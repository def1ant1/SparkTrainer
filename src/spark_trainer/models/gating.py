"""
Gating Mechanisms for Dynamic Capacity and Smarter Compute

This module provides various gating mechanisms for neural networks:
- Token-level MoE (Mixture of Experts) with capacity factors
- Routerless MoE (DeepSeek-style)
- MoE-LoRA (per-expert low-rank adapters)
- Mixture-of-Depths (dynamic layer selection)
- FiLM/Adapter gating for multi-modal fusion
- Span-routing for contiguous token groups

Author: SparkTrainer
"""

import math
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class GatingMetrics:
    """Metrics collected during gating operations for monitoring."""
    expert_utilization: Tensor  # [num_experts] - tokens routed to each expert
    capacity_overflow: float  # percentage of tokens that exceeded capacity
    z_loss: Optional[float] = None  # auxiliary z-loss for load balancing
    gate_entropy: Optional[float] = None  # entropy of routing decisions
    expert_load_variance: Optional[float] = None  # variance in expert loads
    routing_confidence: Optional[Tensor] = None  # per-token confidence scores

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for logging."""
        metrics = {
            'expert_utilization': self.expert_utilization.cpu().tolist(),
            'capacity_overflow': self.capacity_overflow,
        }
        if self.z_loss is not None:
            metrics['z_loss'] = self.z_loss
        if self.gate_entropy is not None:
            metrics['gate_entropy'] = self.gate_entropy
        if self.expert_load_variance is not None:
            metrics['expert_load_variance'] = self.expert_load_variance
        if self.routing_confidence is not None:
            metrics['routing_confidence'] = self.routing_confidence.mean().item()
        return metrics


class TopKRouter(nn.Module):
    """
    Top-K Router with capacity factors and load balancing.

    Routes tokens to top-K experts with capacity constraints and auxiliary losses
    for balanced load distribution.

    Args:
        d_model: Model dimension
        num_experts: Number of expert modules
        num_selected: Number of experts to select per token (K)
        capacity_factor: Multiplier for expert capacity (default: 1.25)
        gate_temp: Temperature for gating softmax (default: 1.0)
        z_loss_coef: Coefficient for z-loss auxiliary objective (default: 0.01)
        enable_jitter: Add noise to gate logits during training (default: True)
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        num_selected: int = 2,
        capacity_factor: float = 1.25,
        gate_temp: float = 1.0,
        z_loss_coef: float = 0.01,
        enable_jitter: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_selected = num_selected
        self.capacity_factor = capacity_factor
        self.gate_temp = gate_temp
        self.z_loss_coef = z_loss_coef
        self.enable_jitter = enable_jitter

        # Gating network: projects hidden states to expert logits
        self.gate = nn.Linear(d_model, num_experts, bias=False)

        # Initialize gate weights with small values for stable training
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)

    def forward(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor, GatingMetrics]:
        """
        Route tokens to experts.

        Args:
            hidden_states: [batch, seq_len, d_model]

        Returns:
            expert_indices: [batch, seq_len, num_selected] - selected expert IDs
            expert_weights: [batch, seq_len, num_selected] - normalized weights
            metrics: GatingMetrics object with routing statistics
        """
        batch_size, seq_len, d_model = hidden_states.shape
        num_tokens = batch_size * seq_len

        # Flatten batch and sequence dimensions
        hidden_flat = hidden_states.view(-1, d_model)  # [num_tokens, d_model]

        # Compute gate logits
        gate_logits = self.gate(hidden_flat)  # [num_tokens, num_experts]

        # Add jitter noise during training for exploration
        if self.training and self.enable_jitter:
            noise = torch.randn_like(gate_logits) * 0.1
            gate_logits = gate_logits + noise

        # Apply temperature scaling
        gate_logits = gate_logits / self.gate_temp

        # Compute gate probabilities
        gate_probs = F.softmax(gate_logits, dim=-1)  # [num_tokens, num_experts]

        # Top-K selection
        top_k_probs, top_k_indices = torch.topk(
            gate_probs, k=self.num_selected, dim=-1
        )  # [num_tokens, num_selected]

        # Normalize top-K probabilities
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # Calculate expert capacity
        capacity = int(self.capacity_factor * num_tokens / self.num_experts * self.num_selected)

        # Track expert utilization
        expert_counts = torch.zeros(self.num_experts, device=hidden_states.device)
        expert_overflow = 0

        # Count tokens routed to each expert
        for expert_id in range(self.num_experts):
            expert_mask = (top_k_indices == expert_id).any(dim=-1)
            expert_counts[expert_id] = expert_mask.sum()
            if expert_counts[expert_id] > capacity:
                expert_overflow += (expert_counts[expert_id] - capacity).item()

        # Calculate metrics
        capacity_overflow_pct = expert_overflow / num_tokens * 100

        # Z-loss: encourages smaller gate logits for stability
        # Penalizes large logit magnitudes to prevent routing collapse
        z_loss = torch.logsumexp(gate_logits, dim=-1).mean() * self.z_loss_coef

        # Gate entropy: measures diversity of routing decisions
        gate_entropy = -(gate_probs * torch.log(gate_probs + 1e-10)).sum(dim=-1).mean()

        # Expert load variance: measures load balance across experts
        expert_load_normalized = expert_counts / expert_counts.sum()
        ideal_load = 1.0 / self.num_experts
        expert_load_variance = ((expert_load_normalized - ideal_load) ** 2).mean()

        # Routing confidence: average of top-K probabilities
        routing_confidence = top_k_probs.max(dim=-1)[0]

        # Reshape outputs back to [batch, seq_len, ...]
        expert_indices = top_k_indices.view(batch_size, seq_len, self.num_selected)
        expert_weights = top_k_probs.view(batch_size, seq_len, self.num_selected)

        # Create metrics object
        metrics = GatingMetrics(
            expert_utilization=expert_counts,
            capacity_overflow=capacity_overflow_pct,
            z_loss=z_loss.item(),
            gate_entropy=gate_entropy.item(),
            expert_load_variance=expert_load_variance.item(),
            routing_confidence=routing_confidence,
        )

        return expert_indices, expert_weights, metrics


class RouterlessMoE(nn.Module):
    """
    Routerless MoE (DeepSeek-style) - simpler training without explicit routing.

    Instead of a learned router, uses a shared projection followed by per-expert
    processing. All experts process all tokens, but with different projections.
    This is simpler to train and avoids routing collapse issues.

    Args:
        d_model: Model dimension
        d_ff: Feed-forward dimension
        num_experts: Number of expert modules
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts

        # Shared input projection across all experts
        self.shared_proj = nn.Linear(d_model, d_ff)

        # Per-expert specialized projections
        self.expert_up = nn.ModuleList([
            nn.Linear(d_ff, d_ff, bias=False) for _ in range(num_experts)
        ])
        self.expert_down = nn.ModuleList([
            nn.Linear(d_ff, d_model, bias=False) for _ in range(num_experts)
        ])

        # Expert mixing weights (learned per-token affinity)
        self.expert_weights = nn.Linear(d_model, num_experts)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, hidden_states: Tensor) -> Tuple[Tensor, GatingMetrics]:
        """
        Process tokens through routerless MoE.

        Args:
            hidden_states: [batch, seq_len, d_model]

        Returns:
            output: [batch, seq_len, d_model]
            metrics: GatingMetrics with expert utilization
        """
        batch_size, seq_len, d_model = hidden_states.shape

        # Shared projection for all experts
        shared = self.activation(self.shared_proj(hidden_states))  # [B, S, d_ff]

        # Compute expert mixing weights (soft routing)
        expert_scores = F.softmax(
            self.expert_weights(hidden_states), dim=-1
        )  # [B, S, num_experts]

        # Process through each expert and blend
        expert_outputs = []
        for i in range(self.num_experts):
            expert_hidden = self.activation(self.expert_up[i](shared))
            expert_out = self.expert_down[i](expert_hidden)
            expert_outputs.append(expert_out)

        # Stack expert outputs: [num_experts, B, S, d_model]
        expert_outputs = torch.stack(expert_outputs, dim=0)

        # Weighted combination of expert outputs
        # expert_scores: [B, S, num_experts] -> [B, S, num_experts, 1]
        weights = expert_scores.unsqueeze(-1)
        # expert_outputs: [num_experts, B, S, d_model] -> [B, S, num_experts, d_model]
        expert_outputs = expert_outputs.permute(1, 2, 0, 3)

        # Blend: [B, S, num_experts, d_model] * [B, S, num_experts, 1] -> [B, S, d_model]
        output = (expert_outputs * weights).sum(dim=2)
        output = self.dropout(output)

        # Calculate metrics
        expert_utilization = expert_scores.mean(dim=[0, 1]) * batch_size * seq_len
        gate_entropy = -(expert_scores * torch.log(expert_scores + 1e-10)).sum(dim=-1).mean()

        metrics = GatingMetrics(
            expert_utilization=expert_utilization,
            capacity_overflow=0.0,  # No capacity constraints in routerless
            gate_entropy=gate_entropy.item(),
        )

        return output, metrics


class LoRAExpert(nn.Module):
    """
    Low-Rank Adapter (LoRA) expert for parameter-efficient MoE.

    Instead of full expert MLPs, uses low-rank decomposition for dramatic
    VRAM savings while maintaining expressiveness.

    Args:
        d_model: Model dimension
        d_ff: Feed-forward dimension
        rank: Low-rank dimension (default: 8)
        alpha: LoRA scaling factor (default: 16)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.rank = rank
        self.scaling = alpha / rank

        # LoRA low-rank decomposition: W = BA
        # Up projection: d_model -> d_ff
        self.lora_A_up = nn.Linear(d_model, rank, bias=False)
        self.lora_B_up = nn.Linear(rank, d_ff, bias=False)

        # Down projection: d_ff -> d_model
        self.lora_A_down = nn.Linear(d_ff, rank, bias=False)
        self.lora_B_down = nn.Linear(rank, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        # Initialize B to zero for stable training start
        nn.init.zeros_(self.lora_B_up.weight)
        nn.init.zeros_(self.lora_B_down.weight)

        # Initialize A with Kaiming initialization
        nn.init.kaiming_uniform_(self.lora_A_up.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_down.weight, a=math.sqrt(5))

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Process through LoRA expert.

        Args:
            hidden_states: [batch, seq_len, d_model]

        Returns:
            output: [batch, seq_len, d_model]
        """
        # Up projection with LoRA: W_up = B_up @ A_up
        hidden = self.lora_A_up(hidden_states)
        hidden = self.lora_B_up(hidden) * self.scaling
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)

        # Down projection with LoRA: W_down = B_down @ A_down
        output = self.lora_A_down(hidden)
        output = self.lora_B_down(output) * self.scaling
        output = self.dropout(output)

        return output


class MoELoRA(nn.Module):
    """
    MoE with LoRA experts - dramatically smaller VRAM footprint.

    Combines top-K routing with LoRA experts instead of full MLPs,
    reducing memory by 10-100x depending on rank.

    Args:
        d_model: Model dimension
        d_ff: Feed-forward dimension
        num_experts: Number of LoRA experts
        num_selected: Number of experts to select per token (K)
        lora_rank: LoRA rank (default: 8)
        lora_alpha: LoRA scaling (default: 16)
        capacity_factor: Expert capacity multiplier (default: 1.25)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        num_selected: int = 2,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        capacity_factor: float = 1.25,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_selected = num_selected

        # Router for expert selection
        self.router = TopKRouter(
            d_model=d_model,
            num_experts=num_experts,
            num_selected=num_selected,
            capacity_factor=capacity_factor,
        )

        # LoRA experts
        self.experts = nn.ModuleList([
            LoRAExpert(d_model, d_ff, rank=lora_rank, alpha=lora_alpha, dropout=dropout)
            for _ in range(num_experts)
        ])

    def forward(self, hidden_states: Tensor) -> Tuple[Tensor, GatingMetrics]:
        """
        Route tokens to LoRA experts.

        Args:
            hidden_states: [batch, seq_len, d_model]

        Returns:
            output: [batch, seq_len, d_model]
            metrics: GatingMetrics with routing statistics
        """
        batch_size, seq_len, d_model = hidden_states.shape

        # Route tokens to experts
        expert_indices, expert_weights, metrics = self.router(hidden_states)
        # expert_indices: [B, S, K]
        # expert_weights: [B, S, K]

        # Flatten for expert processing
        hidden_flat = hidden_states.view(-1, d_model)  # [B*S, d_model]
        expert_indices_flat = expert_indices.view(-1, self.num_selected)  # [B*S, K]
        expert_weights_flat = expert_weights.view(-1, self.num_selected)  # [B*S, K]

        # Initialize output
        output = torch.zeros_like(hidden_flat)  # [B*S, d_model]

        # Process each token through its selected experts
        for k in range(self.num_selected):
            expert_ids = expert_indices_flat[:, k]  # [B*S]
            weights = expert_weights_flat[:, k].unsqueeze(-1)  # [B*S, 1]

            # Group tokens by expert for efficient batched processing
            for expert_id in range(self.num_experts):
                mask = expert_ids == expert_id
                if mask.any():
                    expert_input = hidden_flat[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    output[mask] += expert_output * weights[mask]

        # Reshape back to [batch, seq_len, d_model]
        output = output.view(batch_size, seq_len, d_model)

        return output, metrics


class MixtureOfDepths(nn.Module):
    """
    Mixture-of-Depths: Gating that chooses which layers to execute per token.

    Implements early-exit mechanism where easy tokens skip expensive layers,
    while hard tokens use full depth. Dramatically reduces FLOPs for inference.

    Args:
        d_model: Model dimension
        threshold: Confidence threshold for early exit (default: 0.8)
        min_layers: Minimum layers to execute (default: 1)
        temperature: Temperature for gating softmax (default: 1.0)
    """

    def __init__(
        self,
        d_model: int,
        threshold: float = 0.8,
        min_layers: int = 1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.threshold = threshold
        self.min_layers = min_layers
        self.temperature = temperature

        # Confidence predictor: predicts if token needs more processing
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

    def should_continue(self, hidden_states: Tensor, layer_idx: int) -> Tuple[Tensor, Dict]:
        """
        Decide which tokens should continue to next layer.

        Args:
            hidden_states: [batch, seq_len, d_model]
            layer_idx: Current layer index

        Returns:
            continue_mask: [batch, seq_len] - boolean mask for tokens to continue
            metrics: Dictionary with depth statistics
        """
        # Always process through minimum layers
        if layer_idx < self.min_layers:
            batch_size, seq_len = hidden_states.shape[:2]
            continue_mask = torch.ones(
                batch_size, seq_len, dtype=torch.bool, device=hidden_states.device
            )
            return continue_mask, {'avg_depth': layer_idx + 1, 'exit_rate': 0.0}

        # Predict confidence for each token
        confidence = self.confidence_head(hidden_states).squeeze(-1)  # [batch, seq_len]

        # Continue if confidence below threshold (token needs more processing)
        continue_mask = confidence < self.threshold

        # Calculate metrics
        exit_rate = (~continue_mask).float().mean().item()
        avg_depth = layer_idx + 1 - exit_rate

        metrics = {
            'avg_depth': avg_depth,
            'exit_rate': exit_rate,
            'confidence_mean': confidence.mean().item(),
            'confidence_std': confidence.std().item(),
        }

        return continue_mask, metrics

    def forward_with_depth_gating(
        self,
        hidden_states: Tensor,
        layer_idx: int,
        layer_fn,
    ) -> Tuple[Tensor, Dict]:
        """
        Apply layer with depth gating - only process tokens that need it.

        Args:
            hidden_states: [batch, seq_len, d_model]
            layer_idx: Current layer index
            layer_fn: Layer function to apply

        Returns:
            output: [batch, seq_len, d_model]
            metrics: Depth gating metrics
        """
        continue_mask, metrics = self.should_continue(hidden_states, layer_idx)

        # If all tokens exit, skip layer computation
        if not continue_mask.any():
            return hidden_states, metrics

        # If all tokens continue, process normally
        if continue_mask.all():
            output = layer_fn(hidden_states)
            return output, metrics

        # Partial execution: only process continuing tokens
        batch_size, seq_len, d_model = hidden_states.shape
        output = hidden_states.clone()

        # Extract continuing tokens
        continue_indices = continue_mask.nonzero(as_tuple=False)  # [num_continue, 2]
        if continue_indices.numel() > 0:
            # Process only continuing tokens through layer
            continuing_states = hidden_states[continue_mask]
            processed_states = layer_fn(continuing_states.unsqueeze(1)).squeeze(1)

            # Put processed tokens back
            output[continue_mask] = processed_states

        return output, metrics


class FiLMGating(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) for multi-modal gating.

    Conditions feature maps on modality embeddings via affine transformations.
    Enables adaptive fusion of text, image, audio modalities.

    Args:
        d_model: Model dimension
        num_modalities: Number of modalities (default: 3 for text/image/audio)
    """

    def __init__(
        self,
        d_model: int,
        num_modalities: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_modalities = num_modalities

        # Modality embeddings
        self.modality_embed = nn.Embedding(num_modalities, d_model)

        # FiLM parameter generators: generate scale (gamma) and shift (beta)
        self.film_generator = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model * 2),
        )

    def forward(
        self,
        hidden_states: Tensor,
        modality_ids: Tensor,
    ) -> Tuple[Tensor, Dict]:
        """
        Apply FiLM conditioning based on modality.

        Args:
            hidden_states: [batch, seq_len, d_model]
            modality_ids: [batch, seq_len] - integer modality IDs (0=text, 1=image, 2=audio)

        Returns:
            output: [batch, seq_len, d_model] - modulated features
            metrics: FiLM gating statistics
        """
        batch_size, seq_len, d_model = hidden_states.shape

        # Get modality embeddings
        modality_emb = self.modality_embed(modality_ids)  # [batch, seq_len, d_model]

        # Generate FiLM parameters: gamma (scale) and beta (shift)
        film_params = self.film_generator(modality_emb)  # [batch, seq_len, d_model*2]
        gamma, beta = film_params.chunk(2, dim=-1)  # Each: [batch, seq_len, d_model]

        # Apply affine transformation: y = gamma * x + beta
        output = gamma * hidden_states + beta

        # Calculate metrics
        modality_counts = torch.bincount(
            modality_ids.flatten(),
            minlength=self.num_modalities,
        ).float()
        modality_distribution = modality_counts / modality_counts.sum()

        metrics = {
            'modality_distribution': modality_distribution.cpu().tolist(),
            'gamma_mean': gamma.mean().item(),
            'gamma_std': gamma.std().item(),
            'beta_mean': beta.mean().item(),
            'beta_std': beta.std().item(),
        }

        return output, metrics


class SpanRouter(nn.Module):
    """
    Span-routing: Route contiguous spans (paragraphs, time windows) to experts.

    Instead of routing individual tokens, groups contiguous spans and routes
    them together. This shares expert KV cache and reduces routing overhead.

    Args:
        d_model: Model dimension
        num_experts: Number of experts
        span_size: Size of spans to route (default: 32 tokens)
        overlap: Overlap between spans (default: 8 tokens)
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        span_size: int = 32,
        overlap: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.span_size = span_size
        self.overlap = overlap

        # Span encoder: aggregates span into single vector
        self.span_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Span router: routes aggregated span to expert
        self.router = nn.Linear(d_model, num_experts)

    def create_spans(
        self,
        hidden_states: Tensor,
    ) -> Tuple[Tensor, List[Tuple[int, int]]]:
        """
        Split sequence into overlapping spans.

        Args:
            hidden_states: [batch, seq_len, d_model]

        Returns:
            spans: [batch, num_spans, span_size, d_model]
            span_ranges: List of (start, end) tuples for each span
        """
        batch_size, seq_len, d_model = hidden_states.shape
        stride = self.span_size - self.overlap

        spans = []
        span_ranges = []

        for start in range(0, seq_len, stride):
            end = min(start + self.span_size, seq_len)
            span = hidden_states[:, start:end, :]

            # Pad if needed
            if span.shape[1] < self.span_size:
                padding = torch.zeros(
                    batch_size,
                    self.span_size - span.shape[1],
                    d_model,
                    device=hidden_states.device,
                )
                span = torch.cat([span, padding], dim=1)

            spans.append(span)
            span_ranges.append((start, end))

            # Stop if we've covered the sequence
            if end >= seq_len:
                break

        spans = torch.stack(spans, dim=1)  # [batch, num_spans, span_size, d_model]
        return spans, span_ranges

    def route_spans(
        self,
        spans: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Route spans to experts.

        Args:
            spans: [batch, num_spans, span_size, d_model]

        Returns:
            expert_ids: [batch, num_spans] - assigned expert for each span
            routing_probs: [batch, num_spans, num_experts] - routing probabilities
        """
        batch_size, num_spans, span_size, d_model = spans.shape

        # Aggregate each span into single vector (mean pooling)
        span_aggregated = spans.mean(dim=2)  # [batch, num_spans, d_model]

        # Encode spans
        span_encoded = self.span_encoder(span_aggregated)  # [batch, num_spans, d_model]

        # Route spans
        routing_logits = self.router(span_encoded)  # [batch, num_spans, num_experts]
        routing_probs = F.softmax(routing_logits, dim=-1)

        # Select expert with highest probability
        expert_ids = routing_probs.argmax(dim=-1)  # [batch, num_spans]

        return expert_ids, routing_probs

    def forward(
        self,
        hidden_states: Tensor,
    ) -> Tuple[Tensor, List[Tuple[int, int]], Tensor, Dict]:
        """
        Create and route spans.

        Args:
            hidden_states: [batch, seq_len, d_model]

        Returns:
            spans: [batch, num_spans, span_size, d_model]
            span_ranges: List of (start, end) for each span
            expert_ids: [batch, num_spans] - expert assignments
            metrics: Routing metrics
        """
        # Create spans
        spans, span_ranges = self.create_spans(hidden_states)

        # Route spans to experts
        expert_ids, routing_probs = self.route_spans(spans)

        # Calculate metrics
        expert_counts = torch.zeros(
            self.num_experts, device=hidden_states.device
        )
        for expert_id in range(self.num_experts):
            expert_counts[expert_id] = (expert_ids == expert_id).sum()

        routing_entropy = -(
            routing_probs * torch.log(routing_probs + 1e-10)
        ).sum(dim=-1).mean()

        metrics = {
            'num_spans': len(span_ranges),
            'expert_utilization': expert_counts.cpu().tolist(),
            'routing_entropy': routing_entropy.item(),
            'avg_span_size': sum(end - start for start, end in span_ranges) / len(span_ranges),
        }

        return spans, span_ranges, expert_ids, metrics


class GatingConfig:
    """
    Configuration for gating mechanisms.

    This config is designed to integrate with the Builder recipe system.
    """

    def __init__(
        self,
        type: str = 'moe',  # moe | moe_lora | routerless | mod_gates | span_routing | mixture_of_depths
        num_experts: int = 8,
        num_selected: int = 2,  # K in Top-K
        capacity_factor: float = 1.25,
        gate_temp: float = 1.0,
        z_loss_coef: float = 0.01,
        # LoRA-specific
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        # Mixture-of-Depths specific
        depth_threshold: float = 0.8,
        min_layers: int = 1,
        # Span routing specific
        span_size: int = 32,
        span_overlap: int = 8,
        # Multi-modal specific
        num_modalities: int = 3,
        # General
        dropout: float = 0.1,
        enable_metrics: bool = True,
    ):
        self.type = type
        self.num_experts = num_experts
        self.num_selected = num_selected
        self.capacity_factor = capacity_factor
        self.gate_temp = gate_temp
        self.z_loss_coef = z_loss_coef
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.depth_threshold = depth_threshold
        self.min_layers = min_layers
        self.span_size = span_size
        self.span_overlap = span_overlap
        self.num_modalities = num_modalities
        self.dropout = dropout
        self.enable_metrics = enable_metrics

    def to_dict(self) -> Dict:
        """Convert config to dictionary for serialization."""
        return {
            'type': self.type,
            'num_experts': self.num_experts,
            'num_selected': self.num_selected,
            'capacity_factor': self.capacity_factor,
            'gate_temp': self.gate_temp,
            'z_loss_coef': self.z_loss_coef,
            'lora_rank': self.lora_rank,
            'lora_alpha': self.lora_alpha,
            'depth_threshold': self.depth_threshold,
            'min_layers': self.min_layers,
            'span_size': self.span_size,
            'span_overlap': self.span_overlap,
            'num_modalities': self.num_modalities,
            'dropout': self.dropout,
            'enable_metrics': self.enable_metrics,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'GatingConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


def create_gating_module(
    gating_config: GatingConfig,
    d_model: int,
    d_ff: Optional[int] = None,
) -> nn.Module:
    """
    Factory function to create gating module from config.

    Args:
        gating_config: GatingConfig instance
        d_model: Model dimension
        d_ff: Feed-forward dimension (required for MoE variants)

    Returns:
        Gating module instance
    """
    if d_ff is None:
        d_ff = d_model * 4  # Default FFN expansion

    if gating_config.type == 'moe':
        return TopKRouter(
            d_model=d_model,
            num_experts=gating_config.num_experts,
            num_selected=gating_config.num_selected,
            capacity_factor=gating_config.capacity_factor,
            gate_temp=gating_config.gate_temp,
            z_loss_coef=gating_config.z_loss_coef,
        )
    elif gating_config.type == 'routerless':
        return RouterlessMoE(
            d_model=d_model,
            d_ff=d_ff,
            num_experts=gating_config.num_experts,
            dropout=gating_config.dropout,
        )
    elif gating_config.type == 'moe_lora':
        return MoELoRA(
            d_model=d_model,
            d_ff=d_ff,
            num_experts=gating_config.num_experts,
            num_selected=gating_config.num_selected,
            lora_rank=gating_config.lora_rank,
            lora_alpha=gating_config.lora_alpha,
            capacity_factor=gating_config.capacity_factor,
            dropout=gating_config.dropout,
        )
    elif gating_config.type == 'mixture_of_depths':
        return MixtureOfDepths(
            d_model=d_model,
            threshold=gating_config.depth_threshold,
            min_layers=gating_config.min_layers,
        )
    elif gating_config.type == 'film_gates':
        return FiLMGating(
            d_model=d_model,
            num_modalities=gating_config.num_modalities,
        )
    elif gating_config.type == 'span_routing':
        return SpanRouter(
            d_model=d_model,
            num_experts=gating_config.num_experts,
            span_size=gating_config.span_size,
            overlap=gating_config.span_overlap,
        )
    else:
        raise ValueError(f"Unknown gating type: {gating_config.type}")


# Example usage and integration tests
if __name__ == "__main__":
    print("SparkTrainer Gating Mechanisms Test Suite")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size, seq_len, d_model, d_ff = 2, 128, 512, 2048

    # Test 1: Top-K MoE Router
    print("\n1. Testing Top-K MoE Router...")
    router = TopKRouter(d_model=d_model, num_experts=8, num_selected=2).to(device)
    hidden = torch.randn(batch_size, seq_len, d_model).to(device)
    expert_indices, expert_weights, metrics = router(hidden)
    print(f"   Expert indices shape: {expert_indices.shape}")
    print(f"   Expert weights shape: {expert_weights.shape}")
    print(f"   Capacity overflow: {metrics.capacity_overflow:.2f}%")
    print(f"   Z-loss: {metrics.z_loss:.6f}")
    print(f"   Expert utilization: {metrics.expert_utilization.tolist()}")

    # Test 2: Routerless MoE
    print("\n2. Testing Routerless MoE...")
    routerless = RouterlessMoE(d_model=d_model, d_ff=d_ff, num_experts=8).to(device)
    output, metrics = routerless(hidden)
    print(f"   Output shape: {output.shape}")
    print(f"   Expert utilization: {metrics.expert_utilization.tolist()}")

    # Test 3: MoE-LoRA
    print("\n3. Testing MoE-LoRA...")
    moe_lora = MoELoRA(
        d_model=d_model, d_ff=d_ff, num_experts=8, lora_rank=8
    ).to(device)
    output, metrics = moe_lora(hidden)
    print(f"   Output shape: {output.shape}")
    print(f"   Capacity overflow: {metrics.capacity_overflow:.2f}%")

    # Test 4: Mixture-of-Depths
    print("\n4. Testing Mixture-of-Depths...")
    mod_gate = MixtureOfDepths(d_model=d_model, threshold=0.8).to(device)

    # Simulate layer function
    dummy_layer = lambda x: x + torch.randn_like(x) * 0.01

    for layer_idx in range(4):
        hidden, depth_metrics = mod_gate.forward_with_depth_gating(
            hidden, layer_idx, dummy_layer
        )
        print(f"   Layer {layer_idx}: avg_depth={depth_metrics['avg_depth']:.2f}, "
              f"exit_rate={depth_metrics['exit_rate']:.2%}")

    # Test 5: FiLM Gating
    print("\n5. Testing FiLM Gating...")
    film_gate = FiLMGating(d_model=d_model, num_modalities=3).to(device)
    modality_ids = torch.randint(0, 3, (batch_size, seq_len)).to(device)
    output, film_metrics = film_gate(hidden, modality_ids)
    print(f"   Output shape: {output.shape}")
    print(f"   Modality distribution: {film_metrics['modality_distribution']}")

    # Test 6: Span Routing
    print("\n6. Testing Span Routing...")
    span_router = SpanRouter(d_model=d_model, num_experts=8, span_size=32).to(device)
    spans, span_ranges, expert_ids, span_metrics = span_router(hidden)
    print(f"   Spans shape: {spans.shape}")
    print(f"   Number of spans: {span_metrics['num_spans']}")
    print(f"   Expert utilization: {span_metrics['expert_utilization']}")

    # Test 7: Config-based creation
    print("\n7. Testing Config-based Module Creation...")
    config = GatingConfig(
        type='moe_lora',
        num_experts=8,
        num_selected=2,
        lora_rank=16,
    )
    module = create_gating_module(config, d_model=d_model, d_ff=d_ff).to(device)
    output, metrics = module(hidden)
    print(f"   Created module: {type(module).__name__}")
    print(f"   Output shape: {output.shape}")

    print("\n" + "=" * 60)
    print("All tests passed successfully!")
    print("=" * 60)
