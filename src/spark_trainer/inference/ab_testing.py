"""
A/B Shadow Testing Framework.

Provides:
- Champion vs Challenger model comparison
- Shadow deployment (send traffic to both, only return champion)
- Traffic splitting and routing
- Metrics collection and statistical significance testing
- Automatic promotion based on performance
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import time
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)


class TestType(str, Enum):
    """A/B test types."""
    AB = "ab"  # Split traffic between A and B
    SHADOW = "shadow"  # Send to both, only return A
    CANARY = "canary"  # Gradually roll out B


class DecisionCriteria(str, Enum):
    """Decision criteria for promotion."""
    LATENCY = "latency"
    ACCURACY = "accuracy"
    USER_PREFERENCE = "user_preference"
    COST = "cost"


@dataclass
class ModelVariant:
    """Model variant configuration."""
    name: str
    version: str
    adapter: Any  # ServingAdapter instance
    weight: float = 0.5  # Traffic split weight
    is_champion: bool = False


@dataclass
class TestRequest:
    """A/B test request."""
    request_id: str
    inputs: Any
    parameters: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TestResult:
    """A/B test result."""
    request_id: str
    variant_name: str
    outputs: Any
    latency_ms: float
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ExperimentMetrics:
    """Experiment metrics for a variant."""
    variant_name: str
    total_requests: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    error_rate: float
    custom_metrics: Dict[str, float]


class ABExperiment:
    """
    A/B experiment manager.

    Handles traffic splitting, result collection, and statistical analysis.
    """

    def __init__(
        self,
        experiment_name: str,
        champion: ModelVariant,
        challenger: ModelVariant,
        test_type: TestType = TestType.AB,
        output_dir: Optional[str] = None,
    ):
        self.experiment_name = experiment_name
        self.champion = champion
        self.challenger = challenger
        self.test_type = test_type
        self.output_dir = Path(output_dir or "./ab_experiments") / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.results: List[TestResult] = []
        self.results_lock = threading.Lock()

        # Metrics cache
        self._metrics_cache: Dict[str, List[float]] = defaultdict(list)

        # Start time
        self.start_time = datetime.now()

        logger.info(f"A/B experiment started: {experiment_name} ({test_type.value})")

    def run_test(self, request: TestRequest) -> Any:
        """
        Run A/B test for a request.

        Args:
            request: Test request

        Returns:
            Response (from champion if shadow mode, from selected variant otherwise)
        """
        if self.test_type == TestType.SHADOW:
            return self._run_shadow_test(request)
        elif self.test_type == TestType.AB:
            return self._run_ab_test(request)
        elif self.test_type == TestType.CANARY:
            return self._run_canary_test(request)

    def _run_shadow_test(self, request: TestRequest) -> Any:
        """
        Run shadow test: send to both, return only champion.

        This allows testing challenger without affecting users.
        """
        # Send to both models in parallel
        champion_result = None
        challenger_result = None

        def run_champion():
            nonlocal champion_result
            champion_result = self._execute_request(self.champion, request)

        def run_challenger():
            nonlocal challenger_result
            challenger_result = self._execute_request(self.challenger, request)

        # Execute in parallel
        champion_thread = threading.Thread(target=run_champion)
        challenger_thread = threading.Thread(target=run_challenger)

        champion_thread.start()
        challenger_thread.start()

        champion_thread.join()
        challenger_thread.join()

        # Store both results
        with self.results_lock:
            if champion_result:
                self.results.append(champion_result)
            if challenger_result:
                self.results.append(challenger_result)

        # Return only champion output
        return champion_result.outputs if champion_result else None

    def _run_ab_test(self, request: TestRequest) -> Any:
        """
        Run A/B test: split traffic based on weights.
        """
        # Select variant based on weights
        variant = self._select_variant()

        # Execute request
        result = self._execute_request(variant, request)

        # Store result
        with self.results_lock:
            self.results.append(result)

        return result.outputs

    def _run_canary_test(self, request: TestRequest) -> Any:
        """
        Run canary test: gradually increase challenger traffic.

        Traffic split evolves over time based on challenger performance.
        """
        # Adjust weights based on time and performance
        self._adjust_canary_weights()

        # Run as A/B test with adjusted weights
        return self._run_ab_test(request)

    def _execute_request(
        self,
        variant: ModelVariant,
        request: TestRequest,
    ) -> TestResult:
        """Execute request on a specific variant."""
        start_time = time.time()

        try:
            # Create inference request
            from .serving_adapters import InferenceRequest

            inf_request = InferenceRequest(
                inputs=request.inputs,
                parameters=request.parameters,
                metadata=request.metadata,
            )

            # Run inference
            response = variant.adapter.predict(inf_request)

            latency_ms = response.latency_ms

            # Create result
            result = TestResult(
                request_id=request.request_id,
                variant_name=variant.name,
                outputs=response.outputs,
                latency_ms=latency_ms,
                timestamp=datetime.now().isoformat(),
                metadata={
                    'variant_version': variant.version,
                    'is_champion': variant.is_champion,
                },
            )

            return result

        except Exception as e:
            logger.error(f"Request failed for {variant.name}: {e}")

            # Return error result
            latency_ms = (time.time() - start_time) * 1000
            return TestResult(
                request_id=request.request_id,
                variant_name=variant.name,
                outputs=None,
                latency_ms=latency_ms,
                timestamp=datetime.now().isoformat(),
                metadata={'error': str(e)},
            )

    def _select_variant(self) -> ModelVariant:
        """Select variant based on weights."""
        import random

        total_weight = self.champion.weight + self.challenger.weight
        rand = random.random() * total_weight

        if rand < self.champion.weight:
            return self.champion
        else:
            return self.challenger

    def _adjust_canary_weights(self):
        """
        Adjust canary weights based on performance.

        Gradually increases challenger weight if performing well.
        """
        # Calculate time since start
        elapsed_hours = (datetime.now() - self.start_time).total_seconds() / 3600

        # Get current metrics
        metrics = self.get_metrics()

        if not metrics:
            return

        champion_metrics = next((m for m in metrics if m.variant_name == self.champion.name), None)
        challenger_metrics = next((m for m in metrics if m.variant_name == self.challenger.name), None)

        if not champion_metrics or not challenger_metrics:
            return

        # Check if challenger is performing well
        challenger_ok = (
            challenger_metrics.error_rate <= champion_metrics.error_rate * 1.1 and
            challenger_metrics.avg_latency_ms <= champion_metrics.avg_latency_ms * 1.2
        )

        if challenger_ok:
            # Gradually increase challenger weight
            max_weight = min(0.5, elapsed_hours * 0.1)  # 10% per hour, max 50%
            self.challenger.weight = max_weight
            self.champion.weight = 1.0 - max_weight
            logger.info(f"Canary weights adjusted: champion={self.champion.weight:.2f}, challenger={self.challenger.weight:.2f}")
        else:
            # Reduce challenger weight
            self.challenger.weight = max(0.05, self.challenger.weight * 0.5)
            self.champion.weight = 1.0 - self.challenger.weight
            logger.warning(f"Challenger underperforming, weights adjusted: challenger={self.challenger.weight:.2f}")

    def get_metrics(self) -> List[ExperimentMetrics]:
        """
        Calculate metrics for all variants.

        Returns:
            List of metrics per variant
        """
        with self.results_lock:
            # Group results by variant
            variant_results = defaultdict(list)
            for result in self.results:
                variant_results[result.variant_name].append(result)

            metrics = []

            for variant_name, results in variant_results.items():
                if not results:
                    continue

                # Calculate latencies
                latencies = [r.latency_ms for r in results if r.outputs is not None]
                errors = [r for r in results if r.outputs is None]

                if not latencies:
                    continue

                metrics.append(ExperimentMetrics(
                    variant_name=variant_name,
                    total_requests=len(results),
                    avg_latency_ms=float(np.mean(latencies)),
                    p50_latency_ms=float(np.percentile(latencies, 50)),
                    p95_latency_ms=float(np.percentile(latencies, 95)),
                    p99_latency_ms=float(np.percentile(latencies, 99)),
                    error_rate=len(errors) / len(results),
                    custom_metrics={},
                ))

            return metrics

    def statistical_test(
        self,
        metric: str = "latency_ms",
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Run statistical significance test.

        Uses t-test to determine if challenger is significantly different from champion.

        Args:
            metric: Metric to test
            alpha: Significance level

        Returns:
            Test results with p-value and decision
        """
        with self.results_lock:
            champion_values = [
                r.latency_ms for r in self.results
                if r.variant_name == self.champion.name and r.outputs is not None
            ]

            challenger_values = [
                r.latency_ms for r in self.results
                if r.variant_name == self.challenger.name and r.outputs is not None
            ]

        if len(champion_values) < 30 or len(challenger_values) < 30:
            return {
                'test': 't-test',
                'metric': metric,
                'sufficient_data': False,
                'message': 'Need at least 30 samples per variant',
            }

        # Run t-test
        from scipy import stats

        t_statistic, p_value = stats.ttest_ind(challenger_values, champion_values)

        # Calculate effect size (Cohen's d)
        mean_diff = np.mean(challenger_values) - np.mean(champion_values)
        pooled_std = np.sqrt((np.var(challenger_values) + np.var(champion_values)) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

        # Determine if significantly better (lower is better for latency)
        is_significant = p_value < alpha
        is_better = mean_diff < 0 if metric == "latency_ms" else mean_diff > 0

        decision = "promote" if is_significant and is_better else "reject"

        return {
            'test': 't-test',
            'metric': metric,
            'sufficient_data': True,
            'p_value': float(p_value),
            't_statistic': float(t_statistic),
            'cohens_d': float(cohens_d),
            'alpha': alpha,
            'is_significant': is_significant,
            'is_better': is_better,
            'decision': decision,
            'champion_mean': float(np.mean(champion_values)),
            'challenger_mean': float(np.mean(challenger_values)),
            'champion_std': float(np.std(champion_values)),
            'challenger_std': float(np.std(challenger_values)),
        }

    def should_promote_challenger(
        self,
        criteria: List[DecisionCriteria],
        min_samples: int = 100,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine if challenger should be promoted.

        Args:
            criteria: Decision criteria to evaluate
            min_samples: Minimum samples required

        Returns:
            Tuple of (should_promote, decision_details)
        """
        metrics = self.get_metrics()

        if not metrics:
            return False, {'reason': 'No metrics available'}

        champion_metrics = next((m for m in metrics if m.variant_name == self.champion.name), None)
        challenger_metrics = next((m for m in metrics if m.variant_name == self.challenger.name), None)

        if not champion_metrics or not challenger_metrics:
            return False, {'reason': 'Missing metrics for one variant'}

        if challenger_metrics.total_requests < min_samples:
            return False, {'reason': f'Insufficient samples: {challenger_metrics.total_requests} < {min_samples}'}

        decision_details = {
            'metrics': {
                'champion': asdict(champion_metrics),
                'challenger': asdict(challenger_metrics),
            },
            'checks': {},
        }

        # Check each criterion
        promote = True

        for criterion in criteria:
            if criterion == DecisionCriteria.LATENCY:
                # Challenger should be faster or within 10%
                latency_ok = challenger_metrics.avg_latency_ms <= champion_metrics.avg_latency_ms * 1.1
                decision_details['checks']['latency'] = {
                    'passed': latency_ok,
                    'champion': champion_metrics.avg_latency_ms,
                    'challenger': challenger_metrics.avg_latency_ms,
                }
                promote = promote and latency_ok

            elif criterion == DecisionCriteria.ACCURACY:
                # Would need custom accuracy tracking
                decision_details['checks']['accuracy'] = {
                    'passed': True,
                    'note': 'Custom accuracy tracking not implemented',
                }

        # Run statistical test
        stat_test = self.statistical_test()
        decision_details['statistical_test'] = stat_test

        if stat_test.get('sufficient_data'):
            promote = promote and stat_test['decision'] == 'promote'

        return promote, decision_details

    def save_results(self):
        """Save experiment results to disk."""
        results_path = self.output_dir / "results.jsonl"

        with open(results_path, 'w') as f:
            for result in self.results:
                f.write(json.dumps(asdict(result)) + '\n')

        # Save summary
        metrics = self.get_metrics()
        summary = {
            'experiment_name': self.experiment_name,
            'test_type': self.test_type.value,
            'start_time': self.start_time.isoformat(),
            'duration_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
            'metrics': [asdict(m) for m in metrics],
        }

        summary_path = self.output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Results saved to {self.output_dir}")


class ShadowDeployment:
    """
    Shadow deployment manager.

    Handles multiple concurrent shadow deployments.
    """

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir or "./shadow_deployments")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.active_experiments: Dict[str, ABExperiment] = {}

    def start_experiment(
        self,
        experiment_name: str,
        champion: ModelVariant,
        challenger: ModelVariant,
        test_type: TestType = TestType.SHADOW,
    ) -> ABExperiment:
        """Start new shadow deployment experiment."""
        experiment = ABExperiment(
            experiment_name=experiment_name,
            champion=champion,
            challenger=challenger,
            test_type=test_type,
            output_dir=str(self.output_dir),
        )

        self.active_experiments[experiment_name] = experiment

        logger.info(f"Shadow deployment started: {experiment_name}")

        return experiment

    def stop_experiment(self, experiment_name: str):
        """Stop shadow deployment experiment."""
        if experiment_name not in self.active_experiments:
            raise ValueError(f"Experiment not found: {experiment_name}")

        experiment = self.active_experiments[experiment_name]
        experiment.save_results()

        del self.active_experiments[experiment_name]

        logger.info(f"Shadow deployment stopped: {experiment_name}")

    def get_experiment(self, experiment_name: str) -> ABExperiment:
        """Get active experiment."""
        if experiment_name not in self.active_experiments:
            raise ValueError(f"Experiment not found: {experiment_name}")

        return self.active_experiments[experiment_name]


# Example usage
if __name__ == "__main__":
    from .serving_adapters import VLLMAdapter

    # Create variants
    champion = ModelVariant(
        name="gpt2-champion",
        version="v1",
        adapter=VLLMAdapter("gpt2", "http://localhost:8000"),
        weight=0.8,
        is_champion=True,
    )

    challenger = ModelVariant(
        name="gpt2-challenger",
        version="v2",
        adapter=VLLMAdapter("gpt2-large", "http://localhost:8001"),
        weight=0.2,
        is_champion=False,
    )

    # Start shadow deployment
    deployment = ShadowDeployment()
    experiment = deployment.start_experiment(
        experiment_name="gpt2_comparison",
        champion=champion,
        challenger=challenger,
        test_type=TestType.SHADOW,
    )

    # Run test requests
    for i in range(10):
        request = TestRequest(
            request_id=f"req_{i}",
            inputs=f"Hello, this is test request {i}",
            parameters={"max_tokens": 20},
        )

        output = experiment.run_test(request)
        print(f"Request {i}: {output}")

    # Check if should promote
    should_promote, details = experiment.should_promote_challenger(
        criteria=[DecisionCriteria.LATENCY],
        min_samples=5,
    )

    print(f"Should promote: {should_promote}")
    print(f"Details: {json.dumps(details, indent=2)}")

    # Save results
    experiment.save_results()
