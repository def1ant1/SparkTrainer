import time
from pathlib import Path

import torch

from spark_trainer.inference.ab_testing import ABExperiment, ModelVariant, TestRequest, TestType, DecisionCriteria
from spark_trainer.inference.serving_adapters import InferenceRequest, InferenceResponse
from spark_trainer.models.gating import TopKRouter


class ToyGatedAdapter:
    """A minimal adapter that exercises gating then returns a stub response."""

    def __init__(self, name: str, base_latency_ms: float = 5.0):
        self.name = name
        self.base_latency_ms = base_latency_ms
        # Tiny router to exercise gating path
        self.router = TopKRouter(d_model=8, num_experts=3, num_selected=2, enable_jitter=False)

    def predict(self, request: InferenceRequest) -> InferenceResponse:
        # Run a small gating computation to simulate model work
        with torch.no_grad():
            hidden = torch.randn(1, 2, 8)  # [B=1, S=2, D=8]
            _idx, _w, _metrics = self.router(hidden)

        # Return deterministic latency for test logic
        latency_ms = self.base_latency_ms
        return InferenceResponse(outputs=f"ok:{self.name}", latency_ms=latency_ms)


def test_ab_experiment_with_gated_adapters(tmp_path):
    # Create champion and challenger using gating-backed adapters
    champion = ModelVariant(name="champion", version="v1", adapter=ToyGatedAdapter("A", base_latency_ms=8.0), weight=0.5, is_champion=True)
    challenger = ModelVariant(name="challenger", version="v2", adapter=ToyGatedAdapter("B", base_latency_ms=5.0), weight=0.5, is_champion=False)

    exp = ABExperiment(
        experiment_name="test_gated_exp",
        champion=champion,
        challenger=challenger,
        test_type=TestType.SHADOW,  # ensure both paths execute
        output_dir=str(tmp_path),
    )

    # Run a few requests
    for i in range(10):
        req = TestRequest(request_id=f"r{i}", inputs=f"hello {i}")
        _ = exp.run_test(req)

    # Metrics should include both variants
    metrics = exp.get_metrics()
    names = {m.variant_name for m in metrics}
    assert {"champion", "challenger"}.issubset(names)

    # Challenger should be considered promotable on latency with small sample requirement
    should_promote, details = exp.should_promote_challenger(criteria=[DecisionCriteria.LATENCY], min_samples=5)
    assert should_promote is True

    # Save results
    exp.save_results()
    out_dir = Path(tmp_path) / "test_gated_exp"
    assert (out_dir / "results.jsonl").exists()
    assert (out_dir / "summary.json").exists()
