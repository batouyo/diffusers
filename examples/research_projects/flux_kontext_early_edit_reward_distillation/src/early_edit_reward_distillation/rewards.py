"""Adapters for pluggable image-editing rewards and official EditScore."""
from __future__ import annotations
from typing import Any
from PIL import Image

class EditScoreUnavailable(RuntimeError):
    pass

class EditScoreAdapter:
    """Normalize an evaluator to the project RewardScorer protocol."""
    def __init__(self, evaluator: Any):
        if evaluator is None:
            raise EditScoreUnavailable("official Qwen3-VL 4B EditScore evaluator is required")
        self.evaluator = evaluator
        self.last_details: dict[str, Any] | None = None

    def score_details(self, source: Image.Image, candidate: Image.Image, instruction: str) -> dict[str, Any]:
        if source is None or candidate is None or not instruction.strip():
            raise ValueError("EditScore requires source, candidate and instruction")
        if hasattr(self.evaluator, "evaluate"):
            result = self.evaluator.evaluate([source, candidate], instruction)
            if not isinstance(result, dict) or "overall" not in result:
                raise ValueError("official EditScore result must contain 'overall'")
            details = {key: (value.item() if hasattr(value, "item") else value) for key, value in result.items()}
            details["overall"] = float(details["overall"])
        else:
            details = {"overall": float(self.evaluator(source=source, candidate=candidate, instruction=instruction))}
        self.last_details = details
        return details

    def score(self, source: Image.Image, candidate: Image.Image, instruction: str) -> float:
        return float(self.score_details(source, candidate, instruction)["overall"])
    def score_many(self, source, candidates, instruction):
        return [self.score(source, candidate, instruction) for candidate in candidates]


def build_official_editscore(model_name_or_path: str = "/data15/hyp/weight/Qwen3-VL-4B-Instruct", lora_path: str = "/data15/hyp/weight/EditScore-Qwen3-VL-4B-Instruct", *, score_range: int = 25, num_pass: int = 1) -> EditScoreAdapter:
    """Load the locally cached official EditScore evaluator."""
    try:
        from editscore import EditScore
        evaluator = EditScore(backbone="qwen3vl", model_name_or_path=model_name_or_path, lora_path=lora_path, score_range=score_range, num_pass=num_pass)
    except Exception as exc:
        raise EditScoreUnavailable(f"cannot load official EditScore: {exc}") from exc
    return EditScoreAdapter(evaluator)
