"""Strict adapter boundary for the official EditScore implementation."""
from PIL import Image
class EditScoreUnavailable(RuntimeError): pass
class EditScoreAdapter:
    def __init__(self, evaluator):
        if evaluator is None: raise EditScoreUnavailable("official Qwen3-VL 4B EditScore evaluator is required")
        self.evaluator = evaluator
    def score(self, source: Image.Image, candidate: Image.Image, instruction: str) -> float:
        if source is None or candidate is None or not instruction.strip(): raise ValueError("EditScore requires source, candidate and instruction")
        if hasattr(self.evaluator, "evaluate"):
            result = self.evaluator.evaluate([source, candidate], instruction)
            if not isinstance(result, dict) or "overall" not in result:
                raise ValueError("official EditScore result must contain 'overall'")
            return float(result["overall"])
        return float(self.evaluator(source=source, candidate=candidate, instruction=instruction))
