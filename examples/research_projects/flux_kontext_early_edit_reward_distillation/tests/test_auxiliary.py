import json
import torch
from PIL import Image
from early_edit_reward_distillation.cache import load_teacher_record, save_teacher_record
from early_edit_reward_distillation.metrics import region_l1
from early_edit_reward_distillation.scheduler_debug import inspect_scheduler
from early_edit_reward_distillation.token_layout import audit_token_layout

class FakeScheduler:
    def set_timesteps(self, steps):
        self.timesteps = torch.arange(steps, -1, -1, dtype=torch.float32)
        self.sigmas = torch.linspace(1, 0, steps + 1)

def test_scheduler_audit_uses_nonzero_transitions():
    report = inspect_scheduler(FakeScheduler(), 4)
    assert report["selected"][0]["index"] == 1
    assert report["selected"][0]["post_step_index"] == 2

def test_cache_roundtrip_and_layout(tmp_path):
    save_teacher_record(tmp_path, "sample", {"baseline_state": torch.ones(2, 3)}, {"teacher_step_indices": [1, 2]})
    tensors, metadata = load_teacher_record(tmp_path, "sample")
    assert torch.equal(tensors["baseline_state"], torch.ones(2, 3)) and metadata["schema_version"] == 1
    report = audit_token_layout(generated_tokens=4, source_tokens=8, text_tokens=12, module_shapes={"a": (1, 4, 8)})
    assert report["mask_target"] == "generated_image_tokens_only"

def test_region_l1_respects_preserve_complement():
    a = Image.new("RGB", (2, 2), (0, 0, 0)); b = a.copy(); b.putpixel((0, 0), (255, 0, 0)); mask = Image.new("L", (2, 2), 0); mask.putpixel((0, 0), 255)
    assert region_l1(a, b, mask) > 0 and region_l1(a, b, mask, preserve=True) == 0
