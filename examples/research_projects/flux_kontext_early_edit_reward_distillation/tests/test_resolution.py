import pytest
from early_edit_reward_distillation.resolution import resolve_dimensions

def test_flux_packed_resolution_accounting():
    result = resolve_dimensions(512, 512, 8)
    assert result["resolved_height"] == 512 and result["latent_height"] == 64 and result["generated_image_tokens"] == 1024

def test_resolution_rejects_non_cell_image():
    with pytest.raises(ValueError): resolve_dimensions(8, 8, 8)
