import numpy as np
from PIL import Image
from early_edit_reward_distillation.pie import parse_flat_mask

def test_parse_pie_flat_mask():
    mask = parse_flat_mask("0 2 5 1", (3, 2))
    assert np.asarray(mask).tolist() == [[255, 255, 0], [0, 0, 255]]
