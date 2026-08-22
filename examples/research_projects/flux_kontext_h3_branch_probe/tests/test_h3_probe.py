from __future__ import annotations

import numpy as np
from PIL import Image

from h3_metrics import masked_errors
from pie_bench import decode_mask


def test_decode_full_mask() -> None:
    mask = decode_mask("0 262144")
    assert mask.shape == (512, 512)
    assert int(mask.sum()) == 262144


def test_decode_intervals() -> None:
    mask = decode_mask("0 2 10 3", size=4)
    assert mask.reshape(-1).tolist() == [True, True, False, False, False, False, False, False, False, False, True, True, True, False, False, False]


def test_masked_errors_use_preservation_complement() -> None:
    source = Image.new("RGB", (4, 4), (0, 0, 0))
    output = source.copy()
    pixels = output.load()
    for y in range(2):
        for x in range(4):
            pixels[x, y] = (255, 0, 0)
    edit_mask = np.zeros((4, 4), dtype=bool)
    edit_mask[:2] = True
    l1, l2 = masked_errors(source, [output], edit_mask)
    assert l1[0] == 0.0
    assert l2[0] == 0.0


def test_full_mask_has_no_preservation_error() -> None:
    source = Image.new("RGB", (4, 4), (0, 0, 0))
    output = Image.new("RGB", (4, 4), (255, 0, 0))
    l1, l2 = masked_errors(source, [output], np.ones((4, 4), dtype=bool))
    assert l1 == [None]
    assert l2 == [None]
