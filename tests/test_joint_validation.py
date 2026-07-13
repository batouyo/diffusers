from __future__ import annotations

import torch

from scripts.run_joint_validation import arms, random_controls


class ToyTransformer:
    def __init__(self):
        self.transformer_blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(4)])
        self.single_transformer_blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(6)])


def test_random_controls_match_stream_counts_and_exclude_candidates():
    candidates = [0, 2, 5]
    controls = random_controls(10, 4, candidates, count=5, seed=7)
    assert len(controls) == len(set(controls)) == 5
    for control in controls:
        assert len(control) == 3
        assert sum(index < 4 for index in control) == 2
        assert not set(control) & set(candidates)


def test_arms_include_required_controls_and_budget_match():
    values = arms(ToyTransformer(), [0, 5], alpha=1.5, random_sets=3, seed=9)
    lookup = {name: (mode, blocks, alpha) for name, mode, blocks, alpha in values}
    assert "candidate_combo" in lookup
    assert "all_blocks" in lookup
    assert "all_blocks_budget_matched" in lookup
    assert "textailor_flux1dev_control" in lookup
    assert "candidate_disable_g000" in lookup
    assert len([name for name in lookup if name.startswith("random_")]) == 3
    assert lookup["candidate_disable_g000"][0] == "disable_text"
    assert lookup["all_blocks_budget_matched"][2] == 1.0 + 2 / 10 * 0.5
