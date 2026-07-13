from scripts.make_image_grids import find_arm


def test_candidate_grid_panel_uses_sparse_combo_not_first_single():
    records = [
        {"sample_id": "sample", "seed": 42, "arm": "candidate_single_g003", "output_path": "single.png"},
        {"sample_id": "sample", "seed": 42, "arm": "candidate_combo", "output_path": "combo.png"},
        {"sample_id": "sample", "seed": 1234, "arm": "candidate_combo", "output_path": "other-seed.png"},
    ]
    assert find_arm(records, "sample", "candidate", 3) == "combo.png"


def test_grid_panels_select_exact_joint_arm_names():
    records = [
        {"sample_id": "sample", "seed": 42, "arm": "baseline", "output_path": "base.png"},
        {"sample_id": "sample", "seed": 42, "arm": "candidate_disable_g003", "output_path": "disable.png"},
        {"sample_id": "sample", "seed": 42, "arm": "random_00", "output_path": "random.png"},
        {"sample_id": "sample", "seed": 42, "arm": "all_blocks", "output_path": "all.png"},
    ]
    assert find_arm(records, "sample", "baseline", 3) == "base.png"
    assert find_arm(records, "sample", "disable", 3) == "disable.png"
    assert find_arm(records, "sample", "random", 3) == "random.png"
    assert find_arm(records, "sample", "all", 3) == "all.png"
