from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from verify_alpha_scan import expected_alpha_jobs


def test_formal_alpha_matrix_covers_every_candidate_seed_sample_and_alpha(tmp_path):
    manifest = tmp_path / "dataset.jsonl"
    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    manifest.write_text(
        '{"id":"sample","image":"'
        + str(source)
        + '","instruction":"edit","target_description":"target","category":"object","split":"discovery"}\n',
        encoding="utf-8",
    )
    (tmp_path / "probe_config.yaml").write_text(
        f"project:\n  dataset_manifest: {manifest}\n  output_root: {tmp_path}\n  run_id: run\n"
        "inference:\n  seeds: [1, 2, 3]\n  pilot_seed: 1\n  resolution: 512\n  alpha: 1.5\n"
        "  alpha_grid: [1.1, 1.25, 1.5, 1.75, 2.0]\n"
        "dataset:\n  categories: [object]\n  pilot_per_category: 1\n",
        encoding="utf-8",
    )
    jobs, _ = expected_alpha_jobs(tmp_path, "formal", [0, 2])
    assert list(jobs) == ["1.1", "1.25", "1.5", "1.75", "2.0"]
    assert all(len(items) == 6 for items in jobs.values())
    assert sum(len(items) for items in jobs.values()) == 30
