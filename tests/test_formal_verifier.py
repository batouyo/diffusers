from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from verify_formal_complete import expected_formal_jobs, job_key


@dataclass
class FakeJob:
    sample_id: str = "sample"
    seed: int = 42
    mode: str = "enhance_text"
    global_block_index: int = 3
    alpha: float = 1.5
    resolution: int = 512
    split: str = "discovery"


def test_job_key_is_identical_for_dataclass_and_metadata():
    job = FakeJob()
    metadata = {
        "sample_id": "sample",
        "seed": 42,
        "mode": "enhance_text",
        "global_block_index": 3,
        "alpha": 1.5,
        "resolution": 512,
        "split": "discovery",
    }
    assert job_key(job) == job_key(metadata)


def test_expected_formal_jobs_cover_exact_mode_matrix():
    config = {
        "inference": {"seeds": [1, 2, 3], "resolution": 512, "alpha": 1.5},
        "dataset": {"categories": ["object"]},
    }
    dataset = [
        {
            "id": "sample",
            "image": "source.png",
            "instruction": "edit",
            "target_description": "target",
            "category": "object",
            "split": "discovery",
        }
    ]
    groups = expected_formal_jobs(config, dataset, total_blocks=4, stage2=[0, 2], stage3=[2])
    assert {mode: len(jobs) for mode, jobs in groups.items()} == {
        "baseline": 3,
        "enhance_text": 12,
        "disable_text": 6,
        "remove_block": 3,
    }
    keys = [job_key(job) for jobs in groups.values() for job in jobs]
    assert len(keys) == len(set(keys)) == 24
