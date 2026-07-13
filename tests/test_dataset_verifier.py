import hashlib
from pathlib import Path
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from scripts.verify_dataset import inspect_rows


def test_dataset_verifier_checks_split_counts_hashes_and_images(tmp_path):
    rows = []
    for index, split in enumerate(["discovery", "heldout"]):
        image_path = tmp_path / f"image-{index}.png"
        Image.new("RGB", (512, 512), (index * 50, 100, 150)).save(image_path)
        rows.append(
            {
                "id": f"sample-{index}",
                "image": str(image_path),
                "instruction": "Change the object to blue",
                "target_description": "The object is blue",
                "category": "color",
                "split": split,
                "source_sha256": hashlib.sha256(f"source-{index}".encode()).hexdigest(),
                "prepared_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                "reference_target_image": str(image_path),
                "qc_status": "automatic_pass_pending_human_review",
            }
        )
    config = {
        "dataset": {
            "categories": ["color"],
            "discovery_per_category": 1,
            "heldout_per_category": 1,
        }
    }
    report = inspect_rows(config, rows)
    assert report["status"] == "pass"
    assert report["unique_source_hashes"] == 2
    assert report["observed_sizes"] == {"(512, 512)": 2}

    rows[1]["prepared_sha256"] = "0" * 64
    failed = inspect_rows(config, rows)
    assert failed["status"] == "fail"
    assert any("checksum mismatch" in error for error in failed["errors"])
