"""Export 80 blinded examples for the preregistered single-rater VLM calibration."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import shutil
from collections import defaultdict
from pathlib import Path

import yaml


ROOT = Path("/home/hyp/Code/flux-kontext-block-probing")


def stable_key(value: str) -> str:
    return hashlib.sha256(f"20260714:{value}".encode()).hexdigest()


def main() -> None:
    config = yaml.safe_load((ROOT / "probe_config.yaml").read_text(encoding="utf-8"))
    dataset = {
        row["id"]: row
        for row in (json.loads(line) for line in Path(config["project"]["dataset_manifest"]).read_text(encoding="utf-8").splitlines())
    }
    run_root = Path(config["project"]["output_root"]) / config["project"]["run_id"]
    grouped = defaultdict(list)
    for meta_path in (run_root / "images").rglob("*.json"):
        if meta_path.name.endswith(".eval.json"):
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status") != "complete" or meta.get("split") != "discovery":
            continue
        eval_path = meta_path.with_suffix(".eval.json")
        if not eval_path.exists():
            continue
        evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
        if not evaluation.get("vlm_parse_ok"):
            continue
        grouped[meta["category"]].append((meta, evaluation))

    output = run_root / "calibration"
    images = output / "images"
    images.mkdir(parents=True, exist_ok=True)
    selected = []
    for category in config["dataset"]["categories"]:
        items = grouped[category]
        baselines = sorted((item for item in items if item[0]["mode"] == "baseline"), key=lambda item: stable_key(item[0]["sample_id"]))
        enhanced = sorted(
            (item for item in items if item[0]["mode"] == "enhance_text"),
            key=lambda item: stable_key(f"{item[0]['sample_id']}:{item[0]['global_block_index']}"),
        )
        category_items = baselines[:2]
        used_samples = {item[0]["sample_id"] for item in category_items}
        for item in enhanced:
            if item[0]["sample_id"] in used_samples and len(used_samples) < 5:
                continue
            category_items.append(item)
            used_samples.add(item[0]["sample_id"])
            if len(category_items) == 10:
                break
        if len(category_items) != 10:
            raise RuntimeError(f"{category}: expected 10 calibration items, got {len(category_items)}")
        selected.extend(category_items)

    blinded_rows = []
    key_rows = {}
    html_cards = []
    for index, (meta, evaluation) in enumerate(selected):
        calibration_id = f"cal_{index:03d}"
        row = dataset[meta["sample_id"]]
        source_name = f"{calibration_id}_source.png"
        output_name = f"{calibration_id}_output.png"
        shutil.copy2(row["image"], images / source_name)
        shutil.copy2(meta["output_path"], images / output_name)
        subset = "prompt_calibration" if index % 10 < 5 else "locked_validation"
        blinded_rows.append(
            {
                "calibration_id": calibration_id,
                "subset": subset,
                "category": meta["category"],
                "instruction": row["instruction"],
                "target_description": row["target_description"],
                "source_image": f"images/{source_name}",
                "output_image": f"images/{output_name}",
                "human_score_0_to_4": "",
                "human_evidence": "",
            }
        )
        key_rows[calibration_id] = {
            "sample_id": meta["sample_id"],
            "mode": meta["mode"],
            "global_block_index": meta["global_block_index"],
            "alpha": meta["alpha"],
            "output_sha256": meta["output_sha256"],
            "vlm_score_0_to_4": evaluation["vlm_rubric"]["score_0_to_4"],
            "vlm_raw": evaluation["vlm_raw"],
        }
        html_cards.append(
            f"<section><h3>{calibration_id} — {html.escape(meta['category'])}</h3>"
            f"<p><b>Instruction:</b> {html.escape(row['instruction'])}</p>"
            f"<div><figure><img src='images/{source_name}'><figcaption>Source</figcaption></figure>"
            f"<figure><img src='images/{output_name}'><figcaption>Output</figcaption></figure></div>"
            f"<p>Score 0–4: ____ &nbsp; Evidence: ____________________</p></section>"
        )
    with open(output / "blinded_labels.csv", "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(blinded_rows[0]))
        writer.writeheader()
        writer.writerows(blinded_rows)
    (output / "sealed_key.json").write_text(json.dumps(key_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    document = """<!doctype html><html><head><meta charset='utf-8'><style>
body{font-family:Arial,sans-serif;max-width:1200px;margin:auto}section{border-bottom:1px solid #ccc;padding:20px}
section div{display:flex;gap:20px}figure{margin:0}img{width:512px;max-height:512px;object-fit:contain;background:#eee}
</style></head><body><h1>Blinded edit-completion calibration</h1>
<p>Judge only whether the requested target edit appears on the correct object/region. Ignore aesthetics.</p>
""" + "\n".join(html_cards) + "</body></html>"
    (output / "index.html").write_text(document, encoding="utf-8")
    print(json.dumps({"examples": len(blinded_rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()

