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
            f"<section id='{calibration_id}' data-calibration-id='{calibration_id}'>"
            f"<h3>{calibration_id} — {html.escape(meta['category'])}</h3>"
            f"<p><b>Instruction:</b> {html.escape(row['instruction'])}</p>"
            f"<div><figure><img src='images/{source_name}'><figcaption>Source</figcaption></figure>"
            f"<figure><img src='images/{output_name}'><figcaption>Output</figcaption></figure></div>"
            f"<fieldset><legend>Requested edit completion (0–4)</legend>"
            + "".join(
                f"<label class='score'><input type='radio' name='{calibration_id}_score' value='{score}'> {score}</label>"
                for score in range(5)
            )
            + "</fieldset>"
            f"<label class='evidence'>Evidence <textarea rows='2' placeholder='Briefly describe visible evidence for the score'></textarea></label>"
            "</section>"
        )
    with open(output / "blinded_labels.csv", "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(blinded_rows[0]))
        writer.writeheader()
        writer.writerows(blinded_rows)
    (output / "sealed_key.json").write_text(json.dumps(key_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    browser_rows = json.dumps(blinded_rows, ensure_ascii=False).replace("</", "<\\/")
    document = """<!doctype html><html><head><meta charset='utf-8'><style>
body{font-family:Arial,sans-serif;max-width:1200px;margin:auto;padding:0 18px 80px;color:#202124}
.toolbar{position:sticky;top:0;z-index:2;background:#fff;border-bottom:1px solid #ccc;padding:12px 0;display:flex;gap:16px;align-items:center}
.toolbar button{padding:9px 16px;font-weight:700}.status{font-variant-numeric:tabular-nums}
section{border-bottom:1px solid #ccc;padding:20px 0;scroll-margin-top:70px}
section>div{display:flex;gap:20px;flex-wrap:wrap}figure{margin:0}img{width:512px;max-width:100%;height:512px;object-fit:contain;background:#eee}
fieldset{border:0;padding:12px 0;margin:0}.score{display:inline-block;border:1px solid #aaa;border-radius:6px;padding:8px 16px;margin-right:8px;cursor:pointer}
.score:has(input:checked){background:#dceeff;border-color:#1267b1}.evidence{display:block;font-weight:700}.evidence textarea{display:block;width:min(1050px,95%);margin-top:6px;padding:8px}
.incomplete{outline:3px solid #c62828;outline-offset:-3px}.complete h3::after{content:' ✓';color:#16823b}
</style></head><body><h1>Blinded edit-completion calibration</h1>
<p>Judge only whether the requested target edit appears on the correct object/region. Ignore aesthetics. Scale: 0=no requested edit; 1=barely visible/mostly wrong; 2=partial; 3=mostly correct; 4=clear and correct.</p>
<div class='toolbar'><strong class='status'>Completed 0/80</strong><button type='button' id='export'>Validate &amp; export CSV</button><span>Answers are saved in this browser.</span></div>
""" + "\n".join(html_cards) + f"""
<script>
const rows = {browser_rows};
const storageKey = 'flux-kontext-calibration-v1';
let state = {{}};
try {{ state = JSON.parse(localStorage.getItem(storageKey) || '{{}}'); }} catch (_) {{ state = {{}}; }}
const save = () => localStorage.setItem(storageKey, JSON.stringify(state));
function update() {{
  let complete = 0;
  for (const row of rows) {{
    const section = document.getElementById(row.calibration_id);
    const value = state[row.calibration_id] || {{}};
    const ok = value.score !== undefined && String(value.evidence || '').trim().length > 0;
    section.classList.toggle('complete', ok);
    if (ok) complete += 1;
  }}
  document.querySelector('.status').textContent = `Completed ${{complete}}/${{rows.length}}`;
}}
for (const row of rows) {{
  const section = document.getElementById(row.calibration_id);
  const value = state[row.calibration_id] || {{}};
  if (value.score !== undefined) {{
    const radio = section.querySelector(`input[value="${{value.score}}"]`);
    if (radio) radio.checked = true;
  }}
  section.querySelector('textarea').value = value.evidence || '';
  section.querySelectorAll('input[type=radio]').forEach(input => input.addEventListener('change', event => {{
    state[row.calibration_id] = {{...(state[row.calibration_id] || {{}}), score:Number(event.target.value)}}; save(); update();
  }}));
  section.querySelector('textarea').addEventListener('input', event => {{
    state[row.calibration_id] = {{...(state[row.calibration_id] || {{}}), evidence:event.target.value}}; save(); update();
  }});
}}
const csvCell = value => '"' + String(value ?? '').replaceAll('"', '""') + '"';
document.getElementById('export').addEventListener('click', () => {{
  document.querySelectorAll('section').forEach(section => section.classList.remove('incomplete'));
  const missing = rows.filter(row => {{
    const value = state[row.calibration_id] || {{}};
    return value.score === undefined || !String(value.evidence || '').trim();
  }});
  if (missing.length) {{
    missing.forEach(row => document.getElementById(row.calibration_id).classList.add('incomplete'));
    document.getElementById(missing[0].calibration_id).scrollIntoView({{behavior:'smooth'}});
    alert(`${{missing.length}} examples still need both a score and evidence.`); return;
  }}
  const fields = ['calibration_id','subset','category','instruction','target_description','source_image','output_image','human_score_0_to_4','human_evidence'];
  const lines = [fields.map(csvCell).join(',')];
  for (const row of rows) {{
    const value = state[row.calibration_id];
    const complete = {{...row, human_score_0_to_4:value.score, human_evidence:value.evidence.trim()}};
    lines.push(fields.map(field => csvCell(complete[field])).join(','));
  }}
  const blob = new Blob(['\\ufeff' + lines.join('\\r\\n')], {{type:'text/csv;charset=utf-8'}});
  const link = document.createElement('a'); link.href = URL.createObjectURL(blob); link.download = 'blinded_labels.csv'; link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}});
update();
</script></body></html>"""
    (output / "index.html").write_text(document, encoding="utf-8")
    print(json.dumps({"examples": len(blinded_rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
