"""Generate the final eight-question research report from verified artifacts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path("/home/hyp/Code/flux-kontext-block-probing")


def load_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def verdict(comparison: dict | None) -> str:
    if not comparison:
        return "未运行或缺少有效配对。"
    direction = "优于" if comparison["ci_low"] > 0 else "尚未证实优于"
    return f"{direction}；配对差值={comparison['mean']:.4f}，95% CI=[{comparison['ci_low']:.4f}, {comparison['ci_high']:.4f}]。"


def main() -> None:
    config = yaml.safe_load((ROOT / "probe_config.yaml").read_text(encoding="utf-8"))
    output_root = Path(config["project"]["output_root"])
    run_root = output_root / config["project"]["run_id"]
    structure = load_json(output_root / "preflight" / "structure_report.json", {})
    identity = load_json(output_root / "preflight" / "identity_report.json", {})
    selection = load_json(run_root / "selected_blocks.json", {})
    alpha = load_json(run_root / "selected_alpha.json", {})
    joint = load_json(run_root / "joint_validation.json", {})
    calibration = load_json(run_root / "calibration" / "calibration_report.json", {})
    summary = pd.read_csv(run_root / "block_summary.csv") if (run_root / "block_summary.csv").exists() else pd.DataFrame()
    candidates = [int(value) for value in selection.get("selected_global_blocks", [])]
    stream_lookup = {}
    if not summary.empty:
        stream_lookup = {
            int(row["global_block_index"]): row["block_type"] for _, row in summary.iterrows()
        }
    selected_streams = {"double": 0, "single": 0}
    for index in candidates:
        stream = stream_lookup.get(index)
        if stream in selected_streams:
            selected_streams[stream] += 1
    destructive = []
    if not summary.empty:
        risky = summary.sort_values(["preservation_cost", "bad_image_rate"], ascending=False).head(8)
        destructive = [
            {
                "global_block_index": int(row["global_block_index"]),
                "stream": row["block_type"],
                "semantic_gain": None if pd.isna(row["semantic_gain"]) else float(row["semantic_gain"]),
                "preservation_cost": None if pd.isna(row["preservation_cost"]) else float(row["preservation_cost"]),
                "bad_image_rate": float(row["bad_image_rate"]),
            }
            for _, row in risky.iterrows()
        ]
    comparisons = joint.get("comparisons", {})
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = "VALIDATED" if joint.get("status") == "validated" and calibration.get("gate_pass") else "NOT VALIDATED"
    final = f"""# FLUX.1-Kontext-dev Text Block Probing — Final Report

## Executive status

**{status}**

- Project commit: `{commit}`
- Model revision: `{config['model']['revision']}`
- Diffusers commit: `{config['model']['diffusers_commit']}`
- Runtime blocks: {structure.get('double_block_count')} double + {structure.get('single_block_count')} single = {structure.get('total_block_count')}
- Alpha=1 identity: `{identity.get('status')}`
- Human/VLM calibration gate: `{calibration.get('gate_pass', False)}`
- Selected universal blocks: `{selection.get('universal_blocks', [])}`
- Selected category-specific blocks: `{selection.get('category_specific_blocks', {})}`
- Common alpha*: `{alpha.get('alpha', 'not selected')}`

## Required questions

### 1. FLUX-Kontext 的编辑语义是否集中在少数 Block？

通过预注册门槛的独立候选为 `{candidates}`（{len(candidates)} / {structure.get('total_block_count', 0)}）。若列表为空，结论是当前实验未证明语义集中，而不是强行返回 top-k。

### 2. 这些 Block 主要位于 double-stream 还是 single-stream？

候选构成为 `{selected_streams}`。完整逐层结果见 `block_summary.csv` 和三条 global-index 曲线。

### 3. 不同编辑类别是否共享同一组敏感 Block？

Universal：`{selection.get('universal_blocks', [])}`。Category-specific：`{selection.get('category_specific_blocks', {})}`。类别曲线见 `plots/category_block_response_curves.png`。

### 4. 文本增强最有效的 alpha 范围是什么？

在 `[1.1, 1.25, 1.5, 1.75, 2.0]` 中、通过保持与坏图门槛后选择的公共 alpha 为 `{alpha.get('alpha', '未通过门槛')}`。完整结果见 `alpha_summary.csv`。

### 5. 增强这些 Block 是否比增强全部 Block 更有效？

- 标准 all-block：{verdict(comparisons.get('all_blocks'))}
- 预算匹配 all-block：{verdict(comparisons.get('all_blocks_budget_matched'))}
- 随机对照经验 p：`{joint.get('random_empirical_p', '未运行')}`。

### 6. FLUX.1-Dev 的 TexTailor Block 能否迁移？

TexTailor 编号只在候选锁定后作为对照使用。候选组合相对其结果：{verdict(comparisons.get('textailor_flux1dev_control'))}

### 7. 哪些 Block 适合作为编辑强度控制安装位置？

只有 `{candidates}` 可进入安装候选；仍须以 `joint_validation.json` 的状态为最终约束。当前联合状态为 `{joint.get('status', '未运行')}`。

### 8. 哪些 Block 影响语义但会严重破坏源图，应排除？

按 DINO 保持代价和坏图率排序的风险 Block：

```json
{json.dumps(destructive, ensure_ascii=False, indent=2)}
```

## Joint validation

```json
{json.dumps(joint, ensure_ascii=False, indent=2)}
```

## Evidence

- `raw_metrics.csv`, `block_summary.csv`, `alpha_summary.csv`
- `selected_blocks.json`, `joint_metrics.csv`, `joint_summary.csv`, `joint_validation.json`
- `plots/` and `plots/image_grids/`
- `calibration/` and `completion_audit.json`
"""
    (run_root / "FINAL_REPORT.md").write_text(final, encoding="utf-8")
    print(run_root / "FINAL_REPORT.md")


if __name__ == "__main__":
    main()
