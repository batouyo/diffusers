#!/usr/bin/env python3
"""Persist official EditScore output for an existing source/candidate pair."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="/data15/hyp/weight/Qwen3-VL-4B-Instruct")
    parser.add_argument("--lora", default="/data15/hyp/weight/EditScore-Qwen3-VL-4B-Instruct")
    args = parser.parse_args()
    try:
        from editscore import EditScore
        scorer = EditScore(backbone="qwen3vl", model_name_or_path=args.model, lora_path=args.lora, score_range=25, num_pass=1)
        result = scorer.evaluate([Image.open(args.source), Image.open(args.candidate)], args.instruction)
        payload = {"status": "ok", "source": args.source, "candidate": args.candidate, "instruction": args.instruction, "model": args.model, "lora": args.lora, "result": {key: (float(value) if hasattr(value, "item") else value) for key, value in result.items()}}
    except Exception as exc:
        payload = {"status": "failed", "source": args.source, "candidate": args.candidate, "instruction": args.instruction, "error": f"{type(exc).__name__}: {exc}"}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        raise
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
