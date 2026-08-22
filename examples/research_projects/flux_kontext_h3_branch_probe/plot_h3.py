from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


def _mean(values: list[float]) -> float | None:
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["branching_step"]), str(row["category"]))].append(row)
    output: list[dict[str, Any]] = []
    for (step, category), items in sorted(grouped.items()):
        metrics = {}
        for key in items[0]:
            if key in {"sample_id", "category", "branching_step", "strengths"} or key == "clip_scores":
                continue
            values = []
            for item in items:
                try:
                    value = float(item[key])
                    if value == value:
                        values.append(value)
                except (KeyError, TypeError, ValueError):
                    pass
            if values:
                metrics[key] = sum(values) / len(values)
        output.append({"branching_step": step, "category": category, "n": len(items), **metrics})
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(rows: list[dict[str, Any]], plot_root: str | Path) -> None:
    from PIL import Image, ImageDraw

    plot_root = Path(plot_root)
    plot_root.mkdir(parents=True, exist_ok=True)
    categories = sorted({str(row["category"]) for row in rows})
    steps = sorted({int(row["branching_step"]) for row in rows})

    colors = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189), (255, 127, 14)]

    def draw(metric: str, filename: str, ylabel: str) -> None:
        width, height = 960, 620
        left, top, right, bottom = 90, 45, 900, 520
        image = Image.new("RGB", (width, height), "white")
        canvas = ImageDraw.Draw(image)
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        if not values:
            canvas.text((left, height // 2), f"No finite {ylabel} values", fill="black")
            image.save(plot_root / filename)
            return
        ymin, ymax = min(values), max(values)
        if ymin == ymax:
            ymin, ymax = ymin - 1.0, ymax + 1.0
        canvas.line((left, top, left, bottom), fill="black", width=2)
        canvas.line((left, bottom, right, bottom), fill="black", width=2)
        canvas.text((left, 12), ylabel, fill="black")
        canvas.text((right - 100, bottom + 20), "branching step k", fill="black")
        for color_index, category in enumerate(categories):
            points = []
            for step in steps:
                matches = [row for row in rows if int(row["branching_step"]) == step and row["category"] == category and row.get(metric) is not None]
                if matches:
                    x = left + (right - left) * (step - min(steps)) / max(1, max(steps) - min(steps))
                    y = bottom - (bottom - top) * (float(matches[0][metric]) - ymin) / (ymax - ymin)
                    points.append((int(x), int(y)))
            color = colors[color_index % len(colors)]
            if len(points) > 1:
                canvas.line(points, fill=color, width=3)
            for point in points:
                canvas.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill=color)
        image.save(plot_root / filename)

    draw("edit_dynamic_range", "edit_dynamic_range_vs_branching.png", "CLIP directional range")
    draw("preserve_l1_mean", "preservation_error_vs_branching.png", "masked preservation L1")
    draw("monotonicity", "monotonicity_vs_branching.png", "monotonicity score")

    image = Image.new("RGB", (960, 620), "white")
    canvas = ImageDraw.Draw(image)
    left, top, right, bottom = 90, 45, 900, 520
    valid_rows = [row for row in rows if row.get("preserve_l1_mean") is not None and row.get("edit_dynamic_range") is not None]
    if valid_rows:
        xs = [float(row["preserve_l1_mean"]) for row in valid_rows]
        ys = [float(row["edit_dynamic_range"]) for row in valid_rows]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        if xmin == xmax:
            xmin, xmax = xmin - 1.0, xmax + 1.0
        if ymin == ymax:
            ymin, ymax = ymin - 1.0, ymax + 1.0
        canvas.line((left, top, left, bottom), fill="black", width=2)
        canvas.line((left, bottom, right, bottom), fill="black", width=2)
        canvas.text((left, 12), "editability vs preservation", fill="black")
        for index, row in enumerate(valid_rows):
            x = left + (right - left) * (float(row["preserve_l1_mean"]) - xmin) / (xmax - xmin)
            y = bottom - (bottom - top) * (float(row["edit_dynamic_range"]) - ymin) / (ymax - ymin)
            color = colors[index % len(colors)]
            canvas.ellipse((int(x) - 6, int(y) - 6, int(x) + 6, int(y) + 6), fill=color)
    else:
        canvas.text((left, 250), "No finite preservation/editability pairs", fill="black")
    image.save(plot_root / "editability_structure_pareto.png")
