"""Render results/metrics.csv into the README results table.

Every run appended by train.py, evaluate.py or pipeline.py becomes one row,
written between the `<!-- results:start -->` / `<!-- results:end -->` markers.
"""
import argparse
import csv
from pathlib import Path

MARK_START = "<!-- results:start -->"
MARK_END = "<!-- results:end -->"

COLUMNS = ["run_id", "seed", "region", "model", "split", "iou", "f1",
           "baseline_iou", "hectares_cleared"]
HEADERS = ["Run", "Seed", "Region", "Model", "Split", "IoU", "F1",
           "IoU (NDVI)", "Hectares cleared"]


def read_metrics(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _fmt(row, key):
    val = (row.get(key) or "").strip()
    if not val:
        return ""
    try:
        return f"{float(val):.4f}"
    except ValueError:
        return val


def render_table(rows):
    lines = ["| " + " | ".join(HEADERS) + " |",
             "|" + "|".join("---" for _ in HEADERS) + "|"]
    if not rows:
        cells = (["_no runs yet — run train.py / evaluate.py / pipeline.py_"]
                 + [""] * (len(HEADERS) - 1))
        lines.append("| " + " | ".join(cells) + " |")
    for r in rows:
        cells = [str(r.get("run_id", "")), str(r.get("seed", "")),
                 str(r.get("region", "")), str(r.get("model", "")),
                 str(r.get("split", "")),
                 _fmt(r, "iou"), _fmt(r, "f1"), _fmt(r, "baseline_iou"),
                 str(r.get("hectares_cleared", ""))]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def update_readme(readme_path, table):
    path = Path(readme_path)
    text = path.read_text(encoding="utf-8")
    if MARK_START not in text or MARK_END not in text:
        raise SystemExit(f"{readme_path} is missing {MARK_START} / {MARK_END} markers")
    pre = text.split(MARK_START)[0]
    post = text.split(MARK_END)[1].lstrip("\n")
    path.write_text(f"{pre}{MARK_START}\n{table}\n{MARK_END}\n{post}", encoding="utf-8")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", default="results/metrics.csv")
    p.add_argument("--readme", default="README.md")
    args = p.parse_args()

    rows = read_metrics(args.metrics)
    update_readme(args.readme, render_table(rows))
    print(f"wrote {len(rows)} run(s) from {args.metrics} into {args.readme}")


if __name__ == "__main__":
    main()
