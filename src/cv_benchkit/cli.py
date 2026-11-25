"""CLI entry point for cv-benchkit.

Subcommands
-----------
- ``cv-benchkit convert`` — convert between dataset formats
- ``cv-benchkit validate`` — validate a COCO JSON file
- ``cv-benchkit stats`` — print summary statistics of a COCO JSON file
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from cv_benchkit import __version__
from cv_benchkit.converters import (
    COCODataset,
    coco_to_voc,
    coco_to_yolo,
    validate_coco_file,
    voc_to_coco,
    yolo_to_coco,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cv-benchkit",
        description="Computer-vision benchmark toolkit: convert, validate and summarize datasets.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── convert ──────────────────────────────────────────────────────────
    convert_p = sub.add_parser("convert", help="Convert between dataset formats.")
    convert_p.add_argument(
        "--from",
        dest="src_format",
        required=True,
        choices=["coco", "yolo", "voc"],
        help="Source format.",
    )
    convert_p.add_argument(
        "--to",
        dest="dst_format",
        required=True,
        choices=["coco", "yolo", "voc"],
        help="Target format.",
    )
    convert_p.add_argument(
        "--input",
        dest="input_path",
        required=True,
        help="Input file or directory.",
    )
    convert_p.add_argument(
        "--output",
        dest="output_path",
        required=True,
        help="Output directory.",
    )
    convert_p.add_argument(
        "--image-width",
        type=int,
        default=None,
        help="Image width (required for YOLO source).",
    )
    convert_p.add_argument(
        "--image-height",
        type=int,
        default=None,
        help="Image height (required for YOLO source).",
    )
    convert_p.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Category names (required for YOLO/VOC source).",
    )

    # ── validate ─────────────────────────────────────────────────────────
    validate_p = sub.add_parser("validate", help="Validate a COCO JSON file.")
    validate_p.add_argument("file", help="Path to the COCO JSON file.")

    # ── stats ────────────────────────────────────────────────────────────
    stats_p = sub.add_parser("stats", help="Print summary statistics of a COCO JSON file.")
    stats_p.add_argument("file", help="Path to the COCO JSON file.")
    stats_p.add_argument(
        "--json",
        action="store_true",
        help="Print the statistics as JSON.",
    )

    return parser


def _cmd_convert(args: argparse.Namespace) -> int:
    src = args.src_format
    dst = args.dst_format
    inp = Path(args.input_path)
    out = Path(args.output_path)

    if src == dst:
        print(f"Error: source and target format are the same ({src}).", file=sys.stderr)
        return 1

    if not inp.exists():
        print(f"Error: input path '{inp}' not found.", file=sys.stderr)
        return 1

    if src == "coco" and inp.is_dir():
        print(
            f"Error: input path '{inp}' is a directory, expected a COCO JSON file.",
            file=sys.stderr,
        )
        return 1

    if src in ("yolo", "voc") and not inp.is_dir():
        print(
            f"Error: input path '{inp}' must be a directory for {src.upper()} source format.",
            file=sys.stderr,
        )
        return 1

    if src == "coco":
        coco = validate_coco_file(inp)
        paths = coco_to_yolo(coco, out) if dst == "yolo" else coco_to_voc(coco, out)
        print(f"Wrote {len(paths)} files to {out}")
        return 0

    if src == "yolo":
        cats = args.categories
        if not cats:
            print("Error: --categories required for YOLO source.", file=sys.stderr)
            return 1
        w = args.image_width
        h = args.image_height
        if w is None or h is None:
            print(
                "Error: --image-width and --image-height required for YOLO source.", file=sys.stderr
            )
            return 1
        if w <= 0 or h <= 0:
            print(
                "Error: --image-width and --image-height must be positive integers.",
                file=sys.stderr,
            )
            return 1
        stems = [p.stem for p in inp.glob("*.txt")]
        image_sizes = {s: (w, h) for s in stems}
        coco = yolo_to_coco(inp, image_sizes, cats)
        if dst == "coco":
            out.mkdir(parents=True, exist_ok=True)
            json_path = out / "annotations.json"
            json_path.write_text(
                json.dumps(coco.model_dump(), indent=2),
                encoding="utf-8",
            )
            print(f"Wrote {json_path}")
        else:  # voc
            paths = coco_to_voc(coco, out)
            print(f"Wrote {len(paths)} files to {out}")
        return 0

    if src == "voc":
        cats = args.categories
        if not cats:
            print("Error: --categories required for VOC source.", file=sys.stderr)
            return 1
        coco = voc_to_coco(inp, cats)
        if dst == "coco":
            out.mkdir(parents=True, exist_ok=True)
            json_path = out / "annotations.json"
            json_path.write_text(
                json.dumps(coco.model_dump(), indent=2),
                encoding="utf-8",
            )
            print(f"Wrote {json_path}")
        else:  # yolo
            paths = coco_to_yolo(coco, out)
            print(f"Wrote {len(paths)} files to {out}")
        return 0

    return 1  # pragma: no cover


def _cmd_validate(args: argparse.Namespace) -> int:
    path = Path(args.file)
    if not path.exists():
        print(f"Error: file {path} not found.", file=sys.stderr)
        return 1
    if path.is_dir():
        print(f"Error: {path} is a directory, expected a JSON file.", file=sys.stderr)
        return 1
    try:
        coco = validate_coco_file(path)
    except Exception as exc:  # noqa: BLE001
        print(f"Validation FAILED: {exc}", file=sys.stderr)
        return 1
    msg = (
        f"Valid COCO dataset: {len(coco.images)} images, "
        f"{len(coco.annotations)} annotations, {len(coco.categories)} categories."
    )
    print(msg)
    return 0


# COCO area thresholds: small < 32**2 <= medium < 96**2 <= large.
_SMALL_AREA = 32.0**2
_LARGE_AREA = 96.0**2


def _percent(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def _compute_stats(coco: COCODataset) -> dict[str, Any]:
    """Compute the summary statistics printed by ``cv-benchkit stats``.

    Box scales use the annotation ``area`` with the COCO thresholds. Aspect
    ratios are ``w / h`` of the bbox and skip boxes with zero height.
    """
    n_images = len(coco.images)
    n_annotations = len(coco.annotations)

    per_image = Counter(ann.image_id for ann in coco.annotations)
    counts_per_image = [per_image.get(img.id, 0) for img in coco.images]

    small = sum(1 for ann in coco.annotations if ann.area < _SMALL_AREA)
    large = sum(1 for ann in coco.annotations if ann.area >= _LARGE_AREA)
    medium = n_annotations - small - large

    per_category = Counter(ann.category_id for ann in coco.annotations)
    names = {cat.id: cat.name for cat in coco.categories}
    category_ids = list(names) + [cid for cid in per_category if cid not in names]
    categories: list[dict[str, Any]] = [
        {
            "id": cid,
            "name": names.get(cid, f"<unknown id {cid}>"),
            "count": per_category.get(cid, 0),
            "percent": _percent(per_category.get(cid, 0), n_annotations),
        }
        for cid in category_ids
    ]
    categories.sort(key=lambda c: (-c["count"], c["name"]))

    ratios = [ann.bbox[2] / ann.bbox[3] for ann in coco.annotations if ann.bbox[3] > 0]

    return {
        "overview": {
            "images": n_images,
            "annotations": n_annotations,
            "categories": len(coco.categories),
            "mean_annotations_per_image": (
                statistics.fmean(counts_per_image) if counts_per_image else 0.0
            ),
            "median_annotations_per_image": (
                float(statistics.median(counts_per_image)) if counts_per_image else 0.0
            ),
            "unannotated_images": sum(1 for c in counts_per_image if c == 0),
        },
        "box_scales": {
            name: {"count": count, "percent": _percent(count, n_annotations)}
            for name, count in (("small", small), ("medium", medium), ("large", large))
        },
        "categories": categories,
        "aspect_ratio": {
            "min": min(ratios) if ratios else None,
            "median": float(statistics.median(ratios)) if ratios else None,
            "max": max(ratios) if ratios else None,
            "mean": statistics.fmean(ratios) if ratios else None,
        },
    }


def _format_stats(stats: dict[str, Any]) -> str:
    def num(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.2f}"

    overview = stats["overview"]
    lines = [
        "Dataset overview",
        f"  Images:                    {overview['images']}",
        f"  Annotations:               {overview['annotations']}",
        f"  Categories:                {overview['categories']}",
        (
            f"  Annotations per image:     mean {num(overview['mean_annotations_per_image'])}, "
            f"median {num(overview['median_annotations_per_image'])}"
        ),
        f"  Unannotated images:        {overview['unannotated_images']}",
        "",
        "Box scales (COCO area)",
    ]
    labels = {
        "small": "Small  (< 32^2)",
        "medium": "Medium (32^2 to 96^2)",
        "large": "Large  (>= 96^2)",
    }
    for name, label in labels.items():
        scale = stats["box_scales"][name]
        lines.append(f"  {label:<24} {scale['count']:>8}  {scale['percent']:6.2f}%")
    lines += ["", "Category distribution"]
    if stats["categories"]:
        width = max(len(c["name"]) for c in stats["categories"])
        for cat in stats["categories"]:
            lines.append(f"  {cat['name']:<{width}}  {cat['count']:>8}  {cat['percent']:6.2f}%")
    else:
        lines.append("  (no categories)")
    ratio = stats["aspect_ratio"]
    lines += [
        "",
        "Aspect ratio (w / h)",
        (
            f"  min {num(ratio['min'])}, median {num(ratio['median'])}, "
            f"max {num(ratio['max'])}, mean {num(ratio['mean'])}"
        ),
    ]
    return "\n".join(lines)


def _cmd_stats(args: argparse.Namespace) -> int:
    path = Path(args.file)
    if not path.exists():
        print(f"Error: file {path} not found.", file=sys.stderr)
        return 1
    if path.is_dir():
        print(f"Error: {path} is a directory, expected a JSON file.", file=sys.stderr)
        return 1
    try:
        coco = validate_coco_file(path)
    except Exception as exc:  # noqa: BLE001
        print(f"Validation FAILED: {exc}", file=sys.stderr)
        return 1
    stats = _compute_stats(coco)
    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print(_format_stats(stats))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "convert":
        return _cmd_convert(args)
    if args.command == "validate":
        return _cmd_validate(args)
    if args.command == "stats":
        return _cmd_stats(args)

    parser.print_help()
    return 1  # pragma: no cover
