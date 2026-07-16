from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from nodes import build_region_tile_plan  # noqa: E402


COLORS = [
    (255, 64, 64),
    (64, 180, 255),
    (255, 192, 64),
    (160, 96, 255),
    (64, 224, 128),
    (255, 96, 192),
    (64, 224, 224),
    (224, 224, 64),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mask", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--max-long-side", type=int, default=1536)
    parser.add_argument("--max-short-side", type=int, default=1024)
    parser.add_argument("--max-pixels", type=int, default=1572864)
    parser.add_argument("--context-pixels", type=int, default=192)
    parser.add_argument("--multiple", type=int, default=16)
    parser.add_argument("--support-threshold", type=float, default=0.001)
    parser.add_argument("--min-component-pixels", type=int, default=1)
    parser.add_argument("--min-target-extent", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mask_u8 = np.asarray(Image.open(args.mask).convert("L"), dtype=np.uint8)
    mask = mask_u8.astype(np.float32) / 255.0
    plan = build_region_tile_plan(
        mask,
        max_long_side=args.max_long_side,
        max_short_side=args.max_short_side,
        max_pixels=args.max_pixels,
        context_pixels=args.context_pixels,
        multiple=args.multiple,
        support_threshold=args.support_threshold,
        min_component_pixels=args.min_component_pixels,
        min_target_extent=args.min_target_extent,
    )

    report_path = args.output_dir / "03_tile_plan.json"
    report_path.write_text(
        json.dumps(plan["report"], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    union_u8 = np.rint(plan["union_mask"] * 255.0).astype(np.uint8)
    union_path = args.output_dir / "03_ownership_union.png"
    Image.fromarray(union_u8).save(union_path)

    original = None
    original_array = None
    if args.original is not None:
        original = Image.open(args.original).convert("RGB")
        if original.size != (plan["report"]["image_width"], plan["report"]["image_height"]):
            raise ValueError(
                f"Original size {original.size} does not match mask "
                f"{plan['report']['image_width']}x{plan['report']['image_height']}"
            )
        original_array = np.asarray(original, dtype=np.uint8).copy()

    tiles_dir = args.output_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    for tile in plan["tiles"]:
        index = tile["tile_index"]
        x = tile["x"]
        y = tile["y"]
        width = tile["width"]
        height = tile["height"]
        local_mask_u8 = np.rint(tile["ownership_mask"] * 255.0).astype(np.uint8)
        Image.fromarray(local_mask_u8).save(
            tiles_dir / f"tile_{index:02d}_ownership_mask.png"
        )
        if original is not None:
            original.crop((x, y, x + width, y + height)).save(
                tiles_dir / f"tile_{index:02d}_input_crop.png"
            )

            active = local_mask_u8 > 0
            local = original_array[y : y + height, x : x + width]
            color = np.asarray(COLORS[index % len(COLORS)], dtype=np.float32)
            local[active] = np.rint(local[active] * 0.45 + color * 0.55).astype(np.uint8)

    overlay_path = None
    if original_array is not None:
        overlay = Image.fromarray(original_array)
        draw = ImageDraw.Draw(overlay)
        for tile in plan["tiles"]:
            index = tile["tile_index"]
            x = tile["x"]
            y = tile["y"]
            width = tile["width"]
            height = tile["height"]
            color = COLORS[index % len(COLORS)]
            draw.rectangle((x, y, x + width - 1, y + height - 1), outline=color, width=8)
            draw.rectangle((x + 8, y + 8, x + 150, y + 58), fill=(0, 0, 0))
            draw.text((x + 18, y + 18), f"tile {index}", fill=color)
        overlay_path = args.output_dir / "03_tile_plan_overlay.png"
        overlay.save(overlay_path)

    print(
        json.dumps(
            {
                "report": str(report_path),
                "ownership_union": str(union_path),
                "overlay": str(overlay_path) if overlay_path else None,
                "tiles_dir": str(tiles_dir),
                "tile_count": plan["count"],
                "ownership_overlap_pixels": plan["report"]["ownership_overlap_pixels"],
                "ownership_union_mismatch_pixels": plan["report"][
                    "ownership_union_mismatch_pixels"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
