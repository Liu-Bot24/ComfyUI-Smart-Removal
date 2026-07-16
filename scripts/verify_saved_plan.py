from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_mask", type=Path)
    parser.add_argument("plan_json", type=Path)
    parser.add_argument("tiles_dir", type=Path)
    parser.add_argument("output_report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = json.loads(args.plan_json.read_text(encoding="utf-8"))
    width = int(report["image_width"])
    height = int(report["image_height"])
    source = np.asarray(Image.open(args.source_mask).convert("L"), dtype=np.uint8)
    if source.shape != (height, width):
        raise ValueError(
            f"Source mask shape {source.shape} does not equal plan {(height, width)}"
        )

    reconstructed = np.zeros((height, width), dtype=np.uint8)
    write_count = np.zeros((height, width), dtype=np.uint16)
    tile_checks = []
    caps = report["caps"]

    for tile in report["tiles"]:
        index = int(tile["tile_index"])
        x = int(tile["x"])
        y = int(tile["y"])
        tile_width = int(tile["width"])
        tile_height = int(tile["height"])
        mask_path = args.tiles_dir / f"tile_{index:02d}_ownership_mask.png"
        crop_path = args.tiles_dir / f"tile_{index:02d}_input_crop.png"
        local = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
        crop_size = Image.open(crop_path).size

        expected_shape = (tile_height, tile_width)
        shape_ok = local.shape == expected_shape
        crop_size_ok = crop_size == (tile_width, tile_height)
        in_bounds = (
            x >= 0
            and y >= 0
            and x + tile_width <= width
            and y + tile_height <= height
        )
        multiple_ok = (
            tile_width % int(caps["multiple"]) == 0
            and tile_height % int(caps["multiple"]) == 0
        )
        cap_ok = (
            max(tile_width, tile_height) <= int(caps["max_long_side"])
            and min(tile_width, tile_height) <= int(caps["max_short_side"])
            and tile_width * tile_height <= int(caps["max_pixels"])
        )

        if not all((shape_ok, crop_size_ok, in_bounds, multiple_ok, cap_ok)):
            raise RuntimeError(f"Tile {index} failed structural checks")

        region = reconstructed[y : y + tile_height, x : x + tile_width]
        np.maximum(region, local, out=region)
        write_count[y : y + tile_height, x : x + tile_width] += (
            local > 0
        ).astype(np.uint16)

        tile_checks.append(
            {
                "tile_index": index,
                "x": x,
                "y": y,
                "width": tile_width,
                "height": tile_height,
                "input_pixels": tile_width * tile_height,
                "active_ownership_pixels": int(np.count_nonzero(local)),
                "mask_shape_ok": shape_ok,
                "crop_size_ok": crop_size_ok,
                "in_bounds": in_bounds,
                "multiple_ok": multiple_ok,
                "caps_ok": cap_ok,
                "mask_sha256": sha256(mask_path),
                "crop_sha256": sha256(crop_path),
            }
        )

    source_active = source > 0
    reconstructed_active = reconstructed > 0
    overlap_pixels = int(np.count_nonzero(write_count > 1))
    gap_pixels = int(np.count_nonzero(source_active & ~reconstructed_active))
    extra_pixels = int(np.count_nonzero(~source_active & reconstructed_active))
    value_mismatch_pixels = int(np.count_nonzero(source != reconstructed))
    max_abs_diff = int(
        np.abs(source.astype(np.int16) - reconstructed.astype(np.int16)).max()
    )

    passed = all(
        (
            overlap_pixels == 0,
            gap_pixels == 0,
            extra_pixels == 0,
            value_mismatch_pixels == 0,
            max_abs_diff == 0,
            len(tile_checks) == int(report["tile_count"]),
        )
    )
    result = {
        "passed": passed,
        "source_mask": str(args.source_mask.resolve()),
        "source_mask_sha256": sha256(args.source_mask),
        "plan_json": str(args.plan_json.resolve()),
        "plan_json_sha256": sha256(args.plan_json),
        "canvas_width": width,
        "canvas_height": height,
        "source_active_pixels": int(np.count_nonzero(source_active)),
        "reconstructed_active_pixels": int(np.count_nonzero(reconstructed_active)),
        "tile_count": len(tile_checks),
        "ownership_overlap_pixels": overlap_pixels,
        "ownership_gap_pixels": gap_pixels,
        "ownership_extra_pixels": extra_pixels,
        "ownership_value_mismatch_pixels": value_mismatch_pixels,
        "ownership_max_abs_diff": max_abs_diff,
        "tiles": tile_checks,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
