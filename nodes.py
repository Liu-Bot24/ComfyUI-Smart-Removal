from __future__ import annotations

import json
import math
import re
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


MAX_RESOLUTION = 16384


TILE_CONTROL_PROFILES = {
    "保守（小块）": (1024, 768, 786432, 160),
    "标准（已验证）": (1536, 1024, 1572864, 192),
    "大块（高显存）": (2048, 1280, 2621440, 256),
}

GROW_CONTROL_PROFILES = {
    "精细（8）": 8,
    "小范围（16）": 16,
    "标准（32）": 32,
    "中范围（64）": 64,
    "大范围（128）": 128,
    "超大范围（192）": 192,
}

BLUR_CONTROL_PROFILES = {
    "硬边（0）": 0,
    "精细（4）": 4,
    "标准（8）": 8,
    "柔和（16）": 16,
    "大范围（32）": 32,
    "超柔和（64）": 64,
}

GROW_OVERRIDE_VALUES = {0, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256}
BLUR_OVERRIDE_VALUES = {0, 2, 4, 8, 12, 16, 24, 32, 48, 64}


def _to_numpy_mask(mask: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        value = mask.detach().to(device="cpu", dtype=torch.float32).numpy()
    else:
        value = np.asarray(mask, dtype=np.float32)

    if value.ndim == 4:
        value = value.max(axis=0)
        if value.shape[-1] == 1:
            value = value[..., 0]
    if value.ndim == 3:
        value = value.max(axis=0)
    if value.ndim != 2:
        raise ValueError(f"MASK must resolve to HxW, received shape {value.shape}")

    return np.clip(value, 0.0, 1.0).astype(np.float32, copy=False)


def _axis_crop(
    lower: int,
    upper: int,
    context: int,
    image_size: int,
    multiple: int,
) -> tuple[int, int] | None:
    """Return an in-bounds start/length containing [lower, upper) with context."""
    requested_lower = max(0, lower - context)
    requested_upper = min(image_size, upper + context)
    requested_length = requested_upper - requested_lower
    length = int(math.ceil(requested_length / multiple) * multiple)
    maximum_aligned_length = (image_size // multiple) * multiple
    length = min(length, maximum_aligned_length)

    if length <= 0 or upper - lower > length:
        return None

    center = (requested_lower + requested_upper) / 2.0
    start = int(round(center - length / 2.0))
    start = max(0, min(start, image_size - length))

    if start > lower:
        start = lower
    if start + length < upper:
        start = upper - length
    start = max(0, min(start, image_size - length))

    if not (start <= lower and start + length >= upper):
        return None
    return start, length


def _make_crop(
    xs: np.ndarray,
    ys: np.ndarray,
    image_width: int,
    image_height: int,
    context_pixels: int,
    multiple: int,
) -> dict[str, int] | None:
    min_x = int(xs.min())
    max_x = int(xs.max()) + 1
    min_y = int(ys.min())
    max_y = int(ys.max()) + 1

    horizontal = _axis_crop(min_x, max_x, context_pixels, image_width, multiple)
    vertical = _axis_crop(min_y, max_y, context_pixels, image_height, multiple)
    if horizontal is None or vertical is None:
        return None

    x, width = horizontal
    y, height = vertical
    return {
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "target_bbox_xyxy": [min_x, min_y, max_x, max_y],
    }


def _make_crop_from_bounds(
    min_x: int,
    min_y: int,
    max_x: int,
    max_y: int,
    image_width: int,
    image_height: int,
    context_pixels: int,
    multiple: int,
) -> dict[str, int] | None:
    horizontal = _axis_crop(min_x, max_x, context_pixels, image_width, multiple)
    vertical = _axis_crop(min_y, max_y, context_pixels, image_height, multiple)
    if horizontal is None or vertical is None:
        return None
    x, width = horizontal
    y, height = vertical
    return {
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "target_bbox_xyxy": [min_x, min_y, max_x, max_y],
    }


def _fits_caps(
    width: int,
    height: int,
    max_long_side: int,
    max_short_side: int,
    max_pixels: int,
) -> bool:
    long_side = max(width, height)
    short_side = min(width, height)
    return (
        long_side <= max_long_side
        and short_side <= max_short_side
        and width * height <= max_pixels
    )


def _choose_split_axis(
    xs: np.ndarray,
    ys: np.ndarray,
    context_pixels: int,
    max_long_side: int,
    max_short_side: int,
) -> int:
    width = int(xs.max() - xs.min() + 1) + context_pixels * 2
    height = int(ys.max() - ys.min() + 1) + context_pixels * 2
    if width >= height:
        overflow_x = width / max_long_side
        overflow_y = height / max_short_side
    else:
        overflow_x = width / max_short_side
        overflow_y = height / max_long_side
    if abs(overflow_x - overflow_y) < 1e-9:
        return 1 if width >= height else 0
    return 1 if overflow_x > overflow_y else 0


def _choose_cut(values: np.ndarray, min_target_extent: int) -> int:
    lower = int(values.min())
    upper = int(values.max()) + 1
    if upper - lower <= 1:
        raise ValueError("Unable to split a one-pixel target extent")

    histogram = np.bincount(values - lower, minlength=upper - lower)
    cumulative = np.cumsum(histogram)
    total = int(values.size)
    max_hist = max(1, int(histogram.max()))

    candidates: list[tuple[float, int]] = []
    for cut in range(lower + 1, upper):
        left_count = int(cumulative[cut - lower - 1])
        right_count = total - left_count
        if left_count == 0 or right_count == 0:
            continue
        balance = min(left_count, right_count) / total
        if balance < 0.30:
            continue
        if cut - lower < min_target_extent or upper - cut < min_target_extent:
            continue

        seam_index = cut - lower
        seam_load = int(histogram[seam_index - 1])
        if seam_index < len(histogram):
            seam_load += int(histogram[seam_index])
        balance_penalty = abs(left_count - right_count) / total
        score = seam_load / max_hist + balance_penalty * 0.35
        candidates.append((score, cut))

    if candidates:
        candidates.sort(key=lambda item: (item[0], abs(item[1] - (lower + upper) / 2.0)))
        return candidates[0][1]

    ordered = np.sort(values)
    cut = int(ordered[len(ordered) // 2])
    if cut <= lower:
        cut = lower + 1
    if cut >= upper:
        cut = upper - 1
    return cut


def _choose_maximal_prefix_cut(
    xs: np.ndarray,
    ys: np.ndarray,
    axis: int,
    image_width: int,
    image_height: int,
    max_long_side: int,
    max_short_side: int,
    max_pixels: int,
    context_pixels: int,
    multiple: int,
    min_target_extent: int,
) -> int | None:
    values = xs if axis == 1 else ys
    other = ys if axis == 1 else xs
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_other = other[order]
    if int(sorted_values[0]) == int(sorted_values[-1]):
        return None

    prefix_other_min = np.minimum.accumulate(sorted_other)
    prefix_other_max = np.maximum.accumulate(sorted_other)
    boundaries = np.nonzero(sorted_values[1:] != sorted_values[:-1])[0]
    if boundaries.size == 0:
        return None

    total_span = int(sorted_values[-1] - sorted_values[0] + 1)
    require_min_extent = total_span >= min_target_extent * 2
    candidates: list[tuple[int, int, int]] = []
    for boundary in boundaries:
        cut = int(sorted_values[boundary + 1])
        left_span = cut - int(sorted_values[0])
        right_span = int(sorted_values[-1]) + 1 - cut
        if require_min_extent and (
            left_span < min_target_extent or right_span < min_target_extent
        ):
            continue

        if axis == 1:
            bounds = (
                int(sorted_values[0]),
                int(prefix_other_min[boundary]),
                int(sorted_values[boundary]) + 1,
                int(prefix_other_max[boundary]) + 1,
            )
        else:
            bounds = (
                int(prefix_other_min[boundary]),
                int(sorted_values[0]),
                int(prefix_other_max[boundary]) + 1,
                int(sorted_values[boundary]) + 1,
            )

        crop = _make_crop_from_bounds(
            *bounds,
            image_width,
            image_height,
            context_pixels,
            multiple,
        )
        if crop is None or not _fits_caps(
            crop["width"], crop["height"], max_long_side, max_short_side, max_pixels
        ):
            continue

        seam_load = int(boundary + 1)
        candidates.append((seam_load, left_span, cut))

    if not candidates:
        return None

    # Pack as much target as possible into the current safe tile. The second
    # key favors spatial extent when sparse masks contain uneven pixel density.
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def _sort_leaves_along_principal_axis(
    leaves: list[tuple[np.ndarray, np.ndarray, dict[str, int], int]],
    component_xs: np.ndarray,
    component_ys: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, dict[str, int], int]]:
    centered_x = component_xs.astype(np.float64) - float(component_xs.mean())
    centered_y = component_ys.astype(np.float64) - float(component_ys.mean())
    covariance = np.array(
        [
            [np.mean(centered_x * centered_x), np.mean(centered_x * centered_y)],
            [np.mean(centered_x * centered_y), np.mean(centered_y * centered_y)],
        ],
        dtype=np.float64,
    )
    _, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, -1]
    dominant = int(np.argmax(np.abs(direction)))
    if direction[dominant] < 0:
        direction = -direction

    return sorted(
        leaves,
        key=lambda leaf: float(leaf[0].mean()) * direction[0]
        + float(leaf[1].mean()) * direction[1],
    )


def _split_component(
    xs: np.ndarray,
    ys: np.ndarray,
    image_width: int,
    image_height: int,
    max_long_side: int,
    max_short_side: int,
    max_pixels: int,
    context_pixels: int,
    multiple: int,
    min_target_extent: int,
    depth: int = 0,
) -> list[tuple[np.ndarray, np.ndarray, dict[str, int], int]]:
    crop = _make_crop(xs, ys, image_width, image_height, context_pixels, multiple)
    if crop is not None and _fits_caps(
        crop["width"], crop["height"], max_long_side, max_short_side, max_pixels
    ):
        return [(xs, ys, crop, depth)]

    if depth >= 64:
        raise ValueError("Tile recursion exceeded 64 levels")

    primary_axis = _choose_split_axis(
        xs, ys, context_pixels, max_long_side, max_short_side
    )
    axes = (primary_axis, 1 - primary_axis)
    for axis in axes:
        values = xs if axis == 1 else ys
        if int(values.max()) == int(values.min()):
            continue
        cut = _choose_maximal_prefix_cut(
            xs,
            ys,
            axis,
            image_width,
            image_height,
            max_long_side,
            max_short_side,
            max_pixels,
            context_pixels,
            multiple,
            min_target_extent,
        )
        if cut is None:
            continue
        left = values < cut
        right = ~left
        if not left.any() or not right.any():
            continue
        return _split_component(
            xs[left],
            ys[left],
            image_width,
            image_height,
            max_long_side,
            max_short_side,
            max_pixels,
            context_pixels,
            multiple,
            min_target_extent,
            depth + 1,
        ) + _split_component(
            xs[right],
            ys[right],
            image_width,
            image_height,
            max_long_side,
            max_short_side,
            max_pixels,
            context_pixels,
            multiple,
            min_target_extent,
            depth + 1,
        )

    for axis in axes:
        values = xs if axis == 1 else ys
        if int(values.max()) == int(values.min()):
            continue
        cut = _choose_cut(values, min_target_extent)
        left = values < cut
        right = ~left
        if not left.any() or not right.any():
            continue
        return _split_component(
            xs[left],
            ys[left],
            image_width,
            image_height,
            max_long_side,
            max_short_side,
            max_pixels,
            context_pixels,
            multiple,
            min_target_extent,
            depth + 1,
        ) + _split_component(
            xs[right],
            ys[right],
            image_width,
            image_height,
            max_long_side,
            max_short_side,
            max_pixels,
            context_pixels,
            multiple,
            min_target_extent,
            depth + 1,
        )

    raise ValueError(
        "Target cannot fit the configured tile caps while retaining the requested context"
    )


def build_region_tile_plan(
    mask: torch.Tensor | np.ndarray,
    max_long_side: int = 1536,
    max_short_side: int = 1024,
    max_pixels: int = 1572864,
    context_pixels: int = 192,
    multiple: int = 16,
    support_threshold: float = 0.001,
    min_component_pixels: int = 1,
    min_target_extent: int = 256,
) -> dict[str, Any]:
    if multiple <= 0:
        raise ValueError("multiple must be positive")
    if max_long_side < max_short_side:
        raise ValueError("max_long_side must be greater than or equal to max_short_side")
    if max_long_side % multiple or max_short_side % multiple:
        raise ValueError("tile side caps must be divisible by multiple")
    if context_pixels < 0:
        raise ValueError("context_pixels cannot be negative")

    source = _to_numpy_mask(mask)
    height, width = source.shape
    sanitized = source.copy()
    sanitized[sanitized < support_threshold] = 0.0
    active = sanitized > 0.0
    if not active.any():
        raise ValueError("Mask contains no active pixels at the configured support_threshold")

    structure = ndimage.generate_binary_structure(2, 2)
    labels, component_count_raw = ndimage.label(active, structure=structure)
    object_slices = ndimage.find_objects(labels)

    component_records: list[dict[str, Any]] = []
    dropped_pixels = 0
    for component_id, obj_slice in enumerate(object_slices, start=1):
        if obj_slice is None:
            continue
        local = labels[obj_slice] == component_id
        local_y, local_x = np.nonzero(local)
        pixel_count = int(local_x.size)
        if pixel_count < min_component_pixels:
            dropped_pixels += pixel_count
            continue
        global_y = local_y + obj_slice[0].start
        global_x = local_x + obj_slice[1].start
        component_records.append(
            {
                "component_id": component_id,
                "xs": global_x.astype(np.int32, copy=False),
                "ys": global_y.astype(np.int32, copy=False),
                "bbox": [
                    int(global_x.min()),
                    int(global_y.min()),
                    int(global_x.max()) + 1,
                    int(global_y.max()) + 1,
                ],
                "pixels": pixel_count,
            }
        )

    if not component_records:
        raise ValueError("All mask components were smaller than min_component_pixels")

    component_records.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    planned_target = np.zeros_like(sanitized)
    tiles: list[dict[str, Any]] = []
    component_summaries: list[dict[str, Any]] = []

    for ordered_component_id, component in enumerate(component_records):
        xs = component["xs"]
        ys = component["ys"]
        leaves = _split_component(
            xs,
            ys,
            width,
            height,
            max_long_side,
            max_short_side,
            max_pixels,
            context_pixels,
            multiple,
            min_target_extent,
        )
        leaves = _sort_leaves_along_principal_axis(leaves, xs, ys)

        component_tile_indexes: list[int] = []
        for component_tile_index, (leaf_x, leaf_y, crop, depth) in enumerate(leaves):
            x = crop["x"]
            y = crop["y"]
            tile_width = crop["width"]
            tile_height = crop["height"]
            ownership_mask = np.zeros((tile_height, tile_width), dtype=np.float32)
            ownership_mask[leaf_y - y, leaf_x - x] = sanitized[leaf_y, leaf_x]
            planned_target[leaf_y, leaf_x] = sanitized[leaf_y, leaf_x]

            tile_index = len(tiles)
            component_tile_indexes.append(tile_index)
            tiles.append(
                {
                    "tile_index": tile_index,
                    "component_id": ordered_component_id,
                    "source_component_label": component["component_id"],
                    "component_tile_index": component_tile_index,
                    "x": x,
                    "y": y,
                    "width": tile_width,
                    "height": tile_height,
                    "input_pixels": tile_width * tile_height,
                    "ownership_pixels": int(leaf_x.size),
                    "target_bbox_xyxy": crop["target_bbox_xyxy"],
                    "split_depth": depth,
                    "ownership_mask": ownership_mask,
                }
            )

        component_summaries.append(
            {
                "component_id": ordered_component_id,
                "source_component_label": component["component_id"],
                "bbox_xyxy": component["bbox"],
                "pixels": component["pixels"],
                "tile_indexes": component_tile_indexes,
            }
        )

    union = np.zeros_like(sanitized)
    ownership_count = np.zeros_like(active, dtype=np.uint16)
    for tile in tiles:
        x = tile["x"]
        y = tile["y"]
        tile_width = tile["width"]
        tile_height = tile["height"]
        ownership_mask = tile["ownership_mask"]
        region = union[y : y + tile_height, x : x + tile_width]
        np.maximum(region, ownership_mask, out=region)
        ownership_count[y : y + tile_height, x : x + tile_width] += (
            ownership_mask > 0
        ).astype(np.uint16)

    overlap_pixels = int(np.count_nonzero(ownership_count > 1))
    union_mismatch_pixels = int(np.count_nonzero(union != planned_target))
    if overlap_pixels != 0 or union_mismatch_pixels != 0:
        raise RuntimeError(
            f"Tile ownership invariant failed: overlap={overlap_pixels}, "
            f"union_mismatch={union_mismatch_pixels}"
        )

    report_tiles = [
        {key: value for key, value in tile.items() if key != "ownership_mask"}
        for tile in tiles
    ]
    report = {
        "image_width": width,
        "image_height": height,
        "support_threshold": support_threshold,
        "active_pixels": int(np.count_nonzero(planned_target)),
        "dropped_component_pixels": dropped_pixels,
        "raw_component_count": int(component_count_raw),
        "kept_component_count": len(component_records),
        "tile_count": len(tiles),
        "caps": {
            "max_long_side": max_long_side,
            "max_short_side": max_short_side,
            "max_pixels": max_pixels,
            "context_pixels": context_pixels,
            "multiple": multiple,
            "min_target_extent": min_target_extent,
        },
        "ownership_overlap_pixels": overlap_pixels,
        "ownership_union_mismatch_pixels": union_mismatch_pixels,
        "components": component_summaries,
        "tiles": report_tiles,
    }

    return {
        "tiles": tiles,
        "count": len(tiles),
        "union_mask": union,
        "report": report,
    }


class MaskRegionTilePlanner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "max_long_side": (
                    "INT",
                    {"default": 1536, "min": 256, "max": MAX_RESOLUTION, "step": 16},
                ),
                "max_short_side": (
                    "INT",
                    {"default": 1024, "min": 256, "max": MAX_RESOLUTION, "step": 16},
                ),
                "max_pixels": (
                    "INT",
                    {"default": 1572864, "min": 65536, "max": 67108864, "step": 16384},
                ),
                "context_pixels": (
                    "INT",
                    {"default": 192, "min": 0, "max": 2048, "step": 16},
                ),
                "multiple": (
                    "INT",
                    {"default": 16, "min": 1, "max": 256, "step": 1},
                ),
                "support_threshold": (
                    "FLOAT",
                    {"default": 0.001, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "min_component_pixels": (
                    "INT",
                    {"default": 1, "min": 1, "max": 1000000, "step": 1},
                ),
                "min_target_extent": (
                    "INT",
                    {"default": 256, "min": 1, "max": 4096, "step": 16},
                ),
            }
        }

    RETURN_TYPES = ("REGION_TILE_PLAN", "INT", "MASK", "STRING")
    RETURN_NAMES = ("plan", "tile_count", "ownership_union", "report_json")
    FUNCTION = "plan"
    CATEGORY = "mask/region tiles"
    DESCRIPTION = (
        "Splits an existing mask into context-rich generation crops whose ownership masks "
        "are disjoint and whose union exactly equals the planned target mask."
    )

    def plan(
        self,
        mask,
        max_long_side,
        max_short_side,
        max_pixels,
        context_pixels,
        multiple,
        support_threshold,
        min_component_pixels,
        min_target_extent,
    ):
        plan = build_region_tile_plan(
            mask=mask,
            max_long_side=max_long_side,
            max_short_side=max_short_side,
            max_pixels=max_pixels,
            context_pixels=context_pixels,
            multiple=multiple,
            support_threshold=support_threshold,
            min_component_pixels=min_component_pixels,
            min_target_extent=min_target_extent,
        )
        union = torch.from_numpy(plan["union_mask"].copy()).unsqueeze(0)
        report_json = json.dumps(plan["report"], ensure_ascii=False, indent=2)
        return plan, plan["count"], union, report_json


class MaskRegionTileAtIndex:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": ("REGION_TILE_PLAN",),
                "index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK", "INT", "INT", "INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "ownership_mask",
        "x",
        "y",
        "width",
        "height",
        "component_id",
        "tile_index",
        "metadata_json",
    )
    FUNCTION = "get_tile"
    CATEGORY = "mask/region tiles"

    def get_tile(self, plan, index):
        if index < 0 or index >= plan["count"]:
            raise IndexError(f"Tile index {index} is outside 0..{plan['count'] - 1}")
        tile = plan["tiles"][index]
        ownership_mask = torch.from_numpy(tile["ownership_mask"].copy()).unsqueeze(0)
        metadata = {key: value for key, value in tile.items() if key != "ownership_mask"}
        return (
            ownership_mask,
            tile["x"],
            tile["y"],
            tile["width"],
            tile["height"],
            tile["component_id"],
            tile["tile_index"],
            json.dumps(metadata, ensure_ascii=False, indent=2),
        )


def _image_batch(image: torch.Tensor) -> torch.Tensor:
    value = image.detach().to(dtype=torch.float32)
    if value.ndim == 3:
        value = value.unsqueeze(0)
    if value.ndim != 4 or value.shape[-1] not in (1, 3, 4):
        raise ValueError(f"IMAGE must be BHWC, received shape {tuple(value.shape)}")
    if value.shape[0] != 1:
        raise ValueError("The universal local-edit workflow currently accepts one input image at a time")
    return value[..., :3]


def _mask_batch(mask: torch.Tensor | np.ndarray, height: int, width: int) -> torch.Tensor:
    if isinstance(mask, torch.Tensor):
        value = mask.detach().to(device="cpu", dtype=torch.float32)
    else:
        value = torch.from_numpy(np.asarray(mask, dtype=np.float32))
    if value.ndim == 4:
        if value.shape[-1] == 1:
            value = value[..., 0]
        elif value.shape[1] == 1:
            value = value[:, 0]
        else:
            value = value.amax(dim=-1)
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3:
        raise ValueError(f"MASK must be BHW, received shape {tuple(value.shape)}")
    if value.shape[0] != 1:
        value = value.amax(dim=0, keepdim=True)
    if value.shape[-2:] != (height, width):
        value = torch.nn.functional.interpolate(
            value.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False
        ).squeeze(1)
    return value.clamp(0.0, 1.0)


def _first(value):
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("Expected a non-empty list input")
        return value[0]
    return value


def _grid_positions(size: int, window: int, overlap: int) -> list[int]:
    if size <= window:
        return [0]
    step = window - overlap
    if step <= 0:
        raise ValueError("SAM window overlap must be smaller than the window size")
    positions = list(range(0, size - window + 1, step))
    last = size - window
    if positions[-1] != last:
        positions.append(last)
    return positions


def _flatten_bounding_boxes(value: Any) -> list[dict[str, float]]:
    boxes: list[dict[str, float]] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict) and {"x", "y", "width", "height"}.issubset(item):
            boxes.append(
                {
                    "x": float(item["x"]),
                    "y": float(item["y"]),
                    "width": float(item["width"]),
                    "height": float(item["height"]),
                    "score": float(item.get("score", 1.0)),
                }
            )
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return boxes


def _box_iou(left: dict[str, float], right: dict[str, float]) -> float:
    left_x2 = left["x"] + left["width"]
    left_y2 = left["y"] + left["height"]
    right_x2 = right["x"] + right["width"]
    right_y2 = right["y"] + right["height"]
    intersection_width = max(0.0, min(left_x2, right_x2) - max(left["x"], right["x"]))
    intersection_height = max(0.0, min(left_y2, right_y2) - max(left["y"], right["y"]))
    intersection = intersection_width * intersection_height
    union = left["width"] * left["height"] + right["width"] * right["height"] - intersection
    return intersection / union if union > 0 else 0.0


class BoundingBoxCropBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "bboxes": ("BOUNDING_BOX", {"forceInput": True}),
                "minimum_score": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "context_pixels": (
                    "INT",
                    {"default": 256, "min": 0, "max": 4096, "step": 16},
                ),
                "multiple": (
                    "INT",
                    {"default": 16, "min": 1, "max": 256, "step": 1},
                ),
                "deduplicate_iou": (
                    "FLOAT",
                    {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("native_regions", "x", "y", "width", "height", "report_json")
    OUTPUT_IS_LIST = (True, True, True, True, True, False)
    FUNCTION = "crop"
    CATEGORY = "image/universal local edit"
    DESCRIPTION = "Converts SAM3 full-image detections into scored, deduplicated native-resolution context crops."

    def crop(self, image, bboxes, minimum_score, context_pixels, multiple, deduplicate_iou):
        source = _image_batch(image)
        height, width = int(source.shape[1]), int(source.shape[2])
        candidates = [
            box
            for box in _flatten_bounding_boxes(bboxes)
            if box["score"] >= float(minimum_score) and box["width"] > 0 and box["height"] > 0
        ]
        candidates.sort(key=lambda box: (-box["score"], box["y"], box["x"]))
        kept: list[dict[str, float]] = []
        for candidate in candidates:
            if any(_box_iou(candidate, existing) >= float(deduplicate_iou) for existing in kept):
                continue
            kept.append(candidate)
        if not kept:
            raise ValueError("SAM3 locator returned no bounding boxes above minimum_score")

        regions: list[torch.Tensor] = []
        out_x: list[int] = []
        out_y: list[int] = []
        out_width: list[int] = []
        out_height: list[int] = []
        records: list[dict[str, Any]] = []
        for index, box in enumerate(kept):
            min_x = max(0, min(width - 1, int(math.floor(box["x"]))))
            min_y = max(0, min(height - 1, int(math.floor(box["y"]))))
            max_x = max(min_x + 1, min(width, int(math.ceil(box["x"] + box["width"]))))
            max_y = max(min_y + 1, min(height, int(math.ceil(box["y"] + box["height"]))))
            crop = _make_crop_from_bounds(
                min_x,
                min_y,
                max_x,
                max_y,
                width,
                height,
                int(context_pixels),
                int(multiple),
            )
            if crop is None:
                raise ValueError(f"Unable to create an aligned crop for SAM3 bbox {index}")
            px, py = crop["x"], crop["y"]
            crop_width, crop_height = crop["width"], crop["height"]
            regions.append(source[:, py : py + crop_height, px : px + crop_width, :].clone())
            out_x.append(px)
            out_y.append(py)
            out_width.append(crop_width)
            out_height.append(crop_height)
            records.append(
                {
                    "index": index,
                    "score": box["score"],
                    "detected_bbox_xywh": [box["x"], box["y"], box["width"], box["height"]],
                    "crop_xywh": [px, py, crop_width, crop_height],
                }
            )
        report = {
            "image_width": width,
            "image_height": height,
            "raw_bbox_count": len(_flatten_bounding_boxes(bboxes)),
            "score_filtered_bbox_count": len(candidates),
            "crop_count": len(regions),
            "minimum_score": float(minimum_score),
            "context_pixels": int(context_pixels),
            "multiple": int(multiple),
            "deduplicate_iou": float(deduplicate_iou),
            "regions": records,
        }
        return regions, out_x, out_y, out_width, out_height, json.dumps(report, ensure_ascii=False, indent=2)


class ImageGridWindows:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "window_size": (
                    "INT",
                    {"default": 1536, "min": 256, "max": MAX_RESOLUTION, "step": 16},
                ),
                "overlap": (
                    "INT",
                    {"default": 384, "min": 0, "max": 4096, "step": 16},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("windows", "x", "y", "width", "height", "report_json")
    OUTPUT_IS_LIST = (True, True, True, True, True, False)
    FUNCTION = "split"
    CATEGORY = "image/universal local edit"
    DESCRIPTION = "Splits one native-resolution image into overlapping SAM3 scan windows without resizing."

    def split(self, image, window_size, overlap):
        source = _image_batch(image)
        height, width = int(source.shape[1]), int(source.shape[2])
        effective_window = min(int(window_size), max(height, width))
        if overlap >= effective_window and min(height, width) > 1:
            raise ValueError("SAM window overlap must be smaller than the effective window size")
        xs = _grid_positions(width, min(window_size, width), min(overlap, max(0, min(window_size, width) - 1)))
        ys = _grid_positions(height, min(window_size, height), min(overlap, max(0, min(window_size, height) - 1)))
        windows: list[torch.Tensor] = []
        out_x: list[int] = []
        out_y: list[int] = []
        out_width: list[int] = []
        out_height: list[int] = []
        for y in ys:
            for x in xs:
                crop_width = min(window_size, width - x)
                crop_height = min(window_size, height - y)
                windows.append(source[:, y : y + crop_height, x : x + crop_width, :].clone())
                out_x.append(int(x))
                out_y.append(int(y))
                out_width.append(int(crop_width))
                out_height.append(int(crop_height))
        report = {
            "image_width": width,
            "image_height": height,
            "window_size": int(window_size),
            "overlap": int(overlap),
            "window_count": len(windows),
            "windows": [
                {"index": i, "x": out_x[i], "y": out_y[i], "width": out_width[i], "height": out_height[i]}
                for i in range(len(windows))
            ],
        }
        return windows, out_x, out_y, out_width, out_height, json.dumps(report, ensure_ascii=False, indent=2)


class MaskGridMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "window_masks": ("MASK",),
                "x": ("INT", {"forceInput": True}),
                "y": ("INT", {"forceInput": True}),
                "width": ("INT", {"forceInput": True}),
                "height": ("INT", {"forceInput": True}),
                "detection_threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "minimum_votes": (
                    "INT",
                    {"default": 2, "min": 1, "max": 16, "step": 1},
                ),
                "allow_single_coverage": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("merged_mask", "report_json")
    OUTPUT_IS_LIST = (False, False)
    INPUT_IS_LIST = True
    FUNCTION = "merge"
    CATEGORY = "mask/universal local edit"
    DESCRIPTION = "Merges dynamic SAM3 window masks by overlap-consensus voting, rejecting one-window false positives."

    def merge(
        self,
        original_image,
        window_masks,
        x,
        y,
        width,
        height,
        detection_threshold,
        minimum_votes,
        allow_single_coverage,
    ):
        source = _image_batch(_first(original_image)).to(device="cpu")
        full_height, full_width = int(source.shape[1]), int(source.shape[2])
        threshold = float(_first(detection_threshold))
        required_minimum = int(_first(minimum_votes))
        allow_single = bool(_first(allow_single_coverage))
        lengths = {len(window_masks), len(x), len(y), len(width), len(height)}
        if len(lengths) != 1 or not window_masks:
            raise ValueError("SAM window mask and coordinate list lengths must match and be non-empty")
        merged = torch.zeros((1, full_height, full_width), dtype=torch.float32)
        coverage = torch.zeros((full_height, full_width), dtype=torch.int16)
        votes = torch.zeros((full_height, full_width), dtype=torch.int16)
        for index, mask in enumerate(window_masks):
            px, py = int(x[index]), int(y[index])
            crop_width, crop_height = int(width[index]), int(height[index])
            if px < 0 or py < 0 or px + crop_width > full_width or py + crop_height > full_height:
                raise ValueError(f"SAM window {index} is outside the original image")
            local = _mask_batch(mask, crop_height, crop_width)
            region = merged[:, py : py + crop_height, px : px + crop_width]
            torch.maximum(region, local, out=region)
            coverage[py : py + crop_height, px : px + crop_width] += 1
            votes[py : py + crop_height, px : px + crop_width] += (
                local[0] >= threshold
            ).to(torch.int16)
        if torch.any(coverage == 0):
            raise RuntimeError("SAM scan windows do not cover every input-image pixel")
        required = torch.full_like(coverage, required_minimum)
        if allow_single:
            required = torch.minimum(required, coverage)
        accepted = votes >= required
        merged = torch.where(accepted.unsqueeze(0), merged, torch.zeros_like(merged))
        report = {
            "window_count": len(window_masks),
            "full_width": full_width,
            "full_height": full_height,
            "active_pixels": int(torch.count_nonzero(merged > 0).item()),
            "detection_threshold": threshold,
            "minimum_votes": required_minimum,
            "allow_single_coverage": allow_single,
            "rejected_detected_pixels": int(torch.count_nonzero((votes > 0) & ~accepted).item()),
            "minimum_window_coverage": int(coverage.min().item()),
            "maximum_window_coverage": int(coverage.max().item()),
        }
        return merged, json.dumps(report, ensure_ascii=False, indent=2)


class MaskUnionManualProtect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "automatic_mask": ("MASK",),
                "manual_add_mask": ("MASK",),
                "manual_mode": (
                    [
                        "automatic_plus_add_minus_erase",
                        "automatic_plus_manual",
                        "automatic_only",
                        "manual_only",
                        "automatic_minus_manual",
                    ],
                    {"default": "automatic_plus_add_minus_erase"},
                ),
                "support_threshold": (
                    "FLOAT",
                    {"default": 0.001, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
            },
            "optional": {
                "manual_erase_mask": ("MASK",),
                "protection_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "MASK", "MASK", "IMAGE")
    RETURN_NAMES = (
        "final_target_mask",
        "automatic_mask_resized",
        "manual_mask_resized",
        "protection_mask_resized",
        "overlay",
    )
    FUNCTION = "combine"
    CATEGORY = "mask/universal local edit"
    DESCRIPTION = "Combines SAM3 with LoadImage Mask Editor additions or erasures, then subtracts optional protection."

    def combine(
        self,
        image,
        automatic_mask,
        manual_add_mask,
        manual_mode,
        support_threshold,
        manual_erase_mask=None,
        protection_mask=None,
    ):
        source = _image_batch(image).to(device="cpu")
        height, width = int(source.shape[1]), int(source.shape[2])
        automatic = _mask_batch(automatic_mask, height, width)
        manual = _mask_batch(manual_add_mask, height, width)
        if manual_erase_mask is None:
            erase = torch.zeros_like(automatic)
        else:
            erase = _mask_batch(manual_erase_mask, height, width)
        if protection_mask is None:
            protection = torch.zeros_like(automatic)
        else:
            protection = _mask_batch(protection_mask, height, width)
        if manual_mode == "automatic_plus_add_minus_erase":
            target = torch.clamp(torch.maximum(automatic, manual) - erase, 0.0, 1.0)
        elif manual_mode == "automatic_plus_manual":
            target = torch.maximum(automatic, manual)
        elif manual_mode == "automatic_only":
            target = automatic.clone()
        elif manual_mode == "manual_only":
            target = manual.clone()
        elif manual_mode == "automatic_minus_manual":
            target = torch.clamp(automatic - manual, 0.0, 1.0)
        else:
            raise ValueError(f"Unknown manual mask mode: {manual_mode}")
        target = torch.clamp(target - protection, 0.0, 1.0)
        target[target < float(support_threshold)] = 0.0
        alpha = target.unsqueeze(-1)
        red = torch.zeros_like(source)
        red[..., 0] = 1.0
        overlay = source * (1.0 - alpha * 0.52) + red * (alpha * 0.52)
        return target, automatic, manual, protection, overlay


def _parse_tile_overrides(
    tile_count: int,
    default_value: int,
    text: str,
    *,
    allowed: set[int],
    setting_name: str,
) -> list[int]:
    if default_value not in allowed:
        raise ValueError(f"默认{setting_name}只能是 {sorted(allowed)} 之一")
    values = [default_value] * tile_count
    stripped = text.strip()
    if not stripped:
        return values
    tokens = [token.strip() for token in stripped.replace(";", ",").split(",") if token.strip()]
    assignment_mode = any("=" in token for token in tokens)
    if assignment_mode:
        if not all("=" in token for token in tokens):
            raise ValueError(f"逐块{setting_name}不能混用位置列表和“块号=数值”两种写法")
        for token in tokens:
            index_text, value_text = (part.strip() for part in token.split("=", 1))
            human_index = int(index_text)
            value = int(value_text)
            if human_index < 1 or human_index > tile_count:
                raise ValueError(
                    f"逐块{setting_name}中的块号 {human_index} 无效；当前预览共有 {tile_count} 块，"
                    f"请填写 1 到 {tile_count}"
                )
            if value not in allowed:
                raise ValueError(
                    f"第 {human_index} 块的{setting_name}值 {value} 无效；只能使用 {sorted(allowed)}"
                )
            values[human_index - 1] = value
    else:
        if len(tokens) > tile_count:
            raise ValueError(
                f"逐块{setting_name}填写了 {len(tokens)} 个值，但当前预览只有 {tile_count} 块"
            )
        for index, token in enumerate(tokens):
            value = int(token)
            if value not in allowed:
                raise ValueError(
                    f"第 {index + 1} 块的{setting_name}值 {value} 无效；只能使用 {sorted(allowed)}"
                )
            values[index] = value
    return values


def _parse_expansion_overrides(tile_count: int, default_expand: int, text: str) -> list[int]:
    """Legacy 128/192/224/256 parser retained for existing saved workflows."""
    return _parse_tile_overrides(
        tile_count,
        default_expand,
        text,
        allowed={128, 192, 224, 256},
        setting_name="扩展",
    )


def _preview_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\arial.ttf",
        "DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _label_tile_preview(
    tile_image: torch.Tensor,
    human_index: int,
    width: int,
    height: int,
    grow: int,
    blur: int,
) -> torch.Tensor:
    """Return a preview-only copy with a visible 1-based tile label."""
    batch = _image_batch(tile_image).to(device="cpu", dtype=torch.float32)
    labeled: list[torch.Tensor] = []
    font_size = max(18, min(42, int(min(width, height) * 0.035)))
    font = _preview_font(font_size)
    label = f"块 {human_index}  |  {width}×{height}  |  外扩 {grow}  |  羽化 {blur}"
    for frame in batch:
        array = (frame.clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
        image = Image.fromarray(array)
        draw = ImageDraw.Draw(image)
        box = draw.textbbox((0, 0), label, font=font)
        padding = max(8, font_size // 3)
        box_width = min(image.width, box[2] - box[0] + padding * 2)
        box_height = min(image.height, box[3] - box[1] + padding * 2)
        draw.rectangle((0, 0, box_width, box_height), fill=(12, 18, 24))
        draw.text((padding, padding), label, font=font, fill=(255, 232, 90))
        labeled.append(torch.from_numpy(np.asarray(image).copy()).to(dtype=torch.float32) / 255.0)
    return torch.stack(labeled, dim=0)


class SAMPromptAutoEnglish:
    """Translate Chinese SAM prompts with the already-installed offline Argos model."""

    CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "default": "black cable",
                        "multiline": False,
                        "tooltip": "中文会在节点内部离线翻译成英文；英文原样通过。不会联网或下载模型。",
                    },
                )
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("sam_english_prompt",)
    FUNCTION = "translate"
    CATEGORY = "conditioning/universal local edit"
    DESCRIPTION = "SAM 专用内部离线中译英；英文直接通过，不调用外部程序或网络服务。"

    def translate(self, text):
        source = str(text).strip()
        if not source:
            raise ValueError("SAM 识别对象不能为空")
        if not self.CJK_PATTERN.search(source):
            return (source,)

        try:
            import argostranslate.translate as argos_translate
        except ImportError as exc:
            raise ValueError("未找到内建离线翻译组件 Argos Translate；SAM 中文提示词无法安全翻译") from exc

        languages = {language.code: language for language in argos_translate.get_installed_languages()}
        chinese = languages.get("zh")
        english = languages.get("en")
        if chinese is None or english is None:
            raise ValueError("未找到已安装的中文→英文离线翻译模型；不会自动联网下载")
        try:
            translation = chinese.get_translation(english)
            translated = str(translation.translate(source)).strip()
        except Exception as exc:
            raise ValueError(f"SAM 中文提示词离线翻译失败：{exc}") from exc
        if not translated or self.CJK_PATTERN.search(translated):
            raise ValueError(f"SAM 中文提示词未能完整翻译为英文：{translated or '<空>'}")
        return (translated,)


class LocalEditTileControls:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "edit_instruction": (
                    "STRING",
                    {
                        "default": "移除选中内容，保持其他内容不变。",
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "这里只写要删除或替换成什么；该文本会直接送入 FLUX.2 Klein 编辑流程。",
                    },
                ),
                "tile_profile": (
                    list(TILE_CONTROL_PROFILES),
                    {
                        "default": "标准（已验证）",
                        "tooltip": "手动显存档位；不会伪装成自动显存检测。标准档沿用当前已验证参数。",
                    },
                ),
                "default_grow": (
                    list(GROW_CONTROL_PROFILES),
                    {"default": "标准（32）", "tooltip": "生成范围在目标遮罩外侧增加的像素；与羽化完全独立。"},
                ),
                "tile_grow_overrides": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "例如：2=16,5=64；块号与分块预览一致并从1开始。留空表示全部使用默认外扩。",
                    },
                ),
                "default_blur": (
                    list(BLUR_CONTROL_PROFILES),
                    {"default": "标准（8）", "tooltip": "只控制合成边缘柔和程度，不再随外扩自动增大。"},
                ),
                "tile_blur_overrides": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "例如：2=4,5=16；留空表示全部使用默认羽化。",
                    },
                ),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = (
        "max_long_side",
        "max_short_side",
        "max_pixels",
        "context_pixels",
        "default_grow",
        "tile_grow_overrides",
        "default_blur",
        "tile_blur_overrides",
        "edit_instruction",
    )
    FUNCTION = "resolve"
    CATEGORY = "image/universal local edit"
    DESCRIPTION = "集中输出编辑要求、显存分块档位，以及彼此独立的外扩和羽化设置。"

    def resolve(
        self,
        edit_instruction,
        tile_profile,
        default_grow,
        tile_grow_overrides,
        default_blur,
        tile_blur_overrides,
    ):
        if tile_profile not in TILE_CONTROL_PROFILES:
            raise ValueError(f"未知分块档位：{tile_profile}")
        if default_grow not in GROW_CONTROL_PROFILES:
            raise ValueError(f"未知外扩档位：{default_grow}")
        if default_blur not in BLUR_CONTROL_PROFILES:
            raise ValueError(f"未知羽化档位：{default_blur}")
        max_long_side, max_short_side, max_pixels, context_pixels = TILE_CONTROL_PROFILES[tile_profile]
        return (
            max_long_side,
            max_short_side,
            max_pixels,
            context_pixels,
            GROW_CONTROL_PROFILES[default_grow],
            str(tile_grow_overrides),
            BLUR_CONTROL_PROFILES[default_blur],
            str(tile_blur_overrides),
            str(edit_instruction),
        )


class MaskRegionTileBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "plan": ("REGION_TILE_PLAN",),
                "protection_mask": ("MASK",),
                "default_expand": (["128", "192", "224", "256"], {"default": "128"}),
                "tile_expansion_overrides": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "例如：2=192,5=256；块号与分块预览一致并从1开始。留空表示全部使用默认值。",
                    },
                ),
            }
        }

    RETURN_TYPES = (
        "IMAGE",
        "MASK",
        "MASK",
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "STRING",
    )
    RETURN_NAMES = (
        "tile_images",
        "ownership_masks",
        "protection_crops",
        "x",
        "y",
        "width",
        "height",
        "grow",
        "blur",
        "tile_index",
        "report_json",
    )
    OUTPUT_IS_LIST = (True, True, True, True, True, True, True, True, True, True, False)
    FUNCTION = "prepare"
    CATEGORY = "mask/region tiles"
    DESCRIPTION = "Emits a dynamic list of native-resolution tile images, masks, coordinates, and adjustable 128/192/224/256 settings."

    def prepare(self, image, plan, protection_mask, default_expand, tile_expansion_overrides):
        expansions = _parse_expansion_overrides(plan["count"], int(default_expand), tile_expansion_overrides)
        blurs = [grow // 2 + 1 for grow in expansions]
        return self._prepare_with_values(
            image,
            plan,
            protection_mask,
            expansions,
            blurs,
            {
                "default_expand": int(default_expand),
                "tile_expansion_overrides": tile_expansion_overrides,
                "legacy_coupled_blur": True,
            },
        )

    def _prepare_with_values(
        self,
        image,
        plan,
        protection_mask,
        grow_values: list[int],
        blur_values: list[int],
        report_settings: dict[str, object],
    ):
        source = _image_batch(image).to(device="cpu")
        full_height, full_width = int(source.shape[1]), int(source.shape[2])
        protection = _mask_batch(protection_mask, full_height, full_width)
        if plan["report"]["image_width"] != full_width or plan["report"]["image_height"] != full_height:
            raise ValueError("Tile plan dimensions do not match the input image")
        if len(grow_values) != plan["count"] or len(blur_values) != plan["count"]:
            raise ValueError("外扩或羽化设置数量与分块数量不一致")
        tile_images: list[torch.Tensor] = []
        ownership_masks: list[torch.Tensor] = []
        protection_crops: list[torch.Tensor] = []
        xs: list[int] = []
        ys: list[int] = []
        widths: list[int] = []
        heights: list[int] = []
        grows: list[int] = []
        blurs: list[int] = []
        indexes: list[int] = []
        report_tiles = []
        for tile, grow, blur in zip(plan["tiles"], grow_values, blur_values):
            x, y = int(tile["x"]), int(tile["y"])
            width, height = int(tile["width"]), int(tile["height"])
            tile_images.append(source[:, y : y + height, x : x + width, :].clone())
            ownership_masks.append(torch.from_numpy(tile["ownership_mask"].copy()).unsqueeze(0))
            protection_crops.append(protection[:, y : y + height, x : x + width].clone())
            xs.append(x)
            ys.append(y)
            widths.append(width)
            heights.append(height)
            grows.append(grow)
            blurs.append(blur)
            indexes.append(int(tile["tile_index"]))
            report_tiles.append(
                {
                    "tile_index": int(tile["tile_index"]),
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "grow": grow,
                    "blur": blur,
                }
            )
        report = {
            "tile_count": plan["count"],
            **report_settings,
            "tiles": report_tiles,
        }
        return (
            tile_images,
            ownership_masks,
            protection_crops,
            xs,
            ys,
            widths,
            heights,
            grows,
            blurs,
            indexes,
            json.dumps(report, ensure_ascii=False, indent=2),
        )


class MaskRegionTileBatchControlled(MaskRegionTileBatch):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "plan": ("REGION_TILE_PLAN",),
                "protection_mask": ("MASK",),
                "default_grow": ("INT", {"forceInput": True}),
                "tile_grow_overrides": ("STRING", {"forceInput": True}),
                "default_blur": ("INT", {"forceInput": True}),
                "tile_blur_overrides": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = MaskRegionTileBatch.RETURN_TYPES + ("IMAGE",)
    RETURN_NAMES = MaskRegionTileBatch.RETURN_NAMES + ("labeled_preview_images",)
    OUTPUT_IS_LIST = MaskRegionTileBatch.OUTPUT_IS_LIST + (True,)
    FUNCTION = "prepare_controlled"
    DESCRIPTION = (
        "由集中控制区分别提供外扩和羽化设置，并额外输出带1起始块号、尺寸、外扩和羽化值的预览图；"
        "实际生成仍使用未写字的原始局部块。"
    )

    def prepare_controlled(
        self,
        image,
        plan,
        protection_mask,
        default_grow,
        tile_grow_overrides,
        default_blur,
        tile_blur_overrides,
    ):
        grows = _parse_tile_overrides(
            plan["count"],
            int(default_grow),
            tile_grow_overrides,
            allowed=GROW_OVERRIDE_VALUES,
            setting_name="外扩",
        )
        blurs = _parse_tile_overrides(
            plan["count"],
            int(default_blur),
            tile_blur_overrides,
            allowed=BLUR_OVERRIDE_VALUES,
            setting_name="羽化",
        )
        result = self._prepare_with_values(
            image,
            plan,
            protection_mask,
            grows,
            blurs,
            {
                "default_grow": int(default_grow),
                "tile_grow_overrides": tile_grow_overrides,
                "default_blur": int(default_blur),
                "tile_blur_overrides": tile_blur_overrides,
                "legacy_coupled_blur": False,
            },
        )
        tile_images, _, _, _, _, widths, heights, resolved_grows, resolved_blurs, indexes, _ = result
        previews = [
            _label_tile_preview(tile, index + 1, width, height, grow, blur)
            for tile, index, width, height, grow, blur in zip(
                tile_images,
                indexes,
                widths,
                heights,
                resolved_grows,
                resolved_blurs,
            )
        ]
        return result + (previews,)


class MaskRegionWeightedMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "destination": ("IMAGE",),
                "candidate_tiles": ("IMAGE",),
                "composite_masks": ("MASK",),
                "x": ("INT", {"forceInput": True}),
                "y": ("INT", {"forceInput": True}),
                "width": ("INT", {"forceInput": True}),
                "height": ("INT", {"forceInput": True}),
                "support_cutoff": (
                    "FLOAT",
                    {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "edge_ramp_pixels": (
                    "INT",
                    {"default": 128, "min": 1, "max": 2048, "step": 1},
                ),
                "edge_confidence_floor": (
                    "FLOAT",
                    {"default": 0.05, "min": 0.001, "max": 1.0, "step": 0.001},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("merged_source", "strict_union_mask", "overlap_heatmap", "ownership_map", "report_json")
    OUTPUT_IS_LIST = (False, False, False, False, False)
    INPUT_IS_LIST = True
    FUNCTION = "merge"
    CATEGORY = "image/universal local edit"
    DESCRIPTION = "Combines a dynamic tile list by normalized mask and edge-confidence weights; emits a strict-zero union mask for the final core ImageCompositeMasked node."

    def merge(
        self,
        destination,
        candidate_tiles,
        composite_masks,
        x,
        y,
        width,
        height,
        support_cutoff,
        edge_ramp_pixels,
        edge_confidence_floor,
    ):
        base = _image_batch(_first(destination)).to(device="cpu", dtype=torch.float32)
        full_height, full_width = int(base.shape[1]), int(base.shape[2])
        lengths = {len(candidate_tiles), len(composite_masks), len(x), len(y), len(width), len(height)}
        if len(lengths) != 1 or not candidate_tiles:
            raise ValueError("Candidate tile, mask, and coordinate list lengths must match and be non-empty")
        cutoff = float(_first(support_cutoff))
        ramp_pixels = float(_first(edge_ramp_pixels))
        confidence_floor = float(_first(edge_confidence_floor))
        merged = base.clone()
        accumulated = torch.zeros((full_height, full_width), dtype=torch.float32)
        overlap_count = torch.zeros((full_height, full_width), dtype=torch.int16)
        max_score = torch.zeros((full_height, full_width), dtype=torch.float32)
        owner = torch.full((full_height, full_width), -1, dtype=torch.int16)
        tile_reports = []
        for index, (candidate, mask) in enumerate(zip(candidate_tiles, composite_masks)):
            px, py = int(x[index]), int(y[index])
            tile_width, tile_height = int(width[index]), int(height[index])
            if px < 0 or py < 0 or px + tile_width > full_width or py + tile_height > full_height:
                raise ValueError(f"Candidate tile {index} is outside the destination")
            source = _image_batch(candidate).to(device="cpu", dtype=torch.float32)
            if source.shape[1:3] != (tile_height, tile_width):
                raise ValueError(
                    f"Candidate tile {index} has shape {tuple(source.shape[1:3])}, expected {(tile_height, tile_width)}"
                )
            local_mask = _mask_batch(mask, tile_height, tile_width)[0]
            support = local_mask > cutoff
            yy, xx = torch.meshgrid(
                torch.arange(tile_height, dtype=torch.float32),
                torch.arange(tile_width, dtype=torch.float32),
                indexing="ij",
            )
            edge = torch.minimum(
                torch.minimum(xx + 0.5, yy + 0.5),
                torch.minimum(tile_width - 0.5 - xx, tile_height - 0.5 - yy),
            )
            confidence = confidence_floor + (1.0 - confidence_floor) * torch.clamp(
                edge / ramp_pixels, 0.0, 1.0
            )
            score = (local_mask * confidence) * support.to(dtype=torch.float32)
            accumulated_region = accumulated[py : py + tile_height, px : px + tile_width]
            denominator = accumulated_region + score
            alpha = torch.where(denominator > 0, score / denominator, torch.zeros_like(score))
            destination_region = merged[:, py : py + tile_height, px : px + tile_width, :]
            merged[:, py : py + tile_height, px : px + tile_width, :] = (
                destination_region * (1.0 - alpha[None, ..., None]) + source * alpha[None, ..., None]
            )
            accumulated_region += score
            overlap_count[py : py + tile_height, px : px + tile_width] += support.to(torch.int16)
            max_region = max_score[py : py + tile_height, px : px + tile_width]
            owner_region = owner[py : py + tile_height, px : px + tile_width]
            wins = score > max_region
            max_region[wins] = score[wins]
            owner_region[wins] = index
            tile_reports.append(
                {
                    "tile_index": index,
                    "x": px,
                    "y": py,
                    "width": tile_width,
                    "height": tile_height,
                    "support_pixels": int(torch.count_nonzero(support).item()),
                }
            )
        union = accumulated > 0
        heatmap = base * 0.25
        heatmap[:, overlap_count == 1, :] = torch.tensor((0.10, 0.42, 1.0), dtype=torch.float32)
        heatmap[:, overlap_count > 1, :] = torch.tensor((1.0, 0.10, 0.10), dtype=torch.float32)
        colors = torch.tensor(
            (
                (1.0, 0.25, 0.25),
                (1.0, 0.72, 0.20),
                (0.25, 0.85, 0.42),
                (0.20, 0.55, 1.0),
                (0.72, 0.30, 1.0),
                (0.20, 0.90, 0.90),
            ),
            dtype=torch.float32,
        )
        ownership = torch.full_like(base, 0.05)
        for index in range(len(candidate_tiles)):
            ownership[:, owner == index, :] = colors[index % len(colors)]
        report = {
            "tile_count": len(candidate_tiles),
            "support_cutoff": cutoff,
            "edge_ramp_pixels": ramp_pixels,
            "edge_confidence_floor": confidence_floor,
            "union_support_pixels": int(torch.count_nonzero(union).item()),
            "overlap_pixels": int(torch.count_nonzero(overlap_count > 1).item()),
            "max_overlap_count": int(overlap_count.max().item()),
            "outside_union_is_exact_destination": True,
            "normalized_candidate_weight_sum_in_union": 1.0,
            "tiles": tile_reports,
        }
        return (
            merged.clamp(0.0, 1.0),
            union.to(dtype=torch.float32).unsqueeze(0),
            heatmap.clamp(0.0, 1.0),
            ownership,
            json.dumps(report, ensure_ascii=False, indent=2),
        )


class AppendPreservationPrompt:
    PRESERVATION_SUFFIX = (
        "Keep everything outside the selected mask unchanged. "
        "Inside the selected mask, reconstruct only the requested result and match the surrounding "
        "material, texture, color, lighting, sharpness, and image grain. "
        "Do not introduce unrelated objects, patterns, or materials."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "edit_instruction": (
                    "STRING",
                    {
                        "default": "Remove the selected object.",
                        "multiline": True,
                        "dynamicPrompts": True,
                    },
                )
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("complete_prompt",)
    FUNCTION = "build"
    CATEGORY = "conditioning/universal local edit"
    DESCRIPTION = "Appends the fixed preservation sentence while leaving the requested edit instruction fully user-editable."

    def build(self, edit_instruction):
        instruction = str(edit_instruction).strip()
        if not instruction:
            raise ValueError("Edit instruction cannot be empty")
        if self.PRESERVATION_SUFFIX.lower() in instruction.lower():
            return (instruction,)
        return (f"{instruction.rstrip()} {self.PRESERVATION_SUFFIX}",)
