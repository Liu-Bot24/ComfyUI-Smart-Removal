import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "nodes.py"
SPEC = importlib.util.spec_from_file_location("mask_region_tile_wire_nodes", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _draw_curved_wire(mask, x_start, x_stop, center_y, amplitude, thickness):
    """Draw one connected, gently curved cable without OpenCV test helpers."""
    for x in range(x_start, x_stop):
        y = int(round(center_y + amplitude * np.sin((x - x_start) / 71.0)))
        mask[y - thickness : y + thickness + 1, x] = 1.0


def _gradient_destination(height, width):
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, height),
        torch.linspace(0.0, 1.0, width),
        indexing="ij",
    )
    return torch.stack((xx, yy, (xx + yy) * 0.25), dim=-1).unsqueeze(0)


def _prepare_wire_case(mask):
    height, width = mask.shape
    plan = MODULE.build_region_tile_plan(
        mask,
        max_long_side=1024,
        max_short_side=512,
        max_pixels=524288,
        context_pixels=192,
        min_target_extent=128,
    )
    destination = _gradient_destination(height, width)
    prepared = MODULE.MaskRegionTileBatchControlled().prepare_controlled(
        destination,
        plan,
        torch.zeros((1, height, width), dtype=torch.float32),
        32,
        "",
        4,
        "",
    )
    return plan, destination, prepared


def _candidate_color(index):
    palette = (
        (0.95, 0.05, 0.10),
        (0.05, 0.20, 0.95),
        (0.10, 0.90, 0.20),
        (0.90, 0.15, 0.85),
        (0.95, 0.75, 0.05),
        (0.05, 0.85, 0.90),
    )
    return torch.tensor(palette[index % len(palette)], dtype=torch.float32)


def _merge_synthetic_candidates(destination, prepared, edge_ramp_pixels=8):
    tile_images = prepared[0]
    ownership_masks = prepared[1]
    xs, ys, widths, heights = prepared[3:7]
    shared_masks = prepared[12]
    candidates = []
    for index, tile in enumerate(tile_images):
        color = _candidate_color(index)
        candidates.append(color.view(1, 1, 1, 3).expand_as(tile).clone())
    result = MODULE.MaskRegionWeightedMerge().merge(
        [destination],
        candidates,
        shared_masks,
        xs,
        ys,
        widths,
        heights,
        [0.02],
        [edge_ramp_pixels],
        [0.05],
        ownership_masks=ownership_masks,
    )
    return candidates, result


def _expected_owner_and_seam(prepared, cutoff, edge_ramp_pixels):
    ownership_masks = prepared[1]
    xs, ys, widths, heights = prepared[3:7]
    shared_masks = prepared[12]
    full_height, full_width = prepared[13].shape[1:]

    coverage = np.zeros((full_height, full_width), dtype=np.uint16)
    core_owner = np.full((full_height, full_width), -1, dtype=np.int16)
    propagated_owner = np.full((full_height, full_width), -1, dtype=np.int16)
    best_distance = np.full((full_height, full_width), np.inf, dtype=np.float32)

    for index, (owner_mask, shared_mask, px, py, tile_width, tile_height) in enumerate(
        zip(ownership_masks, shared_masks, xs, ys, widths, heights)
    ):
        support = shared_mask[0].numpy() > cutoff
        core = (owner_mask[0].numpy() > 0.5) & support
        core_region = core_owner[py : py + tile_height, px : px + tile_width]
        if np.any((core_region >= 0) & core):
            raise AssertionError("test fixture contains overlapping ownership cores")
        core_region[core] = index

        coverage[py : py + tile_height, px : px + tile_width] += support.astype(
            np.uint16, copy=False
        )
        local_distance = MODULE.ndimage.distance_transform_edt(~core).astype(
            np.float32, copy=False
        )
        distance_region = best_distance[py : py + tile_height, px : px + tile_width]
        owner_region = propagated_owner[py : py + tile_height, px : px + tile_width]
        wins = support & (local_distance < distance_region)
        distance_region[wins] = local_distance[wins]
        owner_region[wins] = index

    union = coverage > 0
    propagated_owner[core_owner >= 0] = core_owner[core_owner >= 0]
    seam = MODULE._ownership_seam_mask(propagated_owner) & union
    seam_radius = min(
        MODULE.SEAM_GUARD_PIXELS,
        max(1, int(round(edge_ramp_pixels / 2.0))),
    )
    seam_band = (
        MODULE._expand_binary_mask(seam, seam_radius)
        & union
        & (coverage >= 2)
    )
    return core_owner, propagated_owner, union, seam_band


def _assert_exact_owner_merge(testcase, destination, candidates, prepared, merge_result):
    merged, union_tensor, _, _, report_json = merge_result
    report = json.loads(report_json)
    core_owner, propagated_owner, expected_union, seam_band = _expected_owner_and_seam(
        prepared,
        cutoff=0.02,
        edge_ramp_pixels=8,
    )

    actual_union = union_tensor[0].numpy() > 0.5
    testcase.assertTrue(np.array_equal(actual_union, expected_union))
    testcase.assertTrue(torch.equal(merged[0][~union_tensor[0].bool()], destination[0][~union_tensor[0].bool()]))

    mixed_from_owner = np.zeros(expected_union.shape, dtype=bool)
    for index, candidate in enumerate(candidates):
        expected_color = candidate[0, 0, 0]
        owned = propagated_owner == index
        if np.any(owned):
            different = torch.any(
                torch.abs(merged[0] - expected_color) > 1.0e-6,
                dim=-1,
            ).numpy()
            mixed_from_owner |= owned & different

        exact_core = (core_owner == index) & ~seam_band
        testcase.assertGreater(int(np.count_nonzero(exact_core)), 0)
        testcase.assertTrue(
            torch.allclose(
                merged[0][torch.from_numpy(exact_core)],
                expected_color.expand(int(np.count_nonzero(exact_core)), 3),
                atol=0.0,
                rtol=0.0,
            )
        )

    testcase.assertFalse(np.any(mixed_from_owner & ~seam_band))
    testcase.assertEqual(report["weighting"], "exclusive_owner_with_explicit_seam_crossfade")
    testcase.assertEqual(report["owner_core_overlap_pixels"], 0)
    testcase.assertEqual(report["outside_seam_mixed_pixels"], 0)
    testcase.assertTrue(report["outside_union_is_exact_destination"])
    return mixed_from_owner, seam_band, report


class WireAlgorithmRegressionTests(unittest.TestCase):
    def test_long_continuous_wire_crosses_tiles_with_only_narrow_seam_blending(self):
        mask = np.zeros((384, 1800), dtype=np.float32)
        _draw_curved_wire(mask, 80, 1720, center_y=190, amplitude=18, thickness=4)

        plan, destination, prepared = _prepare_wire_case(mask)
        self.assertGreater(plan["count"], 1)
        self.assertEqual(plan["report"]["raw_component_count"], 1)
        self.assertTrue(np.array_equal(plan["union_mask"], mask))
        self.assertEqual(plan["report"]["ownership_overlap_pixels"], 0)
        self.assertEqual(plan["report"]["ownership_union_mismatch_pixels"], 0)

        prepare_report = json.loads(prepared[10])
        self.assertGreater(prepare_report["shared_generation"]["overlap_pixels"], 0)
        self.assertEqual(
            prepare_report["shared_generation"]["seam_guard_single_coverage_pixels"],
            0,
        )

        candidates, merge_result = _merge_synthetic_candidates(destination, prepared)
        mixed, seam_band, report = _assert_exact_owner_merge(
            self, destination, candidates, prepared, merge_result
        )
        self.assertGreater(int(np.count_nonzero(mixed)), 0)
        self.assertTrue(np.all(mixed <= seam_band))
        self.assertEqual(report["seam_half_width_pixels"], 4)
        self.assertLess(report["seam_band_pixels"], report["union_support_pixels"])

    def test_two_far_apart_wires_stay_separate_and_each_is_covered_exactly(self):
        mask = np.zeros((384, 1800), dtype=np.float32)
        _draw_curved_wire(mask, 80, 650, center_y=100, amplitude=10, thickness=4)
        _draw_curved_wire(mask, 1150, 1720, center_y=280, amplitude=12, thickness=4)

        plan, destination, prepared = _prepare_wire_case(mask)
        self.assertEqual(plan["report"]["raw_component_count"], 2)
        self.assertEqual(plan["report"]["kept_component_count"], 2)
        self.assertEqual(plan["report"]["cluster_count"], 2)
        self.assertEqual(plan["count"], 2)
        self.assertTrue(np.array_equal(plan["union_mask"], mask))

        labels = [tuple(tile["source_component_labels"]) for tile in plan["tiles"]]
        self.assertEqual(len(set(labels)), 2)
        for component in plan["report"]["components"]:
            self.assertEqual(len(component["tile_indexes"]), 1)

        candidates, merge_result = _merge_synthetic_candidates(destination, prepared)
        mixed, seam_band, report = _assert_exact_owner_merge(
            self, destination, candidates, prepared, merge_result
        )
        self.assertEqual(int(np.count_nonzero(seam_band)), 0)
        self.assertEqual(int(np.count_nonzero(mixed)), 0)
        self.assertEqual(report["seam_band_pixels"], 0)
        self.assertEqual(report["overlap_pixels"], 0)


if __name__ == "__main__":
    unittest.main()
