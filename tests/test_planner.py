import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "nodes.py"
SPEC = importlib.util.spec_from_file_location("mask_region_tile_nodes", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
build_region_tile_plan = MODULE.build_region_tile_plan


def assert_plan_invariants(plan):
    report = plan["report"]
    assert report["ownership_overlap_pixels"] == 0
    assert report["ownership_union_mismatch_pixels"] == 0
    assert plan["count"] == len(plan["tiles"])
    for tile in plan["tiles"]:
        assert tile["width"] % report["caps"]["multiple"] == 0
        assert tile["height"] % report["caps"]["multiple"] == 0
        assert max(tile["width"], tile["height"]) <= report["caps"]["max_long_side"]
        assert min(tile["width"], tile["height"]) <= report["caps"]["max_short_side"]
        assert tile["input_pixels"] <= report["caps"]["max_pixels"]


class PlannerTests(unittest.TestCase):
    def test_small_component_stays_single_tile_and_preserves_soft_values(self):
        mask = np.zeros((512, 768), dtype=np.float32)
        mask[220:235, 180:500] = 0.75
        plan = build_region_tile_plan(mask, context_pixels=96, min_target_extent=64)
        self.assertEqual(plan["count"], 1)
        self.assertTrue(np.array_equal(plan["union_mask"], mask))
        assert_plan_invariants(plan)

    def test_long_thin_target_splits_without_overlap_or_gaps(self):
        mask = np.zeros((1400, 4200), dtype=np.float32)
        for x in range(200, 4000):
            y = 300 + x // 5
            mask[y : y + 12, x] = 1.0
        plan = build_region_tile_plan(
            mask,
            max_long_side=1536,
            max_short_side=1024,
            max_pixels=1572864,
            context_pixels=192,
            min_target_extent=256,
        )
        self.assertGreater(plan["count"], 1)
        self.assertTrue(np.array_equal(plan["union_mask"], mask))
        assert_plan_invariants(plan)

    def test_slightly_over_profile_splits_into_two_balanced_tiles(self):
        mask = np.zeros((1800, 2600), dtype=np.float32)
        mask[600:1000, 200:2300] = 1.0
        plan = build_region_tile_plan(
            mask,
            max_long_side=1536,
            max_short_side=1024,
            max_pixels=1572864,
            context_pixels=192,
            min_target_extent=256,
        )
        self.assertEqual(plan["count"], 2)
        tile_areas = [tile["input_pixels"] for tile in plan["tiles"]]
        self.assertGreaterEqual(min(tile_areas) / max(tile_areas), 0.75)
        self.assertTrue(np.array_equal(plan["union_mask"], mask))
        assert_plan_invariants(plan)

    def test_disconnected_components_remain_distinct_and_exact(self):
        mask = np.zeros((1000, 1800), dtype=np.float32)
        mask[100:130, 100:900] = 1.0
        mask[700:760, 1200:1650] = 0.5
        plan = build_region_tile_plan(mask, context_pixels=128, min_target_extent=64)
        self.assertEqual(plan["report"]["kept_component_count"], 2)
        self.assertEqual(plan["report"]["cluster_count"], 2)
        self.assertEqual(plan["count"], 2)
        self.assertTrue(np.array_equal(plan["union_mask"], mask))
        assert_plan_invariants(plan)

    def test_nearby_disconnected_fragments_merge_into_one_large_profile_tile(self):
        mask = np.zeros((1600, 2000), dtype=np.float32)
        mask[400:450, 400:500] = 1.0
        mask[520:570, 650:750] = 1.0
        mask[650:710, 820:920] = 1.0
        mask[790:850, 1000:1080] = 1.0
        plan = build_region_tile_plan(
            mask,
            max_long_side=2048,
            max_short_side=1536,
            max_pixels=3145728,
            context_pixels=256,
            min_target_extent=256,
        )
        self.assertEqual(plan["report"]["kept_component_count"], 4)
        self.assertEqual(plan["report"]["cluster_count"], 1)
        self.assertEqual(plan["count"], 1)
        self.assertTrue(np.array_equal(plan["union_mask"], mask))
        assert_plan_invariants(plan)

    def test_large_profile_keeps_a_nearby_fragment_group_in_one_tile(self):
        mask = np.zeros((2000, 2400), dtype=np.float32)
        mask[300:360, 300:500] = 1.0
        mask[600:680, 800:1000] = 1.0
        mask[1000:1080, 1300:1500] = 1.0
        plan = build_region_tile_plan(
            mask,
            max_long_side=2048,
            max_short_side=1536,
            max_pixels=3145728,
            context_pixels=256,
            min_target_extent=256,
        )
        self.assertEqual(plan["report"]["kept_component_count"], 3)
        self.assertEqual(plan["report"]["cluster_count"], 1)
        self.assertEqual(plan["count"], 1)
        self.assertTrue(np.array_equal(plan["union_mask"], mask))
        assert_plan_invariants(plan)

    def test_planner_chooses_combined_split_when_it_uses_fewer_tiles(self):
        mask = np.zeros((1800, 2400), dtype=np.float32)
        mask[300:1000, 300:1000] = 1.0
        mask[300:550, 1200:1550] = 1.0
        plan = build_region_tile_plan(
            mask,
            max_long_side=1536,
            max_short_side=1024,
            max_pixels=1572864,
            context_pixels=192,
            min_target_extent=256,
        )
        self.assertEqual(plan["report"]["cluster_count"], 1)
        self.assertEqual(plan["count"], 2)
        self.assertTrue(np.array_equal(plan["union_mask"], mask))
        assert_plan_invariants(plan)

    def test_planner_chooses_split_then_merge_when_it_uses_fewer_tiles(self):
        mask = np.zeros((2400, 3800), dtype=np.float32)
        rectangles = [
            (2540, 2826, 182, 375),
            (1625, 2507, 390, 801),
            (2597, 3504, 632, 987),
            (1379, 1623, 801, 889),
            (0, 896, 1446, 1703),
        ]
        for x1, x2, y1, y2 in rectangles:
            mask[y1:y2, x1:x2] = 1.0
        plan = build_region_tile_plan(
            mask,
            max_long_side=1024,
            max_short_side=768,
            max_pixels=786432,
            context_pixels=128,
            min_target_extent=128,
        )
        self.assertEqual(plan["report"]["cluster_count"], 2)
        self.assertEqual(plan["count"], 6)
        self.assertTrue(
            any(tile["source_component_labels"] == [2, 4] for tile in plan["tiles"])
        )
        self.assertTrue(np.array_equal(plan["union_mask"], mask))
        assert_plan_invariants(plan)

    def test_empty_mask_is_rejected_instead_of_creating_a_bogus_tile(self):
        mask = np.zeros((512, 512), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "no active pixels"):
            build_region_tile_plan(mask)


if __name__ == "__main__":
    unittest.main()
