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

    def test_disconnected_components_remain_distinct_and_exact(self):
        mask = np.zeros((1000, 1800), dtype=np.float32)
        mask[100:130, 100:900] = 1.0
        mask[700:760, 1200:1650] = 0.5
        plan = build_region_tile_plan(mask, context_pixels=128, min_target_extent=64)
        self.assertEqual(plan["report"]["kept_component_count"], 2)
        self.assertTrue(np.array_equal(plan["union_mask"], mask))
        assert_plan_invariants(plan)

    def test_empty_mask_is_rejected_instead_of_creating_a_bogus_tile(self):
        mask = np.zeros((512, 512), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "no active pixels"):
            build_region_tile_plan(mask)


if __name__ == "__main__":
    unittest.main()
