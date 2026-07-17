import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "nodes.py"
SPEC = importlib.util.spec_from_file_location("mask_region_tile_universal_nodes", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class UniversalWorkflowNodeTests(unittest.TestCase):
    def test_sam_locator_boxes_become_native_aligned_context_crops(self):
        image = torch.zeros((1, 700, 1000, 3), dtype=torch.float32)
        boxes = [[
            {"x": 100.3, "y": 200.2, "width": 400.1, "height": 80.6, "score": 0.92},
            {"x": 103.0, "y": 202.0, "width": 398.0, "height": 79.0, "score": 0.88},
            {"x": 700.0, "y": 50.0, "width": 100.0, "height": 100.0, "score": 0.2},
        ]]
        regions, xs, ys, widths, heights, report = MODULE.BoundingBoxCropBatch().crop(
            image,
            boxes,
            minimum_score=0.5,
            context_pixels=64,
            multiple=16,
            deduplicate_iou=0.85,
        )
        self.assertEqual(len(regions), 1)
        self.assertEqual(widths[0] % 16, 0)
        self.assertEqual(heights[0] % 16, 0)
        self.assertLessEqual(xs[0], 100)
        self.assertLessEqual(ys[0], 200)
        self.assertGreaterEqual(xs[0] + widths[0], 501)
        self.assertGreaterEqual(ys[0] + heights[0], 281)
        self.assertEqual(tuple(regions[0].shape[1:3]), (heights[0], widths[0]))
        self.assertIn('"crop_count": 1', report)

    def test_native_scan_windows_cover_arbitrary_image_without_resizing(self):
        image = torch.zeros((1, 100, 250, 3), dtype=torch.float32)
        windows, xs, ys, widths, heights, _ = MODULE.ImageGridWindows().split(
            image, window_size=128, overlap=32
        )
        coverage = np.zeros((100, 250), dtype=np.uint8)
        for window, x, y, width, height in zip(windows, xs, ys, widths, heights):
            self.assertEqual(tuple(window.shape[1:3]), (height, width))
            coverage[y : y + height, x : x + width] += 1
        self.assertTrue(np.all(coverage > 0))
        self.assertGreater(len(windows), 1)

    def test_sam_window_masks_merge_back_to_exact_full_coordinates(self):
        image = torch.zeros((1, 6, 8, 3), dtype=torch.float32)
        masks = [
            torch.ones((1, 4, 5)),
            torch.full((1, 4, 5), 0.5),
            torch.full((1, 4, 5), 0.25),
            torch.full((1, 4, 5), 0.5),
        ]
        merged, _ = MODULE.MaskGridMerge().merge(
            [image],
            masks,
            [0, 3, 0, 3],
            [0, 0, 2, 2],
            [5, 5, 5, 5],
            [4, 4, 4, 4],
            [0.2],
            [1],
            [True],
        )
        self.assertEqual(tuple(merged.shape), (1, 6, 8))
        self.assertTrue(torch.all(merged[:, 0:4, 0:5] >= 1.0))
        self.assertTrue(torch.all(merged[:, 4:6, 3:8] == 0.5))

    def test_automatic_manual_union_and_protection_subtraction(self):
        image = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
        automatic = torch.zeros((1, 8, 8))
        manual = torch.zeros((1, 8, 8))
        protection = torch.zeros((1, 8, 8))
        automatic[:, 2:4, 1:5] = 1.0
        manual[:, 5:7, 4:7] = 1.0
        protection[:, 2:3, 2:4] = 1.0
        target, auto_out, manual_out, protect_out, overlay = MODULE.MaskUnionManualProtect().combine(
            image,
            automatic,
            manual,
            "automatic_plus_manual",
            0.001,
            protection_mask=protection,
        )
        self.assertEqual(int(torch.count_nonzero(target).item()), 12)
        self.assertTrue(torch.all(target[:, 2:3, 2:4] == 0))
        self.assertTrue(torch.equal(auto_out, automatic))
        self.assertTrue(torch.equal(manual_out, manual))
        self.assertTrue(torch.equal(protect_out, protection))
        self.assertEqual(tuple(overlay.shape), tuple(image.shape))

    def test_sam_consensus_rejects_a_single_window_false_positive(self):
        image = torch.zeros((1, 4, 6, 3), dtype=torch.float32)
        left = torch.zeros((1, 4, 4), dtype=torch.float32)
        right = torch.zeros((1, 4, 4), dtype=torch.float32)
        left[:, 1:3, 2:4] = 1.0
        right[:, 1:3, 0:2] = 1.0
        left[:, 0, 2] = 1.0
        merged, report = MODULE.MaskGridMerge().merge(
            [image],
            [left, right],
            [0, 2],
            [0, 0],
            [4, 4],
            [4, 4],
            [0.5],
            [2],
            [False],
        )
        self.assertTrue(torch.all(merged[:, 1:3, 2:4] == 1.0))
        self.assertEqual(float(merged[0, 0, 2]), 0.0)
        self.assertIn('"rejected_detected_pixels": 1', report)

    def test_manual_mask_can_erase_automatic_false_positive(self):
        image = torch.zeros((1, 4, 4, 3), dtype=torch.float32)
        automatic = torch.ones((1, 4, 4), dtype=torch.float32)
        correction = torch.zeros((1, 4, 4), dtype=torch.float32)
        correction[:, 1:3, 1:3] = 1.0
        target, *_ = MODULE.MaskUnionManualProtect().combine(
            image,
            automatic,
            correction,
            "automatic_minus_manual",
            0.001,
        )
        self.assertTrue(torch.all(target[:, 1:3, 1:3] == 0.0))
        self.assertEqual(int(torch.count_nonzero(target).item()), 12)

    def test_manual_add_and_erase_are_applied_in_one_pass(self):
        image = torch.zeros((1, 5, 5, 3), dtype=torch.float32)
        automatic = torch.zeros((1, 5, 5), dtype=torch.float32)
        addition = torch.zeros((1, 5, 5), dtype=torch.float32)
        erasure = torch.zeros((1, 5, 5), dtype=torch.float32)
        automatic[:, 1:3, 1:3] = 1.0
        addition[:, 3:5, 3:5] = 1.0
        erasure[:, 1, 1] = 1.0
        target, *_ = MODULE.MaskUnionManualProtect().combine(
            image,
            automatic,
            addition,
            "automatic_plus_add_minus_erase",
            0.001,
            manual_erase_mask=erasure,
        )
        self.assertEqual(float(target[0, 1, 1]), 0.0)
        self.assertTrue(torch.all(target[:, 3:5, 3:5] == 1.0))
        self.assertEqual(int(torch.count_nonzero(target).item()), 7)

    def test_tile_expansion_defaults_and_per_tile_overrides(self):
        self.assertEqual(MODULE._parse_expansion_overrides(4, 128, ""), [128, 128, 128, 128])
        self.assertEqual(
            MODULE._parse_expansion_overrides(4, 128, "1=192,3=256"),
            [192, 128, 256, 128],
        )
        self.assertEqual(
            MODULE._parse_expansion_overrides(4, 128, "128,192,224"),
            [128, 192, 224, 128],
        )

    def test_invalid_tile_expansion_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "只能使用"):
            MODULE._parse_expansion_overrides(3, 128, "1=160")

    def test_tile_expansion_rejects_out_of_range_human_block_number(self):
        with self.assertRaisesRegex(ValueError, "块号 0 无效.*1 到 3"):
            MODULE._parse_expansion_overrides(3, 128, "0=192")
        with self.assertRaisesRegex(ValueError, "块号 4 无效.*1 到 3"):
            MODULE._parse_expansion_overrides(3, 128, "4=192")

    def test_local_edit_tile_controls_resolve_manual_profiles(self):
        controls = MODULE.LocalEditTileControls()
        self.assertEqual(
            controls.resolve(
                "移除纹身，保持其他内容不变。",
                "标准（已验证）",
                "标准（32）",
                "2=64",
                "标准（8）",
                "2=16",
            ),
            (1536, 1024, 1572864, 192, 32, "2=64", 8, "2=16", "移除纹身，保持其他内容不变。"),
        )
        self.assertEqual(
            controls.resolve("测试", "保守（小块）", "精细（8）", "", "硬边（0）", ""),
            (1024, 768, 786432, 160, 8, "", 0, "", "测试"),
        )
        self.assertEqual(
            controls.resolve("测试", "大块（高显存）", "大范围（128）", "1=192", "柔和（16）", "1=24"),
            (2048, 1280, 2621440, 256, 128, "1=192", 16, "1=24", "测试"),
        )

    def test_independent_grow_and_blur_overrides(self):
        self.assertEqual(
            MODULE._parse_tile_overrides(
                3, 32, "2=64", allowed=MODULE.GROW_OVERRIDE_VALUES, setting_name="外扩"
            ),
            [32, 64, 32],
        )
        self.assertEqual(
            MODULE._parse_tile_overrides(
                3, 8, "2=4", allowed=MODULE.BLUR_OVERRIDE_VALUES, setting_name="羽化"
            ),
            [8, 4, 8],
        )

    def test_invalid_independent_blur_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "羽化值 7 无效"):
            MODULE._parse_tile_overrides(
                2, 8, "1=7", allowed=MODULE.BLUR_OVERRIDE_VALUES, setting_name="羽化"
            )

    def test_dynamic_tile_batch_matches_planner_coordinates(self):
        mask = np.zeros((512, 1800), dtype=np.float32)
        mask[220:236, 80:1720] = 1.0
        plan = MODULE.build_region_tile_plan(
            mask,
            max_long_side=768,
            max_short_side=512,
            max_pixels=393216,
            context_pixels=96,
            min_target_extent=128,
        )
        image = torch.rand((1, 512, 1800, 3), dtype=torch.float32)
        protection = torch.zeros((1, 512, 1800), dtype=torch.float32)
        result = MODULE.MaskRegionTileBatch().prepare(image, plan, protection, "128", "2=192")
        tile_images, ownership, _, xs, ys, widths, heights, grows, blurs, indexes, _ = result
        self.assertEqual(len(tile_images), plan["count"])
        self.assertEqual(grows[0], 128)
        self.assertEqual(grows[1], 192)
        self.assertEqual(blurs[1], 97)
        for index, tile in enumerate(plan["tiles"]):
            self.assertEqual((xs[index], ys[index], widths[index], heights[index]), (
                tile["x"], tile["y"], tile["width"], tile["height"]
            ))
            self.assertEqual(tuple(tile_images[index].shape[1:3]), (tile["height"], tile["width"]))
            self.assertEqual(tuple(ownership[index].shape[1:]), (tile["height"], tile["width"]))
            self.assertEqual(indexes[index], index)

    def test_controlled_tile_batch_adds_labeled_preview_without_changing_raw_tiles(self):
        mask = np.zeros((256, 768), dtype=np.float32)
        mask[120:136, 40:728] = 1.0
        plan = MODULE.build_region_tile_plan(
            mask,
            max_long_side=512,
            max_short_side=256,
            max_pixels=131072,
            context_pixels=64,
            min_target_extent=128,
        )
        image = torch.full((1, 256, 768, 3), 0.5, dtype=torch.float32)
        protection = torch.zeros((1, 256, 768), dtype=torch.float32)
        result = MODULE.MaskRegionTileBatchControlled().prepare_controlled(
            image, plan, protection, 32, "2=64", 8, "2=4"
        )
        raw_tiles = result[0]
        grows = result[7]
        blurs = result[8]
        previews = result[-1]
        self.assertEqual(len(previews), plan["count"])
        self.assertEqual(grows[0], 32)
        self.assertEqual(blurs[0], 8)
        if plan["count"] > 1:
            self.assertEqual(grows[1], 64)
            self.assertEqual(blurs[1], 4)
        self.assertTrue(torch.all(raw_tiles[0] == 0.5))
        self.assertFalse(torch.equal(previews[0], raw_tiles[0]))
        self.assertEqual(tuple(previews[0].shape), tuple(raw_tiles[0].shape))

    def test_weighted_merge_has_strict_outside_and_normalized_overlap(self):
        destination = torch.full((1, 6, 8, 3), 0.2, dtype=torch.float32)
        candidate0 = torch.full((1, 2, 4, 3), 0.4, dtype=torch.float32)
        candidate1 = torch.full((1, 2, 4, 3), 0.8, dtype=torch.float32)
        masks = [torch.ones((1, 2, 4)), torch.ones((1, 2, 4))]
        merged, union, _, _, report = MODULE.MaskRegionWeightedMerge().merge(
            [destination],
            [candidate0, candidate1],
            masks,
            [1, 3],
            [2, 2],
            [4, 4],
            [2, 2],
            [0.02],
            [128],
            [0.05],
        )
        self.assertTrue(torch.equal(merged[:, 0:2], destination[:, 0:2]))
        self.assertTrue(torch.equal(merged[:, 4:6], destination[:, 4:6]))
        self.assertTrue(torch.all(union[:, 2:4, 1:7] == 1))
        self.assertTrue(torch.all(union[:, :, 0] == 0))
        overlap = merged[:, 2:4, 3:5]
        self.assertTrue(torch.all(overlap > 0.4))
        self.assertTrue(torch.all(overlap < 0.8))
        self.assertIn('"normalized_candidate_weight_sum_in_union": 1.0', report)

    def test_prompt_suffix_is_fixed_but_instruction_is_user_editable(self):
        node = MODULE.AppendPreservationPrompt()
        prompt = node.build("Replace the selected tattoo with natural skin.")[0]
        self.assertTrue(prompt.startswith("Replace the selected tattoo with natural skin."))
        self.assertIn("Keep everything outside the selected mask unchanged.", prompt)
        self.assertIn("match the surrounding material, texture, color, lighting, sharpness", prompt)
        self.assertIn("Do not introduce unrelated objects", prompt)

    def test_sam_prompt_english_passes_through_unchanged(self):
        self.assertEqual(MODULE.SAMPromptAutoEnglish().translate("black cable"), ("black cable",))

    def test_sam_prompt_chinese_uses_installed_offline_translation(self):
        try:
            translated = MODULE.SAMPromptAutoEnglish().translate("手臂上的纹身")[0]
        except ValueError as exc:
            if "未找到内建离线翻译组件" in str(exc):
                self.skipTest("Argos Translate is not installed in this Python runtime")
            raise
        self.assertIn("tattoo", translated.lower())
        self.assertNotRegex(translated, MODULE.SAMPromptAutoEnglish.CJK_PATTERN)


if __name__ == "__main__":
    unittest.main()
