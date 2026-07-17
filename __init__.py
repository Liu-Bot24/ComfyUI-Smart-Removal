from .nodes import (
    AppendPreservationPrompt,
    BoundingBoxCropBatch,
    ImageGridWindows,
    LocalEditTileControls,
    MaskGridMerge,
    MaskRegionTileAtIndex,
    MaskRegionTileBatch,
    MaskRegionTileBatchControlled,
    MaskRegionTilePlanner,
    MaskRegionWeightedMerge,
    MaskUnionManualProtect,
    SAMPromptAutoEnglish,
)


NODE_CLASS_MAPPINGS = {
    "MaskRegionTilePlannerExact": MaskRegionTilePlanner,
    "MaskRegionTileAtIndexExact": MaskRegionTileAtIndex,
    "UniversalImageGridWindowsExact": ImageGridWindows,
    "UniversalBoundingBoxCropBatchExact": BoundingBoxCropBatch,
    "UniversalMaskGridMergeExact": MaskGridMerge,
    "UniversalMaskUnionManualProtectExact": MaskUnionManualProtect,
    "UniversalRegionTileBatchExact": MaskRegionTileBatch,
    "UniversalRegionTileBatchControlledExact": MaskRegionTileBatchControlled,
    "UniversalLocalEditTileControls": LocalEditTileControls,
    "UniversalRegionWeightedMergeExact": MaskRegionWeightedMerge,
    "UniversalAppendPreservationPrompt": AppendPreservationPrompt,
    "UniversalSAMPromptAutoEnglish": SAMPromptAutoEnglish,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskRegionTilePlannerExact": "Mask Region Tile Planner (Exact)",
    "MaskRegionTileAtIndexExact": "Get Region Tile (Exact)",
    "UniversalImageGridWindowsExact": "Universal SAM3 Scan Windows (Native)",
    "UniversalBoundingBoxCropBatchExact": "SAM3 Detections to Native Regions",
    "UniversalMaskGridMergeExact": "Merge SAM3 Window Masks (Exact)",
    "UniversalMaskUnionManualProtectExact": "Automatic + Manual Mask (Protected)",
    "UniversalRegionTileBatchExact": "Prepare Dynamic Region Tiles",
    "UniversalRegionTileBatchControlledExact": "Prepare Controlled Dynamic Region Tiles",
    "UniversalLocalEditTileControls": "Universal Local Edit Controls",
    "UniversalRegionWeightedMergeExact": "Merge Dynamic Region Tiles (Normalized)",
    "UniversalAppendPreservationPrompt": "Edit Instruction + Preserve Suffix",
    "UniversalSAMPromptAutoEnglish": "SAM Prompt Auto English (Offline)",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
