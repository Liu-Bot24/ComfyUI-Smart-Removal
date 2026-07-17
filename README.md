# ComfyUI Mask Region Tile Planner

ComfyUI nodes for complete native-resolution local editing workflows:

- **Mask Region Tile Planner (Exact)** splits an existing mask into bounded crops with context.
- **Get Region Tile (Exact)** returns one tile's coordinates and local ownership mask by index.
- **SAM3 Detections to Native Regions** converts full-image detections into aligned context crops.
- **Universal SAM3 Scan Windows (Native)** and **Merge SAM3 Window Masks (Exact)** provide an optional overlap-consensus scan path.
- **Universal Local Edit Controls** provides one manual tile-size profile, one default expansion, and optional 1-based per-tile overrides such as `2=192,5=256`.
- **Prepare Controlled Dynamic Region Tiles** keeps the generation tiles unchanged and emits a separate preview copy labeled with the tile number, native size, and effective expansion.
- **Automatic + Manual Mask (Protected)** combines an automatic mask with separate hand-painted add, erase, and protection masks.
- **Prepare Dynamic Region Tiles** emits the entire runtime tile list with a default expansion of 128 and per-tile 128/192/224/256 overrides.
- **Merge Dynamic Region Tiles (Normalized)** resolves overlaps with normalized ownership and emits a strict-zero union mask for the final composite.
- **Edit Instruction + Preserve Suffix** keeps the edit request user-editable and appends the fixed preservation sentence.

The nodes do not resize the source image. Their hard invariants are:

- every generation crop stays within the configured long side, short side, and pixel limits;
- crop width and height are divisible by the configured multiple;
- ownership masks do not overlap;
- the union of ownership masks exactly equals the planned target mask.

Connect the dynamic tile outputs directly to a mapped local generation chain. Feed the generated tile list and compositing-mask list to the normalized merge node, then use ComfyUI's core `ImageCompositeMasked` with `resize_source=false` for the final full-resolution composite.

For manual correction, convert an automatic SAM mask into the alpha channel of the source image with core `JoinImageWithAlpha`, then feed that RGBA image to Impact Pack's `PreviewBridge`. This package adds a visible **打开遮罩编辑器** button to `PreviewBridge`, so the editor does not depend on a hidden context-menu action. A user-facing workflow can expose one PreviewBridge for the target mask and a second, clearly named PreviewBridge for the optional protection mask; both reuse the same uploaded source image and each popup has one unambiguous meaning.

## Install

Place this directory in `ComfyUI/custom_nodes/` and restart ComfyUI.

## Test

```powershell
python -m unittest discover -s tests -v
```
