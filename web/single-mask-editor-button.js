import { ComfyApp, app } from "../../scripts/app.js";


app.registerExtension({
    name: "MaskRegionTilePlanner.SingleMaskEditorButton",

    nodeCreated(node) {
        if (node.comfyClass !== "PreviewBridge") {
            return;
        }

        node.addWidget("button", "打开遮罩编辑器", null, () => {
            ComfyApp.copyToClipspace(node);
            ComfyApp.clipspace_return_node = node;
            ComfyApp.open_maskeditor();
        });
    },
});
