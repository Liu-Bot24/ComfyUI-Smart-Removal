# ComfyUI 通用智能消除

这是一套面向高分辨率图片的局部删除/替换工作流及其配套自定义节点。工作流从一张原图开始，使用 ComfyUI 核心 SAM3.1 自动识别目标和保护对象，在原生分辨率下按遮罩区域动态分块，再使用 FLUX.2 Klein 9B 局部生成，并把结果严格合回原图。

仓库内的正式工作流位于 [`workflows/通用智能消除.json`](workflows/%E9%80%9A%E7%94%A8%E6%99%BA%E8%83%BD%E6%B6%88%E9%99%A4.json)。

## 安装方式

### 方式一：ComfyUI Manager 安装缺失节点

这是发布到 Comfy Registry 后的推荐方式。

1. 下载并打开本仓库的 `workflows/通用智能消除.json`。
2. ComfyUI 提示存在缺失节点时，选择“安装全部”或打开 Manager 的“安装缺失节点”。
3. 安装完成后，从原来的 ComfyUI 启动器重启 ComfyUI。
4. 重新打开工作流。

Manager 只能安装节点包和 Python 依赖，不能代替用户接受模型许可或自动安装本页列出的模型文件。

### 方式二：通过 Git URL 安装

适用于仍提供“通过 Git URL 安装”的 ComfyUI Manager 旧版界面。

1. 打开 Manager，点击“通过 Git URL 安装”。
2. 输入：

   ```text
   https://github.com/Liu-Bot24/ComfyUI-Universal-Smart-Removal
   ```

3. 安装完成后，从原来的 ComfyUI 启动器重启 ComfyUI。

### 方式三：手工安装

在 `ComfyUI/custom_nodes/` 下执行：

```powershell
git clone https://github.com/Liu-Bot24/ComfyUI-Universal-Smart-Removal.git
cd ComfyUI-Universal-Smart-Removal
python -m pip install -r requirements.txt
```

必须使用 ComfyUI 自己的 Python 环境执行 `pip`。Windows Portable 应使用其 `python_embeded\python.exe`；整合包或启动器用户应使用该启动器实际调用的 Python，不要安装到系统 Python。

安装后从原来的启动器重启 ComfyUI。

## 工作流还需要的节点包

除本仓库外，工作流使用以下已公开节点包：

- [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)：遮罩叠加预览。
- [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)：`PreviewBridge` 遮罩编辑器和布尔控制。
- [ComfyUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use)：遮罩非空判断。

`SAM3_Detect`、`ReferenceLatent`、`Flux2Scheduler`、`ImageCompositeMasked` 等属于当前 ComfyUI 核心节点。若这些节点缺失，应先更新到支持 SAM3.1 和 FLUX.2 Klein 的新版 ComfyUI，而不是安装同名第三方节点。

## 必需模型

文件名必须与工作流一致：

| 文件 | 放置目录 | 来源 |
|---|---|---|
| `sam3.1_multiplex_fp16.safetensors` | `ComfyUI/models/checkpoints/` | [Comfy-Org SAM3.1](https://huggingface.co/Comfy-Org/sam3.1/blob/main/checkpoints/sam3.1_multiplex_fp16.safetensors) |
| `flux-2-klein-9b-fp8.safetensors` | `ComfyUI/models/diffusion_models/` | [Black Forest Labs FLUX.2 Klein 9B FP8](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8) |
| `qwen_3_8b_fp8mixed.safetensors` | `ComfyUI/models/text_encoders/` | [Comfy-Org 9B text encoder](https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/blob/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors) |
| `full_encoder_small_decoder.safetensors` | `ComfyUI/models/vae/` | [Black Forest Labs FLUX.2 small decoder](https://huggingface.co/black-forest-labs/FLUX.2-small-decoder/blob/main/full_encoder_small_decoder.safetensors) |

FLUX.2 Klein 9B 的上游仓库可能要求登录并接受模型许可。本仓库不包含或重新分发任何模型权重。

## SAM3 中文输入

SAM3 的文本编码器使用英文。本节点包提供 `SAM Prompt Auto English (Offline)`：

- 英文输入原样通过；
- 中文输入使用本机 Argos Translate 中译英；
- 工作流运行时不调用在线翻译服务，也不会自动下载语言模型。

Manager 会根据 `requirements.txt` 安装 Argos Translate 程序库，但中译英语言模型需要一次性安装。在本插件目录内，用 ComfyUI 的 Python 执行：

```powershell
python scripts/install_argos_zh_en.py
```

脚本只在用户明确执行时联网下载 Argos 官方中译英语言包。安装完成后重启 ComfyUI。若不安装语言包，仍可直接给 SAM3 输入英文对象名称。

## 使用顺序

1. 上传一张原图。
2. 在 SAM3 区域填写处理对象和保护对象。
3. 保持“最终生成”关闭，先运行一次并检查自动遮罩和分块范围。
4. 只有自动遮罩不准确时，才打开目标或保护遮罩编辑器进行修正并保存。
5. 在集中控制区填写删除/替换要求。默认使用“标准”分块、外扩 32、羽化 8。
6. 确认白色生成范围正确后开启“最终生成”，再次运行。
7. 最终输出保持原图分辨率；遮罩范围外直接保留原图。

## 本节点包提供的节点

- `Mask Region Tile Planner (Exact)`
- `Get Region Tile (Exact)`
- `Prepare Controlled Dynamic Region Tiles`
- `Universal Local Edit Controls`
- `Merge Dynamic Region Tiles (Normalized)`
- `SAM Prompt Auto English (Offline)`
- 兼容旧工作流的网格、批量分块、手工遮罩和旧合并节点

当前归属合并逻辑要求每个核心像素只有一个所属块；相邻块在明确的内部接缝带内交叉渐变，生成范围之外不混入候选图。节点不会缩放整张原图。

## 常见问题

### `unexpected keyword argument 'ownership_masks'`

工作流比已加载的节点代码新。更新本节点包后，必须完整重启 ComfyUI；只刷新网页不会重新加载 Python 节点。

### 工作流能打开，但模型下拉框为空

这是模型文件缺失或目录不正确，与节点安装无关。按“必需模型”表检查四个文件。

### “安装缺失节点”找不到本节点包

新 Manager 只从 Comfy Registry 安装节点包。Registry 收录前请使用“通过 Git URL 安装”或手工 `git clone`。

## 验证

开发者可在仓库根目录运行：

```powershell
python -m unittest discover -s tests -v
```
