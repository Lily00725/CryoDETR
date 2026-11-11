# CryoDETR 颗粒分类 Grad-CAM 可视化

`visualize_heatmap.py` 将编码器的某一层、某一尺度特征与最终颗粒分类分数结合，
生成叠加在微图上的 Grad-CAM。红色表示当前所选颗粒分类目标的较高正向 Grad-CAM 响应，
可作为分类依据的近似解释。
输入使用训练或验证时同一预处理流程得到的 PNG、JPG 或 TIFF。

## 在训练环境中运行

将 `visualize_heatmap.py` 放在仓库根目录，同时使用本分支的
`models/CryoDETR/backbone.py`。后者增加 `pretrained_backbone` 开关，绘图加载
完整检查点时关闭 ImageNet 权重下载。该开关的默认值仍为 `True`。

使用已经能够训练 CryoDETR 的环境及其编译好的 `MultiScaleDeformableAttention`。
脚本直接使用 `models/CryoDETR` 中的模型类，并从 checkpoint 的 `args` 恢复训练配置。
只有权重、没有 `args` 的文件需要通过 `--config` 补充原训练参数，支持 JSON 或带
`_base_` 的 Python 配置。`--baseline-config` 为基线单独指定配置。
检查点内已保存的参数优先于配置文件。检查点必须来自可信的本地训练结果。

绘制一张模型热图。

```bash
python visualize_heatmap.py \
  --images /root/autodl-tmp/10532/val/example.png \
  --checkpoint /path/to/HAE_experiment/checkpoint0029.pth \
  --output-dir /root/autodl-tmp/heatmaps/10532
```

绘制三栏对比图。

```bash
python visualize_heatmap.py \
  --images /root/autodl-tmp/10532/val/example.png \
  --baseline-checkpoint /path/to/no_HAE_experiment/checkpoint0029.pth \
  --checkpoint /path/to/HAE_experiment/checkpoint0029.pth \
  --encoder-layer -1 \
  --feature-level 0 \
  --class-id 1 \
  --score-threshold 0.3 \
  --top-k 100 \
  --output-dir /root/autodl-tmp/heatmaps/10532
```

路径中的 `example.png` 和检查点路径需要换成实际文件。`--images` 后面可以放多张图片。
原图和热图沿用 PIL 图像坐标，左上角为原点。读取时保留图像方向。
MRC 微图先用训练时的预处理流程生成图片，再将生成的图片传给脚本。

## 对照设置

`--baseline-checkpoint` 需要使用单独训练的、HAE 层数为 0 的模型。
`--checkpoint` 使用 HAE 层数大于 0 的模型。脚本从实际权重的
`std_self_attn` 键识别 HAE 层数，并使用 `strict=True` 加载所有参数。
读取当前模型的真实尺度尺寸后再还原 token，所以四尺度和五尺度模型都可以使用。

如果要隔离 HAE 的效果，两组模型应保持 O2M、解码器、输入尺度、数据划分和训练条件一致。
脚本会报告恢复出的结构参数中超出 HAE 的差异，参数详见输出 JSON。
完整 CryoDETR 对比原始 Deformable DETR 时，图反映两个完整模型的差异。

| 参数 | 含义 |
| --- | --- |
| `--encoder-layer -1` | 最后一层编码器，层编号从 0 开始 |
| `--encoder-layer 1` | 第二层编码器，可观察前两层 HAE 后的表现 |
| `--feature-level 0` | 最高空间分辨率的编码器尺度，适合先查看小颗粒 |
| `--class-id 1` | 仓库 COCO 数据生成脚本使用的颗粒类别编号 |
| `--short-side 800 --max-size 1333` | 默认与 RGB 验证变换一致，复现实验时按实际设置调整 |
| `--cmap jet --alpha 0.45` | 配色和叠加透明度 |

默认分别对两个模型中超过置信度阈值的最多 100 个颗粒 query 的原始 logit 求和，
据此计算梯度。两个模型选中的 query 可以不同，输出 JSON 记录具体数量、编号、
置信度和归一化框。没有 query 达到阈值时脚本会报错，需检查类别及模型输出，
再明确调整阈值。

要围绕同一个颗粒分析，可指定原图中一个矩形区域，顺序为 `X1 Y1 X2 Y2`。

```bash
python visualize_heatmap.py \
  --images /root/autodl-tmp/10532/val/example.png \
  --baseline-checkpoint /path/to/no_HAE_experiment/checkpoint0029.pth \
  --checkpoint /path/to/HAE_experiment/checkpoint0029.pth \
  --target-box 1100 900 1250 1050 \
  --crop 900 700 1500 1300 \
  --output-dir /root/autodl-tmp/heatmaps/10532_single_particle
```

`--target-box` 为每个模型选择与该区域 IoU 最大的一个预测 query，随后对其颗粒分类
logit 求梯度。此时不使用 `--top-k` 和置信度阈值筛选。
坐标需换成实际颗粒位置，并检查 JSON 中选中 query 的框与分数。
`--crop` 仅裁剪最终展示范围，模型仍读取完整微图。

## 输出与解释

每张输入图片生成以下文件。

| 文件 | 内容 |
| --- | --- |
| `example_comparison.png` | 原图及一栏或两栏模型热图，带共享色条 |
| `example_cam.npz` | 插值到原图大小的非负 CAM 和归一化 CAM |
| `example_metadata.json` | 检查点路径、结构、query、各尺度形状和绘图参数 |

CAM 按通道空间平均梯度加权，再通过 ReLU 保留正贡献。
两个模型采用同一组正 CAM 值的第 99 百分位作为色条上界，图中值缩放至 0–1。
该处理用于对照可视化，跨模型的 CAM 绝对幅值仍受各自特征和分类分数尺度影响。
如果当前目标和层的正 CAM 全为零，脚本保留零图并给出提示。

这套图适合展示颗粒分类相关的空间响应变化。若论文要论证感受野范围，应另做输入梯度
感受野实验。若要论证噪声抑制，应结合真实颗粒坐标、污染区域和检测错误统计。
插值只改善展示连续性，空间定位精度仍取决于实际特征尺度和输入缩放。

## 常见运行问题

* `No module named MultiScaleDeformableAttention` 表示当前环境缺少仓库的编译算子。
  在原训练环境运行，或按仓库 `models/CryoDETR/ops/make.sh` 编译。
* `Missing key`、`Unexpected key` 或尺寸不匹配需要核对检查点是否对应此版本模型，
  以及补充的配置是否来自当时训练。
* 显存不足时可先单独绘制一个模型。若调整 `--short-side`，比较两组模型时使用相同值，
  并记录它与原验证设置的差异。
