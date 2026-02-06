# CODA-Prompt 的 Jittor 复现说明

本仓库基于原始 [CODA-Prompt](https://arxiv.org/abs/2211.13218) 代码，对其完整逻辑进行 [Jittor](https://cg.cs.tsinghua.edu.cn/jittor/) 框架的重写，实现与 PyTorch 版本一致的持续学习流程，同时满足国产框架本地运行的需求。

## 核心特性

- 使用纯 Jittor 运算实现的 ViT+Prompt 持续学习框架。
- 与原 `run.py` 参数保持一致的命令行接口，支持配置文件与 CLI 覆盖。
- 在 `coda_jittor` 中复刻 CODA-Prompt、DualPrompt、L2P 等提示学习器。
- 支持分任务数据加载、记忆集挂钩、准确率/时间统计与指标持久化。
- 内置自适应 batch size 回退机制，帮助 6 GB GPU 稳定训练大规模配置。

## 目录结构

```

  run.py              # 命令行入口（python -m coda_jittor.run）
  trainer.py          # 持续学习训练调度
  learners/           # 基于 Jittor 的提示学习器实现
  models/             # ViT 主干与 Prompt 组合模型
  utils/              # 指标、学习率调度、日志工具
configs/
  *.yaml              # 复用自原仓库的实验配置
outputs/              # 默认日志与权重输出目录
```

## 环境准备

1. 建议使用名为 `dl` 的 conda 环境：
   ```bash
   conda create -n dl python=3.8 -y
   conda activate dl
   pip install -r requirements.txt
   ```

2. （可选）验证 Jittor CUDA：
   ```python
   import jittor as jt
   jt.flags.use_cuda = 1
   jt.utils.test_cuda()
   ```

> **提示：** 训练过程中会调用 CuPy (`cupy_cuda11x`) 以加速张量导出，已在 `requirements.txt` 中列出。

## 数据集准备

- **CIFAR-100**：放置在 `data/cifar-100-python` 目录。
- **ImageNet-R / DomainNet**：在 `data/` 下复用原 PyTorch 仓库的数据结构。Jittor 数据加载器继续使用 torchvision 预处理，因此数据布局无需调整。

## 运行实验

请始终在仓库根目录下使用 `conda run -n dl python -u -m coda_jittor.run` 方式启动。

### CIFAR-100 两任务迷你示例

```bash
conda run -n dl python -u -m coda_jittor.run \
  --config configs/cifar-100_2task_mini.yaml \
  --gpuid 0 --repeat 1 --overwrite 1 \
  --learner_type prompt --learner_name CODAPrompt \
  --prompt_param 100 8 0.0 \
  --log_dir outputs/cifar-100/coda-prompt-2task-mini
```

### ImageNet-R 两任务示例

默认情况下，为避免在 6 GB 显存上 OOM，Jittor 训练器会自动把高分辨率输入的 ViT 主干缩减到 `embed_dim=384 / depth=6 / num_heads=6`。若希望完全复现原论文的 ViT-B/16，请显式指定模型参数，例如：

```yaml
# configs/imnet-r_2task_fullvit.yaml
inherit: configs/imnet-r_2task.yaml
model_kwargs:
  img_size: 224
  patch_size: 16
  embed_dim: 768
  depth: 12
  num_heads: 12
```

随后执行：

```bash
conda run -n dl python -u -m coda_jittor.run \
  --config configs/imnet-r_2task_fullvit.yaml \
  --gpuid 0 --repeat 1 --overwrite 1 \
  --learner_type prompt --learner_name CODAPrompt \
  --prompt_param 100 8 0.0 \
  --log_dir outputs/imnet/imnet-r_2task_fullvit
```

### 额外说明

- `--prompt_param` 依次传入：提示池大小、提示长度、正交约束权重。
- `--overwrite 1` 会覆盖已存在的日志与模型，确保结果重新计算。
- 若出现显存不足，训练器会自动将 batch size 减半并同步给优化器。

## 与原实现的差异

| 对比项 | Jittor 版本情况 |
| --- | --- |
| 提示学习逻辑 | 完整复现（CODA、Dual、L2P） |
| 持续学习流程 | 与原版一致，含验证评估和 CSV 输出 |
| 优化器与调度器 | 支持 SGD、Adam、AdamW、余弦退火等同配置 |
| 默认 ViT 主干 | 高分辨率时默认降配，可通过 `model_kwargs` 覆盖 |
| 混合精度 | 暂未启用，默认 FP32 |
| 数据增广 | 继续复用 torchvision 预处理封装 |

若需论文级别结果，请按照上文配置 ViT-B/16，并保证硬件资源充足。

## 常见问题

- **显存不足**：降低 `batch_size`、缩小提示池或保留降配主干。
- **CuPy 缺失**：确保 `cupy_cuda11x` 版本匹配 CUDA 驱动，可用 `pip install --force-reinstall cupy-cuda11x` 重新安装。
- **检测不到 GPU**：将 `jt.flags.use_cuda = 0` 切换到 CPU（训练速度会显著下降，但可验证流程）。
- **准确率偏低**：检查主干参数是否与原版一致，并确认任务顺序固定（可通过 `--repeat` 结合配置种子控制）。

## 引用

若该 Jittor 复现对您的研究有帮助，请同时引用 CODA-Prompt 论文和 Jittor 项目：

```
@InProceedings{Smith_2023_CVPR,
        author    = {Smith, James Seale and Karlinsky, Leonid and Gutta, Vyshnavi and Cascante-Bonilla, Paola and Kim, Donghyun and Arbelle, Assaf and Panda, Rameswar and Feris, Rogerio and Kira, Zsolt},
        title     = {CODA-Prompt: COntinual Decomposed Attention-Based Prompting for Rehearsal-Free Continual Learning},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        month     = {June},
        year      = {2023},
        pages     = {11909-11919}
    }



@article{hu2020jittor,
  title={Jittor: a novel deep learning framework with meta-operators and unified graph execution},
  author={Hu, Shi-Min and Liang, Dun and Yang, Guo-Ye and Yang, Guo-Wei and Zhou, Wen-Yang},
  journal={Science China Information Sciences},
  volume={63},
  number={222103},
  pages={1--21},
  year={2020}
}
```
