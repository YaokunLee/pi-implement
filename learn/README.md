# openpi

openpi 由 [Physical Intelligence 团队](https://www.physicalintelligence.company/) 发布，收录了用于机器人领域的开源模型与软件包。

目前仓库包含三类模型：

- [π₀ 模型](https://www.physicalintelligence.company/blog/pi0)：基于流匹配的视觉-语言-动作模型（VLA）。
- [π₀-FAST 模型](https://www.physicalintelligence.company/research/fast)：一种基于 FAST 动作分词器的自回归 VLA。
- [π₀.₅ 模型](https://www.physicalintelligence.company/blog/pi05)：π₀ 的升级版，通过 [knowledge insulation](https://www.physicalintelligence.company/research/knowledge_insulation) 训练，具备更好的开放世界泛化能力。需注意：在本仓库中，我们目前仅支持 π₀.₅ 的 flow matching 头用于训练和推理。

对所有模型，我们都提供在 1 万+ 小时机器人数据上预训练的「基础模型」检查点，以及开箱即用或在你自有数据集上微调的示例。

这是一次实验：π₀ 针对我们自有的机器人平台开发，这些平台与常用的 [ALOHA](https://tonyzhaozh.github.io/aloha/) 和 [DROID](https://droid-dataset.github.io/) 等不同。我们乐观地认为研究者和从业者能在自己的平台上做出有创意的实验，但并不保证每次尝试都成功。换言之：π₀ 可能或可能不会适用于你的场景，欢迎尝试！

## 更新日志

- 2025 年 9 月：发布 openpi 的 PyTorch 支持。
- 2025 年 9 月：发布提升开放世界泛化能力的 pi05 模型。
- 2025 年 9 月：为 DROID 训练添加了[改进的空闲过滤器](examples/droid/README_train.md#data-filtering)。
- 2025 年 6 月：添加了使用 `openpi` 在完整 [DROID 数据集](https://droid-dataset.github.io/) 上训练 VLA 的[指南](examples/droid/README_train.md)。这是训练 pi0-FAST-DROID 的开源近似实现。

## 环境需求

要运行本仓库的模型，你需要一块 NVIDIA GPU，至少满足下表规格。下面的估算假设使用单 GPU；通过在训练配置中设置 `fsdp_devices` 进行模型并行，也可以使用多卡以降低单卡显存需求。当前训练脚本暂不支持多节点训练。


| 模式         | 所需显存  | 示例 GPU           |
| ------------ | --------- | ------------------ |
| 推理         | > 8 GB    | RTX 4090           |
| 微调（LoRA） | > 22.5 GB | RTX 4090           |
| 微调（全参） | > 70 GB   | A100 (80GB) / H100 |

本仓库已在 Ubuntu 22.04 上通过测试，目前不支持其他操作系统。

## 安装

克隆仓库时务必更新子模块：

```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git

# 如果已克隆仓库：
git submodule update --init --recursive
```

我们使用 [uv](https://docs.astral.sh/uv/) 管理 Python 依赖。参考 [uv 安装指南](https://docs.astral.sh/uv/getting-started/installation/)进行安装。安装 uv 后，运行以下命令以完成环境配置：

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

注意：`GIT_LFS_SKIP_SMUDGE=1` 用于拉取 LeRobot 依赖时跳过 LFS 下载。

**Docker**：如果本地环境遇到问题，可以使用 Docker 方式安装 openpi。详见 [Docker 设置](docs/docker.md)。

## 模型检查点

### 基础模型

我们提供多种基础 VLA 模型检查点，这些模型已在 1 万+ 小时机器人数据上预训练，可直接用于微调。


| 模型         | 用途 | 说明                                                                               | 检查点路径                                     |
| ------------ | ---- | ---------------------------------------------------------------------------------- | ---------------------------------------------- |
| $\pi_0$      | 微调 | 基础[π₀ 模型](https://www.physicalintelligence.company/blog/pi0)                 | `gs://openpi-assets/checkpoints/pi0_base`      |
| $\pi_0$-FAST | 微调 | 基础自回归[π₀-FAST 模型](https://www.physicalintelligence.company/research/fast) | `gs://openpi-assets/checkpoints/pi0_fast_base` |
| $\pi_{0.5}$  | 微调 | 基础[π₀.₅ 模型](https://www.physicalintelligence.company/blog/pi05)             | `gs://openpi-assets/checkpoints/pi05_base`     |

### 微调模型

我们还提供针对多种机器人平台与任务的“专家”检查点。这些模型基于上方基础模型微调，可直接在目标机器人上运行。由于微调数据集较小（多来自 ALOHA 与 DROID Franka），它们不一定泛化到你的机器人，不过我们发现部分模型，尤其 DROID 检查点，在实践中泛化性不错。


| 模型                     | 用途        | 说明                                                                                                                                                                                               | 检查点路径                                            |
| ------------------------ | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| $\pi_0$-FAST-DROID       | 推理        | $\pi_0$-FAST 模型在 [DROID 数据集](https://droid-dataset.github.io/) 上微调：可在 DROID 机器人平台上 0-shot 执行多种简单桌面操作                                                                   | `gs://openpi-assets/checkpoints/pi0_fast_droid`       |
| $\pi_0$-DROID            | 微调        | $\pi_0$ 模型在 [DROID 数据集](https://droid-dataset.github.io/) 上微调：推理更快，但语言跟随可能不如 $\pi_0$-FAST-DROID                                                                            | `gs://openpi-assets/checkpoints/pi0_droid`            |
| $\pi_0$-ALOHA-towel      | 推理        | $\pi_0$ 模型在内部 [ALOHA](https://tonyzhaozh.github.io/aloha/) 数据上微调：可在 ALOHA 机器人平台 0-shot 折叠各类毛巾                                                                              | `gs://openpi-assets/checkpoints/pi0_aloha_towel`      |
| $\pi_0$-ALOHA-tupperware | 推理        | $\pi_0$ 模型在内部 [ALOHA](https://tonyzhaozh.github.io/aloha/) 数据上微调：可完成从保鲜盒取物的任务                                                                                               | `gs://openpi-assets/checkpoints/pi0_aloha_tupperware` |
| $\pi_0$-ALOHA-pen-uncap  | 推理        | $\pi_0$ 模型在公开 [ALOHA](https://dit-policy.github.io/) 数据上微调：可以打开笔帽                                                                                                                 | `gs://openpi-assets/checkpoints/pi0_aloha_pen_uncap`  |
| $\pi_{0.5}$-LIBERO       | 推理        | $\pi_{0.5}$ 模型在 [LIBERO](https://libero-project.github.io/datasets) 基准上微调：达到 SOTA（见 [LIBERO README](examples/libero/README.md)）                                                      | `gs://openpi-assets/checkpoints/pi05_libero`          |
| $\pi_{0.5}$-DROID        | 推理 / 微调 | $\pi_{0.5}$ 模型在 [DROID 数据集](https://droid-dataset.github.io/) 上通过 [knowledge insulation](https://www.physicalintelligence.company/research/knowledge_insulation) 微调：推理快且语言跟随好 | `gs://openpi-assets/checkpoints/pi05_droid`           |

默认情况下，检查点会从 `gs://openpi-assets` 自动下载并缓存在 `~/.cache/openpi`。可通过环境变量 `OPENPI_DATA_HOME` 覆盖下载路径。

## 运行预训练模型推理

以下示例展示了加载并运行我们的 $\pi_0$-FAST-DROID 检查点：

```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")

# 创建训练好的策略
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# 在示例输入上运行推理
dummy_example = {
    "observation/exterior_image_1_left": ...,
    "observation/wrist_image_left": ...,
    ...
    "prompt": "pick up the fork"
}
action_chunk = policy.infer(dummy_example)["actions"]
```

也可以在 [示例 Notebook](examples/inference.ipynb) 中体验。

我们为 [DROID](examples/droid/README.md) 和 [ALOHA](examples/aloha_real/README.md) 机器人提供了逐步推理示例。

**远程推理**：提供了[示例与代码](docs/remote_inference.md)，支持在远程服务器运行模型，通过 websocket 向机器人流式发送动作，方便将计算从机器人侧分离。

**无机器人测试推理**：我们提供了一个[脚本](examples/simple_client/README.md) 用于在无机器人环境下测试推理。脚本会生成随机观测并运行模型推理，详见链接。

## 在自有数据上微调基础模型

我们以在 [LIBERO 数据集](https://libero-project.github.io/datasets) 上微调 $\pi_{0.5}$ 为示例，介绍三个步骤：

1. 将你的数据转换为 LeRobot 数据集格式
2. 定义训练配置并启动训练
3. 启动策略服务器并运行推理

### 1. 将数据转换为 LeRobot 数据集

我们提供了一个最小脚本 [`examples/libero/convert_libero_data_to_lerobot.py`](examples/libero/convert_libero_data_to_lerobot.py) 用于把 LIBERO 数据转换为 LeRobot 数据集。你可以很容易地改写它来处理自己的数据！可从[这里](https://huggingface.co/datasets/openvla/modified_libero_rlds)下载原始 LIBERO 数据，然后运行：

```bash
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/libero/data
```

**提示：** 如果你只想在 LIBERO 上微调，可以跳过这一步，因为我们的 LIBERO 微调配置已经指向预转换的数据集。此步骤主要展示如何适配你的数据。

### 2. 定义训练配置并运行训练

要在自有数据上微调基础模型，需要定义数据处理与训练配置。下面给出带注释的 LIBERO 示例配置，可按需修改：

- [`LiberoInputs` 和 `LiberoOutputs`](src/openpi/policies/libero_policy.py)：定义 LIBERO 环境到模型（及逆向）的数据映射，训练与推理都会用到。
- [`LeRobotLiberoDataConfig`](src/openpi/training/config.py)：定义如何从 LeRobot 数据集中处理原始 LIBERO 数据供训练使用。
- [`TrainConfig`](src/openpi/training/config.py)：定义微调超参、数据配置以及权重加载方式。

我们提供了针对 [π₀](src/openpi/training/config.py)、[π₀-FAST](src/openpi/training/config.py) 和 [π₀.₅](src/openpi/training/config.py) 的 LIBERO 微调示例配置。

开始训练前，需要计算训练数据的归一化统计量。运行：

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero
```

然后用以下命令启动训练（`--overwrite` 用于在重复实验时覆盖现有检查点）：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite
```

命令会在控制台记录训练进度，并把检查点保存到 `checkpoints` 目录。你也可以在 Weights & Biases 面板查看训练。为最大化显存使用，在运行前设置 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`，让 JAX 使用最多 90% GPU 显存（默认 75%）。

**提示：** 我们支持在训练时*复用*预训练阶段的状态/动作归一化统计，对预训练数据中出现过的机器人微调新任务时有益。详见 [norm_stats.md](docs/norm_stats.md)。

### 3. 启动策略服务器并运行推理

训练完成后，可以启动策略服务器并由 LIBERO 评估脚本进行推理。示例（假设使用第 20,000 次迭代的检查点，可按需修改）：

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=checkpoints/pi05_libero/my_experiment/20000
```

该命令会在 8000 端口启动服务，等待接收观测。随后可运行评估脚本（或机器人运行时）访问该服务。

针对 LIBERO 评测，我们提供了推荐的 Docker 工作流同时运行策略服务器和评估脚本。详见 [LIBERO README](examples/libero/README.md)。

如果要在自有机器人运行时中嵌入策略服务器调用，可参考 [远程推理文档](docs/remote_inference.md) 中的最小示例。

### 更多示例

我们还提供了在 ALOHA 平台上微调与推理的示例：

- [ALOHA 模拟器](examples/aloha_sim)
- [ALOHA 实机](examples/aloha_real)
- [UR5](examples/ur5)

## PyTorch 支持

openpi 现已在原有 JAX 版本之外提供 π₀ 与 π₀.₅ 的 PyTorch 实现。该实现已在 LIBERO 基准上通过验证（推理与微调）。当前尚不支持的特性（未来可能更新）：

- π₀-FAST 模型
- 混合精度训练
- FSDP（fully-sharded data parallelism）训练
- LoRA（低秩适配）训练
- 训练时的 EMA（指数滑动平均）权重

### 设置

1. 确保依赖为最新版本：`uv sync`
2. 确认 transformers 版本为 4.53.2：`uv pip show transformers`
3. 应用 transformers 补丁：
   ```bash
   cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
   ```

上述命令会覆盖 transformers 库的部分文件，以添加：1) 支持 AdaRMS；2) 正确控制激活精度；3) 允许在不更新的情况下使用 KV cache。

**警告**：uv 默认使用硬链接模式，该操作会永久影响 uv 缓存中的 transformers，即使重装也会保留，并可能影响其他使用 transformers 的项目。完全撤销需运行 `uv cache clean transformers`。

### 将 JAX 模型转换为 PyTorch

```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /path/to/jax/checkpoint \
    --config_name <config name> \
    --output_path /path/to/converted/pytorch/checkpoint
```

### 使用 PyTorch 运行推理

PyTorch 版本 API 与 JAX 相同，仅需将检查点路径指向转换后的 PyTorch 模型：

```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = "/path/to/converted/pytorch/checkpoint"

# 创建训练好的策略（会自动识别 PyTorch 格式）
policy = policy_config.create_trained_policy(config, checkpoint_dir)

action_chunk = policy.infer(example)["actions"]
```

### 使用 PyTorch 的策略服务器

策略服务器用法相同，只需指向转换后的检查点目录：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid \
    --policy.dir=/path/to/converted/pytorch/checkpoint
```

### 在 PyTorch 中进行微调

1. 将 JAX 基础模型转换为 PyTorch：
   ```bash
   uv run examples/convert_jax_model_to_pytorch.py \
       --config_name <config name> \
       --checkpoint_dir /path/to/jax/base/model \
       --output_path /path/to/pytorch/base/model
   ```
2. 在配置中通过 `pytorch_weight_path` 指定转换后的 PyTorch 模型路径。
3. 选择以下方式之一启动训练：

```bash
# 单卡训练
uv run scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>

# 示例
uv run scripts/train_pytorch.py debug --exp_name pytorch_test
uv run scripts/train_pytorch.py debug --exp_name pytorch_test --resume  # 从最新检查点恢复

# 多 GPU（单节点）
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>

# 示例
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume

# 多节点训练
uv run torchrun \
    --nnodes=<num_nodes> \
    --nproc_per_node=<gpus_per_node> \
    --node_rank=<rank_of_node> \
    --master_addr=<master_ip> \
    --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>
```

### 精度设置

JAX 与 PyTorch 在精度上的处理如下：

**JAX：**

1. 推理：多数权重与计算使用 bfloat16，少量计算使用 float32 保持稳定。
2. 训练：默认混合精度——权重与梯度 float32，(多数) 激活与计算 bfloat16。可在配置中将 `dtype` 设为 float32 进行全精度训练。

**PyTorch：**

1. 推理：与 JAX 一致——多数权重与计算 bfloat16，少量权重转换为 float32 以确保稳定。
2. 训练：支持全 bfloat16（默认）或全 float32，可通过 `pytorch_training_precision` 配置。bfloat16 更省显存但 loss 略高；尚不支持混合精度。

使用 `torch.compile` 时，PyTorch 推理速度与 JAX 相当。

## 故障排查

我们会在此汇总常见问题及解决方案。如未找到答案，请在仓库提 Issue（参考 [CONTRIBUTING.md](CONTRIBUTING.md)）。


| 问题                    | 解决方案                                                                                                                                                                                                                                                     |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `uv sync` 依赖冲突      | 尝试删除虚拟环境目录（`rm -rf .venv`）后重新运行 `uv sync`。如仍有问题，检查 `uv` 是否为最新版（`uv self update`）。                                                                                                                                         |
| 训练显存不足            | 训练前设置`XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`（或更高）以让 JAX 使用更多显存。也可使用 `--fsdp-devices <n>`（n 为 GPU 数）启用 [FSDP](https://engineering.fb.com/2021/07/15/open-source/fsdp/) 以换取更低显存占用（速度会变慢）。若仍不足，可考虑关闭 EMA。 |
| 策略服务器连接错误      | 确认服务器已运行并监听预期端口。检查客户端与服务器之间的网络连接与防火墙设置。                                                                                                                                                                               |
| 训练时报缺失 norm stats | 在训练前使用你的配置运行`scripts/compute_norm_stats.py`。                                                                                                                                                                                                    |
| 数据集下载失败          | 检查网络连接。若使用 HuggingFace 数据集，确保已登录（`huggingface-cli login`）。                                                                                                                                                                             |
| CUDA/GPU 报错           | 确认已正确安装 NVIDIA 驱动。使用 Docker 时需安装 nvidia-container-toolkit。检查 GPU 兼容性。无需系统级安装 CUDA 库——它们会通过 uv 安装。如遇冲突，可尝试卸载系统 CUDA 库。                                                                                 |
| 运行示例时导入错误      | 确保已执行`uv sync` 安装依赖。部分示例可能还有额外需求，详见对应 README。                                                                                                                                                                                    |
| 动作维度不匹配          | 确认数据处理变换与机器人期望的输入/输出维度一致。检查策略类中的动作空间定义。                                                                                                                                                                                |
| 训练 loss 发散          | 查看数据集中的`norm_stats.json` 中 `q01`、`q99` 和 `std`。若某些维度极小，归一化后可能导致状态/动作过大。可手动调整这些统计量作为解决办法。                                                                                                                  |

(base) ➜  openpi git:(main) ✗ GIT_LFS_SKIP_SMUDGE=1 uv sync
Using CPython 3.11.13
Creating virtual environment at: .venv
Resolved 281 packages in 7ms
error: Distribution `jax-cuda12-plugin==0.5.3 @ registry+https://pypi.org/simple` can't be installed because it doesn't have a source distribution or wheel for the current platform

hint: You're on macOS (`macosx_15_0_arm64`), but `jax-cuda12-plugin` (v0.5.3) only has wheels for the following platforms: `manylinux2014_aarch64`, `manylinux2014_x86_64`; consider adding your platform to `tool.uv.required-environments` to ensure uv resolves to a version with compatible wheels
