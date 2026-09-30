# Gemma3 动态图/CINN 复现说明

本文件只描述 Gemma3 的模型专属要求；通用目录约定、入口参数、结果字段和 `run_model_common.sh` 说明见仓库根目录的 [`README.md`](../README.md)。

## 1. 入口

| 文件 | 作用 |
| --- | --- |
| `run_model.sh` | 调用同一个 `run_model.py`，运行纯动态图 |
| `run_model_cinn.sh` | 调用同一个 `run_model.py`，只对已验证的 MLP 子图执行 `to_static(full_graph=True)`+CINN |
| `run_model_common.sh` | 两个入口共享的校验、路径解析、环境设置和日志编排，不直接实现模型计算 |
| `run_model.py` | checkpoint/data 校验、训练或推理、计时和 JSON 输出 |
| `patches/paddleformers-gemma3-cinn.patch` | 固定 PaddleFormers commit 的 Gemma3 数据/checkpoint 适配补丁；不改写动态图模型以迁就 CINN |
| `paddleformers.commit` | 要求的上游 commit |
| `checkpoint/README.md` | checkpoint 结构、精度、大小和 SHA256 |

脚本会先检查 PaddleFormers commit，再幂等应用 patch：已应用则直接运行；未应用且可干净应用则自动应用；源码状态不明确时停止，避免覆盖用户修改。

## 1.1 静态边界与动态图隔离

本版本不再把整个 Gemma3 多模态 forward、训练 loss 或推理 wrapper 包成一个
`to_static(full_graph=True)`。经验证可编译的边界只有语言模型各 decoder layer 的
`mlp` 子层：

```text
CINN:    model.model.language_model.layers[*].mlp
动态图:  vision tower、图像特征/占位符融合、mask、RoPE、RMSNorm、attention、
         decoder layer loop、final norm、lm_head、loss 以及训练/推理 wrapper
```

这样做的含义是：CINN flags 仍在 `cinn` 模式开启，但 flags 本身不会把动态图代码
自动变成静态图；只有 `run_model.py` 显式包住的 MLP 子图会触发 CINN 编译。无法支持
静态图的部分保持原始动态图执行，也不对它们做源码改写。梯度可以跨越 MLP 子图回传，
因此训练的 forward、backward 和 optimizer step 仍是完整路径。

## 2. 环境

第一阶段验证环境：

```text
PaddleFormers commit: 941b463475ef0bbb3f88e2d1e1bcee4181f98e0b
Paddle: 3.4.0.post20260907+4d8f64b7049
Paddle commit: 4d8f64b704921ab3baebbb7f28bba363f5703af8
Python: 3.10.12
CUDA: 13.0
GPU compute capability: 9.0
checkpoint/compute: BF16
loss: FP32
optimizer: AdamW multi_precision=True（正学习率时）
distributed: 单卡，batch size=1，无 DP/PP/MP
```

已知环境风险：Paddle wheel 报告编译 cuDNN 9.13，而验证节点运行库为 cuDNN 9.9。正式性能验收应统一版本。

建议使用项目已验证的 Python 环境，或安装与上述版本兼容的 Paddle/PaddleFormers 依赖。运行前检查：

```bash
python -c 'import paddle; print(paddle.__version__, paddle.version.commit, paddle.version.cuda(), paddle.version.cudnn())'
nvidia-smi
```

## 3. 拉取 PaddleFormers 与测试仓

```bash
git clone <PADDLEFORMERS_URL> /path/to/PaddleFormers
git clone <TEST_REPOSITORY_URL> /path/to/model-test-suite
git -C /path/to/PaddleFormers checkout 941b463475ef0bbb3f88e2d1e1bcee4181f98e0b
```

两个目录可以是同级目录，也可以通过 `PADDLEFORMERS_DIR` 指定其他位置。测试仓当前远端为 `https://github.com/Sainalu/Baidu_HIT_CiNN.git`，但脚本不依赖固定的仓库地址。

## 4. Checkpoint

使用高精度缩层 Paddle checkpoint：

```text
architecture: Gemma3ForConditionalGeneration
text layers: 4
vision layers: 2
weight dtype: BF16
model_state.pdparams bytes: 3517742411
model_state.pdparams SHA256:
33029a880dc17d16abdac2327e7891bbb1fb7483bfaaf4d962916c5fa8f4f75c
```

checkpoint 还必须包含 `config.json`、tokenizer、processor/preprocessor 和 `checkpoint_manifest.json`。`run_model.py` 每次运行前会验证全部文件的大小与 SHA256。

普通 Git 不适合直接保存 3.4 GiB 权重，因此推荐二选一：

1. 通过制品存储下载 checkpoint，然后设置 `CHECKPOINT_DIR`；或
2. 仓库服务支持 Git LFS 时，将文件放在 `checkpoint/files/`。

```bash
export CHECKPOINT_DIR=/path/to/gemma3-4b-text4-vision2-bf16
```

也可以仅在本地创建已被 `.gitignore` 忽略的 `checkpoint/LOCATION.local`，内容为一行 `path=/absolute/path/...`。不要修改受版本控制的 `LOCATION.md`，也不要提交机器路径。

## 5. 数据

仓内 `data/` 保存真实 Gemma3Processor 产生的固定 shape NPZ：

- 8 个训练 batch。
- 6 个 eval/infer batch。
- batch size=1。
- sequence length=512。
- 每个 batch 一张真实图片。
- `manifest.json` 记录来源数据 hash 和每个 NPZ 的 SHA256。

默认 `DATA_DIR=gemma3/data/train`；推理时设置 `DATA_DIR=gemma3/data/eval`。数据预处理发生在计时外。

## 6. 训练冒烟

动态图：

```bash
cd model-test-suite/gemma3
PADDLEFORMERS_DIR=../../PaddleFormers \
CHECKPOINT_DIR=/path/to/checkpoint \
DATA_DIR=data/train \
STEPS=20 WARMUP_STEPS=2 LEARNING_RATE=0 \
./run_model.sh
```

CINN：

```bash
PADDLEFORMERS_DIR=../../PaddleFormers \
CHECKPOINT_DIR=/path/to/checkpoint \
DATA_DIR=data/train \
STEPS=20 WARMUP_STEPS=2 LEARNING_RATE=0 \
./run_model_cinn.sh
```

`LEARNING_RATE=0` 用于 checkpoint 固定权重的精度/性能冒烟。真实参数更新测试设置正学习率，例如：

```bash
LEARNING_RATE=1e-6 STEPS=1000 WARMUP_STEPS=20 ./run_model.sh
LEARNING_RATE=1e-6 STEPS=1000 WARMUP_STEPS=20 ./run_model_cinn.sh
```

不能把零学习率结果称为全量收敛训练。

## 7. 推理

本入口的推理定义为多模态 teacher-forced forward logits，不是完整 autoregressive generate：

```bash
PADDLEFORMERS_DIR=/path/to/PaddleFormers \
CHECKPOINT_DIR=/path/to/checkpoint \
TASK=infer DATA_DIR=data/eval STEPS=20 WARMUP_STEPS=2 \
./run_model.sh

PADDLEFORMERS_DIR=/path/to/PaddleFormers \
CHECKPOINT_DIR=/path/to/checkpoint \
TASK=infer DATA_DIR=data/eval STEPS=20 WARMUP_STEPS=2 \
./run_model_cinn.sh
```

计时包含模型 forward 和 GPU 同步，不含 NPZ 读取、Tensor 创建及 Host-to-Device。输出包含 latency、FPS、首步时间和 logits checksum。

## 8. CINN 生效与计时口径

CINN 唯一可信的生效标志：

```text
Compiling subgraph with CINN backend
```

`run_model_cinn.sh` 会检查该标志，缺失时返回非零。当前 checkpoint 有 4 个文本层，
JSON 的 `static_subgraph_count` 应为 4；日志中的编译 marker 是后端编译事件数，
不保证与包装的 layer 数相等。首次 forward 包含 MLP 子图的懒 trace/CINN 编译；
稳态指标排除 `WARMUP_STEPS`。报告性能时必须分别说明首编译时间和稳态时间，且不能
把未编译的动态图部分当成 CINN 加速范围。

## 9. 输出

默认输出：

```text
gemma3/logs/train-dynamic.log
gemma3/logs/train-cinn.log
gemma3/logs/infer-dynamic.log
gemma3/logs/infer-cinn.log
gemma3/results/train-dynamic.json
gemma3/results/train-cinn.json
gemma3/results/infer-dynamic.json
gemma3/results/infer-cinn.json
```

可以用 `LOG` 和 `OUTPUT` 覆盖。`logs/custom.log` 按 `gemma3/` 目录解析；为兼容旧命令，`gemma3/logs/custom.log` 按仓库根目录解析。正式对比时，动态/CINN 必须使用相同 checkpoint、数据、step、warmup、精度和学习率。

### 9.1 如何阅读结果

训练 JSON 中最重要的字段是：

- `losses`：每一步 FP32 loss。短步测试只检查数值有限，不能据此判断模型已经收敛。
- `first_step_time_s`：首步耗时。CINN 模式包含首次 trace/编译，必须和稳态指标分开报告。
- `mean_steady_step_time_s`、`steady_fps`：去掉 `warmup_steps` 后的训练速度；当前 batch size 为 1。
- `optimizer`、`learning_rate`：确认是否真正更新了参数。`learning_rate=0` 时是 `no-update/clear-grad`，属于性能/功能冒烟。
- `cinn_scope`、`static_subgraph_count`：说明实际进入 CINN 的子图范围和包装数量；动态图模式应分别为 `none` 和 `0`。
- `dynamic_scope`、`outer_graph_mode`：说明被隔离保留在动态图的模块；当前 `outer_graph_mode` 固定为 `dynamic`。

推理 JSON 中最重要的字段是：

- `scope`：当前测试是多模态 teacher-forced forward logits，不是完整的 autoregressive `generate`。
- `first_step_latency_ms`：包含 CINN 首次编译的首步延迟。
- `mean_steady_latency_ms`、`steady_fps`：去掉 warmup 后的稳态延迟和吞吐。
- `logits_checksum`、`logits_abs_checksum`：有限性和粗粒度回归信号，不是任务精度，也不要求动态图与 CINN 位级相等。

性能比较只使用稳态字段：训练吞吐提升为 `(CINN steady_fps / dynamic steady_fps - 1) * 100%`；推理延迟降低为 `(dynamic mean_steady_latency_ms - CINN mean_steady_latency_ms) / dynamic mean_steady_latency_ms * 100%`。首步编译时间单独列出。

### 9.2 本次局部 CINN 清洁克隆冒烟结果

以下结果来自固定 PaddleFormers commit 的干净 checkout、自动应用精简 patch、使用外部
checkpoint 的 20-step 单次测试，`warmup_steps=2`、`learning_rate=0`。旧的“整个
wrapper 全图静态化”结果仅供历史参考，不能与本表直接比较。

| 任务 | 模式 | 首步 | 稳态指标 | 静态范围 |
| --- | --- | ---: | ---: | ---: |
| 训练 | 动态图 | 1.632 s | 14.82 FPS | none |
| 训练 | CINN | 6.236 s | 14.71 FPS | 4 个 MLP 子图 |
| 推理 | 动态图 | 808.890 ms | 30.549 ms（32.73 FPS） | none |
| 推理 | CINN | 3641.503 ms | 34.143 ms（29.29 FPS） | 4 个 MLP 子图 |

本次单次短跑中，训练 FPS 变化为 -0.75%，推理延迟变化为 +11.76%，因此当前局部边界
只能证明隔离和 CINN 编译正确，不能宣称性能提升。训练 20 步最大绝对 loss 差为
0.09351，数值有限；推理 checksum 也有限。正式报告还需要更长 step、多次重复、稳定的
硬件/软件版本和统一的 cuDNN 环境。当前验证节点存在 cuDNN 编译版本 9.13、运行版本
9.9 不一致警告。

## 10. PaddleFormers 修改说明

补丁只包含与静态边界无关的模型接入适配：

- 注册 Gemma3 多模态数据模板。
- 支持高精度缩层 checkpoint 加载时按目标文本层数过滤多余权重。

补丁不再修改 sliding-window mask、`paddle.finfo`、autocast、RoPE、RMSNorm、attention、
动态 tuple、`.item()`/`.place` 条件或 Fleet 路径。它们属于当前不支持或尚未验证的静态
范围，统一留在动态图。真正的 CINN 边界由 `run_model.py` 在模型加载后显式包住 MLP
子层；因此动态模式和 CINN 模式共享同一份模型数学实现，差异只有 MLP 子图是否进入
CINN。

## 11. 最小验收

- PaddleFormers commit 匹配。
- patch 可应用或已应用。
- checkpoint 全部 hash 通过。
- data manifest 与 NPZ hash 通过。
- dynamic train exit 0，loss 有限。
- CINN train exit 0，loss 有限且有编译标志。
- dynamic infer exit 0，logits checksum 有限。
- CINN infer exit 0，checksum 有限且有编译标志。
- CINN 结果的 `cinn_scope` 为 `model.model.language_model.layers[*].mlp`，`static_subgraph_count`
  与 checkpoint 文本层数一致；动态结果的 scope 为 `none`/`0`。

`run_model_common.sh` 会把上述验收串起来：如果 commit、patch、checkpoint/data hash 或 CINN 编译标志不满足要求，入口会返回非零；它本身不改变 Gemma3 的网络数学定义。
