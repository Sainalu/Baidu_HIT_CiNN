# Gemma3 动态图/CINN 复现说明

## 1. 入口

| 文件 | 作用 |
| --- | --- |
| `run_model.sh` | 调用同一个 `run_model.py`，运行纯动态图 |
| `run_model_cinn.sh` | 调用同一个 `run_model.py`，运行 `to_static(full_graph=True)`+CINN |
| `run_model.py` | checkpoint/data 校验、训练或推理、计时和 JSON 输出 |
| `patches/paddleformers-gemma3-cinn.patch` | 固定 PaddleFormers commit 的 Gemma3 CINN 兼容补丁 |
| `paddleformers.commit` | 要求的上游 commit |
| `checkpoint/README.md` | checkpoint 结构、精度、大小和 SHA256 |

脚本会先检查 PaddleFormers commit，再幂等应用 patch：已应用则直接运行；未应用且可干净应用则自动应用；源码状态不明确时停止，避免覆盖用户修改。

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
git clone https://github.com/PaddlePaddle/PaddleFormers.git
git -C PaddleFormers checkout 941b463475ef0bbb3f88e2d1e1bcee4181f98e0b
git clone <TEST_REPOSITORY_URL> model-test-suite
```

两个目录可以是同级目录，也可以通过 `PADDLEFORMERS_DIR` 指定其他位置。

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

也可以仅在本地向 `checkpoint/LOCATION.md` 增加一行 `path=/absolute/path/...`，不要提交机器路径。

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
TASK=infer DATA_DIR=data/eval STEPS=20 WARMUP_STEPS=2 ./run_model.sh
TASK=infer DATA_DIR=data/eval STEPS=20 WARMUP_STEPS=2 ./run_model_cinn.sh
```

计时包含模型 forward 和 GPU 同步，不含 NPZ 读取、Tensor 创建及 Host-to-Device。输出包含 latency、FPS、首步时间和 logits checksum。

## 8. CINN 生效与计时口径

CINN 唯一可信的生效标志：

```text
Compiling subgraph with CINN backend
```

`run_model_cinn.sh` 会检查该标志，缺失时返回非零。首次 forward 包含懒 trace/CINN 编译；稳态指标排除 `WARMUP_STEPS`。报告性能时必须分别说明首编译时间和稳态时间。

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

可以用 `LOG` 和 `OUTPUT` 覆盖。正式对比时，动态/CINN 必须使用相同 checkpoint、数据、step、warmup、精度和学习率。

## 10. PaddleFormers 修改说明

补丁包含：

- 注册 Gemma3 多模态数据模板。
- sliding-window mask 向量化。
- 避免静态 trace 中的 `paddle.finfo`、Python autocast context 和动态 tuple 处理。
- 避免静态图 Tensor `.place` 和 `.item()` Python 条件。
- 支持缩层 checkpoint 按目标层数加载。
- 单卡 TP=1 不访问未初始化 Fleet 通信组。
- 相应等价性/回归单测。

这些修改不改变 attention、MLP、视觉编码器和 loss 的数学定义。动态图和 CINN 使用同一份已打补丁源码。

## 11. 最小验收

- PaddleFormers commit 匹配。
- patch 可应用或已应用。
- checkpoint 全部 hash 通过。
- data manifest 与 NPZ hash 通过。
- dynamic train exit 0，loss 有限。
- CINN train exit 0，loss 有限且有编译标志。
- dynamic infer exit 0，logits checksum 有限。
- CINN infer exit 0，checksum 有限且有编译标志。
