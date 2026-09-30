# PaddleFormers CINN Model Test Suite

这是一个按模型组织的 PaddleFormers 动态图/CINN 可复现测试仓。每个模型目录都提供相同的两个公开入口：动态图入口和 CINN 入口。CINN 入口可以按模型说明只编译可支持的子图，其余模块隔离在动态图；它不默认意味着整个模型都被静态化。模型的 checkpoint、数据、上游 commit、补丁和结果口径由模型目录自己的说明文件定义。

## 模型目录约定

新模型建议按下面的结构创建。`run_model_common.sh` 是推荐的内部公共启动器；如果模型需要不同的启动流程，也可以替换它，但两个公开入口和输出口径应保持一致。

```text
<model>/
├── run_model.sh                 # 动态图公开入口
├── run_model_cinn.sh            # 可支持子图+CINN 公开入口（范围由模型说明定义）
├── run_model.py                 # 模型计算、校验和结果汇总
├── run_model_common.sh          # 两个入口共享的 shell 编排层
├── run_model.md                 # 模型专属环境、数据和验收说明
├── paddleformers.commit         # 要求的上游 PaddleFormers commit
├── patches/                     # 可选，通常放一个兼容补丁
├── data/                        # 固定 shape 的可复现测试数据
├── checkpoint/                  # checkpoint 说明，不默认提交权重
├── logs/                        # 运行时生成的日志
└── results/                     # 运行时生成的 JSON 结果
```

模型目录至少需要 `run_model.sh`、`run_model_cinn.sh`、`run_model.py` 和 `run_model.md`。模型专属的 checkpoint 架构、数据格式和额外环境要求以 `run_model.md` 为准。

## 通用使用流程

下面的命令是模板。将 `<model>`、`/path/to/...` 和上游仓库地址替换成实际值。

### 1. 准备上游代码和测试仓

```bash
git clone <PADDLEFORMERS_URL> /path/to/PaddleFormers
git clone <TEST_REPOSITORY_URL> /path/to/model-test-suite

cd /path/to/model-test-suite/<model>
git -C /path/to/PaddleFormers checkout "$(tr -d '[:space:]' < paddleformers.commit)"
```

`paddleformers.commit` 是模型测试的一部分，不能用任意最新 commit 替代。入口启动时会再次校验当前 HEAD；如果模型目录有一个 `patches/*.patch`，启动器会自动幂等应用它。多个 patch 时通过 `PATCH_FILE` 明确选择。

### 2. 运行动态图和 CINN

```bash
cd /path/to/model-test-suite/<model>

# 动态图训练冒烟
PADDLEFORMERS_DIR=/path/to/PaddleFormers \
CHECKPOINT_DIR=/path/to/checkpoint \
TASK=train STEPS=20 WARMUP_STEPS=2 LEARNING_RATE=0 \
./run_model.sh

# CINN 训练冒烟
PADDLEFORMERS_DIR=/path/to/PaddleFormers \
CHECKPOINT_DIR=/path/to/checkpoint \
TASK=train STEPS=20 WARMUP_STEPS=2 LEARNING_RATE=0 \
./run_model_cinn.sh
```

推理测试使用同一套入口，只需切换 `TASK` 和数据目录：

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

正式比较动态/CINN 时，必须保持 checkpoint、数据、batch size、sequence length、step 数、warmup、学习率、随机种子和硬件一致。`LEARNING_RATE=0` 只表示固定 checkpoint 的前向/反向性能冒烟，不会更新参数，也不能称为收敛训练。

### 3. 参数约定

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `PADDLEFORMERS_DIR` | 模型目录上一级的 `PaddleFormers` | 上游源码目录 |
| `CHECKPOINT_DIR` | 无；也可使用 `checkpoint/LOCATION.local` | 外部 checkpoint 目录 |
| `DATA_DIR` | `data/train`（训练）或 `data/eval`（推理） | 已预处理的测试数据目录 |
| `TASK` | `train` | `train` 或 `infer` |
| `STEPS` | `20` | 总运行步数 |
| `WARMUP_STEPS` | `2` | 不计入稳态指标的步数 |
| `LEARNING_RATE` | `0` | 正值启用 AdamW 更新；零值只做 no-update 冒烟 |
| `GPU_ID` | `0` | 通过 `CUDA_VISIBLE_DEVICES` 选择物理 GPU |
| `PYTHON_BIN` | `python` | Python 解释器 |
| `PATCH_FILE` | 自动选择唯一 `patches/*.patch` | 多补丁模型的显式选择 |
| `OUTPUT` | `results/<task>-<mode>.json` | 结果 JSON 路径 |
| `LOG` | `logs/<task>-<mode>.log` | 原始运行日志路径 |

相对路径以当前模型目录为基准；以模型目录名开头的路径（例如 `<model>/logs/a.log`）以测试仓根目录为基准。为避免歧义，跨目录的 checkpoint、日志和结果建议使用绝对路径。

checkpoint 权重默认不进普通 Git。可以设置 `CHECKPOINT_DIR`，也可以在被忽略的 `checkpoint/LOCATION.local` 中写一行：

```text
path=/absolute/path/to/checkpoint
```

不要把机器路径、访问凭证或私有数据写进受版本控制的文件。

## 测试会产生什么结果

每次运行会同时产生两类输出：

| 输出 | 内容 | 用途 |
| --- | --- | --- |
| `logs/*.log` | 上游 warning、CINN 编译信息、逐步 `STEP_LOSS`/`INFER_STEP` 和 `RUN_SUMMARY` | 人工排查和确认 CINN 是否真正生效 |
| `results/*.json` | 环境、hash、每步记录和汇总指标 | 程序化对比、绘图和归档 |

运行生成的日志和 JSON 默认被 `.gitignore` 忽略，避免把不同机器的结果混入代码提交。需要交付结果时，应将经过确认的汇总或图表单独归档；仓库中的 `results/clone-test-summary.json` 是一份已脱敏的清洁克隆验收记录。

### 训练结果字段

- `losses`/`records[].loss`：每一步的 FP32 loss；只能先判断是否有限（没有 NaN/Inf），不能用 3～20 步冒烟判断收敛。
- `first_step_time_s`：第一个训练 step 的完整耗时。CINN 模式包含首次 trace、编译和执行，通常不能作为稳态性能。
- `mean_steady_step_time_s`：去掉 `warmup_steps` 后的平均 step 时间。
- `steady_fps`：稳态吞吐。当前 Gemma3 runner 的 batch size 是 1，计算为 `1 / mean_steady_step_time_s`；其他模型应在自己的 `run_model.md` 中明确 batch size 和 FPS 定义。
- `optimizer`/`learning_rate`：说明是否真正更新参数；`no-update/clear-grad` 不是训练收敛结果。

### 推理结果字段

- `scope`：当前 Gemma3 runner 是多模态 teacher-forced forward logits，不是 autoregressive `generate`。
- `first_step_latency_ms`：第一个 forward 的端到端模型计算耗时；CINN 模式包含懒编译。
- `mean_steady_latency_ms`：去掉 warmup 后的平均 forward 延迟。
- `steady_fps`：稳态 forward 吞吐。当前 Gemma3 runner 的 batch size 是 1，计算为 `1000 / mean_steady_latency_ms`；其他模型应在自己的 `run_model.md` 中明确计算口径。
- `logits_checksum`/`logits_abs_checksum`：有限性和粗粒度数值回归信号，不是精度指标，也不要求动态图与 CINN 位级相等。
- `cinn_scope`/`static_subgraph_count`：实际进入 CINN 的边界和子图数量；模型若采用局部路线，应同时记录 `dynamic_scope`。

### 动态/CINN 对比口径

训练吞吐提升比例按下面计算：

```text
(CINN steady_fps / 动态图 steady_fps - 1) * 100%
```

推理延迟降低比例按下面计算：

```text
(动态图 mean_steady_latency_ms - CINN mean_steady_latency_ms)
/ 动态图 mean_steady_latency_ms * 100%
```

首步编译时间要单独报告，不能混入稳态提升比例。真正的精度/收敛验收需要正学习率、足够的 step、相同数据和相同 checkpoint，并对比 loss 曲线或任务指标；本仓库的短步结果主要用于功能、数值有限性和性能口径检查。

### 最小验收条件

1. 上游 HEAD 与 `paddleformers.commit` 一致。
2. checkpoint 和数据 manifest 的文件大小/hash 全部通过。
3. 动态图训练 loss 有限，动态图推理 checksum 有限。
4. CINN 训练/推理成功，日志出现 `Compiling subgraph with CINN backend`。
5. JSON 中的环境、精度、数据 hash 和计时口径完整。

## `run_model_common.sh` 是什么

它不是模型网络，也不是第三个用户入口，而是两个公开 shell 入口共用的启动编排层：

```text
run_model.sh       -> 设置 CINN_MODE=dynamic -> run_model_common.sh
run_model_cinn.sh  -> 设置 CINN_MODE=cinn    -> run_model_common.sh
                                             -> run_model.py
```

它按固定顺序完成以下工作：

1. 读取当前模型的 `paddleformers.commit` 并校验上游 HEAD。
2. 自动发现并幂等应用模型的唯一兼容 patch；没有 patch 的模型直接继续。
3. 解析 checkpoint、数据、GPU、step、warmup、输出和日志路径。
4. 设置 `PYTHONPATH`、`CUDA_VISIBLE_DEVICES`，调用模型目录自己的 `run_model.py`。
5. 将标准输出复制到日志文件；CINN 模式额外检查日志中是否出现真实编译标志，缺失就返回非零。

因此，使用者通常只调用 `run_model.sh` 或 `run_model_cinn.sh`，不直接执行 `run_model_common.sh`。模型计算、数据读取、loss、logits 和 JSON schema 在 `run_model.py` 中；公共启动器只负责一致性校验和运行编排。

应用补丁会使上游 PaddleFormers 工作树出现预期的修改；重复运行会识别补丁已应用，不会重复叠加。正式测试建议使用与目标 commit 对齐的独立 checkout，避免和其他开发修改混用。

## 当前模型

- [`gemma3/`](gemma3/run_model.md)：Gemma3 多模态 BF16 高精度缩层 checkpoint，包含训练和 teacher-forced 推理的动态图/CINN 测试。

Gemma3 的具体 commit、checkpoint hash、数据来源、已知环境风险和本次清洁克隆结果见 [`gemma3/run_model.md`](gemma3/run_model.md) 与 [`gemma3/results/clone-test-summary.json`](gemma3/results/clone-test-summary.json)。
