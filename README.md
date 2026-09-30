# PaddleFormers CINN Model Test Suite

这是一个按模型组织的 PaddleFormers 动态图/CINN 可复现测试仓。每个模型目录都提供动态图和 CINN 两个公开入口；CINN 入口只编译模型说明中已经验证的子图，其余路径保持动态图。

logs/ 和 results/ 是运行时输出目录，不是 Git 交付物。仓库只保留 .gitkeep；实际运行产生的日志、JSON 和图片会被 .gitignore 忽略。

## 模型目录

~~~text
<model>/
├── run_model.sh                 # 动态图快速训推总入口，默认 TASK=all
├── run_model_cinn.sh            # CINN 快速训推总入口，默认 TASK=all
├── train.sh                     # 正式单任务训练入口，MODE=dynamic|cinn
├── infer.sh                     # 正式单任务推理入口，MODE=dynamic|cinn
├── run_model.py                 # 一次执行一个 task 的模型计算和 JSON 输出
├── run_model_common.sh          # 内部校验、patch、环境和日志编排
├── run_model.md                 # 模型专属环境、数据和运行说明
├── paddleformers.commit         # 固定的 PaddleFormers commit
├── SOURCE.lock                  # 源码和环境锁定信息（如模型提供）
├── patches/                     # 可选的模型兼容补丁
├── checkpoint/                  # checkpoint 说明和本地位置，不提交权重
├── data/                        # 模型使用的固定输入数据
├── logs/                        # 运行时日志，Git 忽略
├── results/                     # 运行时 JSON/图片，Git 忽略
├── docs/                        # 计划、交付清单和状态（如模型提供）
└── scripts/                     # 校验、汇总、绘图等辅助脚本
~~~

四个必须的公开文件是 run_model.sh、run_model_cinn.sh、run_model.py 和 run_model.md。train.sh 和 infer.sh 对应指导文件要求的一键训推脚本，提供可单独复现的正式任务入口。

## 拉取和准备

~~~bash
git clone <PADDLEFORMERS_URL> /path/to/PaddleFormers
git clone <TEST_REPOSITORY_URL> /path/to/model-test-suite
cd /path/to/model-test-suite/<model>
git -C /path/to/PaddleFormers checkout "$(tr -d '[:space:]' < paddleformers.commit)"
~~~

两个目录可以是同级目录，也可以用 PADDLEFORMERS_DIR 指定其他位置。启动器会校验 PaddleFormers HEAD，并幂等应用模型目录下的唯一 patches/*.patch。多个补丁时必须通过 PATCH_FILE 明确选择。

checkpoint 默认从仓库外部提供：

~~~bash
export CHECKPOINT_DIR=/absolute/path/to/checkpoint
~~~

也可以在被忽略的 checkpoint/LOCATION.local 中写入 path=/absolute/path/to/checkpoint。

## 快速训推

run_model.sh 和 run_model_cinn.sh 默认执行训练和推理两个独立进程。默认是 20 step、2 step warmup、learning_rate=0，只用于快速确认加载、loss、CINN 编译和计时链路。

~~~bash
PADDLEFORMERS_DIR=/path/to/PaddleFormers \
CHECKPOINT_DIR=/path/to/checkpoint \
./run_model.sh

PADDLEFORMERS_DIR=/path/to/PaddleFormers \
CHECKPOINT_DIR=/path/to/checkpoint \
./run_model_cinn.sh
~~~

也可以只运行一个任务：

~~~bash
TASK=train ./run_model.sh
TASK=infer INFER_DATA_DIR=data/eval ./run_model.sh
TASK=train ./run_model_cinn.sh
TASK=infer INFER_DATA_DIR=data/eval ./run_model_cinn.sh
~~~

训练和推理是在 shell 层顺序启动的两个 Python 进程，不在同一个 run_model.py 进程中复用模型和静态图缓存。

## 正式单任务入口

~~~bash
MODE=dynamic STEPS=1000 WARMUP_STEPS=20 LEARNING_RATE=1e-6 ./train.sh
MODE=cinn    STEPS=1000 WARMUP_STEPS=20 LEARNING_RATE=1e-6 ./train.sh
MODE=dynamic INFER_STEPS=100 INFER_WARMUP_STEPS=10 ./infer.sh
MODE=cinn    INFER_STEPS=100 INFER_WARMUP_STEPS=10 ./infer.sh
~~~

LEARNING_RATE=0 是固定 checkpoint 的冒烟测试；全量 loss 和收敛测试必须使用正学习率和真实 optimizer step。训练后如需使用新 checkpoint 推理，必须显式设置 CHECKPOINT_DIR。

## 参数约定

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| PADDLEFORMERS_DIR | 模型目录上一级的 PaddleFormers | 上游源码 |
| CHECKPOINT_DIR | 无；可使用 checkpoint/LOCATION.local | checkpoint 目录 |
| TASK | all（run_model 入口） | train、infer 或 all |
| MODE | dynamic（train/infer 入口） | dynamic 或 cinn |
| STEPS / WARMUP_STEPS | 20 / 2 | 单任务 step 和 warmup |
| TRAIN_STEPS / INFER_STEPS | STEPS | TASK=all 的任务参数 |
| TRAIN_WARMUP_STEPS / INFER_WARMUP_STEPS | WARMUP_STEPS | TASK=all 的任务 warmup |
| TRAIN_DATA_DIR / INFER_DATA_DIR | data/train / data/eval | 任务数据 |
| LEARNING_RATE / TRAIN_LEARNING_RATE | 0 / LEARNING_RATE | 训练学习率 |
| GPU_ID / PYTHON_BIN | 0 / python | GPU 和解释器 |
| PATCH_FILE | 自动选择唯一 patch | 多 patch 时显式指定 |
| RESULTS_DIR / LOGS_DIR | results/ / logs/ | 运行输出目录 |
| TRAIN_OUTPUT / INFER_OUTPUT | results/raw/<mode>-*.json | task JSON |
| TRAIN_LOG / INFER_LOG | logs/<mode>/*.log | task 日志 |
| AGGREGATE_OUTPUT | results/<mode>.json | TASK=all 的聚合 JSON |

TASK=all 时不要设置通用 OUTPUT 或 LOG；使用 TRAIN_OUTPUT、INFER_OUTPUT、TRAIN_LOG 和 INFER_LOG，避免覆盖。

## 运行产物

动态入口生成：

~~~text
logs/dynamic/train.log
logs/dynamic/infer.log
results/raw/dynamic-train.json
results/raw/dynamic-infer.json
results/dynamic.json
~~~

CINN 入口生成对应的 cinn 文件。训练日志包含 STEP_LOSS 和 step 时间；推理日志包含 INFER_STEP、latency、checksum 和 CINN 编译信息。聚合 JSON 包含训练和推理两部分，以及 checkpoint、数据、精度、环境和 CINN 范围。

这些文件只用于本地运行、排错和性能查看，不作为 Git 交付物。空目录通过 .gitkeep 保留。

训练 JSON 重点查看 losses、first_step_time_s、mean_steady_step_time_s、steady_fps、optimizer 和 learning_rate。推理 JSON 重点查看 scope、first_step_latency_ms、mean_steady_latency_ms、steady_fps 和 logits checksum。首步编译时间必须与稳态均值分开。

## run_model_common.sh

这是内部启动器，不是第三个用户入口。它负责校验 commit、幂等应用 patch、解析路径和参数、设置 PYTHONPATH/CUDA_VISIBLE_DEVICES、调用 run_model.py、写日志，以及在 CINN 模式检查 Compiling subgraph with CINN backend。TASK=all 时它分别启动 train/infer，再生成模式级聚合 JSON。

## 当前模型

- gemma3/：Gemma3 多模态 BF16 高精度缩层 checkpoint，包含训练和 teacher-forced 推理的动态图/CINN 测试。
