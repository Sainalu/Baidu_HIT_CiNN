# Gemma3 动态图/CINN 复现说明

本文件描述 Gemma3 的模型专属要求；通用目录、入口参数和运行产物见仓库根目录 README.md。

## 1. 入口和文件职责

| 文件 | 作用 |
| --- | --- |
| run_model.sh | 动态图快速训推总入口，默认依次运行 train 和 infer |
| run_model_cinn.sh | CINN 快速训推总入口，默认依次运行 train 和 infer |
| train.sh | 正式单任务训练入口，MODE=dynamic 或 cinn |
| infer.sh | 正式单任务推理入口，MODE=dynamic 或 cinn |
| run_model.py | 一次运行一个 task，负责加载、计算、计时和 task JSON |
| run_model_common.sh | commit、patch、路径、环境、日志和 all-task 编排 |
| scripts/aggregate_results.py | 合并 train/infer JSON 为 results/<mode>.json |
| patches/paddleformers-gemma3-cinn.patch | Gemma3 数据/checkpoint 接入补丁 |
| paddleformers.commit | 要求的上游 commit |
| checkpoint/README.md | checkpoint 结构、精度、大小和 SHA256 |

run_model.sh 和 run_model_cinn.sh 在 shell 层启动两个独立的 Python 进程，不在同一进程中切换训练和推理。

## 2. 静态边界与动态图隔离

本版本不把整个 Gemma3 多模态 forward、训练 loss 或推理 wrapper 包成一个 to_static 图。经过验证的 CINN 边界只有语言模型各 decoder layer 的 MLP 子层：

~~~text
CINN:    model.model.language_model.layers[*].mlp
动态图:  vision tower、图像特征/占位符融合、mask、RoPE、RMSNorm、
         attention、decoder layer loop、final norm、lm_head、loss、
         训练/推理 wrapper
~~~

CINN flags 只在 cinn 模式开启；flags 本身不会自动把所有动态图代码转换成静态图。无法支持静态图的部分保持原始动态图执行，也不通过补丁改写它们。梯度可以跨越 MLP 子图回传，因此训练仍包含完整的 forward、backward、optimizer step 和 clear-grad。

## 3. 环境

第一阶段验证环境：

~~~text
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
~~~

验证节点曾出现 Paddle 编译 cuDNN 9.13、运行库 cuDNN 9.9 的版本差异。正式性能结果需要统一软件环境。

~~~bash
python -c 'import paddle; print(paddle.__version__, paddle.version.commit, paddle.version.cuda(), paddle.version.cudnn())'
nvidia-smi
~~~

## 4. 拉取上游代码

~~~bash
git clone <PADDLEFORMERS_URL> /path/to/PaddleFormers
git clone <TEST_REPOSITORY_URL> /path/to/model-test-suite
git -C /path/to/PaddleFormers checkout 941b463475ef0bbb3f88e2d1e1bcee4181f98e0b
~~~

两个目录可以是同级目录，也可以通过 PADDLEFORMERS_DIR 指定。启动器会校验 commit，并幂等应用 patches/paddleformers-gemma3-cinn.patch。

## 5. Checkpoint

使用高精度缩层 Paddle checkpoint：

~~~text
architecture: Gemma3ForConditionalGeneration
text layers: 4
vision layers: 2
weight dtype: BF16
model_state.pdparams bytes: 3517742411
model_state.pdparams SHA256:
33029a880dc17d16abdac2327e7891bbb1fb7483bfaaf4d962916c5fa8f4f75c
~~~

checkpoint 必须包含 config、tokenizer、processor/preprocessor 和 checkpoint_manifest.json。每次运行都会验证文件大小和 SHA256。

普通 Git 不保存 3.4 GiB 权重。请设置：

~~~bash
export CHECKPOINT_DIR=/path/to/gemma3-4b-text4-vision2-bf16
~~~

或者在本地创建被忽略的 checkpoint/LOCATION.local，写入一行 path=/absolute/path/...。不要提交机器路径。

## 6. 数据

data/ 保存真实 Gemma3Processor 产生的固定 shape NPZ：

- 8 个训练 batch；
- 6 个 eval/infer batch；
- batch size=1；
- sequence length=512；
- 每个 batch 一张真实图片；
- manifest.json 保存输入来源和每个 NPZ 的 SHA256。

默认训练数据为 data/train，推理数据为 data/eval。数据预处理发生在模型计时之外。

## 7. 快速训推

动态 quick run：

~~~bash
cd model-test-suite/gemma3
PADDLEFORMERS_DIR=/path/to/PaddleFormers \
CHECKPOINT_DIR=/path/to/checkpoint \
./run_model.sh
~~~

CINN quick run：

~~~bash
PADDLEFORMERS_DIR=/path/to/PaddleFormers \
CHECKPOINT_DIR=/path/to/checkpoint \
./run_model_cinn.sh
~~~

默认是 train 和 infer 各 20 step，warmup 2，learning_rate=0。只跑单个任务时可以设置 TASK=train 或 TASK=infer。

正式训练：

~~~bash
MODE=dynamic STEPS=1000 WARMUP_STEPS=20 LEARNING_RATE=1e-6 ./train.sh
MODE=cinn STEPS=1000 WARMUP_STEPS=20 LEARNING_RATE=1e-6 ./train.sh
~~~

零学习率只能用于固定 checkpoint 的功能/性能冒烟，不能称为全量收敛训练。

## 8. 推理定义和命令

当前推理是多模态 teacher-forced forward logits，不是 autoregressive generate。计时包含模型 forward 和 GPU synchronize，不包含 NPZ 读取、Tensor 创建和 Host-to-Device。

~~~bash
MODE=dynamic INFER_STEPS=100 INFER_WARMUP_STEPS=10 ./infer.sh
MODE=cinn INFER_STEPS=100 INFER_WARMUP_STEPS=10 ./infer.sh
~~~

如果训练后要评估新 checkpoint，必须显式设置 CHECKPOINT_DIR。

## 9. 运行产物

run_model.sh 默认生成：

~~~text
logs/dynamic/train.log
logs/dynamic/infer.log
results/raw/dynamic-train.json
results/raw/dynamic-infer.json
results/dynamic.json
~~~

run_model_cinn.sh 生成对应的 cinn 文件。日志和 JSON 都是运行产物，不属于 Git 交付物；它们只在当前机器的 logs/ 和 results/ 下保存，目录通过 .gitkeep 保留。

训练 JSON 重点字段：

- losses 和 records[].loss：逐 step FP32 loss；
- first_step_time_s：包含首步 trace/编译；
- mean_steady_step_time_s、steady_fps：排除 warmup 后的稳态训练速度；
- optimizer、learning_rate：是否执行参数更新；
- cinn_scope、static_subgraph_count：实际静态子图范围和数量。

推理 JSON 重点字段：

- scope：teacher-forced forward 定义；
- first_step_latency_ms：包含 CINN 首次编译；
- mean_steady_latency_ms、steady_fps：稳态推理指标；
- logits_checksum、logits_abs_checksum：有限性和粗粒度回归信号。

训练吞吐提升按 CINN steady_fps / dynamic steady_fps - 1 计算；推理延迟下降按 dynamic latency - CINN latency 再除以 dynamic latency 计算。首步编译时间单独报告。

## 10. CINN 生效判据

CINN 唯一可信的日志标志是：

~~~text
Compiling subgraph with CINN backend
~~~

run_model_cinn.sh 和 train.sh/infer.sh 的 cinn 模式发现该标志缺失时返回非零。当前四层文本 checkpoint 的 static_subgraph_count 应为 4；编译 marker 数量是后端事件数，不一定等于包装层数。

## 11. PaddleFormers 修改说明

补丁只负责：

- 注册 Gemma3 多模态数据模板；
- 支持高精度缩层 checkpoint 加载时过滤多余权重。

补丁不修改 sliding-window mask、paddle.finfo、autocast、RoPE、RMSNorm、attention、动态 tuple、item/place 条件或 Fleet 路径。它们属于当前不支持或尚未验证的静态范围，保持动态图。

## 12. 运行前检查

- PaddleFormers HEAD 与 paddleformers.commit 一致；
- checkpoint manifest、文件大小和 SHA256 通过；
- data manifest 和 NPZ SHA256 通过；
- dynamic train/infer 可运行；
- cinn train/infer 日志出现 CINN 编译标志；
- dynamic 与 cinn 使用相同 checkpoint、数据、step、warmup、精度和硬件。
