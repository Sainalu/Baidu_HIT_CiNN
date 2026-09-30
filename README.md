# PaddleFormers CINN Model Test Suite

该仓库为多个模型提供独立、可复现的 PaddleFormers 动态图/CINN 测试入口。每个模型目录至少包含：

```text
<model>/
├── run_model.sh
├── run_model_cinn.sh
├── run_model.py
└── run_model.md
```

## 使用方式

先拉取并固定 PaddleFormers，再拉取本仓库。以 Gemma3 为例：

```bash
git clone https://github.com/PaddlePaddle/PaddleFormers.git
git -C PaddleFormers checkout 941b463475ef0bbb3f88e2d1e1bcee4181f98e0b
git clone <TEST_REPOSITORY_URL> model-test-suite

cd model-test-suite/gemma3
PADDLEFORMERS_DIR=../../PaddleFormers CHECKPOINT_DIR=/path/to/checkpoint ./run_model.sh
PADDLEFORMERS_DIR=../../PaddleFormers CHECKPOINT_DIR=/path/to/checkpoint ./run_model_cinn.sh
```

Gemma3 需要对固定 PaddleFormers commit 应用兼容 patch；两个入口会校验 commit 并幂等应用 patch。checkpoint 体积较大，因此默认通过 `CHECKPOINT_DIR` 或 `checkpoint/LOCATION.md` 定位，不直接提交普通 Git。若仓库服务支持 Git LFS，可按 `gemma3/checkpoint/README.md` 管理权重。

## 约定

- `run_model.sh`：动态图，不调用 `paddle.jit.to_static`。
- `run_model_cinn.sh`：静态图+CINN，日志必须出现 `Compiling subgraph with CINN backend`。
- 两个入口调用同一个 `run_model.py`，仅模式不同。
- 默认命令做训练冒烟；`TASK=infer` 可做多模态 teacher-forced forward 推理。
- 所有模型必须记录 PaddleFormers commit、Paddle/CUDA/cuDNN、checkpoint hash、数据和精度策略。
- checkpoint、私有数据或访问凭证不得硬编码到脚本或提交历史。

## 当前模型

- `gemma3/`：Gemma3 多模态 BF16 checkpoint，训练及推理动态/CINN 测试。
