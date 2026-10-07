# Molmo 图文 CINN 复现

用于框架测试的随机 tiny，FP32、关闭 TF32、单卡 batch 1。固定真实图片子集：20 张训练、4 张留出，学习类别 ID 数字 token，不是完整模型任务质量评测。两端均启用 molmo_attention_boundary.py 的 attention 边界适配；CINN 编译视觉残差层、文本 MLP 与 attention 核心，RoPE 系数生成与图文组装留在动态侧。没有将整个文本 attention 排除编译。

## 环境与资产

本地环境：V100S 32GB，Python 3.10.20，Paddle 3.5.0.dev20260722（08de6be508389d5277b02e78fbc8742f1ed78160，编译 CUDA 12.6），numpy 1.26.4。安装匹配的 Paddle GPU 包及 PaddleFormers requirements。无 AMP、无分布式。

从 https://github.com/yicycyc/PaddleFormers.git 拉取并 checkout 本目录 paddleformers.commit。入口验证固定提交和干净工作树。模型配置及预处理保留在 run_model.py，不自动替换成完整模型。

权重暂不上传。CHECKPOINT_DIR 放 initial.pdparams 与 inference.pdparams，分别为历史共同随机起点与动态图 30000 步权重。asset_manifest.json 校验权重、数据和 tokenizer 的大小及 SHA256，缺失或不匹配即失败。data/images_manifest.json 记录真实 ImageNet 图片来源，使用须遵守原数据集条款。初始权重根据原 seed 重建，并验证全部参数哈希与历史初始记录完全一致。

## 运行

```bash
export PADDLEFORMERS_DIR=/path/to/PaddleFormers
export CHECKPOINT_DIR=/path/to/local-weights
export PYTHON_BIN=/path/to/python
./run_model.sh
./run_model_cinn.sh
MODE=dynamic STEPS=30000 RESULTS_DIR=results/full ./train.sh
MODE=cinn STEPS=30000 RESULTS_DIR=results/full ./train.sh
MODE=dynamic RESULTS_DIR=results/infer1 ./infer.sh
MODE=cinn RESULTS_DIR=results/infer1 ./infer.sh
```

默认每端 20 步真实 AdamW 更新加独立进程固定前向推理；学习率 2e-5。推理不是 generate，预热 10 次、计时 30 次，GPU 同步，排除预处理、H2D 和首次编译。GPU_ID 默认为 0，重复运行必须指定新的 RESULTS_DIR。CINN 日志缺少编译标志则入口失败。

## 验证边界

训练输出 loss.jsonl、validation.jsonl、首步梯度及权重；独立推理输出 logits.npy、inference.json。两端使用相同 inference.pdparams。吞吐提升=(动态耗时/CINN耗时-1)×100%，与延迟下降不同。运行产物不上传。

历史长跑用于固定子集拟合；本次交付短测仅验证脚本移植及数值、梯度路径，不宣称重新完成长程训练，不代表原始完整模型质量或泛化能力。

2026-10-07 本地交付回归通过：两端各20步，loss最大绝对差4.77e-7；共同权重独立推理logits最大差2.83e-5；72项首步梯度均保留，最大绝对差1.46e-3。本次仅为短程入口验证。
