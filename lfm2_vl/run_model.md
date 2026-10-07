# lfm2_vl 图文 CINN 复现

用于框架测试的随机 tiny、FP32、关闭TF32、单卡batch1；20张训练/4张留出真实图片，类别ID数字token任务，不是原模型caption/OCR/安全质量评测。模型结构和预处理固定在run_model.py，运行生成config/；不更换成完整模型。视觉encoder层和文本层编译；位置插值、形状处理与图文组装动态。保留conv/full_attention/conv文本结构，图像16×16。

## 环境与资产

本地历史环境：V100S 32GB、Python3.10.20、Paddle3.5.0.dev20260722，commit 08de6be508389d5277b02e78fbc8742f1ed78160，编译CUDA12.6，numpy1.26.4、Pillow12.3.0。安装相应Paddle GPU包及PaddleFormers requirements。无AMP、无分布式。

从 https://github.com/yicycyc/PaddleFormers.git 拉取源码并checkout本目录paddleformers.commit（必须是历史测试提交，未合入模型也需要该fork）。入口校验提交与干净工作树，不自动修改PR源码。

权重不上传、不下载，CHECKPOINT_DIR中放initial.pdparams和inference.pdparams，分别为历史随机初始权重和动态图10000步最终权重；两端训推各共用对应权重。asset_manifest.json记录大小/SHA256，不匹配即失败。data/images24.npz是固定真实ImageNet子集，来源列表见images_manifest.json；仅用于此框架实验，使用须遵守原数据集使用条款。

## 运行

```bash
export PADDLEFORMERS_DIR=/path/to/PaddleFormers
export CHECKPOINT_DIR=/path/to/local-weights
export PYTHON_BIN=/path/to/python
./run_model.sh
./run_model_cinn.sh
MODE=dynamic STEPS=10000 RESULTS_DIR=results/full ./train.sh
MODE=cinn STEPS=10000 RESULTS_DIR=results/full ./train.sh
MODE=dynamic RESULTS_DIR=results/infer1 ./infer.sh
MODE=cinn RESULTS_DIR=results/infer1 ./infer.sh
```

两个快速入口分别用独立进程执行20步训练和固定前向推理。AdamW学习率2e-5，其他参数按原脚本，不是零学习率；完整训练目标10000步。推理预热10/计时30，前后GPU同步，排除数据预处理/H2D及首次编译，不是generate。GPU_ID默认0。重复运行指定新RESULTS_DIR，拒绝覆盖。

## 验证边界

CINN显式to_static，日志缺少Compiling subgraph with CINN backend则入口失败。训练loss.jsonl逐步记录，validation.jsonl记录留出结果，gradients.npz保存首步参数梯度；推理保存logits.npy及逐次计时inference.json。原始子集拟合与泛化效果分开，不因短测跑通就宣称收敛。权重、日志和生成结果不提交Git。

本目录是历史已完成长跑脚本的可移植入口；交付回归只做20步及独立推理，不重新宣称执行全部长跑。吞吐提升=(动态ms/CINNms-1)×100%，延迟下降=(1-CINNms/动态ms)×100%。性能只代表当前tiny及编译范围。

2026-10-07 本地交付回归通过：两端各20步，loss最大绝对差9.54e-7；共同权重独立推理logits最大差1.79e-6；72项首步梯度均保留，最大绝对差1.43e-6。本次仅为短程入口验证。
