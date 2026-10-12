# OLMo2 tiny CINN 复现

框架测试用随机缩小模型：2层、hidden64、FFN128、4个query头/2个KV头、head_dim16，保留原OLMo2 tokenizer及词表，共12,919,296参数。单卡V100S 32GB，FP32、TF32关闭、eager attention、无AMP/分布式。不是预训练权重缩层，不代表原始1B模型默认CINN问题已修复。

## 编译边界与训练协议

默认整模型forward经`to_static(full_graph=True, backend="CINN")`编译；显式传入token ID Tensor，返回logits Tensor。cross_entropy和AdamW在动态图侧；配置`get`保留Python读取。没有算子分解黑名单、Norm替换或模型公式修改。

GSM8K固定前24条，20训练/4留出，question+换行+answer分词，取前65 token形成长度64的下一token预测，填充标签忽略。数据来自 https://github.com/openai/grade-school-math （MIT）。这是prefix LM子集拟合，不是answer-only SFT或任务泛化评测。AdamW lr=2e-5、weight_decay=0.01，batch1，seed20261007；动态/CINN共用initial.pdparams，最终各30000步；10000步后恢复完整模型、优化器及随机状态续跑，不重新初始化。保存每步loss及每100步留出loss，每500步保存可恢复状态。

## 环境与源码复现

Python3.10.20、Paddle3.5.0.dev20260722，Paddle commit `08de6be508389d5277b02e78fbc8742f1ed78160`，编译CUDA12.6/cuDNN9.5.1，transformers5.3.0、numpy1.26.4、tokenizers0.22.2，及PaddleFormers requirements。

历史实际测试源码是本地快照`44c39fe19f7bc0b3c45c59562038c0b2706d4624`，不能假定该commit可从GitHub拉取。使用公开基线和补丁还原：

```bash
git clone https://github.com/PaddlePaddle/PaddleFormers.git /path/to/PaddleFormers
git -C /path/to/PaddleFormers checkout 440560443e36364e87522cc1f49a1bb021f66c8d
git -C /path/to/PaddleFormers apply --index /path/to/olmo2/source_snapshot.patch
git -C /path/to/PaddleFormers write-tree
# 预期：511d04c22df3787ab3053c065d5c79d1fa8420ef
```

已实际还原并验证tree哈希与历史快照一致。补丁记录原环境附带的MiniMax注册/实现，不改OLMo2代码；为精确复现保留，不是OLMo2的CINN修复。脚本校验相对公开基线的完整差异，拒绝额外源码修改。

## 运行

权重不上传。外部CHECKPOINT_DIR提供initial.pdparams（共同随机初始化）和inference.pdparams（动态图30000步最终权重），SHA256/大小见asset_manifest.json。资产校验不通过即失败，不自动生成其他随机权重替代。恢复训练仅使用可信本地pdstate。

```bash
export PADDLEFORMERS_DIR=/path/to/PaddleFormers
export CHECKPOINT_DIR=/path/to/weights
export PYTHON_BIN=/path/to/python
./run_model.sh
./run_model_cinn.sh
MODE=dynamic STEPS=30000 RESULTS_DIR=results/full ./train.sh
MODE=cinn STEPS=30000 RESULTS_DIR=results/full ./train.sh
MODE=dynamic RESULTS_DIR=results/infer1 ./infer.sh
MODE=cinn RESULTS_DIR=results/infer1 ./infer.sh
# 续跑时使用相同initial权重，并提供该模式自己的完整状态：
# MODE=dynamic STEPS=30000 RESUME=/path/to/training_state.pdstate RESULTS_DIR=results/resumed ./train.sh
```

快速入口各20步训练加独立进程推理；GPU_ID默认0，重复运行需要新RESULTS_DIR，拒绝覆盖。首次编译耗时不算稳态收益，短入口仅验证打通，不证明长期收敛。

独立推理两端读取同一个inference.pdparams，固定首条训练前缀forward，不是generate。每次10次预热/30次逐次同步计时，记录inference.json及logits.npy。训练性能统计每个进程剔除前100步后取中位数，包含forward/backward/optimizer与Python调度，排除数据组装、评估和保存。吞吐提升=(动态耗时/CINN耗时-1)×100%，不是延迟下降比例。

## 验收范围

2026-10-07已完成两端各30000步：末500步训练loss为0.003588/0.003526，留出loss为11.060065/11.054883。稳态训练10.258/3.231ms，吞吐提升217.47%；同权重独立前向4.687/0.788ms，吞吐提升494.58%，logits最大差1.383e-5（单次配对）。公开基线加补丁后的两端20步入口回归通过，25项参数梯度齐全有限。详见`reference_results/README.md`、原始曲线和JSON；完整逐步日志保存在`reference_results/logs/`，未上传权重。

依据已确认的框架测试口径，验证两端训练子集充分拟合、趋势接近、loss/梯度有限及同权重独立推理。原始loss不缩放、不平滑；同时记录尾段误差和留出结果，不隐瞒过拟合，不把小loss等同于应用质量。旧1B模型的编译阻塞材料仍保留，本次只证明该tiny配置的默认整图路径可行。
