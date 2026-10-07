# MiniMax tiny CINN 复现

随机tiny框架测试，FP32、TF32关闭、batch1、seq64。数据是GSM8K前24条，question+换行+answer，经assets测试tokenizer构造下一token标签；前20训练、4留出。不代表完整MiniMax任务质量。

## 环境

V100S 32GB、Python3.10.20、Paddle3.5.0.dev20260722（08de6be508389d5277b02e78fbc8742f1ed78160）、编译CUDA12.6；transformers5.3.0、tokenizers0.22.2、numpy1.26.4及PaddleFormers requirements。无AMP/分布式。

从 https://github.com/yicycyc/PaddleFormers.git 拉取并checkout paddleformers.commit。脚本验证提交和干净工作树。CINN编译完整forward wrapper，loss/optimizer在动态侧；config.get保留Python。仅CINN安装minimax_moe_functional.py的等价专家累加适配，消除旧静态条件分支梯度丢失；没有移除专家或冻结参数。适配不修改源码工作树。

## 运行

权重不上传。CHECKPOINT_DIR包含initial.pdparams（共同随机起点）及inference.pdparams（动态图30000步权重）；严格校验asset_manifest.json，不匹配即失败。data/gsm8k_24.jsonl来源 https://github.com/openai/grade-school-math （MIT），assets为原测试配置/tokenizer。

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

快速入口20步真实更新训练加独立推理，AdamW lr2e-5/weight_decay0.01。推理固定文本前向，非generate，10预热/30计时，排除预处理/H2D/编译；每步前后GPU同步。GPU_ID默认0。需用新RESULTS_DIR，拒绝覆盖。RESUME可指定可信完整状态文件，两端状态不可互换；模型/优化器/RNG每500步及结束保存。

## 验证

results/<mode>-train包含loss.jsonl、validation.jsonl、first_gradients.pdparams等；<mode>-infer保存logits.npy及inference.json。两端独立推理必须使用同一个inference.pdparams，不能比较各自训练产生的不同权重。入口检查CINN编译日志标志，不满足即失败。吞吐提升=(动态ms/CINNms-1)×100%，不能与延迟下降混用。运行产物与权重不提交。

历史30000步仅证明固定子集充分拟合；本次新入口回归是短程训推和梯度检查，不宣称重新完成长跑。

2026-10-07 本地入口回归通过：两端各20步，loss最大绝对差9.54e-7；41项首步梯度均保留，最大绝对差5.73e-6；共同checkpoint独立推理logits最大绝对差8.83e-6。该短测只用于验证交付入口，不替代长程收敛材料。
