# 本地 checkpoint

本次不上传任何权重。通过 CHECKPOINT_DIR 指定 initial.pdparams 与 inference.pdparams，校验信息见 ../asset_manifest.json。

initial 是随机tiny共同起点；inference 是历史动态图300步权重，复核推理性能必须两端使用同一份。训练恢复状态不作为推理权重加载。权重获取地址待维护者提供，当前只能使用已持有的本地文件。
