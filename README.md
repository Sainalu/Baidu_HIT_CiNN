# PaddleFormers CINN Model Test Suite

这是一个按模型组织的 PaddleFormers 动态图/CINN 测试仓。每个模型目录提供统一的运行入口和模型说明；其他文件由模型实现按需组织，不作为跨模型的固定模板。

## 模型目录

下表说明当前仓库对外文件和目录的功能。合作方可以根据模型需要增删辅助文件，不需要遵循统一的附加目录结构。

| 文件或目录 | 功能 |
| --- | --- |
| `run_model.sh` | 动态图快速训推入口。 |
| `run_model_cinn.sh` | 开启 CINN 的快速训推入口。 |
| `train.sh` | 单独执行训练任务的入口。 |
| `infer.sh` | 单独执行推理任务的入口。 |
| `run_model.py` | 加载模型和数据，执行任务并记录性能结果。 |
| `run_model.md` | 模型专属的环境、数据、运行方式和结果说明。 |
| `checkpoint/` | checkpoint 的说明或外部存放位置记录；权重是否入库由项目决定。 |
| `data/` | 测试所需的数据或数据说明。 |
| `logs/` | 运行产生的训练、推理日志。 |
| `results/` | 运行产生的 JSON、图片和其他结果文件。 |

## 主要交付物说明

- `run_model.sh`、`run_model_cinn.sh`、`train.sh`、`infer.sh`、`run_model.py` 和 `run_model.md` 是每个模型的必需交付文件。
- 每个模型应能从自己的说明文件中找到入口、环境和结果查看方式。
- 除上述交付文件外，其他源码、兼容处理和辅助工具由模型维护者自行选择组织方式。
- 如果需要增加其他模型适配内容，应在对应模型的 `run_model.md` 中说明。
- `logs/` 和 `results/` 中的运行产物默认只保存在本地，是否提交由项目需求决定。

## 运行

先拉取 PaddleFormers 和本测试仓库：

```bash
git clone <PADDLEFORMERS_URL> /path/to/PaddleFormers
git clone <TEST_REPOSITORY_URL> /path/to/model-test-suite
```

进入任一模型目录，执行两个 `run_model` 入口即可完成动态图和 CINN 的快速训推测试：

```bash
cd /path/to/model-test-suite/<model>

PADDLEFORMERS_DIR=/path/to/PaddleFormers \
CHECKPOINT_DIR=/path/to/checkpoint \
./run_model.sh

PADDLEFORMERS_DIR=/path/to/PaddleFormers \
CHECKPOINT_DIR=/path/to/checkpoint \
./run_model_cinn.sh
```

具体环境变量、数据位置、checkpoint 和任务参数以对应模型的 `run_model.md` 为准。后续新增模型沿用同一项目级入口约定。
