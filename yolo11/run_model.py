import argparse
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("--task", choices=["train", "infer"], required=True)
p.add_argument("--mode", choices=["dynamic", "cinn"], required=True)
p.add_argument("--output", required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--epochs", default="1")
a = p.parse_args()
root = Path(__file__).parent
cmd = [
    sys.executable,
    str(root / (a.task + "_impl.py")),
    "--mode",
    a.mode,
    "--output",
    a.output,
]
if "yolo11" != "yolo11":
    cmd += ["--model", "yolo11"]
if a.task == "train":
    os.environ["TRAIN_CHECKPOINT"] = a.checkpoint
    cmd += [
        "--epochs",
        a.epochs,
        "--data",
        str(Path(os.environ["DATA_DIR"]) / "data.npz"),
    ]
    if os.environ.get("RESUME"):
        cmd += ["--resume", os.environ["RESUME"]]
else:
    cmd += ["--checkpoint", a.checkpoint]
subprocess.run(cmd, check=True)
