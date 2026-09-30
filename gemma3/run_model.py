#!/usr/bin/env python3
"""Gemma3 checkpoint-backed multimodal training/inference test runner."""

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dynamic", "cinn"), required=True)
    parser.add_argument("--task", choices=("train", "infer"), default="train")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.0)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


ARGS = parse_args()
USE_CINN = ARGS.mode == "cinn"
os.environ["FLAGS_prim_enable_dynamic"] = "true" if USE_CINN else "false"
os.environ["FLAGS_prim_all"] = "true" if USE_CINN else "false"
os.environ["FLAGS_use_cinn"] = "true" if USE_CINN else "false"

import numpy as np
import paddle

from paddleformers.transformers import Gemma3ForConditionalGeneration


CHECKPOINT_FILES = (
    "added_tokens.json",
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "model_state.pdparams",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)

CINN_SCOPE = "full train/infer wrapper"
DYNAMIC_SCOPE = "checkpoint/data orchestration and optimizer step"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: Path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def synchronize():
    paddle.device.synchronize()


def validate_checkpoint(checkpoint: Path):
    manifest_path = checkpoint / "checkpoint_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"checkpoint manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("architecture") != "Gemma3ForConditionalGeneration":
        raise RuntimeError(f"unexpected checkpoint architecture: {manifest.get('architecture')}")
    if manifest.get("dtype") != "bfloat16":
        raise RuntimeError(f"unexpected checkpoint dtype: {manifest.get('dtype')}")
    if manifest.get("text_layers") != 4 or manifest.get("vision_layers") != 2:
        raise RuntimeError(
            f"unexpected checkpoint layers: text={manifest.get('text_layers')} vision={manifest.get('vision_layers')}"
        )
    entries = {entry["file"]: entry for entry in manifest["files"]}
    missing = sorted(set(CHECKPOINT_FILES) - set(entries))
    if missing:
        raise RuntimeError(f"checkpoint manifest entries missing: {missing}")
    for name, entry in entries.items():
        path = checkpoint / name
        if not path.is_file():
            raise RuntimeError(f"checkpoint file missing: {path}")
        if path.stat().st_size != entry["bytes"]:
            raise RuntimeError(f"checkpoint size mismatch: {path}")
        if sha256(path) != entry["sha256"]:
            raise RuntimeError(f"checkpoint sha256 mismatch: {path}")
    return manifest


def validate_data(data_dir: Path):
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"data manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = sorted(data_dir.glob("batch-*.npz"))
    records = manifest.get("records", [])
    if not files or len(files) != len(records):
        raise RuntimeError(f"data batch count mismatch: files={len(files)} manifest={len(records)}")
    by_name = {record["batch"]: record for record in records}
    for path in files:
        expected = by_name.get(path.name)
        if expected is None or sha256(path) != expected["sha256"]:
            raise RuntimeError(f"data sha256 mismatch: {path}")
    return manifest, files


def load_batch(path: Path, include_labels: bool):
    keys = ["input_ids", "pixel_values", "attention_mask", "position_ids", "token_type_ids"]
    if include_labels:
        keys.append("labels")
    with np.load(path) as data:
        return tuple(paddle.to_tensor(data[key]) for key in keys)


class Gemma3TrainGraph(paddle.nn.Layer):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, pixel_values, attention_mask, position_ids, token_type_ids, labels):
        outputs = self.model.model(
            input_ids,
            pixel_values,
            attention_mask,
            None,
            position_ids,
            None,
            token_type_ids,
            None,
            None,
            False,
            False,
            False,
            False,
        )
        logits = paddle.matmul(outputs[0], self.model.lm_head.weight, transpose_y=True)
        logits = logits.astype("float32").reshape([-1, self.model.config.text_config.vocab_size])
        labels = labels.reshape([-1])
        per_token_loss = paddle.nn.functional.cross_entropy(
            logits, labels, reduction="none", ignore_index=-100
        )
        mask = (labels != -100).astype(per_token_loss.dtype)
        return (per_token_loss * mask).sum() / paddle.clip(mask.sum(), min=1.0)


class Gemma3InferGraph(paddle.nn.Layer):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, pixel_values, attention_mask, position_ids, token_type_ids):
        outputs = self.model.model(
            input_ids,
            pixel_values,
            attention_mask,
            None,
            position_ids,
            None,
            token_type_ids,
            None,
            None,
            False,
            False,
            False,
            False,
        )
        return paddle.matmul(outputs[0], self.model.lm_head.weight, transpose_y=True)


def environment():
    paddleformers_root = Path(__file__).resolve().parents[2] / "PaddleFormers"
    pythonpath_root = os.environ.get("PYTHONPATH", "").split(os.pathsep)[0]
    if pythonpath_root:
        paddleformers_root = Path(pythonpath_root)
    return {
        "python": platform.python_version(),
        "paddle": paddle.__version__,
        "paddle_commit": paddle.version.commit,
        "cuda": paddle.version.cuda(),
        "cudnn_build": paddle.version.cudnn(),
        "device": paddle.device.get_device(),
        "paddleformers_commit": git_commit(paddleformers_root),
    }


def run_train(model, files):
    model.train()
    graph = Gemma3TrainGraph(model)
    if USE_CINN:
        graph = paddle.jit.to_static(graph, full_graph=True)
    optimizer = None
    if ARGS.learning_rate > 0:
        optimizer = paddle.optimizer.AdamW(
            learning_rate=ARGS.learning_rate,
            parameters=graph.parameters(),
            weight_decay=0.0,
            multi_precision=True,
        )
    batches = [load_batch(path, include_labels=True) for path in files]
    records = []
    for step in range(ARGS.steps):
        batch = batches[step % len(batches)]
        synchronize()
        start = time.perf_counter()
        loss = graph(*batch)
        loss.backward()
        if optimizer is None:
            graph.clear_gradients()
        else:
            optimizer.step()
            optimizer.clear_grad()
        synchronize()
        elapsed = time.perf_counter() - start
        loss_value = float(loss.astype("float32"))
        if not math.isfinite(loss_value):
            raise RuntimeError(f"non-finite loss at step {step}: {loss_value}")
        record = {"step": step, "loss": loss_value, "step_time_s": elapsed}
        records.append(record)
        print("STEP_LOSS " + json.dumps(record, sort_keys=True), flush=True)
    steady = records[ARGS.warmup_steps :]
    return {
        "optimizer": "AdamW" if optimizer else "no-update/clear-grad",
        "learning_rate": ARGS.learning_rate,
        "mean_steady_step_time_s": statistics.fmean(r["step_time_s"] for r in steady),
        "steady_fps": 1.0 / statistics.fmean(r["step_time_s"] for r in steady),
        "first_step_time_s": records[0]["step_time_s"],
        "losses": [record["loss"] for record in records],
        "records": records,
    }


def run_infer(model, files):
    model.eval()
    graph = Gemma3InferGraph(model)
    if USE_CINN:
        graph = paddle.jit.to_static(graph, full_graph=True)
    batches = [load_batch(path, include_labels=False) for path in files]
    records = []
    with paddle.no_grad():
        for step in range(ARGS.steps):
            batch = batches[step % len(batches)]
            synchronize()
            start = time.perf_counter()
            logits = graph(*batch)
            synchronize()
            elapsed = time.perf_counter() - start
            check = logits[:, -1, :128].astype("float32")
            checksum = float(check.sum())
            checksum_abs = float(check.abs().sum())
            if not math.isfinite(checksum) or not math.isfinite(checksum_abs):
                raise RuntimeError(f"non-finite logits at step {step}")
            record = {
                "step": step,
                "latency_ms": elapsed * 1000.0,
                "logits_checksum": checksum,
                "logits_abs_checksum": checksum_abs,
            }
            records.append(record)
            print("INFER_STEP " + json.dumps(record, sort_keys=True), flush=True)
    steady = records[ARGS.warmup_steps :]
    mean_ms = statistics.fmean(record["latency_ms"] for record in steady)
    return {
        "scope": "multimodal teacher-forced forward logits; preprocessing and H2D excluded",
        "mean_steady_latency_ms": mean_ms,
        "steady_fps": 1000.0 / mean_ms,
        "first_step_latency_ms": records[0]["latency_ms"],
        "records": records,
    }


def main():
    if not 0 <= ARGS.warmup_steps < ARGS.steps:
        raise ValueError("require 0 <= warmup_steps < steps")
    checkpoint = ARGS.checkpoint.resolve()
    data_dir = ARGS.data.resolve()
    output = ARGS.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_manifest = validate_checkpoint(checkpoint)
    data_manifest, files = validate_data(data_dir)
    paddle.set_device(ARGS.device)
    paddle.seed(ARGS.seed)
    np.random.seed(ARGS.seed)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        str(checkpoint), dtype=checkpoint_manifest["dtype"]
    )
    result = run_train(model, files) if ARGS.task == "train" else run_infer(model, files)
    result.update(
        {
            "model": "gemma3",
            "architecture": checkpoint_manifest["architecture"],
            "checkpoint": str(checkpoint),
            "checkpoint_weight_sha256": next(
                entry["sha256"]
                for entry in checkpoint_manifest["files"]
                if entry["file"] == "model_state.pdparams"
            ),
            "checkpoint_dtype": checkpoint_manifest["dtype"],
            "compute_dtype": "bfloat16",
            "loss_dtype": "float32" if ARGS.task == "train" else None,
            "text_layers": checkpoint_manifest["text_layers"],
            "vision_layers": checkpoint_manifest["vision_layers"],
            "data": str(data_dir),
            "data_input_sha256": data_manifest.get("input_sha256"),
            "data_batches": len(files),
            "mode": ARGS.mode,
            "task": ARGS.task,
            "batch_size": 1,
            "sequence_length": data_manifest["max_length"],
            "steps": ARGS.steps,
            "warmup_steps": ARGS.warmup_steps,
            "first_step_includes_lazy_trace_and_compile": USE_CINN,
            "cinn_scope": CINN_SCOPE if USE_CINN else "none",
            "static_subgraph_count": 1 if USE_CINN else 0,
            "dynamic_scope": DYNAMIC_SCOPE if USE_CINN else "entire model",
            "outer_graph_mode": "static" if USE_CINN else "dynamic",
            "environment": environment(),
        }
    )
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "RUN_SUMMARY "
        + json.dumps(
            {key: value for key, value in result.items() if key not in {"records", "losses"}},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
