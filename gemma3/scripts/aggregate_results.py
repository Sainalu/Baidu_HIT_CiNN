#!/usr/bin/env python3
"""Combine one mode's train and inference JSON into a local run artifact."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dynamic", "cinn"), required=True)
    parser.add_argument("--profile", default="quick")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--infer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def main():
    args = parse_args()
    train = read_json(args.train)
    infer = read_json(args.infer)
    if train.get("mode") != args.mode or infer.get("mode") != args.mode:
        raise ValueError("train and infer results must use the requested mode")
    if train.get("model") != infer.get("model"):
        raise ValueError("train and infer results must use the same model")

    result = {
        "model": train.get("model"),
        "mode": args.mode,
        "profile": args.profile,
        "architecture": train.get("architecture"),
        "checkpoint": {
            "path": train.get("checkpoint"),
            "weight_sha256": train.get("checkpoint_weight_sha256"),
            "dtype": train.get("checkpoint_dtype"),
        },
        "precision": {
            "compute_dtype": train.get("compute_dtype"),
            "train_loss_dtype": train.get("loss_dtype"),
        },
        "environment": train.get("environment"),
        "cinn_scope": train.get("cinn_scope"),
        "static_subgraph_count": train.get("static_subgraph_count"),
        "dynamic_scope": train.get("dynamic_scope"),
        "outer_graph_mode": train.get("outer_graph_mode"),
        "train": train,
        "infer": infer,
        "source_results": {
            "train": str(args.train),
            "infer": str(args.infer),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
