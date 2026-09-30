# Checkpoint record

- Architecture: `Gemma3ForConditionalGeneration`
- Variant: high-precision reduced-depth Paddle checkpoint
- Text layers: 4
- Vision layers: 2
- Weight/model compute dtype: BF16
- Loss dtype: FP32
- Weight file: `model_state.pdparams`
- Weight bytes: `3517742411`
- Weight SHA256: `33029a880dc17d16abdac2327e7891bbb1fb7483bfaaf4d962916c5fa8f4f75c`
- Complete per-file checksums: `checkpoint_manifest.json`

The 3.4 GiB checkpoint is intentionally not duplicated in ordinary Git. Set:

```bash
export CHECKPOINT_DIR=/absolute/path/to/gemma3-4b-text4-vision2-bf16
```

Alternatively, create the ignored file `LOCATION.local` containing `path=/absolute/path/...`. Do not edit the tracked `LOCATION.md` or commit machine-specific paths. If the remote supports Git LFS, put checkpoint files under `checkpoint/files/` and remove that path from the root `.gitignore`.
