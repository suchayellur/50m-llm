# Hackathon LM: from-scratch <50M decoder-only Transformer

This repository is a beginner-friendly PyTorch/Hugging Face project for training a small language model from random weights.

The project intentionally does **not** initialize with pretrained model weights or distillation. The training script constructs:

```python
model = HackathonLMForCausalLM(config)
```

It does **not** call `AutoModelForCausalLM.from_pretrained(...)` to create the model for training.

## What is included

- Decoder-only GPT-style Transformer in `src/modeling_hackathon_lm.py`
- Random initialization only
- Tied token embedding and LM head
- Final config targeting about 49M trainable parameters
- Tiny debug config for fast smoke tests
- 8,192-token tokenizer training from scratch
- Streaming dataset helpers for TinyStories and FineWeb-Edu
- Procedural cryptography data generator
- Kaggle T4-friendly training loop
- Checkpoint and resume support
- BF16 when supported, FP16 otherwise
- Parameter-count assertion script
- lm-evaluation-harness wrapper for HellaSwag, ARC-Easy, PIQA, WinoGrande, and WikiText
- Training metadata for GPU type, time, tokens, and estimated compute

## Parameter budget

The final config is in `configs/final_49m.json`:

- Vocabulary: 8,192
- Layers: 14
- Hidden size: 512
- Attention heads: 8
- MLP intermediate size: 1,408
- Context length: 512

The LM head is tied to the token embedding, so it does not add a second vocabulary projection matrix.

Always check the budget before training:

```bash
python scripts/count_params.py --config configs/debug.json
python scripts/count_params.py --config configs/final_49m.json
```

`count_params.py` asserts that trainable parameters are `<= 50,000,000`.

## Install

Create a fresh environment, then install:

```bash
pip install -r requirements.txt
```

On Kaggle, use an accelerator notebook with a T4 GPU and install the same requirements.

## Step 1: train a tokenizer from scratch

For the tiny debug config:

```bash
python scripts/train_tokenizer.py --dataset tinystories --vocab-size 512 --limit-texts 1000 --output-dir tokenizer_debug
```

For the final model:

```bash
python scripts/train_tokenizer.py --dataset tinystories --vocab-size 8192 --limit-texts 200000 --output-dir tokenizer
```

To use FineWeb-Edu instead:

```bash
python scripts/train_tokenizer.py --dataset fineweb --vocab-size 8192 --limit-texts 200000 --output-dir tokenizer
```

Datasets are streamed when possible, so the full dataset is not stored locally.

## Optional: create cryptography data

```bash
python data/crypto_generator.py --output crypto_train.txt --examples 10000
```

Then train with:

```bash
python scripts/train.py --config configs/debug.json --tokenizer tokenizer_debug --dataset local --local-files crypto_train.txt --steps 50 --output-dir runs/debug_crypto
```

## Step 2: run a tiny debug training job

```bash
python scripts/train.py \
  --config configs/debug.json \
  --tokenizer tokenizer_debug \
  --dataset tinystories \
  --steps 50 \
  --batch-size 4 \
  --grad-accum 2 \
  --output-dir runs/debug
```

This is the first thing to run on a new machine. It catches tokenizer/config mistakes quickly.

## Step 3: train the final model

Kaggle T4 starter command:

```bash
python scripts/train.py \
  --config configs/final_49m.json \
  --tokenizer tokenizer \
  --dataset tinystories \
  --steps 20000 \
  --batch-size 4 \
  --grad-accum 16 \
  --lr 3e-4 \
  --warmup-steps 500 \
  --save-every 1000 \
  --output-dir runs/final
```

For FineWeb-Edu:

```bash
python scripts/train.py --config configs/final_49m.json --tokenizer tokenizer --dataset fineweb --output-dir runs/final
```

The script prints and saves the model config, parameter count, GPU type, precision mode, elapsed time, tokens seen, and estimated FLOPs.

## Resume training

```bash
python scripts/train.py \
  --config configs/final_49m.json \
  --tokenizer tokenizer \
  --dataset tinystories \
  --steps 20000 \
  --resume-from runs/final/checkpoint-1000 \
  --output-dir runs/final
```

## Evaluation

After training, run:

```bash
bash eval/run_lm_eval.sh runs/final/final eval_results
```

On Windows PowerShell:

```powershell
./eval/run_lm_eval.ps1 -ModelDir runs/final/final -OutputDir eval_results
```

Tasks:

- HellaSwag
- ARC-Easy
- PIQA
- WinoGrande
- WikiText

## Keeping the repo small

Generated tokenizers, checkpoints, result folders, and large tensor files are ignored by `.gitignore`. The repository should stay well under 1 GB unless you intentionally keep trained checkpoints in it.

Before adding anything large, pause and decide whether it belongs in Kaggle output storage, Hugging Face Hub, Git LFS, or local ignored folders.

## GitHub to Kaggle workflow

1. Push this repository to GitHub.
2. Open a Kaggle GPU notebook.
3. Clone the repository.
4. Install `requirements.txt`.
5. Train the tokenizer.
6. Run `count_params.py`.
7. Train from random weights.
8. Resume from checkpoints if needed.
9. Evaluate with lm-evaluation-harness.
10. Save final checkpoint and results as Kaggle outputs.
