# Spec-Decode: Qwen LoRA Fine-Tuning + Evaluation on GAIA

This repo contains scripts to fine-tune Qwen (e.g., `Qwen/Qwen3-4B`) via LoRA on GAIA-style tool-call traces, then evaluate the finetuned adapter against the base model. It also includes lightweight dataset utilities for GAIA splits.

## Repository Structure

- `qwen_train.py`: LoRA fine-tuning script for Qwen models using JSONL chat traces.
- `qwen_test.py`: Evaluates a LoRA adapter on GAIA test data; uses Gemini as a text-only judge for tool-call matching.
- `base_qwen_test.py`: Head-to-head comparison of Base vs LoRA outputs with Gemini-based scoring.
- `tool_registry.py`: Tool names catalog referenced by prompts/evaluation.
- `dataset/`: GAIA dataset utilities and split files.
  - `gaia_train.jsonl`, `gaia_val.jsonl`, `gaia_test.jsonl`: JSONL chat traces for training/validation/test.
  - `gaia_split.py`, `gaia_level_count.py`, etc.: Helpers to manipulate GAIA data.
  - `GAIA/2023/...`: Original GAIA data layout (test/validation).
  - `gaia_2023_all_splits/...`: Consolidated GAIA splits.
- `qwen3-gaia-lora/`: Saved LoRA adapter(s) and tokenizer artifacts.
  - `checkpoint-18/`: Example training checkpoint with adapter weights.
- Result artifacts: `results.jsonl`, `comparison_results*.jsonl`, `out*.txt`, etc.

## Data Format

Training/validation/test files are JSONL where each line is an object containing messages:

```json
{
  "id": "optional",
  "messages": [
    {"role": "user", "content": "Question or instruction"},
    {"role": "assistant", "content": "Reasoning trace and tool usage"},
    ...
  ]
}
```

`qwen_train.py` constructs sequences by alternating `<|user|>` and `<|assistant|>` prefixes, masking user tokens from loss and using the tokenizer's EOS for padding.

## Environment Setup

This project uses Hugging Face Transformers, PEFT, Datasets, PyTorch, TQDM, and Google Generative AI (Gemini) for evaluation. The provided `requirements.txt` is empty; install typical dependencies below.

```bash
# Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121  # or cpu
pip install transformers peft datasets accelerate tqdm google-generativeai
```

Notes:
- Replace the PyTorch index URL to match your CUDA/CPU setup.
- Ensure you have access to the base Qwen model on Hugging Face.
- Set `GOOGLE_API_KEY` when using Gemini-based scoring.

## Training (LoRA)

Fine-tune Qwen on GAIA traces with validation:

```bash
python qwen_train.py \
  --train_file dataset/gaia_train.jsonl \
  --val_file dataset/gaia_val.jsonl \
  --model_name Qwen/Qwen3-4B \
  --output_dir qwen3-gaia-lora \
  --max_length 2048 \
  --learning_rate 2e-4 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 3 \
  --eval_steps 200 \
  --save_steps 200 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05
```

Key implementation details:
- Uses tokenizer EOS as `pad_token` to avoid vocabulary mismatch.
- No `resize_token_embeddings` call; keeps base vocab unchanged.
- Left padding for stable generation.

Artifacts are saved under `--output_dir` (e.g., `qwen3-gaia-lora/`).

## Evaluation (LoRA only)

Evaluate adapter outputs on GAIA test data, scoring tool-call alignment via Gemini:

```bash
export GEMINI_API_KEY="<your-key>"
python qwen_test.py \
  --base_model Qwen/Qwen3-4B \
  --lora_dir qwen3-gaia-lora \
  --test_file dataset/gaia_test.jsonl \
  --output_file results.jsonl \
  --gemini_api_key "$GEMINI_API_KEY" \
  --max_new_tokens 1024
```

Outputs include per-example accuracy and latency plus raw strings (`gold_raw`, `pred_raw`).

## Comparative Evaluation (Base vs LoRA)

Run a head-to-head comparison and summary stats:

```bash
export GEMINI_API_KEY="<your-key>"
python base_qwen_test.py \
  --base_model Qwen/Qwen3-4B \
  --lora_dir qwen3-gaia-lora \
  --test_file dataset/gaia_test.jsonl \
  --output_file comparison_results1.jsonl \
  --gemini_api_key "$GEMINI_API_KEY" \
  --max_new_tokens 2048
```

The script:
- Generates traces with base (adapter disabled) and LoRA (adapter enabled).
- Scores each vs gold and asks Gemini to pick a winner.
- Prints aggregate accuracy and win/tie counts.

## Dataset Utilities

Under `dataset/`, you’ll find helpers to:
- Download/prepare GAIA subsets (`gaia_download.py`, `GAIA/2023/...`).
- Split and count levels (`gaia_split.py`, `gaia_level_count.py`).
- Transform model traces (`gemini_traces_to_train_traces.py`).

Review each script for its CLI and expected file layout.

## Tips & Troubleshooting

- If you hit padding/vocab errors, ensure the tokenizer’s `pad_token` is set to its `eos_token` and avoid adding new special tokens.
- Use `device_map=auto` to distribute across available GPUs; adjust `per_device_train_batch_size` and `gradient_accumulation_steps` to fit memory.
- For CPU-only environments, reduce `max_length` and `max_new_tokens` to keep generation feasible.
- Ensure your GAIA JSONL messages follow the expected `role`/`content` schema.

## License

No explicit license file is provided in this repo. Please consult dataset licenses (GAIA) and the Hugging Face model licenses before use.

## Acknowledgments

- Qwen models by Alibaba Cloud (via Hugging Face).
- PEFT/LoRA by Hugging Face ecosystem contributors.
- GAIA dataset authors.
- Google Generative AI (Gemini) used for automated judging.
