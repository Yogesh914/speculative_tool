# Speculative Tool Calling

Reduce LLM agent latency by speculatively pre-executing tool calls in parallel with the actor model's reasoning.

## Core Idea

**Traditional Sequential Execution:**
```
Time: 0s                    10s                    12s
      ├─────────────────────┼──────────────────────┤
      Actor thinking...     Execute tool
                            ▓▓
      Total: 12s
```

**Speculative Parallel Execution:**
```
Time: 0s            2s      4s                     10s    10.001s
      ├─────────────┼───────┼──────────────────────┼──────┤
      Actor:        │                              │
                    └──────────────────────────────┘
                    Thinking...

      Spec:   Predict Execute
              ▓▓      ▓▓
                      └─→ Cache result

                                                   Check cache
                                                   ⚡ Match! Use cached result
                                                   (Tool execution skipped)

      Total: ~10s (2s saved, 17% speedup)
```

**Key insight**: While the actor model (GPT-5) is thinking, a lightweight spec model (GPT-5-mini) predicts and pre-executes the next tool call. If the prediction matches, we use the cached result and skip tool execution entirely.

## Architecture

```
                    ┌─────────────────────┐
                    │      START          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     node_llm        │
                    │                     │
                    │  ┌───────────────┐  │
                    │  │ Actor Model   │  │  Run in parallel
                    │  │ (GPT-5)       │  │  
                    │  └───────────────┘  │
                    │         +           │
                    │  ┌───────────────┐  │
                    │  │ Spec Model    │  │
                    │  │ (GPT-5-mini)  │  │
                    │  │ + Pre-execute │  │
                    │  │ + Cache       │  │
                    │  └───────────────┘  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
              ┌────→│    node_tools       │
              │     │                     │
              │     │  Check cache:       │
              │     │  - Match? Use cache │
              │     │  - No? Execute      │
              │     └──────────┬──────────┘
              │                │
              │                ▼
              │     ┌─────────────────────┐
              │     │   should_end?       │
              │     └──────────┬──────────┘
              │                │
              │         ┌──────┴──────┐
              │         │             │
              │       Done       Continue
              │         │             │
              │         ▼             │
              │     ┌───────┐         │
              │     │  END  │         │
              │     └───────┘         │
              │                       │
              └───────────────────────┘
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download GAIA dataset
python3 download_gaia.py
```

### Option 1: Using OpenAI Models

```bash
# Set API keys - use separate keys to avoid rate limiting when running in parallel
export ACTOR_OPENAI_API_KEY="your-openai-key-1"
export SPEC_OPENAI_API_KEY="your-openai-key-2"
export SERPER_API_KEY="your-serper-key"  # For web search

# Note: If you only have one API key, you can use it for both:
# export ACTOR_OPENAI_API_KEY="your-openai-key"
# export SPEC_OPENAI_API_KEY="your-openai-key"
# However, this may cause request throttling when actor and spec run in parallel

# Configure models (optional, defaults shown)
export MODEL_PROVIDER="openai"
export GAIA_ACTOR_MODEL="gpt-5"
export GAIA_SPEC_MODEL="gpt-5-mini"
```

### Option 2: Using Open-Source Models via vLLM

```bash
# Start vLLM server with tool calling support
CUDA_VISIBLE_DEVICES=7 vllm serve /path/to/your/model \
    --port 8003 \
    --host 0.0.0.0 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes

CUDA_VISIBLE_DEVICES=6 vllm serve /home/nvidia/data/models/Qwen2.5-3B-Instruct \
    --port 8004 \
    --host 0.0.0.0 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes

# Configure to use vLLM
export MODEL_PROVIDER="vllm"
export VLLM_BASE_URL="http://localhost:8003/v1"
export GAIA_ACTOR_MODEL="/path/to/your/model"
export GAIA_SPEC_MODEL="/path/to/your/model"
export SERPER_API_KEY="your-serper-key"  # For web search
```

**Supported vLLM models:**
- Qwen/Qwen2.5-* (recommended, uses `--tool-call-parser hermes`)
- Qwen/QwQ-32B
- meta-llama/Llama-3.1-* (use `--tool-call-parser llama3_json`)
- See [vLLM Tool Calling docs](https://docs.vllm.ai/en/stable/features/tool_calling/) for more

## Running Evaluations

### Single Example (Detailed)

```bash
# Run specific example
python eval.py gaia_dataset/level1/example_000

# With speculation enabled (default)
python eval.py gaia_dataset/level1/example_000

# Baseline (no speculation)
DISABLE_SPECULATION=1 python eval.py gaia_dataset/level1/example_000
```

### Batch Evaluation (Multiple Examples)

```bash
# Run all level 1 examples with speculation
python eval.py --batch --level 1

# Run first 10 examples
python eval.py --batch --level 1 --max 10

# Baseline (no speculation)
DISABLE_SPECULATION=1 python eval.py --batch --level 1 --max 10

# inlcusive start, exclusive end
python eval.py --batch --level 1 --start 10 --end 21

# Custom output directory
python eval.py --batch --level 1 --output my_results/
```

## Configuration

All configuration is via environment variables (see `env.example` for full list):

```bash
# Model Provider
export MODEL_PROVIDER="openai"            # "openai" or "vllm"

# OpenAI API Keys (use separate keys to avoid rate limiting)
export ACTOR_OPENAI_API_KEY="your-key-1"  # For actor model (GPT-5)
export SPEC_OPENAI_API_KEY="your-key-2"   # For spec model (GPT-5-mini)

# Models (format depends on provider)
export GAIA_ACTOR_MODEL="gpt-5"           # Main reasoning model
export GAIA_SPEC_MODEL="gpt-5-mini"       # Speculation model (smaller/faster)

# vLLM Settings (only if MODEL_PROVIDER=vllm)
export VLLM_ACTOR_URL="http://localhost:8003/v1"  # Actor model endpoint
export VLLM_SPEC_URL="http://localhost:8004/v1"   # Spec model endpoint

# Execution Limits
export GAIA_MAX_STEPS="15"                # Max reasoning steps

# Speculation
export DISABLE_SPECULATION="1"            # Set to "1" to disable (baseline mode)
export VERIFICATION_STRATEGY="exact"      # "exact", "tool_name_only", or "none"
```

## Output

**Single mode** shows detailed step-by-step execution:
```
[Step 1] LLM
⏱️  LLM call: 8.5s

🔧 Tool Call:
   Tool: search_with_content
   Args: query = Moon perigee...

[Step 1] TOOLS
⏱️  Execution: 0.001s

✅ USING CACHED RESULT (pre-executed by spec model)
   Saved 1.25s from cache
```

**Batch mode** shows compact progress:
```
[1/10] example_000
✅ CORRECT (89.2s)

[2/10] example_001
❌ WRONG (102.3s)

...

FINAL RESULTS
Accuracy:  8/10 (80.0%)
Avg Time:  95.4s
```

## Results

Results are saved to JSON files:

```json
{
  "task_id": "e1fc63a2-...",
  "question": "If Eliud Kipchoge could maintain...",
  "ground_truth": "17",
  "predicted": "17",
  "correct": true,
  "steps": 6,
  "time_seconds": 89.2,
  "spec_hits": 5,
  "spec_misses": 0,
  "spec_predictions": 5
}
```