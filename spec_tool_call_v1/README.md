# GAIA Agent with Speculative Tool Calling

A powerful ReAct agent system built with **LangGraph** for the GAIA benchmark, featuring **speculative tool calling** for performance optimization, comprehensive file support, and multimodal capabilities.

## 🎯 Key Features

- **🚀 Speculative Tool Calling**: Pre-executes likely tool calls in parallel using a lightweight spec model
  - Automatic cache management for speculation results
  - Significant latency reduction when speculation hits
  - Read-only tool protection (only safe operations are speculated)
- **ReAct Pattern**: Implements Reason → Act → Observe loop for systematic problem-solving
- **8 Specialized Tools**: Web search, file reading (10+ formats), code execution, vision analysis
- **LangGraph Architecture**: Uses LangGraph's native tool calling with `bind_tools()`
- **GPT-5 Support**: Compatible with latest OpenAI models (GPT-5, GPT-4o, GPT-4o-mini)
- **Rich Output**: Detailed execution traces with timing, tool arguments, and speculation metrics
- **Benchmarking Suite**: Measure speculation accuracy and performance gains

## 🏗️ Architecture

### Speculative Tool Calling Flow

```
┌───────────────────────────────────────────────────────────────┐
│              LangGraph Workflow with Speculation              │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐                    ┌──────────────┐        │
│  │   LLM Node   │                    │  Spec Model  │        │
│  │  (Actor)     │                    │ (Predictor)  │        │
│  │              │                    │              │        │
│  │ gpt-5 / 4o   │                    │ gpt-5-mini   │        │
│  └──────┬───────┘                    └──────┬───────┘        │
│         │                                   │                │
│         │ Decides to call tools             │ Predicts      │
│         │                                   │ next tools    │
│         ▼                                   ▼                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Speculation Cache                      │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │    │
│  │  │ search_  │  │  file_   │  │ vision_  │  ...     │    │
│  │  │  web()   │  │  read()  │  │ analyze()│          │    │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘          │    │
│  │       │ Running     │ Running     │ Complete       │    │
│  │       │ async...    │ async...    │ [cached]       │    │
│  │       │             │             │                │    │
│  └───────┼─────────────┼─────────────┼────────────────┘    │
│          │             │             │                     │
│          ▼             ▼             ▼                     │
│  ┌──────────────────────────────────────────────────┐     │
│  │           Tools Node (Execute)                   │     │
│  │                                                  │     │
│  │  1. Check cache for matching tool call           │     │
│  │  2. If HIT: Use cached result (fast! ⚡)         │     │
│  │  3. If MISS: Execute tool normally               │     │
│  │  4. Launch new speculations for next step        │     │
│  │                                                  │     │
│  └──────────────────┬───────────────────────────────┘     │
│                     │                                     │
│                     │ Return results + metrics            │
│                     ▼                                     │
│            Back to LLM Node                               │
│                                                           │
│  Metrics: hits=3, misses=1, launched=5, hit_rate=75%     │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

### Key Benefits

- **⚡ Reduced Latency**: When speculation hits, tool results are ready immediately
- **🔄 Parallel Execution**: Spec model predicts while actor executes
- **🛡️ Safety**: Only read-only tools are speculated (search, read, analyze)
- **📊 Measurable**: Track hit rate, cache performance, and time savings

## 🛠️ Available Tools (8 Total)

### 🔍 Web Search & Browsing
- **`search_web`** - Quick web search (3 results default, 10 with `expand_search=True`)
  - Uses Google Serper API
  - Returns titles, snippets, and URLs
- **`search_with_content`** - Deep search with full content extraction (1 result default, 3 with `expand_search=True`)
  - Fetches and parses actual webpage content
  - More thorough but slower

### 📄 File Reading (10+ formats)
- **`file_read`** - Intelligent file reader with auto-detection:
  - **Text**: `.txt`, `.md`, `.log`
  - **Data**: `.csv`, `.json`, `.jsonl`, `.yaml`, `.yml`
  - **Spreadsheets**: `.xlsx`, `.xls` (with pandas)
  - **Documents**: `.pdf` (with pdfplumber), `.docx`
  - **Markup**: `.xml`, `.html`

### 💻 Code & Computation
- **`calculate`** - Evaluate mathematical expressions
  - Supports Python syntax: `2**10`, `math.sqrt(16)`
  - Safe execution environment
- **`code_exec`** - Execute Python code in sandbox
  - Timeout protection
  - Isolated execution
- **`code_generate`** - Generate Python code using GPT-4o
  - Takes task description and optional context
  - Returns production-ready code

### 👁️ Vision & Multimodality
- **`vision_analyze`** - Analyze images with vision model
  - Can answer specific questions about images
  - Supports common image formats
- **`vision_ocr`** - Extract text from images
  - OCR functionality
  - Useful for diagrams, screenshots, documents

## 🧠 How Speculative Tool Calling Works

### The Problem
Traditional LLM agents execute tools sequentially:
1. LLM thinks and decides which tool to call → **Wait for LLM (slow)**
2. Execute tool → **Wait for tool execution**
3. LLM processes result and decides next step → **Wait for LLM again**

This creates significant latency, especially with powerful but slower models like GPT-5.

### The Solution
**Speculative tool calling** uses a lightweight model to predict and pre-execute tools:

1. **Actor Model** (GPT-5) decides to call `search_web("Eliud Kipchoge marathon")`
2. While actor is thinking, **Spec Model** (GPT-5-mini) predicts likely next tools
3. Spec model launches: `search_web("marathon record")`, `calculate("...")`, etc.
4. When actor decides next step, result may already be cached! ⚡

### Cache Management
- **Normalized Keys**: Tool calls are normalized (e.g., whitespace, case) for better hit rates
- **Read-Only Safety**: Only safe, read-only tools are speculated automatically
- **Async Execution**: All speculations run in parallel without blocking
- **Hit/Miss Tracking**: Detailed metrics show speculation effectiveness

### Example Impact
```
Without Speculation:  Step 1 (2.5s) → Step 2 (2.3s) → Step 3 (2.1s) = 6.9s total
With Speculation:     Step 1 (2.5s) → Step 2 (0.1s) → Step 3 (0.1s) = 2.7s total
                      ⚡ 61% faster!
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create a `.env` file:

```bash
# Required
OPENAI_API_KEY=your_openai_key
SERPER_API_KEY=your_serper_key  # For web search

# Model Configuration
GAIA_ACTOR_MODEL=gpt-5           # Main reasoning model (gpt-5, gpt-4o, gpt-4o-mini)
GAIA_SPEC_MODEL=gpt-5-mini       # Speculation model (should be faster than actor)
GAIA_MAX_STEPS=12                # Max reasoning steps

# Speculation Settings (Advanced)
GAIA_TOPK=3                      # Number of tool calls to speculate
GAIA_CONF_TH=0.35                # Confidence threshold for speculation
DISABLE_SPECULATION=0            # Set to 1 to disable speculation
```

### 3. Download GAIA Dataset

```bash
python download_gaia.py
```

This creates:
```
gaia_dataset/
├── level1/  # Easy questions (165 examples)
│   ├── example_000/
│   │   ├── metadata.json
│   │   └── [attached files]
│   └── ...
├── level2/  # Medium questions
└── level3/  # Hard questions
```

### 4. Run Examples

**Run a specific example:**
```bash
python run_gaia_example.py gaia_dataset/level1/example_000
```

**Run with different models:**
```bash
# With GPT-5
GAIA_ACTOR_MODEL=gpt-5 python run_gaia_example.py gaia_dataset/level3/example_000

# With GPT-4o
GAIA_ACTOR_MODEL=gpt-4o python run_gaia_example.py gaia_dataset/level2/example_005

# With GPT-4o-mini (faster, cheaper)
GAIA_ACTOR_MODEL=gpt-4o-mini python run_gaia_example.py
```

**Run full evaluation:**
```bash
# Evaluate level 1
python main.py --mode eval --level 1

# Evaluate first 5 examples from all levels
python main.py --mode eval --max-examples 5

# Evaluate specific level with limit
python main.py --mode eval --level 2 --max-examples 10
```

## 📊 Example Output

```
================================================================================
GAIA Example: e1fc63a2-da7a-432f-be78-7c4a95598703
Level: 1
================================================================================

Question:
If Eliud Kipchoge could maintain his record-making marathon pace indefinitely...

Ground Truth Answer: 17

================================================================================
Running Actor Model: gpt-5 | Spec Model: gpt-5-mini
Speculation: ENABLED
================================================================================

================================================================================
[Step 1] LLM
================================================================================
⏱️  LLM call: 10.23s
🔧 Decision: Call tool
   Tool: search_web
   Args:
      query = Eliud Kipchoge marathon record pace

================================================================================
[Step 2] TOOLS
================================================================================
⏱️  Execution: 1.52s
📤 Output from 'search_web':
   Found 3 relevant web results:
   1. Eliud Kipchoge - Wikipedia...
   
🚀 Launched 3 speculative tool calls for next step

================================================================================
[Step 3] LLM
================================================================================
⏱️  LLM call: 9.09s
🔧 Decision: Call tool
   Tool: calculate
   Args:
      expression = (356400 * 1000) / (42195 / (2 * 3600 + 1 * 60 + 39))

================================================================================
[Step 4] TOOLS
================================================================================
⚡ CACHE HIT! Using speculated result
⏱️  Execution: 0.05s (saved ~1.5s)
📤 Output from 'calculate':
   Result: 17000

🚀 Launched 2 speculative tool calls for next step

...

================================================================================
✅ EXECUTION COMPLETE
================================================================================

Predicted Answer: 17000
Ground Truth:     17

📊 Speculation Metrics:
   Total Steps:          5
   Cache Hits:           2  (⚡ 40% hit rate)
   Cache Misses:         3
   Speculations Launched: 8
   Time Saved:           ~3.2s

Total Messages: 6
Total Time: 24.5s (vs ~28s without speculation)
```

## 📊 Benchmarking Speculation

### Measure Speculation Accuracy

The benchmark suite measures how well the spec model predicts the actor's tool choices:

```bash
# Benchmark Level 1 (all examples)
python benchmark_speculation.py --level 1

# Benchmark first 10 examples from Level 2
python benchmark_speculation.py --level 2 --max-examples 10

# Benchmark Level 3
python benchmark_speculation.py --level 3
```

### Benchmark Output

```
================================================================================
BENCHMARKING SPECULATION - LEVEL 1
================================================================================
Actor Model:  gpt-5
Spec Model:   gpt-5-mini
================================================================================

Benchmarking: e1fc63a2-da7a-432f-be78-7c4a95598703
Question: If Eliud Kipchoge could maintain his record-making...

[1/2] Getting actor predictions...
  Actor: 1 tool(s) in 8.23s
    - search_web: ['query']

[2/2] Getting speculator predictions...
  Spec:  1 tool(s) in 2.15s
    - search_web: ['query']

  Tool match: ✓
  Args similarity: 85.7%
  Result: ≈ PARTIAL MATCH (tool correct, args 86%)

================================================================================
BENCHMARK SUMMARY
================================================================================
Total Examples:      165
Exact Matches:       98 (59.4%)
Tool Name Matches:   132 (80.0%)
Avg Args Similarity: 72.3%

Results saved to: benchmark_speculation_level1_20251106_172117.json
```

### Analyze Benchmark Results

```bash
# Analyze saved benchmark results
python analyze_benchmark.py benchmark_speculation_level1_20251106_172117.json
```

This shows:
- Tool prediction accuracy
- Argument similarity distribution
- Which tools are hardest to predict
- Potential cache hit rates in real usage

## 📁 Project Structure

```
spec_tool_call/
├── spec_tool_call/                # Main package
│   ├── tools/                     # Tool implementations
│   │   ├── search_tool.py         # Web search (Serper API)
│   │   ├── file_tool.py           # Enhanced file reading
│   │   ├── code_exec_tool.py      # Code execution & generation
│   │   └── vision_tool.py         # Image analysis & OCR
│   ├── config.py                  # Configuration from env vars
│   ├── models.py                  # Pydantic models (Msg, RunState)
│   ├── speculation.py             # Speculation cache & logic
│   ├── tool_registry.py           # Tool registry with normalizers
│   ├── tools_langchain.py         # LangChain tool definitions
│   ├── llm_adapter.py             # LLM initialization & prompts
│   ├── graph.py                   # LangGraph workflow (ReAct loop)
│   └── gaia_eval.py               # GAIA dataset & evaluation
├── main.py                        # Main CLI for batch evaluation
├── run_gaia_example.py            # Run single example with rich output
├── benchmark_speculation.py       # Benchmark speculation accuracy
├── analyze_benchmark.py           # Analyze benchmark results
├── download_gaia.py               # Download GAIA dataset
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## 🔧 Configuration Options

Environment variables (set in `.env` or command line):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | OpenAI API key |
| `SERPER_API_KEY` | *(required)* | Google Serper API key for search |
| `GAIA_ACTOR_MODEL` | `gpt-5` | Main reasoning model (gpt-5, gpt-4o, gpt-4o-mini) |
| `GAIA_SPEC_MODEL` | `gpt-5-mini` | Speculation model for predicting next tools |
| `GAIA_MAX_STEPS` | `12` | Maximum reasoning steps before stopping |
| `GAIA_TOPK` | `3` | Number of tool calls to speculate per step |
| `GAIA_CONF_TH` | `0.35` | Confidence threshold for speculation quality |
| `DISABLE_SPECULATION` | `0` | Set to `1` to disable speculation entirely |
