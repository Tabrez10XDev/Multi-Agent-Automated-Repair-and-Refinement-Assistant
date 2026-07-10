<p align= "center">
<img width="300" height="300" src="maara-logo.png" />
</p>
<p align="center">
    <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+">
    <img src="https://img.shields.io/badge/Ollama-Local%20LLMs-000000?style=for-the-badge&logo=ollama&logoColor=white" alt="Ollama">
    <img src="https://img.shields.io/badge/ChromaDB-Vector%20Store-F97316?style=for-the-badge" alt="ChromaDB">
    <img src="https://img.shields.io/badge/Pytest-Testing-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white" alt="Pytest">
    <img src="https://img.shields.io/badge/RAG-Semantic%20Retrieval-10B981?style=for-the-badge" alt="RAG Semantic Retrieval">
    <img src="https://img.shields.io/badge/Multi--Agent-Orchestration-8B5CF6?style=for-the-badge" alt="Multi-Agent Orchestration">
</p>

# MAARA — Multi-Agent Automated Repair and Refinement Assistant

> A locally deployable multi-agent SDE assistant that autonomously plans, validates, generates, and repairs software artifacts using locally hosted LLMs via Ollama.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Pulling Models](#pulling-models)
- [Configuration](#configuration)
- [Usage](#usage)
- [Running Experiments](#running-experiments)
- [Analyzing Results](#analyzing-results)
- [Project Structure](#project-structure)
- [Known Issues](#known-issues)
- [Authors](#authors)

---

## Overview

MAARA is a five-agent pipeline for automated program repair. Given a natural language issue description and a repository, it:

1. Parses the issue into a structured plan
2. Evaluates the plan before any code is written (Sentinel gate)
3. Retrieves relevant code context from a local vector database (RAG)
4. Generates a fix using a code-specialized LLM
5. Runs tests and iteratively repairs failures

All inference runs locally via Ollama — no cloud API calls, no data sent externally.

---

## Architecture

```
Issue Text
    │
    ▼
┌─────────────┐
│Issue Parser │  → structured plan (intent, affected files, test hints)
└─────────────┘
    │
    ▼
┌─────────────┐
│  Navigator  │  → semantic retrieval from ChromaDB vector store
└─────────────┘
    │
    ▼
┌─────────────┐
│  Sentinel   │  → strategy checking (requirement traceability, consistency, feasibility)
│  Evaluator  │  → rejects bad plans and triggers replanning
└─────────────┘
    │
    ▼
┌─────────────┐     ┌─────────────┐
│    Coder    │ ──► │ Test Runner │
└─────────────┘     └─────────────┘
                          │
                    pass ─┤─ fail
                          │
                    ┌─────────────┐
                    │  Debugger   │  → diagnoses failure, builds repair context
                    └─────────────┘
                          │
                          ▼
                    ┌─────────────┐
                    │   Repair    │  → RAG-grounded patch generation
                    │   (Coder)   │  → iterates up to max_attempts
                    └─────────────┘
```

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- 8GB+ RAM (16GB recommended for larger models)
- MacOS, Linux, or Windows with WSL2

### Python Dependencies

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` yet, install manually:

```bash
pip install ollama chromadb pytest
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Tabrez10XDev/self-healing-agent.git
cd self-healing-agent

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install ollama chromadb pytest

# 4. Make sure Ollama is running
ollama serve
```

---

## Pulling Models

MAARA uses two models by default. Pull them before running anything:

```bash
# Coder and single-agent baseline (4.7GB)
ollama pull qwen2.5-coder:7b

# Evaluator / Planner / Homo config (2.0GB)
ollama pull llama3.2

# Embedding model for RAG / ChromaDB (274MB)
ollama pull nomic-embed-text
```

### Optional larger models (requires 16GB+ RAM)

```bash
# Stronger coder — better repair performance
ollama pull deepseek-coder:33b

# Stronger general reasoning
ollama pull llama3:70b

# Code-specialized alternative
ollama pull codellama:34b
```

### Verify models are available

```bash
ollama list
```

Expected output:
```
NAME                       SIZE
qwen2.5-coder:7b           4.7 GB
llama3.2:latest            2.0 GB
nomic-embed-text:latest    274 MB
```

---

## Configuration

All configuration is controlled through the `run()` function signature in `orchestrator.py` and the `CONFIGURATIONS` list in `benchmark/run_experiments.py`.

### Agent model assignments

Edit these in `orchestrator.py` and `agents/evaluator.py`:

| Agent | Default Model | Purpose |
|---|---|---|
| Issue Parser | `llama3.2` | Parse natural language issue |
| Evaluator (Sentinel) | `llama3.2` | Strategy checking |
| Coder | `qwen2.5-coder:7b` | Code generation |
| Debugger | `qwen2.5-coder:7b` | Failure diagnosis |
| Repair | `qwen2.5-coder:7b` | Iterative patching |

To change the evaluator model, edit `agents/evaluator.py`:

```python
response = ollama.chat(
    model="llama3.2",   # change this to any pulled model
    ...
)
```

### Key parameters

| Parameter | Default | Description |
|---|---|---|
| `max_attempts` | 3 | Max repair iterations per task |
| `MAX_REPLAN` | 1 | Max Sentinel replanning cycles |
| `top_k` | 5 | RAG chunks retrieved per query |
| `config` | `"hetero"` | Pipeline configuration |
| `prompting_strategy` | `"zero_shot"` | Prompting strategy |

### Pipeline configurations

| Config | Description |
|---|---|
| `single_agent` | No orchestration, direct Ollama call |
| `homo_llama3` | All 5 agents use `llama3.2` |
| `hetero` | Best-fit model per agent role |
| `hetero_no_sentinel` | Hetero without the Sentinel gate |
| `hetero_no_rag` | Hetero without ChromaDB retrieval |

### Prompting strategies

| Strategy | Description |
|---|---|
| `zero_shot` | No examples, direct instruction |
| `few_shot` | 2-3 in-context examples from vector DB |
| `chain_of_thought` | Step-by-step reasoning before code |

---

## Usage

### Basic run (single issue)

```bash
python orchestrator.py
```

This runs the default issue defined in the `__main__` block against `target.py` and `test_target.py`.

### Custom issue

Call `run()` directly from your own script:

```python
from orchestrator import run

run(
    issue_text="The calculate_average function crashes on empty lists.",
    repo_path="/path/to/your/repo",
    test_file="test_target.py",
    max_attempts=3,
    config="hetero_no_rag",
    prompting_strategy="few_shot",
    task_id="my_task_001"
)
```

### Monitor progress

Each run appends a JSON line to `benchmark/results.jsonl`:

```bash
# count completed runs
wc -l benchmark/results.jsonl

# live tail
tail -f benchmark/results.jsonl

# see pass/fail summary
cat benchmark/results.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    r = json.loads(line.strip())
    if r:
        status = 'PASS' if r['passed'] else 'FAIL'
        print(f\"{status} | {r['configuration']} | {r['prompting_strategy']} | {r['task_id']}\")
"
```

---

## Running Experiments

The experiment runner evaluates all configurations across 15 HumanEval tasks and 2 prompting strategies (120 total runs).

```bash
python benchmark/run_experiments.py
```

### Resume after crash

The runner automatically skips already-logged tasks. Just re-run the same command:

```bash
python benchmark/run_experiments.py
```

### Customize experiment scope

Edit these three variables at the top of `benchmark/run_experiments.py`:

```python
# cut to fewer tasks
TASKS = TASKS[:10]

# change strategies
PROMPTING_STRATEGIES = ["zero_shot", "few_shot"]

# change configurations
CONFIGURATIONS = [
    {"config": "single_agent",  "models": {"coder": "qwen2.5-coder:7b"}},
    {"config": "homo_llama3",   "models": {"coder": "llama3.2"}},
]
```

### Estimated runtime

| Model | Time per task | 15 tasks |
|---|---|---|
| `qwen2.5-coder:7b` | ~20-30 sec | ~8 min |
| `llama3.2` | ~30-60 sec | ~15 min |

Full 120-run experiment on M3 MacBook Pro: approximately **2-3 hours**.

---

## Analyzing Results

```bash
python benchmark/analyze_results.py
```

Output:

```
============================================================
MAARA RESULTS SUMMARY
============================================================
hetero_no_rag        | few_shot   | pass@1: 100.0%  (n=15)
hetero_no_rag        | zero_shot  | pass@1:  93.3%  (n=15)
single_agent         | few_shot   | pass@1:  86.7%  (n=15)
single_agent         | zero_shot  | pass@1:  86.7%  (n=15)
homo_llama3          | few_shot   | pass@1:  73.3%  (n=15)
hetero_no_sentinel   | few_shot   | pass@1:  66.7%  (n=15)
hetero_no_sentinel   | zero_shot  | pass@1:  60.0%  (n=15)
homo_llama3          | zero_shot  | pass@1:  46.7%  (n=15)
============================================================
```

---

## Project Structure

```
self-healing-agent/
├── orchestrator.py              # Main pipeline entry point
├── agent.py                     # Original single-agent prototype (v1)
├── target.py                    # Target file for repair (used in tests)
├── test_target.py               # Test suite for target.py
│
├── agents/
│   ├── issue_parser.py          # Parses natural language issues into structured plans
│   ├── navigator.py             # Semantic retrieval and file structure analysis
│   ├── evaluator.py             # Sentinel agent — strategy checking
│   ├── coder.py                 # Code generation with prompting strategy support
│   ├── debugger.py              # Failure diagnosis
│   ├── test_runner.py           # Applies fixes and runs pytest
│   └── retriever.py             # ChromaDB index build and retrieval
│
└── benchmark/
    ├── run_experiments.py       # Full experiment runner (120 runs)
    ├── analyze_results.py       # pass@1 reporting from results.jsonl
    └── results.jsonl            # Experiment log (auto-generated)
```

---

## Known Issues

**RAG contamination** — The retriever currently indexes all Python files in the repository including benchmark case files. This pollutes the coder's context with irrelevant code and hurts performance. Fix: filter `benchmark/` from the ChromaDB index in `agents/retriever.py`.

**Evaluator over-rejection** — `llama3.2` at 3B sometimes rejects valid plans with low scores. The score threshold is not enforced by default. Fix: add `and eval_result["score"] >= 6` to the approval check in `orchestrator.py`.

**Large model memory** — Models above 13B parameters may partially offload to CPU on machines with less than 16GB unified memory, significantly slowing inference. Use `ollama ps` to check GPU utilization.

---

## Authors

**Tabrez Mohamed** — tm3578@rit.edu  
**Aparnaa Senthilnathan** — as1567@rit.edu  
**Ashish Ubale** — au9332@rit.edu  

Rochester Institute of Technology
