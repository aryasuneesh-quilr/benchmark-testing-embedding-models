# Benchmark Dataset Generator & Evaluation Pipeline

Transforms raw log JSON into structured retrieval benchmark datasets and automatically evaluates multiple embedding models.

## Overview

This tool provides an end-to-end pipeline to create high-quality retrieval benchmarks from raw log data and instantly test how different embedding models perform on that data.


1. **Generation**: Breaks logs into passages and uses GPT-4.1-mini to generate natural queries and ground-truth labels.
2. **Evaluation**: Runs the generated benchmark against a suite of embedding models (e.g., GTE, MiniLM, MPNet) to calculate retrieval metrics like Recall@K.

---

## Setup & Installation

### 1. Environment Setup

It is recommended to use a virtual environment to manage dependencies.

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

### 2. Configure Credentials

The pipeline requires **Azure OpenAI** for query generation and ground-truth labeling. Create a `.env` file in the root directory:

```ini
AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
AZURE_OPENAI_API_KEY="your-api-key"

```

---

## Quick Start (The Unified Pipeline)

The `run_pipeline.py` script automates both dataset generation and model evaluation in one command.

### Mode A: Generate and Evaluate

Create a new benchmark from your logs and immediately test models against it:

```bash
# Format: python run_pipeline.py <num_chunks> <num_queries>
python run_pipeline.py 500 50

```

### Mode B: Use Existing Benchmark

If you already have a generated benchmark JSON, you can run the evaluation step only:

```bash
# Format: python run_pipeline.py <path_to_json>
python run_pipeline.py benchmark_dataset_500chunks_50queries.json

```

---

## Tool Details

The pipeline is powered by two specialized scripts that handle the transition from unstructured logs to validated performance metrics.

#### 1. Dataset Generation (`generate_benchmark_dataset.py`)

This script automates the creation of a "Golden Dataset" by transforming raw logs into a retrieval-ready format.

* **Intelligent Chunking**: Uses LangChain's `RecursiveCharacterTextSplitter` to break logs into 350-400 token passages, ensuring context is preserved across technical boundaries.
* **Hybrid Query Creation**:
* **Manual**: Randomly samples pre-defined expert queries (from `manual_queries.json`) and uses GPT-4.1-mini to find their "needle in the haystack" matches across the new corpus.
* **Synthetic (Inverted Generation)**: Selects random log passages and tasks the LLM to write the specific query that would naturally lead to that result, ensuring 100% ground-truth accuracy.

#### 2. Model Evaluation (`evaluate_embeddings.py`)

A comprehensive benchmarking suite that measures how "smart" different embedding models are at finding the right logs.

* **Performance Metrics**: Calculates industry-standard Information Retrieval (IR) scores:
* **Recall@K**: Percentage of relevant logs found in the top  results.
* **MRR (Mean Reciprocal Rank)**: Measures how close to the top the first relevant result appeared.
* **NDCG@5**: Evaluates the quality of the ranking order.

---

## Output Files

| File | Description |
| --- | --- |
| `benchmark_dataset_Xchunks_Yqueries.json` | The generated corpus, queries, and ground-truth mappings. |
| `results_benchmark_dataset_...csv` | A detailed CSV report comparing the accuracy of all tested embedding models. |

### Sample Evaluation Output

```text
🚀 STARTING PIPELINE
- Models: prdev/mini-gte, jhu-clsp/ettin-encoder-68m, ...
==================================================

[Step 1/2] Generating Benchmark...
✓ Generated 500 unique passages
✓ Total queries: 50 (manual: 10, synthetic: 40)

[Step 2/2] Evaluating Models...
Evaluating prdev/mini-gte: Recall@5 = 0.88
Evaluating sentence-transformers/all-MiniLM-L6-v2: Recall@5 = 0.82

🎉 PIPELINE COMPLETE!
- Final Report: results_benchmark_dataset_500chunks_50queries.csv

```

---

## Troubleshooting

* **Credential Errors**: Ensure your `.env` file is in the same directory where you run the script.
* **Input File Missing**: The script expects `input_texts_export.json` (a list of log strings) to be present in the root.
* **Memory Issues**: If running on a local machine with limited RAM, reduce the number of models in the `MODELS` list inside `run_pipeline.py`.
