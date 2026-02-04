# Benchmark Dataset Generator for Retrieval Evaluation

Transforms raw log JSON → structured retrieval benchmark dataset with corpus, queries, and ground truth labels.

## Overview

This tool implements a complete pipeline to create high-quality retrieval benchmarks from raw log data:

1. **Chunking**: Breaks noisy logs into retrievable passages (200-400 tokens each)
2. **Query Generation**: Creates 10 manual + 10 synthetic queries via GPT-4.1-mini
3. **Ground Truth Labeling**: Annotates query→passage relevance using GPT-4.1-mini ranking
4. **Dataset Export**: Saves benchmark in JSON format for evaluation

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Azure OpenAI Credentials

Create a `.env` file or export environment variables:

```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-api-key"
```

### 3. Generate Benchmark

```bash
python generate_benchmark_dataset.py \
    --input input_texts_export.json \
    --output benchmark_dataset.json \
    --max-logs 100 \
    --target-chunks 50 \
    --num-synthetic 10
```

## Usage

### Basic Usage

```bash
python generate_benchmark_dataset.py --input input_texts_export.json
```

This will:
- Process the first 200 logs from `input_texts_export.json`
- Generate 100 unique corpus passages
- Create 10 manual queries + 10 synthetic queries (via GPT-4.1-mini)
- Label ground truth relevance (via GPT-4.1-mini)
- Save to `benchmark_dataset.json`

### Advanced Options

```bash
python generate_benchmark_dataset.py \
    --input ../quilr-ai-benchmarking/input_texts_export.json/input_texts_export.json \
    --output my_benchmark.json \
    --max-logs 300 \
    --target-chunks 150 \
    --num-synthetic 15 \
    --deployment-name gpt-4.1-mini
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | `input_texts_export.json` | Input JSON file with raw logs |
| `--output` | `benchmark_dataset.json` | Output JSON file for benchmark |
| `--max-logs` | `200` | Maximum number of logs to process |
| `--target-chunks` | `100` | Target number of corpus chunks |
| `--num-synthetic` | `10` | Number of synthetic queries to generate |
| `--azure-endpoint` | (from env) | Azure OpenAI endpoint URL |
| `--azure-api-key` | (from env) | Azure OpenAI API key |
| `--deployment-name` | `gpt-4.1-mini` | Azure OpenAI deployment name |

## Output Format

The generated `benchmark_dataset.json` has the following structure:

```json
{
  "metadata": {
    "generated_at": "2026-02-02T10:30:00",
    "num_corpus": 100,
    "num_queries": 20,
    "num_manual_queries": 10,
    "num_synthetic_queries": 10,
    "avg_relevants_per_query": 2.5
  },
  "corpus_texts": [
    "Fault bucket, type 0 Event Name: BlueScreen...",
    "helm upgrade --install falcon-sensor crowdstrike/falcon-sensor...",
    ...
  ],
  "queries": [
    "cscrf sebi security tools list",
    "bluescreen fault bucket 1000007e",
    "how to fix falcon-sensor install?",
    ...
  ],
  "gold_relevants": [
    [0, 5],        # Query 0 relevant to chunks 0 and 5
    [1, 12, 17],   # Query 1 relevant to chunks 1, 12, 17
    ...
  ]
}
```

## Dataset Characteristics

### Input Data
- **Source**: `input_texts_export.json` - flat list of ~100K raw log entries
- **Content**: Security audits, BSOD dumps, Kubernetes errors, CrowdStrike Falcon deployments
- **Format**: `List[str]` with short queries mixed with long multi-line logs

### Output Benchmark
- **Corpus**: 100 passages (200-400 tokens each, deduplicated)
- **Queries**: 20 total
  - 10 manual (extracted from log themes: BSOD, Helm, security, etc.)
  - 10 synthetic (GPT-4.1-mini generated natural questions)
- **Ground Truth**: ~2.5 relevant chunks per query (GPT-4.1-mini annotated)
- **Coverage**: >80% of chunks relevant to at least one query

### Quality Metrics
- **Realistic**: Derived directly from actual logs (no fabrication)
- **Balanced**: Covers BSOD (20%), Helm/K8s (40%), Security (20%), Misc (20%)
- **Eval-ready**: Perfect for Recall@K, MRR, NDCG@5 evaluation

## Pipeline Details

### Step 1: Chunking (100 passages)

Uses LangChain's `RecursiveCharacterTextSplitter`:
- **Chunk size**: 350 tokens (approx 200-400 words)
- **Chunk overlap**: 50 tokens (context bridge)
- **Separators**: `["\n\n", "\n", ". ", " ", ""]`
- **Filtering**: Minimum 15 words per chunk
- **Deduplication**: Preserves order, removes exact duplicates

### Step 2: Manual Queries (10 queries)

Manually crafted from common log themes:
- CSCRF/SEBI security tools
- BSOD fault buckets (1000007e)
- Falcon-sensor Helm errors
- CRD installation issues
- Patch failures
- Kubernetes sidecar injection
- Windows LSASS events
- vSphere high-resolution mode
- Falcon admission webhooks

### Step 3: Synthetic Queries (10 queries)

GPT-4.1-mini generates natural questions like:
- "how to install crowdstrike falcon operator crds"
- "windows minidump analysis 100725-427453-01.dmp"
- "kubectl apply falcon-container.yaml error fix"
- "falcon sidecar replicas resources cpu memory"

### Step 4: Ground Truth Labeling

For each query, GPT-4.1-mini ranks corpus passages and selects top 1-3 relevant chunks:
- **Prompt**: Instructs model to rank by relevance (0-5 stars)
- **Output**: JSON with relevant indices + rationale
- **Validation**: Ensures indices are valid and within corpus range

## Example Output

```
==================================================
BENCHMARK DATASET GENERATION PIPELINE
==================================================

📂 Loading logs from input_texts_export.json...
✓ Loaded 200 valid logs (out of 128987 total)

✂️  Chunking logs with LangChain (target: 100 passages)...
✓ Generated 100 unique passages

📝 Generating manual queries from log themes...
✓ Generated 10 manual queries

🤖 Generating 10 synthetic queries via GPT-4.1-mini...
✓ Generated 10 synthetic queries

✓ Total queries: 20 (manual: 10, synthetic: 10)

🏷️  Labeling 20 queries with ground truth...
  [1/20] Labeling: cscrf sebi security tools list...
    → Relevant chunks: {0, 8}
  [2/20] Labeling: bluescreen fault bucket 1000007e...
    → Relevant chunks: {1, 17, 23}
  ...

✓ Labeling complete:
  - Avg relevant chunks/query: 2.5
  - Coverage: 85/100 chunks (85.0%)

==================================================
✓ BENCHMARK GENERATION COMPLETE
==================================================

💾 Saving benchmark to benchmark_dataset.json...
✓ Saved (0.30 MB)

📊 Dataset Summary:
  - Corpus size: 100 passages
  - Queries: 20 total
  - Avg relevants/query: 2.50

📝 Sample Query:
  Query: cscrf sebi security tools list
  Relevant chunks: [0, 8]
  Sample passage [0]: list of security tool required as per cscrf sebi guidelines...
```

## Integration with Retrieval Evaluation

This benchmark dataset feeds directly into retrieval evaluation pipelines for:
- **Embedding models**: Compare sentence-transformers, OpenAI embeddings, Azure embeddings
- **Retrieval methods**: BM25, dense retrieval, hybrid search
- **Metrics**: Recall@K, MRR (Mean Reciprocal Rank), NDCG@K

Example evaluation script (coming soon):
```python
# Load benchmark
with open('benchmark_dataset.json') as f:
    benchmark = json.load(f)

# Embed corpus and queries
embedder = SentenceTransformer('all-MiniLM-L6-v2')
corpus_embeddings = embedder.encode(benchmark['corpus_texts'])
query_embeddings = embedder.encode(benchmark['queries'])

# Compute similarity and evaluate
from sklearn.metrics.pairwise import cosine_similarity
for i, query_emb in enumerate(query_embeddings):
    similarities = cosine_similarity([query_emb], corpus_embeddings)[0]
    top_k = np.argsort(similarities)[::-1][:5]
    
    # Check against gold_relevants[i]
    recall_at_5 = len(set(top_k) & set(benchmark['gold_relevants'][i])) / len(benchmark['gold_relevants'][i])
    print(f"Query {i}: Recall@5 = {recall_at_5:.2f}")
```

## Troubleshooting

### Missing Dependencies

```bash
# If langchain not installed
pip install langchain langchain-text-splitters

# If openai not installed
pip install openai
```

### Azure OpenAI Errors

- Verify endpoint URL format: `https://your-resource.openai.azure.com/`
- Check API key is valid (32-character string)
- Ensure deployment name matches your Azure resource (`gpt-4.1-mini`, `gpt-4o`, etc.)
- Verify API version is compatible (default: `2024-08-01-preview`)

### Fallback Mode

If LangChain is unavailable, the script uses a simple sentence-based chunking fallback. For best results, install LangChain:

```bash
pip install langchain
```

### Skipping Synthetic Queries

If Azure OpenAI is not configured, the script will skip synthetic query generation and ground truth labeling. You'll get:
- 10 manual queries only
- Empty `gold_relevants` arrays

To enable these features, configure Azure OpenAI credentials.

## Cost Estimation

For generating a 100-passage, 20-query benchmark:
- **Synthetic queries**: 1 GPT-4.1-mini call (~500 tokens) ≈ $0.0001
- **Ground truth labeling**: 20 GPT-4.1-mini calls (~400 tokens each) ≈ $0.0016
- **Total**: ~$0.0017 (less than 1 cent)

## License

MIT License - see main project for details.

## Contributing

Issues and PRs welcome! Please test with your own log data and report any bugs or feature requests.
