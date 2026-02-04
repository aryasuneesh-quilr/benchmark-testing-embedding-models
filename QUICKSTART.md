# Quick Start Guide

Get your benchmark dataset generated in 3 minutes!

## Prerequisites

- Python 3.8+ installed
- Azure OpenAI account with GPT-4.1-mini deployment
- Input log data (import it as `input_texts_export.json`)

## Step 1: Setup (1 minute)

### Windows (PowerShell)

```powershell
# Run setup script
.\setup.ps1
```

### Linux/Mac (Bash)

```bash
# Make script executable
chmod +x setup.sh

# Run setup script
./setup.sh
```

### Manual Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

## Step 2: Configure Azure OpenAI (30 seconds)

Edit `.env` file:

```bash
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
```

**Where to find credentials:**
1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to your Azure OpenAI resource
3. Click "Keys and Endpoint"
4. Copy **Endpoint** and **Key 1**

## Step 3: Generate Benchmark (1-2 minutes)

**IMPORTANT**: Make sure your virtual environment is activated first!

### Windows (PowerShell)
```powershell
# Activate venv
.venv\Scripts\activate

# Or use the quick run script
.\run.ps1
```

### Linux/Mac (Bash)
```bash
# Activate venv
source .venv/bin/activate

# Or use the quick run script
./run.sh
```

### Option A: Use CLI (Recommended)

```bash
python generate_benchmark_dataset.py \
    --input ../quilr-ai-benchmarking/input_texts_export.json/input_texts_export.json \
    --output benchmark_dataset.json \
    --max-logs 200 \
    --target-chunks 100 \
    --num-synthetic 10
```

Or simply:
```bash
python generate_benchmark_dataset.py  # Uses defaults
```

### Option B: Use Example Script

```bash
python example_usage.py
```

### Option C: Programmatic Usage

```python
from generate_benchmark_dataset import BenchmarkDatasetGenerator

generator = BenchmarkDatasetGenerator(
    azure_endpoint="https://your-resource.openai.azure.com/",
    azure_api_key="your-api-key"
)

dataset = generator.generate_benchmark(
    input_path="input_texts_export.json",
    max_logs=200,
    target_chunks=100,
    num_synthetic_queries=10
)

generator.save_benchmark(dataset, "benchmark_dataset.json")
```

## What You'll Get

After running, you'll have `benchmark_dataset.json` with:

- **100 corpus passages** - Chunked from your logs
- **20 queries** - 10 manual + 10 synthetic (GPT-4.1-mini generated)
- **Ground truth labels** - Query→passage relevance mappings

## Verify Output

```bash
# Check file was created
ls -lh benchmark_dataset.json

# Inspect contents (first 50 lines)
head -50 benchmark_dataset.json

# Or use Python
python -c "import json; d = json.load(open('benchmark_dataset.json')); print(f\"Corpus: {d['metadata']['num_corpus']} passages\"); print(f\"Queries: {d['metadata']['num_queries']} total\")"
```

## Common Issues

### Issue: "Warning: langchain not installed" or "⚠ Azure OpenAI client not initialized"

**Cause**: Virtual environment not activated

**Solution:**
```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

# Then run the script
python generate_benchmark_dataset.py
```

Or use the quick run script:
```bash
# Windows
.\run.ps1

# Linux/Mac
./run.sh
```

### Issue: `ModuleNotFoundError: No module named 'langchain'`

**Solution:**
```bash
# Make sure venv is activated first!
pip install langchain langchain-text-splitters
```

### Issue: `openai.AuthenticationError`

**Solution:**
- Check `.env` file has correct endpoint and API key
- Ensure no extra spaces or quotes in `.env` values
- Verify API key is valid in Azure Portal

### Issue: `Input file not found`

**Solution:**
```bash
# Check file exists
ls ../quilr-ai-benchmarking/input_texts_export.json/input_texts_export.json

# Or use local copy
python generate_benchmark_dataset.py --input input_texts_export.json
```

### Issue: Script runs but no synthetic queries generated

**Solution:**
- Check Azure OpenAI credentials are set correctly
- Verify deployment name matches your Azure resource
- Check Azure OpenAI quota/rate limits

## Next Steps

1. **Review benchmark**: Open `benchmark_dataset.json` and inspect quality
2. **Evaluate embeddings**: Use benchmark to test different embedding models
3. **Calculate metrics**: Compute Recall@K, MRR, NDCG
4. **Iterate**: Adjust parameters and regenerate if needed

## Cost Estimate

Generating a 100-passage, 20-query benchmark costs approximately:
- **GPT-4.1-mini tokens**: ~12K total
- **Azure OpenAI cost**: < $0.002 (less than 1 cent)

## Help

For detailed documentation, see:
- `README.md` - Full documentation
- `python generate_benchmark_dataset.py --help` - CLI options
- Example code in `example_usage.py`

## Support

Issues? Questions? Open an issue or check logs for error details.
