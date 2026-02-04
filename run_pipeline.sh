# =================================================================
# AUTOMATED EMBEDDING EVALUATION PIPELINE
# =================================================================

Param(
    [string]$firstArg = "500",
    [string]$secondArg = "50"
)

# 1. Configuration
$INPUT_FILE = "input_texts_export.json"
$MODELS = @(
    "prdev/mini-gte",
    "jhu-clsp/ettin-encoder-68m",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2"
)

$CHUNKS = 500
$QUERIES = 50
$SKIP_GEN = $false
$BENCHMARK_FILE = ""

# 2. Smart Argument Parsing
if ($firstArg.EndsWith(".json")) {
    $BENCHMARK_FILE = $firstArg
    if (-not (Test-Path $BENCHMARK_FILE)) {
        Write-Error "Error: Specified benchmark file '$BENCHMARK_FILE' not found."
        exit 1
    }
    $SKIP_GEN = $true
    Write-Host "MODE: Using Existing Benchmark ($BENCHMARK_FILE)" -ForegroundColor Cyan
} else {
    $CHUNKS = $firstArg
    $QUERIES = $secondArg
    $SKIP_GEN = $false
    Write-Host "MODE: Generating New Benchmark ($CHUNKS chunks, $QUERIES queries)" -ForegroundColor Cyan
}

# 3. Setup Dependencies
Write-Host "Checking dependencies..."
pip install -r requirements.txt | Out-Null

# 4. Pipeline Execution
Write-Host "========================================================"
Write-Host "STARTING PIPELINE"
Write-Host "Models: $($MODELS -join ', ')"
Write-Host "========================================================"

if (-not $SKIP_GEN) {
    if (-not (Test-Path $INPUT_FILE)) {
        Write-Error "Error: Input file $INPUT_FILE not found."
        exit 1
    }

    $NUM_LINES = (Get-Content $INPUT_FILE | Measure-Object -Line).Lines
    if ([int]$CHUNKS -gt [int]$NUM_LINES) {
        Write-Error "Error: Chunks ($CHUNKS) > Total Lines ($NUM_LINES)"
        exit 1
    }

    Write-Host "[Step 1/2] Generating Benchmark..." -ForegroundColor Yellow
    python generate_benchmark_final.py --input "$INPUT_FILE" --chunks "$CHUNKS" --queries "$QUERIES"

    $BENCHMARK_FILE = "benchmark_dataset_${CHUNKS}chunks_${QUERIES}queries.json"
}

# --- Step 2: Evaluate ---
Write-Host "[Step 2/2] Evaluating Models..." -ForegroundColor Yellow

$BASENAME = [System.IO.Path]::GetFileNameWithoutExtension($BENCHMARK_FILE)
$OUTPUT_CSV = "results_${BASENAME}.csv"
$MODEL_ARGS = $MODELS -join " "

python evaluate_embeddings.py --benchmark "$BENCHMARK_FILE" --models $MODEL_ARGS --output "$OUTPUT_CSV"

Write-Host "========================================================"
Write-Host "PIPELINE COMPLETE!" -ForegroundColor Green
Write-Host "Final Report: $OUTPUT_CSV"
Write-Host "========================================================"