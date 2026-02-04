import os
import sys
import subprocess
import argparse
from dotenv import load_dotenv

# 1. Configuration
INPUT_FILE = "input_texts_export.json"
MODELS = [
    "prdev/mini-gte",
    "jhu-clsp/ettin-encoder-68m",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2"
]

def run_command(command):
    """Utility to run shell commands and stream output."""
    try:
        process = subprocess.Popen(command, shell=True)
        process.wait()
        if process.returncode != 0:
            print(f"❌ Command failed with exit code {process.returncode}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Execution error: {e}")
        sys.exit(1)

def main():
    load_dotenv()
    # 2. Smart Argument Parsing
    parser = argparse.ArgumentParser(description="Automated Embedding Evaluation Pipeline")
    parser.add_argument("arg1", nargs="?", default="500", help="Chunks count or benchmark JSON file")
    parser.add_argument("arg2", nargs="?", default="50", help="Queries count")
    
    args = parser.parse_args()

    benchmark_file = ""
    chunks = 500
    queries = 50
    skip_gen = False

    # Check if first argument is a JSON file
    if args.arg1.endswith(".json"):
        benchmark_file = args.arg1
        if not os.path.exists(benchmark_file):
            print(f"❌ Error: Specified benchmark file '{benchmark_file}' not found.")
            sys.exit(1)
        skip_gen = True
        print(f"📂 Mode: Using Existing Benchmark ({benchmark_file})")
    else:
        chunks = args.arg1
        queries = args.arg2
        print(f"⚙️  Mode: Generating New Benchmark ({chunks} chunks, {queries} queries)")

    # 3. Setup Dependencies
    print("\n📦 Checking dependencies...")
    
    # Check if environment variables are set
    if not os.getenv("AZURE_OPENAI_ENDPOINT") or not os.getenv("AZURE_OPENAI_API_KEY"):
        print("❌ Error: Azure OpenAI credentials not found. Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env file.")
        sys.exit(1)
    
    # only install dependencies if not already installed
    if not os.path.exists("requirements.txt"):
        print("❌ Error: requirements.txt not found. Please install dependencies manually.")
        sys.exit(1)

    # if venv exists, activate it, and don't install requirements if already installed

    # if os.path.exists("venv") or os.path.exists("benchmark-testing"):
    #     run_command(f"venv/bin/activate" if os.path.exists("venv") else "benchmark-testing/bin/activate")
    # else:
    #     run_command(f"python -m venv benchmark-testing")
    #     run_command(f"benchmark-testing/bin/activate")
    #     run_command(f"{sys.executable} -m pip install -r requirements.txt")

    print("\n" + "="*50)
    print("🚀 STARTING PIPELINE")
    print(f"   - Models: {', '.join(MODELS)}")
    print("="*50)

    # 4. Pipeline Execution
    if not skip_gen:
        # Step 1: Generate
        if not os.path.exists(INPUT_FILE):
            print(f"❌ Error: Input file {INPUT_FILE} not found.")
            sys.exit(1)

        # Line count validation
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            num_lines = sum(1 for line in f)
        
        if int(chunks) > num_lines:
            print(f"❌ Error: Chunks ({chunks}) > Total Lines ({num_lines})")
            sys.exit(1)

        print(f"\n[Step 1/2] Generating Benchmark...")
        run_command(f"python generate_benchmark_dataset.py --input {INPUT_FILE} --chunks {chunks} --queries {queries}")
        
        benchmark_file = f"benchmark_dataset_{chunks}chunks_{queries}queries.json"
        if not os.path.exists(benchmark_file):
            print("❌ Error: Generation failed - benchmark file not created.")
            sys.exit(1)
    else:
        print("\n[Step 1/2] Skipping Generation...")

    # Step 2: Evaluate
    print(f"\n[Step 2/2] Evaluating Models...")
    basename = os.path.splitext(benchmark_file)[0]
    output_csv = f"results_{basename}.csv"
    model_args = " ".join(MODELS)

    run_command(f"python evaluate_embeddings.py --benchmark {benchmark_file} --models {model_args} --output {output_csv}")

    # 5. Completion
    print("\n" + "="*50)
    print("🎉 PIPELINE COMPLETE!")
    print(f"   - Benchmark Used: {benchmark_file}")
    print(f"   - Final Report:   {output_csv}")
    print("="*50)

if __name__ == "__main__":
    main()