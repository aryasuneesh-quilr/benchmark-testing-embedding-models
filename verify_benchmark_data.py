"""
Benchmark Data CSV Exporter
===========================
Converts benchmark JSON -> CSV for easy inspection.
Output columns: Query, Relevant_Chunks (Full Text)

Usage: python verify_benchmark_data.py --input benchmark_dataset.json --output benchmark_verification.csv
"""

import json
import argparse
import os
import csv

def generate_csv_report(input_path: str, output_path: str):
    # 1. Load Data
    if not os.path.exists(input_path):
        print(f"❌ Error: File '{input_path}' not found.")
        return

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Error decoding JSON: {e}")
        return

    print("JSON keys:", data.keys())
    print("queries:", len(data.get("queries", [])))
    print("gold_relevants:", len(data.get("relevant_docs", [])))
    print("corpus_texts:", len(data.get("corpus", [])))

    required_keys = ["corpus", "queries", "relevant_docs"]
    missing = [k for k in required_keys if k not in data]

    if missing:
        raise KeyError(f"Missing required keys in dataset: {missing}")

    corpus = data['corpus']
    queries = data['queries']
    gold_relevants = data['relevant_docs']

    # Validate data lengths match
    if len(queries) != len(gold_relevants):
        print(f"⚠️ Warning: Mismatch in counts (Queries: {len(queries)}, Relevance Sets: {len(gold_relevants)})")

    # 2. Write CSV
    print(f"📝 Writing CSV to {output_path}...")
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        
        # Header
        writer.writerow(['Query', 'Relevant Chunks (Full Text)'])

        count = 0
        for query, rel_ids in zip(queries, gold_relevants):
            chunk_texts = []
            
            # Fetch text for each relevant ID
            for doc_id in rel_ids:
                if 0 <= doc_id < len(corpus):
                    # Add [ID] prefix for clarity in the CSV cell
                    chunk_texts.append(f"[Chunk {doc_id}]\n{corpus[doc_id]}")
                else:
                    chunk_texts.append(f"[INVALID ID {doc_id}]")
            
            # Join multiple chunks with a separator if a query maps to multiple logs
            full_text_cell = "\n----------------------------------------\n".join(chunk_texts)
            
            if not full_text_cell:
                full_text_cell = "(No relevant chunks mapped)"

            writer.writerow([query, full_text_cell])
            count += 1

    print(f"✅ CSV generated successfully with {count} rows.")
    print(f"📂 Output: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="benchmark_dataset_500chunks_100queries.json", help="Path to input JSON")
    parser.add_argument("--output", default="benchmark_verification_500chunks_100queries.csv", help="Path to output CSV")
    args = parser.parse_args()

    generate_csv_report(args.input, args.output)