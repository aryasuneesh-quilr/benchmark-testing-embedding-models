import json
import os
import random
import argparse
from typing import List, Dict, Set, Any
from datetime import datetime

# --- Optional Imports ---
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False

try:
    from openai import AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class BenchmarkGenerator:
    def __init__(self, 
                 azure_endpoint: str = None,
                 azure_api_key: str = None,
                 deployment_name: str = "gpt-4.1-mini",
                 api_version: str = "2024-08-01-preview"):
        
        self.deployment_name = deployment_name
        self.client = None
        
        if OPENAI_AVAILABLE and azure_endpoint and azure_api_key:
            self.client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=azure_api_key,
                api_version=api_version
            )
            print(f"✓ Azure OpenAI initialized ({deployment_name})")
        else:
            print("⚠ OpenAI client NOT initialized. Synthetic generation will fail.")

    def load_and_chunk(self, input_path: str, target_chunks: int = 100) -> List[str]:
        """Loads logs and chunks them into a retrieval corpus."""
        print(f"\n📂 Loading {input_path}...")
        with open(input_path, 'r', encoding='utf-8') as f:
            logs = json.load(f)
        
        # Filter noise (short logs)
        raw_text = [str(log) for log in logs if len(str(log).strip()) > 30]
        
        # Chunking Strategy
        if LANGCHAIN_AVAILABLE:
            print("   Using LangChain RecursiveSplitter...")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=50,
                length_function=len
            )
            all_chunks = []
            for text in raw_text:
                all_chunks.extend(splitter.split_text(text))
        else:
            print("   Using Simple Fallback Splitter...")
            all_chunks = raw_text[:target_chunks * 2] 

        # Deduplicate
        unique_chunks = list(dict.fromkeys([c.strip() for c in all_chunks if len(c) > 40]))
        
        # Random Sampling for Coverage
        if len(unique_chunks) > target_chunks:
            selected_chunks = random.sample(unique_chunks, target_chunks)
        else:
            selected_chunks = unique_chunks
            
        print(f"✓ Corpus Prepared: {len(selected_chunks)} unique passages.")
        return selected_chunks

    def label_manual_queries(self, queries: List[str], corpus: List[str]) -> tuple[List[str], List[List[int]]]:
        """
        Labels manual queries by asking LLM to check ALL chunks.
        """
        if not self.client or not queries:
            return [], []

        print(f"\n🏷️  Labeling {len(queries)} Manual Queries against {len(corpus)} chunks...")
        
        # Create a compressed view of ALL chunks (0 to 99)
        corpus_block = "\n".join([f"[{i}] {t[:150].replace(chr(10), ' ')}" for i, t in enumerate(corpus)])
        
        labeled_queries = []
        labeled_relevance = []

        for q in queries:
            # We strictly ask for IDs from the full list
            prompt = f"""
Query: "{q}"

Identify which Log IDs below are RELEVANT to the query.
- Return JSON: {{"ids": [0, 99, ...]}}
- If none, return empty list.

Logs:
{corpus_block}
"""
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                data = json.loads(response.choices[0].message.content)
                ids = [i for i in data.get("ids", []) if 0 <= i < len(corpus)]
                
                if ids:
                    labeled_queries.append(q)
                    labeled_relevance.append(ids)
                    print(f"   ✓ '{q[:30]}...' -> Matches {ids}")
                else:
                    print(f"   ✗ '{q[:30]}...' -> No matches found. Dropped.")
                    
            except Exception as e:
                print(f"   ⚠ Error labeling '{q}': {e}")

        return labeled_queries, labeled_relevance

    def generate_synthetic_inverted(self, corpus: List[str], count_needed: int) -> tuple[List[str], List[List[int]]]:
        """
        Inverted Generation: Pick a chunk -> Generate Query -> Link guaranteed.
        """
        if not self.client or count_needed <= 0:
            return [], []

        print(f"\n⚡ Generating {count_needed} Synthetic Queries (Inverted Method)...")
        
        # Sample chunks (with replacement if needed)
        if len(corpus) >= count_needed:
            anchor_indices = random.sample(range(len(corpus)), count_needed)
        else:
            anchor_indices = [random.choice(range(len(corpus))) for _ in range(count_needed)]

        syn_queries = []
        syn_relevance = []

        for i, idx in enumerate(anchor_indices):
            anchor_text = corpus[idx]
            
            prompt = f"""
Here is a log snippet:
"{anchor_text[:500]}"

Write a specific, natural search query (as a DevOps engineer) that would lead to finding this specific log.
- Do not quote the log directly.
- Return JSON: {{"query": "..."}}
"""
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.7 
                )
                data = json.loads(response.choices[0].message.content)
                query = data.get("query")
                
                if query:
                    syn_queries.append(query)
                    syn_relevance.append([idx]) 
                    print(f"   [{i+1}/{count_needed}] generated for chunk {idx}")
                
            except Exception as e:
                print(f"   ⚠ Error generating for chunk {idx}: {e}")

        return syn_queries, syn_relevance

    def run(self, input_file: str, output_file: str, total_queries: int = 50, target_chunks: int = 100):
        # 1. Corpus
        corpus = self.load_and_chunk(input_file, target_chunks=target_chunks)

        # 2. Manual Queries (Randomized)
        manual_qs = []
        if os.path.exists("manual_queries.json"):
            with open("manual_queries.json", "r") as f:
                all_manual = json.load(f)
            
            sample_size = min(len(all_manual), total_queries)
            manual_qs = random.sample(all_manual, sample_size)
            print(f"✓ Loaded {len(manual_qs)} randomized manual queries")
            
        final_qs, final_rels = self.label_manual_queries(manual_qs, corpus)

        # 3. Fill remaining with Synthetic
        needed = total_queries - len(final_qs)
        if needed > 0:
            syn_qs, syn_rels = self.generate_synthetic_inverted(corpus, needed)
            final_qs.extend(syn_qs)
            final_rels.extend(syn_rels)

        # 4. Save
        dataset = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "total_queries": len(final_qs),
                "corpus_size": len(corpus)
            },
            "corpus": corpus,
            "queries": final_qs,
            "relevant_docs": final_rels
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2)
            
        print(f"\n✅ Benchmark Saved to {output_file}")
        print(f"   - Corpus: {len(corpus)}")
        print(f"   - Queries: {len(final_qs)} ({len(final_qs)-len(syn_qs)} Manual, {len(syn_qs)} Synthetic)")


if __name__ == "__main__":
    if DOTENV_AVAILABLE: load_dotenv()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="input_texts_export.json")
    
    # We removed the default value here to detect if user provided input
    parser.add_argument("--output", default=None, help="Output filename (optional)") 
    
    parser.add_argument("--queries", type=int, default=50, help="Total queries to generate")
    parser.add_argument("--chunks", type=int, default=100, help="Total log chunks to include in corpus")
    args = parser.parse_args()

    # Dynamic Filename Logic
    if args.output:
        output_filename = args.output
    else:
        # Auto-generate name based on params
        output_filename = f"benchmark_dataset_{args.chunks}chunks_{args.queries}queries.json"

    gen = BenchmarkGenerator(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )
    
    gen.run(args.input, output_filename, total_queries=args.queries, target_chunks=args.chunks)