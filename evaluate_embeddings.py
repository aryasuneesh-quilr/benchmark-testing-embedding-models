"""
Embedding Model Evaluation using Benchmark Dataset
===================================================
Compares embedding models on a benchmark dataset to find the best semantic search quality.

Features:
- Dynamic Model Loading (Hugging Face IDs)
- Latency Tracking (ms per document)
- Standard IR Metrics (Recall, MRR, NDCG)
- CSV Export

Usage:
    python evaluate_embeddings.py --benchmark benchmark_dataset.json --models "prdev/mini-gte" "BAAI/bge-small-en-v1.5"
"""

import json
import numpy as np
import time
import argparse
import csv
import os
from typing import List, Set, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import ndcg_score

# Try importing sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("❌ Error: sentence-transformers not installed. Install with: pip install sentence-transformers")

class EmbeddingEvaluator:
    """Evaluates embedding models on benchmark dataset"""
    
    def __init__(self, benchmark_path: str):
        """Load benchmark dataset"""
        if not os.path.exists(benchmark_path):
            raise FileNotFoundError(f"Benchmark file {benchmark_path} not found.")

        print(f"\n📂 Loading benchmark from {benchmark_path}...")
        with open(benchmark_path, 'r', encoding='utf-8') as f:
            self.benchmark = json.load(f)
        
        self.corpus_texts = self.benchmark['corpus']  # Updated key to match previous script
        self.queries = self.benchmark['queries']
        self.gold_relevants = [set(rel) for rel in self.benchmark['relevant_docs']] # Updated key
        
        print(f"   ✓ Loaded {len(self.corpus_texts)} corpus passages")
        print(f"   ✓ Loaded {len(self.queries)} queries")
    
    def recall_at_k(self, retrieved_indices: List[int], gold_relevant: Set[int], k: int = 5) -> float:
        """Fraction of relevant docs found in top K results."""
        if not gold_relevant: return 0.0
        retrieved_set = set(retrieved_indices[:k])
        return len(retrieved_set & gold_relevant) / len(gold_relevant)
    
    def mrr(self, retrieved_indices: List[int], gold_relevant: Set[int]) -> float:
        """Mean Reciprocal Rank (1/rank of first relevant item)."""
        for rank, idx in enumerate(retrieved_indices, start=1):
            if idx in gold_relevant:
                return 1.0 / rank
        return 0.0
    
    def ndcg_at_k(self, retrieved_indices: List[int], gold_relevant: Set[int], k: int = 5) -> float:
        """Normalized Discounted Cumulative Gain at K."""
        if not gold_relevant: return 0.0
        
        # Binary relevance for standard retrieval evaluation
        relevance = [1 if idx in gold_relevant else 0 for idx in retrieved_indices[:k]]
        
        if sum(relevance) == 0: return 0.0
        
        # Ideal ranking: all 1s at the top
        ideal_relevance = [1] * min(len(gold_relevant), k) + [0] * max(0, k - len(gold_relevant))
        
        try:
            # sklearn expects shape (n_samples, n_labels)
            return ndcg_score([ideal_relevance], [relevance])
        except:
            return 0.0
    
    def evaluate_model(self, model_name: str) -> Dict:
        """Load model, encode, and calculate metrics."""
        print(f"\n🔬 Evaluating Model: {model_name}")
        print("=" * 60)
        
        try:
            model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"❌ Failed to load model {model_name}: {e}")
            return None

        # --- Measure Latency ---
        print(f"   📊 Encoding {len(self.corpus_texts)} passages...")
        start_time = time.time()
        corpus_embeddings = model.encode(self.corpus_texts, convert_to_numpy=True, show_progress_bar=False)
        corpus_time = time.time() - start_time
        latency_ms_per_doc = (corpus_time / len(self.corpus_texts)) * 1000

        print(f"   📊 Encoding {len(self.queries)} queries...")
        query_embeddings = model.encode(self.queries, convert_to_numpy=True, show_progress_bar=False)
        
        # --- Calculate Metrics ---
        recall_1, recall_5, recall_10, mrr_scores, ndcg_5 = [], [], [], [], []
        
        print("   🎯 Computing ranking metrics...")
        for i, query_emb in enumerate(query_embeddings):
            # Cosine Similarity
            similarities = cosine_similarity([query_emb], corpus_embeddings)[0]
            # Argsort gives ascending, so we reverse it for descending similarity
            ranked_indices = np.argsort(similarities)[::-1]
            
            gold = self.gold_relevants[i]
            
            recall_1.append(self.recall_at_k(ranked_indices, gold, k=1))
            recall_5.append(self.recall_at_k(ranked_indices, gold, k=5))
            recall_10.append(self.recall_at_k(ranked_indices, gold, k=10))
            mrr_scores.append(self.mrr(ranked_indices, gold))
            ndcg_5.append(self.ndcg_at_k(ranked_indices, gold, k=5))
        
        results = {
            'Model': model_name,
            'Latency (ms/doc)': round(latency_ms_per_doc, 2),
            'Recall@1': round(np.mean(recall_1), 4),
            'Recall@5': round(np.mean(recall_5), 4),
            'Recall@10': round(np.mean(recall_10), 4),
            'MRR': round(np.mean(mrr_scores), 4),
            'NDCG@5': round(np.mean(ndcg_5), 4)
        }
        
        print(f"   ✅ Latency: {results['Latency (ms/doc)']} ms/doc | MRR: {results['MRR']}")
        return results

    def save_results(self, all_results: List[Dict], output_file: str = "evaluation_results.csv"):
        """Save comparison results to CSV."""
        if not all_results: return
        
        keys = all_results[0].keys()
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(all_results)
        print(f"\n💾 Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Compare Embedding Models")
    parser.add_argument("--benchmark", type=str, default="benchmark_dataset.json", help="Path to benchmark JSON")
    parser.add_argument("--models", nargs='+', default=["prdev/mini-gte", "jhu-clsp/ettin-encoder-68m"], 
                        help="List of HuggingFace model IDs to compare")
    parser.add_argument("--output", type=str, default="evaluation_results.csv", help="Output CSV path")
    
    args = parser.parse_args()
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return

    evaluator = EmbeddingEvaluator(args.benchmark)
    
    all_results = []
    for model_id in args.models:
        res = evaluator.evaluate_model(model_id)
        if res:
            all_results.append(res)
            
    # Print Comparison Table
    print("\n" + "="*95)
    print(f"{'Model':<30} {'Latency':<10} {'R@1':<8} {'R@5':<8} {'MRR':<8} {'NDCG@5':<8}")
    print("-" * 95)
    for res in all_results:
        print(f"{res['Model']:<30} {res['Latency (ms/doc)']:<10} {res['Recall@1']:<8} {res['Recall@5']:<8} {res['MRR']:<8} {res['NDCG@5']:<8}")
    print("="*95)
    
    evaluator.save_results(all_results, args.output)

if __name__ == "__main__":
    main()