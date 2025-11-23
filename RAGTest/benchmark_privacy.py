"""
Benchmark script: Compare RAG performance with and without Privacy-Aware Summary module
"""
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List
import logging

sys.path.append(str(Path(__file__).parent))

from config import Config
from index import get_index
from retriever import get_retriver, response_synthesizer
from process.postprocess_rerank import get_postprocessor
from privacy.privacy_summary import get_privacy_postprocessors
from data.qa_loader import get_qa_dataset
from llms.llm import get_llm
from embs.embedding import get_embedding
from llama_index.core import Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from eval.evaluate_rag import NLGEvaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def benchmark_query_engine(query_engine, queries: List[str], expected_answers: List[str], 
                          privacy_enabled: bool) -> Dict:
    """Benchmark a query engine on a set of queries"""
    results = {
        "privacy_enabled": privacy_enabled,
        "total_queries": len(queries),
        "successful_queries": 0,
        "failed_queries": 0,
        "avg_response_time": 0.0,
        "avg_response_length": 0.0,
        "nlg_scores": {},
        "privacy_stats": {
            "total_pii_detected": 0,
            "total_sentences_removed": 0,
            "avg_risk_score": 0.0,
        }
    }
    
    response_times = []
    response_lengths = []
    nlg_scores_list = []
    pii_counts = []
    removed_counts = []
    risk_scores = []
    
    for i, (query, expected) in enumerate(zip(queries, expected_answers)):
        try:
            start_time = time.time()
            response = query_engine.query(query)
            elapsed = time.time() - start_time
            
            response_times.append(elapsed)
            response_lengths.append(len(response.response))
            
            # NLG evaluation
            nlg_score = NLGEvaluate(response.response, expected)
            nlg_scores_list.append(nlg_score)
            
            # Privacy metadata
            if response.source_nodes:
                for node in response.source_nodes:
                    if "privacy_summary" in node.metadata:
                        privacy_meta = node.metadata["privacy_summary"]
                        pii_counts.append(privacy_meta.get("pii_count", 0))
                        removed_counts.append(privacy_meta.get("eraser_removed_count", 0))
                        risk_scores.append(privacy_meta.get("eraser_average_risk", 0.0))
            
            results["successful_queries"] += 1
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(queries)} queries...")
                
        except Exception as e:
            logger.error(f"Error processing query {i}: {e}")
            results["failed_queries"] += 1
    
    # Calculate averages
    if response_times:
        results["avg_response_time"] = sum(response_times) / len(response_times)
    if response_lengths:
        results["avg_response_length"] = sum(response_lengths) / len(response_lengths)
    
    # Aggregate NLG scores
    if nlg_scores_list:
        for key in nlg_scores_list[0].keys():
            values = [s[key] for s in nlg_scores_list if key in s]
            if values:
                results["nlg_scores"][key] = sum(values) / len(values)
    
    # Privacy stats
    if pii_counts:
        results["privacy_stats"]["total_pii_detected"] = sum(pii_counts)
    if removed_counts:
        results["privacy_stats"]["total_sentences_removed"] = sum(removed_counts)
    if risk_scores:
        results["privacy_stats"]["avg_risk_score"] = sum(risk_scores) / len(risk_scores)
    
    return results


def run_benchmark():
    """Run benchmark comparing with and without privacy module"""
    logger.info("=" * 80)
    logger.info("Privacy Module Benchmark")
    logger.info("=" * 80)
    
    cfg = Config()
    
    # Load data
    logger.info("\nLoading dataset...")
    qa_dataset = get_qa_dataset(cfg.dataset)
    test_size = min(50, len(qa_dataset['question']))  # Test on first 50 queries
    queries = qa_dataset['question'][:test_size]
    expected_answers = qa_dataset['answers'][:test_size]
    
    logger.info(f"Testing on {test_size} queries")
    
    # Setup embeddings and LLM
    embeddings = get_embedding(cfg.embeddings)
    llm = get_llm(cfg.llm)
    Settings.llm = llm
    Settings.embed_model = embeddings
    Settings.chunk_size = cfg.chunk_size
    
    # Build index
    logger.info("\nBuilding index...")
    index, hierarchical_storage_context = get_index(
        qa_dataset, 
        cfg.persist_dir, 
        split_type=cfg.split_type,
        chunk_size=cfg.chunk_size
    )
    
    # Test 1: WITHOUT privacy module
    logger.info("\n" + "=" * 80)
    logger.info("Test 1: WITHOUT Privacy Module")
    logger.info("=" * 80)
    
    cfg.enable_privacy_summary = False
    query_engine_no_privacy = RetrieverQueryEngine(
        retriever=get_retriver(cfg.retriever, index, hierarchical_storage_context=hierarchical_storage_context, cfg=cfg),
        response_synthesizer=response_synthesizer(0),
        node_postprocessors=[get_postprocessor(cfg)]
    )
    
    results_no_privacy = benchmark_query_engine(
        query_engine_no_privacy, queries, expected_answers, privacy_enabled=False
    )
    
    # Test 2: WITH privacy module
    logger.info("\n" + "=" * 80)
    logger.info("Test 2: WITH Privacy Module")
    logger.info("=" * 80)
    
    cfg.enable_privacy_summary = True
    node_postprocessors = [get_postprocessor(cfg)]
    node_postprocessors.extend(get_privacy_postprocessors(cfg))
    
    query_engine_with_privacy = RetrieverQueryEngine(
        retriever=get_retriver(cfg.retriever, index, hierarchical_storage_context=hierarchical_storage_context, cfg=cfg),
        response_synthesizer=response_synthesizer(0),
        node_postprocessors=node_postprocessors
    )
    
    results_with_privacy = benchmark_query_engine(
        query_engine_with_privacy, queries, expected_answers, privacy_enabled=True
    )
    
    # Compare results
    logger.info("\n" + "=" * 80)
    logger.info("Benchmark Results Comparison")
    logger.info("=" * 80)
    
    print("\nResponse Time:")
    print(f"  Without Privacy: {results_no_privacy['avg_response_time']:.3f}s")
    print(f"  With Privacy:    {results_with_privacy['avg_response_time']:.3f}s")
    print(f"  Overhead:         {results_with_privacy['avg_response_time'] - results_no_privacy['avg_response_time']:.3f}s")
    
    print("\nResponse Length:")
    print(f"  Without Privacy: {results_no_privacy['avg_response_length']:.0f} chars")
    print(f"  With Privacy:    {results_with_privacy['avg_response_length']:.0f} chars")
    
    print("\nPrivacy Protection:")
    print(f"  PII Detected:     {results_with_privacy['privacy_stats']['total_pii_detected']}")
    print(f"  Sentences Removed: {results_with_privacy['privacy_stats']['total_sentences_removed']}")
    print(f"  Avg Risk Score:   {results_with_privacy['privacy_stats']['avg_risk_score']:.3f}")
    
    if results_no_privacy['nlg_scores'] and results_with_privacy['nlg_scores']:
        print("\nNLG Scores (with privacy vs without):")
        for metric in results_with_privacy['nlg_scores'].keys():
            if metric in results_no_privacy['nlg_scores']:
                score_no = results_no_privacy['nlg_scores'][metric]
                score_with = results_with_privacy['nlg_scores'][metric]
                diff = score_with - score_no
                print(f"  {metric}: {score_no:.4f} -> {score_with:.4f} (Δ {diff:+.4f})")
    
    # Save results
    output_file = "benchmark_privacy_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "without_privacy": results_no_privacy,
            "with_privacy": results_with_privacy
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nResults saved to {output_file}")
    
    return results_no_privacy, results_with_privacy


if __name__ == "__main__":
    run_benchmark()

