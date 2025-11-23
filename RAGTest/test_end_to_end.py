"""
End-to-end test script: Upstream FedE -> Downstream RAG with Privacy Module
Tests the complete pipeline from federated embedding training to RAG with privacy protection.
"""
import os
import sys
import logging
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / "FedE"))

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_end_to_end_pipeline():
    """Test complete pipeline: FedE embedding -> RAG retrieval -> Privacy protection -> LLM generation"""
    
    logger.info("=" * 80)
    logger.info("End-to-End Pipeline Test: FedE -> RAG -> Privacy -> LLM")
    logger.info("=" * 80)
    
    # Load configuration
    cfg = Config()
    
    # Step 1: Load embeddings (from FedE trained model or default)
    logger.info("\n[Step 1] Loading embeddings...")
    if hasattr(cfg, 'fedrag_model_path') and cfg.fedrag_model_path:
        embeddings = get_embedding(cfg.fedrag_model_path)
        logger.info(f"Using FedE trained model: {cfg.fedrag_model_path}")
    else:
        embeddings = get_embedding(cfg.embeddings)
        logger.info(f"Using default embedding model: {cfg.embeddings}")
    
    # Step 2: Load LLM
    logger.info("\n[Step 2] Loading LLM...")
    llm = get_llm(cfg.llm)
    Settings.llm = llm
    Settings.embed_model = embeddings
    Settings.chunk_size = cfg.chunk_size
    
    # Step 3: Build index from dataset
    logger.info("\n[Step 3] Building index from dataset...")
    qa_dataset = get_qa_dataset(cfg.dataset)
    index, hierarchical_storage_context = get_index(
        qa_dataset, 
        cfg.persist_dir, 
        split_type=cfg.split_type,
        chunk_size=cfg.chunk_size
    )
    logger.info(f"Index built with {len(qa_dataset)} documents")
    
    # Step 4: Setup query engine with privacy module
    logger.info("\n[Step 4] Setting up query engine with privacy protection...")
    node_postprocessors = [get_postprocessor(cfg)]
    privacy_postprocessors = get_privacy_postprocessors(cfg)
    node_postprocessors.extend(privacy_postprocessors)
    
    query_engine = RetrieverQueryEngine(
        retriever=get_retriver(
            cfg.retriever, 
            index, 
            hierarchical_storage_context=hierarchical_storage_context,
            cfg=cfg
        ),
        response_synthesizer=response_synthesizer(0),
        node_postprocessors=node_postprocessors
    )
    
    privacy_enabled = cfg.enable_privacy_summary if hasattr(cfg, 'enable_privacy_summary') else False
    logger.info(f"Privacy module enabled: {privacy_enabled}")
    logger.info(f"Total postprocessors: {len(node_postprocessors)}")
    
    # Step 5: Test queries
    logger.info("\n[Step 5] Testing queries...")
    test_queries = [
        qa_dataset['question'][0] if len(qa_dataset['question']) > 0 else "What is the main topic?",
    ]
    
    if len(qa_dataset['question']) > 1:
        test_queries.append(qa_dataset['question'][1])
    
    for i, query in enumerate(test_queries[:2], 1):  # Test first 2 queries
        logger.info(f"\n--- Test Query {i} ---")
        logger.info(f"Query: {query}")
        
        try:
            response = query_engine.query(query)
            logger.info(f"Response: {response.response[:200]}...")  # First 200 chars
            
            # Check privacy metadata
            if response.source_nodes:
                for j, node in enumerate(response.source_nodes[:1]):  # Check first node
                    if "privacy_summary" in node.metadata:
                        privacy_meta = node.metadata["privacy_summary"]
                        logger.info(f"Privacy metadata on node {j}:")
                        logger.info(f"  - PII detected: {privacy_meta.get('pii_count', 0)}")
                        logger.info(f"  - Sentences removed: {privacy_meta.get('eraser_removed_count', 0)}")
                        logger.info(f"  - Average risk: {privacy_meta.get('eraser_average_risk', 0.0):.3f}")
                        logger.info(f"  - Encrypted: {privacy_meta.get('tenseal_encrypted', False)}")
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
    
    logger.info("\n" + "=" * 80)
    logger.info("End-to-End Test Completed!")
    logger.info("=" * 80)
    
    return True


if __name__ == "__main__":
    test_end_to_end_pipeline()

