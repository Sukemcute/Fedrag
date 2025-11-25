"""
FastAPI backend for RAG Chatbot with Privacy-Aware Summary
"""
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add RAGTest to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from api.models import (
    QueryRequest, QueryResponse, HealthResponse, 
    ErrorResponse, PrivacyStats, SourceNode
)
from api.router import get_router
from config import Config
from privacy import apply_privacy_to_response

# Import RAG components
from llama_index.core import Settings
from index import get_index
from retriever import get_retriver, response_synthesizer
from process.postprocess_rerank import get_postprocessor
from llms.llm import get_llm
from embs.embedding import get_embedding
from llama_index.core.query_engine import RetrieverQueryEngine
from process.query_transform import transform_and_query


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot with Privacy Protection",
    description="Financial Q&A chatbot with privacy-aware summary module",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global variables for RAG components
cfg = None
query_engine = None
router = None


@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global cfg, query_engine, router
    
    logger.info("Starting RAG Chatbot API...")
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        cfg = Config()
        
        # Initialize router
        logger.info("Initializing query router...")
        router = get_router()
        
        # Setup embeddings and LLM
        logger.info("Setting up embeddings and LLM...")
        
        # Check embeddings path
        embeddings_path = cfg.embeddings
        if not embeddings_path or embeddings_path == "embedding path":
            logger.warning("No valid embeddings path in config, using default")
            embeddings_path = "BAAI/bge-base-en"
        
        embeddings = get_embedding(embeddings_path)
        llm = get_llm(cfg.llm)
        
        Settings.llm = llm
        Settings.embed_model = embeddings
        Settings.chunk_size = cfg.chunk_size
        
        # Build index
        logger.info("Building index...")
        index, hierarchical_storage_context = get_index(cfg.index)
        
        # Create query engine
        logger.info("Creating query engine...")
        node_postprocessors = [get_postprocessor(cfg)]
        
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
        
        logger.info("✓ RAG Chatbot API started successfully!")
        logger.info(f"  Privacy module: {'ENABLED' if cfg.privacy.get('enable_privacy_summary') else 'DISABLED'}")
        logger.info(f"  Available retrievers: {router.get_available_retrievers()}")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        logger.error(traceback.format_exc())
        raise


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "service": "RAG Chatbot API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/api/docs"
    }


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global cfg, query_engine, router
    
    if query_engine is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    privacy_enabled = False
    if cfg and hasattr(cfg, 'privacy'):
        privacy_enabled = cfg.privacy.get('enable_privacy_summary', False)
    
    available_retrievers = []
    if router:
        available_retrievers = router.get_available_retrievers()
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        privacy_enabled=privacy_enabled,
        available_retrievers=available_retrievers
    )


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Process user query through RAG pipeline with privacy protection
    """
    global cfg, query_engine, router
    
    if query_engine is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    start_time = time.time()
    
    try:
        question = request.question.strip()
        
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        logger.info(f"Processing query: '{question[:100]}...'")
        
        # Route query (MOCK - currently returns default)
        routing_decision = router.route_query(
            query=question,
            user_preference=request.retriever_type
        )
        routed_to = routing_decision["retriever_type"]
        logger.info(f"Routed to: {routed_to} (reasoning: {routing_decision['reasoning']})")
        
        # Execute query through RAG pipeline
        logger.info("Executing RAG query...")
        response = transform_and_query(question, cfg, query_engine)
        
        # Store original response
        original_answer = response.response
        
        # Apply privacy protection if requested
        privacy_stats_obj = None
        privacy_applied = False
        
        if request.apply_privacy and cfg.privacy.get('enable_privacy_summary', False):
            logger.info("Applying privacy protection...")
            response, privacy_metadata = apply_privacy_to_response(response, question, cfg)
            privacy_applied = True
            
            # Build privacy stats
            if privacy_metadata:
                privacy_stats_obj = PrivacyStats(
                    pii_detected=len(privacy_metadata.get("pii_entities", [])),
                    pii_density=privacy_metadata.get("pii_density", 0.0),
                    sentences_removed=privacy_metadata.get("eraser", {}).get("removed_count", 0),
                    average_risk=privacy_metadata.get("eraser", {}).get("average_risk", 0.0),
                    encrypted=privacy_metadata.get("encryption", {}).get("enabled", False),
                    entities=[e.get("entity_type", "") for e in privacy_metadata.get("pii_entities", [])]
                )
                
                logger.info(f"Privacy stats: PII={privacy_stats_obj.pii_detected}, "
                          f"Removed={privacy_stats_obj.sentences_removed}")
        
        # Extract source nodes
        source_nodes = []
        if hasattr(response, 'source_nodes') and response.source_nodes:
            for node in response.source_nodes[:5]:  # Top 5 sources
                source_nodes.append(SourceNode(
                    text=node.node.get_content()[:500],  # Truncate long text
                    score=node.score if hasattr(node, 'score') else 0.0,
                    metadata=node.node.metadata if hasattr(node.node, 'metadata') else {}
                ))
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Build response
        result = QueryResponse(
            question=question,
            answer=response.response,
            original_answer=original_answer if privacy_applied else None,
            privacy_applied=privacy_applied,
            privacy_stats=privacy_stats_obj,
            source_nodes=source_nodes,
            response_time=response_time,
            routed_to=routed_to
        )
        
        logger.info(f"Query completed in {response_time:.2f}s")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    global cfg
    
    stats = {
        "privacy_enabled": False,
        "total_queries": 0,  # Would track in production
        "avg_response_time": 0.0,  # Would track in production
    }
    
    if cfg and hasattr(cfg, 'privacy'):
        stats["privacy_enabled"] = cfg.privacy.get('enable_privacy_summary', False)
        stats["privacy_config"] = {
            "presidio_entities": cfg.privacy.get('presidio_entities', []),
            "eraser_enabled": cfg.privacy.get('eraser_drop_high_risk', False),
            "tenseal_enabled": cfg.privacy.get('tenseal_enabled', False),
            "flower_enabled": cfg.privacy.get('flower_enabled', False),
        }
    
    return stats


if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

