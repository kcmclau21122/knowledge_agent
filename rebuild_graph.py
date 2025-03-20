#!/usr/bin/env python
"""
Script to rebuild and verify the knowledge graph
"""

import logging
import argparse
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def rebuild_knowledge_graph(force=False):
    """Rebuild the knowledge graph"""
    try:
        # Import necessary modules
        from config import Config
        from vector_database import VectorDatabase
        
        logger.info("Initializing vector database...")
        
        # Initialize vector database with graph support
        vector_db = VectorDatabase(use_graph=True)
        
        # Get collection count
        doc_count = vector_db.collection.count()
        logger.info(f"Vector database contains {doc_count} documents")
        
        if doc_count == 0:
            logger.error("No documents in the vector database. Please ingest documents first.")
            return False
        
        # Check if graph exists
        graph_path = Path(Config.DB_DIR) / "knowledge_graph.pkl"
        
        if os.path.exists(graph_path) and not force:
            logger.info(f"Knowledge graph already exists at {graph_path}")
            logger.info("Verifying graph...")
            
            # Load graph to verify it's working
            graph_loaded = vector_db.knowledge_graph.load_graph()
            
            if graph_loaded and vector_db.knowledge_graph.built:
                logger.info("Knowledge graph verified successfully")
                logger.info(f"Graph contains {len(vector_db.knowledge_graph.nodes)} nodes and approximately {sum(len(edges) for edges in vector_db.knowledge_graph.edges.values())} edges")
                return True
            else:
                logger.warning("Existing graph could not be verified. Rebuilding...")
        
        # Rebuild graph
        logger.info("Building knowledge graph...")
        success = vector_db.knowledge_graph.build_graph(force_rebuild=True)
        
        if success:
            logger.info("Knowledge graph built successfully")
            # Verify it was saved
            if os.path.exists(graph_path):
                logger.info(f"Graph saved to {graph_path}")
                return True
            else:
                logger.error(f"Graph file not found at {graph_path}")
                return False
        else:
            logger.error("Failed to build knowledge graph")
            return False
            
    except Exception as e:
        logger.error(f"Error rebuilding knowledge graph: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    parser = argparse.ArgumentParser(description="Rebuild Knowledge Graph")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if graph exists")
    args = parser.parse_args()
    
    logger.info("=" * 50)
    logger.info("KNOWLEDGE GRAPH REBUILD UTILITY")
    logger.info("=" * 50)
    
    if rebuild_knowledge_graph(args.force):
        logger.info("Knowledge graph rebuilding completed successfully")
        sys.exit(0)
    else:
        logger.error("Knowledge graph rebuilding failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
