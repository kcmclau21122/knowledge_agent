#!/usr/bin/env python
"""
Detailed ChromaDB Diagnostic Script
"""

import logging
import sys
import os
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions
import traceback
import shutil

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('chroma_diagnostic_detailed.log', mode='w')
    ]
)
logger = logging.getLogger("chroma-diagnostic")

def detailed_chromadb_test(db_path, embedding_model):
    """Comprehensive ChromaDB functionality test"""
    logger.info("Starting detailed ChromaDB diagnostic")
    
    try:
        # Detailed Path and Permission Check
        logger.info(f"Checking database path: {db_path}")
        logger.info(f"Absolute path: {Path(db_path).resolve()}")
        
        if not os.path.exists(db_path):
            logger.info(f"Creating database directory: {db_path}")
            os.makedirs(db_path, exist_ok=True)
        
        # Check writability with detailed logging
        try:
            test_file = os.path.join(db_path, '.write_test')
            with open(test_file, 'w') as f:
                f.write('write test')
            os.remove(test_file)
            logger.info("Directory is writable")
        except Exception as write_error:
            logger.error(f"Directory write test failed: {write_error}")
            logger.error(traceback.format_exc())
            return False
        
        # Embedding Function Test
        logger.info("Testing Embedding Function")
        try:
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )
            
            # Test embedding generation
            test_texts = ["This is a test document", "Another test sentence"]
            embeddings = embedding_function(test_texts)
            
            logger.info("Embedding generation successful")
            logger.info(f"Embedding dimensions: {len(embeddings[0])}")
        except Exception as embed_error:
            logger.error(f"Embedding function test failed: {embed_error}")
            logger.error(traceback.format_exc())
            return False
        
        # ChromaDB Client Initialization Test
        logger.info("Testing ChromaDB Client Initialization")
        try:
            client = chromadb.PersistentClient(path=str(db_path))
            logger.info("ChromaDB client initialized successfully")
        except Exception as client_error:
            logger.error(f"ChromaDB client initialization failed: {client_error}")
            logger.error(traceback.format_exc())
            return False
        
        # Collection Creation and Management Test
        logger.info("Testing Collection Operations")
        try:
            # Create a test collection, deleting it first if it exists
            collection_name = "diagnostic_test_collection"
            
            # Remove existing collection if it exists
            try:
                client.delete_collection(name=collection_name)
                logger.info(f"Deleted existing collection '{collection_name}'")
            except:
                # Collection doesn't exist, which is fine
                pass
            
            # Create the collection
            collection = client.create_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            logger.info(f"Collection '{collection_name}' created successfully")
            
            # Add test documents
            test_docs = [
                "This is the first test document about machine learning.",
                "Machine learning is a subset of artificial intelligence.",
                "Artificial intelligence is transforming many industries."
            ]
            
            collection.add(
                ids=[f"doc_{i}" for i in range(len(test_docs))],
                documents=test_docs,
                metadatas=[{"source": "diagnostic"} for _ in test_docs]
            )
            logger.info("Test documents added successfully")
            
            # Verify document count
            doc_count = collection.count()
            logger.info(f"Document count in collection: {doc_count}")
            
            # Perform a query test
            query_results = collection.query(
                query_texts=["machine learning"],
                n_results=2
            )
            
            logger.info("Query test completed successfully")
            logger.info(f"Query results: {query_results}")
            
        except Exception as collection_error:
            logger.error(f"Collection operations test failed: {collection_error}")
            logger.error(traceback.format_exc())
            return False
        
        logger.info("All ChromaDB tests passed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Unexpected error in detailed ChromaDB test: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Main diagnostic entry point"""
    from config import Config
    
    logger.info("=" * 50)
    logger.info("DETAILED CHROMADB DIAGNOSTIC")
    logger.info("=" * 50)
    
    db_path = Config.DB_DIR
    embedding_model = Config.EMBEDDING_MODEL
    
    logger.info(f"Database Path: {db_path}")
    logger.info(f"Embedding Model: {embedding_model}")
    
    success = detailed_chromadb_test(str(db_path), embedding_model)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()