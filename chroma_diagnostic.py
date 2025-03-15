#!/usr/bin/env python
"""
Diagnostic script to check ChromaDB functionality
"""

import logging
import sys
import os
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("chroma-diagnostic")

def check_directory_permissions(directory):
    """Check if the directory exists and has write permissions"""
    directory = Path(directory)
    
    logger.info(f"Checking directory: {directory}")
    
    if not directory.exists():
        logger.info(f"Directory doesn't exist. Attempting to create it...")
        try:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Successfully created directory: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory: {e}")
            return False
    
    # Check if directory is writable
    try:
        test_file = directory / ".write_test"
        with open(test_file, 'w') as f:
            f.write("test")
        test_file.unlink()
        logger.info(f"Directory is writable")
        return True
    except Exception as e:
        logger.error(f"Directory is not writable: {e}")
        return False

def test_embedding_function(model_name):
    """Test if the embedding function can be created"""
    logger.info(f"Testing embedding function with model: {model_name}")
    
    try:
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        
        # Test embedding with a sample text
        sample_text = "This is a test document for embedding"
        embedding = embedding_function([sample_text])
        
        logger.info(f"Successfully created embedding function and generated embeddings")
        logger.info(f"Embedding shape: {len(embedding[0])} dimensions")
        return True
    except Exception as e:
        logger.error(f"Failed to create embedding function: {e}")
        return False

def test_chromadb(db_path, embedding_model):
    """Test basic ChromaDB functionality"""
    logger.info(f"Testing ChromaDB at path: {db_path}")
    
    try:
        # Initialize embedding function
        logger.info("Initializing embedding function")
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        
        # Initialize ChromaDB client
        logger.info("Initializing ChromaDB client")
        client = chromadb.PersistentClient(path=str(db_path))
        
        # Create a test collection
        logger.info("Creating test collection")
        collection_name = "test_collection"
        
        # Try to get the collection if it exists, or create a new one
        try:
            collection = client.get_collection(name=collection_name)
            logger.info(f"Found existing collection: {collection_name}")
        except:
            collection = client.create_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            logger.info(f"Created new collection: {collection_name}")
        
        # Add a test document
        logger.info("Adding test document")
        collection.add(
            ids=["test1"],
            documents=["This is a test document for ChromaDB"],
            metadatas=[{"source": "diagnostic test"}]
        )
        
        # Query the collection
        logger.info("Querying test collection")
        results = collection.query(
            query_texts=["test document"],
            n_results=1
        )
        
        if results and len(results["documents"]) > 0:
            logger.info(f"Successfully queried collection. Results: {results}")
            
            # Get collection count
            count = collection.count()
            logger.info(f"Collection count: {count}")
            
            # List all collections
            collections = client.list_collections()
            logger.info(f"All collections: {collections}")
            
            # Get total document count across all collections
            total_count = 0
            all_collection_stats = []
            
            for coll_name in collections:
                try:
                    coll = client.get_collection(name=coll_name)
                    coll_count = coll.count()
                    total_count += coll_count
                    all_collection_stats.append({
                        "name": coll_name,
                        "count": coll_count
                    })
                except Exception as e:
                    logger.warning(f"Error getting count for collection {coll_name}: {e}")
            
            logger.info(f"Total documents across all collections: {total_count}")
            logger.info(f"Collection statistics: {all_collection_stats}")
            
            return True
        else:
            logger.error("Query returned no results")
            return False
            
    except Exception as e:
        logger.error(f"ChromaDB test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main diagnostic function"""
    from config import Config
    
    logger.info("Starting ChromaDB diagnostic")
    logger.info("=" * 50)
    
    # Check Python version
    logger.info(f"Python version: {sys.version}")
    
    # Check ChromaDB version
    logger.info(f"ChromaDB version: {chromadb.__version__}")
    
    # Get configuration
    db_dir = Config.DB_DIR
    embedding_model = Config.EMBEDDING_MODEL
    
    logger.info(f"DB directory: {db_dir}")
    logger.info(f"Embedding model: {embedding_model}")
    
    # Check directory permissions
    if not check_directory_permissions(db_dir):
        logger.error("Directory permission check failed. Fix permissions before continuing.")
        return False
    
    # Test embedding function
    if not test_embedding_function(embedding_model):
        logger.error("Embedding function test failed. Check your sentence-transformers installation.")
        return False
    
    # Test ChromaDB functionality
    if not test_chromadb(db_dir, embedding_model):
        logger.error("ChromaDB test failed. Check your ChromaDB installation and configuration.")
        return False
    
    logger.info("=" * 50)
    logger.info("ChromaDB diagnostic completed successfully!")
    logger.info("Your ChromaDB installation appears to be working correctly.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)