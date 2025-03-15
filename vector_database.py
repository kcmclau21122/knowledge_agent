"""
Vector database module for storing and retrieving document embeddings
"""

import logging
from typing import List, Dict, Any
import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import traceback

from config import Config

logger = logging.getLogger(__name__)

class VectorDatabase:
    def __init__(self, db_path=None, embedding_model=None, collection_name="knowledge_base"):
        """Initialize the vector database with configuration parameters"""
        self.db_path = db_path or Config.DB_DIR
        self.embedding_model = embedding_model or Config.EMBEDDING_MODEL
        self.collection_name = collection_name
        
        # Ensure database directory exists
        os.makedirs(str(self.db_path), exist_ok=True)
        
        # Initialize embedding function with better error handling
        self.embedding_function = None
        try:
            logger.info(f"Initializing embedding function with model: {self.embedding_model}")
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model
            )
            logger.info("Embedding function initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing SentenceTransformer: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.info("Falling back to default ChromaDB embedding function")
            # We'll continue without an embedding function, ChromaDB will use its default
        
        # Initialize database with better error handling
        try:
            logger.info(f"Initializing ChromaDB at path: {str(self.db_path)}")
            self.db = chromadb.PersistentClient(path=str(self.db_path))
            logger.info("ChromaDB client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Could not initialize ChromaDB. Please check configuration and permissions: {e}")
        
        # Initialize collection
        try:
            self.collection = self._get_or_create_collection()
            logger.info(f"Using collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to get or create collection: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Could not initialize ChromaDB collection: {e}")
    
    def _get_or_create_collection(self):
        """Get or create a collection in the vector database"""
        try:
            # First try to get the collection
            logger.info(f"Attempting to get existing collection: {self.collection_name}")
            if self.embedding_function:
                return self.db.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function
                )
            else:
                return self.db.get_or_create_collection(name=self.collection_name)
        except Exception as e:
            logger.error(f"Error getting collection, attempting to create: {e}")
            # If failed, try to create the collection
            if self.embedding_function:
                return self.db.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function
                )
            else:
                return self.db.create_collection(name=self.collection_name)
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the vector database"""
        if not documents:
            logger.warning("No documents to add")
            return
        
        logger.info(f"Adding {len(documents)} documents to vector database")
        
        try:
            # Prepare data for the collection
            ids = [f"doc_{i}_{hash(doc['text'])}" for i, doc in enumerate(documents)]  # Add hash for uniqueness
            texts = [doc["text"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]
            
            # Add in batches to avoid memory issues
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                end = min(i + batch_size, len(documents))
                batch_start = i
                batch_end = end
                
                logger.info(f"Adding batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} to vector database")
                try:
                    self.collection.add(
                        ids=ids[batch_start:batch_end],
                        documents=texts[batch_start:batch_end],
                        metadatas=metadatas[batch_start:batch_end]
                    )
                    logger.info(f"Successfully added batch {i//batch_size + 1}")
                except Exception as e:
                    logger.error(f"Error adding batch {i//batch_size + 1}: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    # Continue with next batch instead of failing completely
            
            # Verify documents were added
            count = self.collection.count()
            logger.info(f"Total documents in collection after addition: {count}")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector database: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to add documents to vector database: {e}")
    
    def query(self, query_text: str, n_results: int = None) -> List[Dict[str, Any]]:
        """Query the vector database for relevant documents"""
        n_results = n_results or Config.MAX_CONTEXT_DOCS
        
        try:
            logger.info(f"Querying database with: '{query_text[:50]}...' (n_results={n_results})")
            
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            
            documents = []
            if results["documents"] and len(results["documents"]) > 0:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0] if "distances" in results else [0] * len(results["documents"][0])
                )):
                    documents.append({
                        "text": doc,
                        "metadata": metadata,
                        "relevance": 1.0 - min(distance, 1.0) if distance is not None else 1.0  # Convert distance to relevance score
                    })
                
                logger.info(f"Found {len(documents)} relevant documents")
            else:
                logger.warning("No documents found for query")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error querying vector database: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return empty list instead of raising exception
            return []
