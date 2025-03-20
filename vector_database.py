"""
Vector database module for storing and retrieving document embeddings
"""

import logging
from typing import List, Dict, Any
import os
import time
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import traceback
import json
import re
import torch

from config import Config
from knowledge_graph import KnowledgeGraph  # Using the updated KnowledgeGraph

logger = logging.getLogger(__name__)

class VectorDatabase:
    def __init__(self, db_path=None, embedding_model=None, collection_name="knowledge_base", use_graph=None):
        """Initialize the vector database with configuration parameters"""
        self.db_path = db_path or Config.DB_DIR
        self.embedding_model = embedding_model or Config.EMBEDDING_MODEL
        self.collection_name = collection_name
        self.use_graph = use_graph if use_graph is not None else Config.USE_GRAPH
        
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
        
        # Initialize knowledge graph if enabled
        if use_graph:
            self.knowledge_graph = KnowledgeGraph(self)
            logger.info("Knowledge graph initialized")
    
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
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Add documents to the vector database with enhanced logging and error handling
        
        Args:
            documents: List of document dictionaries to add
        
        Returns:
            Number of documents successfully added
        """
        if not documents:
            logger.warning("No documents to add")
            return 0
        
        logger.info(f"Adding {len(documents)} documents to vector database")
        
        # Track successfully added documents
        docs_added = 0
        
        try:
            # Prepare data for the collection with batching
            batch_size = Config.BATCH_SIZE  # Moderate batch size to prevent memory issues
            
            for batch_start in range(0, len(documents), batch_size):
                batch_end = min(batch_start + batch_size, len(documents))
                batch = documents[batch_start:batch_end]
                
                try:
                    # Prepare batch data
                    ids = []
                    texts = []
                    metadatas = []
                    
                    for doc_index, doc in enumerate(batch):
                        # Generate a unique ID for each document
                        doc_id = f"doc_{batch_start + doc_index}_{hash(doc['text'])}"
                        
                        # Truncate very long texts
                        text = doc['text']
                        if len(text) > 10000:
                            text = text[:10000]
                            logger.warning(f"Truncated document {doc_id} to 10,000 characters")
                        
                        # Ensure non-empty text
                        if not text.strip():
                            logger.warning(f"Skipping empty document {doc_id}")
                            continue
                        
                        ids.append(doc_id)
                        texts.append(text)
                        
                        # Flatten and stringify metadata
                        metadata = {}
                        for key, value in doc.get('metadata', {}).items():
                            if value is not None:
                                # Convert complex types to JSON-serializable strings
                                if isinstance(value, (list, dict)):
                                    try:
                                        metadata[key] = json.dumps(value)
                                    except:
                                        metadata[key] = str(value)
                                else:
                                    metadata[key] = str(value)
                        
                        metadatas.append(metadata)
                    
                    # Skip empty batches
                    if not ids:
                        logger.warning(f"Batch {batch_start//batch_size + 1} is empty after processing")
                        continue
                    
                    # Add batch to collection
                    self.collection.add(
                        ids=ids,
                        documents=texts,
                        metadatas=metadatas
                    )
                    
                    # Update count of successfully added documents
                    docs_added += len(ids)
                    
                    logger.info(f"Added batch {batch_start//batch_size + 1}: {len(ids)} documents")
                    
                except Exception as batch_error:
                    logger.error(f"Error adding batch {batch_start//batch_size + 1}: {batch_error}")
                    logger.error(traceback.format_exc())
                    # Continue with next batch instead of failing completely
                    continue
            
            # Verify final count
            final_count = self.collection.count()
            logger.info(f"Total documents in collection after addition: {final_count}")
            
            # Rebuild knowledge graph if enabled
            if self.use_graph and hasattr(self, 'knowledge_graph'):
                try:
                    logger.info("Rebuilding knowledge graph")
                    self.knowledge_graph.build_graph(force_rebuild=True)
                except Exception as graph_error:
                    logger.error(f"Failed to rebuild knowledge graph: {graph_error}")
            
            return docs_added
        
        except Exception as e:
            logger.error(f"Unexpected error adding documents: {e}")
            logger.error(traceback.format_exc())
            return docs_added  
    
    def query(self, query_text: str, n_results: int = None, use_keywords: bool = True, use_graph: bool = True) -> List[Dict[str, Any]]:
        """
        Query the vector database for relevant documents with enhanced retrieval
        
        Args:
            query_text: The query text
            n_results: Number of results to return
            use_keywords: Whether to use keyword matching
            use_graph: Whether to use graph-based expansion
            
        Returns:
            List of relevant documents with metadata and relevance scores
        """
        n_results = n_results or Config.MAX_CONTEXT_DOCS
        
        try:
            # Clear CUDA cache after generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                time.sleep(0.5)  # Short cooldown

            logger.info(f"Querying database with: '{query_text[:50]}...' (n_results={n_results})")
            
            # Multi-strategy retrieval approach
            documents = []
            
            # Strategy 1: Keyword-based search
            if use_keywords:
                keyword_docs = self._keyword_search(query_text, n_results=n_results)
                documents.extend(keyword_docs)
            
            # Strategy 2: Vector similarity search
            vector_docs = self._vector_search(query_text, n_results=n_results)
            
            # Combine results from both strategies, removing duplicates
            all_docs = {}
            
            # Add keyword results (higher priority)
            for doc in documents:
                doc_id = hash(doc["text"])
                if doc_id not in all_docs or all_docs[doc_id]["relevance"] < doc["relevance"]:
                    all_docs[doc_id] = doc
            
            # Add vector results
            for doc in vector_docs:
                doc_id = hash(doc["text"])
                if doc_id not in all_docs or all_docs[doc_id]["relevance"] < doc["relevance"]:
                    all_docs[doc_id] = doc
            
            # Convert to list and sort by relevance
            documents = list(all_docs.values())
            documents.sort(key=lambda x: x["relevance"], reverse=True)
            
            # Strategy 3: Graph-based expansion (if enabled and available)
            if use_graph and self.use_graph and hasattr(self, 'knowledge_graph') and self.knowledge_graph.built:
                try:
                    # Take top results for graph expansion
                    top_results = documents[:min(3, len(documents))]
                    if top_results:
                        expanded_results = self.knowledge_graph.query_with_traversal(
                            top_results, 
                            max_total_results=n_results,
                            max_hops=2
                        )
                        documents = expanded_results
                except Exception as e:
                    logger.error(f"Error in graph-based expansion: {e}")
                    # Continue with existing results
            
            # Return top results
            documents = documents[:n_results]
            
            logger.info(f"Found {len(documents)} relevant documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error querying vector database: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return empty list instead of raising exception
            return []
    
    def _keyword_search(self, query_text: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Search using keyword matching in metadata with fixed operators"""
        # Extract potential keywords from query
        keywords = self._extract_query_keywords(query_text)
        
        if not keywords:
            return []
        
        documents = []
        
        # Try to find matches for each keyword
        for keyword in keywords:
            try:
                # Use $eq or $in operators instead of $contains
                where_filter = None
                
                # Try a direct document search without filters first
                results = self.collection.query(
                    query_texts=[query_text],
                    n_results=n_results
                )
                
                # Process results
                if results["documents"] and len(results["documents"][0]) > 0:
                    for i, (doc, metadata) in enumerate(zip(
                        results["documents"][0],
                        results["metadatas"][0]
                    )):
                        # Check for keyword match in the text manually
                        text_match = keyword.lower() in doc.lower()
                        
                        # Check for keyword match in metadata manually
                        metadata_match = False
                        for meta_key, meta_value in metadata.items():
                            if meta_value and keyword.lower() in str(meta_value).lower():
                                metadata_match = True
                                break
                        
                        # Only include if there's a match
                        if text_match or metadata_match:
                            # Use high base relevance for keyword matches
                            base_relevance = Config.KEYWORD_RELEVANCE_BASE
                            
                            # Get keywords back as a list
                            doc_keywords = []
                            if "keywords" in metadata and metadata["keywords"]:
                                keyword_text = metadata["keywords"]
                                if keyword_text:
                                    if keyword_text.startswith("[") and keyword_text.endswith("]"):
                                        try:
                                            doc_keywords = json.loads(keyword_text)
                                        except:
                                            doc_keywords = [k.strip() for k in keyword_text.split(",")]
                                    else:
                                        doc_keywords = [k.strip() for k in keyword_text.split(",")]
                            
                            # Manually calculate relevance boost based on keyword match quality
                            relevance_boost = 0
                            for doc_keyword in doc_keywords:
                                if keyword.lower() in doc_keyword.lower():
                                    relevance_boost = Config.KEYWORD_RELEVANCE_BOOST  # Direct keyword match
                                    break
                            
                            documents.append({
                                "text": doc,
                                "metadata": metadata,
                                "relevance": min(base_relevance + relevance_boost, 0.98),
                                "match_type": "keyword"
                            })
            except Exception as e:
                logger.error(f"Error in keyword search for '{keyword}': {e}")
                continue
        
        return documents

    
    def _vector_search(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search using vector similarity"""
        try:
            # Use smaller batch if using CUDA
            if torch.cuda.is_available():
                # Process in smaller batches if needed
                results = self.collection.query(
                    query_texts=[query_text],
                    n_results=min(n_results, 20)  # Limit initial results
                )
            else:
                results = self.collection.query(
                    query_texts=[query_text],
                    n_results=n_results
                )
        
            documents = []
            if results["documents"] and len(results["documents"][0]) > 0:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0] if "distances" in results else [0] * len(results["documents"][0])
                )):
                    # Convert distance to relevance score (0-1 scale)
                    relevance = 1.0 - min(distance, 1.0) if distance is not None else 0.7
                    
                    documents.append({
                        "text": doc,
                        "metadata": metadata,
                        "relevance": relevance,
                        "match_type": "vector"
                    })
            
            return documents
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []
    
    def _extract_query_keywords(self, query_text: str) -> List[str]:
        """Extract potential keywords from query text"""
        # Simple stopwords set for filtering
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "by", "for", "with", "about",
                    "from", "to", "of", "is", "are", "was", "were", "be", "been", "being", "have", "has",
                    "had", "do", "does", "did", "can", "could", "will", "would", "shall", "should", "may",
                    "might", "must", "what", "which", "who", "whom", "whose", "when", "where", "why", "how"}
        
        # Split into words, lowercase, and filter out short words and stopwords
        words = [w.lower() for w in re.findall(r'\b[a-zA-Z]\w+\b', query_text)]
        keywords = [w for w in words if len(w) > 3 and w not in stopwords]
        
        # Remove duplicates while preserving order
        unique_keywords = []
        seen = set()
        for kw in keywords:
            if kw not in seen:
                unique_keywords.append(kw)
                seen.add(kw)
        
        return unique_keywords[:Config.MAX_KEYWORDS]  # Return top keywords from config