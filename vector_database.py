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
import hashlib
import math
import numpy as np

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
        self.embedding_cache = {}  # Cache for query embeddings
        self.max_cache_size = 1000  # Maximum number of cached embeddings
        self.preload_embeddings = True
        if self.preload_embeddings and torch.cuda.is_available():
            # Preload common embeddings to GPU
            self._preload_common_embeddings()
        
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
    
    def _preload_common_embeddings(self):
        """Preload common embeddings to GPU memory for faster retrieval"""
        try:
            logger.info("Preloading common embeddings to GPU memory")
            
            # Common query types that might be frequently used
            common_queries = [
                "help", "about", "what is", "how to", "contact",
                "services", "price", "location", "hours"
            ]
            
            # Generate embeddings for these common queries
            if self.embedding_function:
                # This will force the model to load and generate embeddings
                self.embedding_function(common_queries)
                logger.info("Common embeddings preloaded successfully")
            else:
                logger.warning("Embedding function not available for preloading")
                
        except Exception as e:
            logger.warning(f"Failed to preload embeddings: {e}")
            # Non-critical error, just log and continue
    
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
    
    def _detect_information_type(self, query_text: str) -> str:
        """Detect the type of information being requested"""
        query_lower = query_text.lower()
        
        # Common information type patterns
        patterns = {
            "location": [
                r"\b(where|location|address|office|center|branch|site|facility)\b",
                r"\bnear\s+\w+\b",  # "near [city]"
                r"\bin\s+\w+\b"      # "in [city/location]"
            ],
            "contact": [
                r"\b(contact|phone|email|call|reach|touch)\b",
                r"\bhow\s+(can|do|to)\s+(i|we|you)*\s+(contact|reach|email)\b"
            ],
            "hours": [
                r"\b(hour|time|open|close|schedule|when)\b",
                r"\bwhen\s+(are|is|do|can|will)\b"
            ],
            "pricing": [
                r"\b(price|cost|fee|charge|payment|pay|afford|budget|expensive|cheap)\b",
                r"\bhow\s+much\b"
            ],
            "service": [
                r"\b(service|offering|provide|support|assist|help with)\b",
                r"\bdo\s+you\s+(offer|provide|have|support)\b"
            ],
            "howto": [
                r"\bhow\s+to\b",
                r"\b(step|guide|instruction|tutorial|direction|process)\b"
            ],
            "team": [
                r"\b(team|staff|employee|personnel|founder|leadership|management|CEO|executive|director)\b",
                r"\bwho\s+(is|are|works|leads|manages|runs)\b"
            ]
        }
        
        # Check each pattern category
        for info_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, query_lower):
                    return info_type
        
        return "general"

    def _specialized_retrieval(self, query_text: str, info_type: str, n_results: int) -> List[Dict[str, Any]]:
        """Perform specialized retrieval based on information type"""
        # Map of information types to optimized search queries
        type_search_queries = {
            "location": [
                "list of locations",
                "all locations",
                "centers addresses",
                "office locations",
                "where are located"
            ],
            "contact": [
                "contact information",
                "how to contact",
                "phone number email"
            ],
            "hours": [
                "hours of operation",
                "opening hours",
                "business hours",
                "when open"
            ],
            "pricing": [
                "pricing information",
                "cost details",
                "price list",
                "fee structure"
            ],
            "service": [
                "services offered",
                "what we provide",
                "available services"
            ],
            "howto": [
                "how to instructions",
                "step by step guide",
                "tutorial"
            ],
            "team": [
                "our team",
                "staff directory",
                "leadership team"
            ]
        }
        
        combined_results = []
        search_queries = type_search_queries.get(info_type, [])
        
        # Add the original query first for direct matching
        search_queries.insert(0, query_text)
        
        # Try each specialized search query
        for query in search_queries:
            try:
                # First try keyword search which is faster
                keyword_results = self._keyword_search(query, n_results=n_results)
                if keyword_results:
                    # Boost relevance for specialized queries
                    for result in keyword_results:
                        result["relevance"] = min(result.get("relevance", 0) + 0.1, 0.99)
                        result["match_type"] = f"specialized_{info_type}"
                    combined_results.extend(keyword_results)
                
                # Then try vector search
                vector_results = self._vector_search(query, n_results=n_results)
                if vector_results:
                    # Boost relevance for specialized queries
                    for result in vector_results:
                        result["relevance"] = min(result.get("relevance", 0) + 0.05, 0.95)
                        result["match_type"] = f"specialized_{info_type}"
                    combined_results.extend(vector_results)
                    
            except Exception as e:
                logger.debug(f"Error in specialized search for '{query}': {e}")
        
        # Remove duplicates while preserving order
        unique_results = []
        seen_texts = set()
        
        for result in combined_results:
            result_text = result["text"]
            if result_text not in seen_texts:
                seen_texts.add(result_text)
                unique_results.append(result)
        
        # Sort by relevance and return top results
        unique_results.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        return unique_results[:n_results] if unique_results else []

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
        Query the vector database for relevant documents with enhanced retrieval and information type detection
        """
        n_results = n_results or Config.MAX_CONTEXT_DOCS
        
        try:
            # Clear CUDA cache with reduced cooldown time
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                time.sleep(0.1)
            
            # Generate query hash for caching
            query_hash = hashlib.md5(query_text.encode()).hexdigest()
            
            # NEW: Detect information type from query
            info_type = self._detect_information_type(query_text)
            logger.info(f"Query '{query_text[:50]}...' detected as {info_type} query type")
            
            # NEW: Try specialized retrieval for specific information types first
            if info_type != "general":
                specialized_docs = self._specialized_retrieval(query_text, info_type, n_results)
                if specialized_docs:
                    logger.info(f"Found {len(specialized_docs)} documents via specialized {info_type} retrieval")
                    return specialized_docs
            
            # Initialize empty result list for standard retrieval
            documents = []
            
            # Log query information
            logger.info(f"Falling back to standard retrieval for: '{query_text[:50]}...'")
            
            # EXISTING TIERED RETRIEVAL STRATEGY
            
            # Strategy 1: Keyword-based search (fastest, run first)
            if use_keywords:
                start_time = time.time()
                keyword_docs = self._keyword_search(query_text, n_results=n_results)
                keyword_time = time.time() - start_time
                
                documents.extend(keyword_docs)
                logger.info(f"Found {len(keyword_docs)} documents via keyword search in {keyword_time:.3f}s")
                
                # If we have enough highly relevant documents from keywords, skip vector search
                highly_relevant_docs = [doc for doc in keyword_docs if doc.get('relevance', 0) > 0.85]
                if len(highly_relevant_docs) >= min(3, n_results):
                    logger.info(f"Found {len(highly_relevant_docs)} highly relevant documents from keywords, skipping vector search")
                    run_vector_search = False
                else:
                    run_vector_search = True
            else:
                run_vector_search = True  # No keywords, must use vector search
            
            # Strategy 2: Vector similarity search (if needed)
            if run_vector_search:
                # Check embedding cache first
                cached_vector_docs = None
                
                if hasattr(self, 'embedding_cache') and query_hash in self.embedding_cache:
                    cached_vector_docs = self.embedding_cache.get(query_hash)
                    logger.info("Using cached vector search results")
                
                if cached_vector_docs:
                    vector_docs = cached_vector_docs
                else:
                    start_time = time.time()
                    vector_docs = self._vector_search(query_text, n_results=n_results)
                    vector_time = time.time() - start_time
                    logger.info(f"Found {len(vector_docs)} documents via vector search in {vector_time:.3f}s")
                    
                    # Cache vector search results
                    if hasattr(self, 'embedding_cache'):
                        # Initialize cache if not already
                        if not hasattr(self, 'max_cache_size'):
                            self.max_cache_size = 1000
                            
                        # Prune cache if necessary    
                        if len(self.embedding_cache) >= self.max_cache_size:
                            self.embedding_cache.pop(next(iter(self.embedding_cache)))
                            
                        # Store in cache
                        self.embedding_cache[query_hash] = vector_docs
                
                # Add vector results to documents
                documents.extend(vector_docs)
            
            # Combine results from both strategies, removing duplicates
            all_docs = {}
            
            # First add keyword results (higher priority)
            for doc in documents:
                doc_id = hash(doc.get("text", ""))
                if doc_id not in all_docs or all_docs[doc_id]["relevance"] < doc.get("relevance", 0):
                    all_docs[doc_id] = doc
            
            # Convert back to list and sort by relevance
            documents = list(all_docs.values())
            documents.sort(key=lambda x: x.get("relevance", 0), reverse=True)
            
            # Early return for simple queries (avoid graph traversal overhead)
            if len(documents) >= n_results and documents[0].get("relevance", 0) > 0.9:
                logger.info("Found high-relevance direct matches, skipping graph traversal")
                return documents[:n_results]
            
            # Strategy 3: Graph-based expansion (if enabled and available)
            if use_graph and self.use_graph and hasattr(self, 'knowledge_graph') and self.knowledge_graph.built:
                try:
                    # Take top results for graph expansion
                    top_results = documents[:min(3, len(documents))]
                    if top_results:
                        start_time = time.time()
                        expanded_results = self.knowledge_graph.query_with_traversal(
                            top_results, 
                            max_total_results=n_results,
                            max_hops=Config.MAX_GRAPH_HOPS
                        )
                        graph_time = time.time() - start_time
                        
                        logger.info(f"Found {len(expanded_results)} documents after graph traversal in {graph_time:.3f}s")
                        documents = expanded_results
                except Exception as e:
                    logger.error(f"Error in graph-based expansion: {e}")
            
            # Return top results
            return documents[:n_results]
                
        except Exception as e:
            logger.error(f"Error querying vector database: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    def _keyword_search(self, query_text: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Search using fast keyword matching in document text and metadata
        
        Args:
            query_text: Query text to search for
            n_results: Number of results to return
            
        Returns:
            List of document dictionaries with text, metadata and relevance scores
        """
        # Extract potential keywords from query
        keywords = self._extract_query_keywords(query_text)
        
        if not keywords:
            return []
        
        documents = []
        seen_docs = set()  # Track seen documents to avoid duplicates
        
        # Try to find direct matches for each keyword
        for keyword in keywords[:3]:  # Limit to top 3 keywords for performance
            try:
                # Use direct document search for each keyword
                results = self.collection.query(
                    query_texts=[keyword],  # Use the keyword directly for better matching
                    n_results=min(n_results * 2, 10)  # Get more results initially
                )
                
                # Process results
                if results["documents"] and len(results["documents"][0]) > 0:
                    for i, (doc, metadata) in enumerate(zip(
                        results["documents"][0],
                        results["metadatas"][0]
                    )):
                        # Generate a hash for deduplication
                        doc_hash = hash(doc)
                        if doc_hash in seen_docs:
                            continue
                        
                        seen_docs.add(doc_hash)
                        
                        # Check for keyword match in the text
                        text_match = keyword.lower() in doc.lower()
                        
                        # Check for keyword match in metadata
                        metadata_match = False
                        for meta_key, meta_value in metadata.items():
                            if meta_value and keyword.lower() in str(meta_value).lower():
                                metadata_match = True
                                break
                        
                        # Only include if there's a match
                        if text_match or metadata_match:
                            # Calculate relevance score based on match quality
                            base_relevance = Config.KEYWORD_RELEVANCE_BASE
                            
                            # Extract keywords from metadata if available
                            doc_keywords = []
                            if "keywords" in metadata and metadata["keywords"]:
                                keyword_text = metadata["keywords"]
                                if keyword_text:
                                    # Handle different keyword formats (JSON array or comma-separated)
                                    if keyword_text.startswith("[") and keyword_text.endswith("]"):
                                        try:
                                            doc_keywords = json.loads(keyword_text)
                                        except:
                                            doc_keywords = [k.strip() for k in keyword_text.split(",")]
                                    else:
                                        doc_keywords = [k.strip() for k in keyword_text.split(",")]
                            
                            # Apply relevance boost based on match quality
                            relevance_boost = 0
                            
                            # Exact keyword match in document keywords (strongest signal)
                            if any(kw.lower() == keyword.lower() for kw in doc_keywords):
                                relevance_boost = Config.KEYWORD_RELEVANCE_BOOST * 1.5
                            # Partial keyword match in document keywords
                            elif any(keyword.lower() in kw.lower() for kw in doc_keywords):
                                relevance_boost = Config.KEYWORD_RELEVANCE_BOOST
                            # Exact match in document text (good signal)
                            elif re.search(r'\b' + re.escape(keyword.lower()) + r'\b', doc.lower()):
                                relevance_boost = Config.KEYWORD_RELEVANCE_BOOST * 0.8
                            # Partial match in text (weak signal)
                            elif keyword.lower() in doc.lower():
                                relevance_boost = Config.KEYWORD_RELEVANCE_BOOST * 0.5
                            
                            # Further boost if keyword appears multiple times
                            match_count = doc.lower().count(keyword.lower())
                            if match_count > 1:
                                density_boost = min(0.1, 0.02 * match_count)  # Up to +0.1 for many matches
                                relevance_boost += density_boost
                                
                            documents.append({
                                "text": doc,
                                "metadata": metadata,
                                "relevance": min(base_relevance + relevance_boost, 0.98),
                                "match_type": "keyword"
                            })
            except Exception as e:
                logger.error(f"Error in keyword search for '{keyword}': {e}")
                continue
        
        # Sort by relevance for consistent ordering
        documents.sort(key=lambda x: x["relevance"], reverse=True)
        
        # Return top results
        return documents[:n_results]

    def _vector_search(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search using vector similarity with optimized processing
        
        Args:
            query_text: Query text to search for
            n_results: Number of results to return
            
        Returns:
            List of document dictionaries with text, metadata and relevance scores
        """
        try:
            # Adjust batch size based on available memory
            if torch.cuda.is_available():
                # Get available GPU memory in MB
                available_memory = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                free_memory = available_memory - (torch.cuda.memory_allocated() / (1024 * 1024))
                
                # Adaptive batch size based on available memory
                # Use smaller batch for limited memory
                if free_memory < 1000:  # Less than 1GB free
                    batch_results = 10
                elif free_memory < 4000:  # 1-4GB free
                    batch_results = 20
                else:  # More than 4GB free
                    batch_results = 50  # Increase this for faster processing
            else:
                # CPU mode - use requested number
                batch_results = n_results
                
            # Execute vector search efficiently
            logger.debug(f"Executing vector search with batch size {batch_results}")
            results = self.collection.query(
                query_texts=[query_text],
                n_results=batch_results,
                include=["documents", "metadatas", "distances"]
            )
        
            documents = []
            if results["documents"] and len(results["documents"][0]) > 0:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0] if "distances" in results else [0] * len(results["documents"][0])
                )):
                    # Skip empty documents
                    if not doc or not doc.strip():
                        continue
                        
                    # Convert distance to relevance score (0-1 scale) with sigmoid normalization
                    # This gives better differentiation between results
                    if distance is not None:
                        # Lower distance is better, so 1 - distance gives relevance
                        # Clamp to 0-1 range
                        base_relevance = 1.0 - min(distance, 1.0)
                        
                        # Apply sigmoid normalization to enhance contrast between results
                        # Adjust the 10 and -5 constants to tune the curve
                        relevance = 1.0 / (1.0 + math.exp(-10 * (base_relevance - 0.5)))
                    else:
                        # Default relevance if no distance is provided
                        relevance = 0.7
                    
                    documents.append({
                        "text": doc,
                        "metadata": metadata,
                        "relevance": relevance,
                        "match_type": "vector"
                    })
                
                # Sort by relevance in descending order
                documents.sort(key=lambda x: x["relevance"], reverse=True)
                
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