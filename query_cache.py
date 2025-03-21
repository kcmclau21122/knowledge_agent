"""
Query cache module for caching frequent queries and responses
"""

import logging
import time
import json
import os
from pathlib import Path
import pickle
from typing import Dict, Any, List, Optional, Tuple
import hashlib

from config import Config

logger = logging.getLogger(__name__)

class QueryCache:
    def __init__(self, cache_dir=None, max_cache_size=1000, ttl=86400):
        """Initialize the query cache
        
        Args:
            cache_dir: Directory to store persistent cache
            max_cache_size: Maximum number of items in memory cache
            ttl: Time-to-live for cache items in seconds (default: 24 hours)
        """
        self.cache_dir = cache_dir or Path(Config.DATA_DIR) / "query_cache"
        self.max_cache_size = max_cache_size
        self.ttl = ttl
        self.memory_cache = {}  # In-memory cache: {query_hash: (timestamp, result)}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load existing cache on initialization
        self._load_cache()
        
    def _load_cache(self):
        """Load cache from disk to memory"""
        try:
            cache_index_path = self.cache_dir / "cache_index.pkl"
            if os.path.exists(cache_index_path):
                with open(cache_index_path, "rb") as f:
                    self.memory_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.memory_cache)} cached queries")
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            self.memory_cache = {}
    
    def _save_cache(self):
        """Save memory cache index to disk"""
        try:
            cache_index_path = self.cache_dir / "cache_index.pkl"
            with open(cache_index_path, "wb") as f:
                pickle.dump(self.memory_cache, f)
            logger.info(f"Saved cache with {len(self.memory_cache)} entries")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def _hash_query(self, query: str, context_size: int = None) -> str:
        """Generate a hash for the query
        
        Args:
            query: The query string
            context_size: Optional context size modifier for the hash
            
        Returns:
            String hash of the query
        """
        # Normalize the query - lowercase, strip whitespace
        normalized = query.lower().strip()
        
        # Create hash with optional context size
        hash_input = f"{normalized}:{context_size}" if context_size else normalized
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _cache_path(self, query_hash: str) -> Path:
        """Get the file path for a cached result"""
        return self.cache_dir / f"{query_hash}.json"
    
    def get(self, query: str, context_size: int = None) -> Optional[Dict[str, Any]]:
        """Get a cached result for a query if it exists and is valid
        
        Args:
            query: The query string
            context_size: Optional context size modifier for the cache key
            
        Returns:
            Cached result or None if not found or expired
        """
        query_hash = self._hash_query(query, context_size)
        
        # Check in-memory cache first
        if query_hash in self.memory_cache:
            timestamp, _ = self.memory_cache[query_hash]
            
            # Check if expired
            if time.time() - timestamp > self.ttl:
                logger.debug(f"Cache expired for query: {query[:50]}...")
                self.cache_misses += 1
                return None
            
            # Load the actual result from disk
            try:
                cache_path = self._cache_path(query_hash)
                if os.path.exists(cache_path):
                    with open(cache_path, "r") as f:
                        result = json.load(f)
                    self.cache_hits += 1
                    logger.debug(f"Cache hit for query: {query[:50]}...")
                    return result
            except Exception as e:
                logger.error(f"Error reading cache file: {e}")
        
        self.cache_misses += 1
        logger.debug(f"Cache miss for query: {query[:50]}...")
        return None
    
    def put(self, query: str, result: Dict[str, Any], context_size: int = None) -> None:
        """Store a result in the cache
        
        Args:
            query: The query string
            result: The result to cache
            context_size: Optional context size modifier for the cache key
        """
        query_hash = self._hash_query(query, context_size)
        timestamp = time.time()
        
        # Store in memory cache
        self.memory_cache[query_hash] = (timestamp, True)
        
        # Write to disk
        try:
            cache_path = self._cache_path(query_hash)
            with open(cache_path, "w") as f:
                json.dump(result, f)
            logger.debug(f"Cached result for query: {query[:50]}...")
        except Exception as e:
            logger.error(f"Error writing to cache: {e}")
        
        # Prune cache if necessary
        if len(self.memory_cache) > self.max_cache_size:
            self._prune_cache()
        
        # Periodically save the cache index
        if len(self.memory_cache) % 10 == 0:
            self._save_cache()
    
    def _prune_cache(self):
        """Remove oldest items from cache if it exceeds maximum size"""
        logger.info(f"Pruning cache, current size: {len(self.memory_cache)}")
        
        # Sort by timestamp (oldest first)
        sorted_cache = sorted(self.memory_cache.items(), key=lambda x: x[1][0])
        
        # Keep only the newest items up to max_cache_size
        items_to_remove = sorted_cache[:-self.max_cache_size]
        
        for query_hash, _ in items_to_remove:
            # Remove from memory cache
            del self.memory_cache[query_hash]
            
            # Remove from disk (optional)
            try:
                cache_path = self._cache_path(query_hash)
                if os.path.exists(cache_path):
                    os.remove(cache_path)
            except Exception as e:
                logger.error(f"Error removing cache file: {e}")
        
        logger.info(f"Pruned {len(items_to_remove)} items from cache")
    
    def clear(self):
        """Clear the entire cache"""
        try:
            # Clear memory cache
            self.memory_cache = {}
            
            # Clear disk cache
            for file_path in self.cache_dir.glob("*.json"):
                os.remove(file_path)
            
            # Remove index
            index_path = self.cache_dir / "cache_index.pkl"
            if os.path.exists(index_path):
                os.remove(index_path)
                
            self.cache_hits = 0
            self.cache_misses = 0
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cache_size": len(self.memory_cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": f"{hit_rate:.2f}%",
            "total_requests": total_requests
        }
