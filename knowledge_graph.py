"""
Knowledge Graph module for representing and querying relationships between document chunks
with persistence to disk
"""

import logging
import numpy as np
import os
import json
import pickle
from typing import List, Dict, Any, Set
from pathlib import Path

from config import Config

logger = logging.getLogger(__name__)

class KnowledgeGraph:
    def __init__(self, vector_db):
        """Initialize the knowledge graph with connection to vector database"""
        self.vector_db = vector_db
        self.nodes = {}  # Maps document IDs to node data
        self.edges = {}  # Maps document IDs to related document IDs with weights
        self.built = False
        
        # Define graph storage path
        self.graph_path = Path(Config.DB_DIR) / "knowledge_graph.pkl"
        
        # Try to load existing graph
        self.load_graph()
    
    def build_graph(self, force_rebuild=False):
        """Build or rebuild the knowledge graph from vector database"""
        # Skip if already built and not forcing rebuild
        if self.built and not force_rebuild:
            logger.info("Knowledge graph already built, skipping. Use force_rebuild=True to rebuild.")
            return True
            
        logger.info("Building knowledge graph from vector database")
        
        try:
            # Get all documents from the collection
            collection_data = self.vector_db.collection.get()
            if not collection_data or 'ids' not in collection_data or not collection_data['ids']:
                logger.warning("No documents found in vector database to build graph")
                return False
            
            doc_ids = collection_data['ids']
            documents = collection_data['documents']
            metadatas = collection_data['metadatas']
            
            logger.info(f"Building graph with {len(doc_ids)} document nodes")
            
            # Reset graph data if rebuilding
            self.nodes = {}
            self.edges = {}
            
            # First pass: Create nodes
            for i, doc_id in enumerate(doc_ids):
                self.nodes[doc_id] = {
                    'content': documents[i],
                    'metadata': metadatas[i],
                    'embedding_id': i  # To reference embedding position
                }
            
            # Second pass: Create edges based on semantic similarity and metadata
            self._create_edges_from_similarity()
            
            # Third pass: Create edges based on structural relationships (sections, etc.)
            self._create_edges_from_structure()
            
            # Final pass: Normalize edge weights
            self._normalize_edge_weights()
            
            self.built = True
            logger.info(f"Knowledge graph built with {len(self.nodes)} nodes and {sum(len(edges) for edges in self.edges.values())} edges")
            
            # Save the graph to disk
            self.save_graph()
            
            return True
            
        except Exception as e:
            logger.error(f"Error building knowledge graph: {e}")
            return False
    
    def save_graph(self):
        """Save the knowledge graph to disk"""
        try:
            logger.info(f"Saving knowledge graph to {self.graph_path}")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)
            
            # Save a simplified version of the graph (nodes and edges)
            # We don't need to save the full content, just the structure
            simplified_graph = {
                'nodes': {},
                'edges': self.edges,
                'built': self.built
            }
            
            # Simplify node data to save space
            for node_id, node_data in self.nodes.items():
                simplified_graph['nodes'][node_id] = {
                    'metadata': node_data.get('metadata', {}),
                    'embedding_id': node_data.get('embedding_id', 0)
                }
            
            # Save using pickle for efficiency
            with open(self.graph_path, 'wb') as f:
                pickle.dump(simplified_graph, f)
                
            logger.info("Knowledge graph saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving knowledge graph: {e}")
            return False
    
    def load_graph(self):
        """Load the knowledge graph from disk"""
        if not os.path.exists(self.graph_path):
            logger.info(f"No existing knowledge graph found at {self.graph_path}")
            return False
            
        try:
            logger.info(f"Loading knowledge graph from {self.graph_path}")
            
            with open(self.graph_path, 'rb') as f:
                simplified_graph = pickle.load(f)
            
            # Load edges and built status
            self.edges = simplified_graph.get('edges', {})
            self.built = simplified_graph.get('built', False)
            
            # We need to reconstruct nodes with content from the vector DB
            simplified_nodes = simplified_graph.get('nodes', {})
            
            # If we have nodes in the saved graph, try to restore them
            if simplified_nodes:
                try:
                    # Get all documents from the collection to restore content
                    collection_data = self.vector_db.collection.get()
                    
                    if collection_data and 'ids' in collection_data and collection_data['ids']:
                        doc_ids = collection_data['ids']
                        documents = collection_data['documents']
                        
                        # Reconstruct nodes with content from vector DB
                        for i, doc_id in enumerate(doc_ids):
                            if doc_id in simplified_nodes:
                                node_data = simplified_nodes[doc_id]
                                # Add content from vector DB
                                node_data['content'] = documents[i]
                                self.nodes[doc_id] = node_data
                                
                        logger.info(f"Restored {len(self.nodes)} nodes with content")
                        
                    else:
                        logger.warning("Could not get collection data to restore node content")
                        # Still use the simplified nodes, even without content
                        self.nodes = simplified_nodes
                        
                except Exception as collection_error:
                    logger.error(f"Error getting collection data: {collection_error}")
                    # Still use the simplified nodes, even without content
                    self.nodes = simplified_nodes
            
            graph_size = f"{len(self.nodes)} nodes and {sum(len(edges) for edges in self.edges.values())} edges"
            logger.info(f"Knowledge graph loaded successfully: {graph_size}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading knowledge graph: {e}")
            return False
    
    def _create_edges_from_similarity(self):
        """Create edges between nodes based on vector similarity"""
        logger.info("Creating edges based on semantic similarity")
        
        # Get all doc IDs
        doc_ids = list(self.nodes.keys())
        
        # For each document, find similar documents
        for i, doc_id in enumerate(doc_ids):
            # Skip if already has edges
            if doc_id in self.edges:
                continue
            
            try:
                # Query for similar documents
                results = self.vector_db.collection.query(
                    query_texts=[self.nodes[doc_id]['content']],
                    n_results=6  # Get a few more than needed
                )
                
                # Initialize edges for this node
                self.edges[doc_id] = {}
                
                # Add edges to similar documents
                if results and 'ids' in results and results['ids'] and len(results['ids'][0]) > 0:
                    for j, similar_id in enumerate(results['ids'][0]):
                        # Skip self
                        if similar_id == doc_id:
                            continue
                        
                        # Calculate similarity score (1 - distance) and convert to weight
                        if 'distances' in results and results['distances'] and len(results['distances'][0]) > j:
                            distance = results['distances'][0][j]
                            similarity = 1.0 - min(distance, 1.0)
                        else:
                            # Fallback if distances aren't available
                            similarity = 0.7
                        
                        # Only add edge if similar enough (threshold)
                        if similarity > Config.EMBEDDING_SIMILARITY_THRESHOLD:
                            self.edges[doc_id][similar_id] = {
                                'weight': similarity,
                                'type': 'semantic'
                            }
            except Exception as e:
                logger.error(f"Error creating similarity edges for node {doc_id}: {e}")
                # Continue with next node
    
    def _create_edges_from_structure(self):
        """Create edges based on structural relationships (sections, headers, etc.)"""
        logger.info("Creating edges based on document structure")
        
        # Group nodes by source document and section
        doc_sources = {}
        doc_sections = {}
        
        for doc_id, node_data in self.nodes.items():
            metadata = node_data.get('metadata', {})
            source = metadata.get('source', '')
            section = metadata.get('section', '')
            
            # Group by source
            if source not in doc_sources:
                doc_sources[source] = []
            doc_sources[source].append(doc_id)
            
            # Group by source+section
            section_key = f"{source}:{section}"
            if section_key not in doc_sections:
                doc_sections[section_key] = []
            doc_sections[section_key].append(doc_id)
        
        # Connect nodes from same section with high weights
        for section_key, section_nodes in doc_sections.items():
            for i, doc_id in enumerate(section_nodes):
                if doc_id not in self.edges:
                    self.edges[doc_id] = {}
                
                # Connect to other nodes in same section
                for j, other_id in enumerate(section_nodes):
                    if doc_id != other_id:
                        # Closer nodes in sequence get higher weights
                        sequence_distance = abs(i - j)
                        weight = max(0.9 - (sequence_distance * 0.1), 0.6)
                        
                        self.edges[doc_id][other_id] = {
                            'weight': weight,
                            'type': 'section'
                        }
        
        # Connect adjacent chunks from the same source with medium weights
        for source, source_nodes in doc_sources.items():
            # Sort by chunk_index if available
            sorted_nodes = sorted(source_nodes, 
                                 key=lambda id: self.nodes[id].get('metadata', {}).get('chunk_index', 0))
            
            # Connect adjacent nodes
            for i in range(len(sorted_nodes) - 1):
                curr_id = sorted_nodes[i]
                next_id = sorted_nodes[i + 1]
                
                if curr_id not in self.edges:
                    self.edges[curr_id] = {}
                if next_id not in self.edges:
                    self.edges[next_id] = {}
                
                # Bidirectional edges
                self.edges[curr_id][next_id] = {
                    'weight': 0.8,
                    'type': 'adjacent'
                }
                self.edges[next_id][curr_id] = {
                    'weight': 0.8,
                    'type': 'adjacent'
                }
    
    def _normalize_edge_weights(self):
        """Normalize edge weights to prioritize stronger connections"""
        for node_id, edges in self.edges.items():
            if not edges:
                continue
            
            # Get all weights
            weights = [edge_data['weight'] for edge_data in edges.values()]
            
            # If we have enough edges, normalize to emphasize differences
            if len(weights) > 2:
                # Apply softmax-like normalization to emphasize differences
                weights = np.array(weights)
                exp_weights = np.exp((weights - np.min(weights)) * 3)  # Scale differences
                norm_weights = exp_weights / np.sum(exp_weights)
                
                # Update weights with normalized values
                for i, (edge_id, edge_data) in enumerate(edges.items()):
                    edges[edge_id]['weight'] = float(norm_weights[i])
    
    def query_with_traversal(self, initial_results: List[Dict[str, Any]], 
                            max_total_results: int = 5, max_hops: int = None) -> List[Dict[str, Any]]:
        """
        Use graph traversal to expand query results
        
        Args:
            initial_results: Initial vector query results
            max_total_results: Maximum number of results to return
            max_hops: Maximum number of hops in graph traversal
            
        Returns:
            Expanded list of results
        """
        # Use config value if not specified
        max_hops = max_hops if max_hops is not None else Config.MAX_GRAPH_HOPS

        if not self.built or not self.nodes:
            logger.warning("Knowledge graph not built or empty")
            return initial_results
        
        if not initial_results:
            return []
            
        logger.info(f"Expanding {len(initial_results)} initial results using graph traversal")
        
        # Find node IDs for initial results by matching content
        result_nodes = set()
        result_map = {}  # Map from node ID to result dict
        
        # Find the corresponding node IDs for initial results
        for result in initial_results:
            result_text = result['text']
            found = False
            
            # Look for exact match first
            for node_id, node_data in self.nodes.items():
                if 'content' in node_data and node_data['content'] == result_text:
                    result_nodes.add(node_id)
                    result_map[node_id] = result
                    found = True
                    break
            
            # If not found, try matching by embedding similarity
            if not found:
                # This would require additional query, skipping for efficiency
                pass
        
        logger.info(f"Found {len(result_nodes)} matching nodes in the knowledge graph")
        
        if not result_nodes:
            return initial_results
        
        # Start graph traversal from these nodes
        visited = result_nodes.copy()
        frontier = result_nodes.copy()
        discovered = {}  # Maps node_id to (parent_id, weight)
        
        # BFS traversal with limited hops
        for hop in range(max_hops):
            next_frontier = set()
            
            for node_id in frontier:
                if node_id not in self.edges:
                    continue
                    
                # Explore neighbors
                for neighbor_id, edge_data in self.edges[node_id].items():
                    if neighbor_id not in visited:
                        edge_weight = edge_data['weight']
                        
                        # Track discovery with highest weight path
                        if neighbor_id not in discovered or discovered[neighbor_id][1] < edge_weight:
                            discovered[neighbor_id] = (node_id, edge_weight)
                            
                        next_frontier.add(neighbor_id)
                        visited.add(neighbor_id)
            
            # Stop if no new nodes discovered
            if not next_frontier:
                break
                
            frontier = next_frontier
        
        # Calculate relevance scores for discovered nodes based on connection to initial results
        discovered_results = []
        
        for node_id, (parent_id, weight) in discovered.items():
            # Skip initial results (already included)
            if node_id in result_nodes:
                continue
                
            node_data = self.nodes[node_id]
            
            # Skip if node doesn't have content (might happen if graph was loaded without content)
            if 'content' not in node_data:
                continue
            
            # Calculate relevance based on connection strength and hops
            # Start with parent's relevance and multiply by edge weight
            parent_relevance = result_map[parent_id]['relevance'] if parent_id in result_map else 0.7
            relevance = parent_relevance * weight * 0.9  # Decay factor per hop
            
            # Create result structure
            result = {
                'text': node_data['content'],
                'metadata': node_data['metadata'],
                'relevance': relevance,
                'match_type': 'graph' 
            }
            
            discovered_results.append(result)
        
        # Combine and sort all results
        all_results = initial_results + discovered_results
        all_results.sort(key=lambda x: x['relevance'], reverse=True)
        
        # Return top results
        return all_results[:max_total_results]