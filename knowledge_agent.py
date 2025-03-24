"""
Knowledge Agent module for answering questions using the knowledge base
"""

import logging
import traceback
import re
import time
import torch
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import html

from config import Config
from document_processor import DocumentProcessor
from vector_database import VectorDatabase
from llm_manager import LLMManager
from utils import create_directories, log_gpu_memory
from url_detection import convert_urls_to_hyperlinks
from query_cache import QueryCache

# Import semantic processor if available
try:
    from semantic_document_processor import SemanticDocumentProcessor
    SEMANTIC_PROCESSING_AVAILABLE = True
except ImportError:
    SEMANTIC_PROCESSING_AVAILABLE = False

logger = logging.getLogger(__name__)

class KnowledgeAgent:
    def __init__(self, company_name: str = "Our Company", use_graph: bool = True):
            """Initialize the knowledge agent"""
            self.company_name = company_name
            self.use_graph = use_graph
            
            # Initialize LLM manager first since it's needed for semantic processing
            self.llm_manager = LLMManager()
            
            # Initialize document processor with LLM manager for semantic processing
            self.document_processor = DocumentProcessor(llm_manager=self.llm_manager)
            
            # Initialize query cache
            self.query_cache = QueryCache()
            
            # Initialize directories
            create_directories()
            
            # Initialize vector database with better error handling
            try:
                logger.info("Initializing vector database")
                self.vector_db = VectorDatabase(use_graph=use_graph)
                logger.info("Vector database initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize vector database: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise RuntimeError(f"Failed to initialize Knowledge Agent: {e}")
        
    def ingest_knowledge_base(self, knowledge_dir=None):
        """Process and ingest all documents in the knowledge directory with detailed logging"""
        knowledge_dir = knowledge_dir or Config.KNOWLEDGE_DIR
        logger.info(f"Starting knowledge base ingestion from {knowledge_dir}")
        
        try:
            # First load all models to ensure LLM is available for semantic processing
            self.initialize()
            
            # Process documents with semantic chunking
            logger.info("Starting document processing")
            documents = self.document_processor.process_knowledge_base(knowledge_dir)
            
            # Detailed logging about document processing
            logger.info(f"Document processing completed. Total chunks created: {len(documents)}")
            
            if not documents:
                logger.warning("No document chunks were created. Check your knowledge directory and document processor.")
                return 0
            
            # Log a summary of documents
            logger.info("Document processing summary:")
            source_counts = {}
            for doc in documents:
                source = doc.get('metadata', {}).get('source', 'Unknown')
                source_counts[source] = source_counts.get(source, 0) + 1
            
            for source, count in source_counts.items():
                logger.info(f"  {source}: {count} chunks")
            
            # Add to vector database with logging
            logger.info("Starting vector database ingestion")
            docs_added = self.vector_db.add_documents(documents)
            
            logger.info(f"Document ingestion complete. Total documents added: {docs_added}")
            return docs_added
        
        except Exception as e:
            logger.error(f"Comprehensive error during knowledge base ingestion: {e}")
            logger.error(f"Detailed Traceback: {traceback.format_exc()}")
            return 0
    
    def initialize(self):
        """Initialize the knowledge agent by loading the LLM models"""
        logger.info("Initializing knowledge agent")
        try:
            self.llm_manager.load_models()
            logger.info("Knowledge agent initialized")
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to initialize models: {e}")
    
    def answer_question(self, query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        Answer a question using the knowledge base and LLM
        
        Args:
            query: The user's question
            chat_history: List of previous chat messages
            
        Returns:
            Generated response text
        """
        if chat_history is None:
            chat_history = []
            
        # Check if query is in cache (only for queries without chat history)
        if not chat_history:
            cached_response = self.query_cache.get(query)
            if cached_response:
                logger.info(f"Cache hit: Retrieved response for query: {query[:50]}...")
                return cached_response.get('response', '')
            
        # Clear CUDA cache before processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            time.sleep(0.1)  # Short cooldown    
        
        logger.info(f"Processing query: {query}")
        
        try:
            # Analyze query to determine important keywords
            query_keywords = self._extract_query_keywords(query)
            logger.info(f"Extracted keywords from query: {query_keywords}")
            
            # Retrieve relevant documents with enhanced retrieval
            relevant_docs = self.vector_db.query(
                query_text=query,
                n_results=Config.MAX_CONTEXT_DOCS,
                use_keywords=True,
                use_graph=self.use_graph
            )
            
            if not relevant_docs:
                logger.warning("No relevant documents found for query")
            else:
                logger.info(f"Found {len(relevant_docs)} relevant documents")
                
                # Log relevance scores for debugging
                for i, doc in enumerate(relevant_docs):
                    relevance = doc.get('relevance', 0)
                    match_type = doc.get('match_type', 'Unknown')
                    source = doc.get('metadata', {}).get('source', 'Unknown')
                    logger.info(f"  Doc {i}: Relevance={relevance:.2f}, Type={match_type}, Source={source}")
            
            # Build enhanced context from documents with source and summaries
            context_parts = []
            total_tokens = 0
            max_tokens = 1500  # Leave room for the query, system prompt, etc.
            
            # Sort documents by relevance to ensure most relevant are included first
            relevant_docs = sorted(relevant_docs, key=lambda x: x.get('relevance', 0), reverse=True)
            
            for doc in relevant_docs:
                # Format context entry
                metadata = doc.get('metadata', {})
                source = metadata.get('source', 'Unknown')
                section = metadata.get('section', '')
                summary = metadata.get('summary', '')
                
                context_entry = f"Source: {source}"
                if section:
                    context_entry += f" | Section: {section}"
                if summary:
                    context_entry += f"\nSummary: {summary}"
                
                context_entry += f"\nContent: {doc['text']}"
                
                # Rough token count estimate (approx. 4 chars per token)
                entry_tokens = len(context_entry) // 4
                
                # Skip if adding this would exceed our token budget
                if total_tokens + entry_tokens > max_tokens:
                    logger.info(f"Skipping document due to token limit: {source}")
                    continue
                    
                context_parts.append(context_entry)
                total_tokens += entry_tokens
                
                # Check if we've added enough context
                if len(context_parts) >= 4:  # Limit to 4 documents maximum
                    logger.info(f"Stopping at {len(context_parts)} documents to stay within token limit")
                    break
            
            # Join all context parts
            context = "\n\n---\n\n".join(context_parts)
            logger.info(f"Final context size (approximate tokens): {total_tokens}")
                            
            # Format chat history - use up to 5 most recent messages for context
            chat_history_text = ""
                        
            # Filter out messages with empty content
            filtered_history = [msg for msg in chat_history if msg.get("content", "").strip()]
            
            # Format recent messages
            for message in filtered_history[-5:]:
                role = message.get("role", "user")
                content = message.get("content", "")
                # Skip empty messages
                if not content.strip():
                    continue
                chat_history_text += f"<{role}>\n{content}\n</{role}>\n"
            
            # Build the prompt with simplified formatting for Mistral model
            system_prompt = Config.SYSTEM_PROMPT.format(company_name=self.company_name)
                        
            # Format for Mistral GGUF model
            if any(term in query.lower() for term in ["location", "where", "in", "city", "state"]):
                prompt = f"""<s>[INST]
                {system_prompt}

                SPECIAL INSTRUCTION: This is a question about {self.company_name}'s physical locations. ONLY use information about where {self.company_name} has physical facilities or offices. If any information about legislation, studies, or statistics appears in the context, completely ignore it.

                {chat_history_text if chat_history_text else ""}
                
                USER QUESTION: {query}

                KNOWLEDGE BASE INFORMATION:
                {context}
                [/INST]
                """
            else:    
                prompt = Config.CHAT_PROMPT.format(
                    system_prompt=system_prompt,
                    context=context,
                    chat_history=chat_history_text,
                    query=query
                )

            # Generate response
            try:
                response = self.llm_manager.generate_response(prompt)  # Use default values from config
                
                logger.info(f"Generated response for query (length: {len(response)})")
                
                # Ensure we have a valid response
                if not response or len(response.strip()) == 0:
                    logger.warning("Empty response generated, providing fallback")
                    response = "I'm sorry, I couldn't find specific information about that in my knowledge base. Please try rephrasing your question or ask about something else."
                
                # IMPORTANT: Clean the response to remove any leaked system prompt or context
                # This uses regex patterns to remove unwanted content
                import re
                
                # Clean response by removing system prompt, instructions, and context
                # List of patterns that might indicate system content
                system_patterns = [
                    # Match Mistral/Llama instruction pattern at the start of response only
                    r"^\s*\[INST\].*?\[/INST\]\s*",
                    
                    # Match system prompt sections only if they appear at the start
                    r"^\s*You are a helpful customer service assistant.*?(?=\n\n)",
                    
                    # Match knowledge base header but not content
                    r"^\s*KNOWLEDGE BASE INFORMATION:\s*\n",
                    
                    # Only clean source/content markers at the beginning
                    r"^\s*Source:.*?\n\nContent:",
                    
                    # Only remove instruction headers
                    r"^\s*IMPORTANT INSTRUCTIONS?:\s*\n",
                    
                    # Only remove user question prefix
                    r"^\s*USER QUESTION:\s*.*?\n\n"
                ]

                # Apply cleanup patterns
                cleaned_response = response
                for pattern in system_patterns:
                    match = re.search(pattern, cleaned_response, re.DOTALL | re.IGNORECASE)
                    if match:
                        # Only remove matches that occur at the beginning
                        start_pos = match.start()
                        if start_pos <= 20:  # Allow for some whitespace
                            cleaned_response = cleaned_response[match.end():].strip()

                # Log if significant cleaning occurred
                if len(cleaned_response) < len(response) * 0.9:  # If we removed more than 10%
                    logger.info(f"Cleaned response text from {len(response)} to {len(cleaned_response)} characters")
                    # Add a safety check - if too much was removed, keep original
                    if len(cleaned_response) < 50 and len(response) > 200:
                        logger.warning("Response was over-cleaned, reverting to original")
                        cleaned_response = response

                # Process the cleaned response to convert URLs to hyperlinks
                # URL regex pattern - matches common URL formats
                url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/[-\w%!./?=&#+:~]*)*'
                
                # Function to replace each URL with hyperlink
                def replace_url(match):
                    url = match.group(0)
                    # Escape HTML entities in the URL to prevent injection
                    safe_url = html.escape(url)
                    return f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer">{safe_url}</a>'
                
                # Replace URLs with hyperlinks in the cleaned response
                response_with_links = re.sub(url_pattern, replace_url, cleaned_response)
                                    
                # Cache the cleaned, link-enhanced response
                if not chat_history:
                    self.query_cache.put(query, {"response": response_with_links})
                    
                log_gpu_memory()  # Log after model generation
                return response_with_links

            except Exception as e:
                logger.error(f"Error generating response: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                return "I'm sorry, I encountered an error while processing your question. Please try again later or contact support."
                
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"I'm sorry, I encountered an error while processing your question. Please try again later or contact support."
    
    
    def _extract_query_keywords(self, query: str) -> List[str]:
        """Extract key terms from the query for better retrieval"""
        # Simple stopwords set
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "by", "for", "with", "about",
                   "from", "to", "of", "is", "are", "was", "were", "be", "been", "being", "have", "has",
                   "had", "do", "does", "did", "can", "could", "will", "would", "shall", "should", "may",
                   "might", "must", "what", "which", "who", "whom", "whose", "when", "where", "why", "how"}
        
        # Tokenize and filter
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9]+\b', query.lower())
        keywords = [word for word in words if word not in stopwords and len(word) > 3]
        
        # Return unique keywords
        unique_keywords = []
        seen = set()
        for kw in keywords:
            if kw not in seen:
                unique_keywords.append(kw)
                seen.add(kw)
        
        return unique_keywords[:Config.MAX_KEYWORDS]  # Top keywords from config
    
    def add_document(self, file_path: str) -> bool:
        """
        Add a single document to the knowledge base
        
        Args:
            file_path: Path to the document file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Print filename with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"[{timestamp}] Adding document: {file_path}")
            
            # Get file path as Path object
            file_path_obj = Path(file_path)
            
            # First ensure LLM is loaded for semantic processing
            if not hasattr(self.llm_manager, 'model') or self.llm_manager.model is None:
                logger.info("Loading LLM for semantic document processing")
                self.llm_manager.load_models()
            
            # Load document content
            content = self.document_processor.load_document(file_path)
            if not content:
                logger.warning(f"No content extracted from {file_path}")
                return False
            
            # Process document with semantic chunking and LLM enhancement
            document_chunks = self.document_processor.process_text(content, file_path_obj.name)
            
            if not document_chunks:
                logger.warning(f"No chunks created from {file_path}")
                return False
            
            # Add to vector database
            try:
                self.vector_db.add_documents(document_chunks)
                logger.info(f"Added {len(document_chunks)} enhanced chunks to vector database")
            except Exception as e:
                logger.error(f"Failed to add document chunks to vector database: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                return False
            
            # Move the file to the Ingested folder
            self.document_processor.move_to_ingested(file_path_obj)
            
            logger.info(f"Document {file_path} added successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document {file_path}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
