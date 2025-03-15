"""
Knowledge Agent module for answering questions using the knowledge base
"""

import logging
import traceback
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from config import Config
from document_processor import DocumentProcessor
from vector_database import VectorDatabase
from llm_manager import LLMManager
from utils import create_directories

logger = logging.getLogger(__name__)

class KnowledgeAgent:
    def __init__(self, company_name: str = "Our Company"):
        """Initialize the knowledge agent"""
        self.company_name = company_name
        self.document_processor = DocumentProcessor()
        
        # Initialize directories
        create_directories()
        
        # Initialize vector database with better error handling
        try:
            logger.info("Initializing vector database")
            self.vector_db = VectorDatabase()
            logger.info("Vector database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to initialize Knowledge Agent: {e}")
        
        self.llm_manager = LLMManager()
    
    def ingest_knowledge_base(self, knowledge_dir=None):
        """Process and ingest all documents in the knowledge directory"""
        knowledge_dir = knowledge_dir or Config.KNOWLEDGE_DIR
        logger.info(f"Starting knowledge base ingestion from {knowledge_dir}")
        
        try:
            # Process documents
            documents = self.document_processor.process_knowledge_base(knowledge_dir)
            logger.info(f"Document processing completed. {len(documents)} chunks created.")
            
            if not documents:
                logger.warning("No document chunks were created. Check your knowledge directory and document processor.")
                return 0
            
            # Add to vector database with better error handling
            try:
                logger.info(f"Adding {len(documents)} document chunks to vector database")
                self.vector_db.add_documents(documents)
                
                # Verify documents were added by checking collection count
                try:
                    count = self.vector_db.collection.count()
                    logger.info(f"Vector database now contains {count} document chunks")
                except Exception as e:
                    logger.error(f"Could not verify document count: {e}")
                
                logger.info("Knowledge base ingestion complete")
                return len(documents)
            except Exception as e:
                logger.error(f"Failed to add documents to vector database: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                return 0
                
        except Exception as e:
            logger.error(f"Error during knowledge base ingestion: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
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
        
        logger.info(f"Processing query: {query}")
        
        try:
            # Retrieve relevant documents
            relevant_docs = self.vector_db.query(query)
            
            if not relevant_docs:
                logger.warning("No relevant documents found for query")
            else:
                logger.info(f"Found {len(relevant_docs)} relevant documents")
            
            # Build context from documents
            context = "\n\n".join([
                f"Source: {doc['metadata']['source']}\n{doc['text']}"
                for doc in relevant_docs
            ])
            
            # Format chat history
            chat_history_text = ""
            for message in chat_history[-5:]:  # Only use the last 5 messages
                role = message.get("role", "user")
                content = message.get("content", "")
                chat_history_text += f"<{role}>\n{content}\n</{role}>\n"
            
            # Build the prompt
            system_prompt = Config.SYSTEM_PROMPT.format(company_name=self.company_name)
            prompt = Config.CHAT_PROMPT.format(
                system_prompt=system_prompt,
                context=context,
                chat_history=chat_history_text,
                query=query
            )
            
            # Generate response
            response = self.llm_manager.generate_response(prompt)
            
            logger.info(f"Generated response for query")
            return response
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"I'm sorry, I encountered an error while processing your question. Please try again later or contact support."
    
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
            
            # Load and chunk document
            content = self.document_processor.load_document(file_path)
            if not content:
                logger.warning(f"No content extracted from {file_path}")
                return False
            
            chunks = self.document_processor.chunk_text(content)
            
            # Create document entries
            documents = []
            for i, chunk in enumerate(chunks):
                documents.append({
                    "text": chunk,
                    "metadata": {
                        "source": file_path_obj.name,
                        "chunk": i,
                        "created_at": datetime.now().isoformat()
                    }
                })
            
            # Add to vector database
            try:
                self.vector_db.add_documents(documents)
                logger.info(f"Added {len(documents)} chunks to vector database")
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
