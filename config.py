"""
Configuration settings for the Knowledge Agent
"""

import os
from pathlib import Path

class Config:
    # Basic settings
    APP_NAME = "Knowledge Agent"
    VERSION = "1.0.0"
    
    # Paths
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data"
    KNOWLEDGE_DIR = DATA_DIR / "knowledge"
    DB_DIR = DATA_DIR / "vectordb"
    MODEL_DIR = DATA_DIR / "models"
    
    # Web server
    HOST = "localhost"
    PORT = 8000
    
    # LLM settings
    LLM_MODEL = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
    MODEL_PATH = "./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    HUGGINGFACE_TOKEN = "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"
    
    # LLM generation settings
    CONTEXT_WINDOW_SIZE = 4096
    GENERATION_TEMPERATURE = 0.7
    GENERATION_TOP_P = 0.9
    MAX_NEW_TOKENS = 512
    
    # Retrieval settings
    MAX_CONTEXT_DOCS = 8
    MAX_CONTEXT_TOKENS = 2000  # Maximum tokens for context
    MAX_DOC_PARTS = 8  # Maximum document parts to include in context
    CHUNK_SIZE = 768
    CHUNK_OVERLAP = 50
    
    # Vector database settings
    BATCH_SIZE = 100  # Batch size for adding documents to vector database
    KEYWORD_RELEVANCE_BASE = 0.85  # Base relevance score for keyword matches
    KEYWORD_RELEVANCE_BOOST = 0.1  # Relevance boost for direct keyword matches
    MAX_KEYWORDS = 5  # Maximum number of keywords to extract from query
    EMBEDDING_SIMILARITY_THRESHOLD = 0.5  # Threshold for semantic similarity edges
    
    # Knowledge graph settings
    USE_GRAPH = True  # Whether to use knowledge graph for retrieval
    MAX_GRAPH_HOPS = 2  # Maximum number of hops in graph traversal
    
    # Prompt templates
    SYSTEM_PROMPT = """You are a helpful customer service assistant for {company_name}. 
    Your task is to provide accurate, helpful answers based on the company knowledge base.
    Only use information from the provided context.
    If you don't know the answer or it's not in the context, say so honestly.
    Be friendly and concise."""
    
    CHAT_PROMPT = """<s>[INST]
    {system_prompt}

    I'll provide you with relevant information from our knowledge base to help answer the user's question.

    KNOWLEDGE BASE INFORMATION:
    {context}

    PREVIOUS CONVERSATION:
    {chat_history}

    USER QUESTION: {query}
    [/INST]
    """

    @classmethod
    def update_config(cls, **kwargs):
        """
        Update config parameters at runtime
        """
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)