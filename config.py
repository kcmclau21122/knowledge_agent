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
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Retrieval settings
    MAX_CONTEXT_DOCS = 5
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    
    # Prompt templates
    SYSTEM_PROMPT = """You are a helpful customer service assistant for {company_name}. 
    Your task is to provide accurate, helpful answers based on the company knowledge base.
    You must ONLY use information from the provided context.
    If you don't know the answer or if it's not covered in the provided context, say so honestly
    and suggest the customer contact the company directly for more information.
    
    Use a friendly, professional tone. Be concise but thorough."""
    
    CHAT_PROMPT = """<|im_start|>system
{system_prompt}

I'll provide you with relevant information from our knowledge base to help answer the user's question.

KNOWLEDGE BASE INFORMATION:
{context}

PREVIOUS CONVERSATION:
{chat_history}
<|im_end|>

<|im_start|>user
{query}
<|im_end|>

<|im_start|>assistant
"""

    @classmethod
    def update_config(cls, **kwargs):
        """
        Update config parameters at runtime
        """
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)