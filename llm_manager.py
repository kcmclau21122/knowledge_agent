"""
LLM Manager module for handling language model operations
"""

import logging
import torch
import os
import time
import traceback 
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from utils import log_gpu_memory
from llama_cpp import Llama  # This is the new import for GGUF models

from config import Config

logger = logging.getLogger(__name__)

class LLMManager:
    def __init__(self, model_name=None, embedding_model=None, model_path=None):
        """Initialize the LLM Manager with configuration parameters"""
        self.model_name = model_name or Config.LLM_MODEL
        self.model_path = model_path or Config.MODEL_PATH
        self.embedding_model_name = embedding_model or Config.EMBEDDING_MODEL
        
        self.model = None
        self.embedding_model = None
    
    def load_models(self):
        """Load the LLM and embedding models with optimized performance settings"""
        logger.info(f"Loading GGUF model from {self.model_path}")
        log_gpu_memory()  # Log before model loading

        if not os.path.exists(self.model_path):
            logger.error(f"Model file not found at {self.model_path}")
            logger.info("Please run download_gguf_from_config.py to download the model")
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        try:
            # Performance optimizations for GGUF models
            n_gpu_layers = -1  # Use all GPU layers
            n_ctx = Config.CONTEXT_WINDOW_SIZE
            
            # Initialize the GGUF model with optimized parameters
            self.model = Llama(
                model_path=self.model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                verbose=False,
                n_batch=1024,  # Increase batch size for faster processing
                f16_kv=True,  # Use half-precision for key/value cache
                use_mlock=True  # Pin memory to prevent paging
            )
            
            logger.info("GGUF model loaded successfully with performance optimizations")
            
            # Load the embedding model (unchanged)
            try:
                logger.info(f"Loading embedding model from {self.embedding_model_name}")
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                
                # Optimize embedding model if using CUDA
                if torch.cuda.is_available():
                    self.embedding_model = self.embedding_model.to('cuda')
                    logger.info("Moved embedding model to CUDA")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                logger.info("Knowledge agent will run without embedding functionality")
            
        except Exception as e:
            logger.error(f"Error loading GGUF model: {str(e)}")
            logger.error(traceback.format_exc())
            # Fallback handling code...

    def generate_response(self, prompt: str, max_new_tokens: int = None, 
                        temperature: float = None, top_p: float = None) -> str:
        # Use values from Config if not provided
        max_new_tokens = max_new_tokens or Config.MAX_NEW_TOKENS
        temperature = temperature or Config.GENERATION_TEMPERATURE
        top_p = top_p or Config.GENERATION_TOP_P

        """Generate a response using the GGUF model with optimized parameters"""
        log_gpu_memory()  # Log before generation

        if not self.model:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        try:
            # Generate with llama-cpp with performance optimizations
            output = self.model.create_completion(
                prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=["</s>", "<|im_end|>"],  # Common stop tokens for Mistral
                echo=False,
                top_k=40,  # Limit vocabulary search space
                repeat_penalty=1.1  # Slightly penalize repetition for faster completion
            )
            
            # Extract the generated text
            response = output["choices"][0]["text"].strip()
            
            # Log an excerpt of the response for debugging
            response_preview = response[:50] + "..." if len(response) > 50 else response
            logger.debug(f"Generated response preview: {response_preview}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble generating a response at the moment. Please try again."

    
    def get_embeddings(self, texts):
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        if not self.embedding_model:
            raise ValueError("Embedding model not loaded. Call load_models() first.")
            
        return self.embedding_model.encode(texts)