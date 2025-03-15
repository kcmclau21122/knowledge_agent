"""
LLM Manager module for handling language model operations
"""

import logging
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer

from config import Config

logger = logging.getLogger(__name__)

class LLMManager:
    def __init__(self, model_name=None, embedding_model=None):
        """Initialize the LLM Manager with configuration parameters"""
        self.model_name = model_name or Config.LLM_MODEL
        self.embedding_model_name = embedding_model or Config.EMBEDDING_MODEL
        
        self.tokenizer = None
        self.model = None
        self.generator = None
        self.embedding_model = None
    
    def load_models(self):
        """Load the LLM and embedding models"""
        logger.info(f"Loading LLM model from {self.model_name}")
        try:
            # Check if we're using a local model
            if os.path.exists(self.model_name):
                logger.info(f"Loading local model from {self.model_name}")
            
            # Load the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Try loading with optimizations first
            try:
                logger.info("Attempting to load model with 4-bit quantization...")
                
                # Import bitsandbytes to check if it's available
                import bitsandbytes
                
                # Load with 4-bit quantization
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    load_in_4bit=True,
                    low_cpu_mem_usage=True
                )
                logger.info("Successfully loaded model with 4-bit quantization")
                
            except (ImportError, ModuleNotFoundError):
                # Fallback to half precision without 4-bit quantization
                logger.info("bitsandbytes not available, falling back to half precision without 4-bit quantization")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto" if torch.cuda.is_available() else "cpu",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    low_cpu_mem_usage=True
                )
                logger.info("Successfully loaded model with half precision")
            
            # Create a text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto" if torch.cuda.is_available() else "cpu"
            )
            
            # Load the embedding model
            try:
                logger.info(f"Loading embedding model from {self.embedding_model_name}")
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                logger.info("Knowledge agent will run without embedding functionality")
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            logger.info("Attempting to load a fallback model...")
            
            try:
                fallback_model = "facebook/opt-125m"  # Tiny model as fallback
                self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                self.model = AutoModelForCausalLM.from_pretrained(
                    fallback_model,
                    device_map="auto" if torch.cuda.is_available() else "cpu",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    low_cpu_mem_usage=True
                )
                
                self.generator = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device_map="auto" if torch.cuda.is_available() else "cpu"
                )
                
                logger.info(f"Loaded fallback model: {fallback_model}")
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")
                raise RuntimeError("Could not load any language model. Please check your internet connection or specify a smaller model.")
    
    def generate_response(self, prompt: str, max_new_tokens: int = 512, 
                         temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        Generate a response from the LLM
        
        Args:
            prompt: The input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Controls randomness (higher = more random)
            top_p: Controls diversity (higher = more diverse)
            
        Returns:
            Generated text response
        """
        if not self.generator:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        try:
            # Generate with the pipeline
            outputs = self.generator(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract the generated text
            generated_text = outputs[0]["generated_text"]
            
            # Remove the prompt from the generated text
            if generated_text.startswith(prompt):
                response = generated_text[len(prompt):].strip()
            else:
                response = generated_text.strip()
                
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
