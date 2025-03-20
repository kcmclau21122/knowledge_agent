#!/usr/bin/env python
"""
Script to download a quantized version of Mistral-7B-Instruct using the token from config.py
"""

import os
import argparse
import logging
import sys
from pathlib import Path
from huggingface_hub import login, hf_hub_download


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Available quantization types
QUANT_TYPES = {
    "q4_k_m": {
        "description": "4-bit, medium quality (5.1GB)",
        "repo_id": "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        "filename": "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        "file_size": 5100000000,  # Approximate
    },
    "q5_k_m": {
        "description": "5-bit, higher quality (6.3GB)",
        "repo_id": "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        "filename": "mistral-7b-instruct-v0.1.Q5_K_M.gguf",
        "file_size": 6300000000,  # Approximate
    },
    "q8_0": {
        "description": "8-bit, highest quality (9.5GB)",
        "repo_id": "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        "filename": "mistral-7b-instruct-v0.1.Q8_0.gguf",
        "file_size": 9500000000,  # Approximate
    }
}

def get_token_from_config():
    try:
        sys.path.append(os.getcwd())
        from config import Config
        
        if not Config.HUGGINGFACE_TOKEN:
            logger.error("HUGGINGFACE_TOKEN is empty in Config class")
            return None
            
        return Config.HUGGINGFACE_TOKEN
    except ImportError:
        logger.error("Could not import Config from config.py")
        return None
    except Exception as e:
        logger.error(f"Error reading token from config.py: {e}")
        return None

def authenticate_huggingface(token=None):
    """
    Authenticate with Hugging Face using a token
    
    Args:
        token: Hugging Face token, if None will try to get from config
    
    Returns:
        True if authentication successful, False otherwise
    """
    if token is None:
        token = get_token_from_config()
        
    if not token:
        logger.error("No token provided and couldn't get token from config.py")
        return False
    
    try:
        login(token=token)
        logger.info("Successfully authenticated with Hugging Face")
        return True
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        return False

def download_model(output_dir, quant_type="q4_k_m", token=None):
    """
    Download the quantized Mistral-7B-Instruct model
    
    Args:
        output_dir: Directory where the model will be saved
        quant_type: Quantization type (q4_k_m, q5_k_m, q8_0)
        token: Hugging Face token for authentication
    """
    if quant_type not in QUANT_TYPES:
        raise ValueError(f"Invalid quantization type: {quant_type}. Choose from: {', '.join(QUANT_TYPES.keys())}")
    
    # First authenticate with Hugging Face
    if not authenticate_huggingface(token):
        logger.error("Authentication failed. Cannot download model.")
        return None
    
    model_info = QUANT_TYPES[quant_type]
    repo_id = model_info["repo_id"]
    filename = model_info["filename"]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Downloading {model_info['description']} model")
    logger.info(f"From: {repo_id}")
    logger.info(f"File: {filename}")
    logger.info(f"To: {output_dir}")
    logger.info("This might take a while depending on your internet connection...")
    
    try:
        # Use huggingface_hub to download the file
        download_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        
        logger.info(f"Model successfully downloaded to {download_path}")
        logger.info("\nTo use this model in your Knowledge Agent, update config.py:")
        logger.info(f'LLM_MODEL = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"')
        logger.info(f'MODEL_PATH = "{download_path}"')
        
        return download_path
        
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Download Quantized Mistral-7B-Instruct Model")
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./models",
        help="Directory where the model will be saved"
    )
    parser.add_argument(
        "--quant-type",
        type=str,
        default="q4_k_m",
        choices=QUANT_TYPES.keys(),
        help="Quantization type (affects file size and quality)"
    )
    
    args = parser.parse_args()
    
    # Check disk space before downloading
    try:
        import shutil
        disk_usage = shutil.disk_usage(args.output_dir)
        free_space_gb = disk_usage.free / (1024 ** 3)
        
        required_space_gb = QUANT_TYPES[args.quant_type]["file_size"] / (1024 ** 3) * 1.1  # Add 10% margin
        
        if free_space_gb < required_space_gb:
            logger.warning(f"WARNING: You have only {free_space_gb:.1f}GB free disk space.")
            logger.warning(f"Required space: {required_space_gb:.1f}GB")
            
            confirm = input("Continue with download? (y/n): ")
            if confirm.lower() != 'y':
                logger.info("Download cancelled by user")
                return
    except:
        logger.warning("Could not check disk space, continuing anyway")
    
    download_model(args.output_dir, args.quant_type)

if __name__ == "__main__":
    main()