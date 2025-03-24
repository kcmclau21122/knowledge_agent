#!/usr/bin/env python
"""
Script to download GGUF models based on the settings in config.py
"""

import os
import argparse
import logging
import sys
import re
from pathlib import Path
from huggingface_hub import login, hf_hub_download


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add current directory to sys.path to ensure we can import config
sys.path.append(os.getcwd())

# Try to import Config from config.py
try:
    from config import Config
    logger.info("Successfully imported Config from config.py")
except ImportError:
    logger.error("Could not import Config from config.py")
    sys.exit(1)
except Exception as e:
    logger.error(f"Error importing Config from config.py: {e}")
    sys.exit(1)

def get_model_info_from_config():
    """
    Extract model information from Config class
    
    Returns:
        Dict with model information
    """
    try:
        # Get model name from Config
        repo_id = Config.LLM_MODEL

        # Get filename from MODEL_PATH or derive it if needed
        model_path = Config.MODEL_PATH
        filename = os.path.basename(model_path)
        
        # Determine quantization type from filename (if available)
        quant_match = re.search(r'Q(\d+)(_K)?(_M)?', filename)
        quant_type = quant_match.group(0).lower() if quant_match else "unknown"
        
        logger.info(f"Model info from config: repo_id={repo_id}, filename={filename}")
        
        return {
            "repo_id": repo_id,
            "filename": filename,
            "quant_type": quant_type
        }
    except AttributeError as e:
        logger.error(f"Missing required attributes in Config: {e}")
        return None
    except Exception as e:
        logger.error(f"Error extracting model info from Config: {e}")
        return None

def build_quant_types():
    """
    Build the quantization types dictionary based on config settings
    
    Returns:
        Dict with quantization types information
    """
    model_info = get_model_info_from_config()
    if not model_info:
        logger.error("Could not get model info from config")
        return None
    
    # Extract repo parts (username/model-name)
    repo_parts = model_info["repo_id"].split('/')
    if len(repo_parts) != 2:
        logger.warning(f"Unexpected repository format: {model_info['repo_id']}")
        model_name = "model"
    else:
        # Extract model name (e.g., "Mistral-7B-Instruct-v0.1" from filename)
        model_name_match = re.match(r'([a-zA-Z0-9-]+)', model_info["filename"])
        model_name = model_name_match.group(1) if model_name_match else "model"
    
    # Determine file sizes based on quantization
    # These are rough estimates that can be adjusted
    file_sizes = {
        "q4_k_m": 5100000000,  # ~5.1GB
        "q5_k_m": 6300000000,  # ~6.3GB
        "q8_0": 9500000000     # ~9.5GB
    }
    
    # Default to the specific file in the config
    quant_types = {
        "default": {
            "description": f"Config default (from {model_info['filename']})",
            "repo_id": model_info["repo_id"],
            "filename": model_info["filename"],
            "file_size": file_sizes.get(model_info["quant_type"].lower(), 6000000000)  # Default size guess
        }
    }
    
    # Try to add other common quantization options for the same model
    # This is optional and can be removed if not needed
    if "mistral" in model_info["repo_id"].lower() and "instruct" in model_info["repo_id"].lower():
        # For Mistral models, add common quantization types
        quant_types.update({
            "q4_k_m": {
                "description": "4-bit, medium quality (5.1GB)",
                "repo_id": model_info["repo_id"],
                "filename": "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                "file_size": 5100000000,
            },
            "q5_k_m": {
                "description": "5-bit, higher quality (6.3GB)",
                "repo_id": model_info["repo_id"],
                "filename": "mistral-7b-instruct-v0.1.Q5_K_M.gguf",
                "file_size": 6300000000,
            },
            "q8_0": {
                "description": "8-bit, highest quality (9.5GB)",
                "repo_id": model_info["repo_id"],
                "filename": "mistral-7b-instruct-v0.1.Q8_0.gguf",
                "file_size": 9500000000,
            }
        })
    elif "llama" in model_info["repo_id"].lower():
        # For Llama models, add common quantization types
        model_variant = "llama-2-7b"
        if "13b" in model_info["repo_id"].lower():
            model_variant = "llama-2-13b"
        
        quant_types.update({
            "q4_k_m": {
                "description": "4-bit, medium quality",
                "repo_id": model_info["repo_id"],
                f"filename": f"{model_variant}-chat.Q4_K_M.gguf",
                "file_size": 5100000000,
            },
            "q5_k_m": {
                "description": "5-bit, higher quality",
                "repo_id": model_info["repo_id"],
                f"filename": f"{model_variant}-chat.Q5_K_M.gguf",
                "file_size": 6300000000,
            }
        })
    
    return quant_types

def get_token_from_config():
    """
    Get Hugging Face token from Config
    
    Returns:
        Token string or None
    """
    try:
        if not hasattr(Config, 'HUGGINGFACE_TOKEN'):
            logger.error("HUGGINGFACE_TOKEN not found in Config class")
            return None
            
        if not Config.HUGGINGFACE_TOKEN:
            logger.error("HUGGINGFACE_TOKEN is empty in Config class")
            return None
            
        return Config.HUGGINGFACE_TOKEN
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

def download_model(output_dir, quant_type="default", token=None):
    """
    Download the model based on config settings
    
    Args:
        output_dir: Directory where the model will be saved
        quant_type: Quantization type (from QUANT_TYPES dictionary)
        token: Hugging Face token for authentication
    """
    # Get quantization types based on config
    quant_types = build_quant_types()
    if not quant_types:
        logger.error("Failed to build quantization types")
        return None
    
    if quant_type not in quant_types:
        logger.error(f"Invalid quantization type: {quant_type}. Choose from: {', '.join(quant_types.keys())}")
        return None
    
    # First authenticate with Hugging Face
    if not authenticate_huggingface(token):
        logger.error("Authentication failed. Cannot download model.")
        return None
    
    model_info = quant_types[quant_type]
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
            cache_dir=output_dir,
            local_files_only=False
        )
        
        logger.info(f"Model successfully downloaded to {download_path}")
        logger.info("\nTo use this model in your Knowledge Agent, update config.py:")
        logger.info(f'LLM_MODEL = "{repo_id}"')
        logger.info(f'MODEL_PATH = "{download_path}"')
        
        return download_path
        
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return None

def main():
    # Get quantization types
    quant_types = build_quant_types()
    if not quant_types:
        logger.error("Failed to determine model information from config.py")
        sys.exit(1)
    
    # Print available quantization types
    logger.info("Available quantization types:")
    for qt, info in quant_types.items():
        logger.info(f"  {qt}: {info['description']}")
    
    parser = argparse.ArgumentParser(description="Download GGUF Model Based on Config")
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./models",
        help="Directory where the model will be saved"
    )
    parser.add_argument(
        "--quant-type",
        type=str,
        default="default",
        choices=quant_types.keys(),
        help="Quantization type (affects file size and quality)"
    )
    
    args = parser.parse_args()
    
    # Check disk space before downloading
    try:
        import shutil
        disk_usage = shutil.disk_usage(args.output_dir)
        free_space_gb = disk_usage.free / (1024 ** 3)
        
        required_space_gb = quant_types[args.quant_type]["file_size"] / (1024 ** 3) * 1.1  # Add 10% margin
        
        if free_space_gb < required_space_gb:
            logger.warning(f"WARNING: You have only {free_space_gb:.1f}GB free disk space.")
            logger.warning(f"Required space: {required_space_gb:.1f}GB")
            
            confirm = input("Continue with download? (y/n): ")
            if confirm.lower() != 'y':
                logger.info("Download cancelled by user")
                return
    except Exception as e:
        logger.warning(f"Could not check disk space ({e}), continuing anyway")
    
    download_model(args.output_dir, args.quant_type)

if __name__ == "__main__":
    main()