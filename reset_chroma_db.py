#!/usr/bin/env python
"""
Script to reset ChromaDB database to an empty state
"""

import logging
import argparse
import shutil
import os
from pathlib import Path

from config import Config
from utils import create_directories

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def reset_vector_db(db_dir=None, confirm=False):
    """
    Reset the vector database by deleting and recreating the directory
    
    Args:
        db_dir: Directory containing the ChromaDB database
        confirm: Skip confirmation prompt if True
    """
    db_dir = Path(db_dir) if db_dir else Config.DB_DIR
    
    # Ensure the path exists and seems like a ChromaDB directory
    if not db_dir.exists():
        logger.info(f"Database directory {db_dir} doesn't exist. Creating a new one.")
        create_directories()
        return True
    
    # Check for Chroma files to ensure we're deleting the right directory
    chroma_files = list(db_dir.glob("chroma-*"))
    if not chroma_files and not confirm:
        logger.warning(f"Directory {db_dir} doesn't look like a ChromaDB directory. Are you sure?")
        confirmation = input("Type 'yes' to confirm deletion: ")
        if confirmation.lower() != 'yes':
            logger.info("Aborting reset operation.")
            return False
    
    # Confirm the reset if not already confirmed
    if not confirm:
        logger.warning(f"This will delete all data in {db_dir}.")
        confirmation = input("Type 'yes' to confirm: ")
        if confirmation.lower() != 'yes':
            logger.info("Aborting reset operation.")
            return False
    
    try:
        # Delete the directory and its contents
        logger.info(f"Deleting vector database at {db_dir}")
        shutil.rmtree(db_dir, ignore_errors=True)
        
        # Recreate the empty directory
        logger.info(f"Creating new empty database directory at {db_dir}")
        db_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Vector database reset successful")
        return True
        
    except Exception as e:
        logger.error(f"Error resetting vector database: {str(e)}")
        return False

def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description="Reset ChromaDB Vector Database")
    parser.add_argument("--db-dir", type=str, default=None, help="Custom database directory path")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()
    
    db_path = args.db_dir
    
    if reset_vector_db(db_path, args.force):
        logger.info("ChromaDB reset complete. The database is now empty.")
    else:
        logger.error("Failed to reset ChromaDB database.")

if __name__ == "__main__":
    main()
