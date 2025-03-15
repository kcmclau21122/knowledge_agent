#!/usr/bin/env python
"""
Main entry point for the Knowledge Agent application
"""
import argparse
import logging
import logging.handlers
import os
from pathlib import Path

from config import Config
from knowledge_agent import KnowledgeAgent
from web_server import WebServer
from templates import create_templates
from utils import create_directories

logger = logging.getLogger(__name__)

def setup_logging(log_dir=None, log_level=logging.INFO):
    """
    Configure logging to output to both a rotating file and the console
    
    Args:
        log_dir: Directory to store log files (default: 'logs')
        log_level: Logging level (default: logging.INFO)
    """
    # Create log directory if it doesn't exist
    log_dir = Path(log_dir) if log_dir else Path('logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Define log file path
    log_file = log_dir / 'knowledge_agent.log'
    
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Set up rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10 MB per file
        backupCount=5,          # Keep 5 backup files (6 total including the active file)
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    
    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove any existing handlers to avoid duplicates when reconfiguring
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add the new handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Get a logger for this module
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured. Log file: {log_file}")
    logger.info(f"Log level: {logging.getLevelName(log_level)}")
    
    return logger

def main():
    """Main entry point for the application"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Knowledge Agent - Open Source Botpress Alternative")
    parser.add_argument("--company", type=str, default="Our Company", help="Company name")
    parser.add_argument("--host", type=str, default=Config.HOST, help="Web server host")
    parser.add_argument("--port", type=int, default=Config.PORT, help="Web server port")
    parser.add_argument("--knowledge-dir", type=str, default=None, help="Path to knowledge directory")
    parser.add_argument("--ingest", action="store_true", help="Ingest knowledge base")
    parser.add_argument("--model", type=str, default=None, help="LLM model name")
    parser.add_argument("--embedding-model", type=str, default=None, help="Embedding model name")
    parser.add_argument("--simple-mode", action="store_true", help="Run in simple mode without embeddings")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA even if available")
    parser.add_argument("--log-dir", type=str, default="logs", help="Path to log directory")
    parser.add_argument("--log-level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Logging level")
    args = parser.parse_args()
    
    # Setup logging first
    log_level = getattr(logging, args.log_level)
    setup_logging(args.log_dir, log_level)
    
    # Update config if needed
    if args.model:
        Config.LLM_MODEL = args.model
    
    if args.embedding_model:
        Config.EMBEDDING_MODEL = args.embedding_model
    
    if args.knowledge_dir:
        Config.KNOWLEDGE_DIR = Path(args.knowledge_dir)
    
    # Create necessary directories
    create_directories()
    
    # Create templates
    create_templates()
    
    # Initialize knowledge agent
    logger.info(f"Initializing Knowledge Agent for {args.company}")
    agent = KnowledgeAgent(company_name=args.company)
    
    # Ingest knowledge base if specified
    if args.ingest:
        logger.info("Starting knowledge base ingestion")
        agent.ingest_knowledge_base()
    
    # Initialize models
    logger.info("Loading models")
    agent.initialize()
    
    # Start web server
    logger.info(f"Starting web server at http://{args.host}:{args.port}")
    server = WebServer(agent)
    server.run(host=args.host, port=args.port)

if __name__ == "__main__":
    main()