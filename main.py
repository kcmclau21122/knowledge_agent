#!/usr/bin/env python
"""
Main entry point for the Knowledge Agent application
"""
import argparse
import logging
import logging.handlers
import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime

from config import Config
from knowledge_agent import KnowledgeAgent
from web_server import WebServer
from templates import create_templates
from utils import create_directories

def setup_detailed_logging(log_dir=None, log_level=logging.DEBUG):
    """
    Configure extremely detailed logging with robust error handling and directory creation
    """
    # Print initial debugging information
    print("=" * 50)
    print("LOGGING CONFIGURATION DEBUG")
    print("=" * 50)
    print(f"Current Working Directory: {os.getcwd()}")
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {sys.version}")
    
    # Ensure absolute path for log directory
    try:
        # Use provided log directory or default to a standard location
        if log_dir:
            log_dir = Path(log_dir).resolve()
        else:
            # Use a platform-specific default log directory
            if sys.platform == 'win32':
                log_dir = Path(os.path.expanduser('~')) / 'KnowledgeAgentLogs'
            else:
                log_dir = Path('/var/log/knowledge_agent')
        
        # Attempt to create log directory with full permissions
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Debug log directory information
        print(f"Log Directory: {log_dir}")
        print(f"Log Directory Exists: {log_dir.exists()}")
        print(f"Log Directory Writable: {os.access(log_dir, os.W_OK)}")
        
        # Verify directory is writable
        test_file = log_dir / 'write_test.txt'
        try:
            with open(test_file, 'w') as f:
                f.write('Logging directory write test')
            test_file.unlink()  # Remove test file
            print("Log directory write test: PASSED")
        except (IOError, PermissionError) as write_error:
            print(f"WARNING: Cannot write to log directory: {write_error}")
            # Fallback to system temp directory
            log_dir = Path(tempfile.gettempdir()) / 'knowledge_agent_logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            print(f"Falling back to temp directory: {log_dir}")
        
        # Generate timestamped log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f'knowledge_agent_debug_{timestamp}.log'
        
        # Ensure file is writable
        try:
            log_file.touch(exist_ok=True)
            print(f"Log file created: {log_file}")
        except (IOError, PermissionError) as file_error:
            print(f"ERROR: Cannot create log file: {file_error}")
            # Fallback to writing to stdout
            log_file = None
        
        # Configure formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        
        # Clear any existing handlers to prevent duplication
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Configure root logger first
        root_logger.setLevel(log_level)
        
        # Logging handlers
        handlers = []
        
        # File handler (if file is writable)
        if log_file:
            try:
                file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
                file_handler.setFormatter(file_formatter)
                file_handler.setLevel(logging.DEBUG)
                # Add handler directly to root logger
                root_logger.addHandler(file_handler)
                handlers.append(file_handler)
                print(f"Logging to file: {log_file}")
                
                # Force immediate flush to file
                file_handler.flush()
                
            except Exception as handler_error:
                print(f"ERROR setting up file handler: {handler_error}")
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        # Add handler directly to root logger
        root_logger.addHandler(console_handler)
        handlers.append(console_handler)
        
        # Test that logging works
        test_logger = logging.getLogger("logging_test")
        test_logger.info("TEST: Logging system initialized")
        
        # Reduce logging for noisy libraries
        noisy_loggers = [
            'urllib3', 'chromadb', 'transformers', 
            'sentence_transformers', 'httpx', 'httpcore'
        ]
        for logger_name in noisy_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
        
        # Explicit logger for this module
        logger = logging.getLogger(__name__)
        logger.info("Detailed logging successfully configured")
        logger.info(f"Log file location: {log_file}")
        
        # Explicitly flush all handlers
        for handler in root_logger.handlers:
            handler.flush()
        
        return logger
    
    except Exception as e:
        # Absolute fallback logging
        print(f"CRITICAL LOGGING ERROR: {e}")
        print("Falling back to basic stdout logging")
        
        # Clear any existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            
        # Set up basic logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s: %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        
        return logging.getLogger(__name__)

def main():
    """Main entry point for the application"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Knowledge Agent - Open Source Botpress Alternative")
    parser.add_argument("--company", type=str, default="Our Company", help="Company name")
    parser.add_argument("--host", type=str, default=Config.HOST, help="Web server host")
    parser.add_argument("--port", type=int, default=Config.PORT, help="Web server port")
    parser.add_argument("--gpu-memory", type=float, default=0.7, 
                   help="Fraction of GPU memory to use (default: 0.7)")
    parser.add_argument("--offload", action="store_true", 
                    help="Enable CPU offloading for large models")
    parser.add_argument("--knowledge-dir", type=str, default=None, help="Path to knowledge directory")
    parser.add_argument("--ingest", action="store_true", help="Ingest knowledge base")
    parser.add_argument("--model", type=str, default=None, help="LLM model name")
    parser.add_argument("--embedding-model", type=str, default=None, help="Embedding model name")
    parser.add_argument("--simple-mode", action="store_true", help="Run in simple mode without embeddings")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA even if available")
    parser.add_argument("--log-dir", type=str, default=None, help="Path to log directory")
    parser.add_argument("--log-level", type=str, default="DEBUG", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Logging level")
    args = parser.parse_args()
    
    # Setup extremely detailed logging
    log_level = getattr(logging, args.log_level.upper())
    logger = setup_detailed_logging(args.log_dir, log_level)
    
    # Log startup details
    logger.info("=" * 50)
    logger.info("KNOWLEDGE AGENT STARTING")
    logger.info("=" * 50)
    
    # Log detailed system information
    logger.info(f"Current Working Directory: {os.getcwd()}")
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"Python Executable: {sys.executable}")
    
    # Update config if needed
    if args.model:
        Config.LLM_MODEL = args.model
        logger.info(f"Using custom LLM model: {args.model}")
    
    if args.embedding_model:
        Config.EMBEDDING_MODEL = args.embedding_model
        logger.info(f"Using custom embedding model: {args.embedding_model}")
    
    if args.knowledge_dir:
        Config.KNOWLEDGE_DIR = Path(args.knowledge_dir)
        logger.info(f"Using custom knowledge directory: {args.knowledge_dir}")
        
        # Detailed directory check
        knowledge_dir = Config.KNOWLEDGE_DIR
        logger.info(f"Knowledge Directory Path: {knowledge_dir}")
        logger.info(f"Knowledge Directory Exists: {knowledge_dir.exists()}")
        
        if knowledge_dir.exists():
            # List contents of the directory
            try:
                files = list(knowledge_dir.glob('*'))
                logger.info(f"Total files/directories in knowledge directory: {len(files)}")
                for f in files[:20]:  # Log first 20 items
                    logger.info(f"  {f}")
                if len(files) > 20:
                    logger.info(f"  ... and {len(files) - 20} more")
            except Exception as e:
                logger.error(f"Error listing directory contents: {e}")

    parser.add_argument("--no-graph", action="store_true", 
                    help="Disable knowledge graph for retrieval")

    # Create necessary directories
    create_directories()
    
    # Create templates
    create_templates()
    
    # Initialize knowledge agent
    logger.info(f"Initializing Knowledge Agent for {args.company}")
    use_graph = not args.no_graph if args.no_graph else None  # Use config default unless explicitly disabled
    agent = KnowledgeAgent(company_name=args.company, use_graph=use_graph)
    
    # Ingest knowledge base if specified
    if args.ingest:
        logger.info("Starting knowledge base ingestion")
        docs_added = agent.ingest_knowledge_base()
        logger.info(f"Documents added to knowledge base: {docs_added}")
    
    # Initialize models
    logger.info("Loading models")
    agent.initialize()
    
    # Make sure models are loaded before starting the server
    logger.info("Ensuring models are loaded...")
    if not hasattr(agent.llm_manager, 'model') or agent.llm_manager.model is None:
        logger.info("Models not loaded yet, initializing...")
        agent.initialize()
    else:
        logger.info("Models already loaded")

    # Generate a test query to ensure everything is working
    logger.info("Running test query to ensure system is working...")
    test_response = agent.answer_question("Can you help me?")
    logger.info(f"Test query successful, received response of length {len(test_response)}")


    # Start web server
    logger.info(f"Starting web server at http://{args.host}:{args.port}")
    server = WebServer(agent)
    server.run(host=args.host, port=args.port)

if __name__ == "__main__":
    # If you want to override arguments for testing, do it here
    import sys
    if len(sys.argv) == 1:  # Only if no arguments were provided
        print("No arguments provided. Using default test configuration.")
        sys.argv = [
            sys.argv[0], 
            #"--ingest",
            "--log-level", "DEBUG", 
            "--gpu-memory=0.8",
            "--log-dir", "./logs", 
            "--knowledge-dir", "C:/TestData/preborn", 
            "--company", "PreBorn"
        ]
    
    # Call main() which will parse sys.argv
    main()