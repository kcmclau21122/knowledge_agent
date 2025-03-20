"""
Document processing module for loading and chunking documents
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import shutil
import os
import sys

# Document processing libraries
import docx
import PyPDF2
import csv
import pandas as pd
import markdown
from bs4 import BeautifulSoup
import requests
import traceback

from config import Config

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, chunk_size=None, chunk_overlap=None, llm_manager=None):
        """Initialize the document processor with configuration parameters"""
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        self.llm_manager = llm_manager
    
    def move_to_ingested(self, file_path: Path):
        """
        Move a processed file to an 'Ingested' subdirectory within its current directory
        
        Args:
            file_path: Path to the file to be moved
        """
        try:
            # Convert to Path object if not already
            file_path = Path(file_path)
            
            # Ensure the file exists
            if not file_path.exists():
                logger.warning(f"File not found, cannot move: {file_path}")
                return
            
            # Get the parent directory
            parent_dir = file_path.parent
            
            # Create 'Ingested' subdirectory if it doesn't exist
            ingested_dir = parent_dir / 'Ingested'
            ingested_dir.mkdir(exist_ok=True)
            
            # Construct destination path
            # Use original filename, but rename if a file with the same name already exists
            dest_path = ingested_dir / file_path.name
            counter = 1
            while dest_path.exists():
                dest_path = ingested_dir / f"{file_path.stem}_{counter}{file_path.suffix}"
                counter += 1
            
            # Move the file
            shutil.move(str(file_path), str(dest_path))
            
            logger.info(f"Moved processed file to: {dest_path}")
        except Exception as e:
            logger.error(f"Error moving file {file_path} to Ingested directory: {e}")
            logger.error(f"Detailed error: {traceback.format_exc()}")
    
    def load_document(self, file_path: str) -> str:
        """Load a document from various file formats and return its text content"""
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()
        
        try:
            logger.debug(f"Attempting to load document: {file_path}")
            
            if file_extension == '.pdf':
                return self._load_pdf(file_path)
            elif file_extension == '.docx':
                return self._load_docx(file_path)
            elif file_extension == '.txt':
                return self._load_txt(file_path)
            elif file_extension == '.md':
                return self._load_markdown(file_path)
            elif file_extension in ['.csv', '.xlsx', '.xls']:
                return self._load_tabular(file_path)
            elif file_extension in ['.html', '.htm']:
                return self._load_html(file_path)
            else:
                logger.warning(f"Unsupported file format: {file_extension}")
                return ""
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            logger.error(f"Detailed error: {traceback.format_exc()}")
            return ""
    
    def _load_pdf(self, file_path: Path) -> str:
        """Load and extract text from a PDF file"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
        logger.debug(f"Loaded PDF, total characters: {len(text)}")
        return text
    
    def _load_docx(self, file_path: Path) -> str:
        """Load and extract text from a DOCX file"""
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        logger.debug(f"Loaded DOCX, total characters: {len(text)}")
        return text
    
    def _load_txt(self, file_path: Path) -> str:
        """Load text from a TXT file"""
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            text = file.read()
        logger.debug(f"Loaded TXT, total characters: {len(text)}")
        return text
    
    def _load_markdown(self, file_path: Path) -> str:
        """Load and extract text from a Markdown file"""
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            md_content = file.read()
            html_content = markdown.markdown(md_content)
            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text()
        logger.debug(f"Loaded Markdown, total characters: {len(text)}")
        return text
    
    def _load_tabular(self, file_path: Path) -> str:
        """Load and convert tabular data to text"""
        file_extension = file_path.suffix.lower()
        try:
            if file_extension == '.csv':
                df = pd.read_csv(file_path, on_bad_lines='skip')
            else:  # Excel formats
                df = pd.read_excel(file_path)
            
            # Convert DataFrame to a readable text format
            text = df.to_string()
            logger.debug(f"Loaded tabular file, total characters: {len(text)}")
            return text
        except Exception as e:
            logger.error(f"Error processing tabular file {file_path}: {e}")
            return f"Error processing file: {str(e)}"
    
    def _load_html(self, file_path: Path) -> str:
        """Load and extract text from an HTML file"""
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            html_content = file.read()
            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text()
        logger.debug(f"Loaded HTML, total characters: {len(text)}")
        return text
    
    def chunk_text(self, text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
        """
        Split text into chunks with optional overlap
        
        Args:
            text: The input text to chunk
            chunk_size: Size of each chunk (defaults to self.chunk_size)
            chunk_overlap: Number of characters to overlap between chunks (defaults to self.chunk_overlap)
        
        Returns:
            List of text chunks
        """
        # Use instance variables if not provided
        chunk_size = chunk_size or getattr(self, 'chunk_size', 768)
        chunk_overlap = chunk_overlap or getattr(self, 'chunk_overlap', 50)
        
        # Handle extremely short texts
        if len(text) <= chunk_size:
            return [text.strip()]
        
        # Improved chunking algorithm
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end of chunk
            end = min(start + chunk_size, len(text))
            
            # Extract chunk
            chunk = text[start:end].strip()
            
            # Try to find a clean break point
            if end <= len(text):
                # Look for sentence or paragraph break within the last 100 characters
                search_range = min(100, chunk_size // 4)
                
                # Preferred break points
                sentence_breaks = [
                    chunk.rfind('. ', max(0, len(chunk) - search_range)),
                    chunk.rfind('! ', max(0, len(chunk) - search_range)),
                    chunk.rfind('? ', max(0, len(chunk) - search_range))
                ]
                paragraph_break = chunk.rfind('\n\n', max(0, len(chunk) - search_range))
                
                # Find the best break point
                best_break = max(max(sentence_breaks), paragraph_break)
                
                if best_break != -1:
                    chunk = chunk[:best_break + 2].strip()
            
            # Add chunk if not empty
            if chunk:
                chunks.append(chunk)
            
            # Move start point, considering overlap
            if len(chunk) > chunk_overlap:
                start += len(chunk) - chunk_overlap
            elif len(chunk) <= 0:
                start += chunk_overlap
            else:
                start += len(chunk)
            
            # Safety check to prevent infinite loop
            if start >= len(text):
                break
        
        # Remove any empty chunks and extra whitespace
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        
        # Logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Chunked text into {len(chunks)} chunks")
        
        return chunks

    
    def process_knowledge_base(self, knowledge_dir: Path) -> List[Dict[str, Any]]:
        """Process all documents in the knowledge directory with enhanced logging"""
        logger.info("==========================================")
        logger.info("STARTING DOCUMENT PROCESSING")
        logger.info(f"Knowledge directory: {knowledge_dir}")
        logger.info("==========================================")
        
        documents = []
        knowledge_dir = Path(knowledge_dir)
        
        if not knowledge_dir.exists():
            logger.error(f"Knowledge directory {knowledge_dir} does not exist")
            # Print current working directory and check if it exists
            cwd = Path.cwd()
            logger.error(f"Current working directory: {cwd}")
            logger.error(f"Current directory exists: {cwd.exists()}")
            
            # List all drives and paths for further debugging
            if sys.platform == 'win32':
                import string
                import os
                drives = [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:")]
                logger.error(f"Available drives: {drives}")
            
            return documents

        try:
            # Recursively find all files, excluding 'Ingested' directory
            file_paths = [
                path for path in knowledge_dir.rglob('*') 
                if path.is_file() 
                and 'Ingested' not in str(path.parts) 
                and not path.name.startswith('.')  # Ignore hidden files
                and path.suffix.lower() in ['.pdf', '.docx', '.txt', '.md', '.csv', '.html', '.xlsx', '.xls']
            ]
            
            logger.info(f"Found {len(file_paths)} files to process")
            
            # Detailed logging of files found
            for file_path in file_paths:
                logger.debug(f"Will process file: {file_path}")
            
            for file_path in file_paths:
                try:
                    # More detailed logging
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    logger.info(f"[{timestamp}] Processing file: {file_path}")
                    
                    # Extract relative path for metadata
                    rel_path = file_path.relative_to(knowledge_dir)
                    
                    # Log file details
                    logger.debug(f"File details:")
                    logger.debug(f"  Absolute path: {file_path}")
                    logger.debug(f"  Relative path: {rel_path}")
                    logger.debug(f"  File size: {file_path.stat().st_size} bytes")
                    
                    # Load document content
                    content = self.load_document(str(file_path))
                    
                    if not content:
                        logger.warning(f"No content extracted from {file_path}")
                        self.move_to_ingested(file_path)  # Still move empty files
                        continue
                    
                    # Log content length
                    logger.debug(f"Extracted content length: {len(content)} characters")
                    
                    # Chunk the document
                    chunks = self.chunk_text(content)
                    
                    if not chunks:
                        logger.warning(f"No chunks created for {file_path}")
                        continue
                    
                    # Log chunking details
                    logger.debug(f"Created {len(chunks)} chunks from file")
                    
                    # Create document entries for each chunk
                    for i, chunk in enumerate(chunks):
                        documents.append({
                            "text": chunk,
                            "metadata": {
                                "source": str(rel_path),
                                "chunk": i,
                                "total_chunks": len(chunks),
                                "created_at": datetime.now().isoformat()
                            }
                        })
                    
                    # Move the file to the Ingested folder
                    self.move_to_ingested(file_path)
                    
                except Exception as file_error:
                    logger.error(f"Error processing file {file_path}: {str(file_error)}")
                    logger.error(f"Detailed error: {traceback.format_exc()}")
                    # Continue with next file instead of stopping entire process
                    continue
            
        except Exception as e:
            logger.error(f"Comprehensive error processing knowledge base: {str(e)}")
            logger.error(f"Detailed error: {traceback.format_exc()}")
        
        logger.info(f"Processed {len(documents)} document chunks total")
        
        # Additional diagnostic logging
        if documents:
            source_stats = {}
            for doc in documents:
                source = doc['metadata']['source']
                source_stats[source] = source_stats.get(source, 0) + 1
            
            logger.info("Document chunk distribution:")
            for source, count in source_stats.items():
                logger.info(f"  {source}: {count} chunks")
        
        return documents