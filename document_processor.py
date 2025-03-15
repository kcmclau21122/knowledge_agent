"""
Document processing module for loading and chunking documents
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import shutil
import os

# Document processing libraries
import docx
import PyPDF2
import csv
import pandas as pd
import markdown
from bs4 import BeautifulSoup
import requests

from config import Config

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, chunk_size=None, chunk_overlap=None):
        """Initialize the document processor with configuration parameters"""
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
    
    def load_document(self, file_path: str) -> str:
        """Load a document from various file formats and return its text content"""
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()
        
        try:
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
            return ""
    
    def _load_pdf(self, file_path: Path) -> str:
        """Load and extract text from a PDF file"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
        return text
    
    def _load_docx(self, file_path: Path) -> str:
        """Load and extract text from a DOCX file"""
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    
    def _load_txt(self, file_path: Path) -> str:
        """Load text from a TXT file"""
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            return file.read()
    
    def _load_markdown(self, file_path: Path) -> str:
        """Load and extract text from a Markdown file"""
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            md_content = file.read()
            html_content = markdown.markdown(md_content)
            soup = BeautifulSoup(html_content, 'html.parser')
            return soup.get_text()
    
    def _load_tabular(self, file_path: Path) -> str:
        """Load and convert tabular data to text"""
        file_extension = file_path.suffix.lower()
        try:
            if file_extension == '.csv':
                df = pd.read_csv(file_path, on_bad_lines='skip')
            else:  # Excel formats
                df = pd.read_excel(file_path)
            
            # Convert DataFrame to a readable text format
            return df.to_string()
        except Exception as e:
            logger.error(f"Error processing tabular file {file_path}: {e}")
            return f"Error processing file: {str(e)}"
    
    def _load_html(self, file_path: Path) -> str:
        """Load and extract text from an HTML file"""
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            html_content = file.read()
            soup = BeautifulSoup(html_content, 'html.parser')
            return soup.get_text()
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks for vectorization with fixed-size approach"""
        if not text:
            logger.warning("Text is empty, returning empty list")
            return []
        
        # Improved chunking algorithm with fixed segment size
        chunks = []
        
        # If text is very small, just return it as a single chunk
        if len(text) <= self.chunk_size:
            return [text]
            
        # Segment text by sentences or paragraphs first if possible
        segments = []
        # Try to split by paragraphs first (double newline)
        if "\n\n" in text:
            segments = text.split("\n\n")
        # If no paragraphs or too few, split by single newlines
        elif "\n" in text and len(segments) < 2:
            segments = text.split("\n")
        # If still too few segments, split by periods
        elif ". " in text and len(segments) < 2:
            segments = text.split(". ")
            # Add the periods back
            segments = [s + "." if i < len(segments) - 1 else s for i, s in enumerate(segments)]
        # Fallback to just the text itself
        else:
            segments = [text]
        
        current_chunk = ""
        
        for segment in segments:
            # If adding this segment would exceed chunk size
            if len(current_chunk) + len(segment) + 1 > self.chunk_size:
                # If current chunk has content, add it to chunks
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If the segment itself is longer than chunk_size, we need to split it
                if len(segment) > self.chunk_size:
                    # Simple character-based splitting for oversized segments
                    for i in range(0, len(segment), self.chunk_size - self.chunk_overlap):
                        end = min(i + self.chunk_size, len(segment))
                        chunks.append(segment[i:end])
                else:
                    # Start a new chunk with this segment
                    current_chunk = segment
            else:
                # Add a space if the current chunk is not empty
                if current_chunk:
                    current_chunk += " "
                # Add the segment to the current chunk
                current_chunk += segment
        
        # Add the last chunk if it has content
        if current_chunk:
            chunks.append(current_chunk)
        
        logger.info(f"Created a total of {len(chunks)} chunks")
        return chunks
    
    def move_to_ingested(self, file_path: Path) -> None:
        """
        Move processed file to an 'Ingested' subfolder
        
        Args:
            file_path: Path to the file to be moved
        """
        try:
            # Create the 'Ingested' subfolder if it doesn't exist
            ingested_folder = file_path.parent / "Ingested"
            if not ingested_folder.exists():
                logger.info(f"Creating 'Ingested' folder at {ingested_folder}")
                ingested_folder.mkdir(parents=True, exist_ok=True)
            
            # Define the destination path
            dest_path = ingested_folder / file_path.name
            
            # Move the file, overwriting if it already exists
            if dest_path.exists():
                logger.info(f"Removing existing file at {dest_path}")
                dest_path.unlink()
            
            logger.info(f"Moving {file_path} to {dest_path}")
            shutil.move(str(file_path), str(dest_path))
            
        except Exception as e:
            logger.error(f"Error moving file to Ingested folder: {str(e)}")
    
    def process_knowledge_base(self, knowledge_dir: Path) -> List[Dict[str, Any]]:
        """Process all documents in the knowledge directory"""
        logger.info("==========================================")
        logger.info("STARTING DOCUMENT PROCESSING")
        logger.info(f"Knowledge directory: {knowledge_dir}")
        logger.info("==========================================")
        
        documents = []
        knowledge_dir = Path(knowledge_dir)
        
        if not knowledge_dir.exists():
            logger.error(f"Knowledge directory {knowledge_dir} does not exist")
            return documents
        
        try:
            for file_path in list(knowledge_dir.glob('**/*')):
                if file_path.is_file() and "Ingested" not in str(file_path.parts) and not str(file_path).endswith(".gitkeep"):
                    # Print filename with timestamp
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    logger.info(f"[{timestamp}] Processing file: {file_path}")
                    
                    # Extract relative path for metadata
                    rel_path = file_path.relative_to(knowledge_dir)
                    
                    # Load document content
                    content = self.load_document(str(file_path))
                    if not content:
                        logger.warning(f"No content extracted from {file_path}")
                        self.move_to_ingested(file_path)  # Still move empty files
                        continue
                    
                    # Chunk the document
                    chunks = self.chunk_text(content)
                    
                    # Create document entries for each chunk
                    for i, chunk in enumerate(chunks):
                        documents.append({
                            "text": chunk,
                            "metadata": {
                                "source": str(rel_path),
                                "chunk": i,
                                "created_at": datetime.now().isoformat()
                            }
                        })
                    
                    # Move the file to the Ingested folder
                    self.move_to_ingested(file_path)
        except Exception as e:
            logger.error(f"Error processing knowledge base: {str(e)}")
        
        logger.info(f"Processed {len(documents)} document chunks")
        return documents
