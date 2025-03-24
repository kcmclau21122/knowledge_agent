"""
Semantic Document Processor for advanced document chunking and enrichment
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import nltk
from nltk.tokenize import sent_tokenize
from collections import Counter

logger = logging.getLogger(__name__)

class SemanticDocumentProcessor:
    def __init__(self, llm_manager=None, min_chunk_size=100, max_chunk_size=1000, min_chunk_overlap=20):
        """Initialize the enhanced document processor with semantic chunking capabilities"""
        self.llm_manager = llm_manager
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_chunk_overlap = min_chunk_overlap
        
        # Download NLTK resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def process_document(self, text: str, document_name: str) -> List[Dict[str, Any]]:
        """Process a document with enhanced semantic chunking and metadata enrichment"""
        # 1. Extract document structure (headers, sections)
        structured_sections = self._extract_document_structure(text)
        
        # 2. Create semantic chunks from the structured sections
        raw_chunks = self._create_semantic_chunks(structured_sections)
        
        # 3. Enhance chunks with LLM-generated metadata
        enhanced_chunks = []
        for i, chunk in enumerate(raw_chunks):
            chunk_data = {
                "text": chunk["text"],
                "metadata": {
                    "source": document_name,
                    "section": chunk.get("section", ""),
                    "chunk_index": i,
                    "created_at": None  # Will be filled by the document processor
                }
            }
            
            # Add LLM-enhanced metadata if LLM manager is available
            if self.llm_manager:
                enhanced_chunk = self._enhance_chunk_with_llm(chunk_data)
                enhanced_chunks.append(enhanced_chunk)
            else:
                # Basic keyword extraction if no LLM
                keywords = self._extract_basic_keywords(chunk["text"])
                chunk_data["metadata"]["keywords"] = keywords
                enhanced_chunks.append(chunk_data)
        
        return enhanced_chunks
    
    def _extract_document_structure(self, text: str) -> List[Dict[str, Any]]:
        """Extract document structure with headers and sections"""
        # Pattern for headers (Markdown style or plain text with colons)
        header_pattern = re.compile(r'^(#{1,6}\s+.+|[A-Z][A-Za-z0-9\s]+:)$', re.MULTILINE)
        
        sections = []
        lines = text.split('\n')
        current_header = "Introduction"
        current_content = []
        
        # Process line by line
        for line in lines:
            if header_pattern.match(line):
                # Save previous section if it has content
                if current_content:
                    sections.append({
                        "header": current_header,
                        "content": '\n'.join(current_content)
                    })
                
                # Start new section
                current_header = line.strip('# :')
                current_content = []
            else:
                current_content.append(line)
        
        # Add the last section
        if current_content:
            sections.append({
                "header": current_header,
                "content": '\n'.join(current_content)
            })
        
        return sections
    
    def _create_semantic_chunks(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create chunks that preserve semantic meaning"""
        chunks = []
        
        for section in sections:
            section_text = section["content"]
            section_header = section["header"]
            
            # Skip empty sections
            if not section_text.strip():
                continue
            
            # If section is small enough, keep as one chunk
            if len(section_text) <= self.max_chunk_size:
                chunks.append({
                    "text": section_text,
                    "section": section_header
                })
                continue
            
            # Otherwise, split into semantic chunks
            sentences = sent_tokenize(section_text)
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence_length = len(sentence)
                
                # If adding this sentence exceeds max size and we have content
                if current_length + sentence_length > self.max_chunk_size and current_chunk:
                    # Save current chunk
                    chunks.append({
                        "text": ' '.join(current_chunk),
                        "section": section_header
                    })
                    
                    # Start new chunk with overlap
                    overlap_sentences = min(2, len(current_chunk))
                    current_chunk = current_chunk[-overlap_sentences:] + [sentence]
                    current_length = sum(len(s) for s in current_chunk)
                else:
                    # Add sentence to current chunk
                    current_chunk.append(sentence)
                    current_length += sentence_length
            
            # Add the last chunk if it has content and meets minimum size
            if current_chunk and current_length >= self.min_chunk_size:
                chunks.append({
                    "text": ' '.join(current_chunk),
                    "section": section_header
                })
        
        return chunks
    
    def _enhance_chunk_with_llm(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary and keywords for a chunk using LLM"""
        prompt = f"""
        Please analyze the following text and provide:
        1. A concise summary (1-2 sentences)
        2. 5-7 key concepts or keywords representative of the content

        Format your response exactly as:
        SUMMARY: [your summary here]
        KEYWORDS: [comma-separated keywords]

        Text to analyze:
        {chunk['text']}
        """
        
        try:
            response = self.llm_manager.generate_response(prompt, max_new_tokens=256)
            
            # Extract summary and keywords
            summary_match = re.search(r'SUMMARY:\s*(.*?)(?:\n|KEYWORDS:)', response, re.DOTALL)
            keywords_match = re.search(r'KEYWORDS:\s*(.*?)(?:\n|$)', response, re.DOTALL)
            
            summary = summary_match.group(1).strip() if summary_match else ""
            
            keywords = []
            if keywords_match:
                keywords_text = keywords_match.group(1).strip()
                keywords = [k.strip() for k in keywords_text.split(',')]
            
            # Add to chunk metadata
            chunk["metadata"]["summary"] = summary
            chunk["metadata"]["keywords"] = keywords
            
        except Exception as e:
            logger.error(f"Error enhancing chunk with LLM: {e}")
            # Fallback to basic keyword extraction
            keywords = self._extract_basic_keywords(chunk["text"])
            chunk["metadata"]["keywords"] = keywords
        
        return chunk
    
    def _extract_basic_keywords(self, text: str) -> List[str]:
        """Basic keyword extraction without LLM"""
        # Simple stopwords list
        stopwords = {"the", "a", "an", "and", "or", "but", "if", "because", "as", "what",
                     "which", "this", "that", "these", "those", "then", "just", "so", "than",
                     "such", "when", "who", "how", "where", "why", "is", "are", "was", "were",
                     "be", "been", "being", "have", "has", "had", "having", "do", "does", "did",
                     "doing", "to", "for", "with", "about", "against", "between", "into", "through",
                     "during", "before", "after", "above", "below", "from", "up", "down", "in",
                     "out", "on", "off", "over", "under", "again", "further", "then", "once"}
        
        # Tokenize and clean words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out stopwords and count occurrences
        filtered_words = [word for word in words if word not in stopwords]
        word_counts = Counter(filtered_words)
        
        # Get the most common words as keywords
        keywords = [word for word, _ in word_counts.most_common(7)]
        
        return keywords
