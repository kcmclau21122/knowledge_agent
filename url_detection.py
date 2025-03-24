"""
URL detection and hyperlink conversion utility functions with improved response cleaning
"""

import re
import html
import logging

logger = logging.getLogger(__name__)

# Replace the clean_llm_response function in url_detection.py:

def clean_llm_response(text):
    """
    Clean LLM response by removing any leaked system prompts, instructions, or context
    
    Args:
        text: Raw text from LLM response
        
    Returns:
        Cleaned response text
    """
    # If response is empty, return as is
    if not text or not text.strip():
        return text
        
    # Remove numbered responses (like "1." at the beginning)
    cleaned_text = re.sub(r'^\s*\d+\.\s*', '', text)
    
    # Common patterns that might indicate leaked system prompt or context
    system_patterns = [
        # Match Mistral/Llama instruction pattern at the start
        r"^\s*\[INST\].*?\[/INST\]\s*",
        
        # Match system prompt sections at the start
        r"^\s*You are a helpful customer service assistant.*?(?=\n\n)",
        
        # Match knowledge base header at the start (not the content)
        r"^\s*KNOWLEDGE BASE INFORMATION:\s*\n",
    ]
    
    # Try each pattern and remove if found at the start of the text
    for pattern in system_patterns:
        match = re.search(pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
        if match and match.start() < 20:  # Only if near the beginning
            # Remove everything up to the end of the match
            cleaned_text = cleaned_text[match.end():].strip()
    
    # Additional cleanup for any remaining markdown artifacts at the beginning
    cleaned_text = re.sub(r"^\s*---+\s*", "", cleaned_text)  # Remove markdown dividers
    
    # Safety check - if we've removed too much, keep the original
    if len(cleaned_text) < 50 and len(text) > 200:
        logger.warning("Response was over-cleaned, reverting to original")
        return text
    
    # Log the cleanup if significant changes were made
    if len(cleaned_text) < len(text) * 0.9:  # If we removed more than 10%
        logger.info(f"Cleaned response text from {len(text)} to {len(cleaned_text)} characters")
    
    return cleaned_text

def convert_urls_to_hyperlinks(text):
    """
    Convert URLs in text to HTML hyperlinks with improved cleaning
    
    Args:
        text: Input text containing URLs
        
    Returns:
        Text with URLs converted to hyperlinks
    """
    # First, clean the response to remove any leaked system prompt or context
    cleaned_text = clean_llm_response(text)
    
    # URL regex pattern - matches common URL formats
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/[-\w%!./?=&#+:~]*)*'
    
    # Function to replace each URL with hyperlink
    def replace_url(match):
        url = match.group(0)
        # Escape HTML entities in the URL to prevent injection
        safe_url = html.escape(url)
        # Add class or inline style for coloring
        return f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer" style="color:#0066cc;text-decoration:underline;">{safe_url}</a>'
    
    # Replace URLs with hyperlinks
    text_with_links = re.sub(url_pattern, replace_url, cleaned_text)
    
    return text_with_links