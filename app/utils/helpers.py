"""
Utility functions for ThriveBot
"""

import re
from datetime import datetime
from typing import Optional


def format_timestamp(dt: Optional[datetime] = None, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format datetime to string"""
    if dt is None:
        dt = datetime.now()
    return dt.strftime(format_str)


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to max length with suffix"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing invalid characters"""
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip('. ')
    return sanitized


def extract_urls(text: str) -> list:
    """Extract URLs from text"""
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    return re.findall(url_pattern, text)


def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and normalizing"""
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def chunk_list(lst: list, chunk_size: int) -> list:
    """Split a list into chunks of specified size"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def estimate_tokens(text: str) -> int:
    """Estimate token count (approximate: 4 chars per token)"""
    return len(text) // 4


def format_sources_for_display(sources: list, max_sources: int = 3) -> str:
    """Format source list for display"""
    if not sources:
        return "No sources available"
    
    formatted = []
    for i, source in enumerate(sources[:max_sources], 1):
        name = source.get("source", "Unknown")
        if "/" in name:
            name = name.split("/")[-1]
        score = source.get("score", 0)
        formatted.append(f"{i}. {name} ({score:.0%})")
    
    return "\n".join(formatted)


def mask_api_key(key: str, visible_chars: int = 4) -> str:
    """Mask API key for logging, showing only last few characters"""
    if not key or len(key) <= visible_chars:
        return "***"
    return f"***{key[-visible_chars:]}"
