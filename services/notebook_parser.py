"""
Jupyter Notebook Parser for RAG System

Extracts code and markdown cells from .ipynb files, ignoring outputs and metadata.
"""

import json
from typing import Dict, List


def parse_notebook(content: str) -> str:
    """
    Parse a Jupyter notebook and extract code + markdown cells.
    
    Args:
        content: Raw .ipynb file content (JSON string)
        
    Returns:
        Cleaned text with code and markdown cells
    """
    try:
        notebook = json.loads(content)
        cells = notebook.get("cells", [])
        
        extracted_content = []
        
        for cell in cells:
            cell_type = cell.get("cell_type", "")
            source = cell.get("source", [])
            
            # Join source lines (can be list or string)
            if isinstance(source, list):
                cell_content = "".join(source)
            else:
                cell_content = source
            
            if cell_type == "markdown":
                extracted_content.append(f"# Markdown Cell\n{cell_content}\n")
            elif cell_type == "code":
                extracted_content.append(f"# Code Cell\n```python\n{cell_content}\n```\n")
        
        return "\n".join(extracted_content)
    
    except (json.JSONDecodeError, KeyError) as e:
        # If parsing fails, return original content
        return content


def is_notebook_file(filename: str) -> bool:
    """Check if a file is a Jupyter notebook."""
    return filename.lower().endswith(".ipynb")
