"""
Repo Analyzer Service - Objective Intelligence for Repositories
"""
import logging
from typing import List, Dict, Any
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def generate_repo_summary(documents: List[Document], repo_name: str) -> Dict[str, Any]:
    """
    Analyze the repository documents to provide an objective summary.
    
    Args:
        documents: List of loaded documents
        repo_name: Name of the repository
        
    Returns:
        Dictionary containing tech stack, purpose, and key stats
    """
    logger.info(f"📊 Analyzing {len(documents)} documents for {repo_name}...")
    
    # 1. Tech Stack Detection
    extensions = {}
    for doc in documents:
        source = doc.metadata.get('source', '')
        if '.' in source:
            ext = source.split('.')[-1].lower()
            extensions[ext] = extensions.get(ext, 0) + 1
            
    # Sort by frequency
    sorted_exts = sorted(extensions.items(), key=lambda x: x[1], reverse=True)
    top_langs = [ext for ext, count in sorted_exts[:3]]
    
    # Map extensions to languages
    lang_map = {
        'py': 'Python', 'js': 'JavaScript', 'ts': 'TypeScript',
        'tsx': 'React/TypeScript', 'jsx': 'React', 'css': 'CSS',
        'html': 'HTML', 'java': 'Java', 'go': 'Go', 'rs': 'Rust',
        'md': 'Markdown', 'json': 'JSON', 'yml': 'YAML'
    }
    
    tech_stack = [lang_map.get(ext, ext.upper()) for ext in top_langs]
    
    # 2. Key Files Detection
    key_files = []
    important_patterns = ['readme', 'requirements', 'package.json', 'dockerfile', 'main', 'app', 'index']
    
    for doc in documents:
        source = doc.metadata.get('source', '').lower()
        if any(pattern in source for pattern in important_patterns):
            key_files.append(doc.metadata.get('source'))
            
    # Limit key files
    key_files = key_files[:5]
    
    # 3. Construct Summary
    summary = {
        "repo_name": repo_name,
        "doc_count": len(documents),
        "tech_stack": tech_stack,
        "key_files": key_files,
        "is_python": 'py' in extensions,
        "is_web": any(x in extensions for x in ['js', 'ts', 'html', 'css']),
    }
    
    return summary

def format_repo_analysis(summary: Dict[str, Any]) -> str:
    """Format the analysis into a premium markdown message."""
    
    stack_str = ", ".join(summary['tech_stack'])
    
    msg = f"""🎉 **Magical Ingestion Complete!**

I have analyzed `{summary['repo_name']}` and built a knowledge graph from **{summary['doc_count']}** files.

### 🧠 Repository Intelligence
- **Deep Tech Stack:** {stack_str}
- **Primary Type:** {"🐍 Python Backend" if summary.get('is_python') else "🌐 Web Application"}
- **Key Entry Points:**
"""
    
    for file in summary['key_files']:
        msg += f"  - `{file}`\n"
        
    msg += """
### 💡 Suggested Questions
- What is the main purpose of this repo?
- How is the project structured?
- dependency analysis?
"""
    return msg
