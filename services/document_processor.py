"""
Document Processor - Production-Grade Code Splitting for RAG

Advanced splitting strategies:
1. AST-based splitting (keeps functions/classes whole)
2. Import context enrichment (preserves dependencies)
3. Language-aware fallback (smart splitting for large items)

This ensures maximum context preservation for code RAG systems.
"""

import ast
import os
from typing import List, Dict, Optional
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
    except ImportError:
        # Final fallback for very old versions or different structures
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        # Mock Language enum if missing
        class Language:
            PYTHON = "python"
            JS = "js"
            TS = "ts"
            MARKDOWN = "markdown"
            JAVA = "java"
            CPP = "cpp"
            GO = "go"
            RUST = "rust"
from langchain_core.documents import Document
import logging

# Import notebook parser
try:
    from services.notebook_parser import parse_notebook, is_notebook_file
except ImportError:
    # Fallback if notebook_parser is not available
    def parse_notebook(content: str) -> str:
        return content
    def is_notebook_file(filename: str) -> bool:
        return False

logger = logging.getLogger(__name__)


def process_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    strategy: str = "hybrid"
) -> List[Document]:
    """
    Split documents using production-grade hybrid strategy.
    
    Strategies:
    - "hybrid": AST + imports + language-aware (RECOMMENDED)
    - "ast": AST-based only (Python-specific)
    - "language": Language-aware splitting only
    - "basic": Simple recursive splitting
    
    Args:
        documents: List of documents from GitHubLoader
        chunk_size: Maximum size of each chunk (in characters)
        chunk_overlap: Number of characters to overlap between chunks
        strategy: Splitting strategy to use
        
    Returns:
        List of enriched document chunks
    """
    logger.info("=" * 60)
    logger.info("🔪 DOCUMENT SPLITTING (HYBRID STRATEGY)")
    logger.info("=" * 60)
    logger.info(f"Input: {len(documents)} documents")
    logger.info(f"Strategy: {strategy}")
    logger.info(f"Chunk size: {chunk_size} chars")
    logger.info(f"Overlap: {chunk_overlap} chars")
    
    if strategy == "hybrid":
        chunks = _hybrid_split(documents, chunk_size, chunk_overlap)
    elif strategy == "ast":
        chunks = _ast_only_split(documents, chunk_size)
    elif strategy == "language":
        chunks = _language_aware_split(documents, chunk_size, chunk_overlap)
    else:
        chunks = _basic_split(documents, chunk_size, chunk_overlap)
    
    _log_statistics(documents, chunks)
    logger.info("=" * 60)
    return chunks


# ============================================================================
# STRATEGY 1: AST-BASED SPLITTING (For Python)
# ============================================================================

def _split_python_with_ast(document: Document, chunk_size: int) -> List[Document]:
    """
    Parse Python code and extract whole functions/classes.
    
    Benefits:
    - Functions stay intact
    - Metadata includes function/class names
    - Easier to cite exact code locations
    """
    try:
        tree = ast.parse(document.page_content)
    except SyntaxError:
        logger.warning(f"AST parse failed for {document.metadata.get('source')}, falling back")
        return []
    
    chunks = []
    lines = document.page_content.split('\n')
    
    # Extract module-level docstring
    module_docstring = ast.get_docstring(tree) or ""
    
    # Extract imports
    imports = _extract_imports(document.page_content)
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            start_line = node.lineno - 1  # 0-indexed
            end_line = node.end_lineno
            
            # Extract code for this function/class
            code_lines = lines[start_line:end_line]
            code = '\n'.join(code_lines)
            
            # Add imports at the top
            if imports:
                code = f"# Imports from {document.metadata['source']}\n{imports}\n\n{code}"
            
            # Create enriched chunk
            chunk = Document(
                page_content=code,
                metadata={
                    **document.metadata,
                    'chunk_type': 'ast_node',
                    'node_type': 'function' if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else 'class',
                    'node_name': node.name,
                    'start_line': start_line + 1,
                    'end_line': end_line,
                    'has_imports': True
                }
            )
            
            # Only add if within size limit or if it's critical
            if len(code) <= chunk_size * 1.5:  # Allow 50% overflow for functions
                chunks.append(chunk)
            else:
                # Function too large - needs further splitting
                logger.debug(f"Function {node.name} too large ({len(code)} chars), splitting")
                # Will be handled by fallback
    
    return chunks


def _extract_imports(code: str) -> str:
    """Extract all import statements from Python code."""
    imports = []
    for line in code.split('\n'):
        stripped = line.strip()
        if stripped.startswith(('import ', 'from ')):
            imports.append(line)
        elif stripped and not stripped.startswith('#'):
            # Stop at first non-import, non-comment line
            break
    return '\n'.join(imports)


# ============================================================================
# STRATEGY 2: IMPORT CONTEXT ENRICHMENT
# ============================================================================

def _enrich_with_imports(
    chunks: List[Document],
    import_map: Dict[str, str]
) -> List[Document]:
    """
    Add import context to chunks that don't already have it.
    """
    for chunk in chunks:
        # Skip if already has imports (from AST)
        if chunk.metadata.get('has_imports'):
            continue
        
        source = chunk.metadata.get('source', '')
        if source in import_map and import_map[source]:
            chunk.page_content = (
                f"# Imports from {source}\n" +
                import_map[source] + "\n\n" +
                chunk.page_content
            )
            chunk.metadata['has_imports'] = True
    
    return chunks


def _build_import_map(documents: List[Document]) -> Dict[str, str]:
    """Build a mapping of source files to their imports."""
    import_map = {}
    for doc in documents:
        source = doc.metadata.get('source', '')
        if source.endswith('.py'):
            imports = _extract_imports(doc.page_content)
            if imports:
                import_map[source] = imports

    return import_map


# ============================================================================
# STRATEGY 3: LANGUAGE-AWARE SPLITTING (Fallback)
# ============================================================================

def _language_aware_split(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int
) -> List[Document]:
    """Split using language-specific splitters."""
    all_chunks = []
    
    for doc in documents:
        source = doc.metadata.get('source', '')
        ext = os.path.splitext(source)[1].lower()
        
        # Language map using the imported Language class/enum
        language_map = {
            '.py': Language.PYTHON,
            '.js': Language.JS,
            '.ts': Language.TS,
            '.md': Language.MARKDOWN,
            '.java': Language.JAVA,
            '.cpp': Language.CPP,
            '.go': Language.GO,
            '.rs': Language.RUST,
        }
        
        language = language_map.get(ext)
        
        if language:
            if hasattr(RecursiveCharacterTextSplitter, 'from_language'):
                splitter = RecursiveCharacterTextSplitter.from_language(
                    language=language,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
            else:
                 # Fallback if from_language static method is missing (rare but possible in old versions)
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        
        chunks = splitter.split_documents([doc])
        all_chunks.extend(chunks)
    
    return all_chunks


def _basic_split(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int
) -> List[Document]:
    """Basic recursive character splitting."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(documents)


# ============================================================================
# HYBRID STRATEGY (ALL 3 COMBINED)
# ============================================================================

def _hybrid_split(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int
) -> List[Document]:
    """
    Production-grade hybrid approach combining all strategies.
    
    Flow:
    1. Try AST-based splitting for Python files
    2. For non-Python or oversized items, use language-aware splitting
    3. Enrich all chunks with import context
    """
    all_chunks = []
    import_map = _build_import_map(documents)
    
    ast_success = 0
    ast_failed = 0
    fallback_used = 0
    
    for doc in documents:
        source = doc.metadata.get('source', '')
        ext = os.path.splitext(source)[1].lower()
        
        # Preprocess Jupyter notebooks
        if ext == '.ipynb' and is_notebook_file(source):
            logger.debug(f"Parsing notebook: {source}")
            parsed_content = parse_notebook(doc.page_content)
            doc = Document(
                page_content=parsed_content,
                metadata={**doc.metadata, 'preprocessed': 'notebook'}
            )
        
        # Strategy 1: Try AST for Python files
        if ext == '.py':
            ast_chunks = _split_python_with_ast(doc, chunk_size)
            
            if ast_chunks:
                # Check if all chunks are reasonable size
                oversized = [c for c in ast_chunks if len(c.page_content) > chunk_size * 1.5]
                
                if not oversized:
                    all_chunks.extend(ast_chunks)
                    ast_success += 1
                    continue
                else:
                    # Some chunks too large - use fallback
                    logger.debug(f"{source}: {len(oversized)} AST chunks oversized, using fallback")
                    ast_failed += 1
            else:
                ast_failed += 1
        
        # Strategy 2: Language-aware fallback
        fallback_used += 1
        chunks = _language_aware_split([doc], chunk_size, chunk_overlap)
        all_chunks.extend(chunks)
    
    # Strategy 3: Enrich all chunks with imports
    all_chunks = _enrich_with_imports(all_chunks, import_map)
    
    logger.info(f"   AST successful: {ast_success} files")
    logger.info(f"   AST failed/oversized: {ast_failed} files")
    logger.info(f"   Fallback used: {fallback_used} files")
    logger.info(f"   Chunks with imports: {sum(1 for c in all_chunks if c.metadata.get('has_imports'))}")
    
    return all_chunks


def _ast_only_split(documents: List[Document], chunk_size: int) -> List[Document]:
    """AST-only strategy (for comparison/testing)."""
    all_chunks = []
    for doc in documents:
        if doc.metadata.get('source', '').endswith('.py'):
            chunks = _split_python_with_ast(doc, chunk_size)
            all_chunks.extend(chunks)
    return all_chunks


# ============================================================================
# UTILITIES
# ============================================================================

def _log_statistics(original_docs: List[Document], chunks: List[Document]) -> None:
    """Log detailed statistics."""
    logger.info("✅ Splitting complete!")
    logger.info(f"   Original documents: {len(original_docs)}")
    logger.info(f"   Generated chunks: {len(chunks)}")
    
    if original_docs and chunks:
        logger.info(f"   Ratio: 1 doc → {len(chunks)/len(original_docs):.1f} chunks (avg)")
        
        total_chars = sum(len(c.page_content) for c in chunks)
        avg_size = total_chars // len(chunks)
        min_size = min(len(c.page_content) for c in chunks)
        max_size = max(len(c.page_content) for c in chunks)
        
        logger.info(f"   Chunk sizes: min={min_size}, avg={avg_size}, max={max_size} chars")
        
        # Count chunk types
        ast_chunks = sum(1 for c in chunks if c.metadata.get('chunk_type') == 'ast_node')
        if ast_chunks:
            logger.info(f"   AST-based chunks: {ast_chunks}")


def get_chunks_by_source(chunks: List[Document], source_file: str) -> List[Document]:
    """Filter chunks by source file."""
    return [c for c in chunks if c.metadata.get('source') == source_file]


def get_chunk_statistics(chunks: List[Document]) -> dict:
    """Get detailed statistics about chunks."""
    if not chunks:
        return {}
    
    sizes = [len(c.page_content) for c in chunks]
    sources = set(c.metadata.get('source', 'unknown') for c in chunks)
    
    ast_chunks = sum(1 for c in chunks if c.metadata.get('chunk_type') == 'ast_node')
    chunks_with_imports = sum(1 for c in chunks if c.metadata.get('has_imports'))
    
    return {
        'total_chunks': len(chunks),
        'unique_sources': len(sources),
        'ast_based_chunks': ast_chunks,
        'chunks_with_imports': chunks_with_imports,
        'total_characters': sum(sizes),
        'avg_chunk_size': sum(sizes) / len(sizes),
        'min_chunk_size': min(sizes),
        'max_chunk_size': max(sizes),
        'sources': sorted(list(sources))
    }


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    """Test all strategies."""
    from services.github_loader import GitHubLoader
    import os
    
    print("Testing document processor with hybrid strategy...")
    
    loader = GitHubLoader(
        repo="srinath2934/execflow-ai",
        access_token=os.getenv("GITHUB_TOKEN")
    )
    docs = loader.load()
    
    print("\n" + "=" * 60)
    print("STRATEGY COMPARISON")
    print("=" * 60)
    
    # Test each strategy
    strategies = ["basic", "language", "ast", "hybrid"]
    
    for strategy in strategies:
        print(f"\n📊 Testing: {strategy.upper()}")
        chunks = process_documents(docs, chunk_size=1000, strategy=strategy)
        stats = get_chunk_statistics(chunks)
        print(f"   Total chunks: {stats['total_chunks']}")
        print(f"   AST chunks: {stats.get('ast_based_chunks', 0)}")
        print(f"   With imports: {stats.get('chunks_with_imports', 0)}")
        print(f"   Avg size: {stats['avg_chunk_size']:.0f} chars")