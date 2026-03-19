"""
Retrieval Service - Smart Code Search
Learn: How to find relevant code using semantic similarity
"""
from typing import List, Dict
from langchain_core.documents import Document
from services.vector_store import EndeeVectorEngine
from services.embeddings import get_embeddings_model
import logging

logger = logging.getLogger(__name__)


def retrieve_context(
    query: str,
    vectorstore: EndeeVectorEngine,
    index_name: str,
    k: int = 5,
    filter_git_history: bool = True
) -> List[Document]:
    """
    LEARN: Semantic search (meaning-based, not keyword matching)
    
    Example:
    Query: "How do I login?"
    Finds: Functions named authenticate(), verify_user(), etc.
    
    How it works:
    1. Convert query → embedding vector
    2. Find k closest vectors in database (cosine similarity)
    3. Filter out git history if asking about code
    4. Return the corresponding code chunks
    
    Args:
        query: User's question
        vectorstore: The ChromaDB instance
        k: Number of results to return
        filter_git_history: Whether to exclude git history from results
    
    Returns:
        List of relevant code chunks
    """
    logger.info(f"🔍 Searching for: '{query}'")
    
    # Convert query to vector
    embeddings_model = get_embeddings_model()
    query_vector = embeddings_model.embed_query(query)
    
    # Fetch more results to account for filtering
    fetch_k = k * 3 if filter_git_history else k
    
    # Semantic similarity search using Endee
    raw_results = vectorstore.similarity_search(index_name, query_vector, k=fetch_k)
    
    # Convert Endee results to LangChain Documents for consistency
    results = []
    for res in raw_results:
        results.append(Document(
            page_content=res.get("page_content", ""),
            metadata=res.get("metadata", {})
        ))
    
    # Filter out git history if requested
    if filter_git_history:
        code_results = [
            doc for doc in results 
            if doc.metadata.get('source', '') != 'git_history'
        ]
        # Take top k after filtering
        results = code_results[:k]
    
    logger.info(f"✅ Found {len(results)} relevant chunks")
    
    return results


def format_context_for_llm(results: List[Document]) -> str:
    """
    LEARN: Format retrieved code for LLM consumption
    
    Why format?
    - LLMs need clear structure
    - Need to know which file each code came from
    - Better formatting = better answers
    """
    context_parts = []
    
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get('source', 'unknown')
        node_name = doc.metadata.get('node_name', '')
        
        # Format each chunk
        header = f"## Code Reference {i}"
        if node_name:
            header += f" - Function/Class: {node_name}"
        header += f"\nFile: {source}\n"
        
        context_parts.append(f"{header}\n```python\n{doc.page_content}\n```\n")
    
    return "\n".join(context_parts)


def get_citations(results: List[Document]) -> List[Dict]:
    """
    LEARN: Extract source citations for UI display
    
    Why citations?
    - Transparency (user sees where answer came from)
    - Clickable links to GitHub
    - Builds trust
    """
    citations = []
    
    for doc in results:
        start_line = doc.metadata.get('start_line')
        end_line = doc.metadata.get('end_line')
        
        # Format line numbers properly
        if start_line and end_line:
            lines = f"{start_line}-{end_line}"
        else:
            lines = "N/A"  # For git history or files without line info
        
        citation = {
            'file': doc.metadata.get('source', 'unknown'),
            'url': doc.metadata.get('url', ''),
            'node_name': doc.metadata.get('node_name', ''),
            'lines': lines
        }
        citations.append(citation)
    
    return citations
