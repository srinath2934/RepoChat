"""
LLM Service - Groq API Integration
Learn: How to use LLMs for generating answers from code context
"""
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from typing import List, Dict
import logging
import os

logger = logging.getLogger(__name__)


def get_llm():
    """
    Initialize Groq LLM
    
    LEARN: Why Groq?
    - 10x faster than OpenAI
    - Free tier available
    - Llama 3.3 quality
    
    Returns:
        ChatGroq instance
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("❌ GROQ_API_KEY not found. Set it in .env file!")
    
    llm = ChatGroq(
        temperature=0,  # 0 = factual, 1 = creative
        model_name="llama-3.3-70b-versatile",
        groq_api_key=api_key
    )
    
    logger.info("🤖 LLM initialized (Llama 3.3)")
    return llm


def generate_answer(query: str, context: str) -> str:
    """
    Generate answer using RAG
    
    LEARN: RAG prevents hallucinations!
    - LLM can ONLY use the provided code
    - Can't make up functionality
    - Cites specific files
    
    Args:
        query: User's question
        context: Retrieved code (formatted)
    
    Returns:
        AI-generated answer
    """
    logger.info(f"🤖 Generating answer for: '{query}'")
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a senior software engineer helping developers understand code.

RULES:
1. Answer ONLY using the provided code context
2. If the answer isn't in the context, say "I don't have enough information"
3. Cite specific files and function names when possible
4. Use code snippets in your explanation
5. Be concise but thorough

Context:
{context}"""),
        ("human", "{question}")
    ])
    
    # Create chain
    llm = get_llm()
    chain = prompt | llm
    
    # Generate
    response = chain.invoke({
        "context": context,
        "question": query
    })
    
    answer = response.content
    logger.info(f"✅ Generated {len(answer)} character answer")
    
    return answer


def generate_answer_with_citations(
    query: str,
    context: str,
    citations: List[Dict]
) -> Dict:
    """
    Generate answer + return with source citations
    
    Returns:
        {
            'answer': str,
            'sources': [...]
        }
    """
    answer = generate_answer(query, context)
    
    # Format citations
    formatted_sources = []
    for cite in citations:
        formatted_sources.append({
            'file': cite['file'],
            'function': cite.get('node_name', 'N/A'),
            'url': cite['url'],
            'lines': cite.get('lines', 'N/A')
        })
    
    return {
        'answer': answer,
        'sources': formatted_sources
    }

