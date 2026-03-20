import requests
import numpy as np
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class EndeeVectorEngine:
    """
    💎 INTERVIEW SUBMISSION Logic:
    This class handles the direct integration with the local Endee C++ Vector Engine.
    Instead of using standard libraries (like Chroma or Pinecone), I implemented this
    custom connector to demonstrate:
    - Low-level REST API integration.
    - Handling high-performance C++ backend communication.
    - Scalable vector management for large-scale RAG.
    """
    
    def __init__(self, host: str = "http://localhost:9999", auth_token: str = ""):
        self.host = host.rstrip('/')
        self.auth_token = auth_token
        self.headers = {
            "Content-Type": "application/json"
        }
        if auth_token:
            self.headers["Authorization"] = auth_token
            
    def check_health(self) -> bool:
        """Verify the Endee engine is running."""
        try:
            response = requests.get(f"{self.host}/api/v1/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Endee Health Check failed on {self.host}: {e}")
            return False

    def create_index(self, index_name: str, dimension: int, space_type: str = "l2"):
        """Create a new index in the Endee engine."""
        payload = {
            "index_name": index_name,
            "dim": dimension,
            "space_type": space_type,
            "precision": "int16" # Consistent with main.cpp default
        }
        try:
            response = requests.post(
                f"{self.host}/api/v1/index/create",
                headers=self.headers,
                json=payload
            )
            if response.status_code not in [200, 409]: # 409 means already exists
                logger.error(f"Failed to create index: {response.text}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            return False

    def upsert_documents(self, index_name: str, documents: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        Insert documents and their embeddings into the Endee index.
        Endee uses HybridVectorObject which takes id, vector, and meta.
        """
        vectors = []
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            # Handle both Document objects (converted to dict) and direct dicts
            page_content = doc.get("page_content", "")
            metadata = doc.get("metadata", {})
            
            # Endee expected format for /vector/insert
            vectors.append({
                "id": str(i),
                "vector": [float(v) for v in emb],
                "meta": json.dumps({
                    "content": page_content,
                    "metadata": metadata
                })
            })

        try:
            response = requests.post(
                f"{self.host}/api/v1/index/{index_name}/vector/insert",
                headers=self.headers,
                json=vectors
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error upserting to Endee: {e}")
            return False

    def similarity_search(self, index_name: str, query_vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Perform KNN search against the local Endee engine."""
        payload = {
            "vector": [float(v) for v in query_vector],
            "k": k
        }
        try:
            # Note: Endee returns results in msgpack if requested, but we'll use JSON if possible 
            # for simpler python integration unless msgpack is required.
            # However, looking at main.cpp, it seems to prefer msgpack for search results.
            # We might need to handle msgpack or ensure JSON response.
            response = requests.post(
                f"{self.host}/api/v1/index/{index_name}/search",
                headers=self.headers,
                json=payload
            )
            
            if response.status_code == 200:
                import msgpack
                # Endee typically returns a list of results when using the /search endpoint
                try:
                    results = msgpack.unpackb(response.content)
                except Exception:
                    # Fallback to JSON if msgpack fails
                    results = response.json()
                
                # If results is a dict (some engine versions return {"results": [...]})
                if isinstance(results, dict) and "results" in results:
                    results = results["results"]
                
                formatted_results = []
                for res in results:
                    # Endee ResultSet item structure: 'meta' or 'metadata' or 'm'
                    # The engine typically returns a string for meta
                    meta_raw = res.get('meta') or res.get('metadata') or res.get('m') or "{}"
                    
                    try:
                        # If it's already a dict (some versions), use it. Otherwise parse.
                        meta_obj = meta_raw if isinstance(meta_raw, dict) else json.loads(meta_raw)
                        
                        # Extract content and metadata from our standardized upsert format
                        content = meta_obj.get("content") or meta_obj.get("page_content") or ""
                        
                        # Use the entire meta_obj as metadata if our nested structure isn't found
                        metadata = meta_obj.get("metadata") or meta_obj
                        
                        formatted_results.append({
                            "page_content": content,
                            "metadata": metadata,
                            "score": res.get("distance", res.get("score", 0))
                        })
                    except Exception as e:
                        logger.warning(f"Could not parse metadata for a result: {e}")
                        continue
                        
                return formatted_results
            return []
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []