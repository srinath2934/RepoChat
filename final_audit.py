import requests
import json
import msgpack
from sentence_transformers import SentenceTransformer

HOST = "http://localhost:9999"

def test_rag_pipeline():
    print("🔍 Final Engineering Audit: RAG Pipeline Pipeline Check...")
    
    # 1. Health
    try:
        if requests.get(f"{HOST}/api/v1/health").status_code != 200:
            print("❌ Engine offline.")
            return
    except:
        return

    # 2. Aggressive Slugify name
    repo_url = "https://github.com/srinath2934/RepoChat"
    index_name = "".join(c if c.isalnum() else "_" for c in repo_url)
    
    # 3. Embed Query
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query = "What is the purpose of this project?"
    vector = model.encode(query).tolist()
    
    # 4. Pure Msgpack Search
    payload = {"vector": vector, "k": 3}
    resp = requests.post(f"{HOST}/api/v1/index/{index_name}/search", json=payload)
    
    if resp.status_code == 200:
        # Test my new unpacking logic
        try:
            results = msgpack.unpackb(resp.content, raw=False)
            if results:
                print(f"✅ RETRIEVAL SUCCESSFUL!")
                print(f"   Matches: {len(results)}")
                
                # Verify first result has content
                res = results[0]
                meta_raw = "{}"
                if isinstance(res, list):
                    for item in res:
                        if isinstance(item, str) and item.startswith('{"'):
                            meta_raw = item
                else:
                    meta_raw = res.get('meta', '{}')
                
                obj = json.loads(meta_raw)
                print(f"   Content Found: {obj.get('content')[:50]}...")
            else:
                print("❌ Search returned 0 results.")
        except Exception as e:
            print(f"❌ Diagnostic failed: {e}")
    else:
        print(f"❌ Index not found or search failed: {resp.status_code}")

if __name__ == "__main__":
    test_rag_pipeline()
