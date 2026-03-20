"""
🤖 RepoChat - Production Streamlit App
Ask questions about any GitHub repository using AI-powered semantic search
"""

import streamlit as st
import os
from dotenv import load_dotenv
import time
from typing import List, Dict
import logging
import warnings

# Suppress annoying "false alarm" warnings from torch/streamlit on Windows
warnings.filterwarnings("ignore", message="Tried to instantiate class")
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("torch").setLevel(logging.ERROR)
from services.github_loader import GitHubLoader
from services.repo_analyzer import generate_repo_summary, format_repo_analysis
from services.document_processor import process_documents
from services.embeddings import get_embeddings_model
from services.retrieval import retrieve_context, format_context_for_llm, get_citations
from services.vector_store import EndeeVectorEngine
from langchain_groq import ChatGroq
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# PAGE CONFIG - Must be first Streamlit command
# ============================================================================

st.set_page_config(
    page_title="RepoChat: Instant GitHub Q&A",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - Modern, Premium Design
# ============================================================================

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Charter:wght@400;500;600;700&family=Source+Sans+Pro:wght@400;600&display=swap');
    
    /* 
       GLOBAL COLOR OVERRIDE 
       Forces dark text regardless of system dark/light mode for our light beige theme 
    */
    .stApp, .stMarkdown, p, h1, h2, h3, h4, h5, h6, li, span, label, div {
        color: #2d2d2d !important;
    }

    * {
        font-family: 'Source Sans Pro', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main {
        background: #f7f5f2;
        padding: 2rem;
    }
    
    .stApp {
        background: #f7f5f2;
    }
    
    .header {
        background: #ffffff;
        color: #2d2d2d;
        padding: 2.5rem 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid #e8e3dc;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
    }
    
    /* Sidebar glassmorphism-ish style */
    [data-testid="stSidebar"] {
        background: #ebe8e3;
        border-right: 1px solid #d4cfc4;
    }
    
    /* Quick Select Cards */
    .repo-card {
        background: #ffffff;
        border: 1px solid #d4cfc4;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .repo-card:hover {
        border-color: #9a8f7f;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transform: translateY(-2px);
    }

    /* Primary and Accent Buttons */
    .stButton > button {
        background: #ffffff !important;
        color: #2d2d2d !important;
        border: 1px solid #d4cfc4 !important;
        border-radius: 8px;
        padding: 0.625rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background: #f7f5f2 !important;
        border-color: #9a8f7f !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.12);
    }
    
    /* Status feedback */
    .status-badge {
        display: inline-block;
        padding: 0.4rem 0.875rem;
        border-radius: 16px;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .status-warning { background: #fff3e0; color: #e65100; border: 1px solid #ffe0b2; }
    .status-success { background: #e8f5e9; color: #2e7d32; border: 1px solid #c8e6c9; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize all session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None
    
    if 'llm' not in st.session_state:
        st.session_state.llm = None
    
    if 'repo_loaded' not in st.session_state:
        st.session_state.repo_loaded = False
    
    if 'current_repo' not in st.session_state:
        st.session_state.current_repo = None
    
    if 'document_count' not in st.session_state:
        st.session_state.document_count = 0

# ============================================================================
# BACKEND FUNCTIONS
# ============================================================================

@st.cache_resource
def initialize_llm():
    """Initialize the Groq LLM"""
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            st.error("❌ GROQ_API_KEY not found in .env file")
            return None
        
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile",  # Correct working model
            temperature=0.3,
            max_tokens=2048
        )
        
        logger.info("✅ Groq LLM initialized")
        return llm
    except Exception as e:
        st.error(f"❌ Failed to initialize LLM: {e}")
        return None

@st.cache_resource
def initialize_embeddings():
    """Initialize embeddings model"""
    try:
        embeddings = get_embeddings_model()
        logger.info("✅ Embeddings model initialized")
        return embeddings
    except Exception as e:
        st.error(f"❌ Failed to initialize embeddings: {e}")
        return None

def load_github_repo(repo_url: str, branch: str = "main") -> tuple:
    """Load and process GitHub repository"""
    try:
        # Step 1: Load repository
        with st.status("🔄 Loading GitHub repository...", expanded=True) as status:
            st.write("📦 Connecting to GitHub...")
            
            loader = GitHubLoader(
                repo=repo_url,
                branch=branch,
                access_token=os.getenv("GITHUB_TOKEN"),
                commit_history_limit=30
            )
            
            # Create progress bar
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            def update_progress(current, total):
                """Callback to update progress bar"""
                progress = current / total if total > 0 else 0
                progress_bar.progress(progress)
                progress_text.text(f"📥 Downloading files: {current}/{total}")
            
            st.write("📥 Downloading files...")
            documents = loader.load(progress_callback=update_progress, load_commits=False)
            progress_bar.progress(1.0)
            progress_text.text(f"✅ Loaded {len(documents)} documents")
            st.write(f"✅ Loaded {len(documents)} documents")
            
            # Step 2: Process documents
            st.write("🔪 Splitting documents into chunks...")
            chunks = process_documents(
                documents,
                chunk_size=1000,
                chunk_overlap=200,
                strategy="hybrid"
            )
            st.write(f"✅ Created {len(chunks)} chunks")
            
            # Step 3: Create embeddings
            st.write("🧠 Connecting to HuggingFace Inference API...")
            embeddings = initialize_embeddings()
            if not embeddings:
                return None, None, "HUGGINGFACEHUB_API_TOKEN is missing in .env. Please add it to use Cloud Embeddings.", None
            
            # Step 4: Create vector store using Endee
            st.write("💾 Synchronizing with Endee Vector Database...")
            vectorstore = EndeeVectorEngine()
            if not vectorstore.check_health():
                return None, None, f"Endee Engine is not running on {vectorstore.host}. Please start it using Docker.", None
            
            # Create index for this repo (slugify the name aggressively)
            index_name = "".join(c if c.isalnum() else "_" for c in repo_url)
            vectorstore.create_index(index_name, dimension=384) # 384 for all-MiniLM-L6-v2
            
            # 🌟 FIX: Extract texts for embedding first
            st.write("🧠 Generating embeddings...")
            chunk_texts = [c.page_content for c in chunks]
            embeddings_list = embeddings.embed_documents(chunk_texts)
            
            # Convert documents to list of dicts for Endee
            chunk_dicts = [c.dict() for c in chunks]
            
            # Upsert
            vectorstore.upsert_documents(index_name, chunk_dicts, embeddings_list)
            st.session_state.current_index = index_name
            
            status.update(label="✅ Repository loaded and synced with Endee!", state="complete")
        
        return vectorstore, len(chunks), None, documents
        
    except Exception as e:
        return None, None, str(e), None

def get_file_list_from_vectorstore(vectorstore: EndeeVectorEngine) -> str:
    """Get all unique files from the Endee vectorstore metadata"""
    try:
        index_name = st.session_state.get('current_index')
        if not index_name:
            return "### 📁 Repository Structure\n\nNo repository indexed."
            
        # For Endee, we fetch a large batch to extract unique sources
        # Since we don't have a direct 'unique' filter in the engine yet, 
        # we retrieve top 500 vectors with a dummy query
        results = vectorstore.similarity_search(index_name, [0.0]*384, k=500)
        
        logger.info(f"Retrieved {len(results)} documents from Endee for file listing")
        
        logger.info(f"Retrieved {len(all_docs['ids'])} documents from vectorstore")
        
        # Extract unique file paths
        files = set()
        for res in results:
            metadata = res.get('metadata', {})
            source = metadata.get('source', '')
            if source and source != 'git_history' and not source.startswith('[BINARY'):
                files.add(source)
        
        logger.info(f"Found {len(files)} unique files")
        
        # Sort and format
        sorted_files = sorted(list(files))
        
        if not sorted_files:
            return "### ⚠️ No files found in the repository.\n\nThis might indicate an issue with repository loading."
        
        # Group by directory
        file_tree = {}
        for file in sorted_files:
            if '/' in file:
                dir_name = file.rsplit('/', 1)[0]
                file_name = file.rsplit('/', 1)[1]
                if dir_name not in file_tree:
                    file_tree[dir_name] = []
                file_tree[dir_name].append(file_name)
            else:
                if 'root' not in file_tree:
                    file_tree['root'] = []
                file_tree['root'].append(file)
        
        # Format output
        output = ["### 📁 Repository Structure\n"]
        
        # Root files first
        if 'root' in file_tree:
            output.append("**Root files:**")
            for file in sorted(file_tree['root']):
                output.append(f"- `{file}`")
            output.append("")
        
        # Then directories
        for dir_name in sorted([d for d in file_tree.keys() if d != 'root']):
            output.append(f"**`{dir_name}/`**")
            for file in sorted(file_tree[dir_name]):
                output.append(f"  - `{file}`")
            output.append("")
        
        output.append(f"\n**Total files:** {len(sorted_files)}")
        
        return "\n".join(output)
    except Exception as e:
        logger.error(f"Error getting file list: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return f"Unable to retrieve file list. Error: {str(e)}"

def generate_answer(query: str, vectorstore: EndeeVectorEngine, llm) -> Dict:
    """Generate answer using RAG"""
    try:
        # Special case: Detect file listing questions
        file_keywords = ['what files', 'list files', 'show files', 'files in', 'repo structure', 'repository structure', 'what is in']
        if any(keyword in query.lower() for keyword in file_keywords):
            file_list = get_file_list_from_vectorstore(vectorstore)
            return {
                "answer": file_list,
                "citations": [],
                "context_used": 0
            }
        
        # Step 1: Retrieve relevant context
        index_name = st.session_state.get('current_index')
        results = retrieve_context(query, vectorstore, index_name, k=5)
        
        if not results:
            return {
                "answer": "I couldn't find any relevant information in the repository to answer your question.",
                "citations": [],
                "context_used": 0
            }
        
        # Step 2: Format context for LLM
        context = format_context_for_llm(results)
        
        # Step 3: Generate answer
        # UPDATED PROMPT: Enforce structured, precise output
        prompt = f"""You are a senior software engineer assistant. Answer the user's question about the repository.

CONTEXT FROM REPOSITORY:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
1. Be **precise** and avoid fluff.
2. Use **Markdown structure** (headers, bullet points).
3. Start with a direct answer.
4. If showing code, use syntax highlighting.
5. Reference specific files/functions.

FORMAT:
### Summary
[Brief summary of the answer]

### Key Details
- [Bullet point 1]
- [Bullet point 2]
- [Bullet point 3]

### Code Reference (if applicable)
```[language]
[code snippet]
```

### 🛠️ Command Breakdown (if applicable)
- `[command]`: [What this command does and why it is used]
"""
        
        response = llm.invoke(prompt)
        answer = response.content
        
        # Step 4: Get citations
        citations = get_citations(results)
        
        return {
            "answer": answer,
            "citations": citations,
            "context_used": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return {
            "answer": f"Error generating answer: {str(e)}",
            "citations": [],
            "context_used": 0
        }

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_header():
    """Render app header"""
    # Only show the big header if no repository is loaded
    if not st.session_state.repo_loaded:
        st.markdown("""
        <div class="header">
            <h1>🤖 RepoChat</h1>
            <p>Instant AI answers for any GitHub repository</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Minimal header for chat mode
        st.markdown("""
        <div style="padding: 1rem 0; text-align: center; border-bottom: 1px solid #e8e3dc; margin-bottom: 1rem;">
            <span style="font-size: 1.2rem; font-weight: 600; color: #2d2d2d;">🤖 RepoChat</span>
            <span style="color: #d4cfc4; margin: 0 0.5rem;">|</span>
            <span style="color: #6b6b6b;">{repo}</span>
        </div>
        """.format(repo=st.session_state.current_repo), unsafe_allow_html=True)

def render_sidebar():
    """Render sidebar with repo loading"""
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # State management for toggling the input form
        if 'show_load_form' not in st.session_state:
            st.session_state.show_load_form = True
            
        # PROGESSIVE DISCLOSURE: Only show input form if needed
        if not st.session_state.repo_loaded or st.session_state.get('show_load_form', False):
            # 🌟 VISUAL DISCOVERY: Quick-Start Cards
            st.subheader("🌟 Visual Discovery")
            st.write("Pick a popular repo to start instantly:")
            
            # Curated popular repositories
            EXAMPLES = [
                {"repo": "langchain-ai/langchain", "name": "LangChain", "icon": "🦜"},
                {"repo": "facebook/react", "name": "React", "icon": "⚛️"},
                {"repo": "fastapi/fastapi", "name": "FastAPI", "icon": "🚀"},
                {"repo": "srinath2934/execflow-ai", "name": "ExecFlow", "icon": "⚙️"},
            ]
            
            # Create a 2x2 grid for visuals
            col_a, col_b = st.columns(2)
            selected_example = None
            
            for i, example in enumerate(EXAMPLES):
                target_col = col_a if i % 2 == 0 else col_b
                with target_col:
                    if st.button(f"{example['icon']} {example['name']}", use_container_width=True, help=f"Explore {example['repo']}"):
                        selected_example = example['repo']

            st.divider()
            
            # Manual Repository input section
            st.subheader("📦 Manual Load")
            
            repo_url = st.text_input(
                "GitHub Repository",
                placeholder="https://github.com/facebook/react",
                help="Paste the full GitHub URL or enter owner/repo format",
                key="repo_input"
            )
            
            branch = st.text_input(
                "Branch",
                value="main",
                help="Branch name (usually 'main' or 'master')",
                key="branch_input"
            )
            
            load_button = st.button("🚀 Load Repository", use_container_width=True, type="primary")
            
            # Handle loading (either from button or example card)
            active_repo = selected_example if selected_example else (repo_url if load_button else None)
            
            if active_repo:
                # Clear previous state
                st.cache_resource.clear()
                
                with st.spinner(f"✨ Magical extraction of {active_repo}..."):
                    vectorstore, doc_count, error, raw_docs = load_github_repo(active_repo, branch)
                    
                    if error:
                        st.error(f"❌ Error: {error}")
                    else:
                        st.session_state.vectorstore = vectorstore
                        st.session_state.document_count = doc_count
                        st.session_state.repo_loaded = True
                        st.session_state.current_repo = active_repo
                        st.session_state.show_load_form = False
                        
                        # Generate Objective Analysis
                        if raw_docs:
                            analysis_summary = generate_repo_summary(raw_docs, active_repo)
                            welcome_msg = format_repo_analysis(analysis_summary)
                        else:
                            welcome_msg = f"🎉 **Magical Ingestion Complete!**\n\nI've analyzed `{active_repo}` labels."

                        st.session_state.messages = [{
                            "role": "assistant",
                            "content": welcome_msg
                        }]
                        st.balloons()
                        st.rerun()
        
        else:
            # COMPACT STATE: Show when repo is loaded
            st.success(f"✅ **Active:** {st.session_state.current_repo}")
            if st.button("🔄 Load New Repository"):
                st.session_state.show_load_form = True
                st.rerun()
        
        st.divider()
        
        # Current repository status
        st.subheader("📊 Status")
        
        if st.session_state.repo_loaded:
            # Simplified status since top part already confirms usage
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Docs", st.session_state.document_count)
            with col2:
                st.metric("Msgs", len(st.session_state.messages))
        else:
            if not st.session_state.get('show_load_form', True):
                 # Should not happen ideally but fallback
                 st.info("👆 Load a repository above")
            elif 'status_placeholder' in locals():
                 with status_placeholder.container():
                    st.markdown("""
                    <div class="status-badge status-warning">⚠️ No Repository Loaded</div>
                    """, unsafe_allow_html=True)
                    st.info("👆 Load a repository above to start!")
            else:
                 # Initial render state
                 st.markdown("""
                <div class="status-badge status-warning">⚠️ No Repository Loaded</div>
                """, unsafe_allow_html=True)
        
        st.divider()
        
        # ==========================================
        # 🎭 NEW FEATURE: AI Persona Selector
        # ==========================================
        st.subheader("🎭 AI Persona")
        
        # This selectbox returns the string literal chosen by the user
        selected_persona = st.selectbox(
            "Choose your assistant style:",
            ["Senior Engineer", "ELI5 (Explain Like I'm 5)", "Pirate 🏴‍☠️"],
            index=0,
            help="Changes the tone and complexity of the AI's answers",
            key="persona_selector" # Unique key ensures state is preserved
        )
        
        st.divider()
        
        st.subheader("🔧 Quick Actions")
        
        # Use standard buttons which are clearer than the custom dark ones
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("🔄 Reset All"):
            st.session_state.clear()
            st.cache_resource.clear()
            st.rerun()
        
        st.divider()
        
        # Info - More concise
        st.subheader("✨ Features")
        st.markdown("""
        🔍 Smart code search  
        🤖 AI-powered answers  
        📚 Source citations  
        ⚡ Fast & accurate
        """)
        
        st.caption("Powered by Groq LLaMA 3.3 70B")

def render_chat_interface():
    """Render main chat interface"""
    
    # Check if repository is loaded
    if not st.session_state.repo_loaded:
        st.markdown("""
        ##  Welcome to RepoChat!
        
        **Get started in 3 easy steps:**
        
        1. 📦 **Load a repository** from the sidebar
        2. ✍️ **Ask questions** about the code
        3. 📚 **Get AI-powered answers** with source citations
        """)
        
        st.divider()
        
        # Show example queries
        st.subheader("💡 Try These Example Questions:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **General Questions:**
            - What does this repository do?
            - How is the project structured?
            - What are the main features?
            
            **Technical Questions:**
            - How does authentication work?
            - Show me the API endpoints
            - Where is error handling implemented?
            """)
        
        with col2:
            st.markdown("""
            **Code Questions:**
            - What is the main entry point?
            - How is data validated?
            - Show me the database schema
            
            **Deep Dive:**
            - Explain the login function
            - How does caching work?
            - What dependencies are used?
            """)
        
        return
    
    # Display chat messages
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <strong>👤 You:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="bot-message">
                <strong>🤖 Assistant:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
            
            # Show citations if available
            if "citations" in message and message["citations"]:
                with st.expander("📚 View Sources", expanded=False):
                    for idx, citation in enumerate(message["citations"], 1):
                        st.markdown(f"""
                        <div class="citation">
                            <strong>📄 Source {idx}:</strong> {citation['file']}<br>
                            <strong>📍 Lines:</strong> {citation['lines']}<br>
                            {f"<strong>🔧 Function/Class:</strong> {citation['node_name']}<br>" if citation['node_name'] else ""}
                            <a href="{citation['url']}" target="_blank">🔗 View on GitHub</a>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Chat input - MOVED OUTSIDE of any conditions to ensure it's always accessible
    prompt = st.chat_input("💬 Ask me anything about this repository...")
    
    if prompt:
        # Check if repo is loaded
        if not st.session_state.repo_loaded:
            st.error("⚠️ Please load a repository from the sidebar first!")
            return

        # Initialize LLM if needed
        if not st.session_state.llm:
            with st.spinner("🔧 Initializing AI model..."):
                st.session_state.llm = initialize_llm()
        
        if not st.session_state.llm:
            st.error("❌ **Failed to initialize AI model**")
            st.info("💡 **Fix:** Check that your `GROQ_API_KEY` in the `.env` file is correct")
            return
        
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Generator for structured response
        with st.chat_message("assistant", avatar="🤖"):
            try:
                with st.spinner("🤔 Thinking..."):
                    # 1. Retrieve Context (This still takes a moment)
                    index_name = st.session_state.get('current_index')
                    results = retrieve_context(prompt, st.session_state.vectorstore, index_name, k=5)
                    context = format_context_for_llm(results) if results else "No context found."
                
                # 2. Prepare Prompt based on Persona
                persona = st.session_state.get("persona_selector", "Senior Engineer")
                
                # Define persona-specific instructions
                persona_instructions = {
                    "Senior Engineer": "You are a senior technical writer and software engineer. Use professional technical terminology.",
                    "ELI5 (Explain Like I'm 5)": "You are a friendly teacher explaining to a beginner. Use simple analogies, avoid jargon, and explain concepts step-by-step.",
                    "Pirate 🏴‍☠️": "You are a coding pirate captain! Yarr! Use pirate slang (matey, treasure, ship) but keep the technical details accurate."
                }
                
                system_role = persona_instructions.get(persona, persona_instructions["Senior Engineer"])
                
                structured_prompt = f"""{system_role}
                
                CONTEXT:
                {context}
                
                QUESTION:
                {prompt}
                
                STRICT OUTPUT FORMAT (Markdown):
                
                ### 🎯 Summary
                [Direct, 1-sentence answer in the requested persona style]
                
                ### 🔍 Key Details
                - [Bullet point 1]
                - [Bullet point 2]
                - [Bullet point 3]
                
                ### 💻 Code Reference
                (Only if relevant, otherwise omit this section)
                ```[language]
                [code snippet]
                ```
                
                ### 🛠️ Command Breakdown
                (If you suggested terminal commands like pip, git, or python, explain exactly what each flag/part does here)
                - `[command part]`: [Explanation]

                ### 🔗 Source Files
                (List the filenames used)
                """
                
                # 3. Stream the Response
                stream = st.session_state.llm.stream(structured_prompt)
                response_text = st.write_stream(stream)
                
                # 4. Show Citations below the stream
                citations = get_citations(results) if results else []
                if citations:
                    with st.expander("📚 View Sources", expanded=False):
                        for idx, citation in enumerate(citations, 1):
                            st.markdown(f"""
                            <div class="citation">
                                <strong>📄 Source {idx}:</strong> {citation['file']}<br>
                                <strong>📍 Lines:</strong> {citation['lines']}<br>
                                {f"<strong>🔧 Function/Class:</strong> {citation['node_name']}<br>" if citation['node_name'] else ""}
                                <a href="{citation['url']}" target="_blank">🔗 View on GitHub</a>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Add complete message to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "citations": citations
                })
                
            except Exception as e:
                st.error(f"Error: {e}")

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application"""
    
    # Initialize session state
    init_session_state()
    
    # Render UI
    render_header()
    render_sidebar()
    render_chat_interface()

if __name__ == "__main__":
    main()
