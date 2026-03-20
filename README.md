# 🚀 RepoChat: AI-Powered Code Intelligence using Endee

**Interview Submission for Machine Learning Intern — Endee**

---

## 📌 Problem Statement

Developers often struggle to understand large and complex codebases efficiently. Traditional search methods rely on keyword matching and fail to capture semantic relationships within code.

This project solves that problem by enabling **semantic search and question answering over repositories**, transforming codebases into intelligent, queryable knowledge systems.

---

## 💡 Solution Overview

RepoChat is a **Retrieval-Augmented Generation (RAG) system** that allows users to ask natural language questions about a code repository and receive context-aware answers.

The system leverages:

- **Embeddings** for semantic understanding  
- **Endee vector database** for fast similarity search  
- **Groq Llama 3** for generating human-readable responses  

---

## 🏛️ System Architecture

```mermaid
graph TD
    A[User Query] --> B[Embedding Model (Sentence-BERT)]
    B --> C[Endee Vector Database]
    C --> D[Top-K Retrieval]
    D --> E[LLM (Response Generation)]
    E --> F[Answer with Code Context]
```

---

## ⚙️ How Endee is Used

Endee serves as the core vector database powering semantic retrieval:

- **Stores embeddings** of code chunks extracted from repositories
- **Performs high-speed similarity search** using vector indexing
- **Returns top-k relevant code segments** for context generation
- **Enables scalable retrieval** for large codebases

---

## 🔍 Why Endee?

Compared to traditional vector stores:

- **C++ Backend**: Enables high-performance computation and low-level optimization.
- **Efficient Memory**: Optimized for large embedding datasets with hybrid RAM management.
- **Low Latency**: Achieves sub-second retrieval even at scale.
- **Production-Ready**: Designed for real-world AI systems requiring persistence and speed.

---

## 🧠 Key Features

- 📂 **Full Repository Ingestion**: Index any GitHub repository in seconds.
- 🧩 **AST-Based Parsing**: Smart code splitting that respects function and class boundaries.
- 🔎 **Semantic Knowledge**: Uses S-BERT for deep understanding of code intent.
- 🤖 **Interactive RAG**: Conversational interface with memory and technical precision.
- 📌 **Source Citations**: Every answer includes links back to specific source files.
- ⚡ **Flash Performance**: Powered by Endee (C++) and Groq (LPU).

---

## 📊 Evaluation & Results

- **Dataset**: GitHub repositories (multi-file codebases).
- **Retrieval Engine**: Endee Vector Core (L2 Space).
- **Observations**:
    - **Speed**: Average query latency ~1.2s.
    - **Accuracy**: Extremely high contextual relevance due to AST chunking.
    - **Scalability**: Handled 100+ chunks with zero degradation.

---

## 🛠️ Tech Stack

- **Core**: Python 3.12, LangChain
- **Vector DB**: Endee (C++ REST Engine)
- **Embeddings**: Sentence-BERT (Hugging Face Inference API)
- **LLM**: Groq Llama 3.3 70B
- **UI**: Streamlit
- **Infra**: Docker, Docker Compose

---

## 📁 Project Structure
- `app.py`: Main application UI and orchestration.
- `services/`: Core logic for embeddings, retrieval, and analysis.
- `infra/`: Docker configuration for the Endee environment.
- `docs/`: Screenshots, thumbnails, and demo assets.
- `docker-compose.yml`: Automated Endee engine orchestration.
- `fix_engine.ps1`: Automated infrastructure recovery script.

---

## 📸 Demo

### 🖥️ Interface
![Landing Page](docs/screenshots/landing_page.png)
![Repo Loaded](docs/screenshots/repo_loaded.png)
![AI Response](docs/screenshots/ai_response.png)

---

## 🎥 Demo Walkthrough
*Check out the live system in action below. If the player does not appear, your browser may have restricted auto-playback.*

![Watch the RepoChat Demo](docs/demo_recording.mp4)

---

## 🚀 Setup Instructions

### 1. Clone & Configure
```bash
git clone https://github.com/srinath2934/RepoChat
cd RepoChat
```

### 2. Environment Variables
Create a `.env` file with:
```bash
GROQ_API_KEY=your_key
HUGGINGFACEHUB_API_TOKEN=your_token
GITHUB_TOKEN=your_token
```

### 3. Start Infrastructure
```bash
# Automated fix (recommended)
powershell -File fix_engine.ps1

# Or manual
docker-compose up -d
```

### 4. Run Application
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🔮 Future Improvements
- [ ] **Hybrid Search**: Combine vector similarity with BM25 keyword matching.
- [ ] **Multi-Repo Context**: Index multiple repositories for cross-project intelligence.
- [ ] **Local LLM Support**: Integration with Ollama for 100% air-gapped security.

---

## 👨‍💻 Author

**Srinath S**  
AI & Data Science Student  
GitHub: [srinath2934](https://github.com/srinath2934)

⭐ **Submission Note**: This project was built for the **Endee Machine Learning Intern** evaluation, focusing on real-world AI system design, vector search, and scalable retrieval pipelines.
