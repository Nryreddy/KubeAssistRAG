# ☸️ KubeAssistRAG: Kubernetes Operations Copilot

**KubeAssistRAG** is an advanced, Kubernetes-aware Retrieval-Augmented Generation (RAG) system and operational copilot. It combines multi-strategy search (Dense, Sparse, Hybrid), state-of-the-art retrieval techniques (Reranking, CRAG, HyDE, Self-RAG), and Text2SQL routing to intelligently answer complex operational questions, troubleshoot cluster issues, and execute authorized SQL queries against operational databases.

---

## 🚀 Features & Advanced RAG Techniques

KubeAssistRAG implements a modular, LangGraph-orchestrated pipeline that natively supports the following features:

- **Multi-Vector Search Options**:
  - **Dense Search**: Semantic embeddings for conceptual queries (e.g., "How do Pods share resources?").
  - **Sparse Search (BM25)**: Exact token matching, critical for Kubernetes resource names, camelCase identifiers, and specific labels (e.g., `imagePullPolicy`).
  - **Hybrid Search**: Fuses Dense and Sparse retrieval using Reciprocal Rank Fusion (RRF) for the best of both worlds.
- **Cross-Encoder Reranking**: Re-scores and re-orders the top chunks retrieved by the vector database to dramatically boost accuracy.
- **HyDE (Hypothetical Document Embeddings)**: Generates a hypothetical answer to user queries to bridge the vocabulary gap between beginner questions and technical documentation.
- **CRAG (Corrective RAG)**: Automatically evaluates the quality of retrieved documents. If the corpus lacks the answer (e.g., for out-of-corpus questions like new K8s releases), it falls back to a Tavily Web Search.
- **Self-RAG (Self-Reflective RAG)**: Employs an LLM to dynamically critique its own generated answers. If an answer is vague or unhelpful, the system automatically refines the query and retries.
- **Text2SQL Auto-Routing**: Detects questions regarding cluster metrics or incidents and automatically routes them to a specialized Vanna SQL agent to query the internal Postgres database directly.
- **Semantic Caching**: Implements Upstash Redis to cache identical or semantically similar queries, dropping response times from ~9s (cold) to ~3.5s (cached).
- **Prompt Injection Security**: Secures endpoints with Pydantic-based LLM-Guard injection detection.

---

## 🛠️ Tech Stack

- **Backend / API**: FastAPI, Uvicorn
- **Frontend**: Streamlit
- **Agent Orchestration**: LangGraph
- **LLM / Embeddings**: OpenAI, Voyage AI, SentenceTransformers
- **Vector Database**: Qdrant
- **Relational DB**: PostgreSQL (with pgvector)
- **Caching**: Upstash Redis
- **Text2SQL**: Vanna
- **Document Parsing**: Docling

---

## 📦 Setup & Installation

### Prerequisites
- Docker & Docker Compose
- [uv](https://github.com/astral-sh/uv) (Extremely fast Python package installer and resolver)
- API Keys: OpenAI, Upstash Redis, Tavily (optional, for web fallback)

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/KubeAssistRAG.git
cd KubeAssistRAG
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory based on your API keys:
```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
UPSTASH_REDIS_URL=https://...
UPSTASH_REDIS_TOKEN=...
JWT_SECRET=super_secret_string
```

### 3. Start Infrastructure (Postgres & Qdrant)
Run the required databases using Docker Compose:
```bash
docker-compose up -d postgres qdrant
```

### 4. Install Dependencies & Start the API Server
Using `uv`, you can easily sync your environment and start the FastAPI backend:
```bash
uv sync
uv run python scripts/serve.py
```
*The API will be available at `http://localhost:8000`.*

### 5. Launch the Streamlit Frontend
In a separate terminal, run the Copilot UI:
```bash
uv run streamlit run scripts/streamlit_app.py
```
*The UI will automatically open in your browser.*

---

## 💻 Usage

Once the UI is running, you can explore the various pre-configured **RAG Scenarios** directly from the "Query" tab. These scenarios allow you to trigger and observe specific advanced RAG features in isolation:

1. **Sparse Search**: Test exact token matches (`imagePullPolicy`).
2. **Hybrid Search**: See how RRF fuses dense and exact searches.
3. **CRAG**: Ask about something not in the Kubernetes docs (e.g. "latest Kubernetes 1.34 release features") to trigger the Tavily Web Search fallback.
4. **Self-RAG**: Ask a vague question like "how do I scale" to watch the reflection loop evaluate and retry the search.
5. **Incident Database**: Ask "How many P1 incidents occurred last month?" to see the agent seamlessly route to the Text2SQL pipeline.

### Evaluation Dashboard
The application includes a built-in Ragas evaluation framework to track Faithfulness, Context Precision, Context Recall, and Answer Relevancy across different retrieval profiles. View the **Evaluation Results** tab in the UI to compare pipelines (e.g., Naive vs. Hybrid).

---

## 📄 License
This project is licensed under the MIT License.
