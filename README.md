# ThriveBot - RAG-Based Slack FAQ Bot for Thrive Scholars

A production-ready, scalable Slack bot that uses Retrieval-Augmented Generation (RAG) to answer questions from your knowledge base. Built with FastAPI, Slack Bolt, FAISS, and Google's Gemini API.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OFFLINE KNOWLEDGE INGESTION PIPELINE                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  📄 Documents → 📝 Chunking (500 tokens) → 🔢 Gemini Embeddings → 🗄️ FAISS  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ONLINE INFERENCE PIPELINE (REAL-TIME)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  👤 User Query → 🔍 Semantic Search → 📚 Top-K Context → 🤖 Gemini Pro LLM   │
│                                                              │               │
│                                                              ▼               │
│  💬 Slack Response ◄────────────────────────── Generated Answer              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
ThriveBot/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py               # Configuration management
│   ├── slack_bot.py            # Slack Bolt integration
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── document_loader.py  # Load PDFs, text files, etc.
│   │   ├── chunker.py          # Text chunking logic
│   │   └── embedder.py         # Gemini embedding generation
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_store.py     # FAISS vector database
│   │   └── retriever.py        # Semantic search & retrieval
│   ├── generation/
│   │   ├── __init__.py
│   │   └── llm.py              # Gemini Pro LLM integration
│   └── utils/
│       ├── __init__.py
│       └── helpers.py          # Utility functions
├── data/
│   ├── documents/              # Source documents (PDFs, texts)
│   └── vector_store/           # Persisted FAISS index
├── scripts/
│   ├── ingest.py               # Document ingestion script
│   └── test_bot.py             # Local testing script
├── tests/
│   └── test_retrieval.py       # Unit tests
├── .env.example                # Environment variables template
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Google Cloud account with Gemini API access
- Slack workspace with admin access

### 1. Clone and Setup

```bash
cd ThriveBot
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

### 3. Create Slack App

1. Go to [Slack API](https://api.slack.com/apps)
2. Click "Create New App" → "From scratch"
3. Name it "ThriveBot" and select your workspace
4. Enable these features:
   - **OAuth & Permissions**: Add scopes:
     - `app_mentions:read`
     - `chat:write`
     - `channels:history`
     - `groups:history`
     - `im:history`
     - `mpim:history`
   - **Event Subscriptions**: Enable and add:
     - `app_mention`
     - `message.im`
   - **Socket Mode**: Enable for local development
5. Install to workspace and copy tokens to `.env`

### 4. Ingest Documents

```bash
# Add your PDF/text files to data/documents/
python scripts/ingest.py
```

### 5. Run the Bot

```bash
# Development mode
python -m app.main

# Or with uvicorn
uvicorn app.main:app --reload --port 8000
```

## 🐳 Docker Deployment

```bash
# Build and run
docker-compose up --build

# Production deployment
docker-compose -f docker-compose.yml up -d
```

## 🔧 Configuration

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API key | ✅ |
| `SLACK_BOT_TOKEN` | Slack Bot OAuth token (xoxb-...) | ✅ |
| `SLACK_APP_TOKEN` | Slack App-level token (xapp-...) | ✅ |
| `SLACK_SIGNING_SECRET` | Slack signing secret | ✅ |
| `VECTOR_STORE_PATH` | Path to FAISS index | ❌ |
| `CHUNK_SIZE` | Token size for chunks (default: 500) | ❌ |
| `TOP_K` | Number of contexts to retrieve (default: 5) | ❌ |

## 📖 Usage

Once running, interact with ThriveBot in Slack:

- **Direct Message**: Just send a message
- **In Channels**: Mention `@ThriveBot` with your question

Example:
```
@ThriveBot What are the scholarship requirements?
```

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Test locally without Slack
python scripts/test_bot.py "What is Thrive Scholars?"
```

## 📈 Scaling for Production

1. **Azure Container Apps**: Use the provided Dockerfile
2. **Vector Store**: Switch to Azure AI Search for larger datasets
3. **Caching**: Add Redis for response caching
4. **Monitoring**: Integrate with Azure Application Insights

## 📝 License

MIT License - Feel free to use for your internship!

---

Built with ❤️ for Thrive Scholars
