"""
ThriveBot Web Demo - A visual demo interface for screen recording
Run with: python demo/app.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import structlog
import logging

from app.config import settings
from app.ingestion import GeminiEmbedder
from app.retrieval import FAISSVectorStore, RAGRetriever
from app.generation import GeminiLLM

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger()

# Initialize FastAPI
app = FastAPI(title="ThriveBot Demo", description="RAG-Based FAQ Bot Demo")

# Global components
embedder = None
vector_store = None
retriever = None
llm = None


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list
    query: str


@app.on_event("startup")
async def startup():
    """Initialize RAG components on startup"""
    global embedder, vector_store, retriever, llm
    
    logger.info("🚀 Starting ThriveBot Demo...")
    
    embedder = GeminiEmbedder(api_key=settings.gemini_api_key)
    
    vector_store = FAISSVectorStore(
        dimension=embedder.dimension,
        store_path=settings.vector_store_path
    )
    vector_store.load()
    
    retriever = RAGRetriever(
        embedder=embedder,
        vector_store=vector_store,
        top_k=settings.top_k
    )
    
    llm = GeminiLLM(api_key=settings.gemini_api_key)
    
    logger.info(f"✅ Ready! Vector store has {vector_store.size} documents")


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the demo UI"""
    return HTML_TEMPLATE


@app.post("/api/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Process a question and return AI-generated answer"""
    question = request.question.strip()
    
    if not question:
        return QueryResponse(
            answer="Please enter a question!",
            sources=[],
            query=question
        )
    
    # Retrieve context
    context, sources = retriever.retrieve_and_format(question)
    
    # Generate response
    answer = llm.generate(question, context)
    
    # Format sources
    formatted_sources = [
        {
            "name": s.get("source", "Unknown").split("\\")[-1].split("/")[-1],
            "score": f"{s.get('score', 0):.0%}",
            "preview": s.get("content", "")[:150] + "..."
        }
        for s in sources[:3]
    ]
    
    return QueryResponse(
        answer=answer,
        sources=formatted_sources,
        query=question
    )


@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    return {
        "documents": vector_store.size if vector_store else 0,
        "model": "Gemini 2.5 Flash",
        "status": "online"
    }


# Embedded HTML template for the demo UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ThriveBot - AI FAQ Assistant Demo</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
        }
        
        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        .badge {
            background: rgba(255,255,255,0.2);
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
        }
        
        .chat-container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 25px 50px rgba(0,0,0,0.15);
            overflow: hidden;
        }
        
        .chat-header {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            color: white;
            padding: 20px 25px;
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .bot-avatar {
            width: 50px;
            height: 50px;
            background: rgba(255,255,255,0.2);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
        }
        
        .bot-info h2 {
            font-size: 1.2rem;
            font-weight: 600;
        }
        
        .bot-info p {
            font-size: 0.85rem;
            opacity: 0.8;
        }
        
        .status-dot {
            width: 10px;
            height: 10px;
            background: #10b981;
            border-radius: 50%;
            margin-left: auto;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .messages {
            height: 400px;
            overflow-y: auto;
            padding: 25px;
            background: #f8fafc;
        }
        
        .message {
            margin-bottom: 20px;
            animation: fadeIn 0.3s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .message.user {
            display: flex;
            justify-content: flex-end;
        }
        
        .message.bot {
            display: flex;
            justify-content: flex-start;
        }
        
        .message-content {
            max-width: 75%;
            padding: 15px 20px;
            border-radius: 18px;
            line-height: 1.5;
        }
        
        .message.user .message-content {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            color: white;
            border-bottom-right-radius: 5px;
        }
        
        .message.bot .message-content {
            background: white;
            color: #1e293b;
            border-bottom-left-radius: 5px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        
        .sources {
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #e2e8f0;
        }
        
        .sources-title {
            font-size: 0.75rem;
            color: #64748b;
            margin-bottom: 8px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .source-tag {
            display: inline-block;
            background: #f1f5f9;
            color: #475569;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.75rem;
            margin-right: 6px;
            margin-bottom: 6px;
        }
        
        .source-tag .score {
            color: #10b981;
            font-weight: 600;
        }
        
        .input-area {
            padding: 20px 25px;
            background: white;
            border-top: 1px solid #e2e8f0;
            display: flex;
            gap: 15px;
        }
        
        .input-area input {
            flex: 1;
            padding: 15px 20px;
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            font-size: 1rem;
            outline: none;
            transition: border-color 0.2s;
        }
        
        .input-area input:focus {
            border-color: #4f46e5;
        }
        
        .input-area button {
            padding: 15px 30px;
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .input-area button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(79, 70, 229, 0.4);
        }
        
        .input-area button:disabled {
            opacity: 0.7;
            cursor: not-allowed;
            transform: none;
        }
        
        .typing-indicator {
            display: flex;
            gap: 5px;
            padding: 15px 20px;
        }
        
        .typing-indicator span {
            width: 8px;
            height: 8px;
            background: #94a3b8;
            border-radius: 50%;
            animation: bounce 1.4s infinite;
        }
        
        .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
        .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
        
        @keyframes bounce {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-8px); }
        }
        
        .sample-questions {
            margin-top: 25px;
            text-align: center;
        }
        
        .sample-questions p {
            color: rgba(255,255,255,0.8);
            margin-bottom: 12px;
            font-size: 0.9rem;
        }
        
        .sample-btn {
            background: rgba(255,255,255,0.15);
            color: white;
            border: 1px solid rgba(255,255,255,0.3);
            padding: 10px 18px;
            border-radius: 25px;
            margin: 5px;
            cursor: pointer;
            font-size: 0.85rem;
            transition: all 0.2s;
        }
        
        .sample-btn:hover {
            background: rgba(255,255,255,0.25);
            transform: translateY(-2px);
        }
        
        .architecture-badge {
            margin-top: 30px;
            text-align: center;
            color: rgba(255,255,255,0.7);
            font-size: 0.8rem;
        }
        
        .architecture-badge span {
            background: rgba(255,255,255,0.1);
            padding: 5px 12px;
            border-radius: 15px;
            margin: 0 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>
                🎓 ThriveBot
                <span class="badge">RAG Demo</span>
            </h1>
            <p>AI-Powered FAQ Assistant for Thrive Scholars</p>
        </div>
        
        <div class="chat-container">
            <div class="chat-header">
                <div class="bot-avatar">🤖</div>
                <div class="bot-info">
                    <h2>ThriveBot Assistant</h2>
                    <p>Powered by Gemini AI + FAISS Vector Search</p>
                </div>
                <div class="status-dot" title="Online"></div>
            </div>
            
            <div class="messages" id="messages">
                <div class="message bot">
                    <div class="message-content">
                        👋 Hi! I'm ThriveBot, your AI assistant for Thrive Scholars. 
                        <br><br>
                        I can answer questions about our programs, financial aid, mentorship, and more. 
                        Try asking me something!
                    </div>
                </div>
            </div>
            
            <div class="input-area">
                <input 
                    type="text" 
                    id="questionInput" 
                    placeholder="Ask me anything about Thrive Scholars..."
                    onkeypress="if(event.key === 'Enter') askQuestion()"
                >
                <button onclick="askQuestion()" id="sendBtn">
                    Ask →
                </button>
            </div>
        </div>
        
        <div class="sample-questions">
            <p>Try these sample questions:</p>
            <button class="sample-btn" onclick="setQuestion('What are the eligibility requirements?')">
                📋 Eligibility Requirements
            </button>
            <button class="sample-btn" onclick="setQuestion('How do I apply for financial aid?')">
                💰 Financial Aid
            </button>
            <button class="sample-btn" onclick="setQuestion('What mentorship programs are available?')">
                🤝 Mentorship Programs
            </button>
            <button class="sample-btn" onclick="setQuestion('When are the application deadlines?')">
                📅 Deadlines
            </button>
        </div>
        
        <div class="architecture-badge">
            <span>📄 Document Ingestion</span>
            <span>🔢 Gemini Embeddings</span>
            <span>🔍 FAISS Vector Search</span>
            <span>🤖 Gemini 2.5 Flash</span>
        </div>
    </div>
    
    <script>
        const messagesDiv = document.getElementById('messages');
        const input = document.getElementById('questionInput');
        const sendBtn = document.getElementById('sendBtn');
        
        function setQuestion(q) {
            input.value = q;
            input.focus();
        }
        
        function addMessage(content, isUser, sources = null) {
            const msgDiv = document.createElement('div');
            msgDiv.className = `message ${isUser ? 'user' : 'bot'}`;
            
            let html = `<div class="message-content">${formatContent(content)}`;
            
            if (sources && sources.length > 0) {
                html += `<div class="sources">
                    <div class="sources-title">📚 Sources</div>`;
                sources.forEach(s => {
                    html += `<span class="source-tag">${s.name} <span class="score">${s.score}</span></span>`;
                });
                html += `</div>`;
            }
            
            html += `</div>`;
            msgDiv.innerHTML = html;
            messagesDiv.appendChild(msgDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        function formatContent(text) {
            // Convert markdown-like formatting
            return text
                .replace(/\\n/g, '<br>')
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.*?)\*/g, '<em>$1</em>')
                .replace(/^- /gm, '• ')
                .replace(/\\n\\n/g, '<br><br>');
        }
        
        function showTyping() {
            const typingDiv = document.createElement('div');
            typingDiv.className = 'message bot';
            typingDiv.id = 'typing';
            typingDiv.innerHTML = `
                <div class="message-content">
                    <div class="typing-indicator">
                        <span></span><span></span><span></span>
                    </div>
                </div>
            `;
            messagesDiv.appendChild(typingDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        function hideTyping() {
            const typing = document.getElementById('typing');
            if (typing) typing.remove();
        }
        
        async function askQuestion() {
            const question = input.value.trim();
            if (!question) return;
            
            // Add user message
            addMessage(question, true);
            input.value = '';
            sendBtn.disabled = true;
            
            // Show typing indicator
            showTyping();
            
            try {
                const response = await fetch('/api/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question })
                });
                
                const data = await response.json();
                hideTyping();
                addMessage(data.answer, false, data.sources);
                
            } catch (error) {
                hideTyping();
                addMessage('Sorry, I encountered an error. Please try again.', false);
            }
            
            sendBtn.disabled = false;
            input.focus();
        }
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    print("=" * 60)
    print("🎓 ThriveBot Web Demo")
    print("=" * 60)
    print("Open your browser to: http://localhost:8080")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    uvicorn.run(app, host="127.0.0.1", port=8080)
