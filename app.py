import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from model.RAG import RAG
from model.Embedding import EmbeddingModel
from model.LLM import GeminiLLM
from data.preprocessing import load_chunk
import uvicorn
from uuid import uuid4
load_dotenv()

app = FastAPI(
    title="Chatbot API",
    description="Customer support chatbot API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_system = None
embedding_model = None
llm = None

conversation_store: dict = {}


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    # sources: Optional[List[dict]] = []
    # user_role: str # admin, superadmin or household user

class HealthResponse(BaseModel):
    status: str
    message: str

sample_chunks = []

@app.on_event("startup")
async def startup_event():
    global rag_system, embedding_model, llm, sample_chunks
    try:
        print("Embedding Model loading...")
        embedding_model = EmbeddingModel()
        
        print("RAG loading...")
        rag_system = RAG()
        
        print("LLM loading...")
        llm = GeminiLLM()
        
        print("Chunks loading...")
        system_chunks, question_chunks = await load_chunk(embedding_model)
        sample_chunks = system_chunks + question_chunks
        sample_chunks = [{'text': chunk['content'], 
                        'embedding': embedding_model.encode(str(chunk['metadata']['header']) + str(chunk['metadata']['title']))}
                        for chunk in sample_chunks]
        
        print(f"Loaded {len(sample_chunks)} chunks")
        print("Server started successfully")
    except Exception as e:
        print(f"Initialization error: {e}")
        import traceback
        traceback.print_exc()


@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(
        status="ok",
        message="Chatbot API is working properly"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        message="Server is running normally"
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global rag_system, sample_chunks, conversation_store, llm
    
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    if llm is None or rag_system is None:
        raise HTTPException(status_code=503, detail="Server is starting, please try again later")
    
    try:
        # Get or create session
        session_id = request.session_id or str(uuid4())
        
        # Get conversation history from store
        if session_id not in conversation_store:
            conversation_store[session_id] = []
        
        history = conversation_store[session_id]
        
        # Add user message to history
        history.append({"role": "user", "content": request.message})
        
        # Retrieve relevant chunks
        if rag_system and sample_chunks:
            relevant_chunks = await rag_system.retrieve(
                data=request.message,
                embedding_model=embedding_model,
                chunks=sample_chunks,
                top_k=5
            )
        else:
            relevant_chunks = []
        
        # Generate response using LLM with history
        response_text = llm.generate_response(
            query=request.message,
            rag_results=relevant_chunks,
            conversation_history=history
        )
        
        history.append({"role": "assistant", "content": response_text})
        
        if len(history) > 20:
            conversation_store[session_id] = history[-20:]
        
        # sources = []
        # for chunk, score in relevant_chunks[:3]:
        #     sources.append({
        #         "text": chunk.get('text', chunk.get('content', ''))[:200],
        #         "score": round(score, 4)
        #     })
        
        return ChatResponse(
            response=response_text,
            session_id=session_id,
            # sources=sources
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")



if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
