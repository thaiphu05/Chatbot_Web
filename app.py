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

# Initialize FastAPI app
app = FastAPI(
    title="Chatbot API",
    description="API cho chatbot hỗ trợ khách hàng",
    version="1.0.0"
)

# CORS middleware để cho phép frontend gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo None, load trong startup_event
rag_system = None
embedding_model = None
llm = None

# In-memory conversation storage (cho demo, production nên dùng Redis/Database)
conversation_store: dict = {}


# Pydantic models
class ChatMessage(BaseModel):
    role: str  # "user" hoặc "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    # conversation_history được lưu server-side trong conversation_store

class ChatResponse(BaseModel):
    response: str
    session_id: str  # Trả về session_id để client dùng cho request tiếp theo
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
        
        print(f" Loaded {len(sample_chunks)} chunks")
        print("Server started successfully")
    except Exception as e:
        print(f"Lỗi khởi tạo: {e}")
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
    """    
    :param request: message, session_id and conversation_history
    :return: response, session_id and sources
    """
    global rag_system, sample_chunks, conversation_store, llm
    
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message không được để trống")
    
    # Check if models are loaded
    if llm is None or rag_system is None:
        raise HTTPException(status_code=503, detail="Server đang khởi động, vui lòng thử lại sau")
    
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
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")



if __name__ == "__main__":
    # Chạy server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
