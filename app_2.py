from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from model.RAG import RAG
from model.Embedding import EmbeddingModel
from model.LLM import GeminiLLM
from data.preprocessing import load_chunk
from uuid import uuid4
import traceback
import uvicorn
import json

load_dotenv()

app = FastAPI(
    title="Chatbot API",
    description="Customer support chatbot API",
    version="1.0.0"
)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=False,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

rag_system = None
embedding_model = None
llm = None

conversation_store: dict = {}

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
    
    async def send_message(self, message: dict, session_id: str):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(message)

manager = ConnectionManager()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str

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
        traceback.print_exc()

@app.websocket("/v1/chat/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    global rag_system, sample_chunks, conversation_store, llm, embedding_model
    
    origin = websocket.headers.get("origin", "unknown")
    print(f"WebSocket request from origin: {origin}")
    
    try:
        await manager.connect(websocket, session_id)
        print(f"WebSocket accepted: {session_id}")
    except Exception as e:
        print(f"WebSocket accept failed: {e}")
        return
    
    if session_id not in conversation_store:
        conversation_store[session_id] = []
    
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_message = message_data.get("message", "")
            
            if not user_message.strip():
                await manager.send_message({"error": "Message cannot be empty"}, session_id)
                continue
            
            if llm is None or rag_system is None:
                await manager.send_message({"error": "Server is starting"}, session_id)
                continue
            
            history = conversation_store[session_id]
            history.append({"role": "user", "content": user_message})
            
            await manager.send_message({"status": "processing"}, session_id)
            
            if rag_system and sample_chunks:
                relevant_chunks = await rag_system.retrieve(
                    data=user_message,
                    embedding_model=embedding_model,
                    chunks=sample_chunks,
                    top_k=5
                )
            else:
                relevant_chunks = []
            
            response_text = llm.generate_response(
                query=user_message,
                rag_results=relevant_chunks,
                conversation_history=history
            )
            
            history.append({"role": "assistant", "content": response_text})
            
            if len(history) > 20:
                conversation_store[session_id] = history[-20:]
            
            await manager.send_message({
                "response": response_text,
                "session_id": session_id,
                "status": "completed"
            }, session_id)
            
    except WebSocketDisconnect:
        manager.disconnect(session_id)
        print(f"Client {session_id} disconnected")
    except Exception as e:
        await manager.send_message({"error": str(e)}, session_id)
        manager.disconnect(session_id)


@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(status="ok", message="Chatbot API is working properly")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", message="Server is running normally")



@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global rag_system, sample_chunks, conversation_store, llm, embedding_model
    
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    if llm is None or rag_system is None:
        raise HTTPException(status_code=503, detail="Server is starting")
    
    try:
        session_id = request.session_id or str(uuid4())
        
        if session_id not in conversation_store:
            conversation_store[session_id] = []
        
        history = conversation_store[session_id]
        history.append({"role": "user", "content": request.message})
        
        if rag_system and sample_chunks:
            relevant_chunks = await rag_system.retrieve(
                data=request.message,
                embedding_model=embedding_model,
                chunks=sample_chunks,
                top_k=5
            )
        else:
            relevant_chunks = []
        
        response_text = llm.generate_response(
            query=request.message,
            rag_results=relevant_chunks,
            conversation_history=history
        )
        
        history.append({"role": "assistant", "content": response_text})
        
        if len(history) > 20:
            conversation_store[session_id] = history[-20:]
        
        return ChatResponse(response=response_text, session_id=session_id)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)