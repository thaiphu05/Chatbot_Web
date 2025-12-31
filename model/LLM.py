import os
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Optional

load_dotenv()


class GeminiLLM:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        
        self.system_prompt = """You are an AI assistant helping users with the household and population management system.

Response rules:
1. Answer in Vietnamese, concise and easy to understand
2. Only answer based on the provided context
3. If no information is found in the context, say "I could not find information about this issue in the documents"
4. Do not fabricate information
5. If there are multiple steps, list each step clearly
6. Use polite and friendly language, refuse to answer offensive or inappropriate language by saying
"I'm sorry, I cannot answer this question."
"""

    def generate_response(
        self, 
        query: str, 
        rag_results: List[tuple],
        conversation_history: Optional[List[dict]] = None,
        min_score: float = 0.3
    ) -> str:
        """
        Generate response from RAG retrieval results.
        
        :param query: User's question
        :param rag_results: List of (chunk, score) from RAG.retrieve()
        :param conversation_history: Previous conversation messages
        :param min_score: Minimum similarity score to include
        :return: Generated response
        """
        # Filter and extract context from RAG results
        context_parts = []
        for chunk, score in rag_results:
            if score >= min_score:
                text = chunk.get('text', chunk.get('content', ''))
                if text:
                    context_parts.append(f"[Score: {score:.2f}] {text}")
        
        if not context_parts:
            return "Sorry, I could not find relevant information to answer your question."
        
        context = "\n\n".join(context_parts)
        
        # Build conversation history string
        history_str = ""
        if conversation_history and len(conversation_history) > 0:
            recent_history = conversation_history[-6:]
            history_parts = []
            for msg in recent_history:
                role = "User" if msg.get("role") == "user" else "Assistant"
                history_parts.append(f"{role}: {msg.get('content', '')}")
            history_str = "\n".join(history_parts)
        
        prompt = f"""{self.system_prompt}

### Conversation History:
{history_str if history_str else "(None)"}

### Reference Information (Context):
{context}

### Current Question:
{query}

### Answer:"""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Sorry, an error occurred: {str(e)}"
