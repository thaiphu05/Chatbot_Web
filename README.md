# 🤖 Chatbot for Introduction to Software Engineering (IT3180 HUST)

This project is a customer support chatbot developed for the Introduction to Software Engineering course IT3180 at HUST. It utilizes a Retrieval-Augmented Generation (RAG) architecture to provide answers based on a given knowledge base.

## ✨ Features

- 🚀 **FastAPI Backend**: A robust and fast backend serving the chatbot API.
- 🧠 **Retrieval-Augmented Generation (RAG)**: The core of the chatbot, which retrieves relevant information from a document base and uses a Large Language Model (LLM) to generate human-like responses.
- 💎 **Gemini LLM**: Integrated with Google's Gemini for powerful language generation.
- ↔️ **WebSocket Support**: Enables real-time, bidirectional communication for a smoother chat experience (`app_2.py`).
- 🔍 **Vector Embeddings**: Uses sentence transformers to create embeddings for efficient similarity search.
- ⚙️ **Environment-based Configuration**: Easy to configure through a `.env` file.

## 📂 Project Structure

```
.
├── data/
│   └── preprocessing.py      # Scripts for loading and preprocessing data
├── model/
│   ├── Embedding.py          # Handles text embedding
│   ├── LLM.py                # Wrapper for the Gemini LLM
│   ├── RAG.py                # Core RAG implementation
│   └── __init__.py
├── app.py                    # Main FastAPI application with HTTP endpoints
├── app_2.py                  # FastAPI application with WebSocket support
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## 🚀 Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/thaiphu05/Chatbot_Web.git
cd Chatbot_Web
```

### 2. Create and Activate a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

- **Windows**:
  ```bash
  python -m venv venv
  .\venv\Scripts\activate
  ```

- **macOS/Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies

Install all the required Python packages using `pip`.

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the root directory by copying the example file.

```bash
copy .env.example .env
```

Now, open the `.env` file and add your credentials and configuration. You will need a Gemini API key.

```env
# Get your API key from Google AI Studio: https://aistudio.google.com/app/apikey
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"

# Path to the sentence transformer model
MODEL_PATH="all-MiniLM-L6-v2"

# Paths to data files
SYSTEM_DATA_PATH="./data/system.md"
NORMAL_QUESTION_DATA_PATH="./data/normal_question.md"
```

## ▶️ Running the Application

You can run either the standard HTTP version or the WebSocket version of the application.

### Standard HTTP Server

This will run the application defined in `app.py`.

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### WebSocket Server

This will run the application defined in `app_2.py`, which includes WebSocket support for real-time chat.

```bash
uvicorn app_2:app --host 0.0.0.0 --port 8000 --reload
```

The server will be available at `http://localhost:8000`.

## 🔗 API Endpoints

- **`GET /`**: Health check endpoint to see if the API is running.
- **`POST /chat`**: The main endpoint to send a message to the chatbot.
- **`WS /v1/chat/{session_id}`**: The WebSocket endpoint for real-time chat (in `app_2.py`).
