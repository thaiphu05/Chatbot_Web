import re
from typing import List, Dict
from model.Embedding import EmbeddingModel
import os
from dotenv import load_dotenv
load_dotenv()

FEATURE_KEYWORDS = {
    "button": ["Button", "Nút"],
    "form": ["Form"],
    "table": ["Table", "Bảng"],
    "header": ["Header", "Tiêu Đề"],
}

def detect_feature(title: str) -> str:
    for feature, keywords in FEATURE_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in title.lower():
                return feature
    return "general"


def load_system_data_chunks(md_text: str) -> List[Dict]:
    lines = md_text.splitlines()

    chunks = []
    current_chunk = []
    metadata = {
        "header": None,
        "section": None,
        "feature": None,
        "title": None,
    }
    def flush_chunk():
        if current_chunk:
            chunks.append({
                "content": "\n".join(current_chunk).strip(),
                "metadata": metadata.copy()
            })

    for line in lines:
        # ## header / Module
        if line.startswith("## "):
            flush_chunk()
            current_chunk = [ line.replace("##", "").strip("\n") ]
            metadata["header"] = line.replace("##", "").strip()
            metadata["section"] = None
            metadata["feature"] = "header"
            metadata["title"] = metadata["header"]

        # ### Section
        elif line.startswith("### "):
            flush_chunk()
            current_chunk = [line.replace("###", "").strip("\n")]
            metadata["section"] = line.replace("###", "").strip()
            metadata["feature"] = "section"
            metadata["title"] = metadata["section"]

        # #### Feature OR bold feature title
        elif line.startswith("#### ") :
            flush_chunk()
            title = re.sub(r"[*#]", "", line).strip()
            current_chunk = [line.replace("####", "").strip("\n")]
            metadata["title"] = title
            metadata["feature"] = detect_feature(title)

        else:
            current_chunk.append(line)

    flush_chunk()
    return chunks
def load_question_chunks(md_text: str) -> List[Dict]:
    lines = md_text.splitlines()

    chunks = []
    current_answer = []
    metadata = {
        "header": None,
        "title": None,
    }
    
    def flush_chunk():
        if current_answer and metadata["title"]:
            chunks.append({
                "content": "\n".join(current_answer).strip(),
                "metadata": metadata.copy()
            })
    
    for line in lines:
        # ### Header section
        if line.startswith("### "):
            flush_chunk()
            current_answer = []
            metadata["header"] = line.replace("###", "").strip()
            metadata["title"] = None
        
        # **Q: Question
        elif line.startswith("**Q:") or line.startswith("**Q "):
            flush_chunk()
            current_answer = []
            # Extract question text: remove **Q: and trailing **
            question = re.sub(r"^\*\*Q[:\s]*", "", line)
            question = re.sub(r"\*\*$", "", question).strip()
            metadata["title"] = question
        
        # A: Answer line
        elif line.startswith("A:"):
            answer_text = line.replace("A:", "").strip()
            if answer_text:
                current_answer.append(answer_text)
        
        # Continuation of answer (not empty, not a new Q or header)
        elif line.strip() and metadata["title"]:
            current_answer.append(line)
    
    flush_chunk()
    return chunks
        
        
async def load_chunk(embedding: EmbeddingModel):
    with open(os.getenv("SYSTEM_DATA_PATH"), "r", encoding="utf-8") as f:
        system_data = f.read()
    with open(os.getenv("NORMAL_QUESTION_DATA_PATH"), "r", encoding="utf-8") as f:
        question_data = f.read()
    system_chunks = load_system_data_chunks(system_data)
    question_chunks = load_question_chunks(question_data)
    return system_chunks, question_chunks


