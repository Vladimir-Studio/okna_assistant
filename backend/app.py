import os
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

import faiss
import numpy as np
from openai import OpenAI

from .rag_index import load_index, search_similar
from .logger import log_dialog
from .metrics import compute_metrics, update_feedback


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set. Please set it in your environment or in a .env file.")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="FAQ RAG Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_PATH = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_PATH), name="static")

@app.get("/")
async def root():
    return FileResponse(os.path.join(FRONTEND_PATH, "index.html"))


class ChatRequest(BaseModel):
    message: str
    top_k: int = 3


class ChatResponse(BaseModel):
    answer: str
    context: List[Dict[str, Any]]
    dialog_id: str  # используй для отправки фидбека на /feedback


class FeedbackRequest(BaseModel):
    dialog_id: str
    feedback: int  # 1 = хорошо, -1 = плохо, 0 = нейтрально


INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "faiss_index.bin")
META_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "faqs_metadata.npy")

faiss_index, metadata = load_index(INDEX_PATH, META_PATH)

# --- настройки безопасности и качества ---
MAX_INPUT_LEN = 500          # максимальная длина сообщения пользователя
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 300))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.2))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
RAG_DISTANCE_THRESHOLD = 1.0 # FAISS L2-дистанция: выше — контекст нерелевантен
                              # подбери под свои данные, смотри avg_retrieval_distance в /metrics


def embed_text(texts: List[str]) -> np.ndarray:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    vectors = [d.embedding for d in response.data]
    return np.array(vectors, dtype="float32")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    # 1. базовая валидация
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message is empty")

    # 2. защита от prompt injection: обрезаем длину и убираем управляющие символы
    message = req.message.strip()[:MAX_INPUT_LEN]
    message = message.replace("\x00", "").replace("\r", " ")

    # 3. эмбеддинг и поиск
    query_vec = embed_text([message])
    similar_items, retrieval_scores = search_similar(faiss_index, metadata, query_vec, k=req.top_k)

    # 4. RAG validation: если ближайший документ слишком далёк — контекст нерелевантен
    if retrieval_scores and min(retrieval_scores) > RAG_DISTANCE_THRESHOLD:
        fallback = "Извините, я не нашёл подходящей информации по вашему вопросу. Пожалуйста, свяжитесь с нашей поддержкой."
        dialog_id = log_dialog(message, fallback, similar_items, retrieval_scores)
        return ChatResponse(answer=fallback, context=similar_items, dialog_id=dialog_id)

    context_text = "\n\n".join(
        [f"Q: {item['question']}\nA: {item['answer']}" for item in similar_items]
    )

    system_prompt = (
        "Ты — ассистент компании «Континент Окна Крым». Отвечай кратко, по делу, на русском, в дружелюбно-профессиональном стиле. "
        "Используй только FAQ-контекст ниже. Если ответа нет — скажи об этом и предложи связаться с поддержкой. "
        "Не выдумывай факты, не давай гарантий, не выходи за тему компании."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        # пользовательский ввод изолирован от системного промпта отдельным полем
        {"role": "user", "content": f"Контекст FAQ:\n{context_text}\n\nВопрос: {message}"},
    ]

    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    answer = completion.choices[0].message.content

    dialog_id = log_dialog(req.message, answer, similar_items, retrieval_scores)

    return ChatResponse(answer=answer, context=similar_items, dialog_id=dialog_id)


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    if req.feedback not in (-1, 0, 1):
        raise HTTPException(status_code=400, detail="feedback must be -1, 0 or 1")
    found = update_feedback(req.dialog_id, req.feedback)
    if not found:
        raise HTTPException(status_code=404, detail="dialog_id not found")
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    return compute_metrics()


@app.get("/health")
async def health():
    return {"status": "ok"}


