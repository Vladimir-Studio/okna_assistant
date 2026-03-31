"""
Логирование диалогов в JSONL-файл.
Каждая строка — отдельный JSON-объект с полями запроса и ответа.
"""
import json
import os
import uuid
from datetime import datetime, timezone


LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dialogs.jsonl")


def log_dialog(
    message: str,
    answer: str,
    context: list,
    retrieval_scores: list[float],
) -> str:
    """Записывает диалог в JSONL и возвращает уникальный dialog_id."""
    dialog_id = str(uuid.uuid4())
    record = {
        "dialog_id": dialog_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        "message": message,
        "answer": answer,
        "context_questions": [c.get("question", "") for c in context],
        "retrieval_scores": retrieval_scores,
        # среднее расстояние FAISS — чем меньше, тем лучше совпадение
        "avg_retrieval_score": round(sum(retrieval_scores) / len(retrieval_scores), 4) if retrieval_scores else None,
        "feedback": None,  # заполняется через /feedback
    }
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return dialog_id
