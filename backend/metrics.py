"""
Простые метрики качества на основе логов диалогов.
"""
import json
import os
from collections import Counter

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dialogs.jsonl")


def _load_logs() -> list[dict]:
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def update_feedback(dialog_id: str, feedback: int) -> bool:
    """
    Обновляет поле feedback для записи по dialog_id.
    feedback: 1 = хорошо, -1 = плохо, 0 = нейтрально.
    Возвращает True если запись найдена.
    """
    if not os.path.exists(LOG_PATH):
        return False

    logs = _load_logs()
    found = False
    for record in logs:
        if record["dialog_id"] == dialog_id:
            record["feedback"] = feedback
            found = True
            break

    if found:
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            for record in logs:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return found


def compute_metrics() -> dict:
    """Возвращает агрегированные метрики по всем диалогам."""
    logs = _load_logs()
    if not logs:
        return {"total": 0}

    total = len(logs)
    with_feedback = [r for r in logs if r.get("feedback") is not None]
    feedback_counts = Counter(r["feedback"] for r in with_feedback)

    scores = [r["avg_retrieval_score"] for r in logs if r.get("avg_retrieval_score") is not None]
    avg_score = round(sum(scores) / len(scores), 4) if scores else None

    # последние 10 диалогов для быстрого просмотра
    recent = [
        {
            "dialog_id": r["dialog_id"],
            "ts": r["ts"],
            "message": r["message"][:120],
            "feedback": r.get("feedback"),
            "avg_retrieval_score": r.get("avg_retrieval_score"),
        }
        for r in logs[-10:]
    ]

    return {
        "total": total,
        "with_feedback": len(with_feedback),
        "feedback_positive": feedback_counts.get(1, 0),
        "feedback_negative": feedback_counts.get(-1, 0),
        "feedback_neutral": feedback_counts.get(0, 0),
        # satisfaction rate среди оценённых
        "satisfaction_rate": round(feedback_counts.get(1, 0) / len(with_feedback), 3) if with_feedback else None,
        # среднее расстояние FAISS (меньше = релевантнее контекст)
        "avg_retrieval_distance": avg_score,
        "recent": recent,
    }
