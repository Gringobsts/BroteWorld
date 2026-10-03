"""Evaluación de respuestas libres. No importa el currículo ni calcula sus etiquetas."""

import json
import time
import unicodedata
from collections import defaultdict
from hashlib import sha256
from typing import Protocol

from brain.data import Lesson, Prompt


class Predictor(Protocol):
    def predict(self, prompts: list[Prompt], max_new_bytes: int = 80) -> list[str]: ...


def normalize(text: str) -> str:
    # Ignore case, external spacing and an optional final period; keep accents and grammar.
    return " ".join(unicodedata.normalize("NFC", text).casefold().strip().rstrip(".").split())


def evaluate(
    predictor: Predictor,
    lessons: list[Lesson],
    batch_size: int = 64,
    max_new_bytes: int = 80,
    include_predictions: bool = False,
) -> dict:
    if not lessons or batch_size < 1:
        raise ValueError("Se necesitan ejemplos y un lote positivo")
    started = time.perf_counter()
    by_skill = defaultdict(lambda: {"correct": 0, "count": 0})
    by_stage = defaultdict(lambda: {"correct": 0, "count": 0})
    predictions = []
    for offset in range(0, len(lessons), batch_size):
        batch = lessons[offset : offset + batch_size]
        # The predictor receives no reference answer, split, scene group or lesson id.
        outputs = predictor.predict([x.prompt() for x in batch], max_new_bytes)
        if len(outputs) != len(batch):
            raise ValueError("Cantidad de predicciones incorrecta")
        for lesson, output in zip(batch, outputs, strict=True):
            correct = normalize(output) in {normalize(x) for x in lesson.valid_answers}
            for item in (by_skill[lesson.skill], by_stage[str(lesson.stage)]):
                item["count"] += 1
                item["correct"] += int(correct)
            predictions.append({"id": lesson.id, "prediction": output, "correct": correct})
    correct = sum(x["correct"] for x in by_skill.values())
    for item in list(by_skill.values()) + list(by_stage.values()):
        item["accuracy"] = item["correct"] / item["count"]
    elapsed = time.perf_counter() - started
    report = {
        "count": len(lessons),
        "correct": correct,
        "probe_sha256": sha256(
            json.dumps([x.to_dict() for x in lessons], sort_keys=True).encode()
        ).hexdigest(),
        "accuracy": correct / len(lessons),
        "by_skill": dict(by_skill),
        "by_stage": dict(by_stage),
        "seconds": elapsed,
        "ms_per_example": 1000 * elapsed / len(lessons),
    }
    if include_predictions:
        report["predictions"] = predictions
    return report


def retention(before: dict, after: dict, max_drop_pp: float = 5.0) -> dict:
    """Compare the same frozen probe before/after new learning; negative means improvement."""
    if before["probe_sha256"] != after["probe_sha256"]:
        raise ValueError("Las pruebas de retención deben ser comparables")
    drop = 100 * (before["accuracy"] - after["accuracy"])
    drops = {
        skill: 100 * (row["accuracy"] - after["by_skill"][skill]["accuracy"])
        for skill, row in before["by_skill"].items()
    }
    return {
        "drop_pp": drop,
        "by_skill_drop_pp": drops,
        "max_drop_pp": max_drop_pp,
        "passed": drop <= max_drop_pp,
    }
