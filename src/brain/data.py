"""Esquema y lectura. No contiene las reglas lingüísticas del generador."""

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path


@dataclass(frozen=True)
class Prompt:
    situation: str
    question: str
    skill: str


@dataclass(frozen=True)
class Lesson:
    id: str
    stage: int
    skill: str
    situation: str
    question: str
    answer: str
    valid_answers: tuple[str, ...]
    semantic_group: str
    composition_group: str
    template_family: str
    split: str
    provenance: str

    def prompt(self) -> Prompt:
        return Prompt(self.situation, self.question, self.skill)

    def to_dict(self) -> dict:
        return asdict(self)


def read_lessons(path: Path, expected_split: str | None = None) -> list[Lesson]:
    lessons = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        record["valid_answers"] = tuple(record["valid_answers"])
        lesson = Lesson(**record)
        if expected_split and lesson.split != expected_split:
            raise ValueError(f"Partición incorrecta en {lesson.id}: {lesson.split}")
        if not lesson.valid_answers or lesson.answer not in lesson.valid_answers:
            raise ValueError(f"Respuesta inválida en {lesson.id}")
        lessons.append(lesson)
    if not lessons:
        raise ValueError(f"Conjunto vacío: {path}")
    return lessons


def verify_dataset(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, info in manifest["files"].items():
        if sha256((directory / name).read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"Datos modificados: {name}. Crear otra versión del examen.")
    return manifest
