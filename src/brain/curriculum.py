"""Generador controlado de español; el evaluador no importa este módulo."""

import itertools
import json
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

from brain.data import Lesson

NOUNS = (
    ("pelota", "pelotas", "f"),
    ("casa", "casas", "f"),
    ("mesa", "mesas", "f"),
    ("taza", "tazas", "f"),
    ("caja", "cajas", "f"),
    ("libro", "libros", "m"),
    ("gato", "gatos", "m"),
    ("perro", "perros", "m"),
    ("vaso", "vasos", "m"),
    ("zapato", "zapatos", "m"),
)
COLORS = ("rojo", "verde", "azul", "negro", "blanco", "amarillo")
SIZES = ("pequeño", "mediano", "grande")
QUESTIONS = {
    "nombre": (
        "¿Qué objeto hay?",
        "Decí el nombre.",
        "Nombrá el objeto.",
        "Indicá qué objeto aparece.",
        "¿Cuál es el objeto de la situación?",
    ),
    "color": (
        "¿De qué color es?",
        "Decí el color.",
        "Nombrá el color.",
        "Indicá qué color aparece.",
        "¿Cuál es el color del objeto?",
    ),
    "tamaño": (
        "¿De qué tamaño es?",
        "Decí el tamaño.",
        "Nombrá el tamaño.",
        "Indicá qué tamaño aparece.",
        "¿Cuál es el tamaño del objeto?",
    ),
    "concordancia": (
        "Describí su color.",
        "Escribí la frase sobre su color.",
        "Formá una oración sobre su color.",
        "Indicá su color en una oración.",
        "¿Cómo expresarías su color en una frase?",
    ),
}
PROVENANCE = "synthetic:brain-v1; author=Codex; review=code+golden-tests; human-review=pending"


def adjective(word: str, gender: str, number: int) -> str:
    if gender == "f" and word.endswith("o"):
        word = word[:-1] + "a"
    if number == 2:
        word += "es" if word.endswith("l") else "s"
    return word


def question_text(skill: str, family: int, number: int) -> str:
    text = QUESTIONS[skill][family]
    if number == 2:
        text = text.replace("color es?", "color son?").replace("tamaño es?", "tamaño son?")
        text = text.replace("objeto hay", "objetos hay").replace("el objeto", "los objetos")
        text = text.replace("objeto aparece", "objetos aparecen")
        text = text.replace("¿Cuál es los objetos", "¿Cuáles son los objetos")
    return text


def build_lessons() -> dict[str, list[Lesson]]:
    result = defaultdict(list)

    def add(stage, skill, situation, question, answer, group, pair, family, split):
        identity = json.dumps([skill, situation, question], ensure_ascii=False)
        result[split].append(
            Lesson(
                sha256(identity.encode()).hexdigest()[:20],
                stage,
                skill,
                situation,
                question,
                answer,
                (answer,),
                group,
                pair,
                family,
                split,
                PROVENANCE,
            )
        )

    # The entire object/color pair is withheld across number, size, skill and wording.
    for ni, (singular, plural, gender) in enumerate(NOUNS):
        for ci, color in enumerate(COLORS):
            bucket = (ni + ci) % len(COLORS)
            pair = f"{singular}/{color}"
            for number, si in itertools.product((1, 2), range(3)):
                size = SIZES[si]
                group = f"{pair}/{number}/{size}"
                scene_role = (si - ni - ci - number) % 3
                if bucket == 0:
                    split = "test_composition"
                elif bucket == 1:
                    split = "validation_composition"
                else:
                    split = ("train", "validation_scenes", "test_scenes")[scene_role]
                noun = singular if number == 1 else plural
                article = {("f", 1): "La", ("f", 2): "Las", ("m", 1): "El", ("m", 2): "Los"}[
                    (gender, number)
                ]
                colored = adjective(color, gender, number)
                answers = {
                    "nombre": noun,
                    "color": colored,
                    "tamaño": adjective(size, gender, number),
                    "concordancia": f"{article} {noun} {'es' if number == 1 else 'son'} {colored}.",
                }
                fields = [
                    f"objeto={singular}",
                    f"color={color}",
                    f"cantidad={number}",
                    f"tamaño={size}",
                ]
                for skill, answer in answers.items():
                    stage = 2 if skill == "concordancia" or number == 2 else 1
                    families = range(3) if split == "train" else (0,)
                    for family in families:
                        for order in range(2) if split == "train" else (0,):
                            situation = ";".join(fields if order == 0 else fields[::-1])
                            add(
                                stage,
                                skill,
                                situation,
                                question_text(skill, family, number),
                                answer,
                                group,
                                pair,
                                f"t{family}",
                                split,
                            )
                    if split == "train":
                        for family, reserved in (
                            (3, "validation_templates"),
                            (4, "test_templates"),
                        ):
                            add(
                                stage,
                                skill,
                                ";".join(fields),
                                question_text(skill, family, number),
                                answer,
                                group,
                                pair,
                                f"t{family}",
                                reserved,
                            )

    # Separate exact-copy diagnostic. New symbol sequences form its own holdouts.
    for i, symbols in enumerate(itertools.product("aei", repeat=3)):
        word = "".join(symbols)
        split = "test_copy" if i % 9 == 0 else ("validation_copy" if i % 9 == 1 else "train")
        for question in ("Copiá.", "Repetí.") if split == "train" else ("Copiá.",):
            add(0, "copia", word, question, word, f"copy/{word}", f"copy/{word}", "copy", split)
    audit_splits(result)
    return dict(result)


def audit_splits(splits: dict[str, list[Lesson]]) -> None:
    train = splits["train"]
    train_ids = {x.id for x in train}
    train_groups = {x.semantic_group for x in train}
    train_pairs = {x.composition_group for x in train}
    seen_ids = set()
    for split, rows in splits.items():
        for row in rows:
            if row.id in seen_ids:
                raise ValueError(f"Ejemplo duplicado: {row.id}")
            seen_ids.add(row.id)
            if split != "train" and row.id in train_ids:
                raise ValueError("Contaminación de preguntas")
            if split.endswith("composition") and row.composition_group in train_pairs:
                raise ValueError("Contaminación de combinaciones")
            if split.endswith("scenes") and row.semantic_group in train_groups:
                raise ValueError("Contaminación de escenas")
    val_pairs = {x.composition_group for x in splits["validation_composition"]}
    test_pairs = {x.composition_group for x in splits["test_composition"]}
    if val_pairs & test_pairs:
        raise ValueError("Validación y examen comparten combinaciones reservadas")


def write_dataset(directory: Path) -> dict:
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Usá una carpeta vacía para no reemplazar un examen congelado.")
    directory.mkdir(parents=True, exist_ok=True)
    splits = build_lessons()
    manifest = {"version": "brain-v1", "provenance": PROVENANCE, "files": {}}
    for split, rows in sorted(splits.items()):
        name = f"{split}.jsonl"
        payload = "".join(
            json.dumps(x.to_dict(), ensure_ascii=False, sort_keys=True) + "\n" for x in rows
        ).encode()
        (directory / name).write_bytes(payload)
        manifest["files"][name] = {
            "sha256": sha256(payload).hexdigest(),
            "count": len(rows),
            "skills": dict(Counter(x.skill for x in rows)),
            "semantic_groups": len({x.semantic_group for x in rows}),
        }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
