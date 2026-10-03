from dataclasses import replace

import pytest
import torch

from brain.data import Lesson


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


@pytest.fixture
def tiny_lessons():
    base = Lesson(
        "a",
        1,
        "color",
        "pelota;rojo",
        "Color?",
        "roja",
        ("roja",),
        "a",
        "pelota/rojo",
        "t0",
        "train",
        "test-handwritten",
    )
    return [
        base,
        replace(
            base,
            id="b",
            situation="pelota;azul",
            answer="azul",
            valid_answers=("azul",),
            semantic_group="b",
        ),
        replace(
            base,
            id="c",
            situation="libro;rojo",
            answer="rojo",
            valid_answers=("rojo",),
            semantic_group="c",
        ),
        replace(
            base,
            id="d",
            situation="libro;azul",
            answer="azul",
            valid_answers=("azul",),
            semantic_group="d",
        ),
    ]
