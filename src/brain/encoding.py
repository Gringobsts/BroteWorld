"""Bytes UTF-8 y siete delimitadores fijos, sin vocabulario aprendido."""

import torch

from brain.data import Lesson, Prompt

PAD, BOS, SITUATION, QUESTION, ANSWER, EOS, CORRECTION = range(256, 263)
VOCAB_SIZE = 263
IGNORE = -100


def encode_prompt(prompt: Prompt) -> list[int]:
    return (
        [BOS, SITUATION]
        + list(prompt.situation.encode("utf-8"))
        + [QUESTION]
        + list(prompt.question.encode("utf-8"))
        + [ANSWER]
    )


def decode_answer(tokens: list[int]) -> str:
    if EOS in tokens:
        tokens = tokens[: tokens.index(EOS)]
    if any(t >= 256 or t < 0 for t in tokens):
        return "�"
    return bytes(tokens).decode("utf-8", errors="replace")


def training_batch(lessons: list[Lesson]) -> tuple[torch.Tensor, torch.Tensor]:
    if not lessons:
        raise ValueError("Lote vacío")
    sequences, targets = [], []
    for lesson in lessons:
        if lesson.split != "train":
            raise ValueError(f"Prohibido entrenar con {lesson.split}")
        prefix = encode_prompt(lesson.prompt())
        full = prefix + list(lesson.answer.encode("utf-8")) + [EOS]
        sequences.append(full[:-1])
        # The ANSWER delimiter predicts the first answer byte. Prompt and padding are ignored.
        targets.append([IGNORE] * (len(prefix) - 1) + full[len(prefix) :])
    width = max(map(len, sequences))
    inputs = torch.full((len(lessons), width), PAD, dtype=torch.long)
    labels = torch.full_like(inputs, IGNORE)
    for i, (sequence, target) in enumerate(zip(sequences, targets, strict=True)):
        inputs[i, : len(sequence)] = torch.tensor(sequence)
        labels[i, : len(target)] = torch.tensor(target)
    return inputs, labels
