import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from brain.data import Lesson
from brain.encoding import IGNORE, encode_prompt, training_batch
from brain.model import ByteGRU, ModelConfig


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 17
    embedding_size: int = 48
    hidden_size: int = 128
    layers: int = 1
    learning_rate: float = 0.003
    batch_size: int = 32
    steps: int = 1200
    eval_every: int = 200
    validation_limit: int = 96
    max_new_bytes: int = 80
    threads: int = 2

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key != "seed" and value <= 0:
                raise ValueError(f"{key} debe ser positivo")
        if self.layers not in (1, 2):
            raise ValueError("La referencia admite una o dos capas")


def seed_everything(seed: int, threads: int = 2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        seed_everything(config.seed, config.threads)
        self.model = ByteGRU(ModelConfig(config.embedding_size, config.hidden_size, config.layers))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.rng = random.Random(config.seed)
        self.stats = {
            "updates": 0,
            "presentations": 0,
            "input_bytes": 0,
            "answer_bytes": 0,
            "supervised_tokens": 0,
            "training_seconds": 0.0,
        }
        self.seen_ids = set()

    def step(self, lessons: list[Lesson]) -> float:
        self.model.train()
        inputs, labels = training_batch(lessons)
        self.optimizer.zero_grad(set_to_none=True)
        logits, _ = self.model(inputs, None)
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=IGNORE)
        if not torch.isfinite(loss):
            raise FloatingPointError("Pérdida no finita")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
        self.optimizer.step()
        self.stats["updates"] += 1
        self.stats["presentations"] += len(lessons)
        self.stats["input_bytes"] += sum(
            len(x.situation.encode()) + len(x.question.encode()) for x in lessons
        )
        self.stats["answer_bytes"] += sum(len(x.answer.encode()) for x in lessons)
        self.stats["supervised_tokens"] += int((labels != IGNORE).sum())
        self.seen_ids.update(x.id for x in lessons)
        return float(loss.detach())

    def train(self, lessons: list[Lesson], steps: int, callback=None) -> list[dict]:
        if not lessons or any(x.split != "train" for x in lessons):
            raise ValueError("Se requiere entrenamiento sin datos reservados")
        if steps < 1:
            raise ValueError("La cantidad de actualizaciones debe ser positiva")
        history, losses = [], []
        for i in range(steps):
            batch = self.rng.choices(lessons, k=self.config.batch_size)
            started = time.perf_counter()
            loss = self.step(batch)
            self.stats["training_seconds"] += time.perf_counter() - started
            losses.append(loss)
            if self.stats["updates"] % self.config.eval_every == 0 or i == steps - 1:
                row = {
                    "step": self.stats["updates"],
                    "loss": sum(losses) / len(losses),
                    "unique_examples_seen": len(self.seen_ids),
                    **self.stats,
                }
                losses.clear()
                if callback:
                    row.update(callback(self.model))
                history.append(row)
                print(
                    f"step={row['step']} loss={row['loss']:.4f} "
                    f"validation={row.get('validation_accuracy', 'n/a')}",
                    flush=True,
                )
        return history


def training_token_count(lessons: list[Lesson]) -> int:
    return sum(len(encode_prompt(x.prompt())) + len(x.answer.encode()) + 1 for x in lessons)
