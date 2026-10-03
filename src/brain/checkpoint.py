"""Checkpoint de la referencia. No conserva estado oculto ni historial de conversación."""

import os
import random
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from brain.training import TrainConfig, Trainer

FORMAT = 1


def save_checkpoint(path: Path, trainer: Trainer, dataset_manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np_state = np.random.get_state()
    payload = {
        "format": FORMAT,
        "codec": "utf8-bytes+7-v1",
        "config": asdict(trainer.config),
        "model": trainer.model.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "stats": trainer.stats,
        "seen_ids": sorted(trainer.seen_ids),
        "sampler_rng": trainer.rng.getstate(),
        "python_rng": random.getstate(),
        "numpy_rng": [np_state[0], np_state[1].tolist(), *np_state[2:]],
        "torch_rng": torch.get_rng_state(),
        "dataset_manifest": dataset_manifest,
    }
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=path.parent)
    os.close(fd)
    try:
        with open(temporary, "wb") as stream:
            torch.save(payload, stream)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path: Path) -> tuple[Trainer, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["format"] != FORMAT or payload["codec"] != "utf8-bytes+7-v1":
        raise ValueError("Checkpoint incompatible")
    trainer = Trainer(TrainConfig(**payload["config"]))
    trainer.model.load_state_dict(payload["model"])
    trainer.optimizer.load_state_dict(payload["optimizer"])
    trainer.stats = payload["stats"]
    trainer.seen_ids = set(payload["seen_ids"])
    trainer.rng.setstate(payload["sampler_rng"])
    random.setstate(payload["python_rng"])
    state = payload["numpy_rng"]
    np.random.set_state((state[0], np.array(state[1], dtype=np.uint32), *state[2:]))
    torch.set_rng_state(payload["torch_rng"])
    return trainer, payload["dataset_manifest"]
