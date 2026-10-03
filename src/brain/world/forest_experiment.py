"""Independent forest learning and retention measurements; never replace the live mind."""

import argparse
import json
import time
from pathlib import Path

import torch

from brain.world.environment import World
from brain.world.learning import Learner, benchmark


def evaluate(learner):
    return {
        "forest": benchmark(learner, seeds=(1201, 1202, 1203, 1204, 1205), steps=2048, trees=True),
        "walking": benchmark(learner, seeds=(1301, 1302, 1303, 1304, 1305), steps=1024),
    }


def main():
    parser = argparse.ArgumentParser(description="Aprendizaje de tala, retención y reinicio")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--steps", type=int, default=131072)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or (args.out.exists() and any(args.out.iterdir())):
        parser.error("Usá steps positivo y una carpeta vacía")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.checkpoint:
        learner, world = Learner.load(args.checkpoint)
    else:
        learner, world = Learner(), World()
    learner.save(args.out / "initial.pt", world)
    before = evaluate(learner)
    started = time.perf_counter()
    train_seeds = []
    initial_turn = learner.steps
    for i in range(args.steps):
        if i % 4096 == 0:
            learner.update(world.observe())
            seed = 17 + i // 4096
            train_seeds.append(seed)
            world = World(seed)
        learner.live_step(world)
        if (i + 1) % 32768 == 0:
            print(f"Entrenamiento independiente: {i + 1} turnos", flush=True)
    elapsed = time.perf_counter() - started
    after = evaluate(learner)
    learner.save(args.out / "trained.pt", world)
    restored, restored_world = Learner.load(args.out / "trained.pt")
    restart_identical = after == evaluate(restored)
    actions_identical = True
    for _ in range(65):
        actions_identical &= learner.live_step(world) == restored.live_step(restored_world)
    continuation_identical = (
        actions_identical
        and world.state() == restored_world.state()
        and all(
            torch.equal(value, restored.policy.state_dict()[key])
            for key, value in learner.policy.state_dict().items()
        )
    )
    report = {
        "training_turns": args.steps,
        "initial_turn": initial_turn,
        "training_seeds": train_seeds,
        "seconds": elapsed,
        "before": before,
        "after": after,
        "restart_identical": restart_identical,
        "continuation_identical": continuation_identical,
        "note": "Single training run; held-out maps are evaluated without learning. "
        "New training terrain every 4096 turns resets trial missions. "
        "This experiment does not test full mission completion or 50000-turn behavior.",
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
