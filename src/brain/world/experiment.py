import argparse
import json
import time
from pathlib import Path

import torch

from brain.world.environment import World
from brain.world.learning import Learner, LearnerConfig, benchmark


def main():
    parser = argparse.ArgumentParser(
        description="Medir aprendizaje de locomoción en mapas reservados"
    )
    parser.add_argument("--steps", type=int, default=12288)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--out", type=Path, default=Path("runs/world-experiment"))
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Elegí una carpeta de resultados vacía")
    if args.steps < 1:
        parser.error("steps debe ser positivo")
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in args.seeds:
        learner = Learner(LearnerConfig(seed=seed))
        before = benchmark(learner)
        started = time.perf_counter()
        world = World(seed)
        for step in range(args.steps):
            if step % 1024 == 0:
                if step:
                    learner.update(world.observe())
                world = World(seed + step // 1024)
            learner.live_step(world)
            if (step + 1) % 4096 == 0:
                print(f"seed={seed} steps={step + 1} updates={learner.updates}", flush=True)
        seconds = time.perf_counter() - started
        after = benchmark(learner)
        checkpoint = args.out / f"seed-{seed}.pt"
        learner.save(checkpoint, world)
        restored, restored_world = Learner.load(checkpoint)
        # Evaluation doesn't mutate either learner; exact continuation checks training state too.
        identical = after == benchmark(restored)
        for _ in range(65):
            learner.live_step(world)
            restored.live_step(restored_world)
        continuation = all(
            torch.equal(v, restored.policy.state_dict()[k])
            for k, v in learner.policy.state_dict().items()
        )
        row = {
            "seed": seed,
            "steps": args.steps,
            "seconds": seconds,
            "before": before,
            "after": after,
            "restart_identical": identical,
            "continuation_identical": continuation,
        }
        records.append(row)
        print(
            json.dumps(
                {
                    "seed": seed,
                    "before": before["collision_rate"],
                    "after": after["collision_rate"],
                    "restart": identical,
                }
            ),
            flush=True,
        )
    summary = {
        "records": records,
        "algorithm": "PPO; local 3x3 sensors; online updates",
        "heldout_map_seeds": [1001, 1002, 1003, 1004, 1005],
        "mean": {
            phase: {
                key: sum(r[phase][key] for r in records) / len(records)
                for key in ("move_rate", "collision_rate", "coverage")
            }
            for phase in ("before", "after")
        },
    }
    (args.out / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["mean"], indent=2))


if __name__ == "__main__":
    main()
