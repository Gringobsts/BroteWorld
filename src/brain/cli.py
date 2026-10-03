import argparse
import json
import platform
import resource
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from brain.baselines import ByteNGram, Majority, NearestExample
from brain.checkpoint import load_checkpoint, save_checkpoint
from brain.curriculum import write_dataset
from brain.data import Prompt, read_lessons, verify_dataset
from brain.evaluation import evaluate, retention
from brain.training import TrainConfig, Trainer


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_suite(directory, manifest, prefix):
    return {
        name.removesuffix(".jsonl"): read_lessons(directory / name)
        for name in sorted(manifest["files"])
        if name.startswith(prefix)
    }


def balanced_sample(suites: dict, limit: int):
    # A fixed subset, independent of model performance; evenly interleave split/skill buckets.
    buckets = {}
    for split, rows in suites.items():
        for row in rows:
            buckets.setdefault((split, row.skill), []).append(row)
    sample = []
    for i in range(max(map(len, buckets.values()))):
        for key in sorted(buckets):
            if i < len(buckets[key]):
                sample.append(buckets[key][i])
                if len(sample) == limit:
                    return sample
    return sample


def train_command(args):
    manifest = verify_dataset(args.data)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for key in ("seed", "steps"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    config = TrainConfig(**config)
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("La carpeta del experimento debe estar vacía")
    lessons = [
        x
        for x in read_lessons(args.data / "train.jsonl", "train")
        if args.min_stage <= x.stage <= args.max_stage
    ]
    if not lessons:
        raise ValueError("La selección de etapas está vacía")
    trainer = Trainer(config)
    if args.resume:
        trainer, saved_manifest = load_checkpoint(args.resume)
        if saved_manifest != manifest:
            raise ValueError("El checkpoint pertenece a otro conjunto de datos")
        # Keep optimizer, sampler and configuration intact for exact continuation.
        if asdict(trainer.config) | {"steps": config.steps} != asdict(config):
            raise ValueError("Al reanudar solo puede cambiar steps en la configuración")
    args.out.mkdir(parents=True, exist_ok=True)
    validation = balanced_sample(
        load_suite(args.data, manifest, "validation_"), config.validation_limit
    )
    initial = evaluate(trainer.model, validation, max_new_bytes=config.max_new_bytes)
    write_json(args.out / "frozen_control.json", initial)

    def validate(model):
        metric = evaluate(model, validation, max_new_bytes=config.max_new_bytes)
        return {"validation_accuracy": metric["accuracy"], "validation": metric}

    started = time.perf_counter()
    curve = trainer.train(lessons, config.steps, validate)
    save_checkpoint(args.out / "checkpoint.pt", trainer, manifest)
    # Captured from the live trained model, before exiting this training process.
    probe = evaluate(
        trainer.model, validation, max_new_bytes=config.max_new_bytes, include_predictions=True
    )
    write_json(args.out / "restart_probe_before.json", probe)
    write_json(args.out / "curve.json", curve)
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {
        "config": asdict(config),
        "stats": trainer.stats,
        "unique_examples_available": len(lessons),
        "unique_examples_seen": len(trainer.seen_ids),
        "parameters": sum(p.numel() for p in trainer.model.parameters()),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_bytes": usage if sys.platform == "darwin" else usage * 1024,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": "cpu",
        "min_stage": args.min_stage,
        "max_stage": args.max_stage,
        "resumed_from": str(args.resume) if args.resume else None,
        "test_used_for_training_or_selection": False,
        "initial_validation": initial,
        "final_validation": curve[-1]["validation"],
    }
    write_json(args.out / "run.json", report)
    print(
        json.dumps(
            {
                "checkpoint": str(args.out / "checkpoint.pt"),
                "parameters": report["parameters"],
                "stats": trainer.stats,
            },
            indent=2,
        )
    )


def evaluate_command(args):
    manifest = verify_dataset(args.data)
    trainer, saved_manifest = load_checkpoint(args.checkpoint)
    if manifest != saved_manifest:
        raise ValueError("El checkpoint y el examen pertenecen a versiones distintas")
    if args.frozen:
        trainer = Trainer(trainer.config)
    suites = load_suite(args.data, manifest, args.suite + "_")
    if args.suite == "train":
        suites = {"memorization": read_lessons(args.data / "train.jsonl", "train")}
    results = {
        split: evaluate(
            trainer.model,
            lessons,
            max_new_bytes=trainer.config.max_new_bytes,
            include_predictions=args.predictions,
        )
        for split, lessons in suites.items()
    }
    write_json(
        args.out,
        {
            "checkpoint": str(args.checkpoint),
            "frozen": args.frozen,
            "dataset": manifest,
            "results": results,
        },
    )
    print(
        json.dumps(
            {k: {"accuracy": v["accuracy"], "count": v["count"]} for k, v in results.items()},
            indent=2,
        )
    )


def restart_command(args):
    manifest = verify_dataset(args.data)
    trainer, saved_manifest = load_checkpoint(args.checkpoint)
    if manifest != saved_manifest:
        raise ValueError("Datos distintos del entrenamiento")
    validation = balanced_sample(
        load_suite(args.data, manifest, "validation_"), trainer.config.validation_limit
    )
    after = evaluate(
        trainer.model,
        validation,
        max_new_bytes=trainer.config.max_new_bytes,
        include_predictions=True,
    )
    before = json.loads(args.before.read_text(encoding="utf-8"))
    passed = (
        before["probe_sha256"] == after["probe_sha256"]
        and before["predictions"] == after["predictions"]
    )
    result = {
        "passed": passed,
        "before_accuracy": before["accuracy"],
        "after_accuracy": after["accuracy"],
        "after": after,
    }
    write_json(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k != "after"}, indent=2))
    if not passed:
        raise SystemExit(1)


def baselines_command(args):
    manifest = verify_dataset(args.data)
    train = read_lessons(args.data / "train.jsonl", "train")
    suites = load_suite(args.data, manifest, args.suite + "_")
    results = {}
    for name, reference in (
        ("majority", Majority),
        ("nearest", NearestExample),
        ("ngram", ByteNGram),
    ):
        predictor = reference(train)
        results[name] = {split: evaluate(predictor, rows) for split, rows in suites.items()}
        print(name, {k: round(v["accuracy"], 4) for k, v in results[name].items()}, flush=True)
    write_json(args.out, {"dataset": manifest, "results": results})


def retention_command(args):
    manifest = verify_dataset(args.data)
    lessons = [
        x for x in read_lessons(args.data / f"{args.split}.jsonl") if x.stage <= args.max_stage
    ]
    metrics = []
    for checkpoint in (args.before, args.after):
        trainer, saved = load_checkpoint(checkpoint)
        if saved != manifest:
            raise ValueError("Los checkpoints deben usar el mismo conjunto de datos")
        metrics.append(evaluate(trainer.model, lessons, max_new_bytes=trainer.config.max_new_bytes))
    result = {"before": metrics[0], "after": metrics[1], "retention": retention(*metrics)}
    write_json(args.out, result)
    print(json.dumps(result["retention"], indent=2))


def main():
    parser = argparse.ArgumentParser(description="Brain: español experimental desde cero")
    commands = parser.add_subparsers(dest="command", required=True)
    dataset = commands.add_parser("prepare", help="Crear lecciones y un examen congelado")
    dataset.add_argument("--out", type=Path, default=Path("data/v1"))
    train = commands.add_parser("train", help="Entrenar solo con lecciones autorizadas")
    train.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--seed", type=int)
    train.add_argument("--steps", type=int)
    train.add_argument("--resume", type=Path)
    train.add_argument("--min-stage", type=int, choices=range(3), default=0)
    train.add_argument("--max-stage", type=int, choices=range(3), default=2)
    evaluation = commands.add_parser("evaluate", help="Medir generación libre")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument(
        "--suite", choices=("train", "validation", "test"), default="validation"
    )
    evaluation.add_argument("--predictions", action="store_true")
    evaluation.add_argument(
        "--frozen", action="store_true", help="Evaluar la misma inicialización sin actualizaciones"
    )
    evaluation.add_argument("--out", type=Path, required=True)
    baseline = commands.add_parser("baselines", help="Evaluar mayoría, vecino y n-gramas")
    baseline.add_argument("--suite", choices=("validation", "test"), default="validation")
    baseline.add_argument("--out", type=Path, required=True)
    restart = commands.add_parser("verify-restart", help="Comparar respuestas en un proceso nuevo")
    restart.add_argument("--checkpoint", type=Path, required=True)
    restart.add_argument("--before", type=Path, required=True)
    restart.add_argument("--out", type=Path, required=True)
    retain = commands.add_parser("retention", help="Comparar habilidades previas entre checkpoints")
    retain.add_argument("--before", type=Path, required=True)
    retain.add_argument("--after", type=Path, required=True)
    retain.add_argument("--split", default="validation_scenes")
    retain.add_argument("--max-stage", type=int, choices=range(3), default=1)
    retain.add_argument("--out", type=Path, required=True)
    ask = commands.add_parser("ask", help="Consultar al alumno sin entrenarlo")
    ask.add_argument("--checkpoint", type=Path, required=True)
    ask.add_argument("--situation", required=True)
    ask.add_argument("--question", required=True)
    for command in (train, evaluation, baseline, restart, retain):
        command.add_argument("--data", type=Path, default=Path("data/v1"))
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            print(json.dumps(write_dataset(args.out), ensure_ascii=False, indent=2))
        elif args.command == "ask":
            trainer, _ = load_checkpoint(args.checkpoint)
            print(
                trainer.model.predict(
                    [Prompt(args.situation, args.question, "")], trainer.config.max_new_bytes
                )[0]
            )
        else:
            {
                "train": train_command,
                "evaluate": evaluate_command,
                "baselines": baselines_command,
                "verify-restart": restart_command,
                "retention": retention_command,
            }[args.command](args)
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        parser.exit(2, f"Error: {exc}\n")
