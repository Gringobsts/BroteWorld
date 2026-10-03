import json
from dataclasses import replace

import pytest

from brain.curriculum import audit_splits, build_lessons, write_dataset
from brain.data import read_lessons, verify_dataset


def test_splits_hold_out_pairs_scenes_and_template_families():
    splits = build_lessons()
    audit_splits(splits)
    train = splits["train"]
    train_pairs = {x.composition_group for x in train}
    assert train_pairs.isdisjoint(x.composition_group for x in splits["test_composition"])
    assert train_pairs.isdisjoint(x.composition_group for x in splits["validation_composition"])
    train_families = {x.template_family for x in train}
    assert train_families.isdisjoint(x.template_family for x in splits["test_templates"])
    assert train_families.isdisjoint(x.template_family for x in splits["validation_templates"])
    assert {x.semantic_group for x in train}.isdisjoint(
        x.semantic_group for x in splits["test_scenes"]
    )
    for dimension in ("objeto=", "color=", "tamaño="):

        def values(rows, dimension=dimension):
            return {
                part
                for row in rows
                for part in row.situation.split(";")
                if part.startswith(dimension)
            }

        assert values(splits["test_composition"]) <= values(train)


def test_handwritten_spanish_goldens():
    # These labels are independently written; no expected answer is computed by the generator.
    rows = [x for values in build_lessons().values() for x in values]
    goldens = [
        ("objeto=pelota;color=rojo;cantidad=1;tamaño=grande", "concordancia", "La pelota es roja."),
        (
            "objeto=libro;color=azul;cantidad=2;tamaño=pequeño",
            "concordancia",
            "Los libros son azules.",
        ),
        ("objeto=casa;color=verde;cantidad=2;tamaño=grande", "color", "verdes"),
        ("objeto=caja;color=negro;cantidad=2;tamaño=pequeño", "tamaño", "pequeñas"),
        ("objeto=gato;color=blanco;cantidad=1;tamaño=mediano", "nombre", "gato"),
        (
            "objeto=taza;color=amarillo;cantidad=2;tamaño=mediano",
            "concordancia",
            "Las tazas son amarillas.",
        ),
    ]
    for scene, skill, answer in goldens:
        matching = [x for x in rows if x.situation == scene and x.skill == skill]
        assert matching and all(x.answer == answer for x in matching)


def test_manifest_detects_tampering_and_prevents_overwrite(tmp_path):
    write_dataset(tmp_path)
    manifest = verify_dataset(tmp_path)
    assert manifest["files"]["train.jsonl"]["count"] > 1000
    assert read_lessons(tmp_path / "train.jsonl", "train")
    with pytest.raises(FileExistsError):
        write_dataset(tmp_path)
    with (tmp_path / "test_composition.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="modificados"):
        verify_dataset(tmp_path)


def test_audit_rejects_composition_contamination():
    splits = build_lessons()
    leaked = replace(splits["test_composition"][0], id="leaked", split="train")
    splits["train"].append(leaked)
    with pytest.raises(ValueError, match="combinaciones"):
        audit_splits(splits)


def test_serialization_is_reproducible(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    assert write_dataset(a) == write_dataset(b)
    assert json.loads((a / "manifest.json").read_text()) == verify_dataset(b)
