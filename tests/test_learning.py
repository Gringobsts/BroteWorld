import json
import subprocess
import sys
from dataclasses import replace

import pytest
import torch

from brain.baselines import ByteNGram, Majority, NearestExample
from brain.checkpoint import load_checkpoint, save_checkpoint
from brain.encoding import EOS, IGNORE, decode_answer, encode_prompt, training_batch
from brain.evaluation import evaluate, normalize, retention
from brain.model import ByteGRU, ModelConfig
from brain.training import TrainConfig, Trainer


def test_utf8_and_loss_mask(tiny_lessons):
    lesson = replace(
        tiny_lessons[0], situation="tamaño=pequeño", answer="pequeña", valid_answers=("pequeña",)
    )
    inputs, labels = training_batch([lesson, tiny_lessons[1]])
    prefix = encode_prompt(lesson.prompt())
    assert (labels[0, : len(prefix) - 1] == IGNORE).all()
    assert labels[0, len(prefix) - 1 :].tolist() == list("pequeña".encode()) + [EOS]
    assert inputs[0, : len(prefix)].tolist() == prefix
    assert (
        labels[
            1, len(encode_prompt(tiny_lessons[1].prompt())) + len(tiny_lessons[1].answer.encode()) :
        ]
        == IGNORE
    ).all()
    assert decode_answer(list("ñá¿".encode()) + [EOS]) == "ñá¿"


def test_all_training_paths_reject_exam(tiny_lessons):
    exam = [replace(tiny_lessons[0], split="test_composition")]
    for reference in (Majority, NearestExample, ByteNGram, training_batch):
        with pytest.raises(ValueError):
            reference(exam)
    with pytest.raises(ValueError):
        Trainer(TrainConfig()).train(exam, 1)


def test_evaluation_is_blind_and_labels_are_external(tiny_lessons):
    class Spy:
        def predict(self, prompts, max_new_bytes=80):
            assert all(set(vars(x)) == {"situation", "question", "skill"} for x in prompts)
            return ["roja"] * len(prompts)

    metric = evaluate(Spy(), tiny_lessons)
    assert metric["accuracy"] == 0.25
    changed = [replace(x, answer="roja", valid_answers=("roja",)) for x in tiny_lessons]
    assert evaluate(Spy(), changed)["accuracy"] == 1
    assert normalize("  LA pelota es ROJA.  ") == "la pelota es roja"
    assert normalize("pequeña") != normalize("pequena")


def test_evaluation_does_not_change_weights_or_leak_state(tiny_lessons):
    trainer = Trainer(TrainConfig(hidden_size=24, embedding_size=12))
    before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    rows = tiny_lessons[:2]
    first = trainer.model.predict([x.prompt() for x in rows], 8)
    trainer.model.predict([tiny_lessons[3].prompt()], 8)
    assert trainer.model.predict([x.prompt() for x in rows], 8) == first
    assert trainer.model.predict([rows[1].prompt(), rows[0].prompt()], 8) == first[::-1]
    evaluate(trainer.model, rows, max_new_bytes=8)
    assert trainer.model.training
    assert all(torch.equal(v, before[k]) for k, v in trainer.model.state_dict().items())
    assert all(p.grad is None for p in trainer.model.parameters())


def test_recurrent_weights_receive_finite_gradients(tiny_lessons):
    trainer = Trainer(TrainConfig(hidden_size=24, embedding_size=12))
    before = trainer.model.recurrent.weight_hh_l0.detach().clone()
    trainer.step(tiny_lessons)
    assert not torch.equal(before, trainer.model.recurrent.weight_hh_l0)
    for parameter in trainer.model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_recurrent_input_gradient_numerically():
    torch.manual_seed(1)
    model = ByteGRU(ModelConfig(3, 3, 1)).double()
    features = torch.randn(1, 2, 3, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: model.recurrent(x)[0], (features,))


@pytest.mark.integration
def test_learns_tiny_set_and_survives_fresh_process(tmp_path, tiny_lessons):
    trainer = Trainer(
        TrainConfig(
            seed=3,
            hidden_size=48,
            embedding_size=16,
            learning_rate=0.015,
            batch_size=4,
            eval_every=180,
        )
    )
    before = evaluate(trainer.model, tiny_lessons, max_new_bytes=12)
    trainer.train(tiny_lessons, 180)
    after = evaluate(trainer.model, tiny_lessons, max_new_bytes=12, include_predictions=True)
    assert before["accuracy"] == 0
    assert after["accuracy"] == 1
    checkpoint = tmp_path / "trained.pt"
    save_checkpoint(checkpoint, trainer, {"test": "tiny"})
    request = tmp_path / "request.json"
    request.write_text(json.dumps([vars(x.prompt()) for x in tiny_lessons]))
    script = """
import json, sys
from pathlib import Path
from brain.checkpoint import load_checkpoint
from brain.data import Prompt
trainer, _ = load_checkpoint(Path(sys.argv[1]))
prompts = [Prompt(**row) for row in json.loads(Path(sys.argv[2]).read_text())]
print(json.dumps(trainer.model.predict(prompts, 12)))
"""
    output = subprocess.check_output(
        [sys.executable, "-c", script, str(checkpoint), str(request)], text=True
    )
    assert json.loads(output) == [x["prediction"] for x in after["predictions"]]
    # Optimizer momentum, update count and random sampling also resume exactly.
    trainer.train(tiny_lessons, 2)
    restored, manifest = load_checkpoint(checkpoint)
    assert manifest == {"test": "tiny"}
    restored.train(tiny_lessons, 2)
    assert trainer.stats["updates"] == restored.stats["updates"]
    assert trainer.stats["presentations"] == restored.stats["presentations"]
    assert all(
        torch.equal(v, restored.model.state_dict()[k])
        for k, v in trainer.model.state_dict().items()
    )


def test_retention_requires_same_probe(tiny_lessons):
    class Constant:
        def predict(self, prompts, max_new_bytes=80):
            return ["roja"] * len(prompts)

    before = evaluate(Constant(), tiny_lessons)
    assert retention(before, before)["drop_pp"] == 0
    after = {**before, "accuracy": 0.1}
    assert retention(before, after)["drop_pp"] == 15
    assert not retention(before, after)["passed"]
    different = evaluate(Constant(), tiny_lessons[::-1])
    with pytest.raises(ValueError):
        retention(before, different)
