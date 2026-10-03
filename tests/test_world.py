import copy
import json
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest
import torch

from brain.world.clock import TurnClock
from brain.world.environment import COIN_GOAL, GRASS, OBS_SIZE, SIGNALS, World
from brain.world.learning import Learner, benchmark
from brain.world.server import Session, make_handler


@pytest.mark.parametrize(
    "rate,seconds,expected", [(1, 1, 1), (10000, 1, 10000), (100000, 1, 100000), (0.25, 4, 1)]
)
def test_turn_speed_has_no_skipped_actions(rate, seconds, expected):
    clock = TurnClock()
    due = clock.budget(seconds, rate)
    executed = 0
    while due:
        assert due <= 256
        clock.consume(due)
        executed += due
        due = clock.budget(0, rate)
    assert executed == expected
    assert clock.pending == 0
    with pytest.raises(ValueError):
        clock.consume(1)


def test_fractional_time_accumulates():
    clock = TurnClock()
    assert clock.budget(0.3, 1) == 0
    assert clock.budget(0.3, 1) == 0
    assert clock.budget(0.4, 1) == 1
    clock.consume(1)
    assert clock.budget(0, 1) == 0


def test_collision_and_rest_consume_a_turn():
    world = World(17)
    world.x, world.y = 1, 1
    assert world.step(3) < 0
    assert (world.x, world.y, world.steps, world.collisions) == (1, 1, 1, 1)
    world.step(4)
    assert world.steps == 2 and world.rests == 1
    assert len(world.observe()) == OBS_SIZE
    assert all(0 <= value <= 1 for value in world.observe())


@pytest.mark.parametrize("previous_visits", [0, 25])
def test_walking_without_coins_is_neutral_even_when_exploring(previous_visits):
    world = World()
    world.coins = COIN_GOAL  # Isolate movement from the separate coin reward.
    destination = world.x + 1, world.y
    key = world.key(*destination)
    world.overrides[key] = GRASS
    world.visits[key] = previous_visits
    assert world.step(1) == 0
    assert world.last_reward_components == {"avance": 0}
    assert (world.x, world.y) == destination
    assert world.visits[key] == previous_visits + 1
    assert world.moves == world.turn == world.steps == 1


def test_walking_still_receives_the_mission_deadline_penalty():
    world = World()
    world.coins = COIN_GOAL
    world.overrides[world.key(world.x + 1, world.y)] = GRASS
    world.turn = world.next_penalty_turn - 1
    assert world.step(1) == SIGNALS["plazo_sin_cumplir"]
    assert world.last_reward_components == {"avance": 0, "plazo_sin_cumplir": -25}


def test_one_action_per_turn_and_training_continues(tmp_path):
    session = Session(tmp_path / "world.pt")
    session.control({"action": "running", "value": False})
    before = {k: v.clone() for k, v in session.learner.policy.state_dict().items()}
    for i in range(128):
        state = session.control({"action": "step"})
        assert state["clock"]["turn"] == i + 1
        assert state["metrics"]["steps"] == i + 1
        assert state["learning"]["steps"] == i + 1
    assert state["learning"]["updates"] == 2
    assert any(
        not torch.equal(v, before[k]) for k, v in session.learner.policy.state_dict().items()
    )
    session.control({"action": "running", "value": True})
    with pytest.raises(ValueError):
        session.control({"action": "step"})


def test_speed_changes_do_not_change_world_or_train(tmp_path):
    session = Session(tmp_path / "world.pt")
    for speed in (0.25, 1, 10000, 100000):
        state = session.control({"action": "speed", "value": speed})
        assert state["speed"] == speed and state["clock"]["turn"] == 0
    for value in (0, -1, 100001, True, "1000", float("nan")):
        with pytest.raises(ValueError):
            session.control({"action": "speed", "value": value})


def test_change_rate_resets_pacing_origin(tmp_path, monkeypatch):
    session = Session(tmp_path / "world.pt")
    session.clock.pending = 25
    session.pacing_at = 0
    monkeypatch.setattr("brain.world.server.time.monotonic", lambda: 500.0)
    session.control({"action": "speed", "value": 10000})
    assert session.pacing_at == 500.0 and session.clock.pending == 0
    session.clock.pending = 15
    session.control({"action": "running", "value": False})
    assert session.pacing_at == 500.0 and session.clock.pending == 0


def test_evaluation_is_isolated_and_new_terrain_preserves_brain(tmp_path):
    learner, world = Learner(), World()
    for _ in range(70):
        learner.live_step(world)
    original = copy.deepcopy(learner.policy.state_dict())
    rng = learner.rng.get_state().clone()
    buffer = copy.deepcopy(learner.buffer)
    benchmark(learner, seeds=(901,), steps=32)
    assert learner.steps == 70 and learner.buffer == buffer
    assert torch.equal(learner.rng.get_state(), rng)
    assert all(torch.equal(v, original[k]) for k, v in learner.policy.state_dict().items())
    checkpoint = tmp_path / "world.pt"
    learner.save(checkpoint, world)
    session = Session(checkpoint)
    session.control({"action": "terrain"})
    assert session.learner.steps == 70 and session.world.seed == world.seed + 1
    assert session.learner.updates == learner.updates + 1
    assert session.world.steps == 0


def test_world_checkpoint_in_new_process(tmp_path):
    learner, world = Learner(), World()
    for _ in range(69):
        learner.live_step(world)
    checkpoint = tmp_path / "world.pt"
    learner.save(checkpoint, world)
    expected_actions = [learner.live_step(world) for _ in range(70)]
    expected = {
        "actions": expected_actions,
        "x": world.x,
        "y": world.y,
        "updates": learner.updates,
        "turn": learner.steps,
    }
    script = """
import json, sys
from pathlib import Path
from brain.world.learning import Learner
learner, world = Learner.load(Path(sys.argv[1]))
actions = [learner.live_step(world) for _ in range(70)]
print(json.dumps(dict(actions=actions, x=world.x, y=world.y, updates=learner.updates, turn=learner.steps)))
"""
    actual = json.loads(
        subprocess.check_output([sys.executable, "-c", script, str(checkpoint)], text=True)
    )
    assert actual == expected


def test_http_state_and_controls(tmp_path):
    session = Session(tmp_path / "world.pt")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(session))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(url + "/api/state") as response:
            assert json.load(response)["clock"]["actions_per_agent"] == 1
        payload = json.dumps({"action": "speed", "value": 100000}).encode()
        request = urllib.request.Request(
            url + "/api/control", data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request) as response:
            assert json.load(response)["speed"] == 100000
        request.add_header("Origin", "https://unrelated.example")
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        assert error.value.code == 403
        with urllib.request.urlopen(url) as response:
            assert b"Brote" in response.read()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
