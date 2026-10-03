import copy
import json
from dataclasses import asdict

import torch

from brain.world.environment import (
    ACTION_COUNT,
    GRASS,
    MAX_CACHED_CHUNKS,
    OBS_SIZE,
    PATH,
    REGROW_TURNS,
    ROCK,
    STUMP,
    TREE,
    WORLD_SIZE,
    World,
)
from brain.world.learning import Learner
from brain.world.server import Session


def plant(world, dx, dy):
    x, y = world.x + dx, world.y + dy
    world.overrides[world.key(x, y)] = TREE
    return x, y


def test_trillion_block_world_is_sparse_and_deterministic():
    world = World(101)
    assert world.width == world.height == WORLD_SIZE
    assert world.width * world.height == 1_000_000_000_000
    assert len(world.visits) == 1 and world.overrides == {}
    assert len(json.dumps(world.state())) < 2500
    other = World(101)
    for x, y in ((5, 5), (900123, 100123), (999998, 999998), (12345, 654321)):
        assert world.tile_at(x, y) == other.tile_at(x, y)
    for i in range(100):
        world.tile_at(1000 + i * 100, 1000 + i * 100)
    assert len(world._chunks) <= MAX_CACHED_CHUNKS
    assert len(json.dumps(world.state())) < 2500
    world.x, world.y = 999998, 999998
    view = world.viewport()
    assert len(view["tiles"]) == 18 and all(len(row) == 24 for row in view["tiles"])
    assert view["viewport"]["x"] + 24 <= WORLD_SIZE
    assert view["viewport"]["y"] + 18 <= WORLD_SIZE
    assert not world.walkable(-1, 3) and not world.walkable(WORLD_SIZE, 3)


def test_local_sensors_distinguish_tree_from_rock_and_health():
    world = World()
    x, y = plant(world, 0, -1)
    world.overrides[world.key(world.x + 1, world.y)] = ROCK
    before = world.observe()
    assert len(before) == OBS_SIZE == 72
    assert before[2] == 0  # North is blocked.
    assert before[18 + 1] == 1  # North contains a full-health tree.
    assert before[18 + 5] == 0  # East is a rock, not a tree.
    world.step(5)
    assert world.observe()[18 + 1] == 2 / 3
    assert world.tree_health(x, y) == 2


def test_three_hits_cost_three_turns_and_yield_wood_once():
    world = World()
    x, y = plant(world, 0, -1)
    assert world.step(5) > 0 and world.wood == 0
    assert world.step(5) > 0 and world.wood == 0
    assert world.step(5) > 0 and world.wood == 3
    assert world.trees_felled == 1 and world.chop_hits == 3
    assert world.steps == world.turn == 3
    assert world.tile_at(x, y) == STUMP and world.walkable(x, y)
    assert world.stumps[world.key(x, y)] == 3 + 100_000
    assert world.step(5) < 0 and world.wood == 3
    assert world.last_signal == "sin_arbol"


def test_all_invalid_physical_actions_give_negative_signal():
    world = World()
    x, y = world.x + 1, world.y
    for block in (ROCK, TREE):
        world.overrides[world.key(x, y)] = block
        assert world.step(1) < 0 and world.last_signal == "obstaculo"
    for block in (GRASS, PATH, ROCK):
        world.overrides[world.key(x, y)] = block
        assert world.step(6) < 0 and world.last_signal == "sin_arbol"
    assert world.steps == 5 and world.wood == 0


def test_regrowth_exactly_100000_turns_after_felling():
    world = World()
    x, y = plant(world, 0, -1)
    for _ in range(3):
        world.step(5)
    due = world.turn + REGROW_TURNS
    world.turn = due - 2
    world.step(4)
    assert world.tile_at(x, y) == STUMP
    world.step(4)
    assert world.turn == due and world.tile_at(x, y) == TREE
    assert world.tree_health(x, y) == 3
    assert world.wood == 3


def test_tree_does_not_regrow_inside_agent():
    world = World()
    x, y = plant(world, 0, -1)
    for _ in range(3):
        world.step(5)
    due = world.stumps[world.key(x, y)]
    world.step(0)
    world.turn = due - 1
    world.step(4)
    assert world.tile_at(x, y) == STUMP
    world.step(2)
    world.step(4)
    assert world.tile_at(x, y) == TREE


def test_checkpoint_restores_inventory_sparse_changes_and_regrowth(tmp_path):
    learner, world = Learner(), World()
    x, y = plant(world, 0, -1)
    for _ in range(3):
        world.step(5)
    world.step(1)
    path = tmp_path / "forest.pt"
    learner.save(path, world)
    restored, other = Learner.load(path)
    assert other.state() == world.state()
    due = other.stumps[other.key(x, y)]
    other.turn = due - 1
    other.step(4)
    assert other.tile_at(x, y) == TREE and other.wood == 3
    assert restored.policy.actor.out_features == ACTION_COUNT


def test_legacy_migration_preserves_weights_adam_and_original_file(tmp_path):
    old = Learner(observation_size=18, action_count=5)
    observation = [1.0, 0.5] * 9
    action, log_prob, value = old.choose(observation)
    old.buffer = [
        {"obs": observation, "action": action, "log_prob": log_prob, "value": value, "reward": 0.25}
    ]
    old.update(observation)
    tiles = [[2 if x in (0, 23) or y in (0, 17) else 0 for x in range(24)] for y in range(18)]
    visits = [[0] * 24 for _ in range(18)]
    visits[9][12] = 5
    legacy_world = {
        "seed": 17,
        "width": 24,
        "height": 18,
        "tiles": tiles,
        "visits": visits,
        "x": 12,
        "y": 9,
        "steps": 5,
        "moves": 4,
        "collisions": 0,
        "rests": 1,
        "last_action": 4,
        "last_reward": -0.06,
        "recent": [],
        "reachable": 352,
    }
    payload = {
        "format": "brain-world-v1",
        "config": asdict(old.config),
        "weights": old.policy.state_dict(),
        "optimizer": old.optimizer.state_dict(),
        "rng": old.rng.get_state(),
        "buffer": [],
        "steps": 5,
        "updates": old.updates,
        "history": [],
        "loss": old.last_loss,
        "entropy": old.last_entropy,
        "world": legacy_world,
    }
    path = tmp_path / "live.pt"
    torch.save(payload, path)
    original = path.read_bytes()
    learner, world = Learner.load(path, backup_legacy=True)
    assert path.read_bytes() == original
    assert path.with_name("live.v1-backup.pt").read_bytes() == original
    assert world.width == WORLD_SIZE and world.turn == 5
    assert learner.steps == 5 and learner.buffer == []
    a, _ = old.policy(torch.tensor(observation))
    b, _ = learner.policy(torch.tensor(observation + [0.0] * (OBS_SIZE - 18)))
    assert torch.allclose(a, b[:5], atol=1e-7, rtol=1e-6)
    for (name, old_p), (new_name, new_p) in zip(
        old.policy.named_parameters(), learner.policy.named_parameters(), strict=True
    ):
        assert name == new_name
        region = tuple(slice(0, n) for n in old_p.shape)
        assert torch.equal(old_p, new_p[region])
        assert torch.equal(
            old.optimizer.state[old_p]["exp_avg"], learner.optimizer.state[new_p]["exp_avg"][region]
        )
    learner.live_step(world)
    learner.update(world.observe())  # Expanded Adam moments must be executable.


def test_new_terrain_preserves_inventory_and_model(tmp_path):
    session = Session(tmp_path / "world.pt")
    session.world.wood = 12
    before = copy.deepcopy(session.learner.policy.state_dict())
    session.control({"action": "terrain"})
    assert session.world.wood == 12
    assert all(torch.equal(v, before[k]) for k, v in session.learner.policy.state_dict().items())
    state = session.snapshot()
    assert state["world"]["width"] == WORLD_SIZE
    assert state["agent"]["inventory"]["wood"] == 12
    assert state["world"]["regrow_turns"] == 100000


def test_mind_panel_reports_actual_connections_parameters_and_sizes(tmp_path):
    session = Session(tmp_path / "world.pt")
    info = session.snapshot()["learning"]
    assert info["inputs"] == OBS_SIZE and info["actions"] == ACTION_COUNT
    assert info["connections"] == OBS_SIZE * 64 + 64 * 64 + 64 * ACTION_COUNT + 64
    assert info["parameters"] == info["connections"] + 64 + 64 + ACTION_COUNT + 1
    assert info["model_bytes"] == info["parameters"] * 4
    assert info["checkpoint_bytes"] == 0
    session.save()
    assert session.snapshot()["learning"]["checkpoint_bytes"] == session.checkpoint.stat().st_size
