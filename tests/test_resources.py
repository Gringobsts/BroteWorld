import copy
from dataclasses import asdict

import pytest
import torch

from brain.world.environment import (
    ACTION_COUNT,
    COIN_GOAL,
    HARD_TREE,
    HARD_WOOD_GOAL,
    OBS_SIZE,
    PATH,
    REGROW_TURNS,
    ROCK,
    STONE_GOAL,
    TREE,
    WOOD_GOAL,
    World,
    coordinate_hash,
)
from brain.world.learning import Learner
from brain.world.server import Session


def place(world, tile, dx=0, dy=-1):
    x, y = world.x + dx, world.y + dy
    world.overrides[world.key(x, y)] = tile
    return x, y


def find_stone(world):
    for x in range(100, 1000):
        if coordinate_hash(x, 101, world.seed + 7717) % 17 == 0:
            world.x, world.y = x - 1, 101
            place(world, PATH, 0, 0)
            place(world, PATH, 1, 0)
            return x, 101
    raise AssertionError("No stone found")


def test_walk_then_pick_up_are_two_distinct_actions_on_same_cell():
    world = World()
    x, y = find_stone(world)
    assert world.stone_at(x, y) and world.walkable(x, y)
    assert world.observe()[57 + 5] == 1
    world.step(1)
    assert (world.x, world.y, world.steps, world.stone) == (x, y, 1, 0)
    assert world.stone_at(x, y) and world.observe()[57 + 4] == 1
    assert "piedra_suelta" not in world.last_reward_components
    assert world.step(15) > 0
    assert (world.x, world.y, world.steps, world.stone) == (x, y, 2, 1)
    assert not world.stone_at(x, y)
    assert world.step(15) < 0 and world.stone == 1 and world.steps == 3
    assert world.last_signal == "sin_piedra_suelta"


def test_cannot_pick_up_an_adjacent_stone():
    world = World()
    x, y = find_stone(world)
    world.stone_taken[world.key(world.x, world.y)] = True
    assert world.step(15) < 0 and world.stone == 0
    assert world.stone_at(x, y)


@pytest.mark.parametrize("tile", [TREE, HARD_TREE, ROCK])
def test_large_objects_block_movement(tile):
    world = World()
    x, y = place(world, tile)
    assert not world.walkable(x, y)
    start = (world.x, world.y)
    assert world.step(0) < 0
    assert (world.x, world.y) == start and world.steps == 1


def test_soft_tree_bare_hands_and_hard_tree_needs_axe():
    world = World()
    place(world, TREE)
    assert world.axe == world.pickaxe == 0
    for _ in range(3):
        assert world.step(5) > 0
    assert world.wood == 3 and world.hard_wood == 0
    x, y = place(world, HARD_TREE, 1, 0)
    world.pickaxe = 1  # The wrong tool must not work.
    assert world.step(6) < 0 and world.last_signal == "falta_hacha"
    assert world.tree_health(x, y) == 3
    world.axe = 1
    for _ in range(3):
        assert world.step(6) > 0
    assert world.wood == 3 and world.hard_wood == 3
    assert world.walkable(x, y)


@pytest.mark.parametrize("action,tool", [(9, "axe"), (10, "pickaxe")])
def test_recipes_consume_exact_materials_once_and_keep_mission_progress(action, tool):
    world = World()
    world.wood, world.stone = 2, 3
    before = world.goal_state()
    assert world.step(action) > 0
    assert world.steps == 1 and getattr(world, tool) == 1
    assert world.wood == world.stone == 0
    assert world.goal_state()["soft_wood"] == before["soft_wood"] == 2
    assert world.goal_state()["stone"] == before["stone"] == 3
    world.wood, world.stone = 5, 6
    assert world.step(action) < 0
    assert world.wood == 5 and world.stone == 6
    assert world.last_signal == "herramienta_existente"


@pytest.mark.parametrize("action", [9, 10])
@pytest.mark.parametrize("wood,stone", [(1, 3), (2, 2), (0, 0)])
def test_cannot_craft_without_exact_materials(action, wood, stone):
    world = World()
    world.wood, world.stone, world.hard_wood = wood, stone, 1000
    assert world.step(action) < 0
    assert (world.wood, world.stone, world.hard_wood) == (wood, stone, 1000)
    assert world.axe == world.pickaxe == 0
    assert world.last_signal == "materiales_insuficientes"


def test_mining_requires_pickaxe_and_opens_rock_tile_after_three_hits():
    world = World()
    x, y = place(world, ROCK)
    world.axe = 1
    assert world.step(11) < 0 and world.last_signal == "falta_pico"
    assert world.rock_health(x, y) == 3
    world.pickaxe = 1
    for remaining in (2, 1, 0):
        assert world.step(11) > 0
        assert world.rock_health(x, y) == remaining
    assert world.stone == 3 and world.rocks_broken == 1 and world.walkable(x, y)
    assert world.step(11) < 0 and world.stone == 3
    world.step(0)
    assert (world.x, world.y) == (x, y)


@pytest.mark.parametrize("tile", [TREE, HARD_TREE])
def test_both_tree_types_regrow_as_same_type(tile):
    world = World()
    world.axe = 1
    x, y = place(world, tile)
    for _ in range(3):
        world.step(5)
    world.turn += REGROW_TURNS - 1
    world.step(4)
    assert world.tile_at(x, y) == tile and world.tree_health(x, y) == 3


def test_new_sensors_distinguish_types_tools_and_available_materials():
    world = World()
    place(world, HARD_TREE)
    place(world, TREE, 1, 0)
    place(world, ROCK, -1, 0)
    world.wood, world.stone = 2, 2
    world.axe = 1
    obs = world.observe()
    assert len(obs) == OBS_SIZE == 72 and all(0 <= v <= 1 for v in obs)
    assert obs[18 + 1] == obs[18 + 5] == 1
    assert obs[39 + 1] == 1 and obs[39 + 5] == 0
    assert obs[48 + 3] == 1
    assert obs[66:70] == [1, 0, 1, 2 / 3]


@pytest.mark.parametrize("missing", ["coins", "wood", "stone", "hard_wood"])
def test_all_four_goals_are_required_before_renewal(missing):
    world = World()
    world.coins, world.wood = COIN_GOAL, WOOD_GOAL
    world.stone, world.hard_wood = STONE_GOAL, HARD_WOOD_GOAL
    setattr(world, missing, getattr(world, missing) - 1)
    world.turn = world.next_penalty_turn - 1
    world.step(4)
    assert world.missions_completed == 0
    assert "objetivo_completo" not in world.last_reward_components
    assert world.last_reward_components["plazo_sin_cumplir"] == -25


def test_resource_and_tool_state_persists_across_checkpoint_and_terrain(tmp_path):
    session = Session(tmp_path / "live.pt")
    world = session.world
    x, y = find_stone(world)
    world.step(1)
    world.step(15)
    world.wood, world.stone = 4, 6
    world.step(9)
    world.step(10)
    place(world, HARD_TREE)
    world.step(5)
    place(world, ROCK, 1, 0)
    world.step(12)
    session.learner.steps = world.turn
    session.save()
    restored = Session(session.checkpoint)
    assert restored.world.state() == world.state()
    assert not restored.world.stone_at(x, y)
    inventory, goal = restored.world.inventory(), restored.world.goal_state()
    restored.control({"action": "terrain"})
    assert restored.world.inventory() == inventory and restored.world.goal_state() == goal


@pytest.mark.parametrize("pending", [0, 3])
def test_v2_upgrade_finishes_pending_experience_and_preserves_all_old_connections(
    tmp_path, pending
):
    old = Learner(observation_size=39, action_count=9)
    world = World()
    world.wood, world.coins = 345, 50
    for _ in range(4 + pending):
        observation = world.legacy_observation_v2()
        action, log_prob, value = old.choose(observation)
        reward = world.step(action)
        old.steps += 1
        old.buffer.append(
            dict(obs=observation, action=action, log_prob=log_prob, value=value, reward=reward)
        )
        if old.steps == 4:
            old.update(world.legacy_observation_v2())
    state = world.state()
    state.pop("rules_version")
    state.pop("axe")  # V2 had an implicit starter axe in the API.
    payload = dict(
        format="brain-world-v2",
        config=asdict(old.config),
        weights=old.policy.state_dict(),
        optimizer=old.optimizer.state_dict(),
        rng=old.rng.get_state(),
        buffer=old.buffer,
        steps=old.steps,
        updates=old.updates,
        history=old.history,
        loss=old.last_loss,
        entropy=old.last_entropy,
        world=state,
    )
    path = tmp_path / "live.pt"
    torch.save(payload, path)
    expected = copy.deepcopy(old)
    expected.update(world.legacy_observation_v2())
    upgraded, restored = Learner.load(path, backup_legacy=True)
    assert path.read_bytes() == path.with_name("live.v2-backup.pt").read_bytes()
    assert restored.wood == world.wood and restored.coins == world.coins and restored.axe == 1
    assert upgraded.steps == old.steps and upgraded.updates == expected.updates
    assert upgraded.migration["old_experiences_completed"] == pending
    assert upgraded.buffer == [] and torch.equal(upgraded.rng.get_state(), old.rng.get_state())
    for (name, previous), (new_name, current) in zip(
        expected.policy.named_parameters(), upgraded.policy.named_parameters(), strict=True
    ):
        assert name == new_name
        region = tuple(slice(0, n) for n in previous.shape)
        assert torch.equal(previous, current[region])
        assert torch.equal(
            expected.optimizer.state[previous]["exp_avg"],
            upgraded.optimizer.state[current]["exp_avg"][region],
        )
        assert torch.equal(
            expected.optimizer.state[previous]["exp_avg_sq"],
            upgraded.optimizer.state[current]["exp_avg_sq"][region],
        )
    assert upgraded.policy.actor.out_features == ACTION_COUNT == 16
    upgraded.live_step(restored)
    upgraded.update(restored.observe())
