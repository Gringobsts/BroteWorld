import copy

import pytest
import torch

from brain.world.environment import (
    COIN_GOAL,
    GRASS,
    HARD_TREE,
    HARD_WOOD_GOAL,
    ROCK,
    SIGNALS,
    STONE_GOAL,
    TREE,
    WOOD_GOAL,
    World,
    coordinate_hash,
)
from brain.world.learning import Learner
from brain.world.server import Session


@pytest.mark.parametrize(
    "tile,field,goal,action,signal",
    [
        (TREE, "wood", WOOD_GOAL, 5, "arbol_talado"),
        (HARD_TREE, "hard_wood", HARD_WOOD_GOAL, 5, "arbol_duro_talado"),
        (ROCK, "stone", STONE_GOAL, 11, "roca_extraida"),
    ],
)
def test_completed_resource_gives_no_hit_or_extraction_reward(tile, field, goal, action, signal):
    world = World()
    world.axe = world.pickaxe = 1
    setattr(world, field, goal)
    x, y = world.x, world.y - 1
    world.overrides[world.key(x, y)] = tile
    for _ in range(3):
        assert world.step(action) == 0
        assert world.last_reward_reduced
    assert getattr(world, field) == goal + 3  # Physical collection remains possible.
    assert world.last_reward_components == {signal: 0}
    assert world.step(action) < 0  # Wrong actions still get their normal penalty.
    assert not world.last_reward_reduced


@pytest.mark.parametrize(
    "tile,field,goal,action,signal",
    [
        (TREE, "wood", WOOD_GOAL, 5, "arbol_talado"),
        (HARD_TREE, "hard_wood", HARD_WOOD_GOAL, 5, "arbol_duro_talado"),
        (ROCK, "stone", STONE_GOAL, 11, "roca_extraida"),
    ],
)
def test_last_needed_resource_receives_full_reward(tile, field, goal, action, signal):
    world = World()
    world.axe = world.pickaxe = 1
    setattr(world, field, goal - 1)
    x, y = world.x, world.y - 1
    world.overrides[world.key(x, y)] = tile
    for _ in range(2):
        assert world.step(action) == SIGNALS["golpe"]
    assert world.step(action) == SIGNALS[signal]
    assert not world.last_reward_reduced
    assert getattr(world, field) == goal + 2


def set_stone_here(world):
    for x in range(100, 1000):
        if coordinate_hash(x, 101, world.seed + 7717) % 17 == 0:
            world.x, world.y = x, 101
            world.overrides[world.key(x, 101)] = GRASS
            return
    raise AssertionError("No stone")


@pytest.mark.parametrize("completed", [False, True])
def test_floor_stone_reward_depends_on_mission_progress(completed):
    world = World()
    set_stone_here(world)
    world.stone = STONE_GOAL if completed else STONE_GOAL - 1
    assert world.step(15) == (0 if completed else SIGNALS["piedra_suelta"])
    assert world.last_reward_reduced == completed
    assert world.step(15) < 0


def test_spending_materials_does_not_reenable_a_completed_quota():
    world = World()
    world.wood, world.stone = WOOD_GOAL, STONE_GOAL
    world.step(9)
    assert world.wood == WOOD_GOAL - 2 and world.stone == STONE_GOAL - 3
    world.overrides[world.key(world.x, world.y - 1)] = TREE
    assert world.step(5) == 0
    set_stone_here(world)
    assert world.step(15) == 0


def test_completed_resource_does_not_cancel_the_deadline_penalty():
    world = World()
    world.wood = WOOD_GOAL
    world.turn = world.next_penalty_turn - 1
    world.overrides[world.key(world.x, world.y - 1)] = TREE
    assert world.step(5) == SIGNALS["plazo_sin_cumplir"]
    assert world.last_reward_components == {"golpe": 0, "plazo_sin_cumplir": -25}
    assert world.last_reward_reduced


def test_next_mission_reenables_rewards_without_resetting_inventory():
    world = World()
    world.coins, world.stone, world.hard_wood = COIN_GOAL, STONE_GOAL, HARD_WOOD_GOAL
    world.wood = WOOD_GOAL - 1
    x, y = world.x, world.y - 1
    world.overrides[world.key(x, y)] = TREE
    world.damaged[world.key(x, y)] = 1
    assert world.step(5) == SIGNALS["arbol_talado"] + SIGNALS["objetivo_completo"]
    assert world.missions_completed == 1 and world.wood == WOOD_GOAL + 2
    world.overrides[world.key(world.x + 1, world.y)] = TREE
    assert world.step(6) == SIGNALS["golpe"]
    set_stone_here(world)
    assert world.step(15) == SIGNALS["piedra_suelta"]


@pytest.mark.parametrize("previous_version", [None, 2])
def test_old_reward_checkpoint_finishes_pending_experience_without_reset(tmp_path, previous_version):
    learner, world = Learner(), World()
    world.reward_version = previous_version or 1
    world.wood = WOOD_GOAL
    for _ in range(67):
        learner.live_step(world)
    expected = copy.deepcopy(learner)
    expected.update(world.observe())
    path = tmp_path / "live.pt"
    learner.save(path, world)
    payload = torch.load(path, weights_only=True)
    if previous_version is None:
        payload["world"].pop("reward_version")
    payload["world"].pop("last_reward_reduced")
    torch.save(payload, path)
    session = Session(path)
    assert session.world.reward_version == 3
    assert session.learner.steps == learner.steps
    assert session.learner.updates == expected.updates
    assert session.learner.buffer == []
    assert session.world.inventory() == world.inventory()
    assert torch.equal(session.learner.rng.get_state(), learner.rng.get_state())
    for name, value in expected.policy.state_dict().items():
        assert torch.equal(value, session.learner.policy.state_dict()[name])
    for param, other in zip(
        expected.policy.parameters(), session.learner.policy.parameters(), strict=True
    ):
        assert torch.equal(
            expected.optimizer.state[param]["exp_avg"],
            session.learner.optimizer.state[other]["exp_avg"],
        )
    session.save()
    restored = Session(path)
    assert restored.learner.updates == session.learner.updates
    assert restored.world.reward_version == 3
