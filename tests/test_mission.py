import pytest

from brain.world.environment import (
    COIN_GOAL,
    GRASS,
    HARD_TREE,
    HARD_WOOD_GOAL,
    PENALTY_INTERVAL,
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


def place_coin_east(world):
    for x in range(100, 1000):
        y = 101
        if coordinate_hash(x, y, world.seed + 9127 + world.missions_completed * 104729) % 37 == 0:
            world.x, world.y = x - 1, y
            for px in (x - 1, x):
                world.overrides[world.key(px, y)] = GRASS
            return x, y
    raise AssertionError("Missing seeded coin")


def test_coin_sensor_pickup_positive_and_cannot_collect_twice():
    world = World()
    x, y = place_coin_east(world)
    assert world.observe()[27 + 5] == 1
    assert World.restore(world.state()).coin_at(x, y)
    assert world.step(1) == SIGNALS["moneda"]
    assert world.coins == 1 and world.last_signal == "moneda"
    assert world.last_reward_components["moneda"] == SIGNALS["moneda"]
    assert not world.coin_at(x, y)
    world.step(3)
    assert world.step(1) == 0
    assert world.coins == 1


@pytest.mark.parametrize("coins,wood", [(0, 0), (COIN_GOAL, 0), (0, WOOD_GOAL)])
def test_deadlines_repeat_until_both_goals_met(coins, wood):
    world = World()
    world.coins, world.wood = coins, wood
    for boundary in (PENALTY_INTERVAL, 2 * PENALTY_INTERVAL):
        world.turn = boundary - 2
        world.step(4)
        assert "plazo_sin_cumplir" not in world.last_reward_components
        reward = world.step(4)
        assert reward == SIGNALS["espera"] + SIGNALS["plazo_sin_cumplir"]
        assert world.next_penalty_turn == boundary + PENALTY_INTERVAL
        assert world.goal_completed_turn is None
        world.step(4)
        assert "plazo_sin_cumplir" not in world.last_reward_components
    assert world.goal_penalties == 2


@pytest.mark.parametrize("last_resource", ["coins", "wood", "stone", "hard_wood"])
def test_completion_wins_deadline_bonus_once_and_survives_restart(tmp_path, last_resource):
    world = World()
    world.coins, world.wood = COIN_GOAL, WOOD_GOAL
    world.stone, world.hard_wood = STONE_GOAL, HARD_WOOD_GOAL
    if last_resource == "coins":
        world.coins -= 1
        place_coin_east(world)
        action = 1
    elif last_resource == "stone":
        world.stone -= 1
        key = world.key(world.x, world.y - 1)
        world.overrides[key] = ROCK
        world.rock_damaged[key] = 1
        world.pickaxe = 1
        action = 11
    else:
        if last_resource == "wood":
            world.wood -= 1
        else:
            world.hard_wood -= 1
            world.axe = 1
        key = world.key(world.x, world.y - 1)
        world.overrides[key] = TREE if last_resource == "wood" else HARD_TREE
        world.damaged[key] = 1
        action = 5
    world.turn = PENALTY_INTERVAL - 1
    assert world.step(action) >= SIGNALS["objetivo_completo"]
    assert world.goal_completed_turn == PENALTY_INTERVAL
    assert world.goal_penalties == 0
    assert SIGNALS["objetivo_completo"] > max(
        value for key, value in SIGNALS.items() if key != "objetivo_completo"
    )
    goal = world.goal_state()
    assert goal["turns_until_penalty"] == PENALTY_INTERVAL
    assert goal["coins"] == goal["wood"] == 0
    assert goal["round"] == 2 and goal["completed_missions"] == 1
    assert world.coins == COIN_GOAL and world.wood >= WOOD_GOAL
    assert world.observe()[36:39] == [0, 0, 1]
    assert world.observe()[-2:] == [0, 0]
    assert world.steps == 1  # Renewal is part of the action, never an extra turn.
    path = tmp_path / "done.pt"
    Learner().save(path, world)
    _, restored = Learner.load(path)
    assert restored.state() == world.state()
    restored.step(4)
    assert restored.last_reward_components == {"espera": SIGNALS["espera"]}
    restored.turn = 2 * PENALTY_INTERVAL - 1
    restored.step(4)
    assert restored.last_reward_components == {
        "espera": SIGNALS["espera"],
        "plazo_sin_cumplir": SIGNALS["plazo_sin_cumplir"],
    }
    assert restored.goal_penalties == 1
    assert restored.missions_completed == 1


def test_terrain_and_restart_keep_mission_clock_and_inventory(tmp_path):
    path = tmp_path / "live.pt"
    session = Session(path)
    session.learner.steps = session.world.turn = PENALTY_INTERVAL - 1
    session.world.coins, session.world.wood = 10, 90
    session.control({"action": "terrain"})
    assert session.world.goal_state()["turns_until_penalty"] == 1
    session = Session(path)
    assert session.world.coins == 10 and session.world.wood == 90
    session.advance_turn()
    assert session.world.goal_penalties == 1
    assert session.world.last_reward_components["plazo_sin_cumplir"] == -25
    assert any("50.000" in event["text"] for event in session.events)
    assert all(0 <= value <= 1 for value in session.world.observe())


def test_successive_missions_have_fresh_coins_and_require_new_resources():
    world = World()
    original_map = world.viewport()["tiles"]
    original_coins = world.viewport()["coins"]
    world.coins, world.wood = COIN_GOAL, WOOD_GOAL
    world.stone, world.hard_wood = STONE_GOAL, HARD_WOOD_GOAL
    world.step(4)
    assert world.missions_completed == 1
    assert world.viewport()["tiles"] == original_map
    assert world.viewport()["coins"] != original_coins
    x, y = place_coin_east(world)
    assert world.coin_at(x, y)  # Lifetime inventory is 100; this mission still needs 100.
    world.step(1)
    assert world.coins == COIN_GOAL + 1 and world.goal_state()["coins"] == 1
    assert world.missions_completed == 1
    assert "objetivo_completo" not in world.last_reward_components
    world.coins = 2 * COIN_GOAL - 1
    world.wood = 2 * WOOD_GOAL
    world.stone, world.hard_wood = 2 * STONE_GOAL, 2 * HARD_WOOD_GOAL
    assert not world.coin_at(x, y)  # Cannot farm a collected coin in the same round.
    world.coin_taken = {}  # Set up the final pickup without simulating 99 more pickups.
    world.x -= 1
    world.step(1)
    assert world.missions_completed == 2
    assert world.last_reward_components["objetivo_completo"] == 100
    assert world.coins == 2 * COIN_GOAL and world.wood == 2 * WOOD_GOAL
    assert world.goal_state()["round"] == 3
    assert world.goal_state()["coins"] == world.goal_state()["wood"] == 0
    assert world.goal_state()["turns_until_penalty"] == PENALTY_INTERVAL


def test_terrain_and_restart_preserve_second_mission_offsets(tmp_path):
    path = tmp_path / "live.pt"
    session = Session(path)
    session.world.coins, session.world.wood = COIN_GOAL, WOOD_GOAL
    session.world.stone, session.world.hard_wood = STONE_GOAL, HARD_WOOD_GOAL
    session.world.step(4)
    session.world.coins += 7
    session.world.wood += 15
    session.learner.steps = session.world.turn
    goal = session.world.goal_state()
    session.control({"action": "terrain"})
    assert session.world.goal_state() == goal
    restored = Session(path)
    assert restored.world.goal_state() == goal
    assert restored.world.coins == COIN_GOAL + 7 and restored.world.wood == WOOD_GOAL + 15


@pytest.mark.parametrize("completed", [False, True])
def test_old_single_mission_saves_upgrade_without_double_reward(completed):
    world = World()
    world.turn = 70000
    world.coins = 100 if completed else 20
    world.wood = 10000 if completed else 300
    world.goal_completed_turn = 60000 if completed else None
    state = world.state()
    for key in ("missions_completed", "goal_coins_start", "goal_wood_start"):
        state.pop(key)
    restored = World.restore(state)
    goal = restored.goal_state()
    assert restored.coins == world.coins and restored.wood == world.wood
    assert goal["completed_missions"] == int(completed)
    if completed:
        assert goal["coins"] == goal["wood"] == 0
        assert goal["turns_until_penalty"] == PENALTY_INTERVAL
        assert goal["last_completed_turn"] == 60000
    else:
        assert goal["coins"] == 20 and goal["wood"] == 300
        assert goal["next_penalty_turn"] == world.next_penalty_turn
    restored.step(4)
    assert "objetivo_completo" not in restored.last_reward_components
    assert World.restore(restored.state()).goal_state() == restored.goal_state()
