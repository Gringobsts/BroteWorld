"""Sparse procedural terrain. Physics resolves actions; it never selects them."""

import heapq
import math
from collections import OrderedDict

GRASS, PATH, WATER, ROCK, TREE, STUMP, HARD_TREE = range(7)
SOFT_TREE = TREE
DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))
ACTIONS = (*DIRECTIONS, (0, 0), *DIRECTIONS, (0, 0), (0, 0), *DIRECTIONS, (0, 0))
ACTION_NAMES = (
    "norte",
    "este",
    "sur",
    "oeste",
    "quieto",
    "talar al norte",
    "talar al este",
    "talar al sur",
    "talar al oeste",
    "fabricar hacha",
    "fabricar pico",
    "picar al norte",
    "picar al este",
    "picar al sur",
    "picar al oeste",
    "recoger piedra del suelo",
)
OBS_SIZE, ACTION_COUNT = 72, 16
WORLD_SIZE = 1_000_000
TREE_HEALTH, WOOD_PER_TREE, REGROW_TURNS = 3, 3, 100_000
ROCK_HEALTH, STONE_PER_ROCK = 3, 3
TOOL_WOOD_COST, TOOL_STONE_COST = 2, 3
CHUNK_SIZE, MAX_CACHED_CHUNKS = 16, 64
COIN_GOAL, WOOD_GOAL, STONE_GOAL, HARD_WOOD_GOAL = 10_000, 100_000, 100_000, 100_000
PENALTY_INTERVAL = 50_000
SIGNALS = {
    "obstaculo": -0.5,
    "sin_arbol": -0.35,
    "espera": -0.06,
    "golpe": 0.4,
    "arbol_talado": 2.4,
    "avance": 0.02,
    "moneda": 3.0,
    "objetivo_completo": 100.0,
    "plazo_sin_cumplir": -25.0,
    "falta_hacha": -0.5,
    "falta_pico": -0.5,
    "sin_roca": -0.35,
    "materiales_insuficientes": -0.35,
    "herramienta_existente": -0.35,
    "hacha_fabricada": 4.0,
    "pico_fabricado": 4.0,
    "arbol_duro_talado": 2.4,
    "golpe_roca": 0.4,
    "roca_extraida": 2.4,
    "piedra_suelta": 0.8,
    "sin_piedra_suelta": -0.35,
}


def coordinate_hash(x, y, seed):
    n = (x * 374761393 + y * 668265263 + seed * 1442695041) & 0xFFFFFFFF
    n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
    return (n ^ (n >> 16)) & 0xFFFFFFFF


class World:
    def __init__(self, seed=17, width=WORLD_SIZE, height=WORLD_SIZE, trees=True):
        if width < 10 or height < 10:
            raise ValueError("El mundo debe tener al menos 10×10 bloques")
        self.seed, self.width, self.height, self.trees = seed, width, height, trees
        self.x, self.y = width // 2, height // 2
        self.steps = self.turn = self.moves = self.collisions = self.rests = 0
        self.chop_attempts = self.chop_hits = self.trees_felled = self.wood = 0
        self.last_action, self.last_reward, self.last_target = 4, 0.0, None
        self.last_signal = "inicio"
        self.last_reward_components = {}
        self.rules_version = 3
        self.hard_wood = self.stone = 0
        self.wood_spent = self.stone_spent = 0
        self.axe = self.pickaxe = 0
        self.stone_taken = {}
        self.rock_damaged = {}
        self.mined_rocks = {}
        self.mine_attempts = self.mine_hits = self.rocks_broken = 0
        self.goal_stone_start = self.goal_hard_wood_start = 0
        self.coins = 0
        self.coin_taken = {}
        self.goal_start_turn = 0
        self.next_penalty_turn = PENALTY_INTERVAL
        self.goal_completed_turn = None
        self.goal_penalties = 0
        self.missions_completed = 0
        self.goal_coins_start = self.goal_wood_start = 0
        self.visits = {self.key(self.x, self.y): 1}
        self.overrides = {}  # Legacy terrain and future player-built blocks only.
        self.damaged = {}  # Tree health, only for trees that have been hit.
        self.stumps = {}  # key -> scheduled regrowth turn.
        self.regrowth_queue = []
        self.recent = []
        self._chunks = OrderedDict()
        # Placement varies by map seed, independently of the policy and its outcomes.
        for offset in range(64):
            x = width // 2 + (seed * 7 + offset) % 17 - 8
            y = height // 2 + (seed * 11 + offset // 7) % 17 - 8
            if self.walkable(x, y):
                self.x, self.y = x, y
                self.visits = {self.key(x, y): 1}
                break

    def key(self, x, y):
        return y * self.width + x

    def _generate_tile(self, x, y):
        if x in (0, self.width - 1) or y in (0, self.height - 1):
            return WATER
        if (
            abs((x - self.width // 2 + 32) % 64 - 32) <= 1
            or abs((y - self.height // 2 + 32) % 64 - 32) <= 1
        ):
            return PATH
        # Ponds belong to coarse cells; roads remain connected across the whole world.
        cell_x, cell_y = x // 32, y // 32
        pond = coordinate_hash(cell_x, cell_y, self.seed + 99) % 5 == 0
        if pond and ((x % 32 - 12) / 3) ** 2 + ((y % 32 - 12) / 2) ** 2 < 1:
            return WATER
        roll = coordinate_hash(x, y, self.seed) % 100
        if roll < 8:
            return ROCK
        if self.trees and roll < 30:
            return TREE if coordinate_hash(x, y, self.seed + 47) % 2 == 0 else HARD_TREE
        return GRASS

    def base_tile(self, x, y):
        if not (0 <= x < self.width and 0 <= y < self.height):
            return WATER
        key = self.key(x, y)
        if key in self.overrides:
            return self.overrides[key]
        chunk_key = (x // CHUNK_SIZE, y // CHUNK_SIZE)
        chunk = self._chunks.get(chunk_key)
        if chunk is None:
            left, top = chunk_key[0] * CHUNK_SIZE, chunk_key[1] * CHUNK_SIZE
            chunk = bytes(
                self._generate_tile(left + dx, top + dy)
                for dy in range(CHUNK_SIZE)
                for dx in range(CHUNK_SIZE)
            )
            self._chunks[chunk_key] = chunk
            if len(self._chunks) > MAX_CACHED_CHUNKS:
                self._chunks.popitem(last=False)
        else:
            self._chunks.move_to_end(chunk_key)
        return chunk[(y % CHUNK_SIZE) * CHUNK_SIZE + x % CHUNK_SIZE]

    def tile_at(self, x, y):
        if not (0 <= x < self.width and 0 <= y < self.height):
            return WATER
        key = self.key(x, y)
        if key in self.mined_rocks:
            return GRASS
        return STUMP if key in self.stumps else self.base_tile(x, y)

    def tree_health(self, x, y):
        return (
            self.damaged.get(self.key(x, y), TREE_HEALTH)
            if self.tile_at(x, y) in (TREE, HARD_TREE)
            else 0
        )

    def rock_health(self, x, y):
        return (
            self.rock_damaged.get(self.key(x, y), ROCK_HEALTH) if self.tile_at(x, y) == ROCK else 0
        )

    def stone_at(self, x, y):
        return (
            self.tile_at(x, y) in (GRASS, PATH)
            and self.key(x, y) not in self.stone_taken
            and coordinate_hash(x, y, self.seed + 7717) % 17 == 0
        )

    def inventory(self):
        return {
            "coins": self.coins,
            "wood": self.wood,
            "soft_wood": self.wood,
            "hard_wood": self.hard_wood,
            "stone": self.stone,
            "axe": self.axe,
            "pickaxe": self.pickaxe,
        }

    def walkable(self, x, y):
        return self.tile_at(x, y) in (GRASS, PATH, STUMP)

    def coin_at(self, x, y):
        # Random-looking, seeded placement, evaluated only for queried blocks.
        return (
            self.coins - self.goal_coins_start < COIN_GOAL
            and self.tile_at(x, y) in (GRASS, PATH)
            and self.key(x, y) not in self.coin_taken
            and coordinate_hash(x, y, self.seed + 9127 + self.missions_completed * 104729) % 37 == 0
        )

    def _renew_mission(self):
        """Start another collection target without consuming a turn or touching the mind."""
        self.missions_completed += 1
        self.goal_coins_start = self.coins
        self.goal_wood_start = self.wood + self.wood_spent
        self.goal_stone_start = self.stone + self.stone_spent
        self.goal_hard_wood_start = self.hard_wood
        self.goal_start_turn = self.turn
        self.next_penalty_turn = self.turn + PENALTY_INTERVAL
        self.coin_taken = {}

    def goal_state(self):
        return {
            "coins": self.coins - self.goal_coins_start,
            "coin_target": COIN_GOAL,
            "wood": self.wood + self.wood_spent - self.goal_wood_start,
            "soft_wood": self.wood + self.wood_spent - self.goal_wood_start,
            "soft_wood_target": WOOD_GOAL,
            "stone": self.stone + self.stone_spent - self.goal_stone_start,
            "stone_target": STONE_GOAL,
            "hard_wood": self.hard_wood - self.goal_hard_wood_start,
            "hard_wood_target": HARD_WOOD_GOAL,
            "wood_target": WOOD_GOAL,
            "round": self.missions_completed + 1,
            "completed_missions": self.missions_completed,
            "last_completed_turn": self.goal_completed_turn,
            "completed": False,  # The current mission is always active; renewal is immediate.
            "completed_turn": None,
            "started_turn": self.goal_start_turn,
            "next_penalty_turn": self.next_penalty_turn,
            "turns_until_penalty": max(0, self.next_penalty_turn - self.turn),
            "penalties": self.goal_penalties,
            "penalty": SIGNALS["plazo_sin_cumplir"],
            "completion_bonus": SIGNALS["objetivo_completo"],
        }

    def observe(self):
        movement, trees, coins, hard_trees, rocks, stones = [], [], [], [], [], []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                x, y = self.x + dx, self.y + dy
                free = self.walkable(x, y)
                movement.extend(
                    (
                        float(free),
                        1 / math.sqrt(1 + self.visits.get(self.key(x, y), 0)) if free else 0.0,
                    )
                )
                trees.append(self.tree_health(x, y) / TREE_HEALTH)
                coins.append(float(self.coin_at(x, y)))
                hard_trees.append(float(self.tile_at(x, y) == HARD_TREE))
                rocks.append(self.rock_health(x, y) / ROCK_HEALTH)
                stones.append(float(self.stone_at(x, y)))
        mission = [
            min(1, (self.coins - self.goal_coins_start) / COIN_GOAL),
            min(1, (self.wood + self.wood_spent - self.goal_wood_start) / WOOD_GOAL),
            max(0, min(1, (self.next_penalty_turn - self.turn) / PENALTY_INTERVAL)),
        ]
        tools_and_resources = [
            float(bool(self.axe)),
            float(bool(self.pickaxe)),
            min(1, self.wood / TOOL_WOOD_COST),
            min(1, self.stone / TOOL_STONE_COST),
            min(1, (self.stone + self.stone_spent - self.goal_stone_start) / STONE_GOAL),
            min(1, (self.hard_wood - self.goal_hard_wood_start) / HARD_WOOD_GOAL),
        ]
        return (
            movement + trees + coins + mission + hard_trees + rocks + stones + tools_and_resources
        )

    def legacy_observation_v2(self):
        """Reconstruct the old policy's next input before flushing its pending rollout."""
        observation = self.observe()[:39]
        for i, (dy, dx) in enumerate((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)):
            x, y = self.x + dx, self.y + dy
            observation[27 + i] = float(
                self.coins - self.goal_coins_start < 100
                and self.tile_at(x, y) in (GRASS, PATH)
                and self.key(x, y) not in self.coin_taken
                and coordinate_hash(x, y, self.seed + 9127 + self.missions_completed * 104729) % 37
                == 0
            )
        observation[36] = min(1, (self.coins - self.goal_coins_start) / 100)
        observation[37] = min(1, (self.wood - self.goal_wood_start) / 10_000)
        return observation

    def _regrow(self):
        while self.regrowth_queue and self.regrowth_queue[0][0] <= self.turn:
            ready, key = heapq.heappop(self.regrowth_queue)
            if self.stumps.get(key) != ready:
                continue
            if key == self.key(self.x, self.y):
                # Never grow a solid tree through an inhabitant. Retry next turn.
                self.stumps[key] = self.turn + 1
                heapq.heappush(self.regrowth_queue, (self.turn + 1, key))
            else:
                del self.stumps[key]

    def step(self, action):
        if isinstance(action, bool) or action not in range(ACTION_COUNT):
            raise ValueError("Acción desconocida")
        self.turn += 1
        self._regrow()
        moved = collision = hit = felled = False
        self.last_target = None
        if action < 4:
            dx, dy = DIRECTIONS[action]
            moved = self.walkable(self.x + dx, self.y + dy)
            collision = not moved
            if moved:
                self.x += dx
                self.y += dy
                self.moves += 1
                self.last_signal = "avance"
                reward = SIGNALS["avance"] + 0.05 / math.sqrt(
                    1 + self.visits.get(self.key(self.x, self.y), 0)
                )
            else:
                self.collisions += 1
                self.last_signal = "obstaculo"
                reward = SIGNALS["obstaculo"]
        elif action == 4:
            self.rests += 1
            self.last_signal = "espera"
            reward = SIGNALS["espera"]
        elif action < 9:
            self.chop_attempts += 1
            dx, dy = DIRECTIONS[action - 5]
            x, y = self.x + dx, self.y + dy
            self.last_target = [x, y]
            health = self.tree_health(x, y)
            if health and self.tile_at(x, y) == HARD_TREE and not self.axe:
                self.last_signal = "falta_hacha"
                reward = SIGNALS[self.last_signal]
            elif health:
                hit = True
                self.chop_hits += 1
                key = self.key(x, y)
                self.last_signal = "golpe"
                reward = SIGNALS["golpe"]
                if health > 1:
                    self.damaged[key] = health - 1
                else:
                    felled = True
                    self.damaged.pop(key, None)
                    self.stumps[key] = self.turn + REGROW_TURNS
                    heapq.heappush(self.regrowth_queue, (self.turn + REGROW_TURNS, key))
                    self.trees_felled += 1
                    if self.base_tile(x, y) == HARD_TREE:
                        self.hard_wood += WOOD_PER_TREE
                        self.last_signal = "arbol_duro_talado"
                    else:
                        self.wood += WOOD_PER_TREE
                        self.last_signal = "arbol_talado"
                    reward = SIGNALS[self.last_signal]
            else:
                self.last_signal = "sin_arbol"
                reward = SIGNALS["sin_arbol"]
        elif action in (9, 10):
            tool = "axe" if action == 9 else "pickaxe"
            if getattr(self, tool):
                self.last_signal = "herramienta_existente"
            elif self.wood < TOOL_WOOD_COST or self.stone < TOOL_STONE_COST:
                self.last_signal = "materiales_insuficientes"
            else:
                self.wood -= TOOL_WOOD_COST
                self.stone -= TOOL_STONE_COST
                self.wood_spent += TOOL_WOOD_COST
                self.stone_spent += TOOL_STONE_COST
                setattr(self, tool, 1)
                self.last_signal = "hacha_fabricada" if action == 9 else "pico_fabricado"
            reward = SIGNALS[self.last_signal]
        elif action < 15:
            self.mine_attempts += 1
            dx, dy = DIRECTIONS[action - 11]
            x, y = self.x + dx, self.y + dy
            self.last_target = [x, y]
            health = self.rock_health(x, y)
            if not health:
                self.last_signal = "sin_roca"
            elif not self.pickaxe:
                self.last_signal = "falta_pico"
            else:
                self.mine_hits += 1
                key = self.key(x, y)
                if health > 1:
                    self.rock_damaged[key] = health - 1
                    self.last_signal = "golpe_roca"
                else:
                    self.rock_damaged.pop(key, None)
                    self.mined_rocks[key] = True
                    self.rocks_broken += 1
                    self.stone += STONE_PER_ROCK
                    self.last_signal = "roca_extraida"
            reward = SIGNALS[self.last_signal]
        else:
            self.last_target = [self.x, self.y]
            if self.stone_at(self.x, self.y):
                self.stone += 1
                self.stone_taken[self.key(self.x, self.y)] = True
                self.last_signal = "piedra_suelta"
            else:
                self.last_signal = "sin_piedra_suelta"
            reward = SIGNALS[self.last_signal]
        key = self.key(self.x, self.y)
        self.last_reward_components = {self.last_signal: reward}
        if moved and self.coin_at(self.x, self.y):
            self.coins += 1
            self.coin_taken[key] = True
            self.last_signal = "moneda"
            self.last_reward_components["moneda"] = SIGNALS["moneda"]
        # Completion wins if every threshold is met on the exact penalty boundary.
        if (
            self.coins - self.goal_coins_start >= COIN_GOAL
            and self.wood + self.wood_spent - self.goal_wood_start >= WOOD_GOAL
            and self.stone + self.stone_spent - self.goal_stone_start >= STONE_GOAL
            and self.hard_wood - self.goal_hard_wood_start >= HARD_WOOD_GOAL
        ):
            self.goal_completed_turn = self.turn
            self.last_signal = "objetivo_completo"
            self.last_reward_components["objetivo_completo"] = SIGNALS["objetivo_completo"]
            self._renew_mission()
        elif self.turn >= self.next_penalty_turn:
            count = (self.turn - self.next_penalty_turn) // PENALTY_INTERVAL + 1
            self.goal_penalties += count
            self.next_penalty_turn += count * PENALTY_INTERVAL
            self.last_signal = "plazo_sin_cumplir"
            self.last_reward_components["plazo_sin_cumplir"] = count * SIGNALS["plazo_sin_cumplir"]
        reward = sum(self.last_reward_components.values())
        self.visits[key] = self.visits.get(key, 0) + 1
        self.steps += 1
        self.last_action, self.last_reward = action, reward
        self.recent.append(
            {
                "moved": moved,
                "collision": collision,
                "reward": reward,
                "chop": 5 <= action < 9,
                "mine": 11 <= action < 15,
                "hit": hit,
                "felled": felled,
            }
        )
        self.recent = self.recent[-256:]
        return reward

    def metrics(self):
        n = max(1, len(self.recent))
        attempts = sum(r.get("chop", False) for r in self.recent)
        return {
            "steps": self.steps,
            "moves": self.moves,
            "collisions": self.collisions,
            "visited": len(self.visits),
            "total_blocks": self.width * self.height,
            "coverage": len(self.visits) / (self.width * self.height),
            "move_rate": sum(r["moved"] for r in self.recent) / n,
            "collision_rate": sum(r["collision"] for r in self.recent) / n,
            "mean_reward": sum(r["reward"] for r in self.recent) / n,
            "chop_success_rate": sum(r.get("hit", False) for r in self.recent) / max(1, attempts),
            "chop_attempts": self.chop_attempts,
            "chop_hits": self.chop_hits,
            "trees_felled": self.trees_felled,
            "wood": self.wood,
            "coins": self.coins,
            "soft_wood": self.wood,
            "hard_wood": self.hard_wood,
            "stone": self.stone,
            "rocks_broken": self.rocks_broken,
            "mine_success_rate": self.mine_hits / max(1, self.mine_attempts),
            "goal_penalties": self.goal_penalties,
            "cached_chunks": len(self._chunks),
            "changed_blocks": len(self.overrides)
            + len(self.damaged)
            + len(self.stumps)
            + len(self.mined_rocks)
            + len(self.rock_damaged),
        }

    def viewport(self, width=24, height=18):
        width, height = min(width, self.width), min(height, self.height)
        left = max(0, min(self.width - width, self.x - width // 2))
        top = max(0, min(self.height - height, self.y - height // 2))
        return {
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "viewport": {"x": left, "y": top, "width": width, "height": height},
            "tiles": [
                [self.tile_at(x, y) for x in range(left, left + width)]
                for y in range(top, top + height)
            ],
            "visits": [
                [self.visits.get(self.key(x, y), 0) for x in range(left, left + width)]
                for y in range(top, top + height)
            ],
            "tree_health": [
                [self.tree_health(x, y) for x in range(left, left + width)]
                for y in range(top, top + height)
            ],
            "rock_health": [
                [self.rock_health(x, y) for x in range(left, left + width)]
                for y in range(top, top + height)
            ],
            "stones": [
                [int(self.stone_at(x, y)) for x in range(left, left + width)]
                for y in range(top, top + height)
            ],
            "regrow_turns": REGROW_TURNS,
            "coins": [
                [int(self.coin_at(x, y)) for x in range(left, left + width)]
                for y in range(top, top + height)
            ],
        }

    def state(self):
        return {k: v for k, v in vars(self).items() if k != "_chunks"}

    @classmethod
    def restore(cls, state, *, upgrade=True):
        world = cls(state["seed"], state["width"], state["height"], state["trees"])
        for key, value in state.items():
            if key != "_chunks":
                setattr(world, key, value)
        # An older completed save has already paid its bonus. Resume with a fresh mission,
        # without paying twice or charging penalties for time before this upgrade.
        if upgrade and "missions_completed" not in state and world.goal_completed_turn is not None:
            world._renew_mission()
        if upgrade and "rules_version" not in state:
            # The previous world exposed a starter axe. Keep it, plus all accrued wood as soft wood.
            world.axe = 1
            world.rules_version = 3
        heapq.heapify(world.regrowth_queue)
        return world

    @classmethod
    def from_legacy(cls, state, global_turn):
        world = cls(state["seed"])
        ox = world.width // 2 - state["width"] // 2
        oy = world.height // 2 - state["height"] // 2
        world.x, world.y = ox + state["x"], oy + state["y"]
        # Preserve the explored interior, open its former water border into the wider terrain.
        for y in range(1, state["height"] - 1):
            for x in range(1, state["width"] - 1):
                tile = state["tiles"][y][x]
                world.overrides[world.key(ox + x, oy + y)] = tile
        world.visits = {
            world.key(ox + x, oy + y): count
            for y, row in enumerate(state["visits"])
            for x, count in enumerate(row)
            if count
        }
        for key in (
            "steps",
            "moves",
            "collisions",
            "rests",
            "last_action",
            "last_reward",
            "recent",
        ):
            setattr(world, key, state[key])
        world.turn = global_turn
        world.goal_start_turn = global_turn
        world.next_penalty_turn = global_turn + PENALTY_INTERVAL
        # New trees are terrain, not directions or examples supplied to the learner.
        for key, tile in list(world.overrides.items()):
            x, y = key % world.width, key // world.width
            if (
                tile == GRASS
                and coordinate_hash(x, y, world.seed) % 100 < 22
                and (x, y) != (world.x, world.y)
            ):
                world.overrides[key] = TREE
        return world
