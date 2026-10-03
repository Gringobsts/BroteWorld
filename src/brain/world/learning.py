"""Continuous on-policy PPO: random initialization, no scripts or action masks."""

import copy
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Categorical

from brain.world.environment import ACTION_COUNT, OBS_SIZE, World


@dataclass(frozen=True)
class LearnerConfig:
    seed: int = 17
    hidden: int = 64
    rollout: int = 64
    epochs: int = 4
    learning_rate: float = 0.001
    gamma: float = 0.95
    gae_lambda: float = 0.95
    entropy: float = 0.025
    clip: float = 0.2
    exploration: float = 0.1
    temperature: float = 3.0


class Policy(nn.Module):
    def __init__(self, hidden=64, observation_size=OBS_SIZE, action_count=ACTION_COUNT):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(observation_size, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.actor = nn.Linear(hidden, action_count)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, observations):
        features = self.body(observations)
        return self.actor(features), self.critic(features).squeeze(-1)


class Learner:
    def __init__(self, config=None, *, observation_size=OBS_SIZE, action_count=ACTION_COUNT):
        self.config = config = config or LearnerConfig()
        torch.set_num_threads(1)
        with torch.random.fork_rng():
            torch.manual_seed(config.seed)
            self.policy = Policy(config.hidden, observation_size, action_count)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        self.rng = torch.Generator().manual_seed(config.seed + 100)
        self.buffer = []
        self.steps = self.updates = 0
        self.last_loss = self.last_entropy = 0.0
        self.history = []
        self.migration = None

    @torch.no_grad()
    def choose(self, observation, rng=None):
        logits, value = self.policy(torch.tensor(observation, dtype=torch.float32))
        probabilities = self.distribution(logits).probs
        action = torch.multinomial(probabilities, 1, generator=self.rng if rng is None else rng)
        return int(action.item()), float(probabilities[action].log().item()), float(value)

    def distribution(self, logits):
        # The same mixture must be used for sampling AND PPO likelihood ratios.
        # V1 rollouts were collected without this exploration floor.
        epsilon = self.config.exploration if logits.shape[-1] >= 9 else 0.0
        temperature = self.config.temperature if logits.shape[-1] >= 9 else 1.0
        probabilities = (1 - epsilon) * (logits / temperature).softmax(-1) + epsilon / logits.shape[
            -1
        ]
        return Categorical(probs=probabilities)

    def live_step(self, world):
        observation = world.observe()
        action, log_prob, value = self.choose(observation)
        reward = world.step(action)
        self.buffer.append(
            {
                "obs": observation,
                "action": action,
                "reward": reward,
                "log_prob": log_prob,
                "value": value,
            }
        )
        self.steps += 1
        if len(self.buffer) >= self.config.rollout:
            self.update(world.observe())
            self.history.append({"step": self.steps, **world.metrics()})
            self.history = self.history[-160:]
        return action

    def update(self, next_observation):
        if not self.buffer:
            return
        with torch.no_grad():
            _, bootstrap = self.policy(torch.tensor(next_observation, dtype=torch.float32))
        next_value, advantage = float(bootstrap), 0.0
        advantages, returns = [], []
        for row in reversed(self.buffer):
            delta = row["reward"] + self.config.gamma * next_value - row["value"]
            advantage = delta + self.config.gamma * self.config.gae_lambda * advantage
            advantages.append(advantage)
            returns.append(advantage + row["value"])
            next_value = row["value"]
        observations = torch.tensor([r["obs"] for r in self.buffer], dtype=torch.float32)
        actions = torch.tensor([r["action"] for r in self.buffer])
        old_log_probs = torch.tensor([r["log_prob"] for r in self.buffer])
        advantages = torch.tensor(advantages[::-1], dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        returns = torch.tensor(returns[::-1], dtype=torch.float32)
        for _ in range(self.config.epochs):
            logits, values = self.policy(observations)
            distribution = self.distribution(logits)
            log_probs = distribution.log_prob(actions)
            # Stop an update if the new policy drifts too far from the rollout policy.
            if float((old_log_probs - log_probs).mean().detach()) > 0.04:
                break
            ratio = (log_probs - old_log_probs).exp()
            unclipped = ratio * advantages
            clipped = ratio.clamp(1 - self.config.clip, 1 + self.config.clip) * advantages
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = (returns - values).square().mean()
            entropy = distribution.entropy().mean()
            loss = policy_loss + 0.5 * value_loss - self.config.entropy * entropy
            if not torch.isfinite(loss):
                raise FloatingPointError("Actualización no finita")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5, error_if_nonfinite=True)
            self.optimizer.step()
            self.last_loss, self.last_entropy = float(loss.detach()), float(entropy.detach())
        self.updates += 1
        self.buffer.clear()

    def save(self, destination: Path, world: World):
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "brain-world-v3",
            "config": asdict(self.config),
            "weights": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rng": self.rng.get_state(),
            "buffer": self.buffer,
            "steps": self.steps,
            "updates": self.updates,
            "history": self.history,
            "loss": self.last_loss,
            "entropy": self.last_entropy,
            "world": world.state(),
            "migration": self.migration,
        }
        fd, temporary = tempfile.mkstemp(prefix=".brain-world-", dir=destination.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                torch.save(payload, stream)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, source: Path, *, backup_legacy=False):
        payload = torch.load(source, weights_only=True, map_location="cpu")
        if payload["format"] not in ("brain-world-v1", "brain-world-v2", "brain-world-v3"):
            raise ValueError("Formato de mundo incompatible")
        legacy = payload["format"] == "brain-world-v1"
        upgrading = payload["format"] != "brain-world-v3"
        if upgrading and backup_legacy:
            version = "v1" if legacy else "v2"
            backup = source.with_name(source.stem + f".{version}-backup.pt")
            try:
                with backup.open("xb") as stream:
                    stream.write(source.read_bytes())
            except FileExistsError:
                pass
        learner = cls(
            LearnerConfig(**payload["config"]),
            observation_size=payload["weights"]["body.0.weight"].shape[1],
            action_count=payload["weights"]["actor.weight"].shape[0],
        )
        learner.policy.load_state_dict(payload["weights"])
        learner.optimizer.load_state_dict(payload["optimizer"])
        learner.rng.set_state(payload["rng"])
        for key in ("buffer", "steps", "updates", "history"):
            setattr(learner, key, payload[key])
        learner.last_loss, learner.last_entropy = payload["loss"], payload["entropy"]
        learner.migration = payload.get("migration")
        if legacy:
            old = payload["world"]
            pending = len(learner.buffer)
            observation = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    x, y = old["x"] + dx, old["y"] + dy
                    free = (
                        0 <= x < old["width"] and 0 <= y < old["height"] and old["tiles"][y][x] < 2
                    )
                    observation.extend(
                        (float(free), (1 + old["visits"][y][x]) ** -0.5 if free else 0.0)
                    )
            # Finish v1 experience with the v1 policy, before changing the action space.
            learner.update(observation)
            learner.expand_policy()
            learner.migration = {
                "from": "brain-world-v1",
                "turn": learner.steps,
                "old_experiences_completed": pending,
                "optimizer_moments_preserved": True,
            }
            world = World.from_legacy(old, learner.steps)
        elif upgrading:
            old_world = World.restore(payload["world"], upgrade=False)
            pending = len(learner.buffer)
            learner.update(old_world.legacy_observation_v2())
            learner.expand_policy()
            learner.migration = {
                "from": payload["format"],
                "turn": learner.steps,
                "old_experiences_completed": pending,
                "optimizer_moments_preserved": True,
                "previous": learner.migration,
            }
            world = World.restore(payload["world"])
        else:
            world = World.restore(payload["world"])
        return learner, world

    def expand_policy(self):
        """Keep all existing weights and Adam moments; initialize only added dimensions."""
        previous = self.policy
        old_optimizer = self.optimizer
        with torch.random.fork_rng():
            torch.manual_seed(self.config.seed)
            expanded = Policy(self.config.hidden)
        old_parameters = dict(previous.named_parameters())
        with torch.no_grad():
            for name, new in expanded.named_parameters():
                old = old_parameters[name]
                if new.shape != old.shape:
                    new.fill_(-1.0 if name == "actor.bias" else 0.0)
                new[tuple(slice(0, n) for n in old.shape)].copy_(old)
        self.policy = expanded
        group = {
            key: value for key, value in old_optimizer.param_groups[0].items() if key != "params"
        }
        self.optimizer = torch.optim.Adam([{"params": list(expanded.parameters()), **group}])
        for name, new in expanded.named_parameters():
            old = old_parameters[name]
            if old not in old_optimizer.state:
                continue
            state = {}
            for key, value in old_optimizer.state[old].items():
                if torch.is_tensor(value) and value.shape == old.shape and new.shape != old.shape:
                    moment = torch.zeros_like(new)
                    moment[tuple(slice(0, n) for n in old.shape)] = value
                    state[key] = moment
                else:
                    state[key] = copy.deepcopy(value)
            self.optimizer.state[new] = state


@torch.no_grad()
def benchmark(learner, seeds=(1001, 1002, 1003, 1004, 1005), steps=512, *, trees=False):
    """No gradients, no optimizer calls, separate worlds and random generator."""
    results = []
    for seed in seeds:
        world = World(seed, trees=trees)
        rng = torch.Generator().manual_seed(seed)
        for _ in range(steps):
            action, _, _ = learner.choose(world.observe(), rng)
            world.step(action)
        results.append(
            {
                "seed": seed,
                "move_rate": world.moves / steps,
                "collision_rate": world.collisions / steps,
                "coverage": world.metrics()["coverage"],
                "visited": len(world.visits),
                "wood": world.wood,
                "trees_felled": world.trees_felled,
                "coins": world.coins,
                "movement_success_rate": world.moves / max(1, world.moves + world.collisions),
                "chop_success_rate": world.chop_hits / max(1, world.chop_attempts),
                "invalid_chop_rate": (world.chop_attempts - world.chop_hits) / steps,
            }
        )
    return {
        "maps": results,
        **{
            key: sum(r[key] for r in results) / len(results)
            for key in (
                "move_rate",
                "collision_rate",
                "coverage",
                "visited",
                "wood",
                "trees_felled",
                "coins",
                "movement_success_rate",
                "chop_success_rate",
                "invalid_chop_rate",
            )
        },
    }
