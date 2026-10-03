"""Local observation window. The training loop belongs to the server, not the browser."""

import argparse
import copy
import json
import math
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from brain.world.clock import TurnClock
from brain.world.environment import ACTION_NAMES, World
from brain.world.learning import Learner, LearnerConfig


class Session:
    def __init__(self, checkpoint, seed=17, speed=4):
        self.checkpoint = Path(checkpoint)
        if self.checkpoint.exists():
            self.learner, self.world = Learner.load(self.checkpoint, backup_legacy=True)
        else:
            self.learner, self.world = Learner(LearnerConfig(seed=seed)), World(seed)
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.running, self.speed = True, speed
        self.effective_speed = 0.0
        self.clock = TurnClock()
        self.pacing_at = time.monotonic()
        self.saved_at = None
        self.error = None
        self.events = [
            {
                "text": "Nació un nuevo explorador."
                if not self.learner.steps
                else "Despertó conservando lo aprendido.",
                "step": self.learner.steps,
            }
        ]
        self.worker = threading.Thread(target=self.loop, daemon=True)

    def advance_turn(self):
        """One living agent in stage 1: exactly one attempted action per world turn."""
        felled_before = self.world.trees_felled
        self.learner.live_step(self.world)
        for signal_name, message in (
            ("moneda", f"Recogió una moneda. Acumuladas: {self.world.coins}."),
            ("piedra_suelta", "Recogió una piedra del suelo."),
            ("hacha_fabricada", "Fabricó un hacha con 2 de madera blanda y 3 piedras."),
            ("pico_fabricado", "Fabricó un pico con 2 de madera blanda y 3 piedras."),
            ("roca_extraida", "Extrajo una roca grande: 3 de piedra."),
            (
                "objetivo_completo",
                f"Misión {self.world.missions_completed} completada: +100. "
                f"Comienza la misión {self.world.missions_completed + 1}.",
            ),
            ("plazo_sin_cumplir", "50.000 turnos sin completar las cuatro metas: señal de −25."),
        ):
            if signal_name in self.world.last_reward_components:
                self.events.insert(0, {"text": message, "step": self.learner.steps})
                self.events = self.events[:6]
        if self.world.trees_felled > felled_before:
            wood_type = (
                "dura" if "arbol_duro_talado" in self.world.last_reward_components else "blanda"
            )
            self.events.insert(
                0,
                {
                    "text": f"Taló un árbol y recolectó 3 de madera {wood_type}.",
                    "step": self.learner.steps,
                },
            )
            self.events = self.events[:6]
        if self.learner.steps % 256 == 0:
            self.events.insert(
                0,
                {
                    "text": f"{self.world.metrics()['visited']} lugares descubiertos; sigue aprendiendo.",
                    "step": self.learner.steps,
                },
            )
            self.events = self.events[:6]

    def save(self):
        with self.lock:
            self.learner.save(self.checkpoint, self.world)
            self.saved_at = time.time()

    def loop(self):
        last_save = time.monotonic()
        measured_at, measured_turn = last_save, self.learner.steps
        self.pacing_at = last_save
        try:
            while not self.stop.is_set():
                with self.lock:
                    rate = self.speed
                    now = time.monotonic()
                    batch = self.clock.budget(now - self.pacing_at, rate) if self.running else 0
                    self.pacing_at = now
                for _ in range(batch):
                    with self.lock:
                        if not self.running or self.stop.is_set() or self.speed != rate:
                            break
                        self.advance_turn()
                        self.clock.consume(1)
                elapsed = time.monotonic() - measured_at
                if elapsed >= 0.5:
                    self.effective_speed = (self.learner.steps - measured_turn) / elapsed
                    measured_at, measured_turn = time.monotonic(), self.learner.steps
                if time.monotonic() - last_save >= 20:
                    self.save()
                    last_save = time.monotonic()
                delay = min(0.05, max(0, (1 - self.clock.pending) / rate)) if self.running else 0.1
                self.wake.wait(delay)
                self.wake.clear()
        except Exception as exc:
            with self.lock:
                self.error, self.running = str(exc), False
        finally:
            try:
                self.save()
            except Exception as exc:
                self.error = f"No se pudo guardar: {exc}"

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(
                {
                    "world": self.world.viewport(),
                    "agent": {
                        "name": "Brote",
                        "x": self.world.x,
                        "y": self.world.y,
                        "action": self.world.last_action,
                        "action_name": ACTION_NAMES[self.world.last_action],
                        "reward": self.world.last_reward,
                        "sensors": self.world.observe(),
                        "inventory": self.world.inventory(),
                        "target": self.world.last_target,
                        "signal": self.world.last_signal,
                        "reward_components": self.world.last_reward_components,
                    },
                    "metrics": self.world.metrics(),
                    "goal": self.world.goal_state(),
                    "learning": {
                        "steps": self.learner.steps,
                        "updates": self.learner.updates,
                        "loss": self.learner.last_loss,
                        "entropy": self.learner.last_entropy,
                        "buffer": len(self.learner.buffer),
                        "rollout": self.learner.config.rollout,
                        "parameters": sum(p.numel() for p in self.learner.policy.parameters()),
                        "connections": sum(
                            p.numel()
                            for name, p in self.learner.policy.named_parameters()
                            if name.endswith("weight")
                        ),
                        "model_bytes": sum(
                            p.numel() * p.element_size() for p in self.learner.policy.parameters()
                        ),
                        "checkpoint_bytes": self.checkpoint.stat().st_size
                        if self.checkpoint.exists()
                        else 0,
                        "inputs": self.learner.policy.body[0].in_features,
                        "actions": self.learner.policy.actor.out_features,
                        "history": self.learner.history,
                    },
                    "running": self.running,
                    "speed": self.speed,
                    "effective_speed": self.effective_speed if self.running else 0,
                    "saved_at": self.saved_at,
                    "clock": {
                        "turn": self.learner.steps,
                        "seconds_per_turn": 1,
                        "elapsed_seconds": self.learner.steps,
                        "actions_per_agent": 1,
                    },
                    "events": self.events,
                    "error": self.error,
                }
            )

    def control(self, payload):
        with self.lock:
            action = payload.get("action")
            if action == "speed":
                speed = payload.get("value")
                if (
                    isinstance(speed, bool)
                    or not isinstance(speed, (float, int))
                    or not math.isfinite(speed)
                    or not 0.01 <= speed <= 10000
                ):
                    raise ValueError("Velocidad inválida")
                self.speed = speed
                self.clock.reset_pacing()
                self.pacing_at = time.monotonic()
            elif action == "running":
                if not isinstance(payload.get("value"), bool):
                    raise ValueError("Estado inválido")
                if self.error:
                    raise ValueError(self.error)
                self.running = payload["value"]
                self.clock.reset_pacing()
                self.pacing_at = time.monotonic()
            elif action == "save":
                self.save()
            elif action == "step":
                if self.running:
                    raise ValueError("Pausá el mundo antes de avanzar un turno")
                self.advance_turn()
            elif action == "terrain":
                # Finish the current experience before changing the environment, keep the brain.
                self.learner.update(self.world.observe())
                previous = self.world
                self.world = World(previous.seed + 1)
                self.world.wood = previous.wood
                self.world.coins = previous.coins
                for field in (
                    "goal_start_turn",
                    "next_penalty_turn",
                    "goal_completed_turn",
                    "goal_penalties",
                    "missions_completed",
                    "goal_coins_start",
                    "goal_wood_start",
                    "goal_stone_start",
                    "goal_hard_wood_start",
                    "hard_wood",
                    "stone",
                    "wood_spent",
                    "stone_spent",
                    "axe",
                    "pickaxe",
                ):
                    setattr(self.world, field, getattr(previous, field))
                self.world.turn = self.learner.steps
                self.events.insert(
                    0,
                    {"text": "Nuevo terreno. Conserva sus conexiones.", "step": self.learner.steps},
                )
                self.events = self.events[:6]
                self.save()
            else:
                raise ValueError("Acción de control desconocida")
            self.wake.set()
        return self.snapshot()

    def close(self):
        self.stop.set()
        self.wake.set()
        if self.worker.is_alive():
            self.worker.join(timeout=10)
        self.save()


def make_handler(session):
    static = Path(__file__).with_name("static")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def respond(self, code, payload, content_type="application/json; charset=utf-8"):
            body = (
                json.dumps(payload, ensure_ascii=False).encode()
                if isinstance(payload, dict)
                else payload
            )
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def allowed_host(self):
            return self.headers.get("Host") in (
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            )

        def do_GET(self):
            if not self.allowed_host():
                return self.respond(403, {"error": "Host no autorizado"})
            if self.path == "/api/state":
                return self.respond(200, session.snapshot())
            files = {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                "/style.css": ("style.css", "text/css; charset=utf-8"),
            }
            if self.path not in files:
                return self.respond(404, {"error": "No encontrado"})
            name, kind = files[self.path]
            self.respond(200, (static / name).read_bytes(), kind)

        def do_POST(self):
            origin = self.headers.get("Origin")
            allowed = (
                f"http://127.0.0.1:{self.server.server_port}",
                f"http://localhost:{self.server.server_port}",
            )
            if not self.allowed_host() or (origin and origin not in allowed):
                return self.respond(403, {"error": "Origen no autorizado"})
            if self.path != "/api/control":
                return self.respond(404, {"error": "No encontrado"})
            if self.headers.get_content_type() != "application/json":
                return self.respond(415, {"error": "Se requiere JSON"})
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 4096:
                    raise ValueError("Solicitud inválida")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Solicitud inválida")
                self.respond(200, session.control(payload))
            except (ValueError, json.JSONDecodeError) as exc:
                self.respond(400, {"error": str(exc)})

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Brain · Un pequeño mundo que aprende")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--speed", type=float, default=4, help="Turnos/s reales: 0.01 a 10000")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/world/live.pt"))
    parser.add_argument("--paused", action="store_true", help="Iniciar sin avanzar turnos")
    args = parser.parse_args()
    if not math.isfinite(args.speed) or not 0.01 <= args.speed <= 10000:
        parser.error("La velocidad debe estar entre 0.01 y 10000 turnos/s")
    session = Session(args.checkpoint, args.seed, args.speed)
    session.running = not args.paused
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(session))
    server.daemon_threads = True
    session.worker.start()
    print(
        f"Brain abierto en http://127.0.0.1:{args.port} · entrenamiento continuo activo", flush=True
    )

    def terminate(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        session.close()
        server.server_close()


if __name__ == "__main__":
    main()
