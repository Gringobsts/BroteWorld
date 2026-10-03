"""Wall-clock pacing is independent of logical turns and rendering."""

import math


class TurnClock:
    def __init__(self):
        self.pending = 0.0

    def budget(self, real_seconds, turns_per_second, limit=256):
        if real_seconds < 0 or not math.isfinite(real_seconds):
            raise ValueError("Intervalo real inválido")
        self.pending += real_seconds * turns_per_second
        return min(limit, int(self.pending + 1e-9))

    def consume(self, executed):
        if executed < 0 or executed > self.pending + 1e-8:
            raise ValueError("No se pueden inventar turnos")
        self.pending = max(0.0, self.pending - executed)

    def reset_pacing(self):
        self.pending = 0.0
