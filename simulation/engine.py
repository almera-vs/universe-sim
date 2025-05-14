"""
Simulation engine handling tick-based updates.
"""
from world.galaxy import Galaxy

class SimulationEngine:
    def __init__(self, seed, ticks=5000, name=None):
        self.seed = seed
        self.galaxy = Galaxy(name or f"Galaxy_{seed}", seed)
        self.max_ticks = ticks
        self.tick_count = 0

    def run(self):
        for _ in range(self.max_ticks):
            self.tick()
    
    def tick(self):
        self.galaxy.tick()
        self.tick_count += 1
        self.log_tick()

    def log_tick(self):
        print(f"[Simulation] Tick {self.tick_count}")

        