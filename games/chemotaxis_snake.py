"""
chemotaxis_snake.py
Biological Chemotaxis and Klinotaxis Foraging Arena.
The 302-neuron C. elegans navigates a continuous 2D petri dish towards chemical attractants
(food sources) using genuine pirouette mechanisms (Pierce-Shimomura et al. 1999).
"""

import math
import random
from typing import Dict, List, Tuple, Any, Optional
try:
    from ..connectome.biophysical_engine import BiophysicalConnectomeEngine
    from ..connectome.sensorimotor_mapping import ChemotaxisSensoryEncoder, ChemotaxisMotorDecoder
except (ImportError, ValueError):
    from connectome.biophysical_engine import BiophysicalConnectomeEngine
    from connectome.sensorimotor_mapping import ChemotaxisSensoryEncoder, ChemotaxisMotorDecoder

class ChemotaxisSnakeArena:
    def __init__(self, engine: Optional[BiophysicalConnectomeEngine] = None, arena_radius: float = 220.0):
        self.arena_radius = arena_radius
        self.engine = engine or BiophysicalConnectomeEngine()
        self.encoder = ChemotaxisSensoryEncoder(self.engine)
        self.decoder = ChemotaxisMotorDecoder(self.engine)
        
        self.reset()

    def reset(self):
        self.worm_x = 0.0
        self.worm_y = 0.0
        self.heading = random.uniform(0, 2 * math.pi) # radians
        
        # Spawn first food source
        self.food_x = 0.0
        self.food_y = 0.0
        self.score = 0
        self.frames = 0
        self.pirouette_count = 0
        self._spawn_food()
        
        # 16-segment undulating body
        self.body_length = 16
        self.body = [{"x": -i * 5.0, "y": 0.0} for i in range(self.body_length)]
        
        self.engine.reset_state()
        self.encoder.prev_concentration = self._calc_concentration(self.worm_x, self.worm_y)
        
        return self._get_state()

    def _spawn_food(self):
        angle = random.uniform(0, 2 * math.pi)
        r = random.uniform(70.0, self.arena_radius - 40.0)
        self.food_x = r * math.cos(angle)
        self.food_y = r * math.sin(angle)

    def _calc_concentration(self, x: float, y: float) -> float:
        """Inverse-square / Gaussian odor concentration field."""
        dist = math.sqrt((x - self.food_x)**2 + (y - self.food_y)**2)
        # Peak concentration 1.0 at food source, falling off with radius sigma = 80 um
        return float(math.exp(-(dist**2) / (2.0 * (85.0**2))))

    def step(self) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self.frames += 1
        
        # 1. Sense Chemical Concentration & Wall Proximity
        conc = self._calc_concentration(self.worm_x, self.worm_y)
        center_dist = math.sqrt(self.worm_x**2 + self.worm_y**2)
        wall_dist = max(0.0, self.arena_radius - center_dist)
        
        sensory_inputs = self.encoder.encode_and_inject(conc, wall_dist)
        
        # 2. Step Biophysical Engine (4 sub-steps)
        for _ in range(4):
            self.engine.step()
            
        # 3. Decode Movement (speed, turn rate)
        speed, turn_rate, motor_metrics = self.decoder.decode_movement()
        if motor_metrics.get("is_pirouette", False):
            self.pirouette_count += 1
            
        # 4. Turn & Move Head
        self.heading += turn_rate
        dx = speed * math.cos(self.heading)
        dy = speed * math.sin(self.heading)
        self.worm_x += dx
        self.worm_y += dy
        
        # 5. Wall bounce / boundary containment
        if center_dist > self.arena_radius - 5.0:
            # Rebound towards center
            inward_angle = math.atan2(-self.worm_y, -self.worm_x)
            self.heading = inward_angle + random.uniform(-0.4, 0.4)
            self.worm_x = (self.arena_radius - 6.0) * math.cos(math.atan2(self.worm_y, self.worm_x))
            self.worm_y = (self.arena_radius - 6.0) * math.sin(math.atan2(self.worm_y, self.worm_x))

        # 6. Update Body Segments (undulating wave propagation)
        self.body[0] = {"x": self.worm_x, "y": self.worm_y}
        for i in range(1, self.body_length):
            prev = self.body[i - 1]
            curr = self.body[i]
            # Follow previous segment with kinematic constraint
            seg_dx = prev["x"] - curr["x"]
            seg_dy = prev["y"] - curr["y"]
            seg_dist = math.sqrt(seg_dx**2 + seg_dy**2)
            desired_dist = 5.0
            if seg_dist > 0.001:
                curr["x"] = prev["x"] - (seg_dx / seg_dist) * desired_dist
                curr["y"] = prev["y"] - (seg_dy / seg_dist) * desired_dist

        # 7. Food Consumption Check
        dist_to_food = math.sqrt((self.worm_x - self.food_x)**2 + (self.worm_y - self.food_y)**2)
        reward = conc
        eaten = False
        if dist_to_food < 15.0:
            self.score += 1
            reward += 10.0
            eaten = True
            self._spawn_food()

        state = self._get_state()
        info = {
            "sensory_inputs": sensory_inputs,
            "motor_metrics": motor_metrics,
            "conc": conc,
            "food_eaten": eaten,
            "dist_to_food": round(dist_to_food, 1)
        }
        
        return state, reward, False, info

    def _get_state(self) -> Dict[str, Any]:
        return {
            "worm_x": round(self.worm_x, 1),
            "worm_y": round(self.worm_y, 1),
            "heading": round(self.heading, 3),
            "food_x": round(self.food_x, 1),
            "food_y": round(self.food_y, 1),
            "score": self.score,
            "frames": self.frames,
            "pirouettes": self.pirouette_count,
            "arena_radius": self.arena_radius,
            "body": [{"x": round(s["x"], 1), "y": round(s["y"], 1)} for s in self.body]
        }
