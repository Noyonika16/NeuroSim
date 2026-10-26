"""
lunar_worm.py
Lunar Nematode: Cybernetic soft-touch descent and landing.
Sensory mechanoreception (PLM posterior touch / PVD proprioceptors) measures descent velocity
and altitude radar; motor neurons fire deceleration thrusters for a gentle touchdown.
"""

import math
import random
from typing import Dict, List, Tuple, Any, Optional
try:
    from ..connectome.biophysical_engine import BiophysicalConnectomeEngine
except (ImportError, ValueError):
    from connectome.biophysical_engine import BiophysicalConnectomeEngine

class LunarWormLander:
    def __init__(self, engine: Optional[BiophysicalConnectomeEngine] = None, screen_w: int = 400, screen_h: int = 500):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.engine = engine or BiophysicalConnectomeEngine()
        self.reset()

    def reset(self):
        self.x = self.screen_w * 0.5
        self.y = 50.0
        self.vx = random.uniform(-1.0, 1.0)
        self.vy = 0.5
        self.gravity = 0.12
        self.thruster_power = 0.35
        self.fuel = 100.0
        self.is_landed = False
        self.is_crashed = False
        self.score = 0
        self.frames = 0
        self.engine.reset_state()
        return self._get_state()

    def step(self) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.is_landed or self.is_crashed:
            return self._get_state(), 0.0, True, {"done": True}

        self.frames += 1
        ground_y = self.screen_h - 40.0
        altitude = max(0.0, ground_y - self.y)
        
        # Sensory Encoding:
        # Mechanoreceptors PLM / PVD sense altitude proximity and downward speed
        currents: Dict[str, float] = {}
        if altitude < 200.0:
            radar = 35.0 * ((200.0 - altitude) / 200.0)
            currents["PLML"] = radar
            currents["PLMR"] = radar
            
        if self.vy > 1.0:
            speed_sense = min(30.0, self.vy * 8.0)
            currents["PVDL"] = speed_sense
            currents["PVDR"] = speed_sense

        # Horizontal drift: ALM / AVM
        if abs(self.vx) > 0.5:
            currents["ALML"] = 15.0 if self.vx > 0 else 0.0
            currents["ALMR"] = 15.0 if self.vx < 0 else 0.0

        self.engine.set_sensory_inputs_bulk(currents)
        
        # Step connectome
        for _ in range(4):
            self.engine.step()
            
        readouts = self.engine.get_motor_readouts()
        v_avb = readouts["v_avb"]
        db_act = readouts["db_activity"]
        
        # Thruster activation: when forward drive and dorsal motor pool depolarize
        thrust_active = False
        if (db_act > -54.0 or v_avb > -48.0) and self.fuel > 0.0:
            thrust_active = True
            self.vy -= self.thruster_power
            self.fuel = max(0.0, self.fuel - 0.4)
            
        # Lateral correction via turning motor neurons
        self.vx += readouts["turn_bias"] * 0.04
        
        # Physics update
        self.vy += self.gravity
        self.y += self.vy
        self.x += self.vx
        
        # Touchdown check
        done = False
        reward = 0.0
        if self.y >= ground_y:
            self.y = ground_y
            done = True
            if abs(self.vy) <= 1.8 and abs(self.vx) <= 1.2:
                self.is_landed = True
                self.score = 100 + int(self.fuel)
                reward = 100.0
            else:
                self.is_crashed = True
                reward = -50.0

        state = self._get_state()
        info = {
            "altitude": round(altitude, 1),
            "vy": round(self.vy, 2),
            "vx": round(self.vx, 2),
            "thrust_active": thrust_active,
            "readouts": readouts
        }
        return state, reward, done, info

    def _get_state(self) -> Dict[str, Any]:
        return {
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "vx": round(self.vx, 2),
            "vy": round(self.vy, 2),
            "fuel": round(self.fuel, 1),
            "is_landed": self.is_landed,
            "is_crashed": self.is_crashed,
            "score": self.score,
            "frames": self.frames
        }
