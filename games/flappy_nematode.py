"""
flappy_nematode.py
Cybernetic Flappy Bird environment played entirely by the C. elegans 302-neuron connectome.
Sensory inputs: pipe distance, vertical gap alignment, ceiling/floor hazard.
Motor output: Ventral/Dorsal muscle contraction (DB vs VB) triggering flaps.
"""

import math
import random
from typing import Dict, List, Tuple, Any, Optional
try:
    from ..connectome.biophysical_engine import BiophysicalConnectomeEngine
    from ..connectome.sensorimotor_mapping import FlappyBirdSensoryEncoder, FlappyBirdMotorDecoder
except (ImportError, ValueError):
    from connectome.biophysical_engine import BiophysicalConnectomeEngine
    from connectome.sensorimotor_mapping import FlappyBirdSensoryEncoder, FlappyBirdMotorDecoder

class FlappyNematodeGame:
    def __init__(self, engine: Optional[BiophysicalConnectomeEngine] = None, screen_w: int = 400, screen_h: int = 500):
        self.screen_w = screen_w
        self.screen_h = screen_h
        
        self.engine = engine or BiophysicalConnectomeEngine()
        self.encoder = FlappyBirdSensoryEncoder(self.engine)
        self.decoder = FlappyBirdMotorDecoder(self.engine, flap_threshold=3.2)
        
        self.reset()

    def reset(self):
        self.bird_x = 75.0
        self.bird_y = self.screen_h * 0.45
        self.bird_vy = 0.0
        self.gravity = 0.42
        self.flap_strength = -6.5
        
        # Pipe obstacles
        self.pipe_w = 52.0
        self.pipe_gap = 120.0
        self.pipe_speed = 3.2
        self.pipes: List[Dict[str, float]] = []
        self._spawn_pipe(x=self.screen_w + 30.0)
        self._spawn_pipe(x=self.screen_w + 220.0)
        
        self.score = 0
        self.frames = 0
        self.is_game_over = False
        self.flap_count = 0
        self.engine.reset_state()
        
        # Undulating worm body segments (12 segments for visual rendering)
        self.body_segments = [{"x": self.bird_x - i * 5.0, "y": self.bird_y} for i in range(12)]
        
        return self._get_state()

    def _spawn_pipe(self, x: float):
        min_y = 70.0
        max_y = self.screen_h - self.pipe_gap - 70.0
        gap_y = random.uniform(min_y, max_y)
        self.pipes.append({
            "x": x,
            "gap_y": gap_y,
            "passed": False
        })

    def _get_upcoming_pipe(self) -> Dict[str, float]:
        for p in self.pipes:
            if p["x"] + self.pipe_w > self.bird_x - 15.0:
                return p
        return self.pipes[0]

    def step(self) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Runs one physical frame of the game and advances connectome simulation.
        """
        if self.is_game_over:
            return self._get_state(), 0.0, True, {"info": "Game Over"}

        self.frames += 1
        
        # 1. Query upcoming obstacle
        pipe = self._get_upcoming_pipe()
        pipe_dist_x = pipe["x"] - self.bird_x
        pipe_center_y = pipe["gap_y"] + (self.pipe_gap * 0.5)
        
        # 2. Inject sensory currents into C. elegans connectome
        sensory_inputs = self.encoder.encode_and_inject(
            bird_y=self.bird_y,
            bird_vy=self.bird_vy,
            pipe_x=pipe_dist_x,
            pipe_gap_y=pipe_center_y,
            pipe_gap_size=self.pipe_gap,
            screen_height=self.screen_h
        )
        
        # 3. Step Biophysical Engine (run 4 sub-steps of 2ms = 8ms biological time)
        for _ in range(4):
            self.engine.step()
            
        # 4. Decode motor command from ventral/dorsal motor neurons
        should_flap, drive, motor_metrics = self.decoder.decode_action()
        
        if should_flap:
            self.bird_vy = self.flap_strength
            self.flap_count += 1
            
        # 5. Apply Environmental Physics
        self.bird_vy += self.gravity
        self.bird_vy = min(8.5, max(-9.0, self.bird_vy))
        self.bird_y += self.bird_vy
        
        # Move pipes
        for p in self.pipes:
            p["x"] -= self.pipe_speed
            if not p["passed"] and p["x"] + self.pipe_w < self.bird_x:
                p["passed"] = True
                self.score += 1
                
        # Remove offscreen pipes and spawn new ones
        if self.pipes and self.pipes[0]["x"] < -self.pipe_w:
            self.pipes.pop(0)
            self._spawn_pipe(x=self.pipes[-1]["x"] + 190.0)
            
        # 6. Update undulating nematode body posture
        # Head follows bird_y
        self.body_segments[0] = {"x": self.bird_x, "y": self.bird_y}
        for i in range(1, len(self.body_segments)):
            prev = self.body_segments[i - 1]
            curr = self.body_segments[i]
            # Undulatory wave based on dorsal-ventral bias
            wave = math.sin((self.frames * 0.18) - (i * 0.45)) * 4.0
            target_y = prev["y"] + wave * 0.4
            curr["x"] = self.bird_x - i * 5.0
            curr["y"] += (target_y - curr["y"]) * 0.5
            
        # 7. Collision Detection
        # Floor & Ceiling
        if self.bird_y < 12.0 or self.bird_y > self.screen_h - 18.0:
            self.is_game_over = True
            
        # Pipe collisions
        bird_radius = 11.0
        for p in self.pipes:
            # Check horizontal overlap
            if (self.bird_x + bird_radius > p["x"]) and (self.bird_x - bird_radius < p["x"] + self.pipe_w):
                # Check if outside gap
                top_pipe_bottom = p["gap_y"]
                bottom_pipe_top = p["gap_y"] + self.pipe_gap
                if (self.bird_y - bird_radius < top_pipe_bottom) or (self.bird_y + bird_radius > bottom_pipe_top):
                    self.is_game_over = True
                    break
                    
        reward = 1.0 if not self.is_game_over else -10.0
        
        state = self._get_state()
        info = {
            "sensory_inputs": sensory_inputs,
            "motor_metrics": motor_metrics,
            "did_flap": should_flap
        }
        
        return state, reward, self.is_game_over, info

    def _get_state(self) -> Dict[str, Any]:
        pipe = self._get_upcoming_pipe()
        return {
            "bird_x": self.bird_x,
            "bird_y": self.bird_y,
            "bird_vy": self.bird_vy,
            "score": self.score,
            "frames": self.frames,
            "flap_count": self.flap_count,
            "is_game_over": self.is_game_over,
            "pipe_x": pipe["x"],
            "pipe_gap_y": pipe["gap_y"],
            "pipe_gap": self.pipe_gap,
            "pipes": [{"x": p["x"], "gap_y": p["gap_y"]} for p in self.pipes],
            "body_segments": self.body_segments
        }
