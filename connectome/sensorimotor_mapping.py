"""
sensorimotor_mapping.py
Translates virtual game environment dynamics into biological sensory neuron currents,
and decodes motor neuron voltages into physical movements and game controls.
"""

import numpy as np
from typing import Dict, Tuple, Any
from .biophysical_engine import BiophysicalConnectomeEngine

class FlappyBirdSensoryEncoder:
    """
    Encodes Flappy Bird game state into C. elegans sensory currents.
    - Pipe proximity -> ASH (polymodal avoidance), FLP (head mechanosensory)
    - Gap alignment (above/below gap center) -> CEPD/CEPV, IL1D/IL1V
    - Vertical velocity / proprioception -> PVD mechanosensors
    """
    def __init__(self, engine: BiophysicalConnectomeEngine):
        self.engine = engine

    def encode_and_inject(self, bird_y: float, bird_vy: float, pipe_x: float, pipe_gap_y: float, pipe_gap_size: float, screen_height: float):
        """
        bird_y: 0 (top) to screen_height (bottom)
        pipe_x: distance to upcoming pipe (pixels)
        pipe_gap_y: vertical center of gap
        """
        currents: Dict[str, float] = {}
        
        # 1. Horizontal Proximity to obstacle (0 to 1)
        # Closer obstacle -> higher nociceptive ASH avoidance current
        proximity = max(0.0, 1.0 - (pipe_x / 320.0))
        ash_current = 28.0 * (proximity ** 1.8)
        currents["ASHL"] = ash_current
        currents["ASHR"] = ash_current
        currents["FLPL"] = ash_current * 0.8
        currents["FLPR"] = ash_current * 0.8
        
        # 2. Gap Elevation Error: delta_y = bird_y - pipe_gap_y
        # In screen coords: larger y means lower down.
        # If bird_y > pipe_gap_y, bird is below the gap -> needs to flap UP -> excite ventral touch / dorsal motor drive!
        # If bird_y < pipe_gap_y, bird is above the gap -> needs to glide DOWN -> excite dorsal touch / inhibit flap!
        delta_y = (bird_y - pipe_gap_y) / (pipe_gap_size * 0.5) # normalized error
        
        if delta_y > 0:
            # Below gap center: excite ventral touch CEPV / IL1V (stimulates upward escape bend)
            stim = min(35.0, 18.0 * abs(delta_y) * (0.4 + 0.6 * proximity))
            currents["CEPVL"] = stim
            currents["CEPVR"] = stim
            currents["IL1VL"] = stim * 0.7
            currents["IL1VR"] = stim * 0.7
            # Inhibit dorsal
            currents["CEPDL"] = 0.0
            currents["CEPDR"] = 0.0
        else:
            # Above gap center: excite dorsal touch CEPD / IL1D (relaxes upward bend, dives)
            stim = min(35.0, 18.0 * abs(delta_y) * (0.4 + 0.6 * proximity))
            currents["CEPDL"] = stim
            currents["CEPDR"] = stim
            currents["IL1DL"] = stim * 0.7
            currents["IL1DR"] = stim * 0.7
            currents["CEPVL"] = 0.0
            currents["CEPVR"] = 0.0

        # 3. Ground / Floor Collision Hazard
        ground_dist = screen_height - bird_y
        if ground_dist < 80.0:
            ground_danger = 35.0 * ((80.0 - ground_dist) / 80.0) ** 2
            currents["ALML"] = currents.get("ALML", 0.0) + ground_danger
            currents["ALMR"] = currents.get("ALMR", 0.0) + ground_danger
            currents["AVM"] = ground_danger

        # 4. Proprioception: vertical velocity
        # Positive vy means falling fast; negative vy means rising fast
        if bird_vy > 1.0: # falling fast
            currents["PVDL"] = min(20.0, bird_vy * 2.5)
            currents["PVDR"] = min(20.0, bird_vy * 2.5)

        self.engine.set_sensory_inputs_bulk(currents)
        return currents

class FlappyBirdMotorDecoder:
    """
    Decodes motor neuron outputs into Flappy Bird control (Flap vs Glide).
    Uses dorsal B-motor pool (DB) vs ventral B-motor pool (VB) and forward command (AVB).
    """
    def __init__(self, engine: BiophysicalConnectomeEngine, flap_threshold: float = 3.2):
        self.engine = engine
        self.flap_threshold = flap_threshold
        self.cooldown = 0
        self.cooldown_steps = 7 # Refractory period between flaps

    def decode_action(self) -> Tuple[bool, float, Dict[str, float]]:
        """
        Returns:
            should_flap: bool
            flap_intensity: float
            metrics: dict of muscle activations
        """
        readouts = self.engine.get_motor_readouts()
        
        # Calculate Dorsal vs Ventral drive
        db_act = readouts["db_activity"] # resting around -56 mV
        vb_act = readouts["vb_activity"]
        dv_bias = readouts["dorsal_ventral_bias"]
        v_avb = readouts["v_avb"]
        
        # Net Flap Muscle Drive
        # In C. elegans, dorsal muscle contraction pulls head and body dorsally (upward bending wave).
        # We also factor in command forward excitation.
        drive = dv_bias + 0.15 * (v_avb - (-55.0))
        
        should_flap = False
        if self.cooldown > 0:
            self.cooldown -= 1
        elif drive > self.flap_threshold:
            should_flap = True
            self.cooldown = self.cooldown_steps
            
        return should_flap, drive, {
            "drive": round(drive, 3),
            "db_act": round(db_act, 2),
            "vb_act": round(vb_act, 2),
            "dv_bias": round(dv_bias, 3),
            "v_avb": round(v_avb, 2),
            "threshold": self.flap_threshold
        }

class ChemotaxisSensoryEncoder:
    """
    Encodes chemical food gradient into C. elegans gustatory/olfactory amphid sensory neurons:
    - ASEL: activated when dC/dt > 0 (salt/food concentration increasing, ON-cell)
    - ASER: activated when dC/dt < 0 (salt/food concentration decreasing, OFF-cell)
    - AWC: senses absolute food concentration C
    - ASH: senses obstacle/wall proximity (repellent)
    """
    def __init__(self, engine: BiophysicalConnectomeEngine):
        self.engine = engine
        self.prev_concentration = 0.0

    def encode_and_inject(self, current_conc: float, wall_distance: float):
        dC_dt = current_conc - self.prev_concentration
        self.prev_concentration = current_conc
        
        currents: Dict[str, float] = {}
        
        # Chemosensory ON-cell (ASEL) vs OFF-cell (ASER)
        if dC_dt > 0:
            asel_stim = min(30.0, 120.0 * dC_dt)
            currents["ASEL"] = asel_stim
            currents["ASER"] = 0.0
            currents["AWCL"] = min(20.0, 15.0 * current_conc)
            currents["AWCR"] = min(20.0, 15.0 * current_conc)
        else:
            aser_stim = min(30.0, 120.0 * abs(dC_dt))
            currents["ASER"] = aser_stim
            currents["ASEL"] = 0.0
            
        # Wall avoidance
        if wall_distance < 35.0:
            ash_stim = 30.0 * ((35.0 - wall_distance) / 35.0)
            currents["ASHL"] = ash_stim
            currents["ASHR"] = ash_stim

        self.engine.set_sensory_inputs_bulk(currents)
        return currents

class ChemotaxisMotorDecoder:
    """
    Decodes motor neuron outputs into Klinotaxis steering and forward velocity.
    - Forward speed proportional to AVB (forward command interneuron)
    - Steering angle driven by sublateral motor neurons (SMD, RMD) and pirouette command (AVA)
    """
    def __init__(self, engine: BiophysicalConnectomeEngine):
        self.engine = engine

    def decode_movement(self) -> Tuple[float, float, Dict[str, float]]:
        readouts = self.engine.get_motor_readouts()
        
        # Forward speed (clamped 0.5 to 3.5)
        v_avb = readouts["v_avb"]
        v_ava = readouts["v_ava"]
        
        # When AVA is high (reversal/pirouette), forward speed slows and turning rate spikes!
        is_pirouette = (v_ava > -38.0)
        
        if is_pirouette:
            speed = 0.8
            # Sharp pirouette turn
            turn_rate = (readouts["turn_bias"] * 0.12) + (np.random.choice([-1.0, 1.0]) * 0.35)
        else:
            speed = max(1.0, min(3.8, 1.8 + 0.12 * (v_avb - (-50.0))))
            turn_rate = readouts["turn_bias"] * 0.05
            
        return speed, turn_rate, {
            "speed": round(speed, 2),
            "turn_rate": round(turn_rate, 3),
            "is_pirouette": is_pirouette,
            "v_ava": readouts["v_ava"],
            "v_avb": readouts["v_avb"]
        }
