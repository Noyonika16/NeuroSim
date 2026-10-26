"""
test_simulation.py
Automated validation test suite for NEURO-ELEGANS:
- Connectome integrity (exact 302 neurons, synapses, coordinates)
- Biophysical numerical stability & ODE bounds
- Sensory-to-motor reflex propagation
- Laser ablation & optogenetic modulation
- Retro game closed-loop mechanics
"""

import unittest
import numpy as np
import os
import sys

# Ensure local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from connectome.data_loader import Connectome
from connectome.biophysical_engine import BiophysicalConnectomeEngine
from connectome.graph_analytics import ConnectomeGraphAnalytics
from games.flappy_nematode import FlappyNematodeGame
from games.chemotaxis_snake import ChemotaxisSnakeArena

class TestNeuroElegans(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.connectome = Connectome()
        cls.engine = BiophysicalConnectomeEngine(cls.connectome)

    def test_connectome_integrity(self):
        """Validates that exactly 302 canonical neurons and real synapses are loaded."""
        self.assertEqual(len(self.connectome.neuron_names), 302)
        self.assertEqual(len(self.connectome.neurons), 302)
        self.assertGreater(len(self.connectome.chemical_synapses), 2000)
        self.assertGreater(len(self.connectome.gap_junctions), 500)
        
        # Verify crucial command and sensory neurons exist
        for key_neuron in ["AVAL", "AVAR", "AVBL", "AVBR", "ASHL", "ALML", "PLML", "ASEL", "ASER", "DB1", "VB1", "DD1", "VD1"]:
            self.assertIn(key_neuron, self.connectome.neurons)

    def test_biophysical_stability(self):
        """Simulates 100 steps and ensures bounded membrane potentials without NaNs."""
        self.engine.reset_state()
        for _ in range(100):
            v = self.engine.step()
            self.assertFalse(np.isnan(v).any(), "NaN detected in membrane potentials")
            self.assertFalse(np.isinf(v).any(), "Inf detected in membrane potentials")
            self.assertTrue((v >= -85.0).all(), "Voltage fell below physiological minimum")
            self.assertTrue((v <= 35.0).all(), "Voltage exceeded physiological maximum")

    def test_anterior_touch_reflex(self):
        """Confirms that anterior touch stimulus depolarizes AVA reversal command circuit."""
        self.engine.reset_state()
        # Baseline
        for _ in range(40):
            self.engine.step()
        base_v_ava = self.engine.get_neuron_voltage("AVAL")

        # Inject anterior touch into ALML and ASHL
        self.engine.set_sensory_input("ALML", 25.0)
        self.engine.set_sensory_input("ASHL", 25.0)
        for _ in range(50):
            self.engine.step()
        stim_v_ava = self.engine.get_neuron_voltage("AVAL")

        self.assertGreater(stim_v_ava, base_v_ava, "Anterior touch stimulus failed to depolarize AVA")

    def test_laser_ablation(self):
        """Confirms that laser ablation silences the target neuron to resting potential."""
        self.engine.reset_state()
        self.engine.laser_ablate_neuron("AVAL")
        self.engine.set_sensory_input("AVAL", 50.0) # Try to force stimulate
        for _ in range(30):
            self.engine.step()
        self.assertEqual(self.engine.get_neuron_voltage("AVAL"), self.engine.E_leak)
        self.engine.reset_ablations()

    def test_graph_analytics(self):
        """Verifies small-world network analysis."""
        analytics = ConnectomeGraphAnalytics(self.connectome)
        stats = analytics.get_summary_stats()
        self.assertEqual(stats["num_nodes"], 302)
        self.assertGreater(stats["avg_clustering_coefficient"], 0.15)
        self.assertLess(stats["avg_shortest_path_length"], 4.5)

    def test_flappy_nematode_environment(self):
        """Tests closed-loop Flappy Bird gameplay."""
        game = FlappyNematodeGame(self.engine)
        for _ in range(50):
            state, reward, done, info = game.step()
            self.assertIn("bird_y", state)
            self.assertIn("pipes", state)
            if done:
                break

    def test_chemotaxis_arena(self):
        """Tests continuous 2D klinotaxis arena."""
        arena = ChemotaxisSnakeArena(self.engine)
        for _ in range(50):
            state, reward, done, info = arena.step()
            self.assertIn("worm_x", state)
            self.assertIn("food_x", state)
            self.assertFalse(done)

if __name__ == "__main__":
    unittest.main()
