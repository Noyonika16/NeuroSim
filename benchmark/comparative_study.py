"""
comparative_study.py
Research benchmark:
Directly tests:
1. Authentic C. elegans biological connectome (zero-shot, 0 backprop training)
2. Scrambled / Random Topology network (Erdos-Renyi with identical node/edge counts)
3. Shuffled Synapse Weights (identical topology, scrambled weights)
4. Laser-Ablated Circuit (AVA / command reversal killed)
5. Classical Feedforward Controller / Random Agent

Measures survival frames, pipes cleared, motor efficiency, and parameter counts.
"""

import copy
import random
import numpy as np
from typing import Dict, List, Any
try:
    from ..connectome.biophysical_engine import BiophysicalConnectomeEngine
    from ..connectome.data_loader import Connectome
    from ..games.flappy_nematode import FlappyNematodeGame
except (ImportError, ValueError):
    from connectome.biophysical_engine import BiophysicalConnectomeEngine
    from connectome.data_loader import Connectome
    from games.flappy_nematode import FlappyNematodeGame

class ComparativeStudyBenchmark:
    def __init__(self, trials_per_condition: int = 5):
        self.trials = trials_per_condition
        self.base_connectome = Connectome()

    def run_full_suite(self) -> Dict[str, Any]:
        """Runs comparative benchmark across all agent architectures."""
        results = {}
        
        # 1. Biological C. elegans Connectome (Zero-Shot)
        print("Benchmarking Authentic Biological Connectome...")
        bio_engine = BiophysicalConnectomeEngine(self.base_connectome)
        results["Biological Connectome (Authentic)"] = self._evaluate_engine(bio_engine)

        # 2. Shuffled Synapse Weights (Topology Preserved, Synaptic Weights Scrambled)
        print("Benchmarking Shuffled Weights Condition...")
        shuffled_conn = copy.deepcopy(self.base_connectome)
        weights = [syn["weight"] for syn in shuffled_conn.chemical_synapses]
        random.shuffle(weights)
        for i, syn in enumerate(shuffled_conn.chemical_synapses):
            syn["weight"] = weights[i]
        shuffled_engine = BiophysicalConnectomeEngine(shuffled_conn)
        results["Shuffled Weights (Topology Preserved)"] = self._evaluate_engine(shuffled_engine)

        # 3. Random Erdős-Rényi Connectome (Scrambled Topology)
        print("Benchmarking Random Topology Control...")
        random_conn = copy.deepcopy(self.base_connectome)
        neuron_list = random_conn.neuron_names
        for syn in random_conn.chemical_synapses:
            syn["pre"] = random.choice(neuron_list)
            syn["post"] = random.choice(neuron_list)
        random_engine = BiophysicalConnectomeEngine(random_conn)
        results["Random Topology (Erdos-Renyi)"] = self._evaluate_engine(random_engine)

        # 4. Command Circuit Ablation (AVA / AVD killed)
        print("Benchmarking Command Ablation Condition...")
        ablated_engine = BiophysicalConnectomeEngine(self.base_connectome)
        for n in ["AVAL", "AVAR", "AVDL", "AVDR"]:
            ablated_engine.laser_ablate_neuron(n)
        results["Ablated Command Circuit (AVA/AVD Silent)"] = self._evaluate_engine(ablated_engine)

        # 5. Untrained Random Action Baseline
        print("Benchmarking Random Action Agent...")
        results["Random Action Agent"] = self._evaluate_random_agent()

        return results

    def _evaluate_engine(self, engine: BiophysicalConnectomeEngine) -> Dict[str, Any]:
        scores = []
        frames_survived = []
        flaps = []
        
        for t in range(self.trials):
            game = FlappyNematodeGame(engine=engine)
            done = False
            total_reward = 0.0
            
            # Cap at 1200 frames to avoid infinite runs during testing
            while not done and game.frames < 1200:
                state, reward, done, info = game.step()
                total_reward += reward
                
            scores.append(game.score)
            frames_survived.append(game.frames)
            flaps.append(game.flap_count)
            
        return {
            "mean_score": round(float(np.mean(scores)), 2),
            "max_score": int(np.max(scores)),
            "mean_frames": round(float(np.mean(frames_survived)), 1),
            "mean_flaps": round(float(np.mean(flaps)), 1),
            "trainable_parameters": 0,
            "biological_neurons": 302,
            "requires_backpropagation": False
        }

    def _evaluate_random_agent(self) -> Dict[str, Any]:
        scores = []
        frames_survived = []
        for _ in range(self.trials):
            game = FlappyNematodeGame()
            done = False
            while not done and game.frames < 1200:
                # Random flap with 15% probability
                if random.random() < 0.15:
                    game.bird_vy = game.flap_strength
                    game.flap_count += 1
                game.bird_vy += game.gravity
                game.bird_y += game.bird_vy
                game.frames += 1
                for p in game.pipes:
                    p["x"] -= game.pipe_speed
                    if not p["passed"] and p["x"] + game.pipe_w < game.bird_x:
                        p["passed"] = True
                        game.score += 1
                if game.bird_y < 10.0 or game.bird_y > game.screen_h - 15.0:
                    done = True
            scores.append(game.score)
            frames_survived.append(game.frames)

        return {
            "mean_score": round(float(np.mean(scores)), 2),
            "max_score": int(np.max(scores)),
            "mean_frames": round(float(np.mean(frames_survived)), 1),
            "mean_flaps": 0,
            "trainable_parameters": 0,
            "biological_neurons": 0,
            "requires_backpropagation": False
        }

if __name__ == "__main__":
    benchmark = ComparativeStudyBenchmark(trials_per_condition=3)
    results = benchmark.run_full_suite()
    import json
    print(json.dumps(results, indent=2))
