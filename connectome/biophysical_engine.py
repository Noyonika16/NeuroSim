"""
biophysical_engine.py
Membrane potential differential equation solver for the 302-neuron C. elegans connectome.
Implements:
- Graded synaptic transmission (Wicks et al. 1996, Lockery & Goodman, Roberts et al. 2016)
- Symmetrical electrical gap junction coupling via graph Laplacian
- Regenerative calcium plateau potentials (EGL-19 dynamics)
- Sensory input currents, optogenetic excitation/inhibition, and laser cell ablations
- High-efficiency vectorized NumPy integration (Euler & RK2)
"""

import numpy as np
from typing import Dict, List, Set, Optional, Tuple, Any
from .data_loader import Connectome

class BiophysicalConnectomeEngine:
    """
    Simulates membrane potential dynamics for all 302 neurons in C. elegans.
    """
    def __init__(self, connectome: Optional[Connectome] = None, dt: float = 0.002):
        """
        dt: Integration step size in seconds (default 2 ms = 0.002s).
        """
        self.dt = dt
        self.connectome = connectome or Connectome()
        self.num_neurons = len(self.connectome.neuron_names)
        self.neuron_names = self.connectome.neuron_names
        self.name_to_idx = {name: i for i, name in enumerate(self.neuron_names)}
        
        # Calibrated Biophysical Parameters
        self.tau = 0.030         # Membrane time constant (30 ms)
        self.E_leak = -60.0      # Resting membrane potential (mV)
        
        # Graded Synapse Parameters (Wicks 1996)
        self.syn_theta = -45.0   # Activation midpoint (mV)
        self.syn_beta = 4.5      # Slope factor (mV)
        self.g_chem_scale = 0.028 # Synaptic weight scale
        self.g_gap_scale = 0.018  # Gap junction scale
        
        # Regenerative Calcium Plateau Current (EGL-19 dynamics)
        self.enable_calcium = True
        self.g_Ca = 0.08         # Calcium conductance relative to leak
        self.E_Ca = 45.0         # Calcium reversal potential (mV)
        self.Ca_theta = -42.0    # Activation threshold (mV)
        self.Ca_sigma = 5.0      # Slope (mV)
        
        # Construct Connectivity Matrices
        self._build_matrices()
        
        # State Vectors (shape: (302,))
        self.V = np.full(self.num_neurons, self.E_leak, dtype=np.float64)
        self.I_sensory = np.zeros(self.num_neurons, dtype=np.float64)
        self.I_opto = np.zeros(self.num_neurons, dtype=np.float64)
        
        # Experimental Perturbations
        self.ablated_neurons: Set[int] = set()
        self.neuromodulation: Dict[str, float] = {
            "dopamine": 0.0,    # Slowing / sensory adaptation
            "serotonin": 0.0,   # Enhanced motor activity / pharyngeal pumping
            "octopamine": 0.0   # Roaming / fight-or-flight acceleration
        }
        
        # Activity History Buffer for Oscilloscope / Traces (last 300 steps)
        self.history_size = 300
        self.history_V = np.full((self.history_size, self.num_neurons), self.E_leak, dtype=np.float32)
        self.history_idx = 0
        self.time_seconds = 0.0

    def _build_matrices(self):
        """Constructs chemical weight matrix, reversal potentials, and gap Laplacian."""
        N = self.num_neurons
        
        # Chemical Synapse Matrices:
        self.W_chem = np.zeros((N, N), dtype=np.float64)
        self.E_syn = np.zeros((N, N), dtype=np.float64)
        
        for syn in self.connectome.chemical_synapses:
            if syn["pre"] in self.name_to_idx and syn["post"] in self.name_to_idx:
                pre_idx = self.name_to_idx[syn["pre"]]
                post_idx = self.name_to_idx[syn["post"]]
                w = syn["weight"] * self.g_chem_scale
                self.W_chem[pre_idx, post_idx] += w
                self.E_syn[pre_idx, post_idx] = syn["esyn"]
                
        # Gap Junction Laplacian:
        self.L_gap = np.zeros((N, N), dtype=np.float64)
        for gap in self.connectome.gap_junctions:
            if gap["n1"] in self.name_to_idx and gap["n2"] in self.name_to_idx:
                i = self.name_to_idx[gap["n1"]]
                j = self.name_to_idx[gap["n2"]]
                g = gap["weight"] * self.g_gap_scale
                self.L_gap[i, j] += g
                self.L_gap[j, i] += g
                
        # Symmetrical Laplacian row diagonal
        for i in range(N):
            self.L_gap[i, i] = -np.sum(self.L_gap[i, :])

    def set_sensory_input(self, neuron_name: str, current_mV: float):
        """Injects sensory current (in mV equivalent) into a specific neuron."""
        if neuron_name in self.name_to_idx:
            idx = self.name_to_idx[neuron_name]
            self.I_sensory[idx] = current_mV

    def set_sensory_inputs_bulk(self, current_dict: Dict[str, float]):
        """Injects currents into multiple sensory neurons simultaneously."""
        for name, current in current_dict.items():
            if name in self.name_to_idx:
                self.I_sensory[self.name_to_idx[name]] = current

    def clear_sensory_inputs(self):
        """Resets all sensory inputs to zero."""
        self.I_sensory.fill(0.0)

    def set_optogenetic_stimulation(self, neuron_name: str, current_mV: float):
        """Applies optogenetic stimulation (ChR2: +current, NpHR/Arch: -current)."""
        if neuron_name in self.name_to_idx:
            idx = self.name_to_idx[neuron_name]
            self.I_opto[idx] = current_mV

    def clear_optogenetics(self):
        """Clears all optogenetic inputs."""
        self.I_opto.fill(0.0)

    def laser_ablate_neuron(self, neuron_name: str):
        """Irreversibly silences a neuron (sets ablated state)."""
        if neuron_name in self.name_to_idx:
            self.ablated_neurons.add(self.name_to_idx[neuron_name])

    def restore_neuron(self, neuron_name: str):
        """Restores an ablated neuron."""
        if neuron_name in self.name_to_idx:
            self.ablated_neurons.discard(self.name_to_idx[neuron_name])

    def reset_ablations(self):
        """Restores all ablated neurons."""
        self.ablated_neurons.clear()

    def set_neuromodulation(self, dopamine: float = 0.0, serotonin: float = 0.0, octopamine: float = 0.0):
        """Sets global neuromodulatory levels (0.0 to 1.0)."""
        self.neuromodulation["dopamine"] = max(0.0, min(1.0, dopamine))
        self.neuromodulation["serotonin"] = max(0.0, min(1.0, serotonin))
        self.neuromodulation["octopamine"] = max(0.0, min(1.0, octopamine))

    def reset_state(self):
        """Resets membrane potentials and histories."""
        self.V.fill(self.E_leak)
        self.I_sensory.fill(0.0)
        self.I_opto.fill(0.0)
        self.history_V.fill(self.E_leak)
        self.history_idx = 0
        self.time_seconds = 0.0

    def compute_derivatives(self, V_current: np.ndarray) -> np.ndarray:
        """
        Computes dV/dt according to biophysical differential equations.
        dV/dt = ( - (V - E_leak) + L_gap @ V + I_chem + I_Ca + I_sensory + I_opto + noise ) / tau
        """
        # 1. Passive Leak
        dV_leak = -(V_current - self.E_leak)
        
        # 2. Electrical Gap Junction Coupling
        dV_gap = self.L_gap @ V_current
        
        # 3. Graded Chemical Synaptic Transmission
        # Presynaptic activation: s = 1 / (1 + exp(-(V_pre - theta) / beta))
        s_pre = 1.0 / (1.0 + np.exp(-np.clip((V_current - self.syn_theta) / self.syn_beta, -20.0, 20.0)))
        
        # Zero out transmission from ablated neurons
        if self.ablated_neurons:
            s_pre[list(self.ablated_neurons)] = 0.0
            
        # Neuromodulatory tuning
        mod_factor = (1.0 + 0.35 * self.neuromodulation["serotonin"] - 0.20 * self.neuromodulation["dopamine"])
        s_pre_mod = s_pre * mod_factor
        
        effective_G = s_pre_mod[:, None] * self.W_chem
        G_in = np.sum(effective_G, axis=0)
        GE_in = np.sum(effective_G * self.E_syn, axis=0)
        dV_chem = GE_in - G_in * V_current
        
        # 4. Regenerative Calcium Currents (EGL-19 / UNC-2)
        if self.enable_calcium:
            m_inf = 1.0 / (1.0 + np.exp(-np.clip((V_current - self.Ca_theta) / self.Ca_sigma, -20.0, 20.0)))
            dV_Ca = self.g_Ca * m_inf * (self.E_Ca - V_current)
        else:
            dV_Ca = 0.0
            
        # 5. Stochastic Thermal Noise
        noise = np.random.normal(0.0, 0.25, size=self.num_neurons)
        
        total_forcing = dV_leak + dV_gap + dV_chem + dV_Ca + self.I_sensory + self.I_opto + noise
        
        return total_forcing / self.tau

    def step(self) -> np.ndarray:
        """
        Advances the simulation by dt using 2nd-order Runge-Kutta (RK2 / Midpoint).
        Returns current voltage array V.
        """
        # RK2 Midpoint Step
        k1 = self.compute_derivatives(self.V)
        V_mid = self.V + 0.5 * self.dt * k1
        
        if self.ablated_neurons:
            for idx in self.ablated_neurons:
                V_mid[idx] = self.E_leak
                
        k2 = self.compute_derivatives(V_mid)
        self.V = self.V + self.dt * k2
        
        # Enforce physical biological boundaries: -80 mV to +25 mV
        np.clip(self.V, -80.0, 25.0, out=self.V)
        
        # Clamp ablated neurons to resting leak
        if self.ablated_neurons:
            for idx in self.ablated_neurons:
                self.V[idx] = self.E_leak
                
        # Record into history buffer
        self.history_V[self.history_idx] = self.V
        self.history_idx = (self.history_idx + 1) % self.history_size
        self.time_seconds += self.dt
        
        return self.V

    def get_neuron_voltage(self, neuron_name: str) -> float:
        """Returns membrane potential of a neuron in mV."""
        if neuron_name in self.name_to_idx:
            return float(self.V[self.name_to_idx[neuron_name]])
        return self.E_leak

    def get_activity_snapshot(self) -> Dict[str, float]:
        """Returns dictionary of all neuron voltages."""
        return {name: float(self.V[i]) for i, name in enumerate(self.neuron_names)}

    def get_motor_readouts(self) -> Dict[str, float]:
        """
        Decodes motor activity:
        - Forward vs Backward locomotion drive (AVB/PVC vs AVA/AVD)
        - Dorsal vs Ventral muscle contraction drive (DB vs VB)
        - Head Turning drive (SMDD/SMDV, RMD)
        """
        def mean_v(prefixes: List[str]) -> float:
            vals = [self.V[i] for i, name in enumerate(self.neuron_names) if any(name.startswith(p) for p in prefixes)]
            return float(np.mean(vals)) if vals else self.E_leak

        v_ava = (self.get_neuron_voltage("AVAL") + self.get_neuron_voltage("AVAR")) / 2.0
        v_avb = (self.get_neuron_voltage("AVBL") + self.get_neuron_voltage("AVBR")) / 2.0
        v_pvc = (self.get_neuron_voltage("PVCL") + self.get_neuron_voltage("PVCR")) / 2.0
        v_avd = (self.get_neuron_voltage("AVDL") + self.get_neuron_voltage("AVDR")) / 2.0
        
        # Forward command index: forward circuit (AVB, PVC) minus reversal circuit (AVA, AVD)
        forward_drive = ((v_avb + v_pvc) / 2.0) - ((v_ava + v_avd) / 2.0)
        
        # Dorsal vs Ventral body motor pools
        db_activity = mean_v(["DB"])
        vb_activity = mean_v(["VB"])
        dorsal_ventral_bias = db_activity - vb_activity
        
        # Head steering (left vs right)
        smdd_l = self.get_neuron_voltage("SMDDL")
        smdd_r = self.get_neuron_voltage("SMDDR")
        smdv_l = self.get_neuron_voltage("SMDVL")
        smdv_r = self.get_neuron_voltage("SMDVR")
        
        turn_bias = (smdd_l + smdv_l) - (smdd_r + smdv_r)
        
        return {
            "v_ava": round(v_ava, 2),
            "v_avb": round(v_avb, 2),
            "forward_drive": round(forward_drive, 3),
            "db_activity": round(db_activity, 2),
            "vb_activity": round(vb_activity, 2),
            "dorsal_ventral_bias": round(dorsal_ventral_bias, 3),
            "turn_bias": round(turn_bias, 3)
        }
