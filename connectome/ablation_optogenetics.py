"""
ablation_optogenetics.py
In-silico neuroscience experimental suite:
- Single-cell and circuit-level laser ablation
- Optogenetic excitation (ChR2 @ 470nm) and hyperpolarization (NpHR @ 590nm)
- Neuromodulatory bath application (Dopamine, Serotonin, Octopamine)
- Automated phenotypic evaluation
"""

from typing import Dict, List, Set, Tuple, Any, Optional
from .biophysical_engine import BiophysicalConnectomeEngine

class ExperimentalLab:
    """
    Experimental interface for manipulating the living C. elegans connectome in real-time.
    """
    def __init__(self, engine: BiophysicalConnectomeEngine):
        self.engine = engine
        self.active_optogenetics: Dict[str, Dict[str, Any]] = {}
        self.ablated_circuits: Set[str] = set()

    def apply_ablation(self, target: str) -> List[str]:
        """
        Ablates single neuron or predefined functional group:
        'AVA_pair', 'AVB_pair', 'command_reversal', 'gaba_cross_inhibitory', 'touch_mechanoreceptors', 'amphid_chemotaxis'
        """
        killed = []
        if target == "AVA_pair":
            for n in ["AVAL", "AVAR"]:
                self.engine.laser_ablate_neuron(n)
                killed.append(n)
        elif target == "AVB_pair":
            for n in ["AVBL", "AVBR"]:
                self.engine.laser_ablate_neuron(n)
                killed.append(n)
        elif target == "command_reversal":
            for n in ["AVAL", "AVAR", "AVDL", "AVDR", "AVEL", "AVER"]:
                self.engine.laser_ablate_neuron(n)
                killed.append(n)
        elif target == "gaba_cross_inhibitory":
            # Kill DD and VD motor neurons
            for n in self.engine.neuron_names:
                if n.startswith("DD") or n.startswith("VD"):
                    self.engine.laser_ablate_neuron(n)
                    killed.append(n)
        elif target == "touch_mechanoreceptors":
            for n in ["ALML", "ALMR", "AVM", "PLML", "PLMR"]:
                self.engine.laser_ablate_neuron(n)
                killed.append(n)
        elif target == "amphid_chemotaxis":
            for n in ["ASEL", "ASER", "AWCL", "AWCR", "AIYL", "AIYR", "AIBL", "AIBR"]:
                self.engine.laser_ablate_neuron(n)
                killed.append(n)
        else:
            # Single neuron name
            if target in self.engine.name_to_idx:
                self.engine.laser_ablate_neuron(target)
                killed.append(target)
                
        self.ablated_circuits.add(target)
        return killed

    def restore_all(self):
        """Restores all ablated neurons and clears optogenetic protocols."""
        self.engine.reset_ablations()
        self.engine.clear_optogenetics()
        self.active_optogenetics.clear()
        self.ablated_circuits.clear()

    def activate_optogenetic_channel(self, opsin_type: str, neuron_name: str, intensity: float = 25.0):
        """
        opsin_type: 'ChR2' (Channelrhodopsin-2, blue light, depolarizing +current)
                    'NpHR' (Halorhodopsin, yellow light, hyperpolarizing -current)
        """
        if opsin_type == "ChR2":
            current = abs(intensity)
            wavelength_nm = 470
            color = "#00d2ff"
        else: # NpHR / Arch
            current = -abs(intensity)
            wavelength_nm = 590
            color = "#ffb300"

        self.engine.set_optogenetic_stimulation(neuron_name, current)
        self.active_optogenetics[neuron_name] = {
            "opsin": opsin_type,
            "current": current,
            "wavelength": wavelength_nm,
            "color": color
        }

    def deactivate_optogenetic_channel(self, neuron_name: str):
        if neuron_name in self.active_optogenetics:
            self.engine.set_optogenetic_stimulation(neuron_name, 0.0)
            del self.active_optogenetics[neuron_name]

    def set_neuromodulatory_bath(self, dopamine: float, serotonin: float, octopamine: float):
        self.engine.set_neuromodulation(dopamine=dopamine, serotonin=serotonin, octopamine=octopamine)
