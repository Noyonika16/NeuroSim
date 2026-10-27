# NEURO-ELEGANS v13.3

**A connectome-based, closed-loop sensorimotor simulation of *C. elegans*.** This project uses a canonical 302-neuron wiring dataset, simplified membrane-potential dynamics, synthetic sensory encoders, and explicit neural-output decoders to control two small game environments.

> **Scientific scope:** this is a computational model inspired by the *C. elegans* connectome, not a living worm or a validated whole-animal simulation. The connectome dynamics and synaptic weights are frozen at runtime, but the trained-hybrid game readouts are trained offline to imitate the assisted reference policy. Sensory encoders, membrane parameters, motor decoding, and game physics are engineered assumptions. The default Neuron-only controller has no game-aware steering or boundary-turn override. Scores do not establish biological intelligence.

## What's in v13.3

- Interactive Three.js view of the 302-neuron network, live voltage visualization, individual neuron ablation, and modeled AVA optogenetic current.
- **Flappy Nematode:** synthetic obstacle-edge/floor sensory currents feed the network. The default Neuron-only mode uses an untrained DB−VB motor-neuron contrast to trigger flaps. The separate Trained neural hybrid uses a 99-feature regularized logistic readout trained offline to imitate the assisted policy.
- **Chemotaxis:** temporal concentration change is encoded through ASEL/ASER-like inputs; a synthetic bilateral head-side concentration probe supplies local AWCL/AWCR currents. A 105-feature ridge-regression readout predicts turn corrections from neuron voltages only at inference. It was trained offline to imitate assisted angular corrections. The trained hybrid includes a separate ASER-associated reorientation adapter; it no longer has a wall-turn safety override. The default Neuron-only Chemotaxis mode retains its untrained lateral readout and has no boundary-turn override; wall proximity may be encoded into ASH sensory input, but only the neuron-state readout changes heading.
- **Ablation handling:** selected features belonging to ablated neurons are omitted from trained readouts (zero standardized contribution), rather than interpreted as measured leak voltage. Remaining neurons still reflect network-wide effects of the ablation. Population motor means exclude ablated cells. Trial-start and record-time ablations are recorded separately.
- **Reproducible random streams:** environment placement, stochastic reorientation, and neuron noise use separate seeded streams. Noise is pre-drawn per neuron so an ablated neuron does not shift the noise sequence seen by surviving cells; ablation also does not silently change future pipe or food placement.
- **Optogenetics:** AVA stimulation is scheduled before the target simulation-frame update; telemetry advances only when the game advances. Early game-over marks a pulse as incomplete. No-pulse controls are labeled distinctly.
- **Game metrics and causal logs:** Flappy records specific collision reason, score, frames, flap count, readout probability, feature coverage, and why each flap was triggered. Chemotaxis records food, reorientations, turn episodes, pirouettes, body length, feature coverage, last decoded turn, terminal reason, and logged turn triggers. The matched AVA optogenetic protocol also forces Neuron-only mode so interventions are evaluated through the untrained neuron-state action path. Neuron-only Flappy records per-frame motor-drive min/max/mean, threshold crossings, cooldown-blocked crossings, action-eligible crossings, and actual flap actions; its threshold is not changed without evidence.

## Controller modes

| Mode | Flappy action selection | Chemotaxis action selection | Interpretation |
|---|---|---|---|
| **Assisted baseline** | Direct bird-to-gap vertical error | Direct food-bearing steering | Reference policy; neural activity is telemetry only. Ablations should not materially change its actions. |
| **Neuron-only (default)** | Untrained baseline-adjusted DB−VB motor-neuron contrast | Untrained command/motor neuron-state asymmetry | No trained decoder or game-aware action override; virtual sensors and actuator mapping (including a fixed turn gain) remain model assumptions. |
| **Trained neural hybrid** | Trained 99-feature readout over frozen-connectome voltages | Trained 105-feature readout over frozen-connectome voltages plus ASER-associated adapter | Offline imitation readouts; not zero-shot behavior. |

The default *Neuron-only* mode uses untrained readouts of simulated neuron states and applies no game-state action policy, trained decoder, direct ASER-to-turn rule, or boundary-turn override. Synthetic sensory transduction and the mapping from motor-neuron activity to game actions are still engineered abstractions. This is a connectome-inspired simulation, not literal biological tissue or a validated whole-worm model. Assisted-mode performance is not evidence of neural control; the trained hybrid imitates the assisted policy.


### Graph metric definition

The small-world KPI is calculated once per dashboard render and the same result is reused in the top KPI and Graph Theory tab. It uses the **largest connected component** of the undirected union graph (chemical edges treated as undirected, plus undirected gap junctions). Clustering coefficient `C` and mean shortest-path length `L` are both measured on this same component. The null baseline is the mean of 20 **connected** Erdős–Rényi `G(n,p)` graphs with the component's node count and edge density, using seed `20261027`. The displayed index is `sigma = (C/C_random)/(L/L_random)`. The metric is deterministic for the included graph and seed; it is a structural comparison, not evidence of biological function or game performance. If the null calculation fails, the dashboard displays `N/A` rather than a hard-coded value.

### Numerical solver scope

The Python solver uses a **2 ms midpoint RK2 integration step**, a **30 ms membrane time constant**, voltages in **mV**, and model-scaled sensory/opto/noise inputs (not calibrated physical current units). Voltage is clipped to `[-80, 25] mV`. The voltage-dependent calcium-like inward current is a sigmoid conductance approximation. It has **no explicit intracellular calcium state or channel-gating kinetics**, so this implementation does not establish a biologically validated calcium plateau mechanism. A supplied solver seed makes its stochastic noise trajectory reproducible.

## Setup

Requires Python 3.10+ and Node.js only for the optional JavaScript data-collection scripts.

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

For a standalone browser view, open `web/index.html` from a local static server rather than relying on `file://` if the browser blocks assets. The Streamlit embedding is generated from the same `web/index.html`, CSS, JS, connectome JSON, and both decoder JSON files.

## Reproducible experiments

1. Restore all neurons, select a controller, set a seed, restart, and let the run finish.
2. Record the run and compare same-seed conditions. Apply ablations before restarting for a matched intervention trial.
3. Compare Neuron-only, trained neural hybrid, and assisted baseline. Do not treat assisted-mode behavior as neural output.
4. For ablation, record both the initial ablation set and the final set. A loss of behavior after targeted ablation can be a meaningful result; do not remove collision rules to conceal it.
5. For AVA stimulation, use the dedicated no-pulse and scheduled-pulse controls with identical seed, game, controller, and ablation set. Compare AVA voltage, motor contrast, action changes, and score over multiple seeds. A voltage response alone is not a behavioral effect.

### Chemotaxis metrics

- **Food:** collected targets; each food adds one body segment.
- **Reorientations:** heading changes generated by the selected controller; Neuron-only has no safety-guard turns.
- **Turn episodes:** sustained periods of continuous heading change; these are not the same as discrete reorientations.
- **Pirouettes:** ASER-associated stochastic reorientations only.
- **Terminal reason:** boundary collision, self-collision, or 1,800-frame time limit.

These are operational game metrics, not biological constants. Self-collision is part of this Snake-like task; early collision should be reported rather than silently suppressed.

## Readout calibration and reproducibility

The bundled Chemotaxis readout was trained on **13,898 frames from 20 seeds**, using a group split by seed (15 training seeds, 5 held-out seeds). On held-out teacher angular-correction labels, it achieved runtime-clipped MAE **0.0233 rad**, RMSE **0.0346 rad**, and R² **0.646**. These metrics evaluate the deployed ±0.045 rad/frame-clipped output against teacher turn labels; they are action-imitation metrics, not foraging-success metrics. Runtime turn output is conservatively clipped to ±0.045 rad/frame to reduce curvature-induced self-collision.

To regenerate Chemotaxis training data and its readout:

```powershell
node benchmark/collect_chemotaxis_decoder.js
python benchmark/train_chemotaxis_decoder.py
```

This creates `benchmark/chemotaxis_readout_training.jsonl` locally. That raw 78 MB training file is not bundled; the collector reproduces it from the included simulation. The scripts and trained `web/chemotaxis_decoder.json` / `.js` are included.

To retrain the Flappy readout, first run `node benchmark/collect_flappy_decoder.js`, then `python benchmark/train_flappy_decoder.py`. Both training workflows imitate the assisted policy; they do not train or modify connectome synaptic weights.

## Validation

Run:

```powershell
python -m unittest discover -s tests -v
node --check web/app.js
python -m py_compile app.py web/embed_helper.py
```

The bundled headless report (`benchmark/v13_headless_validation.json`) uses the six seeds `20261027`, `1`, `42`, `100`, `20261028`, and `123456`. On the trained-hybrid Chemotaxis mode without the former wall-turn override, scores were **1, 7, 16, 11, 9, 3** (mean **7.83**); one trial reached the time limit and five ended in boundary collision. In the default Neuron-only mode, all three tested seeds scored **0 food** and ended in boundary collision. This exposes a real limitation of the current untrained motor readout; it is not hidden with a safety heuristic. On Flappy, all six neural trials reached score **6** by the 800-frame harness cap; these were not run to natural game-over. The same-seed Assisted control produced identical metrics with and without AVA/AVD ablation, as expected because its action policy bypasses the network and environment RNG is isolated from neural noise. The small-world dashboard KPI is now computed from the seeded 20-graph baseline rather than a hard-coded number. Neuron-only Flappy runs record the untrained motor-drive range, mean, sample count, threshold crossings, and flap actions; the threshold is not tuned blindly to improve a score.

Automated tests check code contracts, data integrity, game metrics, ablation handling, and protocol bookkeeping. Headless simulation smoke tests are exploratory; neither automated tests nor held-out imitation metrics constitute biological validation or replace live-browser testing.

## Project structure

```text
neuro_elegans_final_v13_3/
├── app.py
├── connectome_builder.py
├── white_1986_whole.csv
├── requirements.txt
├── connectome/                 # loader, dynamics, sensorimotor mapping, ablation
├── games/                      # Flappy, Chemotaxis, Lunar Worm task modules
├── benchmark/
│   ├── collect_flappy_decoder.js
│   ├── train_flappy_decoder.py
│   ├── collect_chemotaxis_decoder.js
│   ├── train_chemotaxis_decoder.py
│   └── chemotaxis_v13_protocol.md
├── tests/
└── web/
    ├── index.html
    ├── app.js
    ├── connectome_data.json
    ├── flappy_decoder.json / .js
    ├── chemotaxis_decoder.json / .js
    └── embed_helper.py
```
