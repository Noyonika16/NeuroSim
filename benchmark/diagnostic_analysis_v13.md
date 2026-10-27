# NEURO-ELEGANS v13.1 — Diagnostic analysis and release notes

## Root causes identified

1. **Misleading Flappy HUD:** the canvas always printed an Assisted-baseline label, even when the Neural sensorimotor or Neuron-only mode was selected. This obscured which policy was actually active. The HUD now reports the selected controller and a specific collision reason.
2. **Ablation-induced decoder distribution shift:** a silenced neuron is clamped at the model's leak voltage. Feeding that voltage to a trained readout treated an artificial clamp as an observed sample and could saturate the output. Ablated decoder features are now omitted (zero standardized contribution), surviving network states remain in the readout, and feature coverage is recorded. This is not a guarantee that behavior survives a biologically meaningful ablation.
3. **Confounded random streams:** one RNG controlled neuron noise, policy reorientations, and pipe/food placement. Ablating a neuron changed how many noise draws were consumed and could therefore change future environments even when Assisted mode bypassed the network. v13 separates environment, policy, and network-noise streams and pre-draws noise by neuron index. Headless matched controls now show identical Assisted metrics with and without AVA/AVD ablation.
4. **Optogenetic frame accounting:** the scheduled pulse could be activated after the target frame's network update, and telemetry could advance while the game was stopped. The pulse is now scheduled before its target simulation update; telemetry advances only on simulation frames; early game-over is recorded as an incomplete pulse.
5. **Chemotaxis lacked directional local information and could turn in tight loops:** a scalar concentration derivative alone does not indicate which way to turn. v13 adds an explicitly synthetic bilateral head-side concentration probe (AWCL/AWCR input currents) alongside temporal ASEL/ASER coding, trains a turn readout from neural voltages, subtracts tonic biases in the untrained neuron-only readout, gates ASER reorientation on a falling signal, and the trained hybrid retains a disclosed ASER-associated adapter. Runtime decoding does not compute food bearing. Neuron-only removes the direct turning adapter and all boundary-turn overrides; wall proximity is only a sensory input.
6. **Ambiguous counters and records:** reorientation events, turn episodes, and pirouettes are recorded separately. Records include trial-start and record-time ablations, decoder feature coverage, seed, body length, and end reason.

## Current headless results

See [`v13_headless_validation.json`](v13_headless_validation.json). The harness executes the actual `web/app.js` simulation functions in Node, but does not render a browser UI. Six trained-hybrid Chemotaxis runs collected 1, 7, 16, 11, 9, and 3 food (mean 7.83); one reached the 1,800-frame limit and five ended in boundary collision. The three tested Neuron-only runs each scored 0 and ended in boundary collision, demonstrating that the untrained readout is currently ineffective for this task without the removed safety override. Six Flappy neural runs each reached score 6 by the harness's 800-frame cap; those Flappy trials were not run to natural game-over. The sensory CEP ablation caused immediate floor failure; AVA/AVD group ablation did not change the 800-frame Flappy score in this test, so it is a null behavioral result for that configuration rather than a claim of universal irrelevance.

The Chemotaxis readout was trained on 13,898 frames from 20 seeds (15 training seeds, 5 held-out seeds). With the deployed +/-0.045 rad/frame output clip, held-out action imitation metrics were MAE 0.0233 rad, RMSE 0.0346 rad, and R² 0.646. These metrics evaluate imitation of the assisted turn target, not foraging success or biological fidelity.

## Known limitations

- The neural readouts are trained to imitate the assisted policy; connectome synaptic weights are frozen, but the whole control system is not zero-shot.
- Sensory currents, bilateral virtual probe geometry, membrane parameters, output decoder, trained-hybrid ASER adapter, body mechanics, collision rules, and game physics are engineered abstractions.
- Some seeds still end in boundary or self-collision. Do not suppress these outcomes to make the score look better.
- Automated tests and headless results passed, but the live Streamlit browser has not been visually verified in this environment.


## v13.1 corrective pass

- Replaced the stale hard-coded top KPI (`3.42`) with the same deterministic small-world result used by the topology tab. The null baseline uses 20 connected ER graphs, seed `20261027`, and the largest connected component of the undirected union graph.
- Changed summary clustering to use that same largest connected component and documented how disconnected nodes and random nulls are handled.
- Replaced misleading model-size/FLOPs comparisons with an inventory that separates frozen connectome connections, trained imitation readouts, and untrained Neuron-only decoders.
- Corrected the solver description: it uses a 2 ms midpoint RK2 step and a voltage-only calcium-like inward current, not a fully validated calcium-plateau mechanism. Added optional seeded noise for reproducible Python solver trials.
- Added Neuron-only Flappy decoder distribution telemetry (minimum, maximum, mean, sample count, threshold crossings, and action count). The 0.180 threshold is intentionally not changed without examining these distributions and matched outcomes.
- Applied a consistent dark-navy/slate theme and responsive sidebar layout.
