"""Regression guards for the browser experiment UI's controller/measurement contract."""
from pathlib import Path
import json
import unittest

ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
HTML = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
DECODER = json.loads((ROOT / "web" / "flappy_decoder.json").read_text(encoding="utf-8"))
CHEMO_DECODER = json.loads((ROOT / "web" / "chemotaxis_decoder.json").read_text(encoding="utf-8"))


class TestBrowserExperimentContracts(unittest.TestCase):




    def test_experiment_records_explain_flap_and_turn_triggers(self):
        self.assertIn("actionLog: []", JS)
        self.assertIn("assisted direct bird-to-gap rule", JS)
        self.assertIn("trained voltage readout probability met threshold", JS)
        self.assertIn("untrained DB−VB motor drive exceeded threshold", JS)
        self.assertIn("ASER-associated stochastic pirouette after falling concentration", JS)
        self.assertIn("Action trigger log", JS)
        self.assertIn("actionLog: (isFlappy ? flappy.actionLog : chemotaxis.actionLog).slice()", JS)

    def test_strict_flappy_logs_full_motor_distribution_without_retuning(self):
        start = JS.index("function updateFlappyGame()")
        end = JS.index("function updateChemotaxisGame()", start)
        flappy = JS[start:end]
        self.assertIn("if (controllerMode === 'strict')", flappy)
        self.assertIn("flappy.motorDriveSamples++", flappy)
        self.assertIn("flappy.motorDriveMin", flappy)
        self.assertIn("flappy.motorDriveMax", flappy)
        self.assertIn("flappy.motorThresholdCrossings++", flappy)
        self.assertIn("motorThresholdCrossingsDuringCooldown", flappy)
        self.assertIn("motorThresholdCrossingsActionEligible", flappy)
        self.assertIn("const MOTOR_OUTPUT_THRESHOLD = 0.180;", flappy)
        self.assertIn("strictMotorDriveStats", JS)
        self.assertIn("Threshold unchanged pending distribution review", flappy)

    def test_solver_description_does_not_claim_validated_calcium_plateaus(self):
        app = (ROOT / "app.py").read_text(encoding="utf-8")
        engine = (ROOT / "connectome" / "biophysical_engine.py").read_text(encoding="utf-8")
        self.assertIn("does **not** include an intracellular calcium state", app)
        self.assertIn("not a validated plateau model", engine)
        self.assertIn("seed: Optional[int] = None", engine)

    def test_dashboard_small_world_kpi_is_computed_once_and_reused(self):
        app = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertNotIn("3.42", app)
        self.assertIn("small_world_display", app)
        self.assertIn("sw = small_world_stats", app)
        self.assertIn("n_random_samples=20, seed=20261027", app)
        self.assertIn("connected Erdős–Rényi G(n,p)", app)
        self.assertIn("No fallback value is displayed", app)

    def test_dashboard_distinguishes_frozen_synapses_from_trained_readouts(self):
        app = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("Backprop-Trained Connectome Weights", app)
        self.assertIn("Separate supervised action readouts are trained offline", app)
        self.assertIn("not a zero-shot controller", app)

    def test_streamlit_theme_has_high_contrast_widget_and_metric_styles(self):
        app = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("--primary-color: #22d3ee", app)
        self.assertIn("color: #67e8f9 !important", app)
        self.assertIn("[data-baseweb=\"tag\"]", app)
        self.assertNotIn("V13.1</span>", app)

    def test_dark_navy_theme_is_used_across_dashboard_and_webgl(self):
        css = (ROOT / "web" / "style.css").read_text(encoding="utf-8")
        self.assertIn("--bg-dark: #0b1220", css)
        self.assertIn("--panel-bg: rgba(15, 27, 46, 0.96)", css)
        self.assertIn("new THREE.Color(0x0b1220)", JS)

    def test_environment_policy_and_neural_noise_use_separate_seeded_streams(self):
        self.assertIn("let environmentRngState", JS)
        self.assertIn("let networkRngState", JS)
        self.assertIn("function environmentRandom01()", JS)
        self.assertIn("function networkRandom01()", JS)
        self.assertIn("noiseByNeuron[i] = (networkRandom01() - 0.5) * 0.4", JS)
        self.assertIn("gap_y: 38 + environmentRandom01() * 38", JS)
        self.assertIn("chemotaxis.food_x = 60 + environmentRandom01() * 300", JS)

    def test_seeded_random_stream_is_used_for_simulation(self):
        self.assertIn("function resetRandomStream()", JS)
        self.assertIn("function random01()", JS)
        self.assertIn('id="experiment-seed"', HTML)
        self.assertNotIn("Math.random()", JS)

    def test_assisted_and_neural_policies_are_explicitly_separated(self):
        self.assertIn("const targetHeading = Math.atan2(dy, dx);", JS)
        self.assertIn("if (controllerMode === 'assisted')", JS)
        self.assertIn("direct bird-to-gap error", JS)
        self.assertIn("directly steers toward the food bearing", HTML)
        self.assertIn("btn-controller-strict", HTML)
        self.assertIn("Neuron-only · no overrides", HTML)

    def test_trained_readout_uses_known_neuron_features_only_at_runtime(self):
        self.assertIn("function predictFlappyReadoutProbability()", JS)
        self.assertIn("const probability = predictFlappyReadoutProbability();", JS)
        self.assertIn("probability >= threshold", JS)
        self.assertEqual(len(DECODER["feature_names"]), len(DECODER["weights"]))
        self.assertEqual(len(DECODER["feature_names"]), len(DECODER["mean"]))
        self.assertEqual(len(DECODER["feature_names"]), len(DECODER["scale"]))
        self.assertLess(len(DECODER["feature_names"]), 302)
        self.assertTrue(DECODER["connectome_weights_frozen"])
        self.assertFalse(DECODER["runtime_game_state_access"])
        self.assertIn("trained to imitate the assisted", DECODER["notes"])

    def test_strict_connectome_decoder_is_retained_as_control(self):
        start = JS.index("function updateFlappyGame()")
        end = JS.index("function updateChemotaxisGame()", start)
        flappy = JS[start:end]
        self.assertIn("controllerMode === 'strict'", JS)
        self.assertIn("const MOTOR_OUTPUT_THRESHOLD = 0.180;", flappy)
        self.assertIn("Neuron-only · no overrides", HTML)

    def test_flappy_sensory_encoder_includes_edge_motion_without_action_label(self):
        start = JS.index("function updateFlappyGame()")
        end = JS.index("function updateChemotaxisGame()", start)
        flappy = JS[start:end]
        self.assertIn("const SENSOR_RANGE = 520.0", flappy)
        self.assertIn("upperClosingSpeed", flappy)
        self.assertIn("lowerClosingSpeed", flappy)
        self.assertIn("const floorThreat =", flappy)
        neural_branch = flappy[flappy.index("} else if (controllerMode === 'neural')"):flappy.index("} else {", flappy.index("} else if (controllerMode === 'neural')"))]
        self.assertNotIn("verticalError", neural_branch)
        self.assertNotIn("gapCenterY", neural_branch)
        self.assertNotIn("bird_vy", neural_branch)

    def test_chemotaxis_metrics_have_distinct_definitions(self):
        self.assertIn("chemotaxis.turnEpisodes++;", JS)
        self.assertIn("chemotaxis.pirouettes++;", JS)
        self.assertIn("chemotaxis.turns++;", JS)
        self.assertIn("REORIENTATIONS: ${chemotaxis.turns}", JS)
        self.assertIn("Pirouettes: ${r.pirouettes ?? 0}", JS)
        self.assertIn("turnCandidateFrames >= 3", JS)
        self.assertIn("turnStableFrames >= 8", JS)

    def test_chemotaxis_uses_lateral_asymmetry_not_constant_ava_avb_turning(self):
        self.assertIn("const commandTurn = ((neuronVoltage('AVAL') - neuronVoltage('AVAR')) - chemotaxis.turnBaselineCommand) / 180.0;", JS)
        self.assertIn("const motorBias = ((rightMotor - leftMotor) - chemotaxis.turnBaselineMotor) / 240.0;", JS)
        self.assertIn("const fallingGradient = dC < -0.00005;", JS)
        self.assertIn("controllerMode === 'neural' && Math.abs(neuralTurn) < 0.025 && aserResponseActive && fallingGradient", JS)
        self.assertNotIn("const commandTurn = (ava - avb) / 80.0;", JS)
        self.assertIn("chemotaxis.lastPirouetteFrame", JS)
        self.assertIn("const aserResponse = neuronVoltage('ASER') - chemotaxis.aserBaselineVoltage;", JS)

    def test_neuron_only_controller_has_no_game_aware_action_overrides(self):
        start = JS.index("function updateChemotaxisGame()")
        end = JS.index("function renderGameCanvas()", start)
        chemo = JS[start:end]
        self.assertIn("let controllerMode = 'strict'", JS)
        self.assertIn("id=\"btn-controller-strict\"", HTML)
        self.assertIn("Neuron-only · no overrides", HTML)
        self.assertIn("chemotaxis.heading += neuralTurn;", chemo)
        self.assertIn("5.0 * (commandTurn + motorBias)", chemo)
        self.assertNotIn("nearestEdge.distance < 16", chemo)
        self.assertNotIn("engineered boundary safety guard", chemo)
        self.assertNotIn("engineered wall-safety reflex", chemo)
        self.assertIn("const wallThreat =", chemo)  # sensory input only
        self.assertIn("injectCurrent(\"ASHL\"", chemo)
        self.assertIn("if (controllerMode === 'neural'", chemo)  # hybrid-only ASER adapter
        self.assertIn("No controller may silently override a motor command", chemo)

    def test_chemotaxis_has_explicit_termination_conditions(self):
        self.assertIn("chemotaxis.game_over_reason = 'Boundary collision';", JS)
        self.assertIn("chemotaxis.game_over_reason = 'Self-collision';", JS)
        self.assertIn("chemotaxis.maxFrames = 1800;", JS)
        self.assertIn("Time limit (1,800 frames)", JS)

    def test_food_increases_body_length_and_record_captures_length(self):
        self.assertIn("chemotaxis.body.push({", JS)
        self.assertIn("bodyLength: isFlappy ? null : chemotaxis.body.length", JS)
        self.assertIn("LENGTH: ${chemotaxis.body.length}", JS)

    def test_chemotaxis_body_following_preserves_spacing(self):
        self.assertIn("chemotaxis.pathHistory.unshift({ x: chemotaxis.worm_x, y: chemotaxis.worm_y });", JS)
        self.assertIn("const targetDistance = i * segmentSpacing;", JS)
        self.assertIn("for (let i = 8; i < chemotaxis.body.length; i++)", JS)
        self.assertNotIn("curr.x = prev.x - (dx / bdist) * 5.0", JS)

    def test_ablation_records_individual_neuron_names(self):
        self.assertIn("Single-neuron laser ablation:", JS)
        self.assertIn("ablatedNames: [...ablatedSet].map(i => neuronNames[i]).sort()", JS)
        self.assertIn('id="ablated-neuron-list"', HTML)

    def test_optogenetic_experiment_tracks_neural_motor_and_behavior(self):
        for field in ("baselineAvaVoltage", "peakAvaVoltage", "baselineMotor", "endMotor", "scoreDelta", "headingDeltaRad"):
            self.assertIn(field, JS)
        self.assertIn("Measuring AVA voltage → DB/VB motor contrast → behavioral response", JS)

    def test_record_captures_seed_and_intervention_snapshot(self):
        self.assertIn("seed: experimentSeed", JS)
        self.assertIn("interventionLog: interventionLog.slice()", JS)
        self.assertIn("ablated at record time", JS.lower())

    def test_controller_policy_and_scientific_limits_are_visible(self):
        self.assertIn("Controller modes: what is overridden?", HTML)
        self.assertIn("directly uses bird-to-gap vertical error", HTML)
        self.assertIn("99 sensory, command-interneuron, and motor-neuron voltages", HTML)
        self.assertIn("not zero-shot emergent behavior", HTML)
        self.assertIn("Neuron-only (default)", HTML)

    def test_matched_opto_protocol_is_frame_scheduled_and_seeded(self):
        self.assertIn("btn-controlled-control", HTML)
        self.assertIn("btn-controlled-pulse", HTML)
        self.assertIn("CONTROLLED_PULSE_FRAME = 10", JS)
        self.assertIn("OPTO_DURATION_FRAMES = 27", JS)
        self.assertIn("condition: withPulse ? 'matched AVA pulse' : 'matched no-pulse control'", JS)
        self.assertIn("optoExperiment.elapsedFrames >= optoExperiment.durationFrames", JS)
        self.assertIn("if (controllerMode !== 'strict') setControllerMode('strict');", JS)
        self.assertIn("controller: 'Neuron-only (no overrides)'", JS)

    def test_streamlit_embed_uses_same_html_and_inlines_decoder(self):
        helper = (ROOT / "web" / "embed_helper.py").read_text(encoding="utf-8")
        self.assertIn('"index.html"', helper)
        self.assertIn('"app.js"', helper)
        self.assertIn("window.CONNECTOME_DATA = ", helper)
        self.assertIn("window.FLAPPY_DECODER = ", helper)
        self.assertIn("window.CHEMOTAXIS_DECODER = ", helper)
        self.assertIn('replace(\'<script src="flappy_decoder.js"></script>\'', helper)
        self.assertNotIn("Self-contained HTML", helper)




    def test_neural_chemotaxis_does_not_compute_teacher_food_bearing(self):
        start = JS.index("function updateChemotaxisGame()")
        end = JS.index("// Render Retro Game Canvas", start)
        game = JS[start:end]
        teacher = game.index("if (controllerMode === 'assisted')")
        target = game.index("const targetHeading = Math.atan2(dy, dx);")
        neural = game.index("const learnedTurn = controllerMode === 'neural' ? predictChemotaxisTurn() : null;")
        self.assertLess(teacher, target)
        self.assertLess(target, neural)
        self.assertIn("Bilateral local-gradient probe", game)
        self.assertIn("local concentration samples", HTML)

    def test_chemotaxis_decoder_is_loaded_and_runtime_does_not_read_food_bearing(self):
        self.assertEqual(len(CHEMO_DECODER["feature_names"]), len(CHEMO_DECODER["weights"]))
        self.assertEqual(len(CHEMO_DECODER["feature_names"]), len(CHEMO_DECODER["mean"]))
        self.assertEqual(len(CHEMO_DECODER["feature_names"]), len(CHEMO_DECODER["scale"]))
        self.assertEqual(len(CHEMO_DECODER["feature_names"]), 105)
        self.assertTrue(CHEMO_DECODER["connectome_weights_frozen"])
        self.assertFalse(CHEMO_DECODER["runtime_food_bearing_access"])
        self.assertAlmostEqual(CHEMO_DECODER["runtime_turn_limit_radians"], 0.045)
        start = JS.index("function predictChemotaxisTurn()")
        end = JS.index("function currentGameFrame()", start)
        predictor = JS[start:end]
        self.assertIn("chemotaxisDecoder.feature_names", predictor)
        self.assertIn("if (ablatedSet.has(i)) continue;", predictor)
        self.assertNotIn("food_x", predictor)
        self.assertNotIn("Math.atan2", predictor)
        self.assertIn('injectCurrent("AWCL"', JS)
        self.assertIn('injectCurrent("AWCR"', JS)

    def test_embed_inlines_both_trained_readouts(self):
        helper = (ROOT / "web" / "embed_helper.py").read_text(encoding="utf-8")
        self.assertIn("chemotaxis_decoder.json", helper)
        self.assertIn("window.CHEMOTAXIS_DECODER = ", helper)
        self.assertIn('<script src="chemotaxis_decoder.js"></script>', HTML)

    def test_flappy_hud_reports_the_selected_controller(self):
        self.assertIn("controllerMode === 'assisted' ? 'ASSISTED BASELINE · DIRECT GAME-STATE CONTROL'", JS)
        self.assertIn("'NEURON-ONLY · NO TRAINED READOUT / NO OVERRIDES'", JS)
        self.assertIn("'NEURAL SENSORIMOTOR · TRAINED READOUT'", JS)
        self.assertNotIn("status.textContent=flappy.is_game_over?'RUN OVER · PRESS RESTART':'ASSISTED BASELINE · NEURAL TELEMETRY'", JS)

    def test_ablation_masks_missing_features_in_trained_decoder(self):
        self.assertIn("if (ablatedSet.has(i)) continue;", JS)
        self.assertIn("flappy.decoderFeatureCoverage = activeFeatures / flappyDecoder.feature_names.length;", JS)
        self.assertIn("decoderFeatureCoverage:", JS)
        self.assertIn("Ablated at trial start", JS)
        self.assertIn("Ablated at record time", JS)

    def test_motor_population_readout_excludes_ablated_cells(self):
        start = JS.index("function getPrefixMean(prefix)")
        end = JS.index("// Trained linear readout", start)
        fn = JS[start:end]
        self.assertIn("n.startsWith(prefix) && !ablatedSet.has(i)", fn)

    def test_optogenetic_telemetry_advances_only_on_simulation_frames(self):
        self.assertIn("if (frameAfter > frameBefore) updateOptogeneticTelemetry();", JS)
        self.assertIn("frameBefore + 1 >= controlledProtocol.pulseAtFrame", JS)
        self.assertIn("controlledProtocol.pulseAttempted = true;", JS)
        self.assertIn("game ended before the scheduled pulse duration completed", JS)

    def test_flappy_collision_reason_is_specific(self):
        for reason in ("Ceiling collision", "Floor collision", "Pipe collision"):
            self.assertIn(f"flappy.game_over_reason = '{reason}'", JS)

    def test_controlled_no_pulse_is_not_misreported_as_missing_data(self):
        self.assertIn("Matched no-pulse control (no stimulation scheduled)", JS)
        self.assertIn("Pulse scheduled but never started (run ended before trigger)", JS)


if __name__ == "__main__":
    unittest.main()
