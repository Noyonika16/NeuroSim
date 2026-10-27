/**
 * NEURO-ELEGANS: 3D Connectome & Cybernetic Game Engine
 * Runs a simplified connectome-based model with 302 neuron states in real time.
 */

// Global State
let connectomeData = null;
let flappyDecoder = (typeof window !== 'undefined' ? window.FLAPPY_DECODER : null);
let chemotaxisDecoder = (typeof window !== 'undefined' ? window.CHEMOTAXIS_DECODER : null);
let scene, camera, renderer;
let neuronMeshes = [];
let synapseLines = null;
let raycaster, mouse;
let hoveredNeuron = null;
let selectedNeuron = null;

// Biophysical Engine State in Browser (60 FPS)
const NUM_NEURONS = 302;
let neuronNames = [];
let nameToIdx = {};
let V = new Float32Array(NUM_NEURONS);
let I_sensory = new Float32Array(NUM_NEURONS);
let I_opto = new Float32Array(NUM_NEURONS);
let ablatedSet = new Set();
let optoActiveSet = new Map(); // name -> { type, current }

// Connectivity
let W_chem = null; // Float32Array (N * N)
let E_syn = null;  // Float32Array (N * N)
let L_gap = null;  // Float32Array (N * N)

// Biophysical Parameters
const TAU = 0.030; // 30ms time constant
const E_LEAK = -60.0;
const SYN_THETA = -45.0;
const SYN_BETA = 4.5;
const G_CHEM_SCALE = 0.028;
const G_GAP_SCALE = 0.018;

// Neuromodulation
let neuromod = { dopamine: 0.0, serotonin: 0.0 };

// Oscilloscope History
const SCOPE_HISTORY = 180;
let scopeTraces = {
  "AVAL": new Float32Array(SCOPE_HISTORY).fill(E_LEAK),
  "AVBL": new Float32Array(SCOPE_HISTORY).fill(E_LEAK),
  "ASHL": new Float32Array(SCOPE_HISTORY).fill(E_LEAK),
  "DB1": new Float32Array(SCOPE_HISTORY).fill(E_LEAK),
  "VB1": new Float32Array(SCOPE_HISTORY).fill(E_LEAK)
};

// Game mode and controller mode are separate so experiments can compare a
// transparent assisted reference against connectome-output-driven control.
let currentGameMode = 'flappy';
let controllerMode = 'strict'; // assisted, neural (trained hybrid), or strict (neuron-state-only output; no trained readout or action overrides)
let lastIntervention = 'None';
let optoTimeout = null;
let experimentRecords = [];
let runStartedAt = Date.now();
let experimentSeed = 20261027;
let rngState = experimentSeed >>> 0; // policy/stochastic-reflex stream
let environmentRngState = (experimentSeed ^ 0x9E3779B9) >>> 0;
let networkRngState = (experimentSeed ^ 0x85EBCA6B) >>> 0;
let optoExperiment = null;
let interventionLog = [];
let controlledProtocol = null; // paired no-pulse vs scheduled-pulse experiment
const CONTROLLED_PULSE_FRAME = 10; // Must match the visible protocol description
const OPTO_DURATION_FRAMES = 27; // ~450 ms at 60 FPS; simulation-frame based, not wall-clock based

// Independent seeded streams keep environment randomness, stochastic policy
// events, and neural thermal noise from perturbing one another. In particular,
// ablating neurons must not silently change future pipe/food placement.
function resetRandomStream() {
  const input = document.getElementById('experiment-seed');
  const parsed = Number.parseInt(input?.value ?? String(experimentSeed), 10);
  experimentSeed = Number.isFinite(parsed) ? (parsed >>> 0) : 20261027;
  if (input) input.value = String(experimentSeed);
  const seed = experimentSeed || 1;
  rngState = seed;
  environmentRngState = (seed ^ 0x9E3779B9) >>> 0 || 1;
  networkRngState = (seed ^ 0x85EBCA6B) >>> 0 || 1;
}
function random01() {
  rngState = (1664525 * rngState + 1013904223) >>> 0;
  return rngState / 4294967296;
}
function environmentRandom01() {
  environmentRngState = (1664525 * environmentRngState + 1013904223) >>> 0;
  return environmentRngState / 4294967296;
}
function networkRandom01() {
  networkRngState = (1664525 * networkRngState + 1013904223) >>> 0;
  return networkRngState / 4294967296;
}
function wrapAngle(angle) {
  while (angle > Math.PI) angle -= 2 * Math.PI;
  while (angle < -Math.PI) angle += 2 * Math.PI;
  return angle;
}
function neuronVoltage(name) {
  const idx = nameToIdx[name];
  return idx === undefined ? E_LEAK : V[idx];
}
function activeAvaVoltage() {
  const values = ['AVAL', 'AVAR'].filter(n => nameToIdx[n] !== undefined && !ablatedSet.has(nameToIdx[n])).map(neuronVoltage);
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : E_LEAK;
}
function motorDriveSnapshot() {
  return { db: getPrefixMean('DB'), vb: getPrefixMean('VB'), smddl: neuronVoltage('SMDDL'), smddr: neuronVoltage('SMDDR') };
}
function recordIntervention(message) {
  lastIntervention = message;
  interventionLog.push({ at: Date.now(), message, ablated: [...ablatedSet].map(i => neuronNames[i]).sort() });
  updateAblatedListUI();
  const note = document.getElementById('controller-note');
  if (note) note.textContent = `${message}. Active ablation set (${ablatedSet.size}): ${[...ablatedSet].map(i => neuronNames[i]).sort().join(', ') || 'none'}.`;
}

// --- Flappy Nematode Game State ---
let flappy = {
  bird_x: 75,
  bird_y: 120,
  bird_vy: 0,
  gravity: 0.22,
  flap_strength: -4.2,
  pipe_w: 50,
  pipe_gap: 126,
  pipe_speed: 2.0,
  pipes: [],
  score: 0,
  high_score: 0,
  is_game_over: false,
  flap_cooldown: 0,
  frames: 0,
  flap_count: 0,
  neuralFlapDecisions: 0,
  neuralMotorBaseline: 0,
  lastNeuralDrive: 0,
  lastReadoutProbability: null,
  peakAbsNeuralDrive: 0,
  motorDriveSum: 0,
  motorDriveSamples: 0,
  motorDriveMin: null,
  motorDriveMax: null,
  motorThresholdCrossings: 0,
  motorThresholdCrossingsDuringCooldown: 0,
  motorThresholdCrossingsActionEligible: 0,
  actionLog: [],
  decoderFeatureCoverage: 1.0,
  ablatedAtStartNames: [],
  game_over_reason: '',
  body_segments: []
};

// --- Chemotaxis Snake State ---
let chemotaxis = {
  worm_x: 210,
  worm_y: 120,
  heading: 0,
  food_x: 100,
  food_y: 80,
  score: 0,
  frames: 0,
  pirouettes: 0,
  turns: 0,                 // completed reorientation events, counted at most once per episode
  turnEpisodes: 0,          // continuous periods of sustained heading change
  turnAngleAccumulator: 0,
  turnCandidateFrames: 0,
  turnStableFrames: 0,
  turnEventCounted: false,
  turning: false,
  actionLog: [],
  aserResponseActive: false,
  aserBaselineVoltage: E_LEAK,
  lastPirouetteFrame: -1000,
  lastWallTurnFrame: -1000,
  turnBaselineCommand: 0,
  turnBaselineMotor: 0,
  ablatedAtStartNames: [],
  decoderFeatureCoverage: 1.0,
  lastDecoderTurn: 0,
  prev_conc: 0,
  body: [],
  pathHistory: [], // head trajectory used to place body segments at fixed arc-length intervals
  is_game_over: false,
  game_over_reason: '',
  maxFrames: 1800
};

// Colors by Functional Class
const CLASS_COLORS = {
  sensory: 0x00f2fe,     // Cyan
  command: 0xf857a6,     // Magenta
  motor: 0x00ff87,       // Lime Green
  interneuron: 0xffb300  // Amber
};

// Initialize Application
async function init() {
  if (window.CONNECTOME_DATA) {
    connectomeData = window.CONNECTOME_DATA;
  } else {
    try {
      const resp = await fetch('connectome_data.json');
      connectomeData = await resp.json();
    } catch (e) {
      console.error("Failed to load connectome JSON:", e);
      return;
    }
  }

  setupBiophysicalMatrices();
  initThreeJS();
  initGames();
  setupUIEventListeners();
  animate();
}

function setupBiophysicalMatrices() {
  neuronNames = Object.keys(connectomeData.neurons).sort();
  if (flappyDecoder && Array.isArray(flappyDecoder.feature_names) &&
      flappyDecoder.feature_names.some(name => !Object.prototype.hasOwnProperty.call(connectomeData.neurons, name))) {
    console.error('Flappy decoder references unknown neuron features; trained readout disabled.');
    flappyDecoder = null;
  }
  if (chemotaxisDecoder && Array.isArray(chemotaxisDecoder.feature_names) &&
      chemotaxisDecoder.feature_names.some(name => !Object.prototype.hasOwnProperty.call(connectomeData.neurons, name))) {
    console.error('Chemotaxis decoder references unknown neuron features; trained readout disabled.');
    chemotaxisDecoder = null;
  }
  neuronNames.forEach((name, i) => {
    nameToIdx[name] = i;
  });

  const N = neuronNames.length;
  W_chem = new Float32Array(N * N);
  E_syn = new Float32Array(N * N);
  L_gap = new Float32Array(N * N);

  // Chemical Synapses
  connectomeData.chemical_synapses.forEach(syn => {
    const pre = nameToIdx[syn.pre];
    const post = nameToIdx[syn.post];
    if (pre !== undefined && post !== undefined) {
      W_chem[pre * N + post] += syn.weight * G_CHEM_SCALE;
      E_syn[pre * N + post] = syn.esyn;
    }
  });

  // Gap Junctions
  connectomeData.gap_junctions.forEach(gap => {
    const i = nameToIdx[gap.n1];
    const j = nameToIdx[gap.n2];
    if (i !== undefined && j !== undefined) {
      const g = gap.weight * G_GAP_SCALE;
      L_gap[i * N + j] += g;
      L_gap[j * N + i] += g;
    }
  });

  // Gap Laplacian Diagonals
  for (let i = 0; i < N; i++) {
    let rowSum = 0;
    for (let j = 0; j < N; j++) {
      rowSum += L_gap[i * N + j];
    }
    L_gap[i * N + i] = -rowSum;
  }

  V.fill(E_LEAK);
}

// --- Three.js 3D Connectome Visualization ---
function initThreeJS() {
  const container = document.getElementById('viewport-container');
  const width = container.clientWidth;
  const height = container.clientHeight;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b1220);
  scene.fog = new THREE.FogExp2(0x0b1220, 0.0012);

  camera = new THREE.PerspectiveCamera(45, width / height, 1, 3000);
  camera.position.set(0, 150, 450);

  renderer = new THREE.WebGLRenderer({ canvas: document.getElementById('three-canvas'), antialias: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  // Controls
  if (typeof THREE.OrbitControls !== 'undefined') {
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.target.set(0, 0, 100);
  }

  // Lights
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
  scene.add(ambientLight);
  const dirLight = new THREE.DirectionalLight(0x00f2fe, 1.2);
  dirLight.position.set(100, 200, 200);
  scene.add(dirLight);

  // Nematode Body Outline Mesh / Spline
  createNematodeCuticle();

  // 3D Neurons
  createNeuronNodes();

  // Synapse Fiber Network
  createSynapseFibers();

  // Raycaster for hover/clicks
  raycaster = new THREE.Raycaster();
  mouse = new THREE.Vector2();

  window.addEventListener('resize', onWindowResize);
  renderer.domElement.addEventListener('mousemove', onMouseMove);
  renderer.domElement.addEventListener('click', onCanvasClick);
}

function createNematodeCuticle() {
  // Translucent nematode glass cuticle to frame the 302 neurons
  const points = [];
  for (let z = 0; z <= 1000; z += 50) {
    // Body width profile: tapered at head (z=0) and tail (z=1000), widest midbody
    const frac = z / 1000;
    const r = 32 * Math.sin(Math.PI * Math.pow(frac, 0.65)) + 4;
    points.push(new THREE.Vector3(0, 0, (z - 500) * 0.7));
  }
  const curve = new THREE.CatmullRomCurve3(points);
  const tubeGeo = new THREE.TubeGeometry(curve, 60, 28, 16, false);
  const tubeMat = new THREE.MeshBasicMaterial({
    color: 0x1e293b,
    transparent: true,
    opacity: 0.12,
    wireframe: true
  });
  const cuticle = new THREE.Mesh(tubeGeo, tubeMat);
  scene.add(cuticle);
}

function createNeuronNodes() {
  const sphereGeo = new THREE.SphereGeometry(2.4, 16, 16);

  neuronNames.forEach((name, idx) => {
    const n = connectomeData.neurons[name];
    const nClass = n.type || 'interneuron';
    const baseColor = CLASS_COLORS[nClass] || 0xffb300;

    const mat = new THREE.MeshStandardMaterial({
      color: baseColor,
      emissive: baseColor,
      emissiveIntensity: 0.35,
      roughness: 0.2,
      metalness: 0.8
    });

    const mesh = new THREE.Mesh(sphereGeo, mat);
    // Transform coordinates: C. elegans coordinates (x, y, z um)
    // Scale and center along z axis
    mesh.position.set(n.x * 2.2, n.y * 2.2, (n.z - 500) * 0.7);
    mesh.userData = { name: name, idx: idx, data: n, baseColor: baseColor };

    scene.add(mesh);
    neuronMeshes.push(mesh);
  });
}

function createSynapseFibers() {
  // Render subset of key synapses as glowing lines
  const lineMat = new THREE.LineBasicMaterial({
    color: 0x334155,
    transparent: true,
    opacity: 0.22,
    blending: THREE.AdditiveBlending
  });

  const positions = [];
  const N = neuronNames.length;

  connectomeData.chemical_synapses.slice(0, 800).forEach(syn => {
    const p1 = neuronMeshes[nameToIdx[syn.pre]];
    const p2 = neuronMeshes[nameToIdx[syn.post]];
    if (p1 && p2) {
      positions.push(p1.position.x, p1.position.y, p1.position.z);
      positions.push(p2.position.x, p2.position.y, p2.position.z);
    }
  });

  const lineGeo = new THREE.BufferGeometry();
  lineGeo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  synapseLines = new THREE.LineSegments(lineGeo, lineMat);
  scene.add(synapseLines);
}

function onWindowResize() {
  const container = document.getElementById('viewport-container');
  const width = container.clientWidth;
  const height = container.clientHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
}

function onMouseMove(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObjects(neuronMeshes);

  const tooltip = document.getElementById('neuron-tooltip');
  if (intersects.length > 0) {
    const hit = intersects[0].object;
    hoveredNeuron = hit.userData;
    
    tooltip.style.display = 'block';
    tooltip.style.left = `${event.clientX + 15}px`;
    tooltip.style.top = `${event.clientY - 15}px`;
    
    document.getElementById('tt-name').textContent = hoveredNeuron.name;
    document.getElementById('tt-type').textContent = hoveredNeuron.data.type.toUpperCase();
    document.getElementById('tt-nt').textContent = hoveredNeuron.data.transmitter;
    document.getElementById('tt-voltage').textContent = `${V[hoveredNeuron.idx].toFixed(1)} mV`;
    document.getElementById('tt-status').textContent = ablatedSet.has(hoveredNeuron.idx) ? 'ABLATED (SILENT)' : 'ACTIVE';
  } else {
    hoveredNeuron = null;
    tooltip.style.display = 'none';
  }
}

function onCanvasClick() {
  if (hoveredNeuron) {
    const idx = hoveredNeuron.idx;
    if (ablatedSet.has(idx)) {
      ablatedSet.delete(idx);
      recordIntervention(`Single-neuron restore: ${hoveredNeuron.name}`);
    } else {
      ablatedSet.add(idx);
      recordIntervention(`Single-neuron laser ablation: ${hoveredNeuron.name}`);
    }
  }
}

// --- High-Speed In-Browser Biophysical Step ---
function stepBiophysicalNetwork(dt = 0.002) {
  const N = NUM_NEURONS;
  
  // 1. Presynaptic activations
  const s_pre = new Float32Array(N);
  const seroFactor = 1.0 + 0.35 * neuromod.serotonin - 0.20 * neuromod.dopamine;
  
  for (let i = 0; i < N; i++) {
    if (ablatedSet.has(i)) {
      s_pre[i] = 0;
    } else {
      const expVal = Math.exp(-Math.max(-20, Math.min(20, (V[i] - SYN_THETA) / SYN_BETA)));
      s_pre[i] = (1.0 / (1.0 + expVal)) * seroFactor;
    }
  }

  // 2. Compute Currents & Updates. Pre-draw noise for every neuron before
  // applying ablation so intervention status cannot shift other cells' noise.
  const noiseByNeuron = new Float32Array(N);
  for (let i = 0; i < N; i++) noiseByNeuron[i] = (networkRandom01() - 0.5) * 0.4;
  for (let i = 0; i < N; i++) {
    if (ablatedSet.has(i)) {
      V[i] = E_LEAK;
      continue;
    }

    // Passive Leak
    let dV = -(V[i] - E_LEAK);

    // Gap Junction Laplacian
    for (let j = 0; j < N; j++) {
      const g = L_gap[i * N + j];
      if (g !== 0) {
        dV += g * V[j];
      }
    }

    // Chemical Synaptic Transmission
    for (let pre = 0; pre < N; pre++) {
      const w = W_chem[pre * N + i];
      if (w > 0) {
        const esyn = E_syn[pre * N + i];
        dV += w * s_pre[pre] * (esyn - V[i]);
      }
    }

    // Injected currents (sensory + optogenetic + thermal noise)
    const noise = noiseByNeuron[i];
    const totalCurrent = dV + I_sensory[i] + I_opto[i] + noise;

    V[i] += (totalCurrent / TAU) * dt;

    // Biological voltage clamp bounds (-80mV to +25mV)
    if (V[i] < -80.0) V[i] = -80.0;
    if (V[i] > 25.0) V[i] = 25.0;
  }
}


// Warm the resting network briefly with all external currents cleared, then
// capture its tonic DB-VB motor bias. This is an empirical model baseline, not
// a biologically measured resting potential. Paired trials use the same seed.
function primeNetworkMotorBaseline() {
  I_sensory.fill(0);
  I_opto.fill(0);
  for (let i = 0; i < 80; i++) stepBiophysicalNetwork(0.002);
  flappy.neuralMotorBaseline = (getPrefixMean("DB") - getPrefixMean("VB")) / 10.0;
  flappy.lastNeuralDrive = 0;
  flappy.lastReadoutProbability = null;
  flappy.peakAbsNeuralDrive = 0;
  flappy.motorDriveSum = 0;
  flappy.motorDriveSamples = 0;
  flappy.motorDriveMin = null;
  flappy.motorDriveMax = null;
  flappy.motorThresholdCrossings = 0;
}

// --- Retro Game Loops ---
function initGames() {
  resetFlappy();
  resetChemotaxis();
}

function resetNeuralStateForTrial() {
  // End and discard a pulse from the previous trial before clearing currents.
  if (optoTimeout) { clearTimeout(optoTimeout); optoTimeout = null; }
  if (optoExperiment && !optoExperiment.endedAt) finishOptogeneticPulse('trial reset before pulse duration completed');
  optoExperiment = null;
  // Keep ablation selections intact but remove residual membrane state and
  // stimulation currents so restarted runs begin from a comparable baseline.
  V.fill(E_LEAK);
  I_sensory.fill(0);
  I_opto.fill(0);
  Object.values(scopeTraces).forEach(trace => trace.fill(E_LEAK));
}

function resetFlappy() {
  controlledProtocol = null;
  resetRandomStream();
  resetNeuralStateForTrial();
  // Establish a trial-specific motor baseline with no sensory/opto input. This
  // prevents the decoder from mistaking tonic network bias for a new motor command.
  primeNetworkMotorBaseline();
  runStartedAt = Date.now();
  flappy.bird_x = 75;
  flappy.bird_y = 120;
  flappy.bird_vy = 0;
  flappy.score = 0;
  flappy.frames = 0;
  flappy.flap_count = 0;
  flappy.neuralFlapDecisions = 0;
  flappy.lastNeuralDrive = 0;
  flappy.lastReadoutProbability = null;
  flappy.peakAbsNeuralDrive = 0;
  flappy.motorDriveSum = 0;
  flappy.motorDriveSamples = 0;
  flappy.motorDriveMin = null;
  flappy.motorDriveMax = null;
  flappy.motorThresholdCrossings = 0;
  flappy.motorThresholdCrossingsDuringCooldown = 0;
  flappy.motorThresholdCrossingsActionEligible = 0;
  flappy.actionLog = [];
  flappy.decoderFeatureCoverage = 1.0;
  flappy.ablatedAtStartNames = [...ablatedSet].map(i => neuronNames[i]).sort();
  flappy.flap_cooldown = 0;
  flappy.is_game_over = false;
  flappy.game_over_reason = '';
  flappy.pipes = [
    { x: 450, gap_y: 38 + environmentRandom01() * 38, passed: false },
    { x: 680, gap_y: 38 + environmentRandom01() * 38, passed: false }
  ];
  flappy.body_segments = [];
  for (let i = 0; i < 12; i++) {
    flappy.body_segments.push({ x: flappy.bird_x - i * 5, y: flappy.bird_y });
  }
}

function resetChemotaxis() {
  controlledProtocol = null;
  resetRandomStream();
  resetNeuralStateForTrial();
  // Settle the network without sensory drive so the ASER response is measured
  // relative to a trial-specific resting baseline, not the initial transient.
  I_sensory.fill(0);
  I_opto.fill(0);
  for (let i = 0; i < 80; i++) stepBiophysicalNetwork(0.002);
  chemotaxis.aserBaselineVoltage = neuronVoltage('ASER');
  // Center lateral motor/command asymmetry around each trial's own resting bias.
  // Otherwise a tonic left-right voltage offset accumulates into a tight circle.
  chemotaxis.turnBaselineCommand = neuronVoltage('AVAL') - neuronVoltage('AVAR');
  chemotaxis.turnBaselineMotor = neuronVoltage('SMDDR') - neuronVoltage('SMDDL');
  chemotaxis.decoderFeatureCoverage = 1.0;
  chemotaxis.lastDecoderTurn = 0;
  chemotaxis.ablatedAtStartNames = [...ablatedSet].map(i => neuronNames[i]).sort();
  chemotaxis.lastPirouetteFrame = -1000;
  chemotaxis.lastWallTurnFrame = -1000;
  runStartedAt = Date.now();
  chemotaxis.worm_x = 210;
  chemotaxis.worm_y = 120;
  chemotaxis.heading = 0;
  chemotaxis.score = 0;
  chemotaxis.frames = 0;
  chemotaxis.pirouettes = 0;
  chemotaxis.turns = 0;
  chemotaxis.turnEpisodes = 0;
  chemotaxis.turnAngleAccumulator = 0;
  chemotaxis.turnCandidateFrames = 0;
  chemotaxis.turnStableFrames = 0;
  chemotaxis.turnEventCounted = false;
  chemotaxis.turning = false;
  chemotaxis.actionLog = [];
  chemotaxis.aserResponseActive = false;
  chemotaxis.lastPirouetteFrame = -1000;
  chemotaxis.lastWallTurnFrame = -1000;
  chemotaxis.is_game_over = false;
  chemotaxis.game_over_reason = '';
  chemotaxis.maxFrames = 1800;
  chemotaxis.food_x = 90 + environmentRandom01() * 240;
  chemotaxis.food_y = 45 + environmentRandom01() * 150;
  chemotaxis.prev_conc = Math.exp(-(((chemotaxis.food_x - chemotaxis.worm_x) ** 2) + ((chemotaxis.food_y - chemotaxis.worm_y) ** 2)) / (2.0 * 80.0 * 80.0));
  chemotaxis.body = [];
  for (let i = 0; i < 16; i++) {
    chemotaxis.body.push({ x: chemotaxis.worm_x - i * 5, y: chemotaxis.worm_y });
  }
  // Store the head path at approximately one-pixel initial intervals. Body
  // segments are sampled from this trajectory by arc length, as in a proper
  // snake trail, instead of snapping each segment toward the head every frame.
  chemotaxis.pathHistory = [];
  for (let d = 0; d <= (chemotaxis.body.length + 5) * 5; d += 1) {
    chemotaxis.pathHistory.push({ x: chemotaxis.worm_x - d, y: chemotaxis.worm_y });
  }
}

function updateFlappyGame() {
  // Keep the final state visible until the user explicitly restarts.
  if (flappy.is_game_over) return;
  flappy.frames++;

  // Find upcoming pipe
  let pipe = flappy.pipes[0];
  for (let p of flappy.pipes) {
    if (p.x + flappy.pipe_w > flappy.bird_x - 12) {
      pipe = p;
      break;
    }
  }

  const pipeDistX = pipe.x - flappy.bird_x;

  // Explicit, synthetic obstacle-sensing model. The encoder represents local
  // obstacle proximity/clearance; it does NOT label the desired action as
  // "flap" or directly feed the gap-centering error into the action decoder.
  // This is an engineered sensory abstraction, not a claim that C. elegans sees.
  I_sensory.fill(0);
  const SENSOR_RANGE = 520.0; // start sensing before gravity alone can end the trial
  const proximity = Math.max(0, Math.min(1, 1.0 - (pipeDistX / SENSOR_RANGE)));
  const ashCurrent = 42.0 * Math.pow(proximity, 1.25);
  injectCurrent("ASHL", ashCurrent);
  injectCurrent("ASHR", ashCurrent);

  // Encode the clearance to each pipe edge separately. These are sensory
  // observations, not a target-centering error passed directly to the decoder.
  const upperClearance = flappy.bird_y - pipe.gap_y;
  const lowerClearance = (pipe.gap_y + flappy.pipe_gap) - flappy.bird_y;
  const EDGE_SENSING_DISTANCE = 62.0;
  // Add a motion-sensitive component: an edge is more urgent when the bird
  // is moving toward it. This encodes measured relative motion into sensory
  // currents; it does not select an action or pass a target action to the decoder.
  const upperClosingSpeed = Math.max(0, -flappy.bird_vy);
  const lowerClosingSpeed = Math.max(0, flappy.bird_vy);
  const upperThreat = Math.max(0, Math.min(1,
    (EDGE_SENSING_DISTANCE - upperClearance + 5.0 * upperClosingSpeed) / EDGE_SENSING_DISTANCE));
  const lowerThreat = Math.max(0, Math.min(1,
    (EDGE_SENSING_DISTANCE - lowerClearance + 5.0 * lowerClosingSpeed) / EDGE_SENSING_DISTANCE));
  const approachGain = 0.25 + 0.75 * proximity;
  // Dorsal/ventral CEP channels are used as synthetic directional mechanosensory
  // channels. A stronger current means a nearer edge, not "the correct action".
  injectCurrent("CEPDL", 48.0 * upperThreat * approachGain);
  injectCurrent("CEPDR", 48.0 * upperThreat * approachGain);
  injectCurrent("CEPVL", 48.0 * lowerThreat * approachGain);
  injectCurrent("CEPVR", 48.0 * lowerThreat * approachGain);

  // A floor-proximity/touch signal provides an early warning before the game
  // boundary. This prevents the first trial from ending before the first pipe
  // reaches the bird; it remains an input to the network, not an action rule.
  const floorThreat = Math.max(0, Math.min(1, (flappy.bird_y - 145.0) / 70.0));
  if (floorThreat > 0) {
    injectCurrent("ALML", 55.0 * floorThreat);
    injectCurrent("ALMR", 55.0 * floorThreat);
    injectCurrent("AVM", 35.0 * floorThreat);
  }

  // Step connectome (4 sub-steps)
  for (let s = 0; s < 4; s++) {
    stepBiophysicalNetwork(0.002);
  }

  // Motor Readout (DB vs VB and AVB)
  const dbMean = getPrefixMean("DB");
  const vbMean = getPrefixMean("VB");
  const vAvb = V[nameToIdx["AVBL"] || 0];
  const flapDrive = (dbMean - vbMean) + 0.15 * (vAvb - (-55.0));

  // Collect the Neuron-only decoder's motor-output distribution on every simulation
  // frame (including action-cooldown frames), before applying its threshold.
  // This is diagnostic telemetry only; it does not alter the policy.
  if (controllerMode === 'strict') {
    const observedDrive = (dbMean - vbMean) / 10.0 - flappy.neuralMotorBaseline;
    flappy.lastNeuralDrive = observedDrive;
    flappy.motorDriveSum += observedDrive;
    flappy.motorDriveSamples++;
    flappy.motorDriveMin = flappy.motorDriveMin === null ? observedDrive : Math.min(flappy.motorDriveMin, observedDrive);
    flappy.motorDriveMax = flappy.motorDriveMax === null ? observedDrive : Math.max(flappy.motorDriveMax, observedDrive);
    if (observedDrive > 0.180) {
      flappy.motorThresholdCrossings++;
      if (flappy.flap_cooldown > 0) flappy.motorThresholdCrossingsDuringCooldown++;
      else flappy.motorThresholdCrossingsActionEligible++;
    }
    flappy.peakAbsNeuralDrive = Math.max(flappy.peakAbsNeuralDrive, Math.abs(observedDrive));
  }

  // Update meter HUD
  updateGauges(ashCurrent, flapDrive, dbMean, vbMean);
  if ((controllerMode === 'neural' || controllerMode === 'strict') && flappy.frames % 45 === 0) {
    const note = document.getElementById('controller-note');
    if (controllerMode === 'neural') {
      const p = flappy.lastReadoutProbability;
      if (note) note.textContent = `Trained neural readout · probability of flap: ${p == null ? 'decoder unavailable' : p.toFixed(3)} · active decoder features: ${(100 * flappy.decoderFeatureCoverage).toFixed(0)}% · trained on assisted-policy labels; runtime uses neuron voltages only. Ablated features are omitted.`;
    } else {
      const motorDelta = (dbMean - vbMean) / 10.0 - flappy.neuralMotorBaseline;
      const meanDrive = flappy.motorDriveSamples ? flappy.motorDriveSum / flappy.motorDriveSamples : 0;
      const minDrive = flappy.motorDriveMin === null ? 0 : flappy.motorDriveMin;
      const maxDrive = flappy.motorDriveMax === null ? 0 : flappy.motorDriveMax;
      if (note) note.textContent = `Neuron-only diagnostics · DB−VB drive now ${motorDelta.toFixed(3)}; range [${minDrive.toFixed(3)}, ${maxDrive.toFixed(3)}], mean ${meanDrive.toFixed(3)}; threshold 0.180 crossed ${flappy.motorThresholdCrossings}/${flappy.motorDriveSamples} frames (${flappy.motorThresholdCrossingsDuringCooldown} during cooldown, ${flappy.motorThresholdCrossingsActionEligible} action-eligible); flap actions ${flappy.neuralFlapDecisions}. Threshold unchanged pending distribution review.`;
    }
  }

  // Teacher labels are computed only in the assisted/offline collection path.
  // The calibrated neural action path never computes bird-to-gap target error.
  if (controllerMode === 'assisted') {
    const teacherGapCenterY = pipe.gap_y + flappy.pipe_gap * 0.5;
    const verticalError = flappy.bird_y - teacherGapCenterY;
    if (window.__COLLECT_FLAPPY_ROWS && Array.isArray(window.__FLAPPY_TRAINING_ROWS)) {
      const teacherAction = flappy.flap_cooldown <= 0 &&
        (verticalError > 5 || (flappy.bird_vy > 2.0 && verticalError > -10)) ? 1 : 0;
      window.__FLAPPY_TRAINING_ROWS.push({
        seed: experimentSeed,
        frame: flappy.frames,
        features: Array.from(V),
        target: teacherAction
      });
    }
  }
  if (flappy.flap_cooldown > 0) {
    flappy.flap_cooldown--;
  } else {
    let shouldFlap = false;
    let actionReason = null;
    let actionEvidence = null;
    let actionThreshold = null;
    if (controllerMode === 'assisted') {
      const gapCenterY = pipe.gap_y + flappy.pipe_gap * 0.5;
      const verticalError = flappy.bird_y - gapCenterY;
      shouldFlap = verticalError > 5 || (flappy.bird_vy > 2.0 && verticalError > -10);
      actionReason = shouldFlap ? 'assisted direct bird-to-gap rule' : null;
      actionEvidence = { verticalError, birdVerticalVelocity: flappy.bird_vy };
    } else if (controllerMode === 'neural') {
      // Learned linear readout of the full frozen connectome state. Training
      // labels came from the assisted reference policy, but inference has no
      // access to game geometry or bird kinematics.
      const probability = predictFlappyReadoutProbability();
      flappy.lastReadoutProbability = probability;
      flappy.lastNeuralDrive = probability == null ? 0 : probability;
      flappy.peakAbsNeuralDrive = Math.max(flappy.peakAbsNeuralDrive, probability == null ? 0 : probability);
      const threshold = Number(flappyDecoder?.threshold ?? 0.45);
      const fallbackDrive = (dbMean - vbMean) / 10.0 - flappy.neuralMotorBaseline;
      shouldFlap = probability == null
        ? fallbackDrive > 0.180
        : probability >= threshold;
      actionThreshold = probability == null ? 0.180 : threshold;
      actionReason = shouldFlap
        ? (probability == null ? 'neural readout unavailable; motor fallback exceeded threshold' : 'trained voltage readout probability met threshold')
        : null;
      actionEvidence = probability == null ? { fallbackDrive } : { probability };
      if (shouldFlap) flappy.neuralFlapDecisions++;
    } else {
      // Neuron-only, untrained connectome-output comparison retained as the default.
      const rawMotorDrive = (dbMean - vbMean) / 10.0;
      const neuralDrive = rawMotorDrive - flappy.neuralMotorBaseline;
      flappy.lastNeuralDrive = neuralDrive;
      flappy.peakAbsNeuralDrive = Math.max(flappy.peakAbsNeuralDrive, Math.abs(neuralDrive));
      const MOTOR_OUTPUT_THRESHOLD = 0.180;
      actionThreshold = MOTOR_OUTPUT_THRESHOLD;
      actionReason = neuralDrive > MOTOR_OUTPUT_THRESHOLD ? 'untrained DB−VB motor drive exceeded threshold' : null;
      actionEvidence = { dbMean, vbMean, baseline: flappy.neuralMotorBaseline, motorDrive: neuralDrive };
      // This deliberately does not retune the threshold or use game geometry.
      shouldFlap = neuralDrive > MOTOR_OUTPUT_THRESHOLD;
      if (shouldFlap) flappy.neuralFlapDecisions++;
    }
    if (shouldFlap) {
      flappy.actionLog.push({
        frame: flappy.frames, mode: controllerMode, action: 'flap', reason: actionReason,
        evidence: actionEvidence, threshold: actionThreshold,
        environmentSnapshot: { birdY: flappy.bird_y, birdVerticalVelocity: flappy.bird_vy, pipeX: pipe.x, gapY: pipe.gap_y }
      });
      flappy.flap_count++;
      flappy.bird_vy = flappy.flap_strength;
      flappy.flap_cooldown = controllerMode === 'assisted' ? 10 : (controllerMode === 'neural' ? 18 : 20);
    }
  }

  // Physics
  flappy.bird_vy += flappy.gravity;
  flappy.bird_vy = Math.min(7.5, Math.max(-8.5, flappy.bird_vy));
  flappy.bird_y += flappy.bird_vy;

  // Move Pipes
  for (let p of flappy.pipes) {
    p.x -= flappy.pipe_speed;
    if (!p.passed && p.x + flappy.pipe_w < flappy.bird_x) {
      p.passed = true;
      flappy.score++;
      if (flappy.score > flappy.high_score) flappy.high_score = flappy.score;
    }
  }

  // Respawn pipes
  if (flappy.pipes[0].x < -flappy.pipe_w) {
    flappy.pipes.shift();
    const lastX = flappy.pipes[flappy.pipes.length - 1].x;
    flappy.pipes.push({
      x: lastX + 190,
      gap_y: 38 + environmentRandom01() * 38,
      passed: false
    });
  }

  // Update Body Segments
  flappy.body_segments[0] = { x: flappy.bird_x, y: flappy.bird_y };
  for (let i = 1; i < flappy.body_segments.length; i++) {
    const prev = flappy.body_segments[i - 1];
    const curr = flappy.body_segments[i];
    const wave = Math.sin((Date.now() * 0.015) - (i * 0.45)) * 3.5;
    curr.x = flappy.bird_x - i * 5;
    curr.y += ((prev.y + wave * 0.3) - curr.y) * 0.5;
  }

  // Collision
  if (flappy.bird_y < 8) {
    flappy.is_game_over = true;
    flappy.game_over_reason = 'Ceiling collision';
  } else if (flappy.bird_y > 232) {
    flappy.is_game_over = true;
    flappy.game_over_reason = 'Floor collision';
  }
  for (let p of flappy.pipes) {
    if (flappy.bird_x + 9 > p.x && flappy.bird_x - 9 < p.x + flappy.pipe_w) {
      if (flappy.bird_y - 9 < p.gap_y || flappy.bird_y + 9 > p.gap_y + flappy.pipe_gap) {
        flappy.is_game_over = true;
        flappy.game_over_reason = 'Pipe collision';
      }
    }
  }
}

function updateChemotaxisGame() {
  if (chemotaxis.is_game_over) return;
  chemotaxis.frames++;
  // Finite, reproducible episodes: at most 1,800 simulation frames.
  // This is an experiment time limit, not a biological claim.
  // Sensory coding: concentration and its change are sent to the connectome.
  let dx = chemotaxis.food_x - chemotaxis.worm_x;
  let dy = chemotaxis.food_y - chemotaxis.worm_y;
  let dist = Math.hypot(dx, dy);
  const conc = Math.exp(-(dist * dist) / (2.0 * 80.0 * 80.0));
  const dC = conc - chemotaxis.prev_conc;
  chemotaxis.prev_conc = conc;

  I_sensory.fill(0);
  // Temporal concentration coding: ASEL responds to rising concentration and
  // ASER to falling concentration. This is a simplified sensory abstraction.
  if (dC > 0) injectCurrent("ASEL", Math.min(30.0, 150.0 * dC));
  else injectCurrent("ASER", Math.min(30.0, 150.0 * Math.abs(dC)));

  // Bilateral local-gradient probe: two virtual head-side sensors sample only
  // concentration at small offsets from the head, not the food bearing or target
  // heading. AWCL/AWCR currents encode those local samples into the network.
  // This is an explicit engineered sensor model, not a literal anatomical claim.
  const probeOffset = 10.0;
  const leftProbeX = chemotaxis.worm_x - Math.sin(chemotaxis.heading) * probeOffset;
  const leftProbeY = chemotaxis.worm_y + Math.cos(chemotaxis.heading) * probeOffset;
  const rightProbeX = chemotaxis.worm_x + Math.sin(chemotaxis.heading) * probeOffset;
  const rightProbeY = chemotaxis.worm_y - Math.cos(chemotaxis.heading) * probeOffset;
  const leftFoodDx = chemotaxis.food_x - leftProbeX;
  const leftFoodDy = chemotaxis.food_y - leftProbeY;
  const rightFoodDx = chemotaxis.food_x - rightProbeX;
  const rightFoodDy = chemotaxis.food_y - rightProbeY;
  const leftProbeConc = Math.exp(-(leftFoodDx * leftFoodDx + leftFoodDy * leftFoodDy) / (2.0 * 80.0 * 80.0));
  const rightProbeConc = Math.exp(-(rightFoodDx * rightFoodDx + rightFoodDy * rightFoodDy) / (2.0 * 80.0 * 80.0));
  injectCurrent("AWCL", Math.min(38.0, 14.0 * leftProbeConc + 800.0 * Math.max(0, leftProbeConc - rightProbeConc)));
  injectCurrent("AWCR", Math.min(38.0, 14.0 * rightProbeConc + 800.0 * Math.max(0, rightProbeConc - leftProbeConc)));
  // Directional wall-looming encoder: determine the nearest boundary along the
  // current heading, then encode which side of the body's heading axis the
  // obstacle occupies into the bilateral ASH channels. This is a sensory input,
  // not a direct food-bearing or target-action signal.
  const hx = Math.cos(chemotaxis.heading);
  const hy = Math.sin(chemotaxis.heading);
  const wallCandidates = [];
  if (hx > 0.001) wallCandidates.push({ distance: (406 - chemotaxis.worm_x) / hx, angle: 0 });
  if (hx < -0.001) wallCandidates.push({ distance: (14 - chemotaxis.worm_x) / hx, angle: Math.PI });
  if (hy > 0.001) wallCandidates.push({ distance: (226 - chemotaxis.worm_y) / hy, angle: Math.PI / 2 });
  if (hy < -0.001) wallCandidates.push({ distance: (14 - chemotaxis.worm_y) / hy, angle: -Math.PI / 2 });
  const forwardWall = wallCandidates.filter(w => w.distance >= 0).sort((a, b) => a.distance - b.distance)[0];
  const wallClearance = forwardWall ? forwardWall.distance : 999;
  const wallThreat = Math.max(0, Math.min(1, (58 - wallClearance) / 58));
  if (wallThreat > 0 && forwardWall) {
    const relativeWallAngle = wrapAngle(forwardWall.angle - chemotaxis.heading);
    // ASHL is the synthetic left-side danger channel; ASHR is the right-side
    // danger channel. The polarity is an explicit adapter assumption.
    if (relativeWallAngle > 0.12) injectCurrent("ASHL", 65.0 * wallThreat);
    else if (relativeWallAngle < -0.12) injectCurrent("ASHR", 65.0 * wallThreat);
    else if (random01() < 0.5) injectCurrent("ASHL", 65.0 * wallThreat);
    else injectCurrent("ASHR", 65.0 * wallThreat);
  }
  for (let i = 0; i < 4; i++) stepBiophysicalNetwork(0.002);

  // Read out biological state for the telemetry panel.
  const vAva = V[nameToIdx["AVAL"] || 0];
  const vAvb = V[nameToIdx["AVBL"] || 0];
  const smdd = (V[nameToIdx["SMDDL"] || 0] - V[nameToIdx["SMDDR"] || 0]) * 0.04;

  const headingBeforeAction = chemotaxis.heading;
  let headingActionReason = controllerMode === 'assisted'
    ? 'assisted direct food-bearing steering'
    : (controllerMode === 'strict' ? 'neuron-state-only lateral motor readout (fixed actuator gain; no action override)' : 'trained neural turn readout');
  let headingActionEvidence = null;

  // Only the assisted teacher computes food bearing. Calibrated neural and
  // strict branches use local concentration inputs and simulated neuron states.
  if (controllerMode === 'assisted') {
    const targetHeading = Math.atan2(dy, dx);
    let teacherAngleError = targetHeading - chemotaxis.heading;
    while (teacherAngleError > Math.PI) teacherAngleError -= 2 * Math.PI;
    while (teacherAngleError < -Math.PI) teacherAngleError += 2 * Math.PI;
    const teacherTurn = Math.max(-0.10, Math.min(0.10, teacherAngleError));
    headingActionEvidence = { teacherTurn, teacherAngleError };
    if (window.__COLLECT_CHEMO_ROWS && Array.isArray(window.__CHEMO_TRAINING_ROWS)) {
      window.__CHEMO_TRAINING_ROWS.push({ seed: experimentSeed, frame: chemotaxis.frames, features: Array.from(V), target: teacherTurn });
    }
    chemotaxis.heading += teacherTurn;
  } else {
    const ava = V[nameToIdx["AVAL"] ?? 0];
    const avb = V[nameToIdx["AVBL"] ?? 0];
    const leftMotor = V[nameToIdx["SMDDL"] ?? 0];
    const rightMotor = V[nameToIdx["SMDDR"] ?? 0];
    // AVA/AVB encode backward/forward command state, not left/right steering.
    // Use bilateral command and motor asymmetry for smooth curvature instead of
    // the old AVA−AVB absolute voltage difference that forced a constant circle.
    // Remove each trial's tonic left/right bias before decoding curvature. A
    // constant membrane offset is not evidence for a continuously changing turn.
    const commandTurn = ((neuronVoltage('AVAL') - neuronVoltage('AVAR')) - chemotaxis.turnBaselineCommand) / 180.0;
    const motorBias = ((rightMotor - leftMotor) - chemotaxis.turnBaselineMotor) / 240.0;
    // Fixed virtual actuator gain converts the small voltage-derived motor
    // asymmetry into arena curvature. It does not inspect food or boundaries;
    // it is an explicit output-scale assumption, not a steering policy.
    const strictTurn = Math.max(-0.045, Math.min(0.045, 5.0 * (commandTurn + motorBias)));
    const learnedTurn = controllerMode === 'neural' ? predictChemotaxisTurn() : null;
    const neuralTurn = learnedTurn === null ? strictTurn : learnedTurn;
    chemotaxis.lastDecoderTurn = neuralTurn;
    headingActionEvidence = { decodedTurn: neuralTurn, commandTurn, motorBias, readout: learnedTurn === null ? 'untrained neuron-state readout' : 'trained readout' };

    // Neuron-only mode applies the motor readout directly. There is deliberately
    // no wall-turn override here: wall proximity may be encoded as synthetic
    // sensory input to ASH above, but only the neuron-state motor readout changes
    // heading. The calibrated neural mode retains its separately disclosed
    // ASER-associated adapter for comparison.
    chemotaxis.heading += neuralTurn;

    const aserResponse = neuronVoltage('ASER') - chemotaxis.aserBaselineVoltage;
    const aserResponseActive = aserResponse > 0.55;
    // Require both a modeled ASER response and a falling concentration signal.
    // The sensory derivative changes the probability of a tumble; no food-bearing
    // angle is passed to the action decoder. One frame can cause at most one turn.
    const fallingGradient = dC < -0.00005;
    if (controllerMode === 'neural' && Math.abs(neuralTurn) < 0.025 && aserResponseActive && fallingGradient &&
        chemotaxis.frames - chemotaxis.lastPirouetteFrame >= 100) {
      const turnSign = random01() < 0.5 ? -1 : 1;
      const turnMagnitude = 0.55 + random01() * 0.75;
      chemotaxis.heading += turnSign * turnMagnitude;
      chemotaxis.pirouettes++;
      chemotaxis.turns++;
      headingActionReason = 'ASER-associated stochastic pirouette after falling concentration';
      headingActionEvidence = { aserResponse, dC, turnRadians: turnSign * turnMagnitude };
      chemotaxis.lastPirouetteFrame = chemotaxis.frames;
    }
    chemotaxis.aserResponseActive = aserResponseActive;
  }
  // No controller may silently override a motor command based on arena position.
  const loggedHeadingDelta = wrapAngle(chemotaxis.heading - headingBeforeAction);
  if (Math.abs(loggedHeadingDelta) >= 0.01) {
    chemotaxis.actionLog.push({
      frame: chemotaxis.frames, mode: controllerMode, action: 'turn',
      reason: headingActionReason, appliedTurnRadians: loggedHeadingDelta,
      evidence: headingActionEvidence
    });
  }
  const speed = controllerMode === 'assisted' ? (dist < 24 ? 1.2 : 1.8) : 1.35;
  const previousHeadX = chemotaxis.worm_x;
  const previousHeadY = chemotaxis.worm_y;
  chemotaxis.worm_x += speed * Math.cos(chemotaxis.heading);
  chemotaxis.worm_y += speed * Math.sin(chemotaxis.heading);
  const headStepDistance = Math.hypot(
    chemotaxis.worm_x - previousHeadX,
    chemotaxis.worm_y - previousHeadY
  );

  // Explicit arena boundary collision. The head is clamped only for clean
  // rendering after the terminal event; it is not reflected or steered inward.
  const margin = 14;
  const hitBoundary = chemotaxis.worm_x < margin || chemotaxis.worm_x > 406 ||
                      chemotaxis.worm_y < margin || chemotaxis.worm_y > 226;
  if (hitBoundary) {
    chemotaxis.worm_x = Math.max(margin, Math.min(406, chemotaxis.worm_x));
    chemotaxis.worm_y = Math.max(margin, Math.min(226, chemotaxis.worm_y));
    chemotaxis.is_game_over = true;
    chemotaxis.game_over_reason = 'Boundary collision';
  }

  // Turn events are counted once per sustained reorientation, not every time
  // accumulated frame-wise curvature crosses 0.60 rad. Require three consecutive
  // meaningful heading changes to start an episode, then eight quiet frames to
  // re-arm the detector. Thresholds are explicit engineering choices.
  const actualHeadingDelta = wrapAngle(chemotaxis.heading - headingBeforeAction);
  const absHeadingDelta = Math.abs(actualHeadingDelta);
  if (!chemotaxis.turning) {
    if (absHeadingDelta >= 0.04) {
      chemotaxis.turnCandidateFrames++;
    } else {
      chemotaxis.turnCandidateFrames = 0;
    }
    if (chemotaxis.turnCandidateFrames >= 3) {
      chemotaxis.turning = true;
      chemotaxis.turnEpisodes++;
      chemotaxis.turnAngleAccumulator = 0;
      chemotaxis.turnStableFrames = 0;
      chemotaxis.turnEventCounted = false;
    }
  }
  if (chemotaxis.turning) {
    if (absHeadingDelta >= 0.015) {
      chemotaxis.turnAngleAccumulator += absHeadingDelta;
      chemotaxis.turnStableFrames = 0;
    } else {
      chemotaxis.turnStableFrames++;
    }
    // Discrete turn events are counted at the wall-reflex or pirouette command.
    // This accumulator is retained only to track smooth turn episodes; it does
    // not increment the discrete-event counter a second time.
    if (chemotaxis.turnStableFrames >= 8) {
      chemotaxis.turning = false;
      chemotaxis.turnCandidateFrames = 0;
      chemotaxis.turnStableFrames = 0;
      chemotaxis.turnAngleAccumulator = 0;
      chemotaxis.turnEventCounted = false;
    }
  }

  // Preserve the actual head trajectory and sample each body segment at a
  // fixed arc-length distance behind it. This prevents body collapse during
  // turns and makes self-collision checks correspond to the path actually taken.
  const segmentSpacing = 5.0;
  chemotaxis.pathHistory.unshift({ x: chemotaxis.worm_x, y: chemotaxis.worm_y });
  // Runs are capped at 1,800 frames; this bound is comfortably longer than
  // the trail needed by the maximum body length in this task.
  if (chemotaxis.pathHistory.length > 3000) chemotaxis.pathHistory.length = 3000;
  let historyIndex = 0;
  let historyDistance = 0;
  for (let i = 0; i < chemotaxis.body.length; i++) {
    const targetDistance = i * segmentSpacing;
    let placed = false;
    while (historyIndex < chemotaxis.pathHistory.length - 1) {
      const newer = chemotaxis.pathHistory[historyIndex];
      const older = chemotaxis.pathHistory[historyIndex + 1];
      const segmentDistance = Math.hypot(newer.x - older.x, newer.y - older.y);
      if (segmentDistance < 0.0001) {
        historyIndex++;
        continue;
      }
      if (historyDistance + segmentDistance >= targetDistance) {
        const fraction = Math.max(0, Math.min(1, (targetDistance - historyDistance) / segmentDistance));
        chemotaxis.body[i] = {
          x: newer.x + (older.x - newer.x) * fraction,
          y: newer.y + (older.y - newer.y) * fraction
        };
        placed = true;
        break;
      }
      historyDistance += segmentDistance;
      historyIndex++;
    }
    if (!placed) {
      const oldest = chemotaxis.pathHistory[chemotaxis.pathHistory.length - 1];
      chemotaxis.body[i] = { x: oldest.x, y: oldest.y };
    }
  }

  // Self-collision checks the head against older trajectory-sampled body
  // segments only. The first eight segments are excluded as the neck/turning arc.
  if (!chemotaxis.is_game_over) {
    for (let i = 8; i < chemotaxis.body.length; i++) {
      if (Math.hypot(chemotaxis.worm_x - chemotaxis.body[i].x,
                     chemotaxis.worm_y - chemotaxis.body[i].y) < 5.5) {
        chemotaxis.is_game_over = true;
        chemotaxis.game_over_reason = 'Self-collision';
        break;
      }
    }
  }

  // Check AFTER moving, otherwise the worm can cross food without scoring.
  // Collision ends the run before food can be harvested at the same location.
  if (!chemotaxis.is_game_over) {
    dx = chemotaxis.food_x - chemotaxis.worm_x;
    dy = chemotaxis.food_y - chemotaxis.worm_y;
    if (Math.hypot(dx, dy) < 14.0) {
      chemotaxis.score++;
      // Grow by one segment behind the current tail, avoiding a duplicate
      // overlapping point that would itself corrupt collision detection.
      const tail = chemotaxis.body[chemotaxis.body.length - 1];
      const beforeTail = chemotaxis.body[Math.max(0, chemotaxis.body.length - 2)];
      let tailDx = tail.x - beforeTail.x;
      let tailDy = tail.y - beforeTail.y;
      let tailNorm = Math.hypot(tailDx, tailDy);
      if (tailNorm < 0.001) {
        tailDx = -Math.cos(chemotaxis.heading);
        tailDy = -Math.sin(chemotaxis.heading);
        tailNorm = 1;
      }
      chemotaxis.body.push({
        x: tail.x + (tailDx / tailNorm) * 5.0,
        y: tail.y + (tailDy / tailNorm) * 5.0
      });
      chemotaxis.food_x = 60 + environmentRandom01() * 300;
      chemotaxis.food_y = 40 + environmentRandom01() * 160;
      chemotaxis.prev_conc = Math.exp(-(((chemotaxis.food_x - chemotaxis.worm_x) ** 2) + ((chemotaxis.food_y - chemotaxis.worm_y) ** 2)) / (2.0 * 80.0 * 80.0));
    }
  }

  if (!chemotaxis.is_game_over && chemotaxis.frames >= chemotaxis.maxFrames) {
    chemotaxis.is_game_over = true;
    chemotaxis.game_over_reason = 'Time limit (1,800 frames)';
  }
}

// Render Retro Game Canvas
function renderGameCanvas() {
  const canvas = document.getElementById('game-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  const sky = ctx.createLinearGradient(0, 0, 0, h);
  sky.addColorStop(0, '#63d8ef'); sky.addColorStop(0.72, '#b8f0ef'); sky.addColorStop(1, '#e4f5c4');
  ctx.fillStyle = sky; ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = 'rgba(54,161,177,0.22)';
  for (let i=0;i<13;i++) { const bx=(i*43+7)%w, bh=14+(i%4)*8; ctx.fillRect(bx,190-bh,24,bh); }

  if (currentGameMode === 'flappy') {
    ctx.fillStyle='rgba(255,255,255,0.82)';
    const shift=(flappy.frames*0.35)%(w+70);
    for(let i=0;i<3;i++){ const cx=((i*170-shift+w+70)%(w+70))-30, cy=28+i*23; ctx.beginPath(); ctx.arc(cx,cy,11,0,Math.PI*2); ctx.arc(cx+13,cy-5,15,0,Math.PI*2); ctx.arc(cx+28,cy,10,0,Math.PI*2); ctx.fill(); }
    for(const p of flappy.pipes){
      const bottomY=p.gap_y+flappy.pipe_gap;
      const drawPipe=(x,y,pw,ph,isTop)=>{ if(ph<=0)return; const grad=ctx.createLinearGradient(x,0,x+pw,0); grad.addColorStop(0,'#2f9d48');grad.addColorStop(.25,'#73d957');grad.addColorStop(.62,'#43bb4c');grad.addColorStop(1,'#197a3a');ctx.fillStyle=grad;ctx.fillRect(x,y,pw,ph);ctx.strokeStyle='#176b35';ctx.lineWidth=2;ctx.strokeRect(x+1,y+1,pw-2,ph-2);ctx.fillStyle='#8be45b';ctx.fillRect(x+7,y,5,ph);const ch=10,cy=isTop?y+ph-ch:y;ctx.fillStyle='#53c84c';ctx.fillRect(x-4,cy,pw+8,ch);ctx.strokeStyle='#176b35';ctx.strokeRect(x-4,cy,pw+8,ch);ctx.fillStyle='rgba(255,255,255,.25)';ctx.fillRect(x+2,cy+2,pw-4,2);};
      drawPipe(p.x,0,flappy.pipe_w,p.gap_y,true); drawPipe(p.x,bottomY,flappy.pipe_w,h-bottomY-16,false);
    }
    ctx.lineCap='round';
    for(let i=flappy.body_segments.length-1;i>=0;i--){const seg=flappy.body_segments[i],r=Math.max(2.2,6.2-i*.28);ctx.fillStyle=i<3?'#16b981':'#079669';ctx.beginPath();ctx.ellipse(seg.x,seg.y,r+1.5,r,-.12,0,Math.PI*2);ctx.fill();ctx.strokeStyle='rgba(221,255,220,.55)';ctx.lineWidth=1;ctx.stroke();}
    ctx.fillStyle='#20d5b0';ctx.beginPath();ctx.ellipse(flappy.bird_x,flappy.bird_y,10,8,-.1,0,Math.PI*2);ctx.fill();ctx.fillStyle='#f8fafc';ctx.beginPath();ctx.arc(flappy.bird_x+4,flappy.bird_y-2,2.5,0,Math.PI*2);ctx.fill();ctx.fillStyle='#102a43';ctx.beginPath();ctx.arc(flappy.bird_x+5,flappy.bird_y-2,1.1,0,Math.PI*2);ctx.fill();
    ctx.fillStyle='#7ccf57';ctx.fillRect(0,h-16,w,7);ctx.fillStyle='#d8d38c';ctx.fillRect(0,h-9,w,9);ctx.fillStyle='#b7b16c';for(let x=0;x<w;x+=16)ctx.fillRect(x,h-5,8,2);
    document.getElementById('hud-score').textContent=`SCORE: ${flappy.score}  •  BEST: ${flappy.high_score}`;
    const status=document.getElementById('hud-status');
    const liveLabel = controllerMode === 'assisted' ? 'ASSISTED BASELINE · DIRECT GAME-STATE CONTROL'
      : (controllerMode === 'strict' ? 'NEURON-ONLY · NO TRAINED READOUT / NO OVERRIDES'
      : 'NEURAL SENSORIMOTOR · TRAINED READOUT');
    status.textContent = flappy.is_game_over ? `RUN OVER · ${flappy.game_over_reason || 'collision'} · PRESS RESTART` : liveLabel;
    status.style.color = flappy.is_game_over ? '#ff3366' : (controllerMode === 'assisted' ? '#00a86b' : (controllerMode === 'strict' ? '#ffb300' : '#00f2fe'));
    if(flappy.is_game_over){ctx.fillStyle='rgba(5,12,24,.72)';ctx.fillRect(0,0,w,h);ctx.textAlign='center';ctx.fillStyle='#fff';ctx.font='bold 23px system-ui,sans-serif';ctx.fillText('RUN OVER',w/2,h/2-9);ctx.font='12px system-ui,sans-serif';ctx.fillStyle='#d4e5f2';ctx.fillText(`Score ${flappy.score}  ·  Best ${flappy.high_score}`,w/2,h/2+15);ctx.fillText(flappy.game_over_reason || 'Collision',w/2,h/2+32);ctx.fillText('Select ↻ Restart to try again',w/2,h/2+48);}
  } else {
    const grad=ctx.createRadialGradient(chemotaxis.food_x,chemotaxis.food_y,3,chemotaxis.food_x,chemotaxis.food_y,100);grad.addColorStop(0,'rgba(255,179,0,.5)');grad.addColorStop(1,'rgba(255,179,0,0)');ctx.fillStyle=grad;ctx.beginPath();ctx.arc(chemotaxis.food_x,chemotaxis.food_y,100,0,Math.PI*2);ctx.fill();ctx.fillStyle='#f59e0b';ctx.beginPath();ctx.arc(chemotaxis.food_x,chemotaxis.food_y,7,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#fff0b3';ctx.lineWidth=2;ctx.stroke();
    for(let i=chemotaxis.body.length-1;i>=0;i--){const seg=chemotaxis.body[i];ctx.fillStyle=i===0?'#22d3ee':'#10b981';ctx.beginPath();ctx.arc(seg.x,seg.y,Math.max(2.2,6-i*.22),0,Math.PI*2);ctx.fill();}
    if (chemotaxis.is_game_over) {
      ctx.fillStyle='rgba(2,6,23,0.72)'; ctx.fillRect(0,0,w,h);
      ctx.textAlign='center'; ctx.fillStyle='#f8fafc'; ctx.font='bold 24px system-ui';
      ctx.fillText('RUN OVER',w/2,h/2-12); ctx.font='13px system-ui';
      ctx.fillText(chemotaxis.game_over_reason,w/2,h/2+12);
      ctx.fillText('Select Restart to try again',w/2,h/2+34);
    }
    document.getElementById('hud-score').textContent=`FOOD: ${chemotaxis.score} • REORIENTATIONS: ${chemotaxis.turns} • LENGTH: ${chemotaxis.body.length}`;document.getElementById('hud-status').textContent=chemotaxis.is_game_over ? `GAME OVER · ${chemotaxis.game_over_reason} · PRESS RESTART` : (controllerMode === 'assisted' ? 'ASSISTED BASELINE · NEURAL TELEMETRY' : (controllerMode === 'strict' ? 'NEURON-ONLY · NO ACTION OVERRIDES' : 'NEURAL HYBRID · TRAINED READOUT + ASER ADAPTER'));document.getElementById('hud-status').style.color=chemotaxis.is_game_over ? '#ff3366' : '#00a86b';
  }
}

function renderOscilloscope() {
  const canvas = document.getElementById('scope-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.fillStyle = '#0d192c';
  ctx.fillRect(0, 0, w, h);

  // Grid
  ctx.strokeStyle = '#1e293b';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let y = 20; y < h; y += 25) {
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
  }
  ctx.stroke();

  // Update trace buffers
  Object.keys(scopeTraces).forEach(name => {
    const idx = nameToIdx[name];
    const val = (idx !== undefined) ? V[idx] : E_LEAK;
    const buf = scopeTraces[name];
    for (let i = 0; i < buf.length - 1; i++) {
      buf[i] = buf[i + 1];
    }
    buf[buf.length - 1] = val;
  });

  const traceConfigs = [
    { name: 'AVAL', color: '#ff3366' },
    { name: 'AVBL', color: '#00ff87' },
    { name: 'ASHL', color: '#00f2fe' },
    { name: 'DB1', color: '#ffb300' }
  ];

  traceConfigs.forEach(cfg => {
    const buf = scopeTraces[cfg.name];
    ctx.strokeStyle = cfg.color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < buf.length; i++) {
      const x = (i / buf.length) * w;
      // Map -80mV to 0mV to canvas height
      const y = h - ((buf[i] - (-75.0)) / 60.0) * h;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  });
}

function updateGauges(ashVal, flapDrive, dbAct, vbAct) {
  const ashBar = document.getElementById('meter-sensory');
  if (ashBar) {
    const pct = Math.min(100, (ashVal / 35.0) * 100);
    ashBar.style.width = `${pct}%`;
    ashBar.style.backgroundColor = ashVal > 15 ? '#ff3366' : '#00f2fe';
  }

  const motorBar = document.getElementById('meter-motor');
  if (motorBar) {
    const pct = Math.min(100, Math.max(0, ((flapDrive + 2.0) / 7.0) * 100));
    motorBar.style.width = `${pct}%`;
    motorBar.style.backgroundColor = flapDrive > 3.0 ? '#00ff87' : '#9b51e0';
  }
}

// Helpers
function injectCurrent(name, current) {
  const idx = nameToIdx[name];
  if (idx !== undefined) {
    I_sensory[idx] = current;
  }
}

function getPrefixMean(prefix) {
  let sum = 0, count = 0;
  neuronNames.forEach((n, i) => {
    if (n.startsWith(prefix) && !ablatedSet.has(i)) {
      sum += V[i];
      count++;
    }
  });
  return count > 0 ? sum / count : E_LEAK;
}

// Trained linear readout over the current 302-neuron voltage state. The
// connectome dynamics remain frozen; only this output decoder is trained.
// At runtime it reads neuron voltages only, never bird height/gap coordinates.
function predictFlappyReadoutProbability() {
  if (!flappyDecoder || !Array.isArray(flappyDecoder.weights) ||
      !Array.isArray(flappyDecoder.feature_names) ||
      flappyDecoder.weights.length !== flappyDecoder.feature_names.length ||
      !Array.isArray(flappyDecoder.mean) || !Array.isArray(flappyDecoder.scale)) {
    return null;
  }
  let logit = Number(flappyDecoder.bias || 0);
  let activeFeatures = 0;
  for (let j = 0; j < flappyDecoder.feature_names.length; j++) {
    const i = nameToIdx[flappyDecoder.feature_names[j]];
    if (i === undefined) return null;
    // An ablated neuron is missing information, not an observed -60 mV sample.
    // Mean-imputation (z=0) prevents the decoder treating the artificial leak
    // clamp as an extreme biological voltage. Network-wide consequences of the
    // ablation remain present in all surviving neuron's actual voltages.
    if (ablatedSet.has(i)) continue;
    activeFeatures++;
    const scale = Number(flappyDecoder.scale[j]) || 1.0;
    const z = Math.max(-3.0, Math.min(3.0, (V[i] - Number(flappyDecoder.mean[j] || 0)) / scale));
    logit += Number(flappyDecoder.weights[j] || 0) * z;
  }
  flappy.decoderFeatureCoverage = activeFeatures / flappyDecoder.feature_names.length;
  logit = Math.max(-30, Math.min(30, logit));
  return 1.0 / (1.0 + Math.exp(-logit));
}

function predictChemotaxisTurn() {
  if (!chemotaxisDecoder || !Array.isArray(chemotaxisDecoder.weights) ||
      !Array.isArray(chemotaxisDecoder.feature_names) ||
      chemotaxisDecoder.weights.length !== chemotaxisDecoder.feature_names.length ||
      !Array.isArray(chemotaxisDecoder.mean) || !Array.isArray(chemotaxisDecoder.scale)) return null;
  let prediction = Number(chemotaxisDecoder.bias || 0);
  let activeFeatures = 0;
  for (let j = 0; j < chemotaxisDecoder.feature_names.length; j++) {
    const i = nameToIdx[chemotaxisDecoder.feature_names[j]];
    if (i === undefined) return null;
    // Omit ablated features instead of treating their artificial leak clamp as data.
    if (ablatedSet.has(i)) continue;
    activeFeatures++;
    const scale = Number(chemotaxisDecoder.scale[j]) || 1.0;
    const z = Math.max(-3.0, Math.min(3.0, (V[i] - Number(chemotaxisDecoder.mean[j] || 0)) / scale));
    prediction += Number(chemotaxisDecoder.weights[j] || 0) * z;
  }
  chemotaxis.decoderFeatureCoverage = activeFeatures / chemotaxisDecoder.feature_names.length;
  // A learned turn is an actuator command; bound it to the assisted reference's
  // maximum angular step and keep the trained output separate from wall reflexes.
  const runtimeLimit = Number(chemotaxisDecoder.runtime_turn_limit_radians ?? 0.045);
  return Math.max(-runtimeLimit, Math.min(runtimeLimit, prediction));
}

function currentGameFrame() {
  return currentGameMode === 'flappy' ? flappy.frames : chemotaxis.frames;
}

function updateOptogeneticTelemetry() {
  if (!optoExperiment || optoExperiment.endedAt) return;
  optoExperiment.elapsedFrames = (optoExperiment.elapsedFrames || 0) + 1;
  const ava = activeAvaVoltage();
  const motor = motorDriveSnapshot();
  optoExperiment.peakAvaVoltage = Math.max(optoExperiment.peakAvaVoltage, ava);
  optoExperiment.minAvaVoltage = Math.min(optoExperiment.minAvaVoltage, ava);
  optoExperiment.peakDbMinusVb = Math.max(optoExperiment.peakDbMinusVb, motor.db - motor.vb);
  optoExperiment.minDbMinusVb = Math.min(optoExperiment.minDbMinusVb, motor.db - motor.vb);
  optoExperiment.latestAvaVoltage = ava;
  optoExperiment.latestMotor = motor;
  optoExperiment.latestScore = currentGameMode === 'flappy' ? flappy.score : chemotaxis.score;
  optoExperiment.latestHeading = chemotaxis.heading;
  // End the pulse after a fixed number of simulation frames so the experiment
  // is not dependent on browser timer scheduling or machine speed.
  if (optoExperiment.elapsedFrames >= optoExperiment.durationFrames) {
    optoExperiment.targetNames.forEach(n => { I_opto[nameToIdx[n]] = 0; });
    finishOptogeneticPulse();
    const note = document.getElementById('controller-note');
    if (note) note.textContent = formatOptogeneticSummary(optoExperiment);
  }
}

function formatOptogeneticSummary(summary) {
  if (!summary) return 'No completed optogenetic experiment.';
  const motorStart = summary.baselineMotor.db - summary.baselineMotor.vb;
  const motorEnd = summary.endMotor.db - summary.endMotor.vb;
  const behavior = currentGameMode === 'flappy'
    ? `flaps Δ${summary.flapDelta}`
    : `heading Δ${Number(summary.headingDeltaRad ?? 0).toFixed(3)} rad`;
  return `Pulse ended (${summary.elapsedFrames}/${summary.durationFrames} active simulation frames; ${summary.endReason || 'end reason not recorded'}): AVA ${summary.baselineAvaVoltage.toFixed(1)}→${summary.peakAvaVoltage.toFixed(1)} mV peak, end ${summary.endAvaVoltage.toFixed(1)} mV; DB−VB ${motorStart.toFixed(2)}→${motorEnd.toFixed(2)}; score Δ${summary.scoreDelta}; ${behavior}. Compare against the matched no-pulse control.`;
}

function finishOptogeneticPulse(endReason = 'pulse duration completed') {
  if (!optoExperiment || optoExperiment.endedAt) return;
  optoExperiment.endedAt = Date.now();
  optoExperiment.endReason = endReason;
  optoExperiment.endAvaVoltage = activeAvaVoltage();
  optoExperiment.endMotor = motorDriveSnapshot();
  optoExperiment.endScore = currentGameMode === 'flappy' ? flappy.score : chemotaxis.score;
  optoExperiment.endHeading = chemotaxis.heading;
  optoExperiment.endFlaps = flappy.flap_count;
  optoExperiment.scoreDelta = optoExperiment.endScore - optoExperiment.startScore;
  optoExperiment.headingDeltaRad = currentGameMode === 'chemotaxis' ? wrapAngle(optoExperiment.endHeading - optoExperiment.startHeading) : null;
  optoExperiment.flapDelta = currentGameMode === 'flappy' ? optoExperiment.endFlaps - optoExperiment.startFlaps : null;
}

function startOptogeneticPulse(label = 'AVA pulse') {
  const targetNames = ['AVAL', 'AVAR'].filter(n => nameToIdx[n] !== undefined && !ablatedSet.has(nameToIdx[n]));
  if (!targetNames.length) {
    recordIntervention('Blue pulse rejected: both AVA neurons are unavailable/ablated');
    return false;
  }
  if (optoExperiment && !optoExperiment.endedAt) finishOptogeneticPulse('replaced by a new pulse');
  const motor = motorDriveSnapshot();
  const score = currentGameMode === 'flappy' ? flappy.score : chemotaxis.score;
  optoExperiment = {
    targetNames, opsin: 'ChR2-like depolarizing current', durationMs: 450, durationFrames: OPTO_DURATION_FRAMES,
    startAt: Date.now(), startFrame: currentGameFrame() + 1, elapsedFrames: 0,
    baselineAvaVoltage: activeAvaVoltage(), peakAvaVoltage: activeAvaVoltage(),
    minAvaVoltage: activeAvaVoltage(), latestAvaVoltage: activeAvaVoltage(),
    baselineMotor: motor, peakDbMinusVb: motor.db - motor.vb,
    minDbMinusVb: motor.db - motor.vb, latestMotor: motor,
    startScore: score, latestScore: score, startHeading: chemotaxis.heading, startFlaps: flappy.flap_count,
    latestHeading: chemotaxis.heading, endedAt: null
  };
  targetNames.forEach(n => { I_opto[nameToIdx[n]] = 40.0; });
  recordIntervention(`${label}: ChR2-like current on ${targetNames.join(', ')} for ${OPTO_DURATION_FRAMES} simulation frames`);
  const note = document.getElementById('controller-note');
  if (note) note.textContent = `${lastIntervention}. Measuring AVA voltage → DB/VB motor contrast → behavioral response. Use the matched no-pulse control for causal comparison.`;
  return true;
}

// UI Event Listeners
function setupUIEventListeners() {
  document.getElementById('btn-mode-flappy')?.addEventListener('click', () => {
    currentGameMode = 'flappy';
    document.getElementById('btn-mode-flappy').classList.add('active');
    document.getElementById('btn-mode-chemo').classList.remove('active');
    resetFlappy();
  });

  document.getElementById('btn-mode-chemo')?.addEventListener('click', () => {
    currentGameMode = 'chemotaxis';
    document.getElementById('btn-mode-chemo').classList.add('active');
    document.getElementById('btn-mode-flappy').classList.remove('active');
    resetChemotaxis();
  });

  document.getElementById('btn-restart-game')?.addEventListener('click', () => {
    if (currentGameMode === 'flappy') resetFlappy(); else resetChemotaxis();
  });

  const setControllerMode = (mode) => {
    controllerMode = mode;
    document.getElementById('btn-controller-assisted')?.classList.toggle('active', mode === 'assisted');
    document.getElementById('btn-controller-neural')?.classList.toggle('active', mode === 'neural');
    document.getElementById('btn-controller-strict')?.classList.toggle('active', mode === 'strict');
    const note = document.getElementById('controller-note');
    if (note) {
      if (mode === 'assisted') {
        note.textContent = 'ASSISTED OVERRIDES: Flappy uses direct bird-to-gap error; Chemotaxis steers toward food bearing. The network runs for telemetry but does not determine assisted actions.';
      } else if (mode === 'neural') {
        note.textContent = 'TRAINED NEURAL HYBRID: Flappy and Chemotaxis use offline-trained neuron-voltage readouts that imitate the assisted reference. Runtime decoders use neuron voltages, not direct game-state/food-bearing targets, but this is not zero-shot behavior; Chemotaxis also includes an ASER-associated turning adapter. Select Neuron-only for untrained motor-output experiments.';
      } else {
        note.textContent = 'NEURON-ONLY MODE: Flappy actions come from the untrained DB−VB motor-neuron contrast; Chemotaxis turns come from untrained command/motor neuron-state asymmetry. No trained decoder, game-state steering, boundary-turn override, or direct ASER-to-turn rule is used. Synthetic sensory encoders and a virtual actuator mapping remain explicit model assumptions; behavior may perform poorly.';
      }
    }
    if (currentGameMode === 'flappy') resetFlappy(); else resetChemotaxis();
  };
  document.getElementById('btn-controller-assisted')?.addEventListener('click', () => setControllerMode('assisted'));
  document.getElementById('btn-controller-neural')?.addEventListener('click', () => setControllerMode('neural'));
  document.getElementById('btn-controller-strict')?.addEventListener('click', () => setControllerMode('strict'));

  // Matched optogenetic experiment: both conditions reset to the same seed and
  // neural/game state; the pulse condition stimulates AVA at the same frame.
  const startControlledTrial = (withPulse) => {
    if (controllerMode !== 'strict') setControllerMode('strict');
    else if (currentGameMode === 'flappy') resetFlappy(); else resetChemotaxis();
    controlledProtocol = {
      condition: withPulse ? 'matched AVA pulse' : 'matched no-pulse control',
      pulseScheduled: withPulse, pulseAtFrame: withPulse ? CONTROLLED_PULSE_FRAME : null,
      seed: experimentSeed, controller: 'Neuron-only (no overrides)', game: currentGameMode
    };
    const note = document.getElementById('controller-note');
    if (note) note.textContent = withPulse
      ? `Controlled trial started. Neuron-only, seed ${experimentSeed}; AVA pulse scheduled at frame ${CONTROLLED_PULSE_FRAME} for ${OPTO_DURATION_FRAMES} frames. Record this run, then run the no-pulse control with the same seed and ablation set.`
      : `Matched control started. Neuron-only, seed ${experimentSeed}; no AVA pulse. Keep game and ablation set unchanged when comparing with the pulse trial.`;
  };
  document.getElementById('btn-controlled-control')?.addEventListener('click', () => startControlledTrial(false));
  document.getElementById('btn-controlled-pulse')?.addEventListener('click', () => startControlledTrial(true));

  document.getElementById('btn-record-result')?.addEventListener('click', () => {
    const isFlappy = currentGameMode === 'flappy';
    const score = isFlappy ? flappy.score : chemotaxis.score;
    const durationSeconds = Math.max(0, Math.round((Date.now() - runStartedAt) / 1000));
    const record = {
      id: Date.now(),
      game: isFlappy ? 'Flappy' : 'Chemotaxis',
      controller: controllerMode === 'assisted' ? 'Assisted baseline' : (controllerMode === 'strict' ? 'Neuron-only (untrained, no overrides)' : 'Neural sensorimotor (trained hybrid)'),
      condition: controlledProtocol?.condition || 'manual run',
      protocol: controlledProtocol ? { ...controlledProtocol } : null,
      score,
      durationSeconds,
      frames: isFlappy ? flappy.frames : chemotaxis.frames,
      flaps: isFlappy ? flappy.flap_count : null,
      neuralFlapDecisions: isFlappy && controllerMode !== 'assisted' ? flappy.neuralFlapDecisions : null,
      neuralMotorBaseline: isFlappy && controllerMode === 'strict' ? flappy.neuralMotorBaseline : null,
      lastNeuralDrive: isFlappy && controllerMode !== 'assisted' ? flappy.lastNeuralDrive : null,
      peakAbsNeuralDrive: isFlappy && controllerMode !== 'assisted' ? flappy.peakAbsNeuralDrive : null,
      strictMotorDriveStats: isFlappy && controllerMode === 'strict' ? {
        min: flappy.motorDriveMin, max: flappy.motorDriveMax,
        mean: flappy.motorDriveSamples ? flappy.motorDriveSum / flappy.motorDriveSamples : null,
        samples: flappy.motorDriveSamples, threshold: 0.180,
        thresholdCrossings: flappy.motorThresholdCrossings,
        thresholdCrossingsDuringCooldown: flappy.motorThresholdCrossingsDuringCooldown,
        thresholdCrossingsActionEligible: flappy.motorThresholdCrossingsActionEligible,
        flapActions: flappy.neuralFlapDecisions
      } : null,
      decoderFeatureCoverage: isFlappy && controllerMode === 'neural' ? flappy.decoderFeatureCoverage : null,
      chemotaxisDecoderFeatureCoverage: !isFlappy && controllerMode === 'neural' ? chemotaxis.decoderFeatureCoverage : null,
      lastChemotaxisDecoderTurn: !isFlappy && controllerMode !== 'assisted' ? chemotaxis.lastDecoderTurn : null,
      lastReadoutProbability: isFlappy && controllerMode === 'neural' ? flappy.lastReadoutProbability : null,
      seed: experimentSeed,
      ablatedAtStartNames: (isFlappy ? flappy.ablatedAtStartNames : chemotaxis.ablatedAtStartNames).slice(),
      ablated: ablatedSet.size,
      ablatedNames: [...ablatedSet].map(i => neuronNames[i]).sort(),
      turns: isFlappy ? null : chemotaxis.turns,
      turnEpisodes: isFlappy ? null : chemotaxis.turnEpisodes,
      pirouettes: isFlappy ? null : chemotaxis.pirouettes,
      bodyLength: isFlappy ? null : chemotaxis.body.length,
      gameOverReason: isFlappy ? (flappy.is_game_over ? (flappy.game_over_reason || 'collision') : '') : chemotaxis.game_over_reason,
      intervention: lastIntervention,
      interventionLog: interventionLog.slice(),
      actionLog: (isFlappy ? flappy.actionLog : chemotaxis.actionLog).slice(),
      optogenetics: optoExperiment ? { ...optoExperiment } : null
    };
    experimentRecords.unshift(record);
    renderExperimentResults();
  });
  document.getElementById('btn-clear-results')?.addEventListener('click', () => {
    experimentRecords = [];
    renderExperimentResults();
  });
  window.addEventListener('keydown', (event) => {
    if (event.key.toLowerCase() === 'r') {
      if (currentGameMode === 'flappy') resetFlappy(); else resetChemotaxis();
    }
  });

  // Ablate Command Interneurons (AVA / AVD)
  document.getElementById('btn-ablate-ava')?.addEventListener('click', () => {
    ['AVAL', 'AVAR', 'AVDL', 'AVDR'].forEach(n => {
      if (nameToIdx[n] !== undefined) ablatedSet.add(nameToIdx[n]);
    });
    recordIntervention('Group laser ablation: AVAL, AVAR, AVDL, AVDR');
    const note = document.getElementById('controller-note');
    if (note) note.textContent = `${lastIntervention}. Restart for a matched trial. In Assisted baseline, actions bypass the neural network, so behavior is not expected to change; compare Neuron-only/trained-hybrid modes for network-dependent effects.`;
  });

  // Restore All
  document.getElementById('btn-restore-all')?.addEventListener('click', () => {
    ablatedSet.clear();
    I_opto.fill(0);
    if (optoTimeout) { clearTimeout(optoTimeout); optoTimeout = null; }
    if (optoExperiment && !optoExperiment.endedAt) finishOptogeneticPulse('manually stopped by Restore All');
    recordIntervention('Restore all neurons; optogenetic current cleared');
    const note = document.getElementById('controller-note');
    if (note) note.textContent = `${lastIntervention}. Restart a game for a clean comparison. The current ablation set is now empty.`;
  });

  // Manual pulse remains available for exploratory runs; controlled buttons above
  // should be preferred for matched comparisons.
  document.getElementById('btn-opto-ava')?.addEventListener('click', () => startOptogeneticPulse('manual AVA pulse'));
}

function renderExperimentResults() {
  const el = document.getElementById('experiment-results');
  if (!el) return;
  if (!experimentRecords.length) {
    el.textContent = 'No runs recorded yet. Choose a seed, restart, run a condition, then record it.';
    return;
  }
  el.innerHTML = experimentRecords.slice(0, 8).map((r, i) => {
    const pulse = r.optogenetics;
    const ablatedText = r.ablatedNames.length ? r.ablatedNames.join(', ') : 'none';
    const historyText = (r.interventionLog || []).map(e => `${new Date(e.at).toLocaleTimeString()} — ${e.message} [ablated: ${e.ablated.join(', ') || 'none'}]`).join('<br>');
    const actionEvents = r.actionLog || [];
    const actionText = actionEvents.slice(-12).map(e => {
      const magnitude = e.appliedTurnRadians === undefined ? '' : ` · applied turn ${Number(e.appliedTurnRadians).toFixed(3)} rad`;
      const evidence = e.evidence ? ` · evidence ${JSON.stringify(e.evidence)}` : '';
      return `Frame ${e.frame}: ${e.action} — ${e.reason || 'policy action'}${magnitude}${e.threshold == null ? '' : ` · threshold ${e.threshold}`} ${e.mode ? `(${e.mode})` : ''}${e.environmentSnapshot ? ` · state ${JSON.stringify(e.environmentSnapshot)}` : ''}${evidence}`;
    }).join('<br>');
    const strictStats = r.strictMotorDriveStats
      ? `Neuron-only motor drive: min ${Number(r.strictMotorDriveStats.min).toFixed(3)}, max ${Number(r.strictMotorDriveStats.max).toFixed(3)}, mean ${Number(r.strictMotorDriveStats.mean).toFixed(3)}; threshold ${r.strictMotorDriveStats.threshold} crossed ${r.strictMotorDriveStats.thresholdCrossings}/${r.strictMotorDriveStats.samples} frames (${r.strictMotorDriveStats.thresholdCrossingsDuringCooldown} during cooldown, ${r.strictMotorDriveStats.thresholdCrossingsActionEligible} action-eligible); flap actions ${r.strictMotorDriveStats.flapActions}.`
      : '';
    const pulseText = pulse
      ? `AVA baseline/peak/end: ${Number(pulse.baselineAvaVoltage).toFixed(1)}/${Number(pulse.peakAvaVoltage).toFixed(1)}/${Number(pulse.endAvaVoltage ?? pulse.latestAvaVoltage).toFixed(1)} mV; DB−VB baseline/end: ${(pulse.baselineMotor.db - pulse.baselineMotor.vb).toFixed(2)}/${((pulse.endMotor ?? pulse.latestMotor).db - (pulse.endMotor ?? pulse.latestMotor).vb).toFixed(2)}; ${r.game === 'Flappy' ? `flap Δ: ${pulse.flapDelta ?? 'pulse still active'}` : `heading Δ: ${pulse.headingDeltaRad === null || pulse.headingDeltaRad === undefined ? 'pulse still active' : pulse.headingDeltaRad.toFixed(3) + ' rad'}`}; score Δ: ${pulse.scoreDelta ?? 'pulse still active'}`
      : (r.protocol && !r.protocol.pulseScheduled ? 'Matched no-pulse control (no stimulation scheduled)' : (r.protocol?.pulseScheduled && !r.protocol.pulseStarted ? (r.protocol.pulseAttempted ? 'Pulse attempt failed (check AVA availability)' : 'Pulse scheduled but never started (run ended before trigger)') : 'No optogenetic pulse recorded'));
    return `<div style="padding:7px 0;border-bottom:1px solid #1e293b;">
      <b style="color:#e2e8f0">#${i + 1} ${r.game} · ${r.controller}</b><br>
      Condition: ${r.condition || 'manual run'}${r.protocol ? ` · pulse scheduled: ${r.protocol.pulseScheduled ? `yes, frame ${r.protocol.pulseAtFrame}` : 'no'}` : ''}<br>
      Score: ${r.score} · Elapsed: ${r.durationSeconds}s · Seed: ${r.seed} · Frames: ${r.frames}${r.turns === null ? '' : ` · Reorientation events: ${r.turns} · Turn episodes: ${r.turnEpisodes ?? 'n/a'} · Pirouettes: ${r.pirouettes ?? 0} · Body length: ${r.bodyLength ?? 'n/a'} · Decoder coverage: ${r.chemotaxisDecoderFeatureCoverage == null ? 'n/a' : (100 * r.chemotaxisDecoderFeatureCoverage).toFixed(0) + '%'} · Last decoded turn: ${r.lastChemotaxisDecoderTurn == null ? 'n/a' : Number(r.lastChemotaxisDecoderTurn).toFixed(3) + ' rad'} · End: ${r.gameOverReason || 'not ended'}`}${r.flaps === null ? '' : ` · Flaps: ${r.flaps}`}${r.neuralFlapDecisions === null || r.neuralFlapDecisions === undefined ? '' : ` · Neural flap decisions: ${r.neuralFlapDecisions}${r.lastReadoutProbability !== null && r.lastReadoutProbability !== undefined ? ` · Final readout probability: ${Number(r.lastReadoutProbability).toFixed(3)} · Peak readout probability: ${Number(r.peakAbsNeuralDrive).toFixed(3)} · Feature coverage: ${(100 * Number(r.decoderFeatureCoverage ?? 1)).toFixed(0)}%` : ` · Motor baseline: ${Number(r.neuralMotorBaseline).toFixed(3)} · Peak baseline-adjusted motor drive: ${Number(r.peakAbsNeuralDrive).toFixed(3)}`}`}<br>
      Ablated at trial start (${(r.ablatedAtStartNames || []).length}): ${(r.ablatedAtStartNames || []).join(', ') || 'none'}<br>
      Ablated at record time (${r.ablated}): ${ablatedText}<br>
      ${strictStats ? `<br><span style="color:#a5b4cc">${strictStats}</span>` : ''}
      <span>${r.intervention}</span><br><span style="color:#cbd5e1">Optogenetics: ${pulseText}</span>
      ${actionEvents.length ? `<details style="margin-top:4px"><summary>Action trigger log (${actionEvents.length}; latest ${Math.min(12, actionEvents.length)} shown)</summary><span>${actionText}</span></details>` : ''}
      ${historyText ? `<details style="margin-top:4px"><summary>Intervention history (${r.interventionLog.length})</summary><span>${historyText}</span></details>` : ''}
    </div>`;
  }).join('');
}

function updateAblatedListUI() {
  const badge = document.getElementById('ablated-count-badge');
  if (badge) badge.textContent = `${ablatedSet.size} ABLATED`;
  const list = document.getElementById('ablated-neuron-list');
  if (list) list.textContent = ablatedSet.size ? [...ablatedSet].map(i => neuronNames[i]).sort().join(', ') : 'No neurons ablated';
}

// Main 60 FPS Render Loop
function animate() {
  requestAnimationFrame(animate);

  // Schedule stimulation before the target simulation frame's network update.
  // This makes the recorded pulse frame count match frames that actually receive current.
  const gameWasOver = currentGameMode === 'flappy' ? flappy.is_game_over : chemotaxis.is_game_over;
  const frameBefore = currentGameFrame();
  if (!gameWasOver && controlledProtocol?.pulseScheduled && !controlledProtocol.pulseAttempted &&
      frameBefore + 1 >= controlledProtocol.pulseAtFrame) {
    controlledProtocol.pulseAttempted = true;
    controlledProtocol.pulseStarted = startOptogeneticPulse('Scheduled controlled AVA pulse');
  }

  // Update Game and Biophysics
  if (currentGameMode === 'flappy') {
    updateFlappyGame();
  } else {
    updateChemotaxisGame();
  }

  const frameAfter = currentGameFrame();
  if (frameAfter > frameBefore) updateOptogeneticTelemetry();
  const gameIsOver = currentGameMode === 'flappy' ? flappy.is_game_over : chemotaxis.is_game_over;
  if (gameIsOver && optoExperiment && !optoExperiment.endedAt) {
    I_opto.fill(0);
    finishOptogeneticPulse('game ended before the scheduled pulse duration completed');
  }

  // Update 3D Neuron Visuals based on Live Voltages
  if (neuronMeshes.length > 0) {
    neuronNames.forEach((name, i) => {
      const mesh = neuronMeshes[i];
      if (ablatedSet.has(i)) {
        mesh.material.color.setHex(0x334155);
        mesh.material.emissive.setHex(0x1e293b);
        mesh.scale.set(0.65, 0.65, 0.65);
      } else {
        const normV = Math.max(0, Math.min(1, (V[i] - (-65.0)) / 35.0));
        // Pulse size and glow
        const scale = 0.85 + 0.65 * normV;
        mesh.scale.set(scale, scale, scale);
        mesh.material.color.setHex(mesh.userData.baseColor);
        mesh.material.emissiveIntensity = 0.25 + 0.95 * normV;
      }
    });
  }

  // Render 2D Overlays
  renderGameCanvas();
  renderOscilloscope();

  // Render 3D Scene
  renderer.render(scene, camera);
}

window.addEventListener('DOMContentLoaded', init);
