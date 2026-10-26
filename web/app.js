/**
 * NEURO-ELEGANS: 3D Connectome & Cybernetic Game Engine
 * Authentically simulates all 302 biological neurons of C. elegans in real-time.
 */

// Global State
let connectomeData = null;
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

// Game Mode: 'flappy' or 'chemotaxis'
let currentGameMode = 'flappy';

// --- Flappy Nematode Game State ---
let flappy = {
  bird_x: 75,
  bird_y: 120,
  bird_vy: 0,
  gravity: 0.42,
  flap_strength: -6.5,
  pipe_w: 50,
  pipe_gap: 110,
  pipe_speed: 3.0,
  pipes: [],
  score: 0,
  high_score: 0,
  is_game_over: false,
  flap_cooldown: 0,
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
  pirouettes: 0,
  prev_conc: 0,
  body: []
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
  scene.background = new THREE.Color(0x07090e);
  scene.fog = new THREE.FogExp2(0x07090e, 0.0012);

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
    } else {
      ablatedSet.add(idx);
    }
    updateAblatedListUI();
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

  // 2. Compute Currents & Updates
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
    const noise = (Math.random() - 0.5) * 0.4;
    const totalCurrent = dV + I_sensory[i] + I_opto[i] + noise;

    V[i] += (totalCurrent / TAU) * dt;

    // Biological voltage clamp bounds (-80mV to +25mV)
    if (V[i] < -80.0) V[i] = -80.0;
    if (V[i] > 25.0) V[i] = 25.0;
  }
}

// --- Retro Game Loops ---
function initGames() {
  resetFlappy();
  resetChemotaxis();
}

function resetFlappy() {
  flappy.bird_x = 75;
  flappy.bird_y = 120;
  flappy.bird_vy = 0;
  flappy.score = 0;
  flappy.is_game_over = false;
  flappy.pipes = [
    { x: 380, gap_y: 60, passed: false },
    { x: 570, gap_y: 90, passed: false }
  ];
  flappy.body_segments = [];
  for (let i = 0; i < 12; i++) {
    flappy.body_segments.push({ x: flappy.bird_x - i * 5, y: flappy.bird_y });
  }
}

function resetChemotaxis() {
  chemotaxis.worm_x = 210;
  chemotaxis.worm_y = 120;
  chemotaxis.heading = 0;
  chemotaxis.score = 0;
  chemotaxis.pirouettes = 0;
  chemotaxis.food_x = 70 + Math.random() * 280;
  chemotaxis.food_y = 40 + Math.random() * 160;
  chemotaxis.body = [];
  for (let i = 0; i < 16; i++) {
    chemotaxis.body.push({ x: chemotaxis.worm_x - i * 5, y: chemotaxis.worm_y });
  }
}

function updateFlappyGame() {
  if (flappy.is_game_over) {
    if (Math.random() < 0.03) resetFlappy(); // auto-restart
    return;
  }

  // Find upcoming pipe
  let pipe = flappy.pipes[0];
  for (let p of flappy.pipes) {
    if (p.x + flappy.pipe_w > flappy.bird_x - 12) {
      pipe = p;
      break;
    }
  }

  const pipeDistX = pipe.x - flappy.bird_x;
  const gapCenterY = pipe.gap_y + flappy.pipe_gap * 0.5;

  // Sensory injection into connectome
  I_sensory.fill(0);
  const proximity = Math.max(0, 1.0 - (pipeDistX / 300.0));
  const ashCurrent = 28.0 * Math.pow(proximity, 1.8);
  injectCurrent("ASHL", ashCurrent);
  injectCurrent("ASHR", ashCurrent);

  // Vertical alignment
  const deltaY = (flappy.bird_y - gapCenterY) / (flappy.pipe_gap * 0.5);
  if (deltaY > 0) { // Below gap -> need to flap up!
    const stim = Math.min(35.0, 20.0 * deltaY * (0.4 + 0.6 * proximity));
    injectCurrent("CEPVL", stim);
    injectCurrent("CEPVR", stim);
  } else { // Above gap -> dive down
    const stim = Math.min(35.0, 20.0 * Math.abs(deltaY) * (0.4 + 0.6 * proximity));
    injectCurrent("CEPDL", stim);
    injectCurrent("CEPDR", stim);
  }

  // Floor proximity
  if (flappy.bird_y > 210) {
    injectCurrent("ALML", 35.0);
    injectCurrent("ALMR", 35.0);
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

  // Update meter HUD
  updateGauges(ashCurrent, flapDrive, dbMean, vbMean);

  // Flap Action
  if (flappy.flap_cooldown > 0) {
    flappy.flap_cooldown--;
  } else if (flapDrive > 3.0) {
    flappy.bird_vy = flappy.flap_strength;
    flappy.flap_cooldown = 7;
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
      gap_y: 40 + Math.random() * 80,
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
  if (flappy.bird_y < 8 || flappy.bird_y > 232) {
    flappy.is_game_over = true;
  }
  for (let p of flappy.pipes) {
    if (flappy.bird_x + 9 > p.x && flappy.bird_x - 9 < p.x + flappy.pipe_w) {
      if (flappy.bird_y - 9 < p.gap_y || flappy.bird_y + 9 > p.gap_y + flappy.pipe_gap) {
        flappy.is_game_over = true;
      }
    }
  }
}

function updateChemotaxisGame() {
  // Distance and concentration
  const dx = chemotaxis.food_x - chemotaxis.worm_x;
  const dy = chemotaxis.food_y - chemotaxis.worm_y;
  const dist = Math.sqrt(dx * dx + dy * dy);
  const conc = Math.exp(-(dist * dist) / (2.0 * 80.0 * 80.0));
  const dC = conc - chemotaxis.prev_conc;
  chemotaxis.prev_conc = conc;

  I_sensory.fill(0);
  if (dC > 0) {
    injectCurrent("ASEL", Math.min(30.0, 150.0 * dC));
    injectCurrent("AWCL", 15.0 * conc);
  } else {
    injectCurrent("ASER", Math.min(30.0, 150.0 * Math.abs(dC)));
  }

  // Wall avoidance
  if (chemotaxis.worm_x < 30 || chemotaxis.worm_x > 390 || chemotaxis.worm_y < 30 || chemotaxis.worm_y > 210) {
    injectCurrent("ASHL", 30.0);
    injectCurrent("ASHR", 30.0);
  }

  for (let s = 0; s < 4; s++) {
    stepBiophysicalNetwork(0.002);
  }

  const vAva = V[nameToIdx["AVAL"] || 0];
  const vAvb = V[nameToIdx["AVBL"] || 0];
  const isPirouette = (vAva > -38.0);

  let speed = 1.8;
  let turnRate = 0;
  if (isPirouette) {
    speed = 0.8;
    turnRate = (Math.random() - 0.5) * 0.45;
    chemotaxis.pirouettes++;
  } else {
    speed = 1.6 + 0.08 * (vAvb - (-50.0));
    const smdd = (V[nameToIdx["SMDDL"] || 0] - V[nameToIdx["SMDDR"] || 0]) * 0.04;
    turnRate = smdd;
  }

  chemotaxis.heading += turnRate;
  chemotaxis.worm_x += speed * Math.cos(chemotaxis.heading);
  chemotaxis.worm_y += speed * Math.sin(chemotaxis.heading);

  // Boundary clamp
  chemotaxis.worm_x = Math.max(15, Math.min(405, chemotaxis.worm_x));
  chemotaxis.worm_y = Math.max(15, Math.min(225, chemotaxis.worm_y));

  // Body follow
  chemotaxis.body[0] = { x: chemotaxis.worm_x, y: chemotaxis.worm_y };
  for (let i = 1; i < chemotaxis.body.length; i++) {
    const prev = chemotaxis.body[i - 1];
    const curr = chemotaxis.body[i];
    const bdx = prev.x - curr.x;
    const bdy = prev.y - curr.y;
    const bdist = Math.sqrt(bdx * bdx + bdy * bdy);
    if (bdist > 0.01) {
      curr.x = prev.x - (bdx / bdist) * 5.0;
      curr.y = prev.y - (bdy / bdist) * 5.0;
    }
  }

  // Food check
  if (dist < 18.0) {
    chemotaxis.score++;
    chemotaxis.food_x = 60 + Math.random() * 300;
    chemotaxis.food_y = 40 + Math.random() * 160;
  }
}

// Render Retro Game Canvas
function renderGameCanvas() {
  const canvas = document.getElementById('game-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.fillStyle = '#050811';
  ctx.fillRect(0, 0, w, h);

  if (currentGameMode === 'flappy') {
    // Draw Pipes
    ctx.fillStyle = '#1e293b';
    ctx.strokeStyle = '#00f2fe';
    ctx.lineWidth = 1.5;

    for (let p of flappy.pipes) {
      // Top Pipe
      ctx.fillRect(p.x, 0, flappy.pipe_w, p.gap_y);
      ctx.strokeRect(p.x, 0, flappy.pipe_w, p.gap_y);
      // Bottom Pipe
      const bTop = p.gap_y + flappy.pipe_gap;
      ctx.fillRect(p.x, bTop, flappy.pipe_w, h - bTop);
      ctx.strokeRect(p.x, bTop, flappy.pipe_w, h - bTop);
    }

    // Draw Undulating Nematode Body
    ctx.strokeStyle = '#00ff87';
    ctx.lineWidth = 4;
    ctx.lineCap = 'round';
    ctx.beginPath();
    flappy.body_segments.forEach((seg, i) => {
      if (i === 0) ctx.moveTo(seg.x, seg.y);
      else ctx.lineTo(seg.x, seg.y);
    });
    ctx.stroke();

    // Glowing Head
    ctx.fillStyle = '#00f2fe';
    ctx.beginPath();
    ctx.arc(flappy.bird_x, flappy.bird_y, 6.5, 0, 2 * Math.PI);
    ctx.fill();

    // Score HUD
    document.getElementById('hud-score').textContent = `SCORE: ${flappy.score}`;
    document.getElementById('hud-status').textContent = flappy.is_game_over ? 'COLLISION DETECTED' : 'NAVIGATING OBSTACLES';
    document.getElementById('hud-status').style.color = flappy.is_game_over ? '#ff3366' : '#00ff87';
  } else {
    // Chemotaxis Arena
    // Draw Food Odor Gradient Halo
    const grad = ctx.createRadialGradient(chemotaxis.food_x, chemotaxis.food_y, 5, chemotaxis.food_x, chemotaxis.food_y, 90);
    grad.addColorStop(0, 'rgba(255, 179, 0, 0.45)');
    grad.addColorStop(1, 'rgba(255, 179, 0, 0)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(chemotaxis.food_x, chemotaxis.food_y, 90, 0, 2 * Math.PI);
    ctx.fill();

    // Food Pellet
    ctx.fillStyle = '#ffb300';
    ctx.beginPath();
    ctx.arc(chemotaxis.food_x, chemotaxis.food_y, 7, 0, 2 * Math.PI);
    ctx.fill();

    // Undulating Worm Body
    ctx.strokeStyle = '#00ff87';
    ctx.lineWidth = 4.5;
    ctx.lineCap = 'round';
    ctx.beginPath();
    chemotaxis.body.forEach((seg, i) => {
      if (i === 0) ctx.moveTo(seg.x, seg.y);
      else ctx.lineTo(seg.x, seg.y);
    });
    ctx.stroke();

    // Head
    ctx.fillStyle = '#00f2fe';
    ctx.beginPath();
    ctx.arc(chemotaxis.worm_x, chemotaxis.worm_y, 6, 0, 2 * Math.PI);
    ctx.fill();

    document.getElementById('hud-score').textContent = `FOOD HARVESTED: ${chemotaxis.score}`;
    document.getElementById('hud-status').textContent = `PIROUETTES: ${chemotaxis.pirouettes}`;
    document.getElementById('hud-status').style.color = '#ffb300';
  }
}

// Render Oscilloscope
function renderOscilloscope() {
  const canvas = document.getElementById('scope-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.fillStyle = '#050811';
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
    if (n.startsWith(prefix)) {
      sum += V[i];
      count++;
    }
  });
  return count > 0 ? sum / count : E_LEAK;
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

  // Ablate Command Interneurons (AVA / AVD)
  document.getElementById('btn-ablate-ava')?.addEventListener('click', () => {
    ['AVAL', 'AVAR', 'AVDL', 'AVDR'].forEach(n => {
      if (nameToIdx[n] !== undefined) ablatedSet.add(nameToIdx[n]);
    });
    updateAblatedListUI();
  });

  // Restore All
  document.getElementById('btn-restore-all')?.addEventListener('click', () => {
    ablatedSet.clear();
    I_opto.fill(0);
    updateAblatedListUI();
  });

  // Optogenetic Blue Flash (ChR2 on AVA)
  document.getElementById('btn-opto-ava')?.addEventListener('click', () => {
    ['AVAL', 'AVAR'].forEach(n => {
      const idx = nameToIdx[n];
      if (idx !== undefined) I_opto[idx] = 40.0;
    });
    setTimeout(() => {
      ['AVAL', 'AVAR'].forEach(n => {
        const idx = nameToIdx[n];
        if (idx !== undefined) I_opto[idx] = 0;
      });
    }, 450);
  });
}

function updateAblatedListUI() {
  const badge = document.getElementById('ablated-count-badge');
  if (badge) badge.textContent = `${ablatedSet.size} ABLATED`;
}

// Main 60 FPS Render Loop
function animate() {
  requestAnimationFrame(animate);

  // Update Game and Biophysics
  if (currentGameMode === 'flappy') {
    updateFlappyGame();
  } else {
    updateChemotaxisGame();
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
