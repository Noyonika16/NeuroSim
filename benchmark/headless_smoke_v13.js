// Headless browser-JS smoke benchmark. Runs the real web/app.js game loop in a
// Node VM with the included connectome and decoder files; no DOM rendering.
// Expected runtime is roughly 1-2 minutes for the default multi-seed matrix.
const fs = require('fs');
const vm = require('vm');
const path = require('path');
const root = path.resolve(__dirname, '..');
const data = JSON.parse(fs.readFileSync(path.join(root, 'web/connectome_data.json'), 'utf8'));
const flappyDecoder = JSON.parse(fs.readFileSync(path.join(root, 'web/flappy_decoder.json'), 'utf8'));
const chemoDecoder = JSON.parse(fs.readFileSync(path.join(root, 'web/chemotaxis_decoder.json'), 'utf8'));
const elements = { 'experiment-seed': { value: '20261027' } };
const document = { getElementById: id => elements[id] || null, querySelectorAll: () => [], addEventListener: () => {}, createElement: () => ({}) };
const window = { CONNECTOME_DATA: data, FLAPPY_DECODER: flappyDecoder, CHEMOTAXIS_DECODER: chemoDecoder, addEventListener: () => {} };
const context = { console, Math, Date, Float32Array, Uint8Array, Set, Map, Array, Object, Number, String, JSON, window, document, requestAnimationFrame: () => {}, setTimeout: () => 0, clearTimeout: () => {}, performance: { now: () => 0 } };
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(root, 'web/app.js'), 'utf8'), context);
vm.runInContext('connectomeData=window.CONNECTOME_DATA; flappyDecoder=window.FLAPPY_DECODER; chemotaxisDecoder=window.CHEMOTAXIS_DECODER; setupBiophysicalMatrices();', context);

function trial(game, mode, seed, ablated = [], maxFrames = 1800) {
  elements['experiment-seed'].value = String(seed);
  vm.runInContext(`ablatedSet.clear(); ${JSON.stringify(ablated)}.forEach(n=>{if(nameToIdx[n]!==undefined)ablatedSet.add(nameToIdx[n])}); experimentSeed=${seed}; controllerMode=${JSON.stringify(mode)}; currentGameMode=${JSON.stringify(game)}; ${game === 'flappy' ? 'resetFlappy()' : 'resetChemotaxis()'}`, context);
  for (let frame = 0; frame < maxFrames; frame++) {
    vm.runInContext(game === 'flappy' ? 'updateFlappyGame()' : 'updateChemotaxisGame()', context);
    if (vm.runInContext(game === 'flappy' ? 'flappy.is_game_over' : 'chemotaxis.is_game_over', context)) break;
  }
  const raw = game === 'flappy'
    ? `JSON.stringify({score:flappy.score,frames:flappy.frames,flaps:flappy.flap_count,reason:flappy.game_over_reason || 'harness frame cap',featureCoverage:flappy.decoderFeatureCoverage,strictMotorDriveStats:controllerMode==='strict'?{min:flappy.motorDriveMin,max:flappy.motorDriveMax,mean:flappy.motorDriveSamples?flappy.motorDriveSum/flappy.motorDriveSamples:null,samples:flappy.motorDriveSamples,threshold:0.180,thresholdCrossings:flappy.motorThresholdCrossings,thresholdCrossingsDuringCooldown:flappy.motorThresholdCrossingsDuringCooldown,thresholdCrossingsActionEligible:flappy.motorThresholdCrossingsActionEligible,flapActions:flappy.neuralFlapDecisions}:null})`
    : `JSON.stringify({score:chemotaxis.score,frames:chemotaxis.frames,reorientations:chemotaxis.turns,turnEpisodes:chemotaxis.turnEpisodes,pirouettes:chemotaxis.pirouettes,bodyLength:chemotaxis.body.length,reason:chemotaxis.game_over_reason || 'harness frame cap',featureCoverage:chemotaxis.decoderFeatureCoverage})`;
  return { game, controller: mode === 'strict' ? 'neuron-only' : (mode === 'neural' ? 'trained-hybrid' : mode), seed, ablated, ...JSON.parse(vm.runInContext(raw, context)) };
}

const seeds = [20261027, 1, 42, 100, 20261028, 123456];
const flappy = seeds.map(seed => trial('flappy', 'neural', seed, [], 800));
const assistedParity = [
  trial('flappy', 'assisted', 20261027, [], 800),
  trial('flappy', 'assisted', 20261027, ['AVAL', 'AVAR', 'AVDL', 'AVDR'], 800),
  trial('chemotaxis', 'assisted', 20261027, [], 700),
  trial('chemotaxis', 'assisted', 20261027, ['AVAL', 'AVAR', 'AVDL', 'AVDR'], 700)
];
const ablation = [
  trial('flappy', 'neural', 20261027, ['AVAL', 'AVAR', 'AVDL', 'AVDR'], 800),
  trial('flappy', 'neural', 20261027, ['CEPDL', 'CEPDR', 'CEPVL', 'CEPVR'], 800),
  trial('flappy', 'strict', 20261027, [], 500),
  trial('chemotaxis', 'strict', 20261027, [], 500)
];
const chemotaxis = seeds.map(seed => trial('chemotaxis', 'neural', seed, [], 1800));
const strictChemotaxisSeeds = [20261027, 42, 123456];
const strictChemotaxis = strictChemotaxisSeeds.map(seed => trial('chemotaxis', 'strict', seed, [], 700));
const food = chemotaxis.map(r => r.score);
const report = {
  version: 'v13.3',
  testType: 'headless Node VM executing web/app.js; no browser rendering',
  seedSet: seeds,
  readoutTraining: {
    chemotaxisRows: 13898,
    trainingSeedCount: chemoDecoder.training_seed_count,
    heldoutSeedCount: chemoDecoder.heldout_seed_count,
    featureCount: chemoDecoder.feature_names.length,
    heldoutActionImitationMetrics: chemoDecoder.heldout_metrics,
    warning: 'Metrics use the deployed +/-0.045 rad/frame runtime clip. Teacher turn imitation metrics are not foraging success metrics and do not validate biological fidelity.'
  },
  flappyNeural800FrameTrials: flappy,
  assistedAblationParityControls: assistedParity,
  ablationAndStrictControls: ablation,
  chemotaxisNeuralFullEpisodes: chemotaxis,
  chemotaxisNeuronOnlyTrials700FrameCap: strictChemotaxis,
  summaries: {
    flappyMeanScoreAt800FrameHarnessCap: flappy.reduce((a, r) => a + r.score, 0) / flappy.length,
    assistedFlappyAblationParity: JSON.stringify(assistedParity[0] && {score:assistedParity[0].score,frames:assistedParity[0].frames,flaps:assistedParity[0].flaps,reason:assistedParity[0].reason}) === JSON.stringify(assistedParity[1] && {score:assistedParity[1].score,frames:assistedParity[1].frames,flaps:assistedParity[1].flaps,reason:assistedParity[1].reason}),
    assistedChemotaxisAblationParity: JSON.stringify(assistedParity[2] && {score:assistedParity[2].score,frames:assistedParity[2].frames,bodyLength:assistedParity[2].bodyLength,reason:assistedParity[2].reason}) === JSON.stringify(assistedParity[3] && {score:assistedParity[3].score,frames:assistedParity[3].frames,bodyLength:assistedParity[3].bodyLength,reason:assistedParity[3].reason}),
    chemotaxisMeanFood: food.reduce((a, n) => a + n, 0) / food.length,
    chemotaxisTimeLimitRuns: chemotaxis.filter(r => r.reason.startsWith('Time limit')).length,
    chemotaxisBoundaryCollisionRuns: chemotaxis.filter(r => r.reason === 'Boundary collision').length,
    chemotaxisSelfCollisionRuns: chemotaxis.filter(r => r.reason === 'Self-collision').length,
    neuronOnlyChemotaxisBoundaryCollisionRuns: strictChemotaxis.filter(r => r.reason === 'Boundary collision').length,
    neuronOnlyChemotaxisSelfCollisionRuns: strictChemotaxis.filter(r => r.reason === 'Self-collision').length,
    neuronOnlyChemotaxisMeanFood: strictChemotaxis.reduce((a, r) => a + r.score, 0) / strictChemotaxis.length
  },
  limitations: ['Flappy trials are capped by the harness at 800 frames, not run to natural game-over.', 'Readout training imitates the assisted policy; it does not train connectome synaptic weights.', 'Neuron-only Flappy motor-drive distribution is recorded at every simulation frame; its threshold remains unchanged pending controlled evaluation. Neuron-only Flappy remains an untrained baseline and may collide with pipes.', 'Neuron-only Chemotaxis has no boundary-turn override; wall proximity is only encoded as synthetic sensory input. Headless smoke results do not replace live-browser validation or establish biological fidelity.']
};
fs.writeFileSync(path.join(__dirname, 'v13_headless_validation.json'), JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report, null, 2));
