// Collect assisted-policy labels and frozen-connectome voltage features for a
// calibrated Chemotaxis turn readout. This is offline teacher distillation only.
const fs = require('fs');
const vm = require('vm');
const path = require('path');
const root = path.resolve(__dirname, '..');
const connectome = JSON.parse(fs.readFileSync(path.join(root, 'web/connectome_data.json'), 'utf8'));
const elements = { 'experiment-seed': { value: '20261027' } };
const document = {
  getElementById: id => elements[id] || null,
  querySelectorAll: () => [],
  addEventListener: () => {},
  createElement: () => ({})
};
const window = { CONNECTOME_DATA: connectome, addEventListener: () => {}, __COLLECT_CHEMO_ROWS: true, __CHEMO_TRAINING_ROWS: [] };
const context = {
  console, Math, Date, Float32Array, Uint8Array, Set, Map, Array, Object, Number, String, JSON,
  window, document, requestAnimationFrame: () => {}, setTimeout: () => 0, clearTimeout: () => {},
  performance: { now: () => 0 }
};
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(root, 'web/app.js'), 'utf8'), context);
vm.runInContext('connectomeData = window.CONNECTOME_DATA; setupBiophysicalMatrices(); controllerMode = "assisted"; currentGameMode = "chemotaxis";', context);
const seeds = [20261027, 1, 42, 100, 20261028, 123456, 17, 29, 73, 314, 2718, 8128, 65537, 2026, 9001, 121, 222, 333, 444, 555];
for (const seed of seeds) {
  elements['experiment-seed'].value = String(seed);
  vm.runInContext(`ablatedSet.clear(); experimentSeed = ${seed}; controllerMode = "assisted"; currentGameMode = "chemotaxis"; resetChemotaxis();`, context);
  for (let frame = 0; frame < 700; frame++) {
    vm.runInContext('updateChemotaxisGame();', context);
    if (vm.runInContext('chemotaxis.is_game_over', context)) break;
  }
}
const rows = window.__CHEMO_TRAINING_ROWS;
const out = path.join(root, 'benchmark', 'chemotaxis_readout_training.jsonl');
fs.writeFileSync(out, rows.map(row => JSON.stringify(row)).join('\n') + '\n');
console.log(JSON.stringify({ rows: rows.length, seeds: seeds.length, positiveTurns: rows.filter(r => r.target > 0).length, negativeTurns: rows.filter(r => r.target < 0).length, output: out }, null, 2));
