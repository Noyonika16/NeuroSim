// Reproducible offline data collector for the trained Flappy action readout.
// Requires Node.js. The teacher is the transparent Assisted baseline; the
// runtime readout uses only the 302 simulated neuron voltages.
const fs = require('fs');
const vm = require('vm');
const path = require('path');
const root = path.resolve(__dirname, '..');
const connectome = JSON.parse(fs.readFileSync(path.join(root, 'web/connectome_data.json'), 'utf8'));
const elements = {};
const document = {
  getElementById: id => elements[id] || null,
  querySelectorAll: () => [],
  addEventListener: () => {},
  createElement: () => ({})
};
const window = {
  CONNECTOME_DATA: connectome,
  addEventListener: () => {},
  __COLLECT_FLAPPY_ROWS: true,
  __FLAPPY_TRAINING_ROWS: []
};
const context = {
  console, Math, Date, Float32Array, Uint8Array, Set, Map, Array, Object, Number,
  String, JSON, window, document, requestAnimationFrame: () => {},
  setTimeout: () => 0, clearTimeout: () => {}, performance: { now: () => 0 }
};
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(root, 'web/app.js'), 'utf8'), context);
vm.runInContext('connectomeData = window.CONNECTOME_DATA; setupBiophysicalMatrices(); controllerMode = "assisted"; currentGameMode = "flappy";', context);
const seeds = [20261027, 1, 42, 100, 20261028, 123456, 17, 29, 73, 314, 2718, 8128, 65537, 2026, 9001, 121, 222, 333, 444, 555, 777, 888, 999, 1234, 9876, 24680, 13579, 31415, 54321, 77777];
for (const seed of seeds) {
  vm.runInContext(`ablatedSet.clear(); experimentSeed = ${seed}; controllerMode = "assisted"; resetFlappy();`, context);
  for (let frame = 0; frame < 520; frame++) {
    vm.runInContext('updateFlappyGame();', context);
    if (vm.runInContext('flappy.is_game_over', context)) break;
  }
}
const rows = window.__FLAPPY_TRAINING_ROWS;
const out = path.join(root, 'benchmark', 'flappy_readout_training.jsonl');
fs.writeFileSync(out, rows.map(row => JSON.stringify(row)).join('\n') + '\n');
const positives = rows.filter(row => row.target === 1).length;
console.log(JSON.stringify({ rows: rows.length, positives, negatives: rows.length - positives, seeds: seeds.length, output: out }, null, 2));
