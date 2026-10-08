const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { JSDOM } = require('jsdom');

function gallery() {
  const publicDir = path.join(__dirname, '../public');
  // JSDOM does not load image resources. Exercise the real UI using filename fixtures.
  const dom = new JSDOM(fs.readFileSync(path.join(publicDir, 'index.html'), 'utf8'),
    { url: 'http://localhost/', runScripts: 'outside-only' });
  const context = dom.getInternalVMContext();
  for (const script of ['anima38-ui.js', 'sample-identities.js', 'app.js']) {
    let source = fs.readFileSync(path.join(publicDir, 'js', script), 'utf8');
    if (script === 'app.js') source = source.replace(/^init\(\);$/m, '// API startup is disabled for this isolated DOM test.');
    vm.runInContext(source, context);
  }
  vm.runInContext('currentJob = "synthetic_job";', context);
  dom.window.localStorage.setItem('samples_limit', 'all');
  return { dom, context };
}

function sample(step, prompt) {
  const name = `style-step${String(step).padStart(8, '0')}__prompt_${String(prompt).padStart(2, '0')}_seed_42.png`;
  return { name, dir: `output/sample/step_${step}`, path: `/samples/${name}`, mtime: step };
}

test('real Samples DOM renders one group per prompt across checkpoints', () => {
  const { dom, context } = gallery();
  context.fixtureImages = [6000, 8000, 250].flatMap(step =>
    Array.from({ length: 13 }, (_, i) => sample(step, i + 1)));
  vm.runInContext('renderSampleGroups(fixtureImages);', context);
  const groups = [...dom.window.document.querySelectorAll('.sample-group')];
  assert.equal(groups.length, 13);
  groups.forEach((group, index) => {
    assert.equal(group.querySelector('.group-header').textContent, `Prompt ${index + 1}`);
    assert.equal(group.querySelectorAll('.sample-card').length, 3);
    assert.equal(group.querySelector('.sample-step').textContent, 'Step 8000');
  });
  assert.equal(dom.window.document.querySelectorAll('.sample-card').length, 39);
  dom.window.close();
});

test('old default placements recover without clearing deliberate prompt moves', () => {
  const { dom, context } = gallery();
  const first = sample(6000, 1);
  const second = sample(6000, 2);
  dom.window.localStorage.setItem('sample_order_synthetic_job', JSON.stringify([
    { path: first.path, group: 'default' }, { path: second.path, group: '2' }
  ]));
  context.fixtureImages = [first, second];
  vm.runInContext('renderSampleGroups(fixtureImages);', context);
  assert.equal(dom.window.document.querySelector('[data-group="0"] .sample-card').dataset.path, first.path);
  assert.equal(dom.window.document.querySelector('[data-group="2"] .sample-card').dataset.path, second.path);
  assert.equal(dom.window.document.querySelector('[data-group="default"]'), null);
  dom.window.close();
});
