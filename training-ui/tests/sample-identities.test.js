const { test } = require('node:test');
const assert = require('node:assert/strict');
const samples = require('../public/js/sample-identities');

test('3.8B checkpoint names preserve prompt numbering and checkpoint identity', () => {
  const first = samples.parse('style-step00006000__prompt_01_seed_42.png');
  assert.equal(first.groupKey, '0');
  assert.equal(first.promptIndex, 0);
  assert.equal(first.step, 6000);
  assert.equal(first.checkpoint, 'style-step00006000.safetensors');
  assert.equal(samples.parse('style-step00006000__prompt_13_seed_42.png').groupKey, '12');
});
test('earlier cloud samples can be grouped without renaming any files', () => {
  const image = { name: 'prompt_01_seed_42.png', dir: 'output/sample/step_000250' };
  // Earlier filenames have no checkpoint prefix. Handle them as native names.
  assert.equal(samples.parse(image).groupKey, '0');
  assert.equal(samples.parse(image).step, 250);
});
test('original Standalone filenames retain their zero-based prompt groups', () => {
  assert.equal(samples.parse('style_006000_00_20261007123456_42.png').groupKey, '0');
  assert.equal(samples.parse('style_006000_12_20261007123456_42_gpu0.png').groupKey, '12');
  assert.equal(samples.parse('style_e000004_00_20261007123456.png').epoch, 4);
  assert.equal(samples.parse('generation_00_20261007123456_42.png').groupKey, '0');
});
test('preview images do not claim to have a saved checkpoint', () => {
  const image = samples.parse('style__preview_step_000001__prompt_01_seed_42.png');
  assert.equal(image.step, 1);
  assert.equal(image.preview, true);
  assert.equal(image.checkpoint, null);
});
test('training steps order comparisons even if file mtimes were changed during transfer', () => {
  const images = [
    { name: 'style-step00000250__prompt_01_seed_42.png', mtime: 100 },
    { name: 'style-step00006000__prompt_01_seed_42.png', mtime: 50 }
  ].sort(samples.compareNewest);
  assert.equal(samples.parse(images[0]).step, 6000);
});
test('a 13-prompt run yields 13 independent galleries with every checkpoint retained', () => {
  const groups = new Map();
  for (const step of [250, 500, 6000, 8000]) for (let prompt = 1; prompt <= 13; prompt++) {
    const name = `style-step${String(step).padStart(8, '0')}__prompt_${String(prompt).padStart(2, '0')}_seed_42.png`;
    const key = samples.parse(name).groupKey;
    groups.set(key, [...(groups.get(key) || []), name]);
  }
  assert.equal(groups.size, 13);
  for (const group of groups.values()) assert.equal(group.length, 4);
});
test('unrecognized images stay in Uncategorized', () => {
  assert.equal(samples.parse('unrelated.png').groupKey, 'default');
});
