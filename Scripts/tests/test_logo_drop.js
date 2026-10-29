// The logo's ball drops only when the user changes page (D48): synthetic DOM, no network.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('Scripts/web_app/static/logo.js', 'utf8');

function load({ reducedMotion = false, logo = undefined, referrer = '', navigationType = 'navigate' } = {}) {
  const calls = [];
  const part = name => ({
    name,
    animate(keyframes, options) { calls.push({ name, keyframes, options }); return { cancel() {} }; },
    getAnimations() { return []; },
  });
  const parts = { '.spx-drop': part('ball'), '.spx-spin': part('spin'), '.spx-stem': part('stem') };
  const svg = { animate() {}, querySelector: selector => parts[selector] };
  const context = vm.createContext({
    window: { matchMedia: () => ({ matches: reducedMotion }) },
    document: { referrer, documentElement: { dataset: logo ? { logo } : {} }, querySelector: s => (s === '.logo-word' ? svg : null) },
    location: { origin: 'https://spix.example' },
    performance: { getEntriesByType: () => [{ type: navigationType }] },
    URL,
  });
  vm.runInContext(source, context);
  return { spixLogo: context.window.spixLogo, calls };
}

// A full page load drops the ball only when it came from another Spix page, and not on a reload.
const { spixLogo } = load();
assert.equal(spixLogo.dropsOnLoad('https://spix.example/static/methodology.html', 'https://spix.example', 'navigate'), true);
assert.equal(spixLogo.dropsOnLoad('https://spix.example/app', 'https://spix.example', 'back_forward'), true);
assert.equal(spixLogo.dropsOnLoad('https://spix.example/app', 'https://spix.example', 'reload'), false);
assert.equal(spixLogo.dropsOnLoad('', 'https://spix.example', 'navigate'), false, 'A first visit is not a page change');
assert.equal(spixLogo.dropsOnLoad('https://www.google.com/', 'https://spix.example', 'navigate'), false);

// Arriving from another Spix page plays the drop once on load; a first visit plays nothing.
assert.equal(load({ referrer: 'https://spix.example/' }).calls.length, 3);
assert.equal(load({ referrer: '' }).calls.length, 0);
assert.equal(load({ referrer: 'https://spix.example/', navigationType: 'reload' }).calls.length, 0);

// A drop animates the ball, its spin and the i together, once, ending where the logo rests.
const page = load();
assert.equal(page.spixLogo.drop(), true);
assert.deepEqual(page.calls.map(call => call.name), ['ball', 'spin', 'stem']);
for (const call of page.calls) {
  assert.ok(Number.isFinite(call.options.duration) && call.options.duration > 0);
  assert.ok(!call.options.iterations || call.options.iterations === 1, 'The drop never loops');
  assert.match(call.keyframes.at(-1).transform, /^(translateY\(0\) scale\(1, 1\)|rotate\(0deg\)|scale\(1, 1\))$/);
}

// Reduced motion and the old logo (?logo=old) never animate.
assert.equal(load({ reducedMotion: true }).spixLogo.drop(), false);
assert.equal(load({ logo: 'old' }).spixLogo.drop(), false);
console.log('Logo drop: page-change, first-visit, reload, reduced-motion and old-logo checks passed.');
