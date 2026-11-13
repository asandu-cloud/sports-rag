// The home page's first section (D98, D100): three packs from one saved Match Read (Brighton v Brentford, 8 Nov),
// the corners pack running through its real lines by itself. The words on the packs and in the run come from the
// site's own chance-words rule, and the run ends on Spix's pick. No browser or network.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const STATIC = path.join(__dirname, '../web_app/static');
const app = fs.readFileSync(path.join(STATIC, 'app.js'), 'utf8');
const helpersStart = app.indexOf('// ----- Pure render helpers');
const helpersEnd = app.indexOf('// ----- End of pure render helpers -----');
assert.ok(helpersStart >= 0 && helpersEnd > helpersStart);
const { chanceWords } = Function('esc', app.slice(helpersStart, helpersEnd) + 'return { chanceWords };')(String);
// The hero's script, with no page: only its data, its pure step builder and its loop.
const hero = Function('document', fs.readFileSync(path.join(STATIC, 'home-hero.js'), 'utf8')
  + 'return { HOME_HERO_LINES, HOME_HERO_PICK, homeHeroSteps, homeHeroLoop };')(undefined);
const page = fs.readFileSync(path.join(STATIC, 'index.html'), 'utf8');

// The run: every line from 6.5 to 11.5 with its chance in words and its typical price, then back to the pick.
const steps = hero.homeHeroSteps(chanceWords);
assert.deepEqual(steps.map(s => s.pick), ['Over 6.5', 'Over 7.5', 'Over 8.5', 'Over 9.5', 'Over 10.5', 'Over 11.5', 'Over 8.5']);
assert.deepEqual(steps.map(s => s.chance),
  ['about 11 in 12', 'about 4 in 5', 'about 2 in 3', 'about 3 in 5', 'about 2 in 5', 'about 1 in 3', 'about 2 in 3']);
assert.deepEqual(steps.map(s => s.price), ['1.07', '1.19', '1.36', '1.65', '2.07', '2.67', '1.36']);
// Only Spix's pick is marked, and the run ends on it.
assert.deepEqual(steps.map(s => s.marked), [false, false, true, false, false, false, true]);
assert.equal(hero.HOME_HERO_PICK, 8.5);
// The lines get harder to clear and pay more as they rise.
for (let i = 1; i < hero.HOME_HERO_LINES.length; i++) {
  assert.ok(hero.HOME_HERO_LINES[i][1] < hero.HOME_HERO_LINES[i - 1][1]);
  assert.ok(hero.HOME_HERO_LINES[i][2] > hero.HOME_HERO_LINES[i - 1][2]);
}

// The packs as printed (also what a visitor without script sees): each chance in words is the site's rule applied
// to the chance written beside it, and the corners pack agrees with the run's pick.
const section = page.slice(page.indexOf('<section class="home-hero"'), page.indexOf('</section>', page.indexOf('<section class="home-hero"')));
assert.ok(section.length > 0, 'the first section is the new hero');
const printed = [...section.matchAll(/<b[^>]*data-chance="([\d.]+)"[^>]*>([^<]+)<\/b>/g)].map(m => [Number(m[1]), m[2]]);
assert.equal(printed.length, 3, 'one chance on each pack');
for (const [chance, words] of printed) assert.equal(words, chanceWords(chance));
assert.deepEqual(printed.map(p => p[1]), ['about 3 in 5', 'about 2 in 3', 'about 1 in 2']);
const pick = steps[steps.length - 1];
assert.match(section, new RegExp(`id="hh-cornersPick"><mark>${pick.pick}</mark>`));
assert.match(section, new RegExp(`id="hh-cornersPrice">${pick.price.replace('.', '\\.')}<`));
// Each pack says it is an example and when its prices were taken; 18+ and the help line sit beside them.
assert.match(section, /Example: Spix&rsquo;s read of Brighton v Brentford, Sun 8 Nov, prices as of 13:56/);
assert.match(section, /18\+/);
assert.match(section, /href="https:\/\/www\.gamcare\.org\.uk\/get-support\/"/);
// The old hero is gone from the page, and so is the play wheel (D100): the corners pack runs by itself.
assert.doesNotMatch(page, /hero-run|landing-intro/);
assert.doesNotMatch(section, /hh-badge|hh-play|hh-said|See how Spix reads a line/);
// Each pack is read as one picture with its pick, so a screen reader never hears a line mid-run.
assert.equal((section.match(/<article class="hh-pack"[^>]*role="img"[^>]*aria-label="[^"]+"/g) || []).length, 3);

// The loop (D100): it starts on Spix's pick, runs every line, comes back to the pick, holds it, and runs again.
const timers = { queue: [], set(fn, ms) { this.queue.push({ fn, ms }); return this.queue.length; }, clear() { this.queue = []; } };
const next = () => { const t = timers.queue.shift(); t.fn(); return t.ms; };
const el = () => ({ innerHTML: '', textContent: '', dataset: {} });
const view = { pack: el(), pick: el(), chance: el(), price: el() };
const loop = hero.homeHeroLoop(view, chanceWords, timers);
loop.start(2500);
assert.equal(timers.queue.length, 1);
assert.equal(timers.queue[0].ms, 2500, 'the pick is seen before the first run');
const seen = [];
const delays = [];
for (let i = 0; i < steps.length * 2; i++) {
  next();
  seen.push([view.pick.innerHTML, view.chance.textContent, view.price.textContent, 'running' in view.pack.dataset]);
  delays.push(timers.queue[0].ms);
}
const run = steps.map((s, i) => [s.marked ? `<mark>${s.pick}</mark>` : s.pick, s.chance, s.price, i < steps.length - 1]);
assert.deepEqual(seen, [...run, ...run], 'two runs in a row, each ending on the pick, the card straight only mid-run');
assert.deepEqual(delays.slice(0, steps.length), [800, 800, 800, 800, 800, 1100, 4000], 'a beat on each line, a longer one before the pick, then a rest');
assert.equal(timers.queue.length, 1, 'one timer at a time');
// Off screen it stops on the pick, with nothing left running; it starts again from the top.
next(); next();
assert.equal(view.pick.innerHTML, 'Over 7.5');
loop.stop();
assert.equal(timers.queue.length, 0);
assert.deepEqual([view.pick.innerHTML, view.chance.textContent, view.price.textContent, 'running' in view.pack.dataset],
  ['<mark>Over 8.5</mark>', 'about 2 in 3', '1.36', false]);
loop.start(1000);
next();
assert.equal(view.pick.innerHTML, 'Over 6.5');
loop.start(1000);
assert.equal(timers.queue.length, 1, 'starting twice keeps one loop');
// With no timers given it uses the page's own. A browser throws "Illegal invocation" when setTimeout is called as a
// method of another object (Node does not), so these stand-ins throw the same way.
const plain = function (name) {
  return function (fn, ms) {
    if (this !== undefined && this !== globalThis) throw new TypeError(`Illegal invocation: ${name}`);
    return ms;
  };
};
const heroInBrowser = Function('document', 'setTimeout', 'clearTimeout',
  fs.readFileSync(path.join(STATIC, 'home-hero.js'), 'utf8') + 'return { homeHeroLoop };')(undefined, plain('setTimeout'), plain('clearTimeout'));
const pageLoop = heroInBrowser.homeHeroLoop({ pack: el(), pick: el(), chance: el(), price: el() }, chanceWords);
pageLoop.start(2500);
pageLoop.stop();

// The owner chose version A (D100): the smaller headline is the only layout, and the ?hero=b switch is gone.
assert.doesNotMatch(fs.readFileSync(path.join(STATIC, 'home-hero.js'), 'utf8'), /hero=b|homeHeroVariant|dataset\.hero/);
assert.doesNotMatch(fs.readFileSync(path.join(STATIC, 'home-hero.css'), 'utf8'), /data-hero|Version B/);

// It fills a laptop screen but stops after its content on a taller window, so the footnote stays close (D104).
assert.match(fs.readFileSync(path.join(STATIC, 'home-hero.css'), 'utf8'),
  /\.hh-inner \{[^}]*min-height: min\(calc\(100svh - var\(--topbar-h\)\), 820px\);/);
assert.match(page, /<link rel="stylesheet" href="\/static\/home-hero\.css\?v=home-20261113a">/);
assert.match(page, /<script src="\/static\/home-hero\.js\?v=home-20261112b"><\/script>/);

console.log('home hero web tests passed');
