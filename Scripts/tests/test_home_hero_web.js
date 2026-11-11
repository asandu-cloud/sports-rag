// The home page's first section (D98): three packs from one saved Match Read (Brighton v Brentford, 8 Nov) and the
// play badge that runs the corners pack through its real lines. The words on the packs and in the run come from the
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
// The hero's script, with no page: only its data and its pure step builder.
const hero = Function('document', fs.readFileSync(path.join(STATIC, 'home-hero.js'), 'utf8')
  + 'return { HOME_HERO_LINES, HOME_HERO_PICK, homeHeroSteps };')(undefined);
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
// The old hero is gone from the page.
assert.doesNotMatch(page, /hero-run|landing-intro/);

console.log('home hero web tests passed');
