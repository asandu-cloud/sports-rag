// The six loading screens (D112, owner 16 Nov): one kit in static/loaders.js and loaders.css. A loader shows only
// after 300 ms, names what is loading in one fixed line for screen readers, and steps through its words in CSS.
// The first-load cover (A) also holds 600 ms once shown and always clears, even when the load fails.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const STATIC = path.join(__dirname, '../web_app/static');
const source = fs.readFileSync(path.join(STATIC, 'loaders.js'), 'utf8');
const css = fs.readFileSync(path.join(STATIC, 'loaders.css'), 'utf8');
const page = fs.readFileSync(path.join(STATIC, 'index.html'), 'utf8');
const kit = Function(source + 'return { SPIX_LOADER_KINDS, spixLoader, spixCover };')();
const text = html => html.replace(/<[^>]+>/g, ' ').replace(/&#39;/g, "'").replace(/&amp;/g, '&').replace(/\s+/g, ' ').trim();

(async () => {
  assert.deepEqual(kit.SPIX_LOADER_KINDS, ['ball', 'pitch', 'last10', 'prices', 'card', 'roll']);

  // Every kind: one status line for screen readers, the art and the steps hidden from them, the delay class.
  for (const kind of kit.SPIX_LOADER_KINDS) {
    const html = kit.spixLoader(kind, {label: 'Loading the match', steps: ['Loading the match', "Getting both teams' last 10", 'Fetching the head to head']});
    assert.match(html, new RegExp(`^<div class="ld ld-${kind}" role="status">`), kind);
    assert.match(html, /<span class="ld-sr">Loading the match<\/span>/, kind);
    assert.match(html, /<div class="ld-art" aria-hidden="true">/, kind);
    assert.match(html, /<p class="ld-steps" aria-hidden="true" style="--n:3"><span>Loading the match…<\/span><span>Getting both teams&#39; last 10…<\/span><span>Fetching the head to head…<\/span><\/p><\/div>$/, kind);
    assert.match(css, new RegExp(`\\.ld-${kind}[ .{]`), `${kind} has styles`);
  }

  // Steps: one or three. One stays still; any other count keeps only the first, so the CSS cycle always fits.
  assert.match(kit.spixLoader('roll', {label: 'Loading', steps: ['Loading the profile']}), /style="--n:1"><span>Loading the profile…<\/span><\/p>/);
  assert.match(kit.spixLoader('roll', {label: 'Loading', steps: ['One', 'Two']}), /style="--n:1"><span>One…<\/span><\/p>/);
  assert.match(kit.spixLoader('roll', {label: 'Loading <b>'}), /<span class="ld-sr">Loading &lt;b&gt;<\/span>.*style="--n:1"><span>Loading &lt;b&gt;…<\/span>/, 'no steps: the label, escaped');
  assert.throws(() => kit.spixLoader('spinner', {label: 'x'}), /Unknown loader/);

  // The art of each kind.
  const ball = kit.spixLoader('ball', {label: 'x', logo: '<svg class="logo-word"><g class="spx-drop"></g></svg>'});
  assert.match(ball, /<div class="ld-art" aria-hidden="true"><svg class="logo-word"><g class="spx-drop"><\/g><\/svg><\/div>/, 'A uses the header logo it is given');
  assert.equal((kit.spixLoader('pitch', {label: 'x'}).match(/class="ld-line"/g) || []).length, 9, 'B: outline, halfway, circle, 2 boxes, 2 six-yard boxes, 2 arcs');
  const last10 = kit.spixLoader('last10', {label: 'x'});
  assert.equal((last10.match(/<i class="ld-bar/g) || []).length, 10, 'C: ten columns');
  assert.equal((last10.match(/<i class="ld-bar over"/g) || []).length, 5, 'C: five over the line');
  const prices = kit.spixLoader('prices', {label: 'x'});
  assert.equal(text(prices), 'x Home . Draw . Away . x…', 'D: three cells, digits drawn by the CSS');
  assert.match(prices, /<span class="ld-cell"><small>Home<\/small><span class="ld-num"><i style="--d:2"><\/i>\.<i style="--d:1"><\/i><i style="--d:0"><\/i><\/span><\/span>/, 'D: each digit rolls to its value');
  assert.doesNotMatch(prices, /Bet365|Pinnacle|Chelsea/, 'D names no bookmaker and no team');
  assert.equal((kit.spixLoader('card', {label: 'x'}).match(/class="ld-fixture"/g) || []).length, 1);
  assert.equal((kit.spixLoader('card', {label: 'x', cards: 3}).match(/class="ld-fixture"/g) || []).length, 3, 'E: as many cards as asked');
  assert.match(kit.spixLoader('roll', {label: 'x', title: 'Chelsea v Leeds'}), /<p class="ld-title">Chelsea v Leeds<\/p>/);
  assert.doesNotMatch(kit.spixLoader('roll', {label: 'x'}), /ld-title/);
  // F (D113, owner 17 Nov): a tube the ball rolls through, cobalt filling in behind it, and the ball is the logo's own
  // football: its patches turn inside the clip while its shade and highlight stay put, as in the header.
  const roll = kit.spixLoader('roll', {label: 'x'});
  assert.match(roll, /<span class="ld-tube"><span class="ld-fill"><\/span><span class="ld-rollball"><svg viewBox="-17\.2 -17\.2 34\.4 34\.4">/);
  assert.match(roll, /<clipPath id="ld-ball-clip"><circle r="16\.36"\/><\/clipPath><circle r="16\.36" fill="#FAFDFA"\/><g clip-path="url\(#ld-ball-clip\)"><g class="ld-spin" stroke="#111E17"/);
  assert.match(roll, /<ellipse cx="-6\.87" cy="-8\.18"[^>]*fill="#FFFFFF" fill-opacity="0\.85"\/><\/g><circle r="15\.95" fill="none" stroke="#111E17" stroke-width="0\.82"\/><\/svg><\/span><\/span>/, 'the highlight and outline sit outside the turning patches');
  const header = page.slice(page.indexOf('<svg class="logo-word'));
  const patches = s => s.slice(s.indexOf('<g stroke-width="0.49">'), s.indexOf('<ellipse cx="-6.87"'));
  assert.equal(patches(roll), patches(header), "the same patches as the header's ball");
  assert.doesNotMatch(roll, /spix-header-ball|spx-/, "no ids or classes shared with the header's logo");
  assert.match(css, /\.ld-tube\{[^}]*height:28px;[^}]*border-radius:14px;[^}]*background:var\(--surface2\);[^}]*overflow:hidden/);
  assert.match(css, /\.ld-fill\{[^}]*background:var\(--accent\);[^}]*animation:ld-fill 2\.6s/);
  assert.match(css, /@keyframes ld-fill\{0%\{width:14px;opacity:1\}86%\{width:calc\(100% - 14px\);opacity:1\}/, 'the fill ends at the middle of the ball');
  assert.match(css, /@keyframes ld-roll-on\{0%\{left:4px;opacity:0\}6%\{opacity:1\}86%\{left:calc\(100% - 24px\);opacity:1\}/);
  assert.match(css, /\.ld-rollball svg\{[^}]*stroke:none/, "research.css strokes every svg; the ball's unstroked parts stay clean");
  assert.match(css, /\.ld-rollball \.ld-spin\{transform-box:view-box;transform-origin:0 0;animation:ld-turn 2\.6s/);

  // The CSS: the 300 ms delay on every loader, a still frame for reduced motion, theme tokens only.
  assert.match(css, /\.ld\{[^}]*animation:ld-in \.2s ease \.3s both/);
  const still = css.match(/@media \(prefers-reduced-motion:reduce\)\{([\s\S]*?)\n\}/);
  assert.ok(still, 'one reduced-motion block');
  assert.match(still[1], /\.ld,\.ld \*,\.ld \*::before,\.ld \*::after\{animation:none!important/);
  for (const kind of ['pitch', 'last10', 'prices', 'roll']) assert.match(still[1], new RegExp(`\\.ld-${kind}`), `${kind} has a still frame`);
  assert.doesNotMatch(css.replace(/\/\*[\s\S]*?\*\//g, ''), /#[0-9a-fA-F]{3,6}\b/, 'colours come from the theme tokens');

  // The page loads the kit before app.js and the Research Area.
  const at = name => page.indexOf(name);
  assert.match(page, /<link rel="stylesheet" href="\/static\/loaders\.css\?v=loaders-20261117a">/);
  assert.match(page, /<script src="\/static\/loaders\.js\?v=loaders-20261117a"><\/script>/);
  assert.ok(at('/static/loaders.js') < at('/static/research/research.js') && at('/static/loaders.js') < at('/static/app.js'));

  // A: the app's first board load runs under the cover, with the header's own logo. Later loads (a new date or
  // league, the 90 s refresh) never call it: init() starts the first load only once.
  const app = fs.readFileSync(path.join(STATIC, 'app.js'), 'utf8');
  assert.match(app, /      this\._initialLoad = this\._loadMatchday\(\);\n      window\.spixCover\?\.\(this\._initialLoad, \{label: "Getting today's matches ready", logo: document\.querySelector\('\.logo-word'\)\?\.outerHTML \|\| ''\}\);\n    \}\n    return this\._initialLoad;/);
  assert.equal((app.match(/spixCover/g) || []).length, 1, 'only the first load');

  // E on the board and D on Best Bets, in their loading states only (a background refresh never draws them).
  // The old line stays as the fallback if the kit is missing.
  assert.match(app, /    if \(state\.loading\) \{\n      container\.innerHTML = window\.spixLoader\?\.\('card', \{label: 'Preparing the matchday board', cards: 3,\n        steps: \['Loading the fixture schedule', 'Loading the published Match Reads', 'Preparing the matchday board'\]\}\)\n        \?\? '<div class="empty-state"><p>Preparing the matchday board…<\/p>/);
  assert.match(app, /    if \(state\.loading\) \{\n      container\.innerHTML = window\.spixLoader\?\.\('prices', \{label: 'Ranking the clearest matchday opportunities'\}\)\n        \?\? '<div class="empty-state"><p>Ranking the clearest matchday opportunities…<\/p><\/div>';/);
  assert.match(app, /if \(!background\) \{[\s\S]{0,200}?this\._renderFixtures\(\{ loading: true \}\);\n      this\._renderBestBets\(\{ loading: true \}\);/, 'only a foreground load');
  assert.match(page, /<script src="\/static\/app\.js\?v=search-20261118a"><\/script>/);

  // B, C and F in the Research Area. Its shadow root adopts the page's own loaders.css (already loaded, so the
  // component's first loader is styled at once); without the kit, the old line.
  const research = fs.readFileSync(path.join(STATIC, 'research/research.js'), 'utf8');
  // Without the adopted sheet (an older browser, or loaders.css failed), a loader would draw unstyled: the old line.
  assert.match(research, /\nlet kitAdopted = false;\nconst loading = \(kind, label, steps\) => \(kitAdopted && window\.spixLoader\?\.\(kind, \{label, steps\}\)\) \|\| `<p class="profile-loading" role="status">\$\{label\}…<\/p>`;\n/);
  assert.match(research, /    if \(kit\) \{ root\.adoptedStyleSheets = \[kit\]; kitAdopted = true; \}\n/);
  assert.match(research, /\nfunction kitSheet\(\) \{\n  const sheet = \[\.\.\.document\.styleSheets\]\.find\(s => s\.href\?\.includes\('\/static\/loaders\.css'\)\);/);
  assert.match(research, /    const root = this\.attachShadow\(\{mode: 'open'\}\);\n    const kit = kitSheet\(\);\n    if \(kit\) \{ root\.adoptedStyleSheets = \[kit\]; kitAdopted = true; \}\n    root\.innerHTML = loading\('roll', 'Loading the Research Area'\);/);
  assert.match(research, /\$\('researchHome'\)\.innerHTML = homeTop\(\) \+ loading\('roll', 'Loading this weekend and the season leaders'\);/, 'F on the home page');
  assert.match(research, /\$\('profileBody'\)\.innerHTML = loading\('last10', 'Loading the profile'\);/, 'C on a profile');
  assert.match(research, /\$\('matchBody'\)\.innerHTML = loading\('pitch', 'Loading the match', \['Loading the match', "Getting both teams' last 10", 'Fetching the head to head'\]\);/, 'B on the match view');
  assert.doesNotMatch(research, /'<p class="profile-loading" role="status">Loading/, 'no plain loading line left');
  assert.match(page, /<script type="module" src="\/static\/research\/research\.js\?v=research-20261117i"><\/script>/);

  // The cover (A), on a fake clock and a fake page.
  function harness() {
    let t = 0, timers = [];
    const nodes = [];
    const doc = {
      body: {append: el => nodes.push(el)},
      createElement: () => {
        const el = {className: '', innerHTML: '', classes: new Set(), removed: false};
        el.classList = {add: c => el.classes.add(c)};
        el.remove = () => { el.removed = true; };
        return el;
      },
    };
    const clock = {
      now: () => t,
      setTimeout: (fn, ms) => { const timer = {at: t + ms, fn}; timers.push(timer); return timer; },
      clearTimeout: timer => { timers = timers.filter(x => x !== timer); },
    };
    const advance = async ms => {
      const end = t + ms;
      for (;;) {
        await null; await null;
        const next = timers.filter(x => x.at <= end).sort((a, b) => a.at - b.at)[0];
        if (!next) break;
        timers = timers.filter(x => x !== next); t = next.at; next.fn();
      }
      t = end; await null; await null;
    };
    return {doc, clock, nodes, advance};
  }
  const deferred = () => { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b; }); return {promise, resolve, reject}; };
  const options = h => ({doc: h.doc, clock: h.clock, logo: '<svg></svg>', label: "Getting today's matches ready", steps: ['Loading the fixture schedule']});

  // A quick load: nothing is ever drawn.
  let h = harness(), load = deferred();
  kit.spixCover(load.promise, options(h));
  await h.advance(250); load.resolve(); await h.advance(1000);
  assert.equal(h.nodes.length, 0, 'under 300 ms: no cover');

  // A slow load: drawn at 300 ms, held to 900 ms though the load ends at 400, faded, then removed.
  h = harness(); load = deferred();
  kit.spixCover(load.promise, options(h));
  await h.advance(300);
  assert.equal(h.nodes.length, 1, 'shown at 300 ms');
  const cover = h.nodes[0];
  assert.equal(cover.className, 'ld-cover');
  assert.match(cover.innerHTML, /^<div class="ld ld-ball" role="status"><span class="ld-sr">Getting today&#39;s matches ready<\/span>/);
  await h.advance(100); load.resolve(); await h.advance(450);
  assert.ok(!cover.classes.has('ld-out'), 'held: 600 ms not yet up at 850 ms');
  await h.advance(60);
  assert.ok(cover.classes.has('ld-out'), 'fading at 900 ms');
  assert.ok(!cover.removed);
  await h.advance(300);
  assert.ok(cover.removed, 'gone after the fade');

  // A failed load clears it too.
  h = harness(); load = deferred();
  kit.spixCover(load.promise, options(h));
  await h.advance(400); load.reject(new Error('down')); await h.advance(1500);
  assert.ok(h.nodes[0].removed, 'a failure clears the cover');

  // A load that never ends: the cover gives up at 15 s and the board's own loader carries on.
  h = harness(); load = deferred();
  kit.spixCover(load.promise, options(h));
  await h.advance(15000);
  assert.ok(h.nodes[0].classes.has('ld-out'), 'gives up at 15 s');

  console.log('test_loaders_web: ok');
})().catch(error => { console.error(error); process.exit(1); });
