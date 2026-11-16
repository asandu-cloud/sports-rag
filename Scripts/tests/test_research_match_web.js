// Research Area overhaul task 6 (D107): the match view's shell. Real B2 bodies from the live database on 16 Nov:
// Chelsea v Leeds (no referee, no read yet) and Atletico Madrid v Barcelona (8 Nov, its read and referee).
process.env.TZ = 'Europe/London'; // kickoffs show in the viewer's local time; the test reads them in London
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');

const text = html => html.replace(/<[^>]+>/g, ' ').replace(/&#39;/g, "'").replace(/&amp;/g, '&').replace(/\s+/g, ' ')
  .replace(/ ([.,:%])/g, '$1').trim();
const fixture = name => JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures', name), 'utf8'));

(async () => {
  const m = await import(pathToFileURL(path.join(__dirname, '../web_app/static/research/match.js')));
  const leeds = fixture('research_match_chelsea_leeds.json');
  const atleti = fixture('research_match_atletico_barcelona.json');
  const now = new Date('2026-11-16T12:00:00Z');

  // The markets, in the spec's order.
  assert.deepEqual(m.MARKETS, ['cards', 'sot', 'goals', 'corners', 'result', 'h2h', 'news']);
  assert.equal(m.MARKET_LABELS.sot, 'Shots on target');

  // The head: both crests, the teams, competition, day and time, referee.
  const head = m.matchHead(leeds, {now});
  assert.match(text(head), /Chelsea v Leeds/);
  assert.match(text(head), /Sat 21 Nov · 15:00 Premier League Referee not named yet Read due Tue/);
  assert.equal((head.match(/class="crest/g) || []).length, 2);
  const named = m.matchHead(atleti, {now});
  assert.match(text(named), /Atletico Madrid v Barcelona/);
  assert.match(named, /data-referee="j-rojas-faae24c2"[^>]*>Javier Alberola Rojas</, 'the referee opens his page');
  assert.match(text(named), /Read ready/);

  // How Spix reads it: Atletico v Barcelona's real 8 Nov read, one figure marked for the open market.
  const reads = m.spixReads(atleti, 'cards', {now});
  assert.match(text(reads), /How Spix reads it/);
  assert.match(text(reads), /Barcelona are favourites\./);
  assert.match(text(reads), /Atletico Madrid 27% Draw 21% Barcelona 52%/);
  assert.match(text(reads), /3\.7 goals Atletico Madrid 1\.6 · Barcelona 2\.2/);
  assert.match(text(reads), /8\.9 corners/);
  assert.match(text(reads), /4 cards/);
  assert.match(text(reads), /70% both teams score/);
  assert.equal((reads.match(/<mark>/g) || []).length, 1);
  assert.match(reads, /<mark>4<\/mark>/, 'cards open: the cards figure is marked');
  assert.match(m.spixReads(atleti, 'goals', {now}), /<mark>3\.7<\/mark>/);
  assert.match(m.spixReads(atleti, 'corners', {now}), /<mark>8\.9<\/mark>/);
  assert.match(m.spixReads(atleti, 'result', {now}), /<mark>52%<\/mark>/, 'result open: the favourite marked');
  assert.doesNotMatch(m.spixReads(atleti, 'h2h', {now}), /<mark>/);
  // Close chances have no favourite.
  const even = structuredClone(atleti);
  even.read.result = {home: 0.36, draw: 0.3, away: 0.34};
  assert.match(text(m.spixReads(even, 'cards', {now})), /No clear favourite\./);

  // Read not out yet: when it comes out, no figures; the history below still shows.
  const waiting = m.spixReads(leeds, 'cards', {now});
  assert.match(text(waiting), /Spix's read comes out Tue 17 Nov\. The past numbers below are ready now\./);
  assert.doesNotMatch(waiting, /class="figs"/);
  // A preliminary read says so.
  const early = structuredClone(atleti);
  early.read.status = 'preliminary';
  assert.match(text(m.spixReads(early, 'cards', {now})), /Preliminary: an early read\. Spix refreshes it nearer kickoff\./);

  // The markets row: only filled markets get a button; the open one is current.
  const row = m.marketsRow('cards', ['cards', 'goals']);
  assert.equal((row.match(/data-market="/g) || []).length, 2);
  assert.match(row, /data-market="cards" aria-current="true"/);
  assert.match(row, /data-market="goals" aria-current="false"/);
  assert.equal(m.marketsRow('cards', []), '');

  // The whole view: back link, head, read, the foot line.
  const view = m.renderMatch(leeds, {market: 'cards', back: {label: 'Chelsea'}, now});
  assert.match(view, /data-match-back/);
  assert.match(text(view), /^Chelsea Chelsea v Leeds/);
  assert.match(text(view), /Past matches are records, not a forecast\. Spix's figures come from statistical models\. 18\+ · GamCare$/);
  assert.match(view, /href="https:\/\/www\.gamcare\.org\.uk\/get-support\/"/);

  // When it cannot load.
  assert.match(text(m.matchError(404)), /This match could not load It may have been moved or cancelled\./);
  assert.match(text(m.matchError(503)), /Research is updating Try again in a few minutes\./);
  assert.match(text(m.matchError(500)), /This match could not load Please try again in a moment\./);
  assert.match(m.matchError(404), /data-match-back/);

  console.log('test_research_match_web: ok');
})().catch(error => { console.error(error); process.exit(1); });
