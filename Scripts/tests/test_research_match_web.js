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

  // Task 7, Cards. Atletico v Barcelona at 3.5: the read's chance and typical price for each side, no Add (cards are
  // not a slip market), then how often each team's, the referee's and the meetings' matches went over.
  assert.equal(m.SECTIONS.cards, m.cardsSection);
  const cards = m.cardsSection(atleti, {line: 3.5});
  assert.match(text(cards), /^Cards Spix expects 4 cards Line 2\.5 3\.5 4\.5 5\.5/);
  assert.equal((cards.match(/data-mline="/g) || []).length, 4);
  assert.match(cards, /data-mline="3\.5" aria-pressed="true"/);
  assert.match(text(cards), /Over 3\.5 cards Spix leans this way Spix 71% · typical price 1\.35/);
  assert.match(text(cards), /Under 3\.5 cards Spix 29% · typical price 3\.07/);
  assert.doesNotMatch(cards, /data-add/, 'no Add on cards');
  assert.equal((cards.match(/class="strip"/g) || []).length, 4);
  assert.match(text(cards), /How often their matches went over 3\.5 cards/);
  assert.match(text(cards), /Atletico Madrid's matches Last 10, every competition 4 of 10 3\.7 a match/);
  assert.match(text(cards), /Barcelona's matches Last 10, every competition 6 of 10 3\.8 a match/);
  assert.match(text(cards), /Javier Alberola Rojas Referee, last 10 matches 7 of 9 4\.3 a match/, 'a match without statistics is left out');
  assert.match(text(cards), /Atletico Madrid v Barcelona Meetings since 2023\/24 3 of 6 4\.5 a match/);
  // Who gets the cards: each side's own and their opponents', one decimal, with the biggest named.
  assert.match(text(cards), /Who gets the cards Last 10 each Atletico Madrid Booked a match 1\.9 Their opponents booked 1\.8 Barcelona Booked a match 1\.5 Their opponents booked 2\.3/);
  assert.match(text(cards), /Teams that play Barcelona pick up the most: 2\.3 cards a match\./);
  // The referee against the competition this season.
  assert.match(text(cards), /The referee Javier Alberola Rojas Cards a match this season 4\.9 7 matches, every competition La Liga this season 4\.7 118 matches/);
  assert.match(cards, /data-referee="j-rojas-faae24c2"/);
  assert.match(text(cards), /7 matches is a small sample/);
  assert.match(text(cards), /Each yellow and each red counts as one card here\. Check how your bookmaker counts a red\./);
  // Players booked most often, up to five a side.
  assert.match(text(cards), /Players booked most often This season, every competition/);
  assert.match(text(cards), /Marc Pubill 4 yellow in 13 matches/);
  assert.match(text(cards), /Robin Le Normand 2 yellow · 1 red in 11 matches/);
  assert.equal((cards.match(/class="mk-booked-row"/g) || []).length, 10);
  // Another line recounts the history; a line the read has no price for says so.
  assert.match(text(m.cardsSection(atleti, {line: 2.5})), /No typical price for 2\.5 cards yet\. Spix's chances cover 3\.5, 4\.5 and 5\.5\./);
  assert.match(text(m.cardsSection(atleti, {line: 2.5})), /Atletico Madrid's matches Last 10, every competition 8 of 10/);
  assert.match(text(m.cardsSection(atleti, {line: 4.5})), /Over 4\.5 cards Spix leans this way Spix 52%/);
  assert.match(m.cardsSection(atleti, {}), /data-mline="3\.5" aria-pressed="true"/, '3.5 by default');

  // Chelsea v Leeds today: no read, no referee. Three rows, and the referee line.
  const waitingCards = m.cardsSection(leeds, {line: 3.5});
  assert.doesNotMatch(text(waitingCards), /Spix expects/);
  assert.match(text(waitingCards), /Spix's chances and typical prices show once the read is out\./);
  assert.equal((waitingCards.match(/class="strip"/g) || []).length, 3);
  assert.match(text(waitingCards), /Chelsea's matches Last 10, every competition 9 of 10 5\.5 a match/);
  assert.match(text(waitingCards), /Leeds' matches Last 10, every competition 7 of 10 4\.2 a match/);
  assert.match(text(waitingCards), /Chelsea v Leeds Meetings since 2023\/24 1 of 2 5 a match/);
  assert.match(text(waitingCards), /Referee not named yet\. Appointments usually come two to three days before kickoff\./);
  assert.match(text(waitingCards), /Teams that play Chelsea pick up the most: 3\.4 cards a match\./);
  const strangers = structuredClone(leeds);
  strangers.h2h = [];
  assert.match(text(m.cardsSection(strangers, {line: 3.5})), /No meetings since 2023\/24\./);
  assert.match(m.renderMatch(atleti, {market: 'cards', back: {label: 'Barcelona'}}), /id="mk-cards"/);
  assert.match(m.marketsRow('cards'), /data-market="cards" aria-current="true"/);

  // Add to slip: the read's own option, handed to app.js as the match page hands its Explore options.
  const add = m.addLegDetail(atleti, 'goals', 'over', 2.5);
  assert.deepEqual(add.fixture, {event_id: '1570443', home: 'Atletico Madrid', away: 'Barcelona',
    kickoff: atleti.fixture.kickoff, league: 'LaLiga'});
  assert.equal(add.leg.label, 'Over 2.5 goals');
  assert.equal(add.leg.typical_price, 1.34);
  assert.equal(add.market, 'goals');
  assert.equal(m.addLegDetail(atleti, 'moneyline', 'away', null).leg.label, 'Barcelona to win');
  assert.equal(m.addLegDetail(atleti, 'cards', 'over', 3.5), null, 'cards are not a slip market');
  assert.equal(m.addLegDetail(leeds, 'goals', 'over', 2.5), null, 'no read, nothing to add');
  const app = fs.readFileSync(path.join(__dirname, '../web_app/static/app.js'), 'utf8');
  assert.match(app, /document\.addEventListener\('research-add-leg', event => \{[\s\S]*?slip\.add\(`\$\{fixture\.home\} vs \$\{fixture\.away\}`, leg\.label, leg\.typical_price \?\? null, recsMarketName\(leg\.market\), \{\s*fixtureId: String\(fixture\.event_id\),/);

  console.log('test_research_match_web: ok');
})().catch(error => { console.error(error); process.exit(1); });
