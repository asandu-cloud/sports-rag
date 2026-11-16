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
  // Atletico v Barcelona's real read, saved after full time; the tests of a read before kickoff see it as not started.
  const played = fixture('research_match_atletico_barcelona.json');
  const atleti = {...played, fixture: {...played.fixture, status: 'NS'}};
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
  // Cards carry no Spix suggestion (D61), so the likelier side reads "More likely" (owner, 16 Nov).
  assert.match(text(cards), /Over 3\.5 cards More likely Spix 71% · typical price 1\.35/);
  assert.doesNotMatch(cards, /Spix leans this way/);
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
  assert.match(text(m.cardsSection(atleti, {line: 4.5})), /Over 4\.5 cards More likely Spix 52%/);
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

  // Task 8, Shots on target: the likely starters' last 10 against 1+ or 2+, both teams or one, most hits first; what
  // each side allows. B2 carries no player prices, so there is no price column at all.
  assert.equal(m.SECTIONS.sot, m.sotSection);
  const sot = m.sotSection(leeds, {line: 1});
  assert.match(text(sot), /^Shots on target Likely starters, by starts in their last 5 Each player 1\+ 2\+ Team Both Chelsea Leeds/);
  assert.match(sot, /data-mline="1" aria-pressed="true"/);
  assert.match(sot, /data-mteam="both" aria-pressed="true"/);
  const sotRows = sot.match(/<tr class="mk-sot-row">[\s\S]*?<\/tr>/g);
  assert.equal(sotRows.length, 16);
  assert.match(text(sotRows[0]), /^Pedro Neto Chelsea · Midfielder 8 of 10 76 min$/, 'most hits first');
  assert.match(text(sot), /Cole Palmer Chelsea · Midfielder 7 of 10 89 min/);
  assert.match(text(sot), /Dominic Calvert-Lewin Leeds · Forward 7 of 10 87 min/);
  assert.doesNotMatch(sot, /Typical price|–/, 'no price column without prices');
  const sot2 = m.sotSection(leeds, {line: 2});
  assert.match(text(sot2), /Pedro Neto Chelsea · Midfielder 4 of 10/);
  assert.match(text(sot2), /Cole Palmer Chelsea · Midfielder 2 of 10/);
  assert.match(sot2, /data-mline="2" aria-pressed="true"/);
  // What each side allows: shots on target faced a match this season and the league rank.
  assert.match(text(sot), /What Leeds allow Shots on target faced a match 4\.3 10th most of 20 in the Premier League/);
  assert.match(text(sot), /What Chelsea allow Shots on target faced a match 5\.2 4th most of 20 in the Premier League/);
  // One team: its players, and only what their opponents allow.
  const leedsOnly = m.sotSection(leeds, {line: 1, team: 'away'});
  assert.equal(leedsOnly.match(/<tr class="mk-sot-row">/g).length, 8);
  assert.doesNotMatch(text(leedsOnly), /Cole Palmer|What Leeds allow/);
  assert.match(text(leedsOnly), /What Chelsea allow/);
  assert.match(leedsOnly, /data-mteam="away" aria-pressed="true"/);
  assert.match(text(sot), /Line-ups are confirmed about an hour before kickoff; this list comes from recent starts\./);
  const noLeague = structuredClone(leeds);
  noLeague.teams.away.allowed = null;
  assert.match(text(m.sotSection(noLeague, {line: 1})), /What Leeds allow No league figures for Leeds this season yet\./);
  // Ranks in words, 1 = fewest.
  assert.equal(m.rankWords(1, 20), 'the fewest of 20');
  assert.equal(m.rankWords(20, 20), 'the most of 20');
  assert.equal(m.rankWords(3, 20), '3rd fewest of 20');
  assert.equal(m.rankWords(10, 20), '10th fewest of 20');
  assert.equal(m.rankWords(11, 20), '10th most of 20');
  assert.match(text(m.sotSection(atleti, {line: 1})), /What Barcelona allow Shots on target faced a match 3\.3 the fewest of 20 in La Liga/);
  // The player page's context panel: what the next opponent allows.
  assert.equal(m.opponentAllows(leeds, 12), 'Leeds allow 4.3 shots on target a match, the 10th most of 20 in the Premier League.');
  assert.equal(m.opponentAllows(leeds, 17), 'Chelsea allow 5.2 shots on target a match, the 4th most of 20 in the Premier League.');
  assert.equal(m.opponentAllows(atleti, 22), 'Barcelona allow 3.3 shots on target a match, the fewest of 20 in La Liga.');
  assert.equal(m.opponentAllows(noLeague, 12), '');
  assert.equal(m.opponentAllows(leeds, 999), '');
  const content = fs.readFileSync(path.join(__dirname, '../web_app/static/research/content.html'), 'utf8');
  assert.match(content, /<section id="opponentContext" class="opponent-context" hidden><h3>The next opponent<\/h3><p id="opponentInsight"><\/p><\/section>/);

  // Task 9. Goals at 2.5: the read's chance and price for each side, Add on both (the slip takes goals), then each
  // team's last 10 and the meetings.
  assert.equal(m.SECTIONS.goals, m.goalsSection);
  const goals = m.goalsSection(atleti, {line: 2.5});
  assert.match(text(goals), /^Goals Spix expects 3\.7 goals Line 1\.5 2\.5 3\.5/);
  assert.match(text(goals), /Over 2\.5 goals Spix leans this way Spix 72% · typical price 1\.34 Add/);
  assert.match(text(goals), /Under 2\.5 goals Spix 28% · typical price 3\.20 Add/);
  assert.match(goals, /data-add="goals\|over\|2\.5"/);
  assert.match(goals, /data-add="goals\|under\|2\.5"/);
  assert.equal((goals.match(/class="strip"/g) || []).length, 3);
  assert.match(text(goals), /Atletico Madrid's matches Last 10 · scored 1\.8, conceded 0\.5 4 of 10 2\.3 a match/);
  assert.match(text(goals), /Barcelona's matches Last 10 · scored 3\.8, conceded 1\.5 10 of 10 5\.3 a match/);
  assert.match(text(goals), /Atletico Madrid v Barcelona Meetings since 2023\/24 5 of 6 3\.5 a match/);
  assert.doesNotMatch(m.goalsSection(leeds, {line: 2.5}), /data-add/, 'no read, no Add');
  assert.match(text(m.goalsSection(leeds, {})), /Spix's chances and typical prices show once the read is out\./);
  assert.match(m.goalsSection(leeds, {}), /data-mline="2\.5" aria-pressed="true"/, '2.5 by default');

  // Corners at 9.5 likewise.
  assert.equal(m.SECTIONS.corners, m.cornersSection);
  const corners = m.cornersSection(atleti, {line: 9.5});
  assert.match(text(corners), /^Corners Spix expects 8\.9 corners Line 8\.5 9\.5 10\.5 11\.5/);
  assert.match(text(corners), /Over 9\.5 corners Spix leans this way Spix 58% · typical price 1\.62 Add/);
  assert.match(text(corners), /Under 9\.5 corners Spix 42% · typical price 2\.14 Add/);
  assert.match(text(corners), /Atletico Madrid's matches Last 10 · won 5\.2, conceded 4\.1 4 of 10 9\.3 a match/);
  assert.match(text(corners), /Barcelona's matches Last 10 · won 6\.3, conceded 2\.8 4 of 10 9\.1 a match/);
  assert.match(text(corners), /Atletico Madrid v Barcelona Meetings since 2023\/24 3 of 6 9\.8 a match/);

  // Result: the three chances with prices and Add, each side's last 10 as W, D and L, home and away form.
  assert.equal(m.SECTIONS.result, m.resultSection);
  const result = m.resultSection(atleti, {});
  assert.match(text(result), /Atletico Madrid to win Spix 27% · typical price 3\.58 Add Draw Spix 21% · typical price 4\.37 Add Barcelona to win Spix leans this way Spix 52% · typical price 1\.85 Add/);
  assert.equal((result.match(/data-add="moneyline\|/g) || []).length, 3);
  assert.match(result, /data-add="moneyline\|draw\|null"/);
  assert.match(text(result), /Atletico Madrid Last 10, every competition W W W W W W D W W L 8 won, 1 drawn, 1 lost/);
  assert.match(text(result), /Barcelona Last 10, every competition W W W W W W L L W W 8 won, 0 drawn, 2 lost/);
  assert.match(text(result), /Atletico Madrid at home Won, drew, lost 4 · 0 · 1 5 home matches in their last 10/);
  assert.match(text(result), /Barcelona away Won, drew, lost 4 · 0 · 1 5 away matches in their last 10/);
  assert.match(text(m.resultSection(leeds, {})), /Spix's chances and typical prices show once the read is out\./);
  assert.doesNotMatch(m.resultSection(leeds, {}), /data-add/);

  // Head to head: one row per meeting, newest first, with what happened in it.
  assert.equal(m.SECTIONS.h2h, m.h2hSection);
  const h2h = m.h2hSection(atleti, {});
  assert.match(text(h2h), /^Head to head Since 2023\/24 Barcelona won 4, Atletico Madrid won 2\./);
  const meetings = h2h.match(/<li class="mk-h2h-row">[\s\S]*?<\/li>/g);
  assert.equal(meetings.length, 6);
  assert.match(text(meetings[0]), /^14 Apr 2026 UEFA Champions League Atletico Madrid 1–2 Barcelona 2 cards · 6 corners · 3 goals$/);
  assert.match(text(m.h2hSection(leeds, {})), /Chelsea 2–2 Leeds 8 cards · 5 corners · 4 goals/);
  assert.match(text(m.h2hSection(strangers, {})), /No meetings since 2023\/24\./);

  // Team news: no line-ups yet, so the likely starters from recent starts, and when line-ups come.
  assert.equal(m.SECTIONS.news, m.newsSection);
  const news = m.newsSection(atleti, {});
  assert.match(text(news), /^Team news Likely outfield starters, from starts in the last 5/);
  assert.equal((news.match(/class="mk-news-row"/g) || []).length, 16);
  assert.match(text(news), /Dávid Hancko Defender · started 5 of last 5/);
  assert.match(text(m.newsSection(leeds, {})), /Josh Acheampong Defender · started 4 of last 5/);
  assert.match(text(news), /Line-ups are confirmed about an hour before kickoff\. Injuries and suspensions are not in Spix's data yet\./);
  assert.deepEqual(m.MARKETS.filter(key => m.SECTIONS[key]), m.MARKETS, 'every market has its section');

  // Final review: values from the API never break out of an attribute, and a missing chance is not "NaN%".
  const hostile = structuredClone(atleti);
  hostile.read.markets.find(x => x.market === 'moneyline').options[0].side = 'home"><b>x';
  assert.doesNotMatch(m.resultSection(hostile, {}), /"><b>x/);
  hostile.read.result.home = '1;background:red';
  assert.doesNotMatch(m.spixReads(hostile, 'result'), /background:red/);
  const gap = structuredClone(atleti);
  gap.read.lines.goals.find(l => l.line === 2.5).under.chance = null;
  assert.doesNotMatch(text(m.goalsSection(gap, {line: 2.5})), /NaN/);
  assert.match(text(m.goalsSection(gap, {line: 2.5})), /Under 2\.5 goals typical price 3\.20/);

  // A match that has kicked off (real-data pass, 16 Nov). The API drops the read once the board does: Atletico v
  // Barcelona as served after full time, status FT and no read. Nothing promises a read that will not come.
  const gone = {...played, read: null, fixture: {...played.fixture, read: {status: 'waiting', read_at: null, opens_at: null, expected: null}}};
  assert.match(text(m.matchHead(gone, {now})), /Referee Javier Alberola Rojas Full time$/);
  assert.equal(text(m.spixReads(gone, 'goals')), "How Spix reads it This match has been played, so Spix's read is no longer shown. The past numbers below run up to its kickoff.");
  const goneGoals = text(m.goalsSection(gone, {line: 2.5}));
  assert.match(goneGoals, /Spix's chances and typical prices are shown before kickoff only\./);
  assert.doesNotMatch(goneGoals, /once the read is out/);
  assert.match(text(m.resultSection(gone, {})), /shown before kickoff only/);
  const inPlay = {...gone, fixture: {...gone.fixture, status: '2H'}};
  assert.match(text(m.matchHead(inPlay, {now})), /In play$/);
  assert.match(text(m.spixReads(inPlay, 'goals')), /This match has kicked off, so Spix's read is no longer shown\. The past numbers below run up to kickoff\./);
  const postponed = {...gone, fixture: {...gone.fixture, status: 'PST'}};
  assert.match(text(m.matchHead(postponed, {now})), /Postponed$/);
  assert.match(text(m.spixReads(postponed, 'goals')), /This match was postponed\. Spix reads it again once it has a new date\./);
  assert.match(text(m.goalsSection(postponed, {line: 2.5})), /Spix's chances and typical prices show once the match has a new date\./);
  // Still carrying its read from before kickoff (the board's Live tab keeps it): said in the past tense, no Add.
  assert.match(text(m.matchHead(played, {now})), /Full time$/);
  assert.match(text(m.spixReads(played, 'goals')), /^How Spix read it before kickoff Barcelona are favourites\./);
  const playedGoals = m.goalsSection(played, {line: 2.5});
  assert.match(text(playedGoals), /Over 2\.5 goals Spix leans this way Spix 72%/);
  assert.doesNotMatch(playedGoals, /data-add/, 'no Add once the match has kicked off');
  assert.doesNotMatch(m.resultSection(played, {}), /data-add/);
  assert.equal(m.matchPhase({status: 'NS'}), null);
  assert.equal(m.matchPhase({}), null, 'no status: as the API gives an upcoming match');

  console.log('test_research_match_web: ok');
})().catch(error => { console.error(error); process.exit(1); });
