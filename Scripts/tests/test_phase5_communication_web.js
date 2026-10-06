// Render the actual selection panel with a minimal DOM; no browser or network.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../web_app/static/app.js'), 'utf8');
const start = source.indexOf('  _renderMatchReadSummary(f) {');
const end = source.indexOf('\n  _buildMarketTabs(f)', start);
assert.ok(start >= 0 && end > start);
let click, adds = 0, now = Date.parse('2023-10-01T12:00:00Z');
const button = {dataset:{matchReadAdd:'0'}, classList:{add(){},remove(){}}, addEventListener(type, fn){click=fn;}};
const container = {innerHTML:'',querySelectorAll(){return this.innerHTML.includes('data-match-read-add')?[button]:[];}};
const esc = value => String(value ?? '').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const obj = Function('document','slip','esc','Date','return ({'+source.slice(start,end)+'})')(
  {getElementById(){return container;}}, {legs:[],add(){adds++;return true;}},esc,{now:()=>now,parse:Date.parse});
const card={id:1,status:'recommended',home:'HomeFC',away:'AwayFC',selections:[{
  recommendation_id:'recommendation.v1:exact',market:{group:'goals'},role:'core',pick:'Over 2.5',odds:2.075,
  value_edge:-.01,confidence:'medium',justification:{},explanation:{
    reasoning:'Goals Over 2.5 @ 2.075 (Book1) <script>bad</script>',
    uncertainty:'Scenario estimate, not a guaranteed return.',price_conditions:'Minimum odds 1.79.',
    expires_at:'2023-10-01T12:04:00Z'}}]};
obj._renderMatchReadSummary(card);
assert.match(container.innerHTML,/data-recommendation-id="recommendation.v1:exact"/);
assert.match(container.innerHTML,/Over 2.5/);
assert.match(container.innerHTML,/>2.075<\/span>/);
assert.doesNotMatch(container.innerHTML,/>2.08<\/span>/);
assert.match(container.innerHTML,/Book1/);
assert.match(container.innerHTML,/Scenario estimate, not a guaranteed return/);
assert.match(container.innerHTML,/Minimum odds 1.79/);
assert.match(container.innerHTML,/-1.0% probability edge/);
assert.doesNotMatch(container.innerHTML,/\+-1/);
assert.match(container.innerHTML,/MEDIUM SUPPORT/);
assert.match(container.innerHTML,/&lt;script&gt;/);
assert.doesNotMatch(container.innerHTML,/<script>/);
click(); assert.equal(adds,1);
now=Date.parse('2023-10-01T12:05:00Z');click();assert.equal(adds,1);assert.equal(button.disabled,true);
obj._renderMatchReadSummary(card);
assert.match(container.innerHTML,/Quote expired/);assert.doesNotMatch(container.innerHTML,/data-match-read-add/);
console.log('Visible selection panel, exact price, escaping, identity and render/click expiry checks passed (synthetic only).');

card.selections[0].value_edge=null;card.selections[0].odds=null;obj._renderMatchReadSummary(card);
assert.doesNotMatch(container.innerHTML,/probability edge/);
assert.match(container.innerHTML,/>—<\/span>/);

// Exercise the actual headline and market entry points after time advances.
const guardStart=source.indexOf('  _matchReadQuoteUsable(f, pick, odds) {');
const fixtureEnd=source.indexOf('  _showMatch(fxId)',guardStart);
const marketStart=source.indexOf('  _toggleSlipFromMarket(fxId, pick, odds) {');
const marketEnd=source.indexOf('  _renderBestBets(',marketStart);
const slipCalls=[];
const entryPoints=Function('document','slip','showToast','Date','return ({'+source.slice(guardStart,fixtureEnd)+source.slice(marketStart,marketEnd)+'})')(
  {querySelectorAll(){return[];}},{add(...args){slipCalls.push(args);return true;}},()=>{}, {now:()=>now,parse:Date.parse});
const fixture={...card,isMatchRead:true,best:true,pick:'Over 2.5',odds:2.075};
fixture.selections[0].odds=2.075;
entryPoints.fixtures=[fixture];entryPoints.fixtureIndex={};
entryPoints._toggleSlipFromFixture(1);entryPoints._toggleSlipFromMarket(1,'Over 2.5',2.075);
assert.equal(slipCalls.length,0);
now=Date.parse('2023-10-01T12:00:00Z');
entryPoints._toggleSlipFromFixture(1);entryPoints._toggleSlipFromMarket(1,'Over 2.5',2.075);
assert.equal(slipCalls.length,2);assert.equal(slipCalls[0][2],2.075);
entryPoints._toggleSlipFromMarket(1,'Under 2.5',2.075);assert.equal(slipCalls.length,2);
assert.doesNotMatch(source,/f\.odds\.toFixed\(2\)|line\.odds\.toFixed\(2\)/);
console.log('Headline/market entry points preserve the exact selected quote and reject expired or unselected quotes.');
