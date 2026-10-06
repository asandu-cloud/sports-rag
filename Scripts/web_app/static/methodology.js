// Educational single-market arithmetic only; never creates a recommendation.
const evForm = document.getElementById('evExample');
function calculateExample(event) {
  event?.preventDefault();
  if (!evForm.reportValidity()) return;
  const probability = Number(document.getElementById('evProbability').value) / 100;
  const odds = Number(document.getElementById('evOdds').value);
  const result = document.getElementById('evResult');
  if (!Number.isFinite(probability) || !Number.isFinite(odds) || probability < 0 || probability > 1 || odds <= 1) {
    result.textContent = 'Enter a probability from 0 to 100 and decimal odds greater than 1.';
    return;
  }
  const implied = 100 / odds, edge = probability * 100 - implied, expected = probability * odds - 1;
  const money = new Intl.NumberFormat('en-GB', { style: 'currency', currency: 'GBP', signDisplay: 'exceptZero' }).format(expected);
  result.innerHTML = `<dl class="info-ev-results"><div><dt>Price-implied probability</dt><dd>${implied.toFixed(1)}%</dd></div><div><dt>Probability difference</dt><dd>${edge > 0 ? '+' : ''}${edge.toFixed(1)} pp</dd></div><div><dt>Estimated profit per £1</dt><dd>${money}</dd></div></dl>`;
}
evForm.addEventListener('submit', calculateExample);
calculateExample();
