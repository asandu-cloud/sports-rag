/* Research Area catalogue: labels and definitions for the statistics served by /api/research/*.
   Keys match the `values` of each profile's `match_log`. No records live here. */
const stat = (label, title, noun, definition, extra = {}) => ({label, title, noun, definition, ...extra});
const EXTRA_TIME = 'Extra time is included when a match went to extra time.';

const playerMetrics = {
  shots: stat('Shots', 'Total shots', 'shots', `All recorded attempts at goal: on target, off target and blocked. ${EXTRA_TIME}`, {verb: 'Took a shot', threshold: 2}),
  shots_on_target: stat('On target', 'Shots on target', 'shots on target', 'Attempts recorded as on target: goals plus efforts saved by the goalkeeper. These are match records, not a probability of doing it again.', {verb: 'Had a shot on target'}),
  goals: stat('Goals', 'Goals', 'goals', 'Goals credited to this player in each match. Own goals are not counted.', {verb: 'Scored'}),
  assists: stat('Assists', 'Assists', 'assists', 'Assists credited to this player in the match record. Bookmakers can use a different definition when they settle a bet.', {verb: 'Recorded an assist'}),
  yellow_cards: stat('Yellow cards', 'Yellow cards', 'yellow cards', 'Recorded yellow cards. Red cards are counted separately. This is not a booking-points total.', {verb: 'Was booked'}),
  passes: stat('Passes', 'Passes attempted', 'passes', 'Passes attempted, successful or not. Minutes played and a player’s role both change the count.', {verb: 'Attempted a pass', threshold: 30, step: 5, limit: 120}),
  tackles: stat('Tackles', 'Tackles', 'tackles', 'Tackles credited to the player in the match record.', {verb: 'Made a tackle'}),
  interceptions: stat('Interceptions', 'Interceptions', 'interceptions', 'Opponent passes intercepted. Team style and match state change how often the chance arises.', {verb: 'Made an interception'}),
  duels_won: stat('Duels won', 'Duels won', 'duels won', 'One-on-one contests for the ball that the player won, on the ground or in the air.', {verb: 'Won a duel', threshold: 3}),
  fouls_committed: stat('Fouls committed', 'Fouls committed', 'fouls', 'Fouls given against this player. A foul does not always lead to a card.', {verb: 'Committed a foul'}),
  fouls_drawn: stat('Fouls won', 'Fouls won', 'fouls won', 'Fouls committed against this player by an opponent.', {verb: 'Won a foul'}),
  red_cards: stat('Red cards', 'Red cards', 'red cards', 'Recorded red cards, including second yellows. Check how a bookmaker defines cards before using this for a bet.', {verb: 'Was sent off'}),
  minutes: stat('Minutes', 'Minutes played', 'minutes', 'Minutes on the pitch in each appearance, including stoppage time as recorded by the data provider.', {verb: 'Played', threshold: 60, step: 15, limit: 120, noRate: true})
};

const teamMetrics = {
  goals_for: stat('Goals', 'Goals scored', 'goals', `Goals scored by this team, taken from the final score. ${EXTRA_TIME} Penalty shoot-outs are not counted.`, {threshold: 2}),
  corners_for: stat('Corners', 'Corners won', 'corners', 'Corners won by this team. Team corners are not the same as total match corners.', {threshold: 5}),
  sot_for: stat('On target', 'Shots on target', 'shots on target', 'Attempts on target by this team, not the match total.', {threshold: 4}),
  cards: stat('Cards', 'Cards received', 'cards', 'Yellow and red cards shown to this team. This is not a booking-points total.', {threshold: 2}),
  xg_for: stat('xG', 'Expected goals (xG)', 'expected goals', 'Expected goals estimates the quality of the chances a team created. 1.0 means chances that would usually produce one goal.', {threshold: 1, decimals: 2}),
  goals_against: stat('Conceded', 'Goals conceded', 'goals conceded', `Goals scored by the opponent, taken from the final score. ${EXTRA_TIME}`, {threshold: 1}),
  xg_against: stat('xG conceded', 'Expected goals conceded', 'expected goals conceded', 'The quality of the chances the opponent created, measured in expected goals.', {threshold: 1, decimals: 2}),
  corners_against: stat('Corners conceded', 'Corners conceded', 'corners conceded', 'Corners won by the opponent in this team’s matches.', {threshold: 5}),
  shots_for: stat('Shots', 'Total shots', 'shots', 'All attempts by this team: on target, off target and blocked.', {threshold: 10, limit: 40}),
  shots_against: stat('Shots faced', 'Shots faced', 'shots faced', 'All attempts by the opponent.', {threshold: 10, limit: 40}),
  sot_against: stat('On target faced', 'Shots on target faced', 'shots on target faced', 'Attempts on target by the opponent.', {threshold: 4}),
  possession: stat('Possession', 'Possession', 'possession', 'The team’s share of the ball. This is a percentage, not a count or a chance of winning.', {percent: true, threshold: 50, step: 5, limit: 100}),
  fouls: stat('Fouls', 'Fouls committed', 'fouls', 'Fouls given against this team. A foul does not always lead to a card.', {threshold: 10, limit: 40}),
  opponent_cards: stat('Opponent cards', 'Cards shown to the opponent', 'opponent cards', 'Yellow and red cards shown to the other team in this team’s matches.', {threshold: 2}),
  red_cards: stat('Red cards', 'Red cards received', 'red cards', 'Red cards shown to this team, including second yellows.'),
  offsides: stat('Offsides', 'Offsides', 'offsides', 'Offside offences given against this team.')
};

const refereeMetrics = {
  cards: stat('Cards', 'Cards shown', 'cards', `Yellow and red cards shown to both teams. ${EXTRA_TIME} Not a booking-points total.`, {threshold: 4}),
  yellows: stat('Yellows', 'Yellow cards shown', 'yellow cards', 'Yellow cards shown to both teams.', {threshold: 4}),
  reds: stat('Reds', 'Red cards shown', 'red cards', 'Red cards shown to both teams, including second yellows.'),
  fouls: stat('Fouls', 'Fouls called', 'fouls', 'Fouls given against both teams. The teams’ style matters as much as the referee.', {threshold: 20, limit: 50}),
  home_cards: stat('Home cards', 'Cards shown to the home team', 'home-team cards', 'Yellow and red cards shown to the home team. Raw home and away counts do not show bias.', {threshold: 2}),
  home_fouls: stat('Home fouls', 'Fouls by the home team', 'home-team fouls', 'Fouls given against the home team.', {threshold: 10, limit: 40}),
  away_cards: stat('Away cards', 'Cards shown to the away team', 'away-team cards', 'Yellow and red cards shown to the away team. Team style, competition and match state all affect this.', {threshold: 2}),
  away_fouls: stat('Away fouls', 'Fouls by the away team', 'away-team fouls', 'Fouls given against the away team.', {threshold: 10, limit: 40})
};

export const catalogues = {
  players: {label: 'Players', singular: 'player', metrics: playerMetrics, defaultStat: 'shots_on_target',
    common: ['shots', 'shots_on_target', 'goals', 'assists', 'yellow_cards'],
    groups: [['Passing and duels', ['passes', 'duels_won']], ['Defending', ['tackles', 'interceptions']], ['Discipline', ['fouls_committed', 'fouls_drawn', 'red_cards']], ['Time on the pitch', ['minutes']]],
    description: 'Search any player with match records in our leagues and European cups.',
    search: 'Search a player', suggestions: ['Haaland', 'Salah', 'Kane', 'Yamal'],
    introduction: 'See how a player compares with others in the same position, how often they reach a number, and every match behind the average.',
    topics: 'Shots · Goals · Minutes'},
  teams: {label: 'Teams', singular: 'team', metrics: teamMetrics, defaultStat: 'goals_for',
    common: ['goals_for', 'corners_for', 'sot_for', 'cards', 'xg_for'],
    groups: [['Attacking', ['shots_for', 'possession', 'offsides']], ['Defending', ['goals_against', 'xg_against', 'corners_against', 'shots_against', 'sot_against']], ['Discipline', ['fouls', 'red_cards', 'opponent_cards']]],
    description: 'Search a club to see what it creates, what it allows and how it ranks in its league.',
    search: 'Search a team', suggestions: ['Arsenal', 'Real Madrid', 'Inter', 'Bayern'],
    introduction: 'Compare a team with the rest of its league, then split any number by home and away matches.',
    topics: 'Scoring · Chances · Discipline'},
  referees: {label: 'Referees', singular: 'referee', metrics: refereeMetrics, defaultStat: 'cards',
    common: ['cards', 'yellows', 'reds', 'fouls'],
    groups: [['Home team', ['home_cards', 'home_fouls']], ['Away team', ['away_cards', 'away_fouls']]],
    description: 'Search an official to see how many cards and fouls their matches produce.',
    search: 'Search a referee', suggestions: ['Taylor', 'Oliver', 'Marciniak'],
    introduction: 'See an official’s card and foul counts against other referees, then look at the matches behind an average.',
    topics: 'Cards · Fouls · Competitions'}
};

/* Profile rank rows: API metric key -> short label and the match-by-match statistic it opens. */
export const rankMetrics = {
  players: {
    goals_per_90: ['Goals per 90', 'goals'], shots_per_90: ['Shots per 90', 'shots'],
    shots_on_target_per_90: ['On target per 90', 'shots_on_target'], assists_per_90: ['Assists per 90', 'assists'],
    conversion_pct: ['Shot conversion', 'goals'], shooting_accuracy_pct: ['Shooting accuracy', 'shots_on_target'],
    duel_win_pct: ['Duels won', 'duels_won'], pass_accuracy_pct: ['Pass accuracy', 'passes'],
    fouls_won_per_90: ['Fouls won per 90', 'fouls_drawn'], fouls_committed_per_90: ['Fouls committed per 90', 'fouls_committed'],
    cards_per_90: ['Cards per 90', 'yellow_cards'], average_rating: ['Average match rating', 'minutes']
  },
  teams: {
    goals_for: ['Goals scored', 'goals_for'], xg_for: ['Expected goals (xG)', 'xg_for'],
    shots_on_target_for: ['Shots on target', 'sot_for'], corners_for: ['Corners won', 'corners_for'],
    possession: ['Possession', 'possession'], goals_against: ['Goals conceded', 'goals_against'],
    xg_against: ['xG conceded', 'xg_against'], shots_on_target_faced: ['On target faced', 'sot_against'],
    corners_against: ['Corners conceded', 'corners_against'], cards: ['Cards', 'cards']
  },
  referees: {
    cards: ['Cards per match', 'cards'], yellows: ['Yellows per match', 'yellows'], reds: ['Reds per match', 'reds'],
    fouls: ['Fouls per match', 'fouls'], cards_per_foul: ['Cards per foul', 'cards']
  }
};

export const rankGroups = {
  players: {
    scoring: ['Scoring and shooting', ['goals_per_90', 'shots_per_90', 'shots_on_target_per_90', 'assists_per_90', 'conversion_pct']],
    duels: ['Duels and passing', ['duel_win_pct', 'pass_accuracy_pct', 'fouls_won_per_90']],
    discipline: ['Discipline', ['fouls_committed_per_90', 'cards_per_90']],
    overall: ['Overall', ['average_rating']],
    order: {G: ['overall', 'duels', 'discipline'], D: ['duels', 'discipline', 'overall', 'scoring'],
      M: ['scoring', 'duels', 'discipline', 'overall'], F: ['scoring', 'duels', 'discipline', 'overall']}
  },
  teams: {
    attack: ['Attack', ['goals_for', 'xg_for', 'shots_on_target_for', 'corners_for', 'possession']],
    defence: ['Defence', ['goals_against', 'xg_against', 'shots_on_target_faced', 'corners_against']],
    discipline: ['Discipline', ['cards']],
    order: {all: ['attack', 'defence', 'discipline']}
  },
  referees: {
    cards: ['Cards and fouls', ['cards', 'yellows', 'reds', 'fouls', 'cards_per_foul']],
    order: {all: ['cards']}
  }
};

export const frequencyLabels = {
  shots_on_target_1_plus: '1+ shot on target', shots_on_target_2_plus: '2+ shots on target', shots_3_plus: '3+ shots',
  scored: 'Scored', assisted: 'Assisted', booked: 'Booked',
  over_2_5_goals: '3+ goals in the match', both_teams_scored: 'Both teams scored', clean_sheet: 'Clean sheet',
  corners_10_plus: '10+ corners in the match', cards_4_plus: '4+ cards in the match',
  cards_5_plus: '5+ cards in the match', red_card_shown: 'Showed a red card', fouls_25_plus: '25+ fouls called'
};
