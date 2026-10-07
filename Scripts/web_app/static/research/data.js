/* Fictional entities and synthetic records for the labelled design preview. */
const playerMetrics = {
    shots: {label:'Shots', title:'Total shots', noun:'shots', verb:'Took a shot', definition:'All recorded attempts at goal, including attempts on target, off target and blocked. The chart shows each match separately.'},
    sot: {label:'On target', title:'Shots on target', noun:'shots on target', verb:'Had a shot on target', definition:'Attempts recorded as on target, usually a goal or an effort saved by the goalkeeper. These are match statistics, not a probability of doing it again.'},
    goals: {label:'Goals', title:'Goals', noun:'goals', verb:'Scored', definition:'Goals credited to this player in each match. Own goals are excluded. A run of goals does not by itself establish a valuable bet.'},
    assists: {label:'Assists', title:'Assists', noun:'assists', verb:'Recorded an assist', definition:'Assists credited to this player in the match record. Providers and bookmakers can use different definitions; a statistical record does not establish bet settlement.'},
    cards: {label:'Cards', title:'Yellow cards', noun:'yellow cards', verb:'Received a yellow card', definition:'Recorded yellow cards, shown separately from red cards. This is not booking points or a bookmaker settlement total.'},
    passes: {label:'Passes', title:'Passes attempted', noun:'passes', verb:'Attempted a pass', threshold:30, step:10, definition:'Recorded passes attempted, including successful and unsuccessful passes. More minutes and a different team role can both affect the count.'},
    tackles: {label:'Tackles', title:'Tackles', noun:'tackles', verb:'Recorded a tackle', definition:'Tackles credited to the player in the source record. Provider definitions matter; these sample values illustrate the design only.'},
    interceptions: {label:'Interceptions', title:'Interceptions', noun:'interceptions', verb:'Recorded an interception', definition:'Recorded interceptions of an opponent’s pass. Match context and a player’s role can change how often these opportunities arise.'},
    fouls: {label:'Fouls committed', title:'Fouls committed', noun:'fouls', verb:'Committed a foul', definition:'Fouls attributed to this player in the match record. A foul does not necessarily result in a card.'},
    foulsDrawn: {label:'Fouls won', title:'Fouls won', noun:'fouls won', verb:'Won a foul', definition:'Fouls recorded as drawn by this player. These are separate from fouls committed.'},
    redCards: {label:'Red cards', title:'Red cards', noun:'red cards', verb:'Received a red card', definition:'Recorded red cards, separate from yellow cards. Check exact source and settlement definitions before using any card statistic for a bet.'}
  };
  const opponents = [['Alderwick','ALD'],['Brookwell','BRK'],['Creston','CRE'],['Dunleigh','DUN'],['Elmbridge','ELM'],['Fairhaven','FAI'],['Glenhurst','GLN'],['Highmoor','HIG'],['Ivydale','IVY'],['Larkspur','LAR']];
  const templates = [
    {id:'jamie-mercer', name:'Jamie Mercer', club:'Northbridge FC', position:'Forward', initials:'JM', next:'Northbridge v Elmbridge'},
    {id:'leo-santos', name:'Leo Santos', club:'Westhaven FC', position:'Midfielder', initials:'LS', next:'Westhaven v Creston'},
    {id:'noah-bennett', name:'Noah Bennett', club:'Kingsmere FC', position:'Defender', initials:'NB', next:'Kingsmere v Highmoor'},
    {id:'rory-vale', name:'Rory Vale', club:'Eastford FC', position:'Goalkeeper', initials:'RV', next:'Eastford v Brookwell'}
  ];
  const minutes = [90,82,70,90,25,88,90,66,81,90,78,90,19,85,0,90,74,90,64,83];
  const shots = [3,2,4,3,1,5,2,0,3,4,3,4,1,2,0,5,2,3,1,4];
  const onTarget = [1,1,2,0,0,3,1,0,2,1,1,2,0,1,null,3,1,null,0,2];
  const players = templates.map((player, p) => ({...player, records:minutes.map((mins,i) => {
    const [opponent,abbr] = opponents[i % opponents.length];
    const day = new Date(Date.UTC(2026,6,2 + i * 5)).toISOString().slice(0,10);
    const played = mins > 0;
    const s = p === 3 ? 0 : Math.max(0, shots[i] - p);
    const target = onTarget[i] == null ? null : Math.min(s, Math.max(0, onTarget[i] - (p === 2 ? 1 : 0)));
    const stats = {
      shots:s, sot:p === 3 ? null : target,
      goals:target == null ? null : Math.min(target, (i+p) % 4 === 1 ? 1 : 0),
      assists:played && (i+p) % 7 === 1 && p !== 3 ? 1 : 0,
      cards:played && (i+p) % (p === 2 ? 4 : 6) === 0 ? 1 : 0,
      passes:Math.round((24 + p*11 + i%5*4) * mins / 90),
      tackles:p === 3 ? 0 : (i+p) % 4,
      interceptions:p === 3 ? 0 : (i+2*p) % 3,
      fouls:p === 3 ? 0 : i % 3,
      foulsDrawn:p === 3 ? 0 : (i+1) % 4,
      redCards:0
    };
    if (!played) Object.keys(stats).forEach(key => { stats[key] = null; });
    return {id:`${player.id}-${i}`, date:day, opponent, abbr, venue:i%2 ? 'away':'home', minutes:p === 1 && i === 18 ? null : mins, appeared:played, stats};
  })}));
const metric = (label,title,noun,definition,extra={}) => ({label,title,noun,definition,...extra});
Object.assign(playerMetrics,{
  keyPasses:metric('Key passes','Key passes','key passes','Recorded passes leading to an attempt at goal. This is separate from an assist.'),
  dribbles:metric('Dribbles','Successful dribbles','successful dribbles','Recorded successful attempts to move past an opponent with the ball.'),
  saves:metric('Saves','Goalkeeper saves','saves','Recorded goalkeeper saves. A save does not imply a clean sheet or predict the next match.')
});
players.forEach((p,index)=>p.records.forEach((r,i)=>{
  Object.assign(r.stats,{keyPasses:r.appeared?(index===3?0:i%4):null,dribbles:r.appeared?(index===3?0:(i+index)%3):null,saves:r.appeared?(index===3?2+i%5:0):null});
}));
const teamMetrics = {
  goals:metric('Goals','Goals scored','goals','Goals scored by this team during regulation time, including stoppage time. Penalty shootouts and extra time are excluded.'),
  corners:metric('Corners','Corners won','corners','Corners credited to the selected team. Team corners are different from total match corners.',{threshold:5}),
  sot:metric('On target','Shots on target','shots on target','Attempts on target by the selected team, rather than the match total.',{threshold:4}),
  cards:metric('Cards','Yellow cards received','yellow cards','Yellow cards recorded for this team. Red cards and bookmaker booking points are separate measures.',{threshold:2}),
  possession:metric('Possession','Possession','possession','The team’s recorded share of possession. This is a percentage, not a count or a probability of winning.',{percent:true,threshold:50,step:5,limit:100}),
  conceded:metric('Goals conceded','Goals conceded','goals conceded','Regulation-time goals scored by the opponent. Lower historical counts alone do not establish a valuable bet.'),
  cornersAgainst:metric('Corners conceded','Corners conceded','corners conceded','Corners credited to the opponent in matches involving this team.',{threshold:5}),
  shots:metric('Shots','Total shots','shots','All attempts by this team, including attempts on target, off target and blocked.',{threshold:10,step:1,limit:30}),
  shotsAgainst:metric('Shots faced','Shots faced','shots faced','All recorded attempts by the opponent.',{threshold:10,limit:30}),
  sotAgainst:metric('On target faced','Shots on target faced','shots on target faced','Attempts on target recorded for the opponent.',{threshold:4}),
  fouls:metric('Fouls','Fouls committed','fouls','Fouls attributed to the selected team. A foul does not necessarily result in a card.',{threshold:10,limit:30}),
  offsides:metric('Offsides','Offsides','offsides','Offside offences attributed to the selected team.'),
  redCards:metric('Red cards','Red cards received','red cards','Recorded red cards for this team. This does not combine yellow cards, second-yellow dismissals or booking points into a betting settlement total.')
};
const teams=templates.slice(0,3).map((p,t)=>({id:p.club.toLowerCase().replaceAll(' ','-'),name:p.club,club:'Example League',position:'Team',initials:p.club.split(' ').map(w=>w[0]).join(''),next:p.next,records:Array.from({length:20},(_,i)=>{
 const [opponent,abbr]=opponents[i%10],goals=(i+t)%4,conceded=(i+2*t)%3,sot=goals+2+i%4;
 return {id:`team-${t}-${i}`,date:new Date(Date.UTC(2026,6,2+i*5)).toISOString().slice(0,10),opponent,abbr,venue:i%2?'away':'home',minutes:90,appeared:true,stats:{goals,conceded,corners:i===17?null:3+(i+t)%7,cornersAgainst:2+(i+2*t)%6,sot,shots:sot+4+i%6,shotsAgainst:conceded+7+i%8,sotAgainst:conceded+1+i%4,cards:(i+t)%5,redCards:i===6&&t===1?1:0,possession:i===12?null:42+(i*3+t*2)%25,fouls:8+(i+t)%9,offsides:i%4}};
})}));
const refereeMetrics = {
 cards:metric('Yellow cards','Yellow cards per match','yellow cards','Recorded yellow cards across both teams in regulation time. These are match totals, not bookmaker booking points.',{threshold:4}),
 redCards:metric('Reds','Red cards per match','red cards','Recorded red cards across both teams. Keep direct reds, second-yellow dismissals and bookmaker definitions separate before drawing settlement conclusions.'),
 fouls:metric('Fouls','Fouls per match','fouls','Recorded fouls across both teams. This reflects the teams and match context as well as the referee.',{threshold:20,step:1,limit:50}),
 penalties:metric('Penalties','Penalties awarded','penalties','Penalties awarded during regulation play, regardless of whether they were scored. Shootout attempts are excluded.'),
 homeCards:metric('Home yellows','Home-team yellow cards','home yellow cards','Yellow cards attributed to the home team. Comparing raw home and away counts does not establish referee bias.',{threshold:2}),
 awayCards:metric('Away yellows','Away-team yellow cards','away yellow cards','Yellow cards attributed to the away team. Team style, competition and match state can affect these counts.',{threshold:2}),
 homeFouls:metric('Home fouls','Home-team fouls','home fouls','Recorded fouls by the home team.',{threshold:10,limit:30}),
 awayFouls:metric('Away fouls','Away-team fouls','away fouls','Recorded fouls by the away team.',{threshold:10,limit:30})
};
const referees=[['alex-harper','Alex Harper','AH'],['morgan-reed','Morgan Reed','MR'],['sam-ellis','Sam Ellis','SE']].map(([id,name,initials],r)=>({id,name,initials,club:'Example League',position:'Referee',records:Array.from({length:20},(_,i)=>{
 const [home,ha]=opponents[i%10],[away,aa]=opponents[(i+3+r)%10];
 const homeCards=1+(i+r)%3,awayCards=(i+2*r)%4,homeFouls=7+(i+r)%8,awayFouls=8+(i+2*r)%9;
 return {id:`ref-${r}-${i}`,date:new Date(Date.UTC(2026,6,2+i*5)).toISOString().slice(0,10),opponent:`${home} v ${away}`,abbr:`${ha}/${aa}`,venue:null,minutes:90,appeared:true,stats:{homeCards,awayCards,cards:i===17?null:homeCards+awayCards,redCards:i===6+r?1:0,homeFouls,awayFouls,fouls:i===12?null:homeFouls+awayFouls,penalties:i%7===r?1:0}};
})}));
export const catalogues={
 players:{label:'Players',singular:'player',defaultStat:'sot',metrics:playerMetrics,entities:players,common:['shots','sot','goals','assists','cards'],groups:[['Passing',['passes','keyPasses']],['Attacking',['dribbles']],['Defending',['tackles','interceptions']],['Discipline',['fouls','foulsDrawn','redCards']],['Goalkeeping',['saves']]],description:'Look beyond the total. Explore recent performances, minutes and match context.',search:'Search a player, club or position',context:'Time on the pitch'},
 teams:{label:'Teams',singular:'team',defaultStat:'goals',metrics:teamMetrics,entities:teams,common:['goals','corners','sot','cards','possession'],groups:[['Attacking',['shots','offsides']],['Defending',['conceded','cornersAgainst','shotsAgainst','sotAgainst']],['Discipline',['fouls','redCards']]],description:'Explore how a team scores, concedes and creates chances, at home and away.',search:'Search a team',context:'Results in this period'},
 referees:{label:'Referees',singular:'referee',defaultStat:'cards',metrics:refereeMetrics,entities:referees,common:['cards','redCards','fouls','penalties'],groups:[['Home team',['homeCards','homeFouls']],['Away team',['awayCards','awayFouls']]],description:'See cards, fouls and penalties across the matches an official has overseen.',search:'Search a referee',context:'Context matters'}
};
