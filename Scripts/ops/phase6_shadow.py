"""Outcome-blind Phase 6 capture. Only this independent shadow namespace is writable.

Reads future schedule/canonical forecasts and archived odds using read-only SQL.
Executes the frozen selection worker without database/network access. Never trains,
grades bets, publishes, imports a delivery service or promotes model artifacts.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime,timedelta,timezone
import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]
BUNDLE=ROOT/'Research/phase6-shadow-2026-10-06'
LEAGUES=('EPL','LaLiga','SerieA','Bundesliga','Ligue1')
MARKETS={'goals','corners','cards','sot','btts','moneyline','spreads'}


def now(): return datetime.now(timezone.utc)
def stamp(value): return value.astimezone(timezone.utc).isoformat()
def encoded(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(value): return hashlib.sha256(encoded(value)).hexdigest()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment_identity():
    return {'python':sys.version.split()[0], 'packages':dict(sorted(
        (d.metadata['Name'].lower().replace('_','-'),d.version) for d in importlib.metadata.distributions()
        if d.metadata.get('Name')))}


def utc(value,*,database=False):
    result=datetime.fromisoformat(str(value).replace('Z','+00:00'))
    if result.tzinfo is None:
        if not database: raise ValueError('Timezone required')
        result=result.replace(tzinfo=timezone.utc)  # platform SQLite stores UTC datetimes
    return result.astimezone(timezone.utc)


def unaliased(path):
    path=Path(path).absolute()
    for part in (path,*path.parents):
        if part.is_symlink(): raise ValueError('Symlink is not an isolated shadow path')
    if path.is_file() and path.stat().st_nlink!=1:
        raise ValueError('Hard-linked shadow destination')
    return path


def atomic_new(path,body):
    path=unaliased(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=body: raise ValueError('Immutable record already exists with different content')
        return
    fd,name=tempfile.mkstemp(prefix='.pending-',dir=path.parent)
    temporary=Path(name)
    try:
        with os.fdopen(fd,'wb') as h: h.write(body);h.flush();os.fsync(h.fileno())
        try: os.link(temporary,path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes()!=body: raise ValueError('Concurrent immutable write differs')
        directory=os.open(path.parent,os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally: temporary.unlink(missing_ok=True)


class Store:
    def __init__(self,root):
        self.root=unaliased(root)
        self.root.mkdir(parents=True,exist_ok=True)
    def put(self,body):
        identifier=hashlib.sha256(body).hexdigest()
        atomic_new(self.root/'objects'/identifier[:2]/(identifier+'.gz'),gzip.compress(body,mtime=0))
        return identifier
    def get(self,identifier):
        if len(identifier)!=64 or any(c not in '0123456789abcdef' for c in identifier): raise ValueError('Invalid object hash')
        path=unaliased(self.root/'objects'/identifier[:2]/(identifier+'.gz'))
        data=gzip.decompress(path.read_bytes())
        if hashlib.sha256(data).hexdigest()!=identifier: raise ValueError('Stored artifact checksum mismatch')
        return data
    def resolve_inputs(self,payload):
        for row in payload['fixtures']:
            if row.get('forecast_sha256'):
                if json.loads(self.get(row['forecast_sha256']))!=row['forecast']:
                    raise ValueError('Forecast reference mismatch')
            if row.get('quote_sha256'):
                raw=json.loads(self.get(row['quote_sha256']))
                matches=[r for r in raw['response'] if str(r['fixture']['id'])==str(row['event']['id'])]
                if row['quote_receipt']['fixture_payload'] not in matches:
                    raise ValueError('Quote body is not present in retained receipt')
        return True


@contextmanager
def read_database(path,columns):
    connection=sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True,timeout=5)
    connection.row_factory=sqlite3.Row
    connection.execute('PRAGMA query_only=ON');connection.execute('BEGIN')
    def authorize(action,table,column,*unused):
        if action==sqlite3.SQLITE_READ and column not in columns.get(table,set()): return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK if action in (sqlite3.SQLITE_SELECT,sqlite3.SQLITE_READ,sqlite3.SQLITE_FUNCTION,sqlite3.SQLITE_RECURSIVE) else sqlite3.SQLITE_DENY
    connection.set_authorizer(authorize)
    try: yield connection
    finally: connection.close()


PLATFORM_COLUMNS={
    'fixtures':set('api_football_id competition_id home_team_id away_team_id kickoff_utc status last_fetched_at'.split()),
    'competitions':{'id','code'},'teams':{'id','name'},
    'match_reads':set('fixture_api_id league kickoff evaluated_at created_at read_json read_key stage'.split())}
ODDS_COLUMNS={
    'fixture_snapshots':{'fixture_id','request_id'},'quotes':{'fixture_id','request_id'},
    'requests':set('id run_id requested_at received_at status raw_sha256'.split()),
    'raw_bodies':{'sha256','body_gzip'}}


def verify_release(root=ROOT,bundle=BUNDLE):
    release=json.loads((bundle/'RELEASE.json').read_text())
    identity='phase6-diagnostic.v1:'+digest({k:v for k,v in release.items() if k!='release_id'})
    if identity!=release['release_id']: raise ValueError('Release identity mismatch')
    if release['environment']!=environment_identity(): raise ValueError('Python environment changed; review before further capture')
    for name,wanted in release['files'].items():
        path=(bundle/name).resolve()
        if not path.is_relative_to(bundle.resolve()) or sha(path)!=wanted:
            raise ValueError('Frozen shadow release changed: '+name)
    for name,wanted in release['operational_files'].items():
        if sha(root/name)!=wanted: raise ValueError('Operational prediction input changed; review a new release: '+name)
    for name,wanted in release['entrypoints'].items():
        if sha(root/name)!=wanted: raise ValueError('Shadow entrypoint changed: '+name)
    return release


def capture(root,store,cutoff,release):
    horizon=cutoff+timedelta(days=14);fixtures=[];raw_cache={}
    with read_database(root/'Index/platform.db',PLATFORM_COLUMNS) as db, \
         read_database(root/'Index/odds_archive/odds.sqlite3',ODDS_COLUMNS) as odds:
        placeholders=','.join('?' for _ in LEAGUES)
        rows=db.execute(f'''SELECT f.api_football_id id,c.code league,h.name home_team,a.name away_team,
                   f.kickoff_utc commence_time,f.status,f.last_fetched_at
            FROM fixtures f JOIN competitions c ON c.id=f.competition_id
            JOIN teams h ON h.id=f.home_team_id JOIN teams a ON a.id=f.away_team_id
            WHERE c.code IN ({placeholders}) AND datetime(f.kickoff_utc)>datetime(?)
              AND datetime(f.kickoff_utc)<datetime(?) ORDER BY f.kickoff_utc,f.api_football_id''',
            (*LEAGUES,stamp(cutoff),stamp(horizon))).fetchall()
        for scheduled in rows:
            event=dict(scheduled);event['id']=str(event['id']);event['commence_time']=stamp(utc(event['commence_time'],database=True))
            row={'event':event,'quote_capture_status':'not_attempted_without_eligible_forecast'};fixtures.append(row)
            if event['last_fetched_at'] and utc(event['last_fetched_at'],database=True)>cutoff:
                row['capture_rejection']='schedule_observed_after_cutoff';continue
            if event['status'] not in ('NS','TBD'):
                row['capture_rejection']='fixture_not_scheduled_pre_match';continue
            source=db.execute('''SELECT read_key,read_json,evaluated_at,created_at FROM match_reads
                WHERE fixture_api_id=? AND league=? AND datetime(kickoff)=datetime(?)
                  AND datetime(evaluated_at)<=datetime(?) AND datetime(created_at)<=datetime(?)
                ORDER BY evaluated_at DESC,created_at DESC,read_key LIMIT 1''',
                (event['id'],event['league'],event['commence_time'],stamp(cutoff),stamp(cutoff))).fetchone()
            if not source:
                row['capture_rejection']='no_pre_cutoff_canonical_forecast';continue
            if utc(source['created_at'],database=True)>cutoff or utc(source['evaluated_at'],database=True)>cutoff:
                row['capture_rejection']='source_observed_after_cutoff';continue
            forecast=json.loads(source['read_json'])
            row['forecast']=forecast;row['forecast_sha256']=store.put(encoded(forecast));row['source_read_key']=source['read_key']
            if cutoff-utc(source['evaluated_at'],database=True)>timedelta(seconds=86400):
                row['capture_rejection']='canonical_forecast_older_than_24_hours';continue
            results=forecast.get('canonical_results',[])
            if len(results)!=7 or {r['market']['group'] for r in results}!=MARKETS:
                row['capture_rejection']='incomplete_market_snapshot';continue
            if str(forecast['fixture']['event_id'])!=event['id'] or forecast['fixture']['league']!=event['league']:
                raise ValueError('Stored forecast fixture identity differs')
            for result in results:
                if (utc(result['provenance']['generated_at'])>cutoff
                        or str(result['fixture']['event_id'])!=event['id']
                        or result['fixture']['league']!=event['league']
                        or result['fixture']['home_team']!=event['home_team']
                        or result['fixture']['away_team']!=event['away_team']
                        or utc(result['fixture']['kickoff'])!=utc(event['commence_time'])
                        or result['context']['prediction_system']['files_sha256']!=release['operational_system_files']):
                    raise ValueError('Forecast clock, kickoff or prediction-source identity mismatch')
            receipt=odds.execute('''SELECT r.id,r.requested_at,r.received_at,r.raw_sha256,b.body_gzip
                FROM requests r JOIN raw_bodies b ON b.sha256=r.raw_sha256
                WHERE r.status='ok' AND datetime(r.received_at)<=datetime(?) AND r.id IN (
                  SELECT request_id FROM fixture_snapshots WHERE fixture_id=?
                  UNION SELECT request_id FROM quotes WHERE fixture_id=?)
                ORDER BY r.received_at DESC,r.id LIMIT 1''', (stamp(cutoff),int(event['id']),int(event['id']))).fetchone()
            row['quote_capture_status']='missing_prior_receipt'
            if receipt:
                key=receipt['raw_sha256']
                if key not in raw_cache:
                    body=gzip.decompress(receipt['body_gzip'])
                    if hashlib.sha256(body).hexdigest()!=key: raise ValueError('Archived odds checksum mismatch')
                    store.put(body);raw_cache[key]=json.loads(body)
                matched=[r for r in raw_cache[key]['response'] if str(r['fixture']['id'])==event['id']]
                if len(matched)!=1: raise ValueError('Ambiguous archived fixture row')
                if utc(matched[0]['fixture']['date'])!=utc(event['commence_time']):
                    row['capture_rejection']='odds_schedule_kickoff_mismatch'
                    row['quote_capture_status']='schedule_mismatch';continue
                if not utc(receipt['requested_at'])<=utc(receipt['received_at'])<=cutoff:
                    raise ValueError('Invalid receipt chronology')
                row['quote_sha256']=key
                row['quote_capture_status']='receipt_captured'
                row['quote_receipt']={'id':receipt['id'],'requested_at':receipt['requested_at'],
                    'received_at':receipt['received_at'],'fixture_payload':matched[0]}
    payload={'schema_version':'phase6-captured-inputs.v1','mode':'outcome_blind_feasibility',
        'cutoff':stamp(cutoff),'outcome_access':False,'fixtures':fixtures,
        'recipe':release['recipe'],'evidence':release['evidence'],
        'release_id':release['release_id'],'supported_betting_scope':[],
        'capture_boundary':'future schedule and immutable pre-match canonical forecasts; no historical outcomes or profiles'}
    store.resolve_inputs(payload)
    return payload


def health(payload,output):
    from collections import Counter
    ages=[];price_counts={}
    counters=Counter({'raw_core_quotes':0,'source_fresh_raw_core_quotes':0})
    for row in payload['fixtures']:
        for book in (row.get('quote_receipt') or {}).get('fixture_payload',{}).get('bookmakers',[]):
            if book['id'] not in (7,8): continue
            raw=row['quote_receipt']['fixture_payload'];source=raw.get('update')
            try: age=(utc(payload['cutoff'])-utc(source)).total_seconds() if source else None
            except (ValueError,TypeError): age=None
            if age is not None: ages.append(age)
            for market in book.get('bets',[]):
                if market['id'] not in (1,4,5,8,45,80,87): continue
                key=f"{book['id']}:{market['id']}";price_counts[key]=price_counts.get(key,0)+len(market['values'])
                counters['raw_core_quotes']+=len(market['values'])
                if age is not None and 0<=age<900: counters['source_fresh_raw_core_quotes']+=len(market['values'])
    counts=Counter(r['status'] for r in output['fixtures'])
    lead=Counter()
    for row in payload['fixtures']:
        hours=(utc(row['event']['commence_time'])-utc(payload['cutoff'])).total_seconds()/3600
        band='under_24_hours' if hours<24 else '24_to_96_hours' if hours<96 else '96_to_336_hours'
        lead[band]+=1
    return {'universe_fixtures':len(payload['fixtures']),'fixture_capture_statuses':dict(counts),
        'fixture_lead_time_bands':dict(lead),
        'quote_capture_statuses':dict(Counter(r['quote_capture_status'] for r in payload['fixtures'])),
        'quote_coverage_scope':'current canonical forecast subset; unavailable forecasts retain explicit not-attempted status',
        'source_age_seconds_min_max':[min(ages),max(ages)] if ages else None,
        'raw_quote_counts':dict(counters),'bookmaker_market_counts':price_counts,
        'rejection_counts':output['rejection_counts'],'no_performance_metrics':True,
        'candidate_selected_bets':sum(len(c['source_card']['selections']) for s in output['slates'] if s['arm']=='candidate' for c in s['audit']['candidates'] if c['selected']),
        'raw_quote_counts_are_not_valid_or_executable_prices':True}


def execute_once(root=ROOT,bundle=BUNDLE,*,clock=now):
    release=verify_release(root,bundle)
    directory=unaliased(root/'Index/phase6_shadow');store=Store(directory)
    lock_path=unaliased(directory/'.capture.lock')
    with lock_path.open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: return {'status':'busy','public_delivery':False}
        cutoff=clock()
        contract_path=directory/'feasibility-contract.json'
        if not contract_path.exists():
            atomic_new(contract_path,encoded({'release_id':release['release_id'],'start':stamp(cutoff),
                'end':stamp(cutoff+timedelta(days=14)),'evaluation_weeks':26,
                'evaluation_start_utc':None,'outcome_access':False,'public_delivery':False}))
        contract=json.loads(unaliased(contract_path).read_text())
        if contract['release_id']!=release['release_id']: raise ValueError('Existing shadow epoch belongs to another release')
        if not utc(contract['start'])<=cutoff<utc(contract['end']):
            return {'status':'feasibility_window_closed','public_delivery':False}
        slot=cutoff.strftime('%Y%m%dT%H')
        journal=unaliased(directory/'runs'/(slot+'.json'))
        if journal.exists():
            existing=json.loads(journal.read_text())
            if digest({k:v for k,v in existing.items() if k!='record_id'})!=existing['record_id']:
                raise ValueError('Existing run journal failed integrity verification')
            for key in ('input_sha256','output_sha256'):
                if existing.get(key): store.get(existing[key])
            return {'status':'already_recorded','run':existing['record_id'],'public_delivery':False}
        if shutil.disk_usage(directory).free<2*1024**3: raise RuntimeError('Less than 2 GiB free; preserving existing evidence')
        work=Path(tempfile.mkdtemp(prefix='.run-',dir=directory))
        try:
            payload=capture(root,store,cutoff,release);input_sha=store.put(encoded(payload))
            (work/'input.json').write_bytes(encoded(payload))
            env={'PATH':os.environ.get('PATH','/usr/bin:/bin'),'PYTHONDONTWRITEBYTECODE':'1',
                 'PYTHONHASHSEED':'0','PYTHONNOUSERSITE':'1','PREDICTION_RELEASE_MODE':'shadow',
                 'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
            child=subprocess.run([sys.executable,'-B',str(bundle/'worker.py'),'--runtime',str(bundle/'runtime'),
                '--input',str(work/'input.json'),'--output',str(work/'output.json')],cwd=bundle/'runtime',env=env,
                capture_output=True,text=True,timeout=240)
            if child.returncode:
                raise RuntimeError('Isolated worker failed: '+child.stderr[-2000:])
            output=json.loads((work/'output.json').read_text())
            if output['input_digest']!=digest(payload) or output.get('public_delivery') is not False or output.get('io_violations'):
                raise ValueError('Worker input identity or isolation failure')
            output_sha=store.put(encoded(output))
            end=clock()
            if end-cutoff>timedelta(seconds=300) or any(utc(r['event']['commence_time'])<=end+timedelta(seconds=60) for r in payload['fixtures']):
                raise ValueError('Capture not completed before cutoff deadline/kickoff')
            store.resolve_inputs(payload)
            record={'schema_version':'phase6-shadow-run.v1','status':'captured','release_id':release['release_id'],
                'cutoff':stamp(cutoff),'recorded_at':stamp(end),'input_sha256':input_sha,
                'output_sha256':output_sha,'health':health(payload,output),
                'public_delivery':False,'performance_evaluated':False,'cohort':'feasibility_not_evaluation'}
            record['record_id']=digest(record)
            atomic_new(journal,encoded(record))
            return {'status':'captured','record_id':record['record_id'],'health':record['health'],'public_delivery':False}
        except Exception as exc:
            # A failed slot remains visible; never backfill it later from refreshed inputs.
            record={'status':'failed','cutoff':stamp(cutoff),'recorded_at':stamp(clock()),
                'release_id':release['release_id'],'error':str(exc),'public_delivery':False}
            record['record_id']=digest(record);atomic_new(journal,encoded(record))
            raise
        finally:
            shutil.rmtree(work)


def status(root=ROOT):
    directory=root/'Index/phase6_shadow'
    contract=directory/'feasibility-contract.json'
    if not contract.exists(): return {'status':'not_started','evaluation_weeks':26}
    paths=sorted((directory/'runs').glob('*.json'))
    return {'contract':json.loads(contract.read_text()),'recorded_runs':len(paths),
            'latest':json.loads(paths[-1].read_text()) if paths else None}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('verify','capture','status'))
    args=parser.parse_args(argv)
    try:
        result=(execute_once() if args.action=='capture' else status() if args.action=='status'
                else {'release_id':verify_release()['release_id'],'status':'verified'})
        print(json.dumps(result,indent=2,sort_keys=True));return 0
    except Exception as exc:
        print(json.dumps({'status':'failed','reason':str(exc),'public_delivery':False}));return 1


if __name__=='__main__': raise SystemExit(main())
