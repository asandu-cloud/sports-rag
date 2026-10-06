"""Control only the outcome-blind Phase 6 capture job; never a delivery worker."""
import argparse
import json
import os
from pathlib import Path
import plistlib
from Scripts.ops.match_read_launchd import build_paths,_assert_installable,_run_launchctl,_write_atomically
from Scripts.ops.phase6_shadow import verify_release

LABEL='com.bettingrag.phase6-shadow'


def render(paths):
    return plistlib.dumps({'Label':LABEL,
        'ProgramArguments':[str(paths.python_executable),'-B','-m','Scripts.ops.phase6_shadow','capture'],
        'WorkingDirectory':str(paths.project_root),'StartInterval':3600,'RunAtLoad':True,'ProcessType':'Background',
        'StandardOutPath':str(paths.log_directory/'phase6-shadow.stdout.log'),
        'StandardErrorPath':str(paths.log_directory/'phase6-shadow.stderr.log')},sort_keys=False).decode()


def operate(action,paths):
    target=f'gui/{os.getuid()}/{LABEL}'
    if action=='render': return plistlib.loads(render(paths).encode())
    if action=='status':
        result=_run_launchctl(['print',target],check=False);details={}
        for line in result.stdout.splitlines():
            key,sep,value=line.strip().partition(' = ')
            if sep and key in ('state','pid','runs','last exit code'): details[key]=value
        return {'loaded':result.returncode==0,'installed':paths.plist_path.is_file(),'label':LABEL,'process':details}
    _assert_installable(paths)
    if paths.plist_path.exists() and plistlib.loads(paths.plist_path.read_bytes()).get('Label')!=LABEL:
        raise ValueError('Refusing to replace another job')
    if action=='stop':
        _run_launchctl(['bootout',target],check=False)
        if _run_launchctl(['print',target],check=False).returncode==0: raise RuntimeError('Shadow job still loaded')
        paths.plist_path.unlink(missing_ok=True)
        return {'stopped':True,'all_evidence_preserved':True}
    if action!='install': raise ValueError('Unknown action')
    verify_release()
    if paths.plist_path.exists():
        if paths.plist_path.read_text()!=render(paths): raise ValueError('Existing shadow schedule differs')
        if _run_launchctl(['print',target],check=False).returncode==0: return {'loaded':True,'unchanged':True}
    paths.log_directory.mkdir(parents=True,exist_ok=True)
    _write_atomically(paths.plist_path,render(paths))
    _run_launchctl(['bootstrap',f'gui/{os.getuid()}',str(paths.plist_path)],check=True)
    return {'loaded':True,'interval_seconds':3600,'label':LABEL,'outcome_access':False,'public_delivery':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('render','status','install','stop'))
    args=parser.parse_args();paths=build_paths(plist_path=Path.home()/'Library/LaunchAgents'/f'{LABEL}.plist')
    try:
        print(json.dumps(operate(args.action,paths),indent=2));return 0
    except (OSError,RuntimeError,ValueError) as exc:
        print(json.dumps({'status':'failed','reason':str(exc)}));return 1


if __name__=='__main__': raise SystemExit(main())
