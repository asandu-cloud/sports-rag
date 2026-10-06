"""Start/stop the owner-authorized local website without changing other jobs."""
import argparse
import json
import os
from pathlib import Path
import plistlib

from Scripts.ops.match_read_launchd import build_paths, _assert_installable, _run_launchctl, _write_atomically

LABEL = 'com.bettingrag.website'


def render(paths):
    return plistlib.dumps({
        'Label': LABEL,
        'ProgramArguments': [str(paths.python_executable), '-B', '-m', 'uvicorn',
                             'Scripts.web_app.api:app', '--host', '127.0.0.1', '--port', '8000'],
        'WorkingDirectory': str(paths.project_root),
        'EnvironmentVariables': {'PYTHONDONTWRITEBYTECODE': '1',
                                 'PYTHONPATH': str(paths.project_root / 'Scripts')},
        'RunAtLoad': True, 'KeepAlive': True, 'ThrottleInterval': 10,
        'StandardOutPath': str(paths.log_directory / 'website.stdout.log'),
        'StandardErrorPath': str(paths.log_directory / 'website.stderr.log'),
    }, sort_keys=False).decode()


def operate(action, paths):
    target = f'gui/{os.getuid()}/{LABEL}'
    if action == 'render':
        return plistlib.loads(render(paths).encode())
    if action == 'status':
        result = _run_launchctl(['print', target], check=False)
        details = {}
        for line in result.stdout.splitlines():
            key, sep, value = line.strip().partition(' = ')
            if sep and key in ('state', 'pid', 'runs', 'last exit code'):
                details[key] = value
        return {'loaded': result.returncode == 0, 'installed': paths.plist_path.is_file(),
                'label': LABEL, 'process': details, 'url': 'http://127.0.0.1:8000'}
    _assert_installable(paths)
    if paths.plist_path.exists() and paths.plist_path.read_text() != render(paths):
        raise ValueError('An existing website job differs; inspect before replacing it')
    if action == 'stop':
        _run_launchctl(['bootout', target], check=False)
        if _run_launchctl(['print', target], check=False).returncode == 0:
            raise RuntimeError('Website job is still loaded')
        paths.plist_path.unlink(missing_ok=True)
        return {'stopped': True, 'data_preserved': True}
    if action != 'install':
        raise ValueError('Unknown action')
    if _run_launchctl(['print', target], check=False).returncode == 0:
        return {'loaded': True, 'unchanged': True}
    paths.log_directory.mkdir(parents=True, exist_ok=True)
    _write_atomically(paths.plist_path, render(paths))
    _run_launchctl(['bootstrap', f'gui/{os.getuid()}', str(paths.plist_path)], check=True)
    return {'loaded': True, 'url': 'http://127.0.0.1:8000', 'label': LABEL}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('render', 'status', 'install', 'stop'))
    args = parser.parse_args()
    paths = build_paths(plist_path=Path.home() / 'Library/LaunchAgents' / f'{LABEL}.plist')
    print(json.dumps(operate(args.action, paths), indent=2))


if __name__ == '__main__':
    main()
