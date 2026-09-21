"""Conservative operator recovery. Run during a maintenance window.

Never kills processes or deletes input/results. An interrupted job is cloned
to a new token so its partial results and original status remain available.
"""
import argparse
import json
import os
import sys
from pathlib import Path
import psutil
import nar_jobs as jobs


def active_processes(path):
    needle=str(path.resolve()).replace('\\','/').lower()
    found=[]
    for proc in psutil.process_iter(['pid','cmdline']):
        if proc.pid==os.getpid():continue
        try:
            args=proc.info['cmdline'] or []
            worker = any(Path(a).name.lower() == 'nar_jobs.py' for a in args) and path.name in args
            if worker or any(needle in a.replace('\\','/').lower() for a in args):found.append(proc.pid)
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            continue
    return found


def execution_health(token):
    """Read-only liveness check; absence of a visible process is not auto-retry authority."""
    import time
    path=jobs.job_path(token)
    state=jobs.read_state(token)
    if state['state'] not in {'running','queued'}:
        return 'not_pending'
    if time.time()-state.get('submitted_unix',time.time()) < 30:
        return 'starting'
    uncertain=False
    for proc in psutil.process_iter():
        if proc.pid == os.getpid():continue
        try:
            if proc.name().lower() not in {'python.exe','pythonw.exe','python','python3'}:
                continue
            args=proc.cmdline()
            worker=any(Path(a).name.lower()=='nar_jobs.py' for a in args) and token in args
            needle=str(path.resolve()).replace('\\','/').lower()
            child=any(needle in a.replace('\\','/').lower() for a in args)
            if worker or child:return 'active'
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            uncertain=True
    return 'unknown' if uncertain else 'no_worker_detected'


def clone_for_recovery(token):
    path=jobs.job_path(token)
    with jobs.worker_slot():
        state=jobs.read_state(token)
        if state['state'] not in {'running','failed'}:raise ValueError('Only interrupted running or failed jobs can be cloned; queued jobs can be resumed directly.')
        if active_processes(path):raise ValueError('An input-processing child is still active; recovery refused.')
        uploads={key:(path/name).read_bytes() for key,name in jobs.FILES.items() if (path/name).exists()}
        new=jobs.submit(state['kind'],state['species'],uploads,launch=False)
        cloned=jobs.read_state(new)
        cloned['recovery_of']=token
        jobs.save_state(jobs.job_path(new),cloned)
    return new


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('token')
    parser.add_argument('--clone-and-run',action='store_true')
    args=parser.parse_args()
    if not args.clone_and_run:
        print(json.dumps(jobs.read_state(args.token),indent=2))
    else:
        new=clone_for_recovery(args.token)
        print('Recovery result link: ?job='+new,flush=True)
        jobs.run_job(new)
