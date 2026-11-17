from __future__ import annotations
# Thread caps must precede all numerical imports, including in spawned workers.
import os
for _key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[_key]='1'
import argparse,json,sys,time,subprocess,copy
from pathlib import Path
import yaml,numpy as np
from .common import *

def load_config(args):
    p=Path(args.config) if args.config else ROOT/'configs'/f'{args.profile}.yaml'
    c=yaml.safe_load(p.read_text());c['offline']=args.offline
    if args.data_root:c['data_root']=str(Path(args.data_root).resolve())
    if args.stop_after=='preview':c['run_seeds']=list(range(5))
    return c

def init_output(c,args):
    from .artifacts import metadata
    out=Path(args.output) if args.output else output_for(c);out.mkdir(parents=True,exist_ok=True)
    if not (out/'protocol_manifest.json').exists():metadata(c,out)
    (ROOT/'CURRENT_OUTPUT').write_text(str(out)+'\n')
    print('OUTPUT '+str(out),flush=True);return out

def prepare(c):
    from .data import prepare_all
    from .evaluation import prepare_references
    from .runner import shared_initial_ids,prepared
    prepare_all(c,ROOT)
    reference_root=Path(c['reference_cache_dir']).parents[1] if c.get('reference_cache_dir') else ROOT
    prepare_references(reference_root,c)
    (ROOT/'data/initial').mkdir(parents=True,exist_ok=True)
    for ds in c['datasets']:
        X=prepared(ds,20)[0]
        for seed in c['run_seeds']:
            p=ROOT/f'data/initial/{ds}_{seed}_{c["n0"]}.npy'
            if not p.exists():np.save(p,shared_initial_ids(X,ds,seed,n0=c['n0']))

def main():
    ap=argparse.ArgumentParser(description='FisherBalancing real-response protocol v2')
    ap.add_argument('command',choices=['doctor','download','prepare','run','benchmark','reevaluate','summarize','verify-artifacts','export-paper','pipeline','status','audit-replay'])
    ap.add_argument('--config');ap.add_argument('--profile',default='server_main');ap.add_argument('--output');ap.add_argument('--runtime-plan',default='auto');ap.add_argument('--resume',action='store_true');ap.add_argument('--missing-only',action='store_true');ap.add_argument('--max-pilot-minutes',type=float,default=10);ap.add_argument('--max-wall-minutes',type=float);ap.add_argument('--stop-after',choices=['prepare','smoke','pilot','preview','main']);ap.add_argument('--offline',action='store_true');ap.add_argument('--data-root');ap.add_argument('--cpu-only',action='store_true');ap.add_argument('--workers',type=int);ap.add_argument('--run-path')
    args=ap.parse_args();c=load_config(args);cmd=args.command;started=time.monotonic();deadline=time.time()+args.max_wall_minutes*60 if args.max_wall_minutes else None
    from . import runtime
    if cmd=='doctor':print(json.dumps(runtime.doctor(c),indent=2));return
    if cmd=='download':
        from .data import download_all
        print(json.dumps(download_all(c,ROOT),default=serial));return
    if cmd in ['prepare','pipeline']:
        print('STAGE prepare',flush=True);runtime.doctor(c);prepare(c)
        if cmd=='prepare' or args.stop_after=='prepare':return
    out=init_output(c,args)
    if cmd=='status':
        print(json.dumps(read_json(out/'progress.json',{}),indent=2));print(json.dumps(read_json(out/'verification.json',{}),indent=2));return
    if cmd=='pipeline':
        runtime.doctor(c,out)
        stage_start=time.monotonic();print('STAGE tests',flush=True)
        testlog=out/'pytest.log'
        with testlog.open('w') as log:p=subprocess.run([sys.executable,'-m','pytest','-q'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        print(testlog.read_text(),flush=True)
        if p.returncode:raise RuntimeError('Correctness tests failed; see '+str(testlog))
        print('STAGE smoke',flush=True);smoke=yaml.safe_load((ROOT/'configs/smoke.yaml').read_text());smokeout=output_for(smoke);smokeout.mkdir(parents=True,exist_ok=True)
        from .artifacts import metadata
        metadata(smoke,smokeout)
        smokeplan={'cpu_workers':min(16,runtime.resource_limits()[0]),'sdp_queue_limit':4}
        result=runtime.run_grid(smoke,smokeout,smokeplan,deadline=deadline)
        if result['failed']:raise RuntimeError('Smoke trajectories failed; see '+str(smokeout))
        if args.stop_after=='smoke':return
        print('STAGE timing_pilot',flush=True)
        plan=runtime.benchmark(c,out,args.max_pilot_minutes)
        if args.workers:plan['cpu_workers']=min(args.workers,runtime.resource_limits()[0]);atomic_json(out/'runtime_plan.json',plan)
        pilot_tasks=[t for t in tasks_for(c) if t[1] in [5,20] and t[2]==0]
        result=runtime.run_grid(c,out,plan,round_limit=2,deadline=deadline,tasks=pilot_tasks)
        if result['failed']:raise RuntimeError('Main-compatible timing pilot failed')
        atomic_json(out/'stage_times.json',{'pre_main_seconds':time.monotonic()-started,'tests_smoke_pilot_seconds':time.monotonic()-stage_start})
        if args.stop_after=='pilot':return
        cmd='run'
    if cmd=='benchmark':runtime.benchmark(c,out,args.max_pilot_minutes);return
    if cmd=='run':
        plan=read_json(out/'runtime_plan.json')
        if plan is None:plan=runtime.benchmark(c,out,args.max_pilot_minutes)
        if args.workers:plan['cpu_workers']=min(args.workers,runtime.resource_limits()[0])
        print('STAGE server_main',flush=True);result=runtime.run_grid(c,out,plan,deadline=deadline)
        if deadline and time.time()>=deadline:
            print('WALL_LIMIT_REACHED checkpoints preserved',flush=True);return
        if result['failed']:print('WARNING failed trajectories remain; reports will expose them',flush=True)
        if args.command!='pipeline':return
        cmd='summarize'
    if cmd=='reevaluate':
        from .runner import reevaluate_one
        for task in tasks_for(c):reevaluate_one(task,c,out,args.missing_only)
        return
    if cmd in ['summarize','export-paper']:
        print('STAGE summarize_and_figures',flush=True)
        from .reporting import summarize
        print(json.dumps(summarize(c,out,seeds=list(range(5)) if args.stop_after=='preview' or args.profile=='server_preview' else None),default=serial),flush=True)
        if args.stop_after=='preview' or args.profile=='server_preview':return
        if args.command!='pipeline':return
        cmd='verify-artifacts'
    if cmd=='verify-artifacts':
        from .artifacts import verify
        result=verify(c,out);result['pipeline_wall_seconds']=time.monotonic()-started
        atomic_json(out/'verification.json',result);print(json.dumps(result,indent=2),flush=True)
        if args.command=='pipeline':atomic_json(out/'pipeline_status.json',{'state':'MAIN_COMPLETE' if result['MAIN_COMPLETE'] else 'INCOMPLETE','finished_at':time.time(),**result})
        return
    if cmd=='audit-replay':
        if not args.run_path:raise ValueError('--run-path required')
        from .runner import checkpoint_files,prepared
        p=Path(args.run_path);identity=read_json(p/'config.json');ds,d,seed,method=identity['task'];X=prepared(ds,d)[0]
        from .audit_replay import replay_batch
        files=checkpoint_files(p)
        for previous,(j,a,obj) in zip(files,files[1:]):
            old=np.load(previous[1],allow_pickle=False)['selected_ids']
            result=replay_batch(X,obj['acquisition_audit'],FAMILIES[ds],rho=c['rho'],labeled_ids=old,config={**identity['config'],'dataset':ds,'seed':seed,'acquisition_floor':1e-12 if ds!='movielens' else 0.})
            print(json.dumps(result,default=serial))

if __name__=='__main__':main()
