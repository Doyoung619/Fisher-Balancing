"""Resource-aware shared CPU budget and measured, class-specific ETA."""
from pathlib import Path
import os,sys,time,json,subprocess,platform,importlib,math,statistics,concurrent.futures as cf
import numpy as np
import psutil
from threadpoolctl import threadpool_info
from .common import *

def command(args):
    try:return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT,timeout=20)
    except Exception as e:return str(e)
def resource_limits():
    cpus=len(os.sched_getaffinity(0));mem=psutil.virtual_memory().available;limits={}
    cg=Path('/sys/fs/cgroup')/Path(Path('/proc/self/cgroup').read_text().strip().split('::')[-1]).relative_to('/')
    for p in [cg,*cg.parents]:
        if not str(p).startswith('/sys/fs/cgroup'):continue
        for f in ['cpu.max','memory.max','cpuset.cpus.effective']:
            if (p/f).exists():
                s=(p/f).read_text().strip();limits[str(p/f)]=s
                if f=='cpu.max' and s.split()[0]!='max':cpus=min(cpus,max(1,int(int(s.split()[0])/int(s.split()[1]))))
                if f=='memory.max' and s!='max':mem=min(mem,int(s))
    return cpus,mem,limits

def doctor(config,out=None):
    versions={}
    for name in ['numpy','scipy','pandas','sklearn','cvxpy','scs','clarabel','torch','cupy','numba','pyarrow','pytest']:
        try:module=importlib.import_module(name);versions[name]=getattr(module,'__version__','unknown')
        except Exception as e:versions[name]={'unavailable':str(e)}
    cpus,mem,limits=resource_limits()
    result={'at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'hostname':platform.node(),'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'physical_cores':psutil.cpu_count(logical=False),'logical_cpus':psutil.cpu_count(),'affinity':sorted(os.sched_getaffinity(0)),'effective_cpu_limit':cpus,'effective_memory_available_bytes':mem,'cgroup_limits':limits,'memory':dict(psutil.virtual_memory()._asdict()),'versions':versions,'blas':threadpool_info(),'gpu_devices':command(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.free,driver_version','--format=csv,noheader']),'gpu_processes':command(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader']),'nvidia_smi':command(['nvidia-smi']),'allowed_gpu_ids':[0,1,2,3],'other_processes':command(['ps','-eo','pid,ppid,pcpu,rss,args','--sort=-pcpu'])[:20000]}
    atomic_json((Path(out) if out else ROOT)/'environment.json',result)
    return result

def _throughput_probe(payload):
    from .runner import prepared
    from .numerics import fit_glm
    from .evaluation import evaluate_checkpoint
    from .selectors import select_batch
    ds,d,n,config,repeat=payload;X,y,Xt,yt=prepared(ds,d);rng=np.random.default_rng(775+repeat)
    ids=rng.choice(len(X),min(n,len(X)-600),replace=False);mask=np.ones(len(X),bool);mask[ids]=False
    K=config['datasets'][ds]['candidate_cap'];m=config['datasets'][ds]['batch_size'];cand=rng.choice(np.flatnonzero(mask),K,replace=False)
    start=time.monotonic()
    theta,audit=fit_glm(X[ids],y[ids],FAMILIES[ds],rho=config['rho'])
    method=['fb_sample','a_opt','bait_glm','d_opt'][repeat%4]
    chosen,_=select_batch(method,X,ids,cand,theta,FAMILIES[ds],m,config['rho'],rng,config,batch_index=10)
    ids=np.r_[ids,chosen];theta,audit=fit_glm(X[ids],y[ids],FAMILIES[ds],rho=config['rho'],theta0=theta)
    # Timing only; evaluator reference here is deliberately isolated from all science artifacts.
    evaluate_checkpoint(X,y,Xt,yt,ids,theta,FAMILIES[ds],config['rho'],theta,theta,audit,extra={'n0':config['n0']})
    return time.monotonic()-start

def benchmark(config,out,max_minutes=10):
    out=Path(out);start=time.monotonic();cpus,mem,_=resource_limits();results=[]
    # All probes use actual real responses and complete fit/select/evaluate operations.
    # Uniform timing-only late prefixes never enter a main trajectory.
    for workers in [min(32,cpus),min(16,cpus),min(64,cpus),min(24,cpus),min(48,cpus)]:
        if workers in [r['workers'] for r in results]:continue
        if time.monotonic()-start>max_minutes*60*.3 and len(results)>=3:break
        payload=[('obd',20,[2080,4080][i%2],config,i) for i in range(96)]
        t=time.monotonic()
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:times=list(ex.map(_throughput_probe,payload))
        wall=time.monotonic()-t
        results.append({'workers':workers,'blas_threads':1,'tasks':len(payload),'wall_seconds':wall,'throughput_tasks_per_second':len(payload)/wall,'median_task_seconds':float(np.median(times)),'p90_task_seconds':float(np.quantile(times,.9))})
        print('CPU_BENCHMARK '+json.dumps(results[-1]),flush=True)
    best=max(results,key=lambda x:x['throughput_tasks_per_second'])
    plan={'backend':'numpy_cpu_float64','cpu_workers':best['workers'],'blas_threads_per_worker':1,'cpu_worker_measurements':results,'sdp_queue_limit':min(8,best['workers']),'cpu_tokens_shared_across_queues':True,'allowed_gpu_ids':[0,1,2,3],'used_gpu_ids':[],'selected_reason':'Fastest measured full fit/select/evaluation CPU throughput on actual OBD late-state timing-only subsets; CUDA audit recorded separately when available.','cuda_measurements':read_json(ROOT/'cuda_benchmark.json',{'status':'not_yet_measured','reason':'PyTorch absent in primary interpreter; available in separate existing environment'}),'benchmark_seconds':time.monotonic()-start,'scientific_rng_independent_of_scheduler':True,'pilot_coverage_notes':'CPU worker search uses OBD d20 late prefixes n=2080/4080 and four selector classes. Main-compatible pilot separately covers all method/family and d5/20 with actual m,K. Full solver queue scaling not exhaustively searched.'}
    env=read_json(out/'environment.json',{});plan['allowed_gpu_uuid_mapping']=env.get('gpu_devices')
    atomic_json(out/'runtime_plan.json',plan);atomic_json(out/'benchmark_runtime.json',plan)
    return plan

def telemetry(out,completed,total,inflight,start,eta_seconds=None):
    p=psutil.Process();children=p.children(recursive=True);rss=sum(c.memory_info().rss for c in children if c.is_running())+p.memory_info().rss
    row={'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'completed':completed,'expected':total,'inflight':inflight,'elapsed_seconds':time.monotonic()-start,'eta_seconds':eta_seconds,'cpu_percent':psutil.cpu_percent(),'rss_bytes':rss,'available_ram_bytes':psutil.virtual_memory().available}
    path=Path(out)/'runtime_telemetry.csv'
    import csv
    with path.open('a') as f:
        w=csv.DictWriter(f,fieldnames=list(row));
        if f.tell()==0:w.writeheader()
        w.writerow(row)
    atomic_json(Path(out)/'progress.json',row)
    return row

def estimate_remaining(pending,inflight,measurements,config,workers,sdp_limit=8):
    costs=[];sdp_costs=[];missing=0
    for task in list(pending)+list(inflight):
        ds,d,s,m=task;key=(ds,d,m);samples=measurements.get(key,[])
        if not samples:
            samples=measurements.get((ds,20,m),[]) or measurements.get((ds,5,m),[])
        if samples:
            cost=float(np.median(samples));costs.append(cost)
            if m=='e_sdp':sdp_costs.append(cost)
        else:missing+=1
    if not costs:return None
    # Longest remaining trajectory and shared worker lower bound; 25% overhead reserve.
    if missing:return None
    return 1.25*max(sum(costs)/workers,sum(sdp_costs)/sdp_limit,max(costs))

def run_grid(config,out,plan,round_limit=None,deadline=None,tasks=None):
    from .runner import run_one,checkpoint_files
    tasks=list(tasks or tasks_for(config));total=len(tasks);out=Path(out)
    existing=[];pending=[];measurements={}
    for key,entry in read_json(out/'runtime_costs.json',{}).items():
        ds,d,m=key.split('/');measurements[(ds,int(d),m)]=[entry['median']]
    for task in tasks:
        s=read_json(run_path(out,task)/'status.json',{});ds,d,seed,m=task
        rounds=int(np.ceil(config['datasets'][ds]['additional_labels']/config['datasets'][ds]['batch_size']))
        if s.get('state')=='EVALUATION_COMPLETE':
            existing.append(task);measurements.setdefault((ds,d,m),[]).append(s['total_elapsed_seconds'])
        elif round_limit is not None and s.get('completed_rounds',-1)>=round_limit:existing.append(task)
        else:pending.append(task)
    # Long jobs first with preview seeds slightly preferred; never a preview barrier.
    pending.sort(key=lambda t:(-(6 if t[0]=='obd' else 1)*(4 if t[3]=='e_sdp' else 2 if t[3]=='e_greedy' else 1)*t[1], t[2]>=5,t[2]))
    workers=plan['cpu_workers'];sdp_limit=plan.get('sdp_queue_limit',8);completed=len(existing);failed=0;inflight={};start=time.monotonic();last_log=0;preview_done=False;preview_process=None
    with cf.ProcessPoolExecutor(max_workers=workers) as executor:
        while pending or inflight:
            if deadline and time.time()>=deadline:pending=[]
            sdp_inflight=sum(t[3]=='e_sdp' for t in inflight.values())
            while len(inflight)<workers and pending:
                idx=next((i for i,t in enumerate(pending) if t[3]!='e_sdp' or sdp_inflight<sdp_limit),None)
                if idx is None:break
                task=pending.pop(idx);f=executor.submit(run_one,task,config,str(out),round_limit,deadline);inflight[f]=task
                if task[3]=='e_sdp':sdp_inflight+=1
            if not inflight:break
            done,_=cf.wait(list(inflight),timeout=10,return_when=cf.FIRST_COMPLETED)
            for f in done:
                task=inflight.pop(f)
                try:result=f.result()
                except Exception as e:result={'state':'FAILED','task':task,'error':repr(e)}
                if result['state']=='FAILED':
                    failed+=1;print('RUN_FAILED '+json.dumps(result),flush=True)
                else:
                    completed+=1
                    ds,d,seed,m=task
                    if result['state']=='EVALUATION_COMPLETE':cost=result['total_elapsed_seconds']
                    else:
                        files=checkpoint_files(run_path(out,task));rows=[v[2]['metrics'] for v in files]
                        sampled=[r['acquisition_seconds']+r['fit_seconds']+r['evaluation_seconds']+r['io_seconds'] for r in rows if r['round']>0]
                        planned_rounds=int(np.ceil(config['datasets'][ds]['additional_labels']/config['datasets'][ds]['batch_size']))
                        cost=float(np.mean(sampled))*planned_rounds if sampled else result.get('wall_seconds',1)*planned_rounds
                    measurements.setdefault((ds,d,m),[]).append(cost)
            if time.monotonic()-last_log>10 or not pending and not inflight:
                eta=estimate_remaining(pending,inflight.values(),measurements,config,workers,sdp_limit)
                row=telemetry(out,completed,total,len(inflight),start,eta);row['failed']=failed
                print('PROGRESS '+json.dumps(row),flush=True);last_log=time.monotonic()
                atomic_json(out/'runtime_costs.json',{f'{ds}/{d}/{m}':{'n':len(v),'median':float(np.median(v)),'p90':float(np.quantile(v,.9))} for (ds,d,m),v in measurements.items()})
            if round_limit is None and not preview_done:
                preview=[t for t in tasks if t[2]<5]
                if all(read_json(run_path(out,t)/'status.json',{}).get('state')=='EVALUATION_COMPLETE' for t in preview):
                    # Lightweight summary runs asynchronously using an available CPU token.
                    atomic_json(out/'preview_ready.json',{'seeds':[0,1,2,3,4],'label':'5-seed preliminary','ready':True,'at':time.time()})
                    if workers < resource_limits()[0] or len(inflight)<workers:
                        preview_config={**config,'run_seeds':[0,1,2,3,4]}
                        atomic_json(out/'preview_config.json',preview_config)
                        preview_code='import json,sys;from pathlib import Path;from real_response.reporting import summarize;p=Path(sys.argv[1]);summarize(json.loads((p/"preview_config.json").read_text()),p,seeds=list(range(5)))'
                        with (out/'preview.log').open('w') as log:
                            preview_process=subprocess.Popen([sys.executable,'-c',preview_code,str(out)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    preview_done=True
    result={'completed':completed,'failed':failed,'expected':total,'wall_seconds':time.monotonic()-start,'round_limit':round_limit}
    atomic_json(out/('pilot_status.json' if round_limit else 'grid_status.json'),result)
    return result
