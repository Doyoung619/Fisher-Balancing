"""Persistent, balanced, resumable scheduling for the continuous OFAT queue.

All mutation is confined to the selected run. The watchdog owns a separate
lock; the scheduler owns a run lock; each worker owns a trajectory lock.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import multiprocessing
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

from .checkpoint import atomic_json, file_lock, load_json

METHODS = ["Ours", "A-opt", "D-opt", "E-greedy", "E-relax", "UCB", "TS", "Random"]


def utc_stamp(timestamp=None):
    return datetime.fromtimestamp(timestamp or time.time(), timezone.utc).isoformat()


def read_jobs(run):
    value = load_json(Path(run) / "jobs.json")
    return value["jobs"] if isinstance(value, dict) else value


def condition_table(config):
    value = config["conditions"]
    return value if isinstance(value, dict) else {row["id"]: row for row in value}


def condition_for(task, config):
    return task["condition"] if isinstance(task["condition"], dict) else condition_table(config)[task["condition"]]


def physical_jobs(jobs):
    return [job for job in jobs if not job.get("alias_of")]


def ordered_jobs(jobs, config):
    """One replicate wave visits all families/conditions/methods before the next."""
    conditions = list(condition_table(config))
    families = config.get("families", ["gaussian", "bernoulli"])
    def key(job):
        cond = condition_for(job, config)["id"]
        return (job["replicate"], conditions.index(cond), METHODS.index(job["method"]), families.index(job["family"]))
    return sorted(physical_jobs(jobs), key=key)


def checkpoint_progress(run, jobs):
    run = Path(run)
    progress = {}
    for task in physical_jobs(jobs):
        value = load_json(run / "checkpoints" / f"{task['id']}.status.json", {})
        result = load_json(run / "task_results" / f"{task['id']}.json", {})
        progress[task["id"]] = max(int(value.get("N", 0)), int(result.get("N", 0)))
    for task in jobs:
        if task.get("alias_of"):
            progress[task["id"]] = progress.get(task["alias_of"], 0)
    return progress


def profile_rows(run):
    profile = load_json(Path(run) / "profile.json", {})
    return profile, profile.get("rows", profile.get("strata", []))


def profile_curve(task, config, rows):
    cond = condition_for(task, config)
    matched = [row for row in rows if row.get("family") == task["family"]
               and row.get("method") == task["method"]
               and int(row.get("d", -1)) == cond["d"]
               and int(row.get("m", -1)) == cond["m"]
               and ("L" not in row or abs(float(row["L"]) - cond["L"]) < 1e-12)
               and row.get("backend", "cpu") == "cpu"]
    # Refuse to estimate from early-only timing, particularly logistic replay.
    if not matched or max(row["history_N"] for row in matched) < .8 * config["N_run"]:
        return None
    grouped = defaultdict(list)
    for row in matched:
        grouped[float(row["history_N"])].append(float(row["batch_seconds"]))
    return sorted((history, sum(values) / len(values)) for history, values in grouped.items())


def timing_stratum(task, config):
    cond=condition_for(task,config)
    return '|'.join(map(str,[task['family'],task['method'],cond['d'],cond['L'],cond['m'],'cpu']))


def integrate_curve(curve, start, target, batch_size):
    """Piecewise linear batch costs integrated across observed history size."""
    if target <= start:
        return 0.0
    def value(x):
        if x <= curve[0][0]:
            return curve[0][1]
        for (x0, y0), (x1, y1) in zip(curve[:-1], curve[1:]):
            if x <= x1:
                return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        return curve[-1][1]
    knots = [float(start)] + [x for x, _ in curve if start < x < target] + [float(target)]
    return sum((right - left) * (value(left) + value(right)) / 2 for left, right in zip(knots[:-1], knots[1:])) / batch_size


def estimate_eta(jobs, config, progress, rows, workers, target, preview=False, correction=1.0, ratios=None, stage_aware=True):
    preview_target=config.get('preview_N',2000)
    preview_jobs=[job for job in physical_jobs(jobs) if job['replicate']<config.get('preview_replicates',32)]
    if stage_aware and not preview and target>preview_target and any(progress.get(job['id'],0)<preview_target for job in preview_jobs):
        prefix=estimate_eta(jobs,config,progress,rows,workers,preview_target,True,correction,ratios,False)
        after_preview=dict(progress)
        for job in preview_jobs:
            after_preview[job['id']]=max(after_preview.get(job['id'],0),preview_target)
        production=estimate_eta(jobs,config,after_preview,rows,workers,target,False,correction,ratios,False)
        if prefix.get('seconds') is None or production.get('seconds') is None:
            # Every unprofiled unfinished prefix is also unprofiled in its
            # production remainder: do not double count those logical tasks.
            result=dict(production if production.get('seconds') is None else prefix)
            result['preview_barrier']={'preview_seconds':prefix.get('seconds'),'production_seconds':production.get('seconds')}
            return result
        result=dict(production)
        seconds=prefix['seconds']+production['seconds']
        finish=time.time()+seconds
        result.update(seconds=seconds,label='measured pilot queue makespan including preview barrier',
                      finish_utc=utc_stamp(finish),finish_America_New_York=datetime.fromtimestamp(finish,ZoneInfo('America/New_York')).isoformat(),
                      preview_barrier={'preview_seconds':prefix['seconds'],'production_seconds':production['seconds']})
        if prefix.get('empirical_range_seconds') and production.get('empirical_range_seconds'):
            result['empirical_range_seconds']=[left+right for left,right in zip(prefix['empirical_range_seconds'],production['empirical_range_seconds'])]
            result['range_qualification']+='; preview and production phase bounds added'
        else:
            result.pop('empirical_range_seconds',None)
            if not prefix.get('empirical_range_seconds'):
                result['range_qualification']=prefix['range_qualification']
        return result
    import heapq
    costs, lower_costs, upper_costs, unprofiled, curves, integrated = [], [], [], [], {}, {}
    used_corrections, range_unavailable={},set()
    for task in physical_jobs(jobs):
        if preview and task["replicate"] >= config.get("preview_replicates", 32):
            continue
        if progress.get(task["id"], 0) >= target:
            continue
        key=(task['family'], str(task['condition']), task['method'])
        if key not in curves:
            curves[key]=profile_curve(task, config, rows)
        curve=curves[key]
        if curve is None:
            unprofiled.append(task["id"])
            continue
        costkey=(key,progress.get(task['id'],0),target)
        if costkey not in integrated:
            integrated[costkey]=integrate_curve(curve, progress.get(task["id"], 0), target, condition_for(task, config)["m"])
        cost=integrated[costkey]
        key=timing_stratum(task,config)
        observed=ratios.get(key,[]) if isinstance(ratios,dict) else []
        observed=sorted(observed)
        local_correction=observed[len(observed)//2] if observed else correction
        used_corrections[key]={'median_ratio':local_correction,'completed_tasks':len(observed)}
        costs.append(cost * local_correction)
        if len(observed)>=10:
            lower_costs.append(cost*observed[int(.1*(len(observed)-1))])
            upper_costs.append(cost*observed[int(.9*(len(observed)-1))])
        else:
            range_unavailable.add(key)
    if unprofiled:
        return {"seconds": None, "label": "calibrating", "unprofiled_tasks": len(unprofiled), "unprofiled_examples": unprofiled[:8]}
    def makespan(values):
        loads = [0.0] * max(1, workers)
        for cost in values:
            ready = heapq.heappop(loads)
            heapq.heappush(loads, ready + cost)
        return max(loads)
    seconds = makespan(costs)
    finish = time.time() + seconds
    result = {"seconds": seconds, "label": "measured pilot queue makespan", "unprofiled_tasks": 0,
              "finish_utc": utc_stamp(finish), "finish_America_New_York": datetime.fromtimestamp(finish, ZoneInfo("America/New_York")).isoformat(),
              "stratified_pilot_to_live_cost_ratios":used_corrections}
    if costs and not range_unavailable:
        result["empirical_range_seconds"] = [makespan(lower_costs),makespan(upper_costs)]
        result["range_qualification"] = "10th–90th percentile task/pilot ratios within family/method/d/L/m/backend; descriptive, not a confidence interval"
    else:
        result["range_qualification"] = f"Empirical range calibrating: {len(range_unavailable)} remaining strata have fewer than 10 completed tasks"
    return result


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True


def current_workers(run, active_ids=None):
    workers = []
    for path in (Path(run) / "workers").glob("*.json"):
        record = load_json(path, {})
        if not record.get("pid") or not pid_alive(record["pid"]):
            continue
        if active_ids is not None and record.get("task_id") not in active_ids:
            continue
        record['process_alive']=True
        updated=record.get('updated_at')
        record['heartbeat_age_seconds']=max(0.,time.time()-float(updated)) if isinstance(updated,(int,float)) else None
        record['heartbeat_stale']=record['heartbeat_age_seconds'] is None or record['heartbeat_age_seconds']>90
        record["rss_bytes"] = None
        try:
            fields = Path(f"/proc/{record['pid']}/statm").read_text().split()
            record["rss_bytes"] = int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            pass
        workers.append(record)
    return workers


def format_eta(value):
    if value.get("seconds") is None:
        return f"calibrating ({value.get('unprofiled_tasks', '?')} unprofiled tasks)"
    uncertainty=value.get('empirical_range_seconds')
    range_text=f"; empirical range {uncertainty[0]/3600:.2f}–{uncertainty[1]/3600:.2f} h" if uncertainty else '; empirical range calibrating'
    return f"{value['seconds']/3600:.2f} h{range_text}; finish {value['finish_utc']} / {value['finish_America_New_York']}"


def status_text(value):
    logical, physical = value["logical"], value["physical"]
    workers = value.get("workers", [])
    heartbeat_ages=[worker['heartbeat_age_seconds'] for worker in workers if worker.get('heartbeat_age_seconds') is not None]
    heartbeat_text=f"oldest {max(heartbeat_ages):.1f}s" if heartbeat_ages else 'no active worker heartbeat'
    lines = [f"{value['state']} | stage={value['stage']} | heartbeat={value['heartbeat']}",
             f"Logical trajectories completed/expected={logical['completed']}/{logical['expected']} running={logical['running']} pending={logical['pending']} failed={logical['failed']}",
             f"Physical trajectories completed/expected={physical['completed']}/{physical['expected']} running={physical['running']} pending={physical['pending']} failed={physical['failed']}",
             f"Preview completed/expected={value['preview']['completed']}/{value['preview']['expected']}",
             f"Responses physical={physical['responses']:,}/{physical['expected_responses']:,}; logical={logical['responses']:,}/{logical['expected_responses']:,}; physical batches={physical['batches']:,}/{physical['expected_batches']:,}",
             f"Elapsed={value['elapsed_seconds']/3600:.3f} h; recent physical throughput={value['recent_responses_per_second']:.3f} responses/s ({value['recent_batches_per_second']:.3f} batches/s)",
             f"ETA preview: {format_eta(value['eta_preview'])}", f"ETA full: {format_eta(value['eta_full'])}",
             f"CPU workers={len(workers)}/{value['worker_capacity']}; GPU workers=0; worker RSS={sum(w.get('rss_bytes') or 0 for w in workers)/2**30:.3f} GiB",
             f"Worker heartbeat: {heartbeat_text}; stale (>90s)={sum(bool(worker.get('heartbeat_stale')) for worker in workers)}; a slow batch remains counted while its process is alive",
             f"Supervisor pid={value['supervisor_pid']} host={value['hostname']} session={value.get('session')}",
             f"Run directory: {value['run_dir']}"]
    return "\n".join(lines) + "\n"


def write_status(run, jobs, config, stage, state, active, failures, started, workers, previous=None, ratios=None, progress_cache=None):
    run = Path(run)
    progress = dict(progress_cache) if progress_cache is not None else checkpoint_progress(run, jobs)
    live = current_workers(run, active)
    # Heartbeats reflect safe completed batch boundaries, even between durable saves.
    for worker in live:
        task_id = worker.get("task_id")
        progress[task_id] = max(progress.get(task_id, 0), int(worker.get("N", 0)))
    for job in jobs:
        if job.get("alias_of"):
            progress[job["id"]] = progress.get(job["alias_of"], 0)
    target = config["N_run"]
    failed = {key for key, val in failures.items() if val.get("attempts", 0) >= 3}
    def counts(queue):
        complete = sum(progress.get(job["id"], 0) >= target for job in queue)
        running = sum(job.get("alias_of", job["id"]) in active and progress.get(job['id'],0)<target for job in queue)
        fail = sum(job.get("alias_of", job["id"]) in failed and progress.get(job["id"], 0) < target for job in queue)
        return {"expected": len(queue), "completed": complete, "running": running, "failed": fail, "pending": len(queue)-complete-running-fail,
                "responses": sum(min(progress.get(job["id"], 0), target) for job in queue), "expected_responses": len(queue)*target,
                "batches": sum(min(progress.get(job["id"], 0), target)//condition_for(job, config)["m"] for job in queue),
                "expected_batches":sum(target//condition_for(job,config)['m'] for job in queue)}
    physical, logical = counts(physical_jobs(jobs)), counts(jobs)
    preview_jobs = [job for job in jobs if job["replicate"] < config.get("preview_replicates", 32)]
    _, rows = profile_rows(run)
    ratios = ratios or {}
    correction = 1.0
    now = time.time()
    dt = now - previous["timestamp"] if previous else 0
    throughput = max(0, physical["responses"]-previous["physical"]["responses"])/dt if dt > 0 else 0
    batch_throughput = max(0, physical["batches"]-previous["physical"]["batches"])/dt if dt > 0 else 0
    manifest = load_json(run / "manifest.json", {})
    value = {"state": state, "stage": stage, "run_dir": str(run.resolve()), "timestamp": now, "heartbeat": utc_stamp(now),
             "elapsed_seconds": now-started, "supervisor_pid": os.getpid(), "hostname": socket.gethostname(),
             "session": os.environ.get("OFAT_TMUX_SESSION"), "worker_capacity": workers, "workers": live, "logical": logical, "physical": physical,
             "preview": {"expected": len(preview_jobs), "completed": sum(progress.get(job["id"], 0) >= config.get("preview_N",2000) for job in preview_jobs)},
             "recent_responses_per_second": throughput, "recent_batches_per_second": batch_throughput,
             "eta_preview": estimate_eta(jobs, config, progress, rows, workers, config.get("preview_N",2000), True, correction, ratios),
             "eta_full": estimate_eta(jobs, config, progress, rows, workers, target, False, correction, ratios),
             "code_hash": manifest.get("code_hash"), "config_hash": manifest.get("config_hash"),
             "shared_work": "Gaussian Ours/A-opt/D-opt/E-greedy aliases share an exact response-independent design and the same response stream; aliases are not independent realizations."}
    atomic_json(run / "status.json", value)
    import json
    with open(run / "progress.jsonl", "a") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
    text = status_text(value)
    with open(run / "logs" / "status.log", "a") as stream:
        stream.write(text + "\n")
    print(text, flush=True)
    return value


def worker_init():
    # BLAS environment is set before spawned interpreter imports numerical code.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Linux parent death requests a safe batch-boundary stop if scheduler dies.
    try:
        import ctypes
        ctypes.CDLL(None).prctl(1, signal.SIGTERM)
    except (ImportError, AttributeError, OSError):
        pass


def execute_task(run, task, target):
    from .simulation import run_task
    run = Path(run)
    with file_lock(run / "task_locks" / f"{task['id']}.lock", blocking=True):
        done = checkpoint_progress(run, [task])[task["id"]]
        if done >= target:
            return {"N": done, "elapsed_seconds": 0., "start_N": done}
        started = time.monotonic()
        atomic_json(run / "workers" / f"{os.getpid()}.json", {"pid": os.getpid(), "task_id": task["id"], "N": done, "updated_at": time.time()})
        try:
            result = dict(run_task(str(run), task, target))
            result.setdefault("elapsed_seconds", time.monotonic()-started)
            result["start_N"] = done
            result["updated_at"] = time.time()
            atomic_json(run / "task_results" / f"{task['id']}.json", result)
            return result
        finally:
            (run / "workers" / f"{os.getpid()}.json").unlink(missing_ok=True)


def start_plot(run, phase):
    """Plot failure is visible and retryable; it never corrupts trajectories."""
    logfile = open(Path(run) / "logs" / f"plot-{phase}.log", "a")
    command = [sys.executable, "-m", "src.plotting", "--run-dir", str(run), "--phase", phase]
    try:
        process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1], stdout=logfile, stderr=subprocess.STDOUT)
    finally:
        logfile.close()
    return process


def run_supervisor(run, workers=None):
    run = Path(run).resolve()
    for directory in ("logs", "workers", "task_results", "task_locks", "checkpoints"):
        (run / directory).mkdir(parents=True, exist_ok=True)
    from .common import validate_run
    validate_run(run)
    config = load_json(run / "config.json")
    jobs = read_jobs(run)
    profile, rows = profile_rows(run)
    workers = workers or profile.get("workers", profile.get("selected", {}).get("workers", config.get('workers',1)))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    with file_lock(run / ".supervisor.lock"):
        launch = load_json(run / "launch.json", {})
        launch.update({"supervisor_pid": os.getpid(), "hostname": socket.gethostname(), "python": sys.executable, "workers": workers,
                       "session": os.environ.get("OFAT_TMUX_SESSION"), "started_at": utc_stamp()})
        atomic_json(run / "launch.json", launch)
        started = load_json(run / "status.json", {}).get("timestamp", time.time())
        old = load_json(run / "status.json", {})
        started -= old.get("elapsed_seconds", 0)
        failures = load_json(run / "failures.json", {})
        ratios = load_json(run / "live_timing_ratios.json", {})
        if not isinstance(ratios,dict):
            ratios={}
        queue = ordered_jobs(jobs, config)
        active, previous, plot_process = {}, None, None
        last_plot=0.
        first_preview_generated=(run/'figures'/'preview'/'figure_manifest.json').exists()
        progress_cache=checkpoint_progress(run,jobs)
        stopping = (run / "STOP_REQUESTED").exists()
        def handle_stop(signum, _frame):
            nonlocal stopping
            stopping = True
            (run / "STOP_REQUESTED").touch()
        signal.signal(signal.SIGTERM, handle_stop)
        signal.signal(signal.SIGINT, handle_stop)
        ctx = multiprocessing.get_context("spawn")
        stage = "preview"
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=worker_init) as pool:
            for stage, target, stage_jobs in (("preview", config.get("preview_N",2000), [job for job in queue if job["replicate"] < config.get("preview_replicates",32)]), ("production", config["N_run"], queue)):
                progress = progress_cache
                pending = [job for job in stage_jobs if progress[job["id"]] < target and failures.get(job["id"],{}).get("attempts",0)<3]
                last_status = 0.
                while pending or active:
                    stopping = stopping or (run / "STOP_REQUESTED").exists()
                    while pending and len(active) < workers and not stopping:
                        task = pending.pop(0)
                        future = pool.submit(execute_task, str(run), task, target)
                        active[future] = task
                    if not active and stopping:
                        break
                    done, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED) if active else (set(),set())
                    for future in done:
                        task = active.pop(future)
                        try:
                            result = future.result()
                            progress_cache[task['id']]=int(result.get('N',0))
                            for alias in jobs:
                                if alias.get('alias_of')==task['id']:
                                    progress_cache[alias['id']]=int(result.get('N',0))
                            if int(result.get("N",0)) < target and not stopping and not (run/"STOP_REQUESTED").exists():
                                raise RuntimeError(f"Task returned N={result.get('N')} before target {target}")
                            curve = profile_curve(task, config, rows)
                            if curve and result.get("elapsed_seconds",0)>0 and result["N"]>result["start_N"]:
                                predicted = integrate_curve(curve,result["start_N"],result["N"],condition_for(task,config)["m"])
                                if predicted>0:
                                    key=timing_stratum(task,config)
                                    ratios.setdefault(key,[]).append(result["elapsed_seconds"]/predicted)
                                    ratios[key]=ratios[key][-128:]
                                    atomic_json(run / "live_timing_ratios.json",ratios)
                            timing_path=run/"timing_profile.csv"
                            write_header=not timing_path.exists()
                            with timing_path.open('a',newline='') as stream:
                                writer=csv.DictWriter(stream,fieldnames=['timestamp','family','condition','method','d','L','m','backend','start_N','end_N','elapsed_seconds','batch_seconds'])
                                if write_header:
                                    writer.writeheader()
                                cond=condition_for(task,config)
                                batches=(int(result['N'])-int(result['start_N']))/cond['m']
                                writer.writerow({'timestamp':utc_stamp(),'family':task['family'],'condition':cond['id'],'method':task['method'],'d':cond['d'],'L':cond['L'],'m':cond['m'],'backend':'cpu','start_N':result['start_N'],'end_N':result['N'],'elapsed_seconds':result.get('elapsed_seconds',0),'batch_seconds':result.get('elapsed_seconds',0)/batches if batches else 0})
                            failures.pop(task["id"],None)
                        except Exception as exc:
                            item = failures.setdefault(task["id"], {"attempts":0})
                            item.update({"attempts":item["attempts"]+1, "last_error":repr(exc), "traceback":traceback.format_exc(), "timestamp":utc_stamp(), "stage":stage})
                            with open(run/"logs"/"failures.log","a") as stream:
                                stream.write(item["timestamp"]+" "+task["id"]+"\n"+item["traceback"]+"\n")
                            if item["attempts"]<3 and not stopping:
                                pending.append(task)
                        atomic_json(run / "failures.json", failures)
                    now=time.monotonic()
                    if now-last_status>=config.get("status_interval_seconds",30) or not pending and not active:
                        previous=write_status(run,jobs,config,stage,"STOPPING" if stopping else "RUNNING",{task["id"] for task in active.values()},failures,started,workers,previous,ratios,progress_cache)
                        last_status=now
                    if not stopping and (plot_process is None or plot_process.poll() is not None):
                        eligible=any(progress_cache.get(job['id'],0)>=config.get('preview_N',2000) for job in stage_jobs if job['replicate']<config.get('preview_replicates',32))
                        if eligible and (not first_preview_generated or now-last_plot>=config.get('plot_interval_seconds',3600)):
                            plot_process=start_plot(run,'preview')
                            first_preview_generated=True
                            last_plot=now
                if stopping:
                    break
                if stage=="preview":
                    if plot_process is not None:
                        plot_process.wait()
                    plot_process=start_plot(run,"preview")
                    last_plot=time.monotonic()
            # Exiting the pool waits for all workers to checkpoint and return.
        if plot_process is not None:
            code=plot_process.wait()
            atomic_json(run/"preview_plot_result.json",{"returncode":code,"finished_at":utc_stamp()})
        if stopping:
            state="STOPPED"
        else:
            progress=checkpoint_progress(run,jobs)
            state="COMPLETE" if all(progress[job["id"]]>=config["N_run"] for job in jobs) else "FINISHED_WITH_FAILURES"
            plot=start_plot(run,"final")
            # Keep heartbeat current while full aggregation is running.
            while plot.poll() is None:
                previous=write_status(run,jobs,config,"final_plotting", "FINALIZING",set(),failures,started,workers,previous,ratios,progress_cache)
                try:
                    plot.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
            atomic_json(run/"final_plot_result.json",{"returncode":plot.returncode,"finished_at":utc_stamp()})
            if plot.returncode:
                state="PLOTTING_FAILED" if state=="COMPLETE" else state
        write_status(run,jobs,config,stage,state,set(),failures,started,workers,previous,ratios,progress_cache)
        return 0 if state in ("COMPLETE","STOPPED","FINISHED_WITH_FAILURES") else 2


def watchdog(run, workers=None):
    run=Path(run).resolve()
    (run/"logs").mkdir(exist_ok=True)
    with file_lock(run/".watchdog.lock"):
        atomic_json(run/"watchdog.json",{"pid":os.getpid(),"hostname":socket.gethostname(),"started_at":utc_stamp(),"session":os.environ.get("OFAT_TMUX_SESSION")})
        def stop(_signum,_frame):
            (run/"STOP_REQUESTED").touch()
        signal.signal(signal.SIGTERM,stop)
        signal.signal(signal.SIGINT,stop)
        retries=0
        while not (run/"STOP_REQUESTED").exists():
            cmd=[sys.executable,"-m","src.supervisor","run","--run-dir",str(run)]
            if workers:
                cmd += ["--workers",str(workers)]
            code=subprocess.call(cmd,cwd=Path(__file__).resolve().parents[1])
            status=load_json(run/"status.json",{})
            if code==0 or status.get("state") in ("COMPLETE","STOPPED","FINISHED_WITH_FAILURES"):
                return code
            retries+=1
            atomic_json(run/"watchdog_last_restart.json",{"returncode":code,"restart":retries,"timestamp":utc_stamp()})
            if retries>=20:
                raise RuntimeError("Supervisor crashed 20 times; inspect logs before resuming")
            time.sleep(min(60,5*retries))
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["run","watch","status","stop"])
    parser.add_argument("--run-dir",required=True)
    parser.add_argument("--workers",type=int)
    args=parser.parse_args()
    if args.command=="run":
        return run_supervisor(args.run_dir,args.workers)
    if args.command=="watch":
        return watchdog(args.run_dir,args.workers)
    if args.command=="stop":
        (Path(args.run_dir)/"STOP_REQUESTED").touch()
        print("Graceful stop requested; workers checkpoint at their next completed batch boundary.")
        return 0
    status=load_json(Path(args.run_dir)/"status.json")
    if status is None:
        print("No status yet: supervisor has not produced its first heartbeat.")
        return 1
    print(status_text(status),end="")
    age=time.time()-status["timestamp"]
    print(f"Heartbeat age: {age:.1f}s; supervisor process alive: {pid_alive(status['supervisor_pid'])}")
    print("Current live worker PIDs: "+", ".join(str(worker["pid"]) for worker in current_workers(args.run_dir)))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
