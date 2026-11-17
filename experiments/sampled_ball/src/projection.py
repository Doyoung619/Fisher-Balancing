"""Transparent preflight wall-clock projection from measured component timings."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np

CONDITIONS = [dict(name='central', d=10, m=10), dict(name='dim5', d=5, m=10),
              dict(name='dim20', d=20, m=10), dict(name='radius05', d=10, m=10),
              dict(name='radius2', d=10, m=10), dict(name='batch5', d=10, m=5),
              dict(name='batch20', d=10, m=20)]
METHODS = ('FB','AOpt','DOpt','EGreedy','UCB','TS','Random')
INFO = METHODS[:4]


def project(benchmark):
    rows = benchmark['rows']

    def subset(component, family, device, d, m, **kw):
        base = [r for r in rows if r['component'] == component and r['family'] == family
                and r['device'].startswith(device) and all(r.get(k) == v for k,v in kw.items())]
        if not base:
            return []
        # d=5 and Gaussian d<20 have no direct measurements: use the next
        # larger measured dimension conservatively, without dimension scaling.
        nearest = min(base, key=lambda r: abs(r['d']-d)*100 + abs(r['m']-m))
        return [r for r in base if r['d'] == nearest['d'] and r['m'] == nearest['m']]

    def acquisition(family, method, device, c, R, K):
        rr = subset('acquisition', family, device, c['d'], c['m'], method=method, K=K)
        reference_R = rr[0]['R']
        # A Gaussian information trajectory has R=1 regardless of the number
        # of response realizations; this factor must never scale with R.
        scale = 1. if family == 'gaussian' and method in INFO else max(.5, R/reference_R)
        return float(np.mean([r['seconds'] for r in rr])) * scale

    def refit(family, device, c, R, N):
        rr = subset('refit', family, device, c['d'], c['m'])
        measured_R = sorted(set(r['R'] for r in rr))
        # Integrate the measured complete-history cost over N. This captures
        # the cumulative logistic refit cost instead of extrapolating only
        # the cheap initial batches.
        refit_by_R = []
        grid = np.linspace(0, N, 201)
        for rcount in measured_R:
            data = sorted([r for r in rr if r['R'] == rcount], key=lambda r:r['N'])
            values = np.interp(grid, [0]+[r['N'] for r in data],
                               [data[0]['seconds']]+[r['seconds'] for r in data])
            refit_by_R.append(float(np.trapezoid(values, grid)/N))
        if R < measured_R[0]:
            return refit_by_R[0] * max(.5, R/measured_R[0])
        if R > measured_R[-1]:
            return refit_by_R[-1] * R/measured_R[-1]
        return float(np.interp(R, measured_R, refit_by_R))

    def cloud(family, device, c, K):
        rr = subset('cloud', family, device, c['d'], c['m'], K=K)
        return float(np.mean([r['seconds'] for r in rr])) if rr else .001

    output = []
    for preset, N, K, fbcentral, fbother, compcentral, compother in [
            ('preferred', 10000,4096,256,128,64,32),
            ('fast',10000,2048,128,64,32,16),
            ('fast_N5000',5000,2048,128,64,32,16)]:
        jobs = []
        for family in ('gaussian','bernoulli'):
            for c in CONDITIONS:
                central = c['name'] == 'central'
                for method in METHODS:
                    R = (fbcentral if central else fbother) if method=='FB' else (compcentral if central else compother)
                    job = dict(id=f"{family}_{c['name']}_{method}", family=family,
                               condition=c['name'], method=method, R=R, components={})
                    for dev in ('cpu','cuda'):
                        acq = acquisition(family,method,dev,c,R,K)
                        fit = refit(family,dev,c,R,N)
                        cl = 0 if method=='Random' else cloud(family,dev,c,K)
                        # Includes response/statistic updates, small records,
                        # periodic recovery writes, and worker startup.
                        job[dev] = (acq + fit + cl + .0005)*N/c['m'] + 2.
                        job['components'][dev] = dict(acquisition_seconds=acq*N/c['m'],
                                                     refit_seconds=fit*N/c['m'],
                                                     cloud_seconds=cl*N/c['m'],
                                                     overhead_seconds=.0005*N/c['m']+2.)
                    jobs.append(job)
        variants=[]
        for assignment in ('all_bernoulli_gpu','bernoulli_light_cpu','bernoulli_only_FB_E_gpu'):
            cpus=[0.]*28;gpus=[0.]*4
            assigned=[]
            for job in sorted(jobs,key=lambda j:max(j['cpu'],j['cuda']),reverse=True):
                use_gpu=job['family']=='bernoulli'
                if assignment=='bernoulli_light_cpu' and job['method'] in ('UCB','TS','Random'):
                    use_gpu=False
                if assignment=='bernoulli_only_FB_E_gpu' and job['method'] not in ('FB','EGreedy'):
                    use_gpu=False
                load = gpus if use_gpu else cpus
                dev = 'cuda' if use_gpu else 'cpu'
                worker=int(np.argmin(load));load[worker]+=job[dev]
                assigned.append(dict(job,assigned_device=dev,chosen_device=dev,
                                     estimated_seconds=job[dev]))
            raw = max(max(cpus),max(gpus))
            variants.append(dict(assignment=assignment,component_projection_seconds=raw,
                                 cpu_makespan_seconds=max(cpus),gpu_makespan_seconds=max(gpus),
                                 conservative_seconds=raw*2.5+180,
                                 caution_seconds=raw*4+300,
                                 cpu_workers=28,gpu_workers=4,jobs=assigned))
        output.append(dict(preset=preset,N_run=N,K=K,
                           best_assignment=min(variants,key=lambda v:v['component_projection_seconds'])['assignment'],
                           variants=variants))
    return dict(status='estimated_not_guaranteed',benchmark_elapsed_seconds=benchmark['elapsed_seconds'],
                assumptions=['Piecewise-linear integral of measured cumulative-history refit time over N.',
                             'Gaussian information acquisition R=1 irrespective of response replicate count.',
                             'Unmeasured smaller dimensions use the measured larger dimension without a speed discount.',
                             'Replicate interpolation for refits; proportional replicate scaling for acquisition.',
                             'One single-thread worker per physical CPU core: 28 CPU workers plus four GPU hosts.',
                             'Conservative estimate = 2.5 * component makespan + 180 seconds for contention and analysis.',
                             'Caution estimate = 4 * component makespan + 300 seconds; not a formal upper bound.'],
                projections=output)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--benchmark',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();result=project(json.loads(Path(args.benchmark).read_text()))
    Path(args.output).write_text(json.dumps(result,indent=2))
    preferred=result['projections'][0]
    chosen=next(v for v in preferred['variants'] if v['assignment']==preferred['best_assignment'])
    Path(args.output).with_name('preferred_job_projection.json').write_text(json.dumps(
        dict(preset='preferred',N_run=10000,K=4096,assignment=chosen['assignment'],
             component_projection_seconds=chosen['component_projection_seconds'],
             conservative_seconds=chosen['conservative_seconds'],
             caution_seconds=chosen['caution_seconds'],
             jobs={j['id']:j for j in chosen['jobs']}),indent=2))
    for row in result['projections']:
        print(row['preset'],row['best_assignment'])
        for v in row['variants']:
            print(v['assignment'], 'component',round(v['component_projection_seconds'],1),
                  'conservative_minutes',round(v['conservative_seconds']/60,2),
                  'caution_minutes',round(v['caution_seconds']/60,2))
