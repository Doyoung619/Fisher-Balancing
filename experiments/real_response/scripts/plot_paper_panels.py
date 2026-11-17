"""Portable reproduction: only Python stdlib, NumPy, Matplotlib are needed."""
from pathlib import Path
import argparse,csv,json,itertools,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COMMON=['fb_opt2','a_opt','d_opt','e_greedy','e_sdp','bait_glm','ucb_glm','ts_glm','random']
EXTRA={'movielens':['b_emcm','lcmd','regvol'],'obd':['entropy','badge','firal']}
LABELS=dict(zip(COMMON,['ours','A-opt','D-opt','E-greedy','E-SDP','BAIT','UCB','TS','Random']))
LABELS.update(b_emcm='B-EMCM',lcmd='LCMD',regvol='RegVol',entropy='Entropy',badge='BADGE',firal='FIRAL')
COLORS=dict(zip(COMMON,['#B55252','#009E73','#CC79A7','#E69F00','#6A3D9A','#56B4E9','#8C564B','#7F7F7F','#222222']))
COLORS.update(b_emcm='#A6761D',lcmd='#739B25',regvol='#3B8098',entropy='#159DAB',badge='#BCBD22',firal='#D89C36')
STYLES=dict(zip(COMMON,['-','--','-.',':','--','-.',':','--',':']))
STYLES.update(b_emcm='--',lcmd='-.',regvol=':',entropy='--',badge='-.',firal=':')
FAMILIES={'movielens':('gaussian','MovieLens',400),'obd':('bernoulli','OBD',4000)}
PANELS=[('mean_lambda_min_reference',r'(a) $\lambda_{\min}(V_{\mathrm{ref},t})$','Minimum eigenvalue'),
        ('mean_lambda_min_plugin',r'(b) $\lambda_{\min}(\widehat{V}_t)$','Minimum eigenvalue'),
        ('R2_ref',r'(c) $R_2(t)$','Root total risk'),('Rdir_ref',r'(d) $R_{\mathrm{dir}}(t)$','Root directional risk')]
PLOT_COLUMNS=['dataset','d','batch_size','method','display_name','round','n_additional','n_total','actual_seed_count']+[p[0] for p in PANELS]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--condition',help='Optional single figure stem, e.g. gaussian_movielens_m20_d05_2x2')
    parser.add_argument('--formats',default='png,pdf,svg')
    args=parser.parse_args();args.output.mkdir(exist_ok=True,parents=True)
    with args.data.open() as f:rows=list(csv.DictReader(f))
    for row in rows:
        for key in ['d','batch_size','round','n_additional','n_total','actual_seed_count']:row[key]=int(float(row[key]))
        for key,_,_ in PANELS:row[key]=float(row[key])
        row['display_name']=LABELS[row['method']]
    assert all(r['actual_seed_count']==10 for r in rows)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':14,'axes.labelsize':11,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    entries=[];plotted=[]
    for ds,m,d in itertools.product(FAMILIES,[20,40,80],[5,10,20]):
        family,title,budget=FAMILIES[ds];stem=f'{family}_{ds}_m{m:02d}_d{d:02d}_2x2'
        if args.condition and args.condition!=stem:continue
        group=[r for r in rows if r['dataset']==ds and r['d']==d and r['batch_size']==m]
        methods=COMMON+EXTRA[ds];assert set(r['method'] for r in group)==set(methods)
        fig,axes=plt.subplots(2,2,figsize=(11.6,8.4),sharex=True)
        for method in methods:
            data=sorted([r for r in group if r['method']==method],key=lambda r:r['n_additional'])
            x=[r['n_additional'] for r in data];assert x==list(range(0,budget+1,m))
            plotted.extend(data)
            for ax,(key,_,_) in zip(axes.flat,PANELS):
                y=np.array([r[key] for r in data]);assert np.isfinite(y).all() and (y>=0).all()
                ax.plot(x,y,label=LABELS[method],color=COLORS[method],linestyle=STYLES[method],
                    linewidth=2.6 if method=='fb_opt2' else 1.45,
                    marker='D' if method=='fb_opt2' else None,markersize=3,
                    markevery=max(1,len(data)//8),zorder=5 if method=='fb_opt2' else 2)
        for ax,(_,label,ylabel) in zip(axes.flat,PANELS):
            ax.set_title(label,pad=10);ax.set_ylabel(ylabel);ax.set_xlim(0,budget)
            ax.grid(alpha=.2,linewidth=.6);ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,4),useMathText=True)
        low=min(ax.get_ylim()[0] for ax in axes[0]);high=max(ax.get_ylim()[1] for ax in axes[0])
        for ax in axes[0]:ax.set_ylim(low,high)
        for ax in axes[1]:ax.set_xlabel(r'Additional observed responses ($mt$)')
        fig.suptitle(f'{title} | {family.capitalize()} | m = {m}, d = {d}',fontsize=16,y=.975)
        fig.text(.5,.927,'10 paired seeds; fixed full-pool reference; initial responses = 80',ha='center',fontsize=10,color='#444444')
        handles,labels=axes[0,0].get_legend_handles_labels();assert labels[0]=='ours'
        fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.012),ncol=6,frameon=False,
                   fontsize=9.5,columnspacing=1.5,handlelength=2.8)
        fig.tight_layout(rect=[.015,.10,.99,.91],h_pad=2.,w_pad=2.)
        names=[]
        for ext in args.formats.split(','):
            name=stem+'.'+ext;fig.savefig(args.output/name,dpi=200,facecolor='white');names.append(name)
        plt.close(fig)
        with (args.output/(stem+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=PLOT_COLUMNS,extrasaction='ignore');writer.writeheader();writer.writerows(group)
        entries.append(dict(dataset=ds,g=family,m=m,d=d,methods=methods,legend_order=labels,
            ours_color=COLORS['fb_opt2'],seed_count=10,additional_budget=budget,files=names,data_csv=stem+'.csv'))
        print('Created',stem,flush=True)
    with (args.output/'plotted_data.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=PLOT_COLUMNS,extrasaction='ignore');writer.writeheader();writer.writerows(plotted)
    manifest=dict(figure_count=len(entries),image_file_count=sum(len(e['files']) for e in entries),
        source_csv=str(args.data),source_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
        script=str(Path(__file__).resolve()),figures=entries,metric_definitions_unchanged=True,
        omitted_legacy_fb_methods=['fb_sample','fb_range','fb_opt1','fb_opt'],ours_internal_method='fb_opt2')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (args.output/'README.txt').write_text('18 Gaussian/Bernoulli figures; 12 eligible methods per figure.\n'
        'ours = FB-opt2, muted red #B55252, first legend entry.\n'
        'Each stem has PNG/PDF/SVG plus the exact matching CSV.\n'
        'Top panels: 10-seed mean minimum eigenvalue, ridge and warm starts included.\n'
        'Bottom panels: sqrt(trace(Q_ref)) and sqrt(lambda_max(Q_ref)), Q=E.T@E/10, no centering.\n'
        'Linear axes, recorded batch checkpoints, no smoothing or confidence bands.\n'
        'Source and figure mapping are in manifest.json.\n')
    print('PLOTS_COMPLETE',len(entries),flush=True)


if __name__=='__main__':main()
