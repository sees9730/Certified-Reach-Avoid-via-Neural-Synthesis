"""Full-domain certificate slices and noisy trajectories; samples never decide SAT."""
import json
import os
import tempfile
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'nova_4d_double_integrator_syn_mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

try:
    from .runtime import Controller
    from .check import fingerprint
    from .animation import animate
except ImportError:
    from runtime import Controller
    from check import fingerprint
    from animation import animate


def regions(ax,p,axes,fixed):
    for name,boxes,color in [('initial',[p['initial']],'orange'),('goal',[p['goal']],'limegreen'),('unsafe',p['unsafe'],'crimson')]:
        labeled=False
        for box in boxes:
            if all(box[k][0]<=fixed[k]<=box[k][1] for k in range(4) if k not in axes):
                i,j=axes
                ax.add_patch(Rectangle((box[i][0],box[j][0]),box[i][1]-box[i][0],box[j][1]-box[j][0],
                                       fill=False,edgecolor=color,lw=2,label=name if not labeled else None))
                labeled=True


def plot(output,rollout_options=None):
    output=Path(output);a=json.loads((output/'candidate.json').read_text());p=a['problem'];model=Controller(a)
    report=json.loads((output/'verification.json').read_text()) if (output/'verification.json').exists() else {}
    status=report.get('status','Unverified') if report.get('candidate_sha256')==fingerprint(a) else 'Unverified'
    initial=np.mean(p['initial'],axis=1);domain=np.array(p['domain'])
    slices=[((0,1),initial),((2,3),np.zeros(4)),((2,3),initial)]
    with PdfPages(output/'certificate_slices.pdf') as pdf:
        for page,(axes,fixed) in enumerate(slices):
            i,j=axes;xx,yy=np.meshgrid(np.linspace(*domain[i],201),np.linspace(*domain[j],201))
            x=np.tile(fixed,(xx.size,1));x[:,i]=xx.ravel();x[:,j]=yy.ravel()
            v,g=model.values(x);v=v.reshape(xx.shape);g=g.reshape(xx.shape)
            fig,plots=plt.subplots(1,2,figsize=(12,5),layout='constrained')
            for ax,field,title in zip(plots,[v,g],['V','GV']):
                extent=float(np.abs(field).max()) or 1
                norm=SymLogNorm(.001,vmin=0 if title=='V' else -extent,vmax=extent)
                mesh=ax.pcolormesh(xx,yy,field,norm=norm,cmap='viridis' if title=='V' else 'coolwarm',shading='auto',rasterized=True)
                fig.colorbar(mesh,ax=ax,label=title)
                if v.min()<p['beta_ra']<v.max():
                    c=ax.contour(xx,yy,v,levels=[p['beta_ra']],colors='black');ax.clabel(c,fmt='V=%g')
                threshold=1 if title=='V' else -p['epsilon']
                if field.min()<threshold<field.max():
                    c=ax.contour(xx,yy,field,levels=[threshold],colors='white');ax.clabel(c,fmt=f'{title}=%g')
                regions(ax,p,axes,fixed);ax.set(xlabel=p['state_order'][i],ylabel=p['state_order'][j],title=title,
                                              xlim=domain[i],ylim=domain[j]);ax.legend(loc='upper left')
            hidden=', '.join(f'{p["state_order"][k]}={fixed[k]:g}' for k in range(4) if k not in axes)
            fig.suptitle(f'{status}: slice at {hidden}. Boxes are actual slice intersections; colors are sampled.')
            pdf.savefig(fig);fig.savefig(output/f'slice_{page+1}.png',dpi=130);plt.close(fig)
    runs=animate(output,**(rollout_options or {}))
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    regions(axes[0],p,(0,1),initial)
    for run in runs:
        rows=np.array(run['samples']);label=f"seed {run['seed']}: {run['outcome']}"
        axes[0].plot(rows[:,1],rows[:,2],label=label)
        axes[1].plot(rows[:,0],rows[:,3],alpha=.6);axes[1].plot(rows[:,0],rows[:,4],alpha=.6,ls='--')
        axes[2].plot(rows[:,0],rows[:,5],alpha=.6);axes[2].plot(rows[:,0],rows[:,6],alpha=.6,ls='--')
    axes[0].set(xlim=domain[0],ylim=domain[1],xlabel='px',ylabel='py',title=f'{len(runs)} noisy position trajectories');axes[0].legend(fontsize=7)
    for b in p['unsafe']:
        for i in (2,3):
            if all(b[j]==p['domain'][j] for j in range(4) if j!=i):
                axes[1].axhspan(*b[i],color='crimson',alpha=.15)
    for magnitude in set(p['control_limit']):
        for limit in (-magnitude,magnitude): axes[2].axhline(limit,color='crimson',lw=1)
    axes[1].set(xlabel='time',ylabel='velocity',title='vx solid; vy dashed',ylim=domain[2])
    limit=max(p['control_limit'])*1.05
    axes[2].set(xlabel='time',ylabel='acceleration',title='ux solid; uy dashed',ylim=(-limit,limit))
    fig.suptitle(f'{status} interval proof; these numerical rollouts do not establish the probability bound')
    fig.savefig(output/'rollouts.pdf');fig.savefig(output/'rollouts.png',dpi=130);plt.close(fig)
    print('Plots:',output/'certificate_slices.pdf','and',output/'rollouts.pdf',flush=True)
