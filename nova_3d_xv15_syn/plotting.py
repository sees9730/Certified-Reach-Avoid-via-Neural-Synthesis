"""Sampled V/GV and policy slices for inspection; plots never decide SAT."""
import json
import os
import tempfile
from pathlib import Path
import numpy as np
import torch
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'nova_3d_xv15_syn_matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Rectangle, Patch

try:
    from .model import Model
    from .interval import I
    from .verify import fingerprint
except ImportError:
    from model import Model
    from interval import I
    from verify import fingerprint


def plot(output):
    output=Path(output);artifact=json.loads((output/'candidate.json').read_text())
    p=artifact['problem'];model=Model(p,False);model.restore(artifact['model'])
    report=json.loads((output/'verification.json').read_text()) if (output/'verification.json').exists() else {}
    if report.get('candidate_sha256')!=fingerprint(artifact): report={}
    labels=['Airspeed [m/s]','Flight-path angle [deg]','Rotor tilt [deg]']
    domain=np.array(p['domain']);initial=np.mean(p['initial'],axis=1);goal=np.mean(p['goal'],axis=1)
    slices=[(0,1,goal),(0,1,initial),(0,2,goal),(0,2,initial),(1,2,goal)]
    with PdfPages(output/'slices.pdf') as pdf:
        for page,(i,j,fixed) in enumerate(slices):
            k=next(k for k in range(3) if k not in (i,j))
            a,b=np.meshgrid(np.linspace(*domain[i],220),np.linspace(*domain[j],220))
            x=np.tile(fixed,(a.size,1));x[:,i]=a.ravel();x[:,j]=b.ravel()
            with torch.no_grad():
                points=I(x);v,g=model.values(points)
                values=[v.lo.numpy().reshape(a.shape),g.lo.numpy().reshape(a.shape)]
                controls=[u.lo.numpy().reshape(a.shape) for u in model.policy(points)]
            fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
            for ax,z,title in zip(axes,values,['V','GV']):
                extent=float(np.abs(z).max()) or 1.
                norm=SymLogNorm(linthresh=.01,vmin=float(z.min()) if title=='V' else -extent,
                                vmax=float(z.max()) if title=='V' else extent)
                field=ax.pcolormesh(a,b,z,cmap='viridis' if title=='V' else 'coolwarm',norm=norm,shading='auto',rasterized=True)
                fig.colorbar(field,ax=ax,label=title)
                if values[0].min()<p['beta_ra']<values[0].max():
                    c=ax.contour(a,b,values[0],levels=[p['beta_ra']],colors='black',linewidths=1.5)
                    ax.clabel(c,fmt={p['beta_ra']:f"V={p['beta_ra']:g}"})
                level=1 if title=='V' else -p['epsilon']
                if z.min()<level<z.max():
                    c=ax.contour(a,b,z,levels=[level],colors='white',linewidths=1)
                    ax.clabel(c,fmt={level:f'{title}={level:g}'})
                for name,boxes,color in [('Initial',[p['initial']],'orange'),('Goal',[p['goal']],'lime'),('Unsafe',p['unsafe'],'red')]:
                    for box in boxes:
                        box=np.array(box)
                        if box[k,0]<=fixed[k]<=box[k,1]:
                            ax.add_patch(Rectangle((box[i,0],box[j,0]),np.diff(box[i])[0],np.diff(box[j])[0],
                                                   fill=False,edgecolor=color,linewidth=2))
                ax.set(xlabel=labels[i],ylabel=labels[j],xlim=domain[i],ylim=domain[j],title=title)
            axes[0].legend(handles=[Patch(fill=False,edgecolor=c,label=n) for n,c in [('Initial','orange'),('Goal','lime'),('Unsafe','red')]],loc='upper left')
            fig.suptitle(f"{report.get('status','Unverified candidate')} — slice at {labels[k]} = {fixed[k]:g}\n"
                         'Region boxes appear only when they intersect this slice; colors are sampled values.')
            pdf.savefig(fig);fig.savefig(output/f'slice_{page+1}.png',dpi=140);plt.close(fig)
            fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
            controls[0]*=p['mass']*p['gravity']
            for ax,z,title in zip(axes,controls,['Thrust [N]','Angle of attack [deg]','Tilt rate [deg/s]']):
                field=ax.pcolormesh(a,b,z,shading='auto',rasterized=True);fig.colorbar(field,ax=ax)
                ax.set(xlabel=labels[i],ylabel=labels[j],title=title)
            fig.suptitle(f'Learned control — {labels[k]} = {fixed[k]:g}')
            pdf.savefig(fig);plt.close(fig)
    print(f'Plots: {output / "slices.pdf"}',flush=True)
