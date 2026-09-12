"""Reproducible Euler-Maruyama rollout and a standalone aircraft animation."""
import csv
import json
from pathlib import Path
import numpy as np

try:
    from .model import drift
    from .verify import fingerprint
except ImportError:
    from model import drift
    from verify import fingerprint


class Evaluator:
    """Fast NumPy evaluation of the same saved smooth model, for simulation only."""
    def __init__(self, artifact):
        if artifact.get('controller_type')!='smooth_affine_softmin': raise ValueError('Expected a smooth controller')
        self.p=artifact['problem'];self.m={k:np.asarray(v) for k,v in artifact['model'].items()}

    def __call__(self, x):
        m=self.m;z=(x-m['center'])/m['scale'];e=np.exp(z[:,None]*m['rates'])
        w=m['weights']/m['normalizers'];a=m['rates']/m['scale'][:,None]
        v=np.sum(w*(e-1)**2)
        grad=np.sum(w*2*a*(e*e-e),axis=1)
        hess=np.sum(w*2*a*a*(2*e*e-e),axis=1)
        phi=np.r_[1,z,z*z,z[0]*z[1],z[0]*z[2],z[1]*z[2]]
        mid=(m['action_lower']+m['action_upper'])/2;half=(m['action_upper']-m['action_lower'])/2
        alpha=mid[1]+half[1]*np.tanh(m['alpha_weights']@phi)
        angle=np.deg2rad(alpha+x[2])
        c=self.p['gravity']*(grad[0]*np.cos(angle)+grad[1]*np.sin(angle)/x[0]*180/np.pi)
        u=np.array([mid[0]-half[0]*np.tanh(half[0]*c/m['temperature']),alpha,
                    mid[2]-half[2]*np.tanh(half[2]*grad[2]/m['temperature'])])
        f=np.asarray(drift(x,u,self.p))
        gv=grad@f+.5*np.sum(np.square(self.p['diffusion'])*hess)
        return v,gv,u,f


def simulate(artifact, seconds=60., dt=.001, seed=0, stochastic=True, initial=None):
    if not np.isfinite([seconds,dt]).all() or seconds<=0 or dt<=0: raise ValueError('Positive finite simulation time and step required')
    evaluator=Evaluator(artifact);p=evaluator.p;rng=np.random.default_rng(seed)
    inside=lambda x,box:bool(np.all((x>=np.asarray(box)[:,0])&(x<=np.asarray(box)[:,1])))
    x=np.mean(p['initial'],axis=1) if initial is None else np.asarray(initial,dtype=float)
    if x.shape!=(3,) or not inside(x,p['initial']): raise ValueError('Animation must start in the initial region')
    position=np.zeros(2);rows=[];time=0.;next_frame=0.;outcome='time_limit'
    sigma=np.asarray(p['diffusion']) if stochastic else np.zeros(3)
    while True:
        if not np.isfinite(x).all(): raise FloatingPointError('Nonfinite rollout; reduce the time step')
        if not inside(x,p['domain']): outcome='domain_exit'
        elif any(inside(x,b) for b in p['unsafe']): outcome='unsafe'
        elif inside(x,p['goal']): outcome='goal'
        terminal=outcome!='time_limit' or time>=seconds
        if x[0]<=0: raise FloatingPointError('Rollout crossed zero airspeed; reduce the time step')
        v,gv,u,f=evaluator(x)
        if time>=next_frame or terminal:
            rows.append([time,*x,*position,u[0]*p['mass']*p['gravity'],u[1],u[2],v,gv])
            next_frame=time+.05
        if terminal: break
        h=min(dt,seconds-time)
        position+=h*x[0]*np.array([np.cos(np.deg2rad(x[1])),np.sin(np.deg2rad(x[1]))])
        x=x+h*f+np.sqrt(h)*sigma*rng.standard_normal(3)
        time=min(seconds,time+h)
    return dict(candidate_sha256=fingerprint(artifact),outcome=outcome,seed=seed,dt=dt,stochastic=stochastic,
                method='Euler-Maruyama; stop events checked at integration steps; illustrative, not a verification proof',
                columns=['time_s','airspeed_m_s','gamma_deg','tilt_deg','distance_m','relative_altitude_m',
                         'thrust_N','alpha_deg','tilt_rate_deg_s','V','GV'],samples=rows)


def animate(output, **options):
    output=Path(output);artifact=json.loads((output/'candidate.json').read_text())
    print('Simulating the smooth XV-15 controller for animation...',flush=True)
    rollout=simulate(artifact,**options)
    report=json.loads((output/'verification.json').read_text()) if (output/'verification.json').exists() else {}
    status=report.get('status','Unverified') if report.get('candidate_sha256')==fingerprint(artifact) else 'Unverified'
    (output/'rollout.json').write_text(json.dumps(rollout,indent=2,allow_nan=False)+'\n')
    with (output/'rollout.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(rollout['columns']);writer.writerows(rollout['samples'])
    data=dict(rollout=rollout,problem=artifact['problem'],verification=status)
    payload=json.dumps(data,allow_nan=False).replace('<','\\u003c')
    template=(Path(__file__).parent/'aircraft.html').read_text()
    (output/'aircraft_animation.html').write_text(template.replace('__ROLLOUT_DATA__',payload))
    print(f"Animation: {output/'aircraft_animation.html'} ({rollout['outcome']} at {rollout['samples'][-1][0]:.2f} s)",flush=True)
    return rollout
