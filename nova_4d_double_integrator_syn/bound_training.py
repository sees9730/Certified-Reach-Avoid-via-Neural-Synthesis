"""Controller parameter search, certificate fitting, and complete bound refinement."""
import itertools
import time
import numpy as np
import torch
from scipy.optimize import linprog
try:
    from .curved import CurvedPair,B,split
    from .curved_check import Proof
except ImportError:
    from curved import CurvedPair,B,split
    from curved_check import Proof


def features(model,cells,caps):
    saved=model.weights.clone();columns=[]
    for j in range(model.weights.numel()):
        model.weights.zero_();model.weights.reshape(-1)[j]=1
        columns.append(model.energy_bounds(cells,caps)[2].numpy())
    model.weights.copy_(saved)
    return np.array(columns).T


def region_features(model,boxes,upper):
    q=model.region_energy(torch.tensor(np.array(boxes),dtype=torch.float64))
    powers=np.arange(1,model.weights.shape[1]+1)
    return np.concatenate([np.array((t.h if upper else t.l)[:,None]/model.scales[i])**powers for i,t in enumerate(q)],axis=1)


def fit_weights(A,b):
    # Scaling improves conditioning; no LP status is accepted as a certificate proof.
    scale=np.maximum(np.max(abs(A),axis=1),1e-4)
    return linprog(np.ones(A.shape[1]),A_ub=A/scale[:,None],b_ub=b/scale,bounds=(0,None),
                   options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})


def propose(model,caps,options):
    """Finite bound constraints produce a proposal, never a SAT declaration."""
    p=model.p;size=options['grid_size'];powers=np.arange(1,model.weights.shape[1]+1)
    a,b=np.meshgrid(np.linspace(0,1,size)**2,np.linspace(0,1,size)**2)
    z=np.stack([a.ravel(),b.ravel()],1)
    radii=np.array([min(-a,b) for a,b in p['goal']])
    qmax=min(radii[0]**2,radii[2]**2/float(1/model.gamma[0]+model.k[0]**2))
    qx=torch.linspace(0,qmax,options['goal_boundary_points'],dtype=torch.float64)
    rx=qx.sqrt();h,hp,_=model.curve(B(-rx,rx));vx=rx*torch.sqrt(1/model.gamma[0]+model.k[0]**2)
    ry=torch.minimum(radii[1]-torch.maximum(h.l.abs(),h.h.abs()),
        (radii[3]-torch.maximum(hp.l.abs(),hp.h.abs())*vx)/torch.sqrt(1/model.gamma[1]+model.k[1]**2)).clamp(min=0)
    boundary=np.stack([qx.numpy()/float(caps[0]),(ry.numpy()**2+1e-9)/float(caps[1])],1)
    # Also cover the vertical boundary where the px/vx goal constraint becomes active.
    vertical=np.stack([np.full(size,(qmax+1e-9)/float(caps[0])),np.linspace(0,1,size)**2],1)
    z=np.concatenate([z,boundary,vertical]);z=z[(z<=1).all(1)]
    cells=torch.tensor(np.stack([z,z],-1),dtype=torch.float64)
    _,_,_,goal=model.energy_bounds(cells,caps);G=features(model,cells,caps)[~goal]
    initial=np.array(list(itertools.product(*p['initial'])))
    R=region_features(model,np.stack([initial,initial],-1),True)
    interior=[b for b in p['unsafe'] if all(p['domain'][i][0]<b[i][0]<=b[i][1]<p['domain'][i][1] for i in (0,1))]
    U=region_features(model,interior,False) if interior else np.empty((0,model.weights.numel()))
    cap_rows=np.zeros((2,model.weights.numel()));degree=len(powers)
    for i in range(2): cap_rows[i,degree*i:degree*(i+1)]=float(caps[i]/model.scales[i])**powers
    A=np.concatenate([R,-U,G,-cap_rows]);b=np.r_[np.full(len(R),options['initial_training_upper']),
        np.full(len(U),-1.04*p['beta_ra']),np.full(len(G),-options['generator_margin_factor']*p['epsilon']),[-1.004*p['beta_ra']]*2]
    result=fit_weights(A,b)
    if result.success: model.weights.copy_(torch.tensor(result.x.reshape(model.weights.shape),dtype=torch.float64))
    return result.success


@torch.no_grad()
def learn(p,options,steps=4000,seconds=120,seed=0):
    torch.set_num_threads(1);torch.manual_seed(seed);start=time.perf_counter();log=[];attempts=[]
    model=CurvedPair(p);before=model.export();caps=torch.tensor(options['energy_caps'],dtype=torch.float64)
    if caps.shape!=(2,) or not torch.isfinite(caps).all() or (caps<=0).any(): raise ValueError('Two positive finite energy caps required')
    for key in ('grid_size','goal_boundary_points','refine_per_step','max_cells'):
        if options[key]<2: raise ValueError('Search counts must be at least two')
    if not 0<options['initial_training_upper']<1 or options['generator_margin_factor']<=1: raise ValueError('Training margins must be strict')
    if not options['curve_amplitudes'] or not options['transverse_damping']: raise ValueError('Empty controller search')
    if not np.isfinite([*options['curve_amplitudes'],*options['transverse_damping'],options['longitudinal_damping']]).all() or min(*options['transverse_damping'],options['longitudinal_damping'])<=0:
        raise ValueError('Finite curve parameters and positive damping required')
    regions={'initial':[np.array(p['initial'],float)],'unsafe':[np.array(b,float) for b in p['unsafe']]}
    rh=[];energies=[np.array([[0.,1.],[0.,1.]])];eh=[];reason='budget exhausted';proposal=False
    for amplitude,damping in itertools.product(options['curve_amplitudes'],options['transverse_damping']):
        model.amplitude.fill_(amplitude);model.d.copy_(torch.tensor([options['longitudinal_damping'],damping],dtype=torch.float64))
        success=propose(model,caps,options);control_ok=False
        if success:
            control_ok,_=Proof(dict(problem=p,model=model.export(),energy_caps=[repr(float(c)) for c in caps])).control_cover()
        attempts.append(dict(amplitude=amplitude,damping=damping,feasible=success,control_cover_passed=control_ok))
        print('Proposal:',attempts[-1],flush=True)
        if success and control_ok: proposal=True;break
        if time.perf_counter()-start>=seconds: break
    if not proposal:
        reason='No feasible proposal within the controller search and budget'
        # Keep an independently checkable complete cover even when proposal fitting fails.
        caps=model.caps(p['beta_ra'])
    limit=np.ptp(np.array(p['domain']),axis=1);batch=options['refine_per_step']
    for step in range(steps if proposal else 1):
        eb=torch.tensor(np.array(energies),dtype=torch.float64)
        low,_,g,goal=model.energy_bounds(eb,caps);active=(low<p['beta_ra'])&~goal
        values={name:region_features(model,cells,name=='initial')@model.weights.numpy().ravel() for name,cells in regions.items()}
        initial=float(values['initial'].max());unsafe=float(values['unsafe'].min());worst=float(g[active].max()) if active.any() else -np.inf
        row=dict(step=step,seconds=time.perf_counter()-start,initial_upper=initial,unsafe_lower=unsafe,
                 generator_upper=worst,energy_cells=len(energies),region_cells=sum(map(len,regions.values())),
                 loss=max(initial-1,0)+max(p['beta_ra']-unsafe,0)+max(worst+p['epsilon'],0))
        log.append(row)
        if step%10==0: print(row,flush=True)
        if initial<=1 and unsafe>=p['beta_ra'] and worst<=-p['epsilon']:
            reason='Floating-point bounds satisfied; independent exact verification required';break
        if not proposal or time.perf_counter()-start>=seconds: break
        # A positive scaling changes all certificate/generator bounds linearly.
        # This bound-based update recovers the requested margin without changing the controller.
        if worst<0 and unsafe>0:
            factor=max(1.,1.01*p['epsilon']/-worst,1.01*p['beta_ra']/unsafe)
            if 1<factor and initial*factor<=.995:
                model.weights.mul_(factor);continue
        bad=((active&(g>-p['epsilon'])).nonzero().flatten()).tolist()
        for index in sorted(bad,key=lambda i:float(g[i]),reverse=True)[:batch]:
            parent=energies[index];width=np.ptp(parent,axis=1)
            if len(energies)>=options['max_cells'] or width.max()<1e-12: continue
            # Balanced bisection prevents one nearly flat direction from starving the other.
            split(energies,eh,index,int(np.argmax(width)))
        for name,cells in regions.items():
            violations=values[name]-.955 if name=='initial' else 1.01*p['beta_ra']-values[name]
            bad=np.where(violations>0)[0];bad=sorted(bad,key=lambda i:violations[i],reverse=True)[:batch]
            for index in bad:
                width=np.ptp(cells[index],axis=1)/limit;axis=max((0,2),key=lambda i:width[i])
                if len(cells)>=options['max_cells'] or width[axis]<1e-12: continue
                split(cells,rh,int(index),axis,name)
        if len(energies)>=options['max_cells'] and all(len(v)>=options['max_cells'] for v in regions.values()):
            reason='Cell budget exhausted';break
    after=model.export();changes={key:float(np.linalg.norm(np.array(after[key],float)-np.array(before[key],float))) for key in ('k','d','weights','amplitude')}
    candidate=dict(format='curved_double_integrator_v2',problem=p,model=after,energy_caps=[repr(float(c)) for c in caps],
        energy_splits=eh,region_splits=rh,parameter_changes=changes,seed=seed,search_options=options,controller_search=attempts,training_stop=reason)
    return candidate,log
