"""Joint bound training for a structured smooth double-integrator controller."""
import time
import numpy as np
import torch


class Pair(torch.nn.Module):
    def __init__(self, problem, degree=6):
        super().__init__()
        self.problem=problem
        self.k=torch.nn.Parameter(torch.full((2,),.6,dtype=torch.float64))
        self.d=torch.nn.Parameter(torch.full((2,),.9,dtype=torch.float64))
        self.weights=torch.nn.Parameter(torch.full((degree,),.005,dtype=torch.float64))
        self.scale=float(self.energy_bounds(torch.tensor([problem['initial']],dtype=torch.float64))[1].detach()[0])

    def energy_bounds(self, boxes):
        """Exact extrema of q on boxes, by separable convex two-variable minimization."""
        lows=[];highs=[]
        for axis,k in enumerate(self.k):
            pl,ph=boxes[:,axis,0],boxes[:,axis,1]
            vl,vh=boxes[:,axis+2,0],boxes[:,axis+2,1]
            q=lambda p,v:p*p+(v+k*p)**2
            corners=[q(p,v) for p in (pl,ph) for v in (vl,vh)]
            candidates=corners+[q(p,torch.clamp(-k*p,min=vl,max=vh)) for p in (pl,ph)]
            candidates += [q(torch.clamp(-k*v/(1+k*k),min=pl,max=ph),v) for v in (vl,vh)]
            candidates += [q(torch.clamp(torch.zeros_like(pl),min=pl,max=ph),torch.clamp(torch.zeros_like(vl),min=vl,max=vh))]
            lows.append(torch.stack(candidates).amin(0));highs.append(torch.stack(corners).amax(0))
        return sum(lows),sum(highs)

    def value(self,q):
        powers=torch.arange(1,len(self.weights)+1,dtype=torch.float64)
        return ((q[...,None]/self.scale)**powers*self.weights).sum(-1)

    def outside_energy(self):
        goal=torch.tensor(self.problem['goal'],dtype=torch.float64)
        radii=torch.minimum(-goal[:,0],goal[:,1])
        return torch.cat([radii[:2]**2,radii[2:]**2/(1+self.k*self.k)]).amin()

    def control_bound(self):
        domain=torch.tensor(self.problem['domain'],dtype=torch.float64)
        radius=domain.abs().amax(-1)
        return (1+self.k*self.d)*radius[:2]+(self.k+self.d)*radius[2:]

    def bounds(self, cuts):
        q0=self.outside_energy()
        q1=self.energy_bounds(torch.tensor([self.problem['domain']],dtype=torch.float64))[1][0]
        q=q0+(q1-q0)*torch.tensor(cuts,dtype=torch.float64)
        lo,hi=q[:-1],q[1:]
        j=torch.arange(1,len(self.weights)+1,dtype=torch.float64)
        decay=torch.cat([self.k,self.d]).amin()
        noise=torch.tensor(self.problem['acceleration_noise'],dtype=torch.float64)**2
        offset=noise.sum()+2*(j-1)*noise.max()
        a=j/self.scale*(lo[:,None]/self.scale)**(j-1)
        b=j/self.scale*(hi[:,None]/self.scale)**(j-1)
        c=-2*decay*hi[:,None]+offset
        e=-2*decay*lo[:,None]+offset
        upper=torch.stack([a*c,a*e,b*c,b*e]).amax(0)@self.weights
        return self.value(lo),self.value(hi),upper

    def export(self):
        # Decimal strings define the exact rational candidate checked independently.
        return dict(k=[repr(float(v)) for v in self.k],d=[repr(float(v)) for v in self.d],
                    weights=[repr(float(v)) for v in self.weights],scale=repr(self.scale))


def learn(problem,steps=4000,seconds=120,lr=.02,seed=0):
    torch.set_num_threads(1);torch.manual_seed(seed)
    pair=Pair(problem);before=pair.export();cuts=[0.,1.];log=[]
    optimizer=torch.optim.Adam([{'params':[pair.weights],'lr':lr},{'params':[pair.k,pair.d],'lr':lr/4}])
    boxes=torch.tensor([problem['initial'],*problem['unsafe']],dtype=torch.float64)
    start=time.perf_counter()
    for step in range(steps):
        qlo,qhi=pair.energy_bounds(boxes)
        vi=pair.value(qhi[0]);vu=pair.value(qlo[1:])
        lower,upper,generator=pair.bounds(cuts)
        active=lower.detach()<problem['beta_ra']*1.02
        bad=torch.relu(generator+.0002)/(1+upper.detach())
        ctrl=pair.control_bound()/torch.tensor(problem['control_limit'])
        loss=20*torch.relu(vi-.95)**2+20*torch.relu(1.05-vu/problem['beta_ra']).square().mean()
        loss=loss+20*torch.relu(ctrl-.98).square().mean()
        if active.any(): loss=loss+bad[active].square().mean()+bad[active].max()
        optimizer.zero_grad();loss.backward();optimizer.step()
        with torch.no_grad():
            pair.weights.clamp_(min=0);pair.k.clamp_(.05,2);pair.d.clamp_(.05,2)
        if step%10==0 or step==steps-1:
            with torch.no_grad():
                lower,upper,generator=pair.bounds(cuts)
                bad=(lower<problem['beta_ra'])&(generator>-problem['epsilon'])
                indices=bad.nonzero().flatten().tolist()
                indices=sorted(indices,key=lambda i:float(generator[i]),reverse=True)[:4]
                cuts=sorted(cuts+[(cuts[i]+cuts[i+1])/2 for i in indices if cuts[i+1]-cuts[i]>1e-12])
                lower,upper,generator=pair.bounds(cuts)
                qlo,qhi=pair.energy_bounds(boxes);vi=pair.value(qhi[0]);vu=pair.value(qlo[1:])
                act=lower<problem['beta_ra'];worst=float(generator[act].max()) if act.any() else None
                control=pair.control_bound()
                row=dict(step=step,seconds=time.perf_counter()-start,loss=float(loss),energy_cells=len(cuts)-1,
                         initial_upper=float(vi),unsafe_lower=float(vu.min()),generator_upper=worst,
                         control_upper=float(control.max()))
                log.append(row)
                if step%100==0: print(row,flush=True)
                if vi<=1 and (vu>=problem['beta_ra']).all() and (control<=torch.tensor(problem['control_limit'])).all() and (generator[act]<=-problem['epsilon']).all(): break
        if time.perf_counter()-start>=seconds: break
    model=pair.export()
    changes={key:float(np.linalg.norm(np.array(model[key],dtype=float)-np.array(before[key],dtype=float))) for key in ('k','d','weights')}
    return dict(format='smooth_double_integrator_v1',problem=problem,model=model,
                energy_partition=[repr(t) for t in cuts],parameter_changes=changes,seed=seed),log
