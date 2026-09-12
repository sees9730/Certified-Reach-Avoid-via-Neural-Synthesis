"""Joint certificate/controller bound training. Run python -m nova_3d_xv15_syn train."""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

try:
    from .interval import I
    from .model import Model
except ImportError:
    from interval import I
    from model import Model

HERE = Path(__file__).resolve().parent


class Partition:
    """A full-domain binary partition, with a replayable coverage witness."""
    def __init__(self, problem):
        self.p = problem
        self.cells = [np.array(problem['domain'], dtype=float)]
        self.history = []

    def split(self, index, axis, cut):
        if type(index) is not int or not 0 <= index < len(self.cells) or type(axis) is not int or not 0 <= axis < 3:
            raise ValueError('Invalid split index/axis')
        parent = self.cells[index]
        if not parent[axis,0] < cut < parent[axis,1]: raise ValueError('Invalid split location')
        left, right = parent.copy(), parent.copy()
        left[axis,1], right[axis,0] = cut, cut
        self.cells[index] = left
        self.cells.append(right)
        self.history.append([index,axis,cut])

    def align(self):
        for axis in range(3):
            cuts=sorted({b[axis][side] for b in [self.p['goal'],*self.p['unsafe']] for side in (0,1)})
            for cut in cuts:
                for index in range(len(self.cells)):
                    if self.cells[index][axis,0] < cut < self.cells[index][axis,1]: self.split(index,axis,cut)

    def tensors(self): return torch.tensor(np.array(self.cells),dtype=torch.float64)


def in_goal(boxes, p):
    goal = torch.tensor(p['goal'],dtype=torch.float64)
    return ((boxes[:,:,0]>=goal[:,0])&(boxes[:,:,1]<=goal[:,1])).all(-1)


def refine(model, partition, maximum, count):
    """Split violated cells along the axis giving the best child upper bound."""
    with torch.no_grad():
        boxes=partition.tensors(); v,g=model.bounds(boxes)
        bad=(v.lo<model.p['beta_ra']) & (g.hi>-model.p['epsilon']) & ~in_goal(boxes,model.p)
        _,center_g=model.bounds(I(boxes.mean(-1)))
        # Train actual counterexamples; splitting them cannot fix the controller.
        bad &= (g.hi-center_g.hi) > .5*torch.relu(center_g.hi+model.p['epsilon'])
        indices=bad.nonzero().flatten()
        if not len(indices): return 0
        count=min(count,len(indices),maximum-len(boxes))
        if count<=0: return len(indices)
        indices=indices[torch.topk(g.hi[indices],count).indices]
        parents=boxes[indices]; mid=parents.mean(-1)
        choices=[]
        for axis in range(3):
            left,right=parents.clone(),parents.clone()
            left[:,axis,1]=mid[:,axis];right[:,axis,0]=mid[:,axis]
            children=torch.cat([left,right]); cv,cg=model.bounds(children)
            score=torch.where((cv.lo>=model.p['beta_ra'])|in_goal(children,model.p),-1e10,cg.hi)
            choices.append(score.reshape(2,-1).amax(0))
        axes=torch.stack(choices).argmin(0)
        for index,axis in zip(indices.tolist(),axes.tolist()):
            partition.split(index,axis,float(boxes[index,axis].mean()))
        return len(indices)


def train(p, args):
    torch.set_num_threads(1); torch.manual_seed(args.seed)
    model=Model(p); initial_state=model.export()
    partition=Partition(p);partition.align()
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    initial=torch.tensor([p['initial']],dtype=torch.float64)
    unsafe=torch.tensor(p['unsafe'],dtype=torch.float64)
    if args.max_cells<len(partition.cells): raise ValueError('max-cells is smaller than the initial partition')
    started=perf_counter();history=[]
    domain=torch.tensor(p['domain'],dtype=torch.float64)
    for step in range(args.steps):
        if step<args.warmup:
            x=domain[:,0]+torch.rand(args.batch,3,dtype=torch.float64)*(domain[:,1]-domain[:,0])
            boxes=torch.stack([x,x],-1)
        else:
            boxes=partition.tensors()
            if len(boxes)>args.batch:
                with torch.no_grad():
                    bv,bg=model.bounds(boxes)
                    score=torch.where((bv.lo<p['beta_ra'])&~in_goal(boxes,p),bg.hi,-1e10)
                    chosen=torch.cat([score.topk(args.batch//2).indices,torch.randperm(len(boxes))[:args.batch-args.batch//2]])
                boxes=boxes[chosen]
        v,g=model.bounds(boxes)
        active=(v.lo.detach()<p['beta_ra']*1.02)&~in_goal(boxes,p)
        iv,_=model.bounds(initial);uv,_=model.bounds(unsafe)
        ig=torch.relu(iv.hi-.95)
        ug=torch.relu(1.05-uv.lo/p['beta_ra'])
        violation=torch.relu(g.hi+.002)/(1+v.hi.detach())
        ag=violation[active]
        loss=10*ig.square().mean()+10*ug.square().mean()
        if len(ag): loss=loss+ag.square().mean()+.2*ag.max()
        opt.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),20)
        opt.step()
        model.project()
        if step%args.refine_every==0 or step==args.steps-1:
            if step>=args.warmup: refine(model,partition,args.max_cells,args.refine_batch)
            with torch.no_grad():
                bv,bg=model.bounds(partition.tensors());iv,_=model.bounds(initial);uv,_=model.bounds(unsafe)
                active=(bv.lo<p['beta_ra'])&~in_goal(partition.tensors(),p)
                bad=int(((bg.hi>-p['epsilon'])&active).sum())
                gv=float(bg.hi[active].max()) if active.any() else -float('inf')
                row=dict(step=step,seconds=perf_counter()-started,loss=float(loss),cells=len(partition.cells),
                         failed_generator=bad,V_initial_upper=float(iv.hi.max()),V_unsafe_lower=float(uv.lo.min()),GV_active_upper=gv)
                history.append(row)
                print(f"step={step:5d} cells={row['cells']:6d} failed_GV={bad:5d} init={row['V_initial_upper']:.4g} unsafe={row['V_unsafe_lower']:.4g} GV={gv:.4g}",flush=True)
                if row['V_initial_upper']<=1 and row['V_unsafe_lower']>=p['beta_ra'] and bad==0:
                    break
        if perf_counter()-started>=args.seconds: break
    args.output.mkdir(parents=True,exist_ok=True)
    with torch.no_grad(): witnesses=model.action_bounds(partition.tensors())[1].hi.argmin(-1).tolist()
    artifact=dict(controller_type='smooth_affine_softmin',problem=p,model=model.export(),partition=partition.history,witness_actions=witnesses,seed=args.seed,
                  parameter_changes={k:float(np.linalg.norm(np.array(model.export()[k])-np.array(initial_state[k])))
                                     for k in ('weights','alpha_weights')})
    (args.output/'candidate.json').write_text(json.dumps(artifact,indent=2,allow_nan=False)+'\n')
    with (args.output/'training.csv').open('w') as stream:
        writer=csv.DictWriter(stream,history[0]);writer.writeheader();writer.writerows(history)
    return artifact


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['train','verify','plot','animate'])
    parser.add_argument('--problem',type=Path,default=HERE/'problem.json')
    parser.add_argument('--output',type=Path,default=HERE/'results')
    parser.add_argument('--steps',type=int,default=10000)
    parser.add_argument('--seconds',type=float,default=600)
    parser.add_argument('--warmup',type=int,default=250)
    parser.add_argument('--lr',type=float,default=.02)
    parser.add_argument('--batch',type=int,default=2048)
    parser.add_argument('--refine-every',type=int,default=25)
    parser.add_argument('--refine-batch',type=int,default=256)
    parser.add_argument('--max-cells',type=int,default=30000)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--no-animation',action='store_true')
    parser.add_argument('--animation-time',type=float,default=60.)
    parser.add_argument('--animation-dt',type=float,default=.001)
    parser.add_argument('--animation-seed',type=int,default=0)
    parser.add_argument('--deterministic',action='store_true',help='Use drift-only animation; verification still uses the full SDE')
    args=parser.parse_args()
    if min(args.steps,args.seconds,args.batch,args.refine_every,args.refine_batch,args.max_cells,args.lr)<=0 or args.warmup<0:
        parser.error('Budgets, batch sizes, intervals and learning rate must be positive; warmup must be nonnegative')
    if args.command=='train': train(json.loads(args.problem.read_text()),args)
    if args.command in ('train','verify'):
        try: from .verify import verify
        except ImportError: from verify import verify
        report=verify(json.loads((args.output/'candidate.json').read_text()))
        (args.output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='cells'},indent=2))
    if args.command in ('train','verify','plot'):
        try: from .plotting import plot
        except ImportError: from plotting import plot
        plot(args.output)
    if args.command in ('train','verify','animate') and not args.no_animation:
        try: from .animation import animate
        except ImportError: from animation import animate
        animate(args.output,seconds=args.animation_time,dt=args.animation_dt,seed=args.animation_seed,stochastic=not args.deterministic)


if __name__=='__main__': main()
