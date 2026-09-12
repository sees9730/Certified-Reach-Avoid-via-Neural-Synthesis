"""XV-15 dynamics, globally nonnegative certificate and bounded learned policy.

Internal angles are degrees; thrust is in units of mass*gravity.
"""
import math
import numpy as np
import torch
from scipy.optimize import linprog

try:
    from .interval import I, quadratic
except ImportError:
    from interval import I, quadratic


def drift(x, u, p):
    sin=lambda a:a.sin() if hasattr(a,'sin') else np.sin(a)
    cos=lambda a:a.cos() if hasattr(a,'cos') else np.cos(a)
    square=lambda a:a.square() if hasattr(a,'square') else np.square(a)
    v, gamma, beta = [x[..., i] for i in range(3)]
    thrust, alpha, delta = [u[i] for i in range(3)]
    radians = math.pi/180
    r = beta/90
    cl = (0.0849*alpha+0.3482)*(1-r) + (0.0646*alpha+0.2709)*r
    cd = (0.00042143*square(alpha)+0.0030*alpha+0.0218)*(1-r) + (0.000473216*square(alpha)+0.00343*alpha+0.2165)*r
    qsm = (0.5*p['density']*p['wing_area']/p['mass'])*square(v)
    angle = (alpha+beta)*radians
    g = p['gravity']
    return [g*thrust*cos(angle)-qsm*cd-g*sin(gamma*radians),
            (g*thrust*sin(angle)+qsm*cl-g*cos(gamma*radians))/v/radians, delta]


class Model(torch.nn.Module):
    def __init__(self, p, initialize=True):
        super().__init__()
        self.p = p
        self.register_buffer('center', torch.tensor(np.mean(p['goal'], axis=1), dtype=torch.float64))
        self.register_buffer('scale', torch.tensor([50.,20.,45.], dtype=torch.float64))
        self.register_buffer('rates', torch.tensor([-4.,-2.,-1.,1.,2.,4.], dtype=torch.float64))
        edges = (torch.tensor(p['domain'], dtype=torch.float64)-self.center[:,None])/self.scale[:,None]
        self.register_buffer('normalizers', torch.expm1(edges[:,:,None]*self.rates).square().amax(1))
        self.weights = torch.nn.Parameter(torch.full((3,6), .001, dtype=torch.float64))
        limits=np.array(p['control_limits'],dtype=float)
        # Inward rounding keeps binary64 parameters inside exact decimal limits.
        self.register_buffer('action_lower',torch.tensor(np.nextafter(limits[:,0],np.inf)))
        self.register_buffer('action_upper',torch.tensor(np.nextafter(limits[:,1],-np.inf)))
        self.alpha_weights=torch.nn.Parameter(torch.zeros(10,dtype=torch.float64))
        self.register_buffer('temperature',torch.tensor(.001,dtype=torch.float64))
        if initialize: self.initialize()

    def policy(self, x):
        """Analytic smooth feedback; no argmin, switching, or clipping at runtime."""
        if not isinstance(x,I): x=I(x)
        if not torch.equal(x.lo,x.hi): raise ValueError('Policy evaluation expects points')
        alpha=self.alpha(x)
        _,grad,_=self.features(x);gradient=(grad*self.weights).sum(-1)
        angle=(alpha+x[...,2])*(math.pi/180)
        c=self.p['gravity']*(gradient[...,0]*angle.cos()+gradient[...,1]*angle.sin()/x[...,0]/(math.pi/180))
        result=[]
        for i,coefficient in [(0,c),(2,gradient[...,2])]:
            mid=(self.action_lower[i]+self.action_upper[i])/2
            half=(self.action_upper[i]-self.action_lower[i])/2
            result.append(mid-half*(coefficient*half/self.temperature).tanh())
        return [result[0],alpha,result[1]]

    def alpha(self,x):
        z=(x-self.center)/self.scale
        phi=[I(torch.ones_like(z.lo[...,0]))]+[z[...,i] for i in range(3)]
        phi += [z[...,i].square() for i in range(3)]+[z[...,i]*z[...,j] for i,j in ((0,1),(0,2),(1,2))]
        raw=sum(h*w for h,w in zip(phi,self.alpha_weights))
        return (self.action_lower[1]+self.action_upper[1])/2+(self.action_upper[1]-self.action_lower[1])/2*raw.tanh()

    def features(self, x):
        z = (x-self.center)/self.scale
        e = (z[..., :,None]*self.rates).exp()
        feature = (e-1).square()/self.normalizers
        grad = quadratic(e,1.,-1.)*(2*self.rates/self.scale[:,None])/self.normalizers
        hess = quadratic(e,2.,-1.)*(2*(self.rates/self.scale[:,None])**2)/self.normalizers
        return feature, grad, hess

    def action_bounds(self, boxes):
        x = boxes if isinstance(boxes,I) else I(boxes[...,0], boxes[...,1])
        feature, grad, hess = self.features(x)
        v = (feature*self.weights).sum(-1).sum(-1)
        gradient=(grad*self.weights).sum(-1)
        noise=sum((hess[...,i,:]*self.weights[i]).sum(-1)*(.5*self.p['diffusion'][i]**2) for i in range(3))
        thrust=torch.stack([self.action_lower[0],self.action_lower[0],self.action_upper[0],self.action_upper[0]])
        delta=torch.stack([self.action_lower[2],self.action_upper[2],self.action_lower[2],self.action_upper[2]])
        f=drift(x[...,None,:],[I(thrust),self.alpha(x)[...,None],I(delta)],self.p)
        gv=sum(gradient[...,i,None]*f[i] for i in range(3))+noise[...,None]
        return v, gv

    def bounds(self, boxes):
        v,g=self.action_bounds(boxes)
        # Smooth binary soft-min costs at most temperature*log(2) per affine control.
        return v,I(g.lo.amin(-1),g.hi.amin(-1)+2*self.temperature*math.log(2))

    def values(self,x):
        """Actual pointwise V and GV for the smooth controller (not proof bounds)."""
        if not isinstance(x,I): x=I(x)
        h,grad,hess=self.features(x);v=(h*self.weights).sum(-1).sum(-1)
        f=drift(x,self.policy(x),self.p)
        gv=sum((grad[...,i,:]*self.weights[i]).sum(-1)*f[i]+(hess[...,i,:]*self.weights[i]).sum(-1)*(.5*self.p['diffusion'][i]**2) for i in range(3))
        return v,gv

    def project(self):
        with torch.no_grad():
            self.weights.clamp_(min=0)

    def initialize(self):
        """LP initializes value separation only; it does not solve the GV condition."""
        p=self.p
        with torch.no_grad():
            fi=self.features(I(np.array(p['initial'])[:,0],np.array(p['initial'])[:,1]))[0].hi.numpy().ravel()
            fu=self.features(I(np.array(p['unsafe'])[:,:,0],np.array(p['unsafe'])[:,:,1]))[0].lo.numpy().reshape(-1,18)
            fit=linprog(fi+1e-6,A_ub=-fu,b_ub=-np.full(len(fu),p['beta_ra']*1.1),bounds=(0,None),method='highs')
            if not fit.success: raise ValueError('Certificate feature family cannot separate these regions')
            self.weights.copy_(torch.tensor(fit.x.reshape(3,6)))
            self.weights.mul_(max(1.,.8/float(fi@fit.x)))
            # Fit an angle-channel initialization to pointwise low-GV proposals.
            rng=np.random.default_rng(0);domain=np.array(p['domain'])
            x=torch.tensor(rng.uniform(domain[:,0],domain[:,1],(4096,3)),dtype=torch.float64)
            _,grad,_=self.features(I(x));gradient=(grad.lo*self.weights).sum(-1)
            choices=torch.tensor([[t,a,0.] for t in [float(self.action_lower[0]),float(self.action_upper[0])]
                                  for a in np.linspace(float(self.action_lower[1]),float(self.action_upper[1]),17)],dtype=torch.float64)
            f=drift(I(x[:,None,:]),[I(choices[:,i]) for i in range(3)],p)
            score=sum(gradient[:,i,None]*f[i].lo for i in range(3));target=choices[score.argmin(-1),1]
            z=(x-self.center)/self.scale
            phi=torch.stack([torch.ones(len(x)),*[z[:,i] for i in range(3)],*[z[:,i]**2 for i in range(3)],
                             *[z[:,i]*z[:,j] for i,j in ((0,1),(0,2),(1,2))]],-1)
            mid=(self.action_lower[1]+self.action_upper[1])/2;half=(self.action_upper[1]-self.action_lower[1])/2
            self.alpha_weights.copy_(torch.linalg.lstsq(phi,torch.atanh(((target-mid)/half).clamp(-.95,.95))).solution)

    def export(self):
        return {name: value.detach().tolist() for name,value in self.state_dict().items()}

    def restore(self, values):
        self.load_state_dict({k:torch.tensor(v,dtype=torch.float64) for k,v in values.items()})
        if not torch.isfinite(torch.cat([p.ravel() for p in self.parameters()])).all() or (self.weights<0).any():
            raise ValueError('Invalid learned parameters')
