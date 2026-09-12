"""Joint bound training in a learned curved coordinate system."""
import numpy as np
import torch


class B:
    def __init__(self,l,h=None):
        self.l=torch.as_tensor(l,dtype=torch.float64);self.h=self.l if h is None else torch.as_tensor(h,dtype=torch.float64)
    def __add__(self,b):
        b=wrap(b);return B(self.l+b.l,self.h+b.h)
    __radd__=__add__
    def __neg__(self): return B(-self.h,-self.l)
    def __sub__(self,b): return self+-wrap(b)
    def __rsub__(self,b): return wrap(b)+-self
    def __mul__(self,b):
        b=wrap(b);v=torch.stack(torch.broadcast_tensors(self.l*b.l,self.l*b.h,self.h*b.l,self.h*b.h));return B(v.amin(0),v.amax(0))
    __rmul__=__mul__
    def square(self):
        return B(torch.where((self.l<=0)&(self.h>=0),0.,torch.minimum(self.l**2,self.h**2)),torch.maximum(self.l**2,self.h**2))
def wrap(x): return x if isinstance(x,B) else B(x)


class CurvedPair(torch.nn.Module):
    def __init__(self,p):
        super().__init__();self.p=p
        self.k=torch.nn.Parameter(torch.tensor([.2,1.],dtype=torch.float64))
        self.d=torch.nn.Parameter(torch.tensor([1.,3.5],dtype=torch.float64))
        self.amplitude=torch.nn.Parameter(torch.tensor(1.1,dtype=torch.float64))
        self.weights=torch.nn.Parameter(torch.full((2,10),.005,dtype=torch.float64))
        with torch.no_grad(): self.weights[0,0]=.4;self.weights[1,0]=.02
        self.gamma=torch.tensor([1.3,1.],dtype=torch.float64)
        c=np.mean(p['initial'],axis=1);self.anchor=float(c[0])
        if not self.anchor: raise ValueError("The curved family requires a nonzero initial px center")
        self.slope=float(c[1]/c[0])
        self.scales=torch.ones(2,dtype=torch.float64)
        q=self.region_energy(torch.tensor([p['initial']],dtype=torch.float64))
        self.scales=torch.tensor([float(t.h.detach()[0]) for t in q],dtype=torch.float64)

    def curve(self,x):
        b=self.slope+self.amplitude;c=-self.amplitude/self.anchor**2
        h=lambda z:b*z+c*z**3
        dh=lambda z:b+3*c*z*z
        span=x.h-x.l
        coefficients=torch.stack([h(x.l),h(x.l)+span*dh(x.l)/3,h(x.h)-span*dh(x.h)/3,h(x.h)])
        return B(coefficients.amin(0),coefficients.amax(0)),b+3*c*x.square(),6*c*x

    def region_energy(self,boxes):
        x,y,vx,vy=[B(boxes[:,i,0],boxes[:,i,1]) for i in range(4)]
        h,dh,_=self.curve(x);s=y-h;t=vy-dh*vx
        def quadratic(p,v,k,gamma):
            q=lambda a,b:a*a+gamma*(b+k*a)**2
            clamp=lambda a,l,h:torch.minimum(h,torch.maximum(l,a))
            corners=[q(a,b) for a in (p.l,p.h) for b in (v.l,v.h)]
            values=corners+[q(a,clamp(-k*a,v.l,v.h)) for a in (p.l,p.h)]
            values += [q(clamp(-gamma*k*b/(1+gamma*k*k),p.l,p.h),b) for b in (v.l,v.h)]
            low=torch.stack(values).amin(0)
            low=torch.where((p.l<=0)&(p.h>=0)&(v.l<=0)&(v.h>=0),0.,low)
            return B(low,torch.stack(corners).amax(0))
        return [quadratic(x,vx,self.k[0],self.gamma[0]),quadratic(s,t,self.k[1],self.gamma[1])]

    def value(self,q,axis):
        j=torch.arange(1,self.weights.shape[1]+1,dtype=torch.float64)
        return ((q[...,None]/self.scales[axis])**j*self.weights[axis]).sum(-1)

    def caps(self,beta):
        with torch.no_grad():
            result=[]
            for i in range(2):
                lo=0.;hi=float(self.scales[i])
                while float(self.value(torch.tensor(hi,dtype=torch.float64),i))<beta: hi*=2
                for _ in range(40):
                    mid=(lo+hi)/2
                    if float(self.value(torch.tensor(mid,dtype=torch.float64),i))>=beta: hi=mid
                    else: lo=mid
                result.append(hi*(1+1e-6))
            return torch.tensor(result,dtype=torch.float64)

    def energy_bounds(self,boxes,caps):
        ql=boxes[:,:,0]*caps;qh=boxes[:,:,1]*caps
        low=sum(self.value(ql[:,i],i) for i in range(2));high=sum(self.value(qh[:,i],i) for i in range(2))
        radius=torch.sqrt(qh[:,0]);_,hp,_=self.curve(B(-radius,radius))
        variance=torch.tensor(self.p['acceleration_noise'],dtype=torch.float64)**2
        noises=[self.gamma[0]*variance[0],self.gamma[1]*(variance[1]+hp.square().h*variance[0])]
        j=torch.arange(1,self.weights.shape[1]+1,dtype=torch.float64)
        gv=0
        for i in range(2):
            lo=ql[:,i,None];hi=qh[:,i,None];nu=wrap(noises[i]).h
            if nu.ndim: nu=nu[:,None]
            factor=B(j/self.scales[i]*(lo/self.scales[i])**(j-1),j/self.scales[i]*(hi/self.scales[i])**(j-1))
            pos=-2*self.k[i]*B(lo,hi)+nu
            vel=-2*self.d[i]*B(lo,hi)+(2*j-1)*nu
            gv+=((factor*B(torch.minimum(pos.l,vel.l),torch.maximum(pos.h,vel.h))).h*self.weights[i]).sum(-1)
        # A conservative physical box enclosing all states in this energy rectangle.
        sy=torch.sqrt(qh[:,1]);vx=radius*torch.sqrt(1/self.gamma[0]+self.k[0]**2)
        h,dh,_=self.curve(B(-radius,radius));py=torch.maximum(h.l.abs(),h.h.abs())+sy
        vy=torch.maximum(dh.l.abs(),dh.h.abs())*vx+sy*torch.sqrt(1/self.gamma[1]+self.k[1]**2)
        limit=torch.tensor([min(-a,b) for a,b in self.p['goal']],dtype=torch.float64)
        goal=(torch.stack([radius,py,vx,vy],-1)<=limit).all(-1)
        return low,high,gv,goal

    def export(self):
        def strings(t): return np.array(t.detach()).astype(str).tolist()
        return dict(k=strings(self.k),d=strings(self.d),gamma=strings(self.gamma),weights=strings(self.weights),
                    amplitude=strings(self.amplitude),anchor=str(self.anchor),slope=str(self.slope),scales=strings(self.scales))


def split(cells,history,index,axis,group=None):
    left=cells[index].copy();right=left.copy();cut=float(left[axis].mean())
    if not left[axis,0]<cut<left[axis,1]: raise ValueError("Cell cannot be split further in floating point")
    left[axis,1]=cut;right[axis,0]=cut;cells[index]=left;cells.append(right)
    history.append(([group] if group else [])+[index,axis,repr(cut)])

