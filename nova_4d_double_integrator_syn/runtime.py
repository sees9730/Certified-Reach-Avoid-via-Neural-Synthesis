"""Floating-point deployment and illustrative SDE rollouts, separate from proof."""
import numpy as np


class Controller:
    def __init__(self,candidate):
        self.problem=candidate['problem'];m=candidate['model']
        self.curved=candidate['format']=='curved_double_integrator_v2'
        self.k=np.array(m['k'],float);self.d=np.array(m['d'],float)
        self.weights=np.array(m['weights'],float)
        if self.curved:
            self.gamma=np.array(m['gamma'],float);self.scales=np.array(m['scales'],float)
            self.b=float(m['slope'])+float(m['amplitude']);self.c=-float(m['amplitude'])/float(m['anchor'])**2
        else: self.scale=float(m['scale'])

    def coordinates(self,x):
        px=x[...,0];vx=x[...,2];hp=self.b+3*self.c*px**2
        s=x[...,1]-(self.b*px+self.c*px**3);t=x[...,3]-hp*vx
        pos=np.stack([px,s],-1);vel=np.stack([vx,t],-1)
        return pos,vel,vel+self.k*pos,hp,6*self.c*px

    def raw_control(self,x):
        pos,vel,_,hp,hpp=self.coordinates(x)
        force=-(1/self.gamma+self.k*self.d)*pos-(self.k+self.d)*vel
        return np.stack([force[...,0],hp*force[...,0]+hpp*x[...,2]**2+force[...,1]],-1)

    def __call__(self,state):
        x=np.asarray(state,float)
        if self.curved:
            raw=self.raw_control(x);limit=np.array(self.problem['control_limit']);core=.95*limit
            r=np.maximum(np.abs(raw)-core,0);z=(limit-core)/np.maximum(r,1e-300)
            tail=core+(limit-core)/(z+np.exp(-z))
            return np.where(r>0,np.sign(raw)*tail,raw)
        return -(1+self.k*self.d)*x[...,:2]-(self.k+self.d)*x[...,2:]

    def values(self,state):
        if self.curved:
            x=np.asarray(state,float);pos,vel,w,hp,_=self.coordinates(x)
            q=pos**2+self.gamma*w**2;j=np.arange(1,self.weights.shape[1]+1)
            ratio=q[...,None]/self.scales[:,None]
            value=np.sum(self.weights*ratio**j,axis=(-2,-1))
            first=np.sum(self.weights*j/self.scales[:,None]*ratio**(j-1),axis=-1)
            second=np.sum(self.weights[:,1:]*j[1:]*(j[1:]-1)/self.scales[:,None]**2*ratio**(j[1:]-2),axis=-1)
            variance=np.square(self.problem['acceleration_noise'])
            noise=np.stack([np.full_like(hp,self.gamma[0]*variance[0]),self.gamma[1]*(variance[1]+hp**2*variance[0])],-1)
            delta=self(x)-self.raw_control(x)
            transformed=np.stack([delta[...,0],delta[...,1]-hp*delta[...,0]],-1)
            gq=-2*(self.k*pos**2+self.gamma*self.d*w**2)+noise+2*self.gamma*w*transformed
            return value,np.sum(first*gq+2*second*self.gamma*noise*w**2,axis=-1)
        x=np.asarray(state,float);p=x[...,:2];z=x[...,2:]+self.k*p
        q=np.sum(p*p+z*z,axis=-1);j=np.arange(1,len(self.weights)+1)
        ratio=q[...,None]/self.scale
        value=np.sum(self.weights*ratio**j,axis=-1)
        first=np.sum(self.weights*j/self.scale*ratio**(j-1),axis=-1)
        second=np.sum(self.weights[1:]*j[1:]*(j[1:]-1)/self.scale**2*ratio**(j[1:]-2),axis=-1)
        variance=np.square(self.problem['acceleration_noise'])
        gq=-2*np.sum(self.k*p*p+self.d*z*z,axis=-1)+np.sum(variance)
        gamma=4*np.sum(variance*z*z,axis=-1)
        return value,first*gq+second*gamma/2


def rollout(controller,seed=0,dt=.002,seconds=15):
    if dt<=0 or seconds<=0: raise ValueError('Positive simulation step/horizon required')
    p=controller.problem;rng=np.random.default_rng(seed);initial=np.array(p['initial'])
    x=rng.uniform(initial[:,0],initial[:,1]);time=0.;rows=[]
    inside=lambda b:np.all((x>=np.array(b)[:,0])&(x<=np.array(b)[:,1]))
    while True:
        status='running'
        if not inside(p['domain']): status='domain_exit'
        elif any(inside(b) for b in p['unsafe']): status='unsafe'
        elif inside(p['goal']): status='goal'
        elif time>=seconds: status='time_limit'
        u=controller(x);v,g=controller.values(x);rows.append([time,*x,*u,float(v),float(g)])
        if status!='running': break
        h=min(dt,seconds-time)
        x=x+h*np.r_[x[2:],u]+np.sqrt(h)*np.r_[0.,0.,np.array(p['acceleration_noise'])*rng.standard_normal(2)]
        time=min(seconds,time+h)
    return dict(seed=seed,dt=dt,outcome=status,columns=['time','px','py','vx','vy','ux','uy','V','GV'],samples=rows)
