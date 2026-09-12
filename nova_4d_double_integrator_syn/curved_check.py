"""Independent rational interval verification for the polynomial shear model."""
from fractions import Fraction as F
from math import isqrt
try:
    from .check import validate, fingerprint, rational
except ImportError:
    from check import validate, fingerprint, rational


class I:
    def __init__(self,l,h=None): self.l=rational(l);self.h=self.l if h is None else rational(h)
    def __add__(self,b):
        b=interval(b);return I(self.l+b.l,self.h+b.h)
    __radd__=__add__
    def __neg__(self): return I(-self.h,-self.l)
    def __sub__(self,b): return self+-interval(b)
    def __rsub__(self,b): return interval(b)+-self
    def __mul__(self,b):
        b=interval(b);v=[self.l*b.l,self.l*b.h,self.h*b.l,self.h*b.h];return I(min(v),max(v))
    __rmul__=__mul__
    def square(self): return I(0 if self.l<=0<=self.h else min(self.l**2,self.h**2),max(self.l**2,self.h**2))
    def absolute(self): return max(abs(self.l),abs(self.h))
def interval(x): return x if isinstance(x,I) else I(x)
def sqrt_upper(x):
    x=rational(x);scale=2**32;n=isqrt(x.numerator*scale**2//x.denominator)
    return F(n if F(n,scale)**2==x else n+1,scale)


class Proof:
    def __init__(self,candidate):
        self.p=candidate['problem'];self.domain,self.initial,self.goal,self.unsafe=validate(self.p)
        m=candidate['model']
        for key in ('k','d','gamma','scales'):
            v=list(map(rational,m[key]))
            if len(v)!=2 or min(v)<=0: raise ValueError('Invalid '+key)
            setattr(self,key,v)
        self.weights=[list(map(rational,w)) for w in m['weights']]
        if len(self.weights)!=2 or any(not w or min(w)<0 or max(w)<=0 for w in self.weights): raise ValueError('Invalid weights')
        a,anchor,slope=[rational(m[k]) for k in ('amplitude','anchor','slope')]
        if not anchor: raise ValueError('Zero curve anchor')
        self.b=slope+a;self.c=-a/anchor**2
        self.beta=rational(self.p['beta_ra']);self.eps=rational(self.p['epsilon'])
        self.variance=[rational(s)**2 for s in self.p['acceleration_noise']]
        self.caps=list(map(rational,candidate['energy_caps']))
        if len(self.caps)!=2 or min(self.caps)<=0 or any(self.value(q,i)<self.beta for i,q in enumerate(self.caps)): raise ValueError('Sublevel energy cover is incomplete')

    def curve(self,x):
        h=lambda z:self.b*z+self.c*z**3;dh=lambda z:self.b+3*self.c*z*z
        span=x.h-x.l;v=[h(x.l),h(x.l)+span*dh(x.l)/3,h(x.h)-span*dh(x.h)/3,h(x.h)]
        return I(min(v),max(v)),self.b+3*self.c*x.square(),6*self.c*x

    def value(self,q,i): return sum(w*(q/self.scales[i])**j for j,w in enumerate(self.weights[i],1))

    def energy(self,box):
        x,y,vx,vy=[I(*b) for b in box];h,dh,_=self.curve(x);s=y-h
        def quadratic(p,v,k,gamma):
            q=lambda a,b:a*a+gamma*(b+k*a)**2
            clamp=lambda a,l,h:min(h,max(l,a))
            corners=[q(a,b) for a in (p.l,p.h) for b in (v.l,v.h)]
            values=corners+[q(a,clamp(-k*a,v.l,v.h)) for a in (p.l,p.h)]
            values += [q(clamp(-gamma*k*b/(1+gamma*k*k),p.l,p.h),b) for b in (v.l,v.h)]
            if p.l<=0<=p.h and v.l<=0<=v.h: values.append(F(0))
            return I(min(values),max(corners))
        return [quadratic(x,vx,self.k[0],self.gamma[0]),quadratic(s,vy-dh*vx,self.k[1],self.gamma[1])]

    def generator(self,box):
        q=[I(a*c,b*c) for (a,b),c in zip(box,self.caps)]
        low=sum(self.value(v.l,i) for i,v in enumerate(q))
        if low>=self.beta: return low,None,'V_lower >= beta_ra'
        rx,ry=[sqrt_upper(v.h) for v in q];h,hp,_=self.curve(I(-rx,rx))
        vx=rx*sqrt_upper(1/self.gamma[0]+self.k[0]**2)
        radii=[rx,h.absolute()+ry,vx,hp.absolute()*vx+ry*sqrt_upper(1/self.gamma[1]+self.k[1]**2)]
        if all(r<=min(-a,b) for r,(a,b) in zip(radii,self.goal)): return low,None,'contained in goal'
        noise=[self.gamma[0]*self.variance[0],self.gamma[1]*(self.variance[1]+hp.square().h*self.variance[0])]
        gv=F(0)
        for i,v in enumerate(q):
            for j,w in enumerate(self.weights[i],1):
                factor=I(F(j)/self.scales[i]*(v.l/self.scales[i])**(j-1),F(j)/self.scales[i]*(v.h/self.scales[i])**(j-1))
                pos=-2*self.k[i]*v+noise[i];vel=-2*self.d[i]*v+(2*j-1)*noise[i]
                gv+=w*(factor*I(min(pos.l,vel.l),max(pos.h,vel.h))).h
        return low,gv,'generator bound'

    def control_cover(self,budget=30000):
        rx=sqrt_upper(self.caps[0]);rw=sqrt_upper(self.caps[0]/self.gamma[0])
        pending=[([(-rx,rx),(-rw,rw)],0)];worst=[F(0),F(0)];leaves=0;checked=0
        cores=[F(19,20)*rational(v) for v in self.p['control_limit']]
        transverse=sqrt_upper((1/self.gamma[1]-self.k[1]**2)**2+(self.k[1]+self.d[1])**2/self.gamma[1])
        while pending:
            box,depth=pending.pop();checked+=1;x,wx=[I(*v) for v in box]
            qx=x.square()+self.gamma[0]*wx.square();remaining=self.beta-self.value(qx.l,0)
            if remaining<=0: leaves+=1;continue
            # Floating-point search proposes an upper root; the rational test certifies it.
            low=0.;high=float(self.caps[1]);target=float(remaining)
            for _ in range(40):
                mid=(low+high)/2
                value=sum(float(w)*(mid/float(self.scales[1]))**j for j,w in enumerate(self.weights[1],1))
                if value>=target: high=mid
                else: low=mid
            hi=rational(high*(1+1e-10))
            if self.value(hi,1)<remaining: hi=self.caps[1]
            ry=sqrt_upper(hi);h,hp,hpp=self.curve(x);vx=wx-self.k[0]*x
            ux=-(1/self.gamma[0]-self.k[0]**2)*x-(self.k[0]+self.d[0])*wx
            uy_center=hp*ux+hpp*vx.square()
            bounds=[ux.absolute(),uy_center.absolute()+transverse*ry]
            ty=ry*sqrt_upper(1/self.gamma[1]+self.k[1]**2)
            state=[x,h+I(-ry,ry),vx,hp*vx+I(-ty,ty)]
            inside=all(a<=v.l and v.h<=b for v,(a,b) in zip(state,self.domain))
            if inside and all(a<=b for a,b in zip(bounds,cores)):
                leaves+=1;worst=[max(a,b) for a,b in zip(worst,bounds)];continue
            if checked>=budget or depth>=32:
                return False,dict(cells=leaves,checked=checked,unresolved=len(pending)+1,raw_control_upper=list(map(float,bounds)),
                    state_box=[[float(v.l),float(v.h)] for v in state],domain_containment_passed=inside)
            axis=max(range(2),key=lambda j:float((box[j][1]-box[j][0])/(rx if j==0 else rw)))
            a,b=box[axis];mid=(a+b)/2;left=list(box);right=list(box);left[axis]=(a,mid);right[axis]=(mid,b)
            pending.extend([(right,depth+1),(left,depth+1)])
        return True,dict(cells=leaves,checked=checked,unresolved=0,raw_control_upper=list(map(float,worst)))


def replay(roots,history):
    cells=[list(b) for b in roots]
    for index,axis,cut in history:
        if not isinstance(index,int) or not 0<=index<len(cells) or not isinstance(axis,int) or not 0<=axis<len(cells[index]): raise ValueError('Invalid split index')
        cut=rational(cut);a,b=cells[index][axis]
        if not a<cut<b: raise ValueError('Invalid partition split')
        left=list(cells[index]);right=list(left);left[axis]=(a,cut);right[axis]=(cut,b);cells[index]=left;cells.append(right)
    return cells


def verify(candidate):
    proof=Proof(candidate);regions={}
    if any(r[0] not in ('initial','unsafe') for r in candidate['region_splits']): raise ValueError('Unknown region group')
    for name,roots in [('initial',[proof.initial]),('unsafe',proof.unsafe)]:
        boxes=replay(roots,[r[1:] for r in candidate['region_splits'] if r[0]==name])
        values=[sum(proof.value(q.h if name=='initial' else q.l,i) for i,q in enumerate(proof.energy(b))) for b in boxes]
        regions[name]=max(values) if name=='initial' else min(values)
    cells=replay([[(F(0),F(1))]*2],candidate['energy_splits']);rows=[];active=[]
    for b in cells:
        low,g,reason=proof.generator(b);passed=g is None or g<=-proof.eps
        if g is not None: active.append(g)
        rows.append(dict(energy_box=[[str(a*c),str(z*c)] for (a,z),c in zip(b,proof.caps)],V_lower=float(low),GV_upper=None if g is None else float(g),GV_upper_exact=None if g is None else str(g),passed=passed,reason=reason))
    controls,control_report=proof.control_cover()
    success=regions['initial']<=1 and regions['unsafe']>=proof.beta and controls and all(r['passed'] for r in rows)
    failures=[]
    if regions['initial']>1: failures.append('initial V upper > 1')
    if regions['unsafe']<proof.beta: failures.append('unsafe V lower < beta_ra')
    if not all(r['passed'] for r in rows): failures.append('active generator upper > -epsilon')
    if not controls: failures.append('sublevel domain containment / raw controller identity not proved')
    summary=dict(V_initial_upper=float(regions['initial']),V_unsafe_lower=float(regions['unsafe']),GV_active_upper=float(max(active)) if active else None,control_absolute_upper=proof.p['control_limit'])
    return dict(status='SAT' if success else 'UNKNOWN',candidate_sha256=fingerprint(candidate),summary=summary,
        arithmetic='Exact rational interval comparisons with rational square-root enclosures; decimal summaries are for display.',
        global_nonnegative=True,controller_smooth=True,boundary_coverage=True,initial_goal_disjoint=True,
        sublevel_contained_in_domain=controls,controls_verified=controls,control_cover=control_report,
        generator_identity_verified=controls,failed_conditions=failures,failed_generator_cells=sum(not r['passed'] for r in rows),
        training_stop=candidate.get('training_stop'),
        beta_ra=float(proof.beta),epsilon=float(proof.eps),reach_avoid_probability_lower=float(1-1/proof.beta) if success else None,
        domain=proof.p['domain'],energy_cells=len(rows),region_cells=2+len(proof.unsafe)-1+len(candidate['region_splits']),cells=rows)
