"""Independent, outward-rounded verification of the saved closed-loop pair.

Only mpmath and the Python standard library are used here. No Torch bounds,
point samples, optimizer state, or existing project implementation is trusted.
"""
import math
import hashlib
import json
from fractions import Fraction
from time import perf_counter

from mpmath import iv, mp


def lower(x): return mp.make_mpf(x._mpi_[0])
def upper(x): return mp.make_mpf(x._mpi_[1])
def decimal(x): return iv.mpf(str(x))


def tanh(x):
    f=lambda t:2/(1+iv.exp(-2*iv.mpf(t)))-1
    return iv.mpf([lower(f(lower(x))),upper(f(upper(x)))])


def fingerprint(artifact):
    return hashlib.sha256(json.dumps(artifact,sort_keys=True,allow_nan=False).encode()).hexdigest()


def quadratic(x, a, b):
    f=lambda t:a*t*t+b*t
    endpoints=[f(iv.mpf(lower(x))),f(iv.mpf(upper(x)))]
    vertex=iv.mpf(-b)/(2*a)
    if lower(x)<=upper(vertex) and upper(x)>=lower(vertex): endpoints.append(f(vertex))
    return iv.mpf([min(lower(t) for t in endpoints),max(upper(t) for t in endpoints)])


class Checker:
    def __init__(self, artifact):
        iv.dps=50;mp.dps=70
        if artifact.get('controller_type')!='smooth_affine_softmin': raise ValueError('Expected a smooth controller artifact; retrain this model version')
        self.p=artifact['problem']; self.m=artifact['model']
        shapes={'weights':(3,6),'alpha_weights':(10,),'center':(3,),
                'scale':(3,),'rates':(6,),'normalizers':(3,6),
                'action_lower':(3,),'action_upper':(3,),'temperature':()}
        def validate(value,shape):
            if shape:
                if not isinstance(value,list) or len(value)!=shape[0]: raise ValueError('Invalid parameter shape')
                for v in value: validate(v,shape[1:])
            elif type(value) not in (int,float) or not math.isfinite(value): raise ValueError('Nonfinite parameter')
        for key,shape in shapes.items(): validate(self.m[key],shape)
        if any(w<0 for row in self.m['weights'] for w in row): raise ValueError('Negative certificate weight')
        if any(v<=0 for v in self.m['scale']) or any(v<=0 for row in self.m['normalizers'] for v in row):
            raise ValueError('Nonpositive normalization')
        if len(self.p['control_limits'])!=3 or any(a>=b for a,b in self.p['control_limits']):
            raise ValueError('Invalid control limits')
        if self.m['temperature']<=0: raise ValueError('Smoothing temperature must be positive')
        if any(a>=b for a,b in zip(self.m['action_lower'],self.m['action_upper'])): raise ValueError('Invalid controller range')
        for action in [self.m['action_lower'],self.m['action_upper']]:
            for value,(lo,hi) in zip(action,self.p['control_limits']):
                if lower(iv.mpf(value))<upper(decimal(lo)) or upper(iv.mpf(value))>lower(decimal(hi)):
                    raise ValueError('Action outside the exact control limits')
        self.beta=decimal(self.p['beta_ra']);self.epsilon=decimal(self.p['epsilon'])
        if lower(self.beta)<=0 or lower(self.epsilon)<=0: raise ValueError('Nonpositive thresholds')
        self.cells=[self.box(self.p['domain'])]
        if self.cells[0][0][0]<=0: raise ValueError('XV-15 requires positive airspeed')
        self.goal=self.box(self.p['goal'])
        self.initial=self.box(self.p['initial'])
        self.unsafe=[self.box(b) for b in self.p['unsafe']]
        for b in [self.goal,self.initial,*self.unsafe]:
            if not self.contains(self.cells[0],b): raise ValueError('Region outside domain')
        for index,axis,cut in artifact['partition']:
            if type(index) is not int or not 0<=index<len(self.cells) or type(axis) is not int or not 0<=axis<3:
                raise ValueError('Invalid partition witness')
            cut=Fraction(cut); parent=self.cells[index]
            if not parent[axis][0]<cut<parent[axis][1]: raise ValueError('Invalid partition cut')
            left,right=list(parent),list(parent)
            left[axis]=(parent[axis][0],cut);right[axis]=(cut,parent[axis][1])
            self.cells[index]=left;self.cells.append(right)
        self.witnesses=artifact.get('witness_actions')
        if self.witnesses is not None and (len(self.witnesses)!=len(self.cells) or any(
                type(i) is not int or not 0<=i<4 for i in self.witnesses)):
            raise ValueError('Invalid action witnesses')

    @staticmethod
    def box(rows):
        if len(rows)!=3 or any(len(row)!=2 for row in rows): raise ValueError('Expected a 3D box')
        result=[tuple(Fraction(str(v)) for v in row) for row in rows]
        if any(a>b for a,b in result): raise ValueError('Reversed box')
        return result

    @staticmethod
    def contains(outer,inner): return all(a<=c<=d<=b for (a,b),(c,d) in zip(outer,inner))

    def evaluate(self, box, generator=True, action=None):
        def rational(v): return iv.mpf(v.numerator)/v.denominator
        x=[iv.mpf([lower(rational(a)),upper(rational(b))]) for a,b in box]
        # Saved model floats denote exact binary64 coefficients, not rounded decimals.
        m=self.m;z=[(x[i]-iv.mpf(m['center'][i]))/iv.mpf(m['scale'][i]) for i in range(3)]
        value=iv.mpf(0);grad=[iv.mpf(0) for _ in x];hess=[iv.mpf(0) for _ in x]
        for i in range(3):
            for j,rate in enumerate(m['rates']):
                if m['weights'][i][j]==0: continue
                weight=iv.mpf(m['weights'][i][j])/iv.mpf(m['normalizers'][i][j])
                e=iv.exp(iv.mpf(rate)*z[i]);a=iv.mpf(rate)/iv.mpf(m['scale'][i])
                value+=weight*(e-1)**2
                if generator:
                    grad[i]+=weight*2*a*quadratic(e,1,-1)
                    hess[i]+=weight*2*a*a*quadratic(e,2,-1)
        if not generator: return value,None
        phi=[iv.mpf(1)]+z+[t**2 for t in z]+[z[i]*z[j] for i,j in ((0,1),(0,2),(1,2))]
        raw=sum(iv.mpf(w)*h for w,h in zip(m['alpha_weights'],phi))
        lo=list(map(iv.mpf,m['action_lower']));hi=list(map(iv.mpf,m['action_upper']))
        alpha=(lo[1]+hi[1])/2+(hi[1]-lo[1])/2*tanh(raw)
        corners=[(t,d) for t in [lo[0],hi[0]] for d in [lo[2],hi[2]]]
        generators=[]
        for thrust,delta in corners if action is None else [corners[action]]:
            v,gamma,beta=x
            deg=iv.pi/180;r=beta/90;g=decimal(self.p['gravity'])
            cl=(decimal('0.0849')*alpha+decimal('0.3482'))*(1-r)+(decimal('0.0646')*alpha+decimal('0.2709'))*r
            cd=(decimal('0.00042143')*alpha**2+decimal('0.0030')*alpha+decimal('0.0218'))*(1-r)+(decimal('0.000473216')*alpha**2+decimal('0.00343')*alpha+decimal('0.2165'))*r
            qsm=decimal('0.5')*decimal(self.p['density'])*decimal(self.p['wing_area'])/decimal(self.p['mass'])*v**2
            f=[g*thrust*iv.cos((alpha+beta)*deg)-qsm*cd-g*iv.sin(gamma*deg),
               (g*thrust*iv.sin((alpha+beta)*deg)+qsm*cl-g*iv.cos(gamma*deg))/v/deg,delta]
            generators.append(sum(grad[i]*f[i]+hess[i]*decimal(self.p['diffusion'][i])**2/2 for i in range(3)))
        penalty=2*iv.mpf(m['temperature'])*iv.ln(2)
        bound=iv.mpf([min(lower(t) for t in generators),min(upper(t) for t in generators)])
        # Each affine-control soft-min adds at most temperature*log(2).
        return value,iv.mpf([lower(bound),upper(bound+penalty)])


def verify(artifact):
    start=perf_counter();engine=Checker(artifact);rows=[]
    for group,boxes in [('initial',[engine.initial]),('unsafe',engine.unsafe),('generator',engine.cells)]:
        for index,box in enumerate(boxes):
            v,_=engine.evaluate(box,False);gv=None;reason='bound'
            if group=='initial': ok=upper(v)<=1
            elif group=='unsafe': ok=lower(v)>=upper(engine.beta)
            elif engine.contains(engine.goal,box): ok=True;reason='goal'
            elif lower(v)>=upper(engine.beta): ok=True;reason='V_lower >= beta_ra'
            else:
                witness=engine.witnesses[index] if engine.witnesses is not None else None
                _,gv=engine.evaluate(box,action=witness)
                ok=upper(gv)<=lower(-engine.epsilon)
            rows.append(dict(region=group,index=index,box=[[float(a),float(b)] for a,b in box],
                             V_lower=float(lower(v)),V_upper=float(upper(v)),
                             GV_upper=float(upper(gv)) if gv is not None else None,passed=bool(ok),reason=reason))
            if group=='generator' and index and index%2000==0:
                print(f'Independent interval verification: {index}/{len(boxes)} cells',flush=True)
    passed=all(r['passed'] for r in rows)
    active=[r['GV_upper'] for r in rows if r['GV_upper'] is not None]
    return dict(status='SAT' if passed else 'UNKNOWN',candidate_sha256=fingerprint(artifact),global_nonnegative=True,
                domain=engine.p['domain'],state_units=engine.p['state_units'],
                beta_ra=engine.p['beta_ra'],epsilon=engine.p['epsilon'],
                controls_verified=True,policy='smooth tanh feedback with a learned quadratic angle channel',
                controller_smooth=True,smoothing_penalty_upper=float(upper(2*iv.mpf(engine.m['temperature'])*iv.ln(2))),precision_decimal_digits=50,
                summary_values='Rounded decimals for display; acceptance used outward interval endpoints.',
                summary=dict(V_initial_upper=rows[0]['V_upper'],V_unsafe_lower=min(r['V_lower'] for r in rows if r['region']=='unsafe'),
                             GV_active_upper=max(active) if active else None),
                failed_cells=sum(not r['passed'] for r in rows),generator_cells=len(engine.cells),active_generator_cells=len(active),
                seconds=perf_counter()-start,cells=rows)


if __name__=='__main__':
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate',type=Path)
    args=parser.parse_args()
    report=verify(json.loads(args.candidate.read_text()))
    (args.candidate.parent/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cells'},indent=2))
