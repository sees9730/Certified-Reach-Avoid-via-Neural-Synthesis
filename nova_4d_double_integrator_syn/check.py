"""Independent exact-rational proof; uses only the Python standard library."""
import hashlib
import json
from fractions import Fraction as F


def rational(value): return F(str(value))
def fingerprint(candidate): return hashlib.sha256(json.dumps(candidate,sort_keys=True).encode()).hexdigest()
def contains(a,b): return all(lo<=x<=y<=hi for (lo,hi),(x,y) in zip(a,b))
def disjoint(a,b): return any(hi<x or y<lo for (lo,hi),(x,y) in zip(a,b))


def box(rows):
    if len(rows)!=4 or any(len(row)!=2 for row in rows): raise ValueError('Expected a four-dimensional box')
    result=[tuple(map(rational,row)) for row in rows]
    if any(lo>hi for lo,hi in result): raise ValueError('Reversed box')
    return result


def validate(problem):
    if problem['state_order']!=['px','py','vx','vy']: raise ValueError('Wrong state order')
    domain,initial,goal=[box(problem[key]) for key in ('domain','initial','goal')]
    unsafe=[box(b) for b in problem['unsafe']]
    if any(lo>=0 or hi<=0 for lo,hi in goal): raise ValueError('The goal must contain the origin in its interior')
    if not disjoint(initial,goal): raise ValueError('Initial and goal regions overlap')
    if any(not contains(domain,b) for b in [initial,goal,*unsafe]): raise ValueError('Region outside domain')
    if any(not disjoint(b,initial) or not disjoint(b,goal) for b in unsafe): raise ValueError('Unsafe overlaps initial or goal')
    for i in range(4):
        for endpoint in domain[i]:
            face=list(domain);face[i]=(endpoint,endpoint)
            if not any(contains(b,face) for b in unsafe): raise ValueError('Unsafe boxes must cover every complete domain face')
    for key in ('acceleration_noise','control_limit'):
        if len(problem[key])!=2 or any(rational(v)<=0 for v in problem[key]): raise ValueError('Two positive noise/control values required')
    if rational(problem['beta_ra'])<=1 or rational(problem['epsilon'])<=0: raise ValueError('Invalid thresholds')
    return domain,initial,goal,unsafe


def energy_range(region,k):
    """Convex pair quadratics: evaluate corners, edge stationary points, and origin."""
    low=high=F(0)
    clamp=lambda x,lo,hi:max(lo,min(hi,x))
    for i,gain in enumerate(k):
        pl,ph=region[i];vl,vh=region[i+2]
        q=lambda p,v:p*p+(v+gain*p)**2
        vertices=[q(p,v) for p in (pl,ph) for v in (vl,vh)]
        candidates=vertices+[q(p,clamp(-gain*p,vl,vh)) for p in (pl,ph)]
        candidates += [q(clamp(-gain*v/(1+gain*gain),pl,ph),v) for v in (vl,vh)]
        if pl<=0<=ph and vl<=0<=vh: candidates.append(F(0))
        low+=min(candidates);high+=max(vertices)
    return low,high


def verify(candidate):
    if candidate.get('format')=='curved_double_integrator_v2':
        try: from .curved_check import verify as curved_verify
        except ImportError: from curved_check import verify as curved_verify
        return curved_verify(candidate)
    if candidate.get('format')!='smooth_double_integrator_v1': raise ValueError('Unknown candidate format')
    p=candidate['problem'];domain,initial,goal,unsafe=validate(p);m=candidate['model']
    k,d,w=[list(map(rational,m[key])) for key in ('k','d','weights')];scale=rational(m['scale'])
    if len(k)!=2 or len(d)!=2 or not w or min(k+d)<=0 or min(w)<0 or scale<=0: raise ValueError('Invalid model')
    value=lambda q:sum(c*(q/scale)**j for j,c in enumerate(w,1))
    initial_upper=value(energy_range(initial,k)[1])
    unsafe_bounds=[value(energy_range(b,k)[0]) for b in unsafe]
    radii=[min(-lo,hi) for lo,hi in goal]
    q0=min([r*r for r in radii[:2]]+[radii[i+2]**2/(1+k[i]**2) for i in range(2)])
    q1=energy_range(domain,k)[1]
    cuts=list(map(rational,candidate['energy_partition']))
    if len(cuts)<2 or cuts[0]!=0 or cuts[-1]!=1 or any(a>=b for a,b in zip(cuts,cuts[1:])): raise ValueError('Partition has a gap, overlap, or missing endpoint')
    beta=rational(p['beta_ra']);eps=rational(p['epsilon']);decay=min(k+d)
    variance=[rational(v)**2 for v in p['acceleration_noise']]
    rows=[];active=[]
    for left,right in zip(cuts,cuts[1:]):
        lo=q0+(q1-q0)*left;hi=q0+(q1-q0)*right
        vl=value(lo)
        if vl>=beta:
            passed=True;gv=None;reason='V_lower >= beta_ra'
        else:
            gv=F(0)
            for j,c in enumerate(w,1):
                a=F(j)/scale*(lo/scale)**(j-1);b=F(j)/scale*(hi/scale)**(j-1)
                offset=sum(variance)+2*(j-1)*max(variance)
                c0=-2*decay*hi+offset;c1=-2*decay*lo+offset
                gv+=c*max(a*c0,a*c1,b*c0,b*c1)
            passed=gv<=-eps;reason='generator bound';active.append(gv)
        rows.append(dict(q_interval=[str(lo),str(hi)],V_lower=float(vl),GV_upper=float(gv) if gv is not None else None,
                         GV_upper_exact=str(gv) if gv is not None else None,passed=passed,reason=reason))
    radius=[max(abs(lo),abs(hi)) for lo,hi in domain]
    control=[(1+k[i]*d[i])*radius[i]+(k[i]+d[i])*radius[i+2] for i in range(2)]
    control_ok=all(bound<=rational(limit) for bound,limit in zip(control,p['control_limit']))
    nearest=[min(-lo,hi) for lo,hi in domain]
    outside_domain_q=min([r*r for r in nearest[:2]]+[nearest[i+2]**2/(1+k[i]**2) for i in range(2)])
    outside_domain_v=value(outside_domain_q)
    success=initial_upper<=1 and min(unsafe_bounds)>=beta and control_ok and outside_domain_v>=beta and all(r['passed'] for r in rows)
    summary=dict(V_initial_upper=float(initial_upper),V_unsafe_lower=float(min(unsafe_bounds)),
                 GV_active_upper=float(max(active)) if active else None,control_absolute_upper=list(map(float,control)))
    return dict(status='SAT' if success else 'UNKNOWN',candidate_sha256=fingerprint(candidate),summary=summary,
                arithmetic='Exact rational comparisons; decimal summary values are for display.',
                global_nonnegative=True,controller_smooth=True,boundary_coverage=True,initial_goal_disjoint=True,
                sublevel_contained_in_domain=outside_domain_v>=beta,outside_domain_V_lower=float(outside_domain_v),
                controls_verified=control_ok,beta_ra=float(beta),epsilon=float(eps),
                reach_avoid_probability_lower=float(1-1/beta) if success else None,
                domain=p['domain'],outside_goal_energy_lower=str(q0),domain_energy_upper=str(q1),
                unsafe_lower_bounds=list(map(float,unsafe_bounds)),energy_cells=len(rows),cells=rows)


if __name__=='__main__':
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('candidate',type=Path);args=parser.parse_args()
    report=verify(json.loads(args.candidate.read_text()))
    (args.candidate.parent/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cells'},indent=2))
