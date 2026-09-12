"""Bound training with adaptive boxes. Run: python nova_5d_gbm_veri/main.py train."""
import argparse
import ast
import csv
import json
from fractions import Fraction as Q
from pathlib import Path
from time import perf_counter

import sympy as sp


def expression(text, symbols):
    """Read polynomial arithmetic only; never evaluate Python from the input."""
    text = str(text)
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return sp.Rational(ast.get_source_segment(text, node))
        if isinstance(node, ast.Name) and node.id in symbols:
            return symbols[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        if isinstance(node, ast.BinOp):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): return a + b
            if isinstance(node.op, ast.Sub): return a - b
            if isinstance(node.op, ast.Mult): return a * b
            if isinstance(node.op, ast.Div) and b.is_Rational and b != 0: return a / b
            if isinstance(node.op, ast.Pow) and b.is_Integer and b >= 0: return a ** int(b)
        raise ValueError("Use polynomial expressions in x1..xn with rational constants")
    return visit(ast.parse(text, mode="eval").body)


def box(rows):
    result = tuple((Q(str(lo)), Q(str(hi))) for lo, hi in rows)
    if not result or any(lo > hi for lo, hi in result): raise ValueError("Invalid box")
    return result


def contains(outer, inner):
    return all(a <= c <= d <= b for (a, b), (c, d) in zip(outer, inner))


class Bounds:
    """V=sum w_j phi_j, w_j>=0. phi_j are nonnegative polynomial features."""
    def __init__(self, problem, degree=8):
        self.problem, self.degree = problem, degree
        self.domain = box(problem["domain"])
        self.n = len(self.domain)
        if type(degree) is not int or degree < 1: raise ValueError("degree must be positive")
        if problem.get("dimension", self.n) != self.n: raise ValueError("Dimension mismatch")
        self.beta, self.epsilon = Q(str(problem["beta_ra"])), Q(str(problem["epsilon"]))
        if self.beta <= 0 or self.epsilon <= 0: raise ValueError("Thresholds must be positive")
        self.goal = [box(b) for b in problem["goal"]]
        self.cells = {key: [box(b) for b in problem[key]] for key in ("initial", "unsafe")}
        self.cells["generator"] = [self.domain]
        for regions in (*self.cells.values(), self.goal):
            if not regions or any(len(b) != self.n or not contains(self.domain, b) for b in regions):
                raise ValueError("All region boxes must lie inside the stated domain")
        self.center = tuple((lo+hi)/2 for lo, hi in self.goal[0])
        self.scale = tuple((hi-lo)/2 for lo, hi in self.domain)
        if min(self.scale) <= 0: raise ValueError("Domain widths must be positive")
        z = sp.symbols(f"z1:{self.n+1}")
        variables = {f"x{i+1}": self.center[i] + self.scale[i]*z[i] for i in range(self.n)}
        drift = problem["drift"]
        diffusion = problem["diffusion"]
        if len(drift) != self.n or len(diffusion) != self.n or not diffusion[0] or any(
                len(row) != len(diffusion[0]) for row in diffusion): raise ValueError("SDE dimensions differ")
        f = sp.Matrix([expression(v, variables)/s for v, s in zip(drift, self.scale)])
        g = sp.Matrix([[expression(v, variables)/self.scale[i] for v in row]
                       for i, row in enumerate(diffusion)])
        covariance = g*g.T
        def generator(h):
            return sp.expand(sum(sp.diff(h, z[i])*f[i] for i in range(self.n)) +
                sum(covariance[i,j]*sp.diff(h, z[i], z[j])/2 for i in range(self.n) for j in range(self.n)))
        def gamma(h):
            return sp.expand(sum(sum(sp.diff(h, z[i])*g[i,j] for i in range(self.n))**2
                                 for j in range(g.cols)))
        # Normalize features from the input geometry, not from a known certificate.
        radius = max(sum(max(((a-c)/s)**2, ((b-c)/s)**2) for (a,b),c,s in
                         zip(cell,self.center,self.scale)) for cell in self.cells["initial"]) or Q(1)
        q = sum(v*v for v in z) / radius
        gq, gamma_q = generator(q), gamma(q)
        features = [q**k for k in range(1, degree+1)]
        generators = [sp.expand(k*q**(k-1)*gq + (k*(k-1)*q**(k-2)*gamma_q/2 if k > 1 else 0))
                      for k in range(1, degree+1)]
        # Coordinate and mixed squares also permit nonradial learned functions.
        forms = list(z) + [z[i]+sign*z[j] for i in range(self.n) for j in range(i+1,self.n) for sign in (-1,1)]
        features += [h*h for h in forms]
        generators += [generator(h*h) for h in forms]
        self.names = [str(h) for h in features]
        def terms(h):
            return [(powers, Q(int(c.p), int(c.q))) for powers,c in sp.Poly(h,*z).terms() if c]
        self.polynomials = [(terms(h),terms(gh)) for h,gh in zip(features,generators)]
        self.cache, self.history = {}, []

    def inside_goal(self, cell):
        return any(contains(goal,cell) for goal in self.goal)

    def bounds(self, cell):
        """Exact interval bounds of every feature and its generator on one cell."""
        if cell in self.cache: return self.cache[cell]
        intervals = [((a-c)/s,(b-c)/s) for (a,b),c,s in zip(cell,self.center,self.scale)]
        powers = {}
        def interval(terms):
            lower = upper = Q(0)
            for exponents, coefficient in terms:
                lo = hi = coefficient
                for i,k in enumerate(exponents):
                    if not k: continue
                    if (i,k) not in powers:
                        a,b = intervals[i]
                        powers[i,k] = ((a**k,b**k) if k%2 else
                            (Q(0) if a <= 0 <= b else min(a**k,b**k), max(a**k,b**k)))
                    a,b = powers[i,k]
                    values = (lo*a,lo*b,hi*a,hi*b)
                    lo,hi = min(values),max(values)
                lower += lo
                upper += hi
            return lower,upper
        result = [(interval(h),interval(gh)) for h,gh in self.polynomials]
        self.cache[cell] = result
        return result

    def split(self, group, index, axis, cut):
        parent = self.cells[group][index]
        if type(axis) is not int or not 0 <= axis < self.n or not parent[axis][0] < cut < parent[axis][1]:
            raise ValueError("Invalid partition split")
        left,right = list(parent),list(parent)
        left[axis],right[axis] = (parent[axis][0],cut),(cut,parent[axis][1])
        self.cells[group][index] = tuple(left)
        self.cells[group].append(tuple(right))
        self.history.append([group,index,axis,str(cut)])

    def replay(self, history):
        for group,index,axis,cut in history:
            if group not in self.cells or type(index) is not int or not 0 <= index < len(self.cells[group]):
                raise ValueError("Invalid partition history")
            self.split(group,index,axis,Q(cut))


def verify(engine, weights):
    """Whole-cell proof. Replayed splits guarantee coverage; no sampling acceptance."""
    w = tuple(Q(v) for v in weights)
    if len(w) != len(engine.names) or min(w) < 0: raise ValueError("Invalid certificate coefficients")
    rows, passed = [], True
    for group,cells in engine.cells.items():
        for cell in cells:
            b = engine.bounds(cell)
            lower = sum(v*h[0][0] for v,h in zip(w,b))
            upper = sum(v*h[0][1] for v,h in zip(w,b))
            gv = sum(v*h[1][1] for v,h in zip(w,b))
            reason = "bound"
            if group == "initial": ok = upper <= 1
            elif group == "unsafe": ok = lower >= engine.beta
            elif engine.inside_goal(cell): ok,reason = True,"goal"
            elif lower >= engine.beta: ok,reason = True,"V_lower >= beta_ra"
            else: ok = gv <= -engine.epsilon
            passed &= ok
            rows.append(dict(region=group, box=[[str(a),str(b)] for a,b in cell],
                             V_lower=str(lower),V_upper=str(upper),GV_upper=str(gv),passed=bool(ok),reason=reason))
    active = [Q(r["GV_upper"]) for r in rows if r["region"] == "generator" and r["reason"] == "bound"]
    summary = {"V_initial_upper": str(max(Q(r["V_upper"]) for r in rows if r["region"]=="initial")),
               "V_unsafe_lower": str(min(Q(r["V_lower"]) for r in rows if r["region"]=="unsafe")),
               "GV_active_upper": str(max(active)) if active else None}
    return dict(status="SAT" if passed else "UNKNOWN",global_nonnegative=True,
                domain=engine.problem["domain"],summary=summary,
                cell_counts={k:len(v) for k,v in engine.cells.items()},cells=rows)


def train(engine, epochs=5000, lr=0.01, refine_every=25, batch=4, max_cells=4096, seconds=120):
    """Projected Adam on the bound loss, interleaved with scored cell splitting."""
    import torch
    if max_cells < sum(map(len,engine.cells.values())): raise ValueError("max_cells is smaller than the initial partition")
    torch.set_num_threads(1)
    w = torch.nn.Parameter(torch.full((len(engine.names),),0.005,dtype=torch.float64))
    optimizer = torch.optim.Adam([w],lr=lr)
    started, log = perf_counter(), []
    def matrices():
        result = {}
        for group,cells in engine.cells.items():
            b = [engine.bounds(cell) for cell in cells]
            result[group] = tuple(torch.tensor([[float(h[i][j]) for h in row] for row in b],dtype=torch.float64)
                                  for i,j in ((0,0),(0,1),(1,1)))
        return result
    data = matrices()
    beta,epsilon = float(engine.beta),float(engine.epsilon)
    def objective():
        _,iu,_ = data["initial"]
        ul,_,_ = data["unsafe"]
        gl,_,gu = data["generator"]
        outside = torch.tensor([not engine.inside_goal(c) for c in engine.cells["generator"]])
        active = outside & (gl@w < beta)
        return (torch.relu(iu@w-0.99).sum(), torch.relu(1.01*beta-ul@w).sum()/beta,
                torch.relu(gu@w+1.1*epsilon)[active].sum()/epsilon)
    report = None
    for epoch in range(epochs+1):
        losses = objective()
        loss = sum(losses)
        if epoch%refine_every == 0 or float(loss.detach()) == 0 or epoch==epochs:
            report = verify(engine,[Q(float(v)) for v in w.detach()])
            elapsed = perf_counter()-started
            record = dict(epoch=epoch,seconds=elapsed,loss=float(loss.detach()),
                          loss_initial=float(losses[0].detach()),loss_unsafe=float(losses[1].detach()),
                          loss_GV=float(losses[2].detach()),**{k+"_cells":v for k,v in report["cell_counts"].items()})
            log.append(record)
            print(f"epoch={epoch:4d} loss={float(loss.detach()):.4g} cells={report['cell_counts']} {report['status']}",flush=True)
            if report["status"] == "SAT": break
            if elapsed >= seconds or epoch==epochs: break
            candidates = []
            for group,cells in engine.cells.items():
                for index,cell in enumerate(cells):
                    lo,hi,gv = (float(a[index]@w.detach()) for a in data[group])
                    score = hi-1 if group=="initial" else (beta-lo)/beta if group=="unsafe" else (
                        (gv+epsilon)/epsilon if lo < beta and not engine.inside_goal(cell) else 0)
                    if score > 0 and any(a < b for a,b in cell): candidates.append((score,group,index))
            room = max_cells-sum(len(c) for c in engine.cells.values())
            for _,group,index in sorted(candidates,reverse=True)[:max(0,min(batch,room))]:
                cell = engine.cells[group][index]
                cuts = [(i,face) for goal in engine.goal if group=="generator" and all(
                    max(a,c) <= min(b,d) for (a,b),(c,d) in zip(cell,goal))
                    for i in range(engine.n) for face in goal[i] if cell[i][0] < face < cell[i][1]]
                axis,cut = cuts[0] if cuts else (None,None)
                if axis is None:
                    axis = max(range(engine.n),key=lambda i:(cell[i][1]-cell[i][0])/engine.scale[i])
                    cut = sum(cell[axis])/2
                engine.split(group,index,axis,cut)
            data = matrices()
            loss = sum(objective())  # Train on the refined boxes, even when refine_every=1.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad(): w.clamp_(min=0)
    return [str(Q(float(v))) for v in w.detach()],report,log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("train","verify"),nargs="?",default="train")
    parser.add_argument("--problem",type=Path,default=Path(__file__).with_name("5d_gbm.json"))
    parser.add_argument("--output",type=Path,default=Path(__file__).with_name("results"))
    parser.add_argument("--degree",type=int,default=8,help="Include radial powers 1..degree and coordinate/mixed squares")
    parser.add_argument("--epochs",type=int,default=5000)
    parser.add_argument("--lr",type=float,default=0.01)
    parser.add_argument("--refine-every",type=int,default=25)
    parser.add_argument("--batch",type=int,default=4)
    parser.add_argument("--max-cells",type=int,default=4096)
    parser.add_argument("--seconds",type=float,default=120)
    parser.add_argument("--no-plots",action="store_true",help="Skip plots after verification")
    parser.add_argument("--axes",nargs="+",help="Plot coordinates, e.g. x1 x3")
    parser.add_argument("--slice",nargs="*",metavar="x3=50",help="Fix unplotted coordinates; otherwise show goal/initial/unsafe slices")
    parser.add_argument("--points",type=int,default=201,help="Plot samples per axis")
    args = parser.parse_args()
    started = perf_counter()
    try:
        problem = json.loads(args.problem.read_text())
        if args.command=="verify":
            artifact = json.loads((args.output/"certificate.json").read_text())
            if artifact["problem"] != problem: raise ValueError("Certificate problem does not match requested input")
            engine = Bounds(problem,artifact["degree"])
            if artifact["features"] != engine.names or artifact["center"] != list(map(str,engine.center)) or artifact["scale"] != list(map(str,engine.scale)):
                raise ValueError("Saved function representation does not match its defining problem/degree")
            engine.replay(artifact["splits"])
            report = verify(engine,artifact["weights"])
        else:
            if min(args.epochs,args.refine_every,args.batch,args.max_cells,args.seconds,args.lr) <= 0:
                raise ValueError("Training budgets and learning rate must be positive")
            engine = Bounds(problem,args.degree)
            weights,report,log = train(engine,args.epochs,args.lr,args.refine_every,args.batch,args.max_cells,args.seconds)
            args.output.mkdir(parents=True,exist_ok=True)
            artifact = dict(status=report["status"],problem=problem,degree=args.degree,weights=weights,
                            training_seconds=perf_counter()-started,iterations=log[-1]["epoch"],
                            center=list(map(str,engine.center)),scale=list(map(str,engine.scale)),
                            features=engine.names,splits=engine.history)
            (args.output/"certificate.json").write_text(json.dumps(artifact,indent=2)+"\n")
            with (args.output/"training.csv").open("w") as f:
                writer = csv.DictWriter(f,fieldnames=log[0].keys()); writer.writeheader(); writer.writerows(log)
        report["total_seconds"] = perf_counter()-started
        (args.output/"verification.json").write_text(json.dumps(report,indent=2)+"\n")
        print(f"{report['status']} in {report['total_seconds']:.3f}s; cells={report['cell_counts']}")
        print("V >= 0 everywhere: nonnegative weights on nonnegative features")
        for key,value in report["summary"].items():
            print(f"{key}: {float(Q(value)):.10g}" if value is not None else f"{key}: no active cells")
        print(f"Exact bounds and domain: {args.output/'verification.json'}")
        if args.command=="verify" and not args.no_plots:
            if __package__: from .plotting import plot
            else: from plotting import plot
            print(f"V/GV plots: {plot(engine,artifact['weights'],args.output,report['status'],args.axes,args.slice,args.points)}")
        return 0 if report["status"]=="SAT" else 1
    except (ValueError,KeyError,TypeError,OSError,ZeroDivisionError,SyntaxError) as exc:
        print(f"ERROR: {exc}"); return 2


if __name__=="__main__": raise SystemExit(main())
