"""Pointwise V/GV slices, confined to the input domain; no proof decisions here."""
from fractions import Fraction as Q
import json
from pathlib import Path

import numpy as np


def sample(engine, weights, axes, anchor, points):
    if len(axes) not in (1,2) or len(set(axes)) != len(axes) or any(i<0 or i>=engine.n for i in axes):
        raise ValueError("Choose one or two distinct coordinates in x1..xn")
    if points < 3: raise ValueError("Plot resolution must be >= 3")
    if any(not a<=v<=b for v,(a,b) in zip(anchor,engine.domain)):
        raise ValueError("Slice coordinates must be inside the problem domain")
    vectors = [np.linspace(float(engine.domain[i][0]),float(engine.domain[i][1]),points) for i in axes]
    grid = np.meshgrid(*vectors,indexing="xy") if len(axes)==2 else vectors
    z = [(v-float(engine.center[i]))/float(engine.scale[i]) for i,v in zip(axes,grid)]
    fixed = [(v-c)/s for v,c,s in zip(anchor,engine.center,engine.scale)]
    fields = []
    for field in (0,1):
        # Substitute fixed coordinates exactly before evaluating the smaller slice polynomial.
        terms = {}
        for w,pair in zip(weights,engine.polynomials):
            w = Q(w)
            if not w: continue
            for powers,c in pair[field]:
                c *= w
                for i in range(engine.n):
                    if i not in axes: c *= fixed[i]**powers[i]
                key = tuple(powers[i] for i in axes)
                terms[key] = terms.get(key,Q(0))+c
        values = np.zeros_like(grid[0])
        for powers,c in terms.items():
            term = float(c)
            for v,k in zip(z,powers): term = term*v**k
            values += term
        if not np.isfinite(values).all(): raise ValueError("Plot evaluation overflowed")
        fields.append(values)
    return grid,fields


def plot(engine, weights, output, status, axes=None, fixed=None, points=201):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import SymLogNorm
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from matplotlib import patheffects

    def index(name):
        if not name.startswith("x") or not name[1:].isdigit() or not 1<=int(name[1:])<=engine.n:
            raise ValueError("Coordinates must be x1..xn")
        return int(name[1:])-1
    axes = tuple(map(index,axes)) if axes else tuple(range(min(2,engine.n)))
    slices = []
    if fixed is not None:
        anchor,seen = list(engine.center),set()
        for item in fixed:
            name,sep,value = item.partition("=")
            if not sep: raise ValueError("Slice entries must be like x3=50")
            i = index(name)
            if i in axes or i in seen: raise ValueError("A fixed coordinate cannot be plotted or repeated")
            anchor[i] = Q(value); seen.add(i)
        slices.append(("custom",anchor))
    else:
        # Separate slices make the disjoint high-dimensional regions visible honestly.
        seen = set()
        for name in ("goal","initial","unsafe"):
            for j,region in enumerate(engine.problem[name]):
                anchor = [sum(Q(str(v)) for v in row)/2 for row in region]
                key = tuple(anchor[i] for i in range(engine.n) if i not in axes)
                if key not in seen: slices.append((f"{name}_{j+1}" if key else "domain",anchor)); seen.add(key)
    output = Path(output)/"plots"; output.mkdir(parents=True,exist_ok=True)
    beta,eps = float(engine.beta),float(engine.epsilon)
    colors = dict(initial="#178d3c",goal="#a15acc",unsafe="#d64226")
    metadata = []
    with PdfPages(output/"V_GV.pdf") as pdf:
        for name,anchor in slices:
            grid,(V,GV) = sample(engine,weights,axes,anchor,points)
            figure = Figure(figsize=(12,6)); FigureCanvasAgg(figure)
            panels = figure.subplots(1,2)
            visible = [(kind,b) for kind in colors for b in engine.problem[kind] if all(
                i in axes or Q(str(a))<=anchor[i]<=Q(str(c)) for i,(a,c) in enumerate(b))]
            crossings = float(V.min()) < beta < float(V.max())
            for ax,values,label in zip(panels,(V,GV),("V","GV")):
                handles = []
                thresholds = [(V,1,"white","--","V = 1"),(V,beta,"#ffb000","-",f"V = beta_ra = {beta:g}")]
                if label=="GV": thresholds.append((GV,-eps,"black",":",f"GV = {-eps:g}"))
                if len(axes)==2:
                    span = max(float(np.abs(values).max()),eps)
                    norm = (SymLogNorm(linthresh=1,vmin=min(0,float(V.min())),vmax=max(beta,float(V.max())))
                            if label=="V" else SymLogNorm(linthresh=eps,vmin=-span,vmax=span))
                    mesh = ax.pcolormesh(*grid,values,shading="auto",norm=norm,
                                         cmap="viridis" if label=="V" else "RdBu_r",rasterized=True)
                    figure.colorbar(mesh,ax=ax,pad=0.05).set_label(f"{label} (symlog scale)")
                    if float(values.min()) < float(values.max()):
                        levels = norm.inverse(np.linspace(float(norm(values.min())),float(norm(values.max())),7)[1:-1])
                        lines = ax.contour(*grid,values,levels=levels,colors="white",linewidths=0.6,alpha=0.6)
                        ax.clabel(lines,fmt=lambda v:f"{v:.2g}",fontsize=7)
                    for field,level,color,style,text in thresholds:
                        if float(field.min()) < level < float(field.max()):
                            contour = ax.contour(*grid,field,levels=[level],colors=[color],linestyles=style,linewidths=1.6)
                            contour.set_path_effects([patheffects.withStroke(linewidth=2.6,foreground="#444444")])
                            handles.append(Line2D([],[],color=color,ls=style,label=text))
                    ax.set_ylim(*map(float,engine.domain[axes[1]])); ax.set_ylabel(f"x{axes[1]+1}")
                    ax.set_aspect("equal",adjustable="box")
                else:
                    ax.plot(grid[0],values,color="#2766a7")
                    ax.set_yscale("symlog",linthresh=1 if label=="V" else eps)
                    for level in ((1,beta) if label=="V" else (-eps,0)):
                        ax.axhline(level,ls="--",color="#aa6500",label=f"{label} = {level:g}")
                    handles.extend(ax.get_legend_handles_labels()[0])
                    ax.set_ylabel(label); ax.grid(alpha=0.2)
                for kind,region in visible:
                    i = axes[0]; lo,hi = map(lambda v:float(Q(str(v))),region[i])
                    if len(axes)==2:
                        bottom,top = map(lambda v:float(Q(str(v))),region[axes[1]])
                        rect = Rectangle((lo,bottom),hi-lo,top-bottom,fill=False,ec=colors[kind],lw=2,zorder=5)
                        rect.set_path_effects([patheffects.withStroke(linewidth=3.4,foreground="white")]); ax.add_patch(rect)
                    else: ax.axvspan(lo,hi,color=colors[kind],alpha=0.15)
                handles += [Line2D([],[],color=colors[k],lw=2,label=f"{k} region") for k in colors if any(t==k for t,_ in visible)]
                if handles: ax.legend(handles=handles,fontsize=8,loc="upper right")
                ax.set_xlim(*map(float,engine.domain[axes[0]])); ax.set_xlabel(f"x{axes[0]+1}"); ax.set_title(label)
            fixed_text = ", ".join(f"x{i+1}={float(v):g}" for i,v in enumerate(anchor) if i not in axes) or "full domain"
            figure.suptitle(f"{engine.problem.get('name','SDE')} | {status} | {fixed_text}")
            note = "" if crossings else f"No V={beta:g} crossing on this sampled slice. "
            figure.text(0.5,0.025,note+"Pointwise plots; exact proof bounds are in verification.json.",ha="center",fontsize=9)
            figure.tight_layout(rect=(0,0.07,1,0.94))
            pdf.savefig(figure); figure.savefig(output/f"{name}.png",dpi=160); figure.clear()
            metadata.append(dict(name=name,axes=[f"x{i+1}" for i in axes],
                fixed={f"x{i+1}":str(v) for i,v in enumerate(anchor) if i not in axes},domain=engine.problem["domain"],
                regions=sorted({kind for kind,_ in visible}),
                V_range=[float(V.min()),float(V.max())],GV_range=[float(GV.min()),float(GV.max())],beta_crossing=crossings))
    (output/"slices.json").write_text(json.dumps(metadata,indent=2)+"\n")
    return output/"V_GV.pdf"
