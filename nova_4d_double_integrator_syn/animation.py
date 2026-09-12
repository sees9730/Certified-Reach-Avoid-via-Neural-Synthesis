"""Export synchronized rollout/control playback as one offline HTML file."""
import json
from pathlib import Path
import numpy as np

try:
    from .runtime import Controller,rollout
    from .check import fingerprint
except ImportError:
    from runtime import Controller,rollout
    from check import fingerprint


def animate(output, count=5, seed=0, dt=.002, seconds=15):
    if count<1 or seed<0 or not np.isfinite([dt,seconds]).all() or min(dt,seconds)<=0:
        raise ValueError('Positive rollout count, step and horizon, and nonnegative seed required')
    output=Path(output);candidate=json.loads((output/'candidate.json').read_text())
    controller=Controller(candidate);digest=fingerprint(candidate)
    report=json.loads((output/'verification.json').read_text()) if (output/'verification.json').exists() else {}
    status=report.get('status','Unverified') if report.get('candidate_sha256')==digest else 'Unverified'
    runs=[rollout(controller,seed=seed+i,dt=dt,seconds=seconds) for i in range(count)]
    data=dict(candidate_sha256=digest,verification=status,problem=candidate['problem'],
              method='Euler-Maruyama; illustrative simulation; events checked at integration steps',runs=runs)
    (output/'rollouts.json').write_text(json.dumps(data,allow_nan=False)+'\n')
    # Escape '<' so even a user-supplied task name cannot close the script element.
    payload=json.dumps(data,allow_nan=False).replace('<','\\u003c')
    template=(Path(__file__).parent/'rollout_player.html').read_text()
    path=output/'rollouts_animation.html'
    path.write_text(template.replace('__ROLLOUT_DATA__',payload))
    print('Rollout/control animation:',path,flush=True)
    return runs
