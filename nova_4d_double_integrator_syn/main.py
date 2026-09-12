"""Train or verify: python -m nova_4d_double_integrator_syn train."""
import argparse
import csv
import json
import math
from pathlib import Path

try:
    from .check import validate,verify,fingerprint
except ImportError:
    from check import validate,verify,fingerprint


def main():
    here=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['train','verify','plot','animate'])
    parser.add_argument('--problem',type=Path,default=here/'problem.json')
    parser.add_argument('--search',type=Path,default=here/'search.json')
    parser.add_argument('--output',type=Path,default=here/'results')
    parser.add_argument('--steps',type=int,default=1000000)
    parser.add_argument('--seconds',type=float,default=1800)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--no-plots',action='store_true')
    parser.add_argument('--rollouts',type=int,default=5)
    parser.add_argument('--rollout-seed',type=int,default=0)
    parser.add_argument('--rollout-dt',type=float,default=.002)
    parser.add_argument('--rollout-seconds',type=float,default=15)
    args=parser.parse_args()
    if args.steps<=0 or args.seconds<=0 or not math.isfinite(args.seconds): parser.error('Positive finite budgets required')
    if args.rollouts<1 or args.rollout_seed<0 or min(args.rollout_dt,args.rollout_seconds)<=0 or not all(map(math.isfinite,[args.rollout_dt,args.rollout_seconds])):
        parser.error('Positive rollout count, step and horizon, and nonnegative rollout seed required')
    if args.command=='train':
        try: from .bound_training import learn
        except ImportError: from bound_training import learn
        problem=json.loads(args.problem.read_text());validate(problem)
        candidate,log=learn(problem,json.loads(args.search.read_text()),steps=args.steps,seconds=args.seconds,seed=args.seed)
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'candidate.json').write_text(json.dumps(candidate,indent=2)+'\n')
        with (args.output/'training.csv').open('w') as stream:
            writer=csv.DictWriter(stream,log[0]);writer.writeheader();writer.writerows(log)
    candidate=json.loads((args.output/'candidate.json').read_text())
    if candidate['problem']!=json.loads(args.problem.read_text()):
        parser.error('The saved candidate uses a different problem. Train this problem, or pass its matching --problem file.')
    if args.command in ('train','verify'):
        try: report=verify(candidate)
        except (ValueError,OverflowError) as error:
            report=dict(status='UNKNOWN',candidate_sha256=fingerprint(candidate),summary={},
                        failed_conditions=[str(error)],training_stop=candidate.get('training_stop'))
        (args.output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='cells'},indent=2))
    options=dict(count=args.rollouts,seed=args.rollout_seed,dt=args.rollout_dt,seconds=args.rollout_seconds)
    if args.command=='animate':
        try: from .animation import animate
        except ImportError: from animation import animate
        animate(args.output,**options)
    elif not args.no_plots:
        try: from .visualize import plot
        except ImportError: from visualize import plot
        plot(args.output,options)


if __name__=='__main__': main()
