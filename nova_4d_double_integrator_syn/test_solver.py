"""Implementation checks only: these tests do not run synthesis or assert SAT."""
import copy
import json
import tempfile
import unittest
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import torch
from .curved import CurvedPair,split
from .curved_check import Proof,replay,sqrt_upper
from .check import box,validate,verify
from .runtime import Controller,rollout
from .animation import animate
from .bound_training import features,fit_weights


class SolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem=json.loads((Path(__file__).parent/'problem.json').read_text())
        cls.pair=CurvedPair(cls.problem)
        cls.candidate=dict(format='curved_double_integrator_v2',problem=cls.problem,model=cls.pair.export(),
            energy_caps=[repr(float(q)) for q in cls.pair.caps(cls.problem['beta_ra'])],energy_splits=[],region_splits=[])

    def test_obstacle_blocks_segment_and_boundaries_remain_covered(self):
        validate(self.problem)
        initial=np.mean(self.problem['initial'],axis=1);goal=np.mean(self.problem['goal'],axis=1)
        points=initial[None]+np.linspace(0,1,1001)[:,None]*(goal-initial)
        obstacle=np.array(self.problem['unsafe'][-1])
        self.assertTrue(np.any(((points[:,:2]>=obstacle[:2,0])&(points[:,:2]<=obstacle[:2,1])).all(1)))
        for i in range(8):
            p=copy.deepcopy(self.problem);p['unsafe'].pop(i)
            with self.assertRaises(ValueError):validate(p)

    def test_interval_bounds_cover_full_four_dimensional_boxes(self):
        model=Controller(self.candidate);proof=Proof(self.candidate);rng=np.random.default_rng(2)
        for _ in range(8):
            lo=rng.uniform(-3,2,4);hi=lo+rng.uniform(.01,1,4);region=np.stack([lo,hi],-1)
            x=rng.uniform(lo,hi,(200,4));pos,_,w,_,_=model.coordinates(x);q=pos**2+model.gamma*w**2
            exact=proof.energy(box(region.tolist()));floating=self.pair.region_energy(torch.tensor(region[None]))
            for i in range(2):
                np.testing.assert_allclose([floating[i].l.item(),floating[i].h.item()],[float(exact[i].l),float(exact[i].h)],rtol=1e-12)
                self.assertGreaterEqual(q[:,i].min(),float(exact[i].l)-1e-10)
                self.assertLessEqual(q[:,i].max(),float(exact[i].h)+1e-10)

    def test_generator_matches_autograd_including_saturated_feedback(self):
        model=Controller(self.candidate)
        for state in [[-3,1.5,.4,-1.07],[.1,-.2,-.1,.3],[4,4,7,6]]:
            x=torch.tensor(state,dtype=torch.float64,requires_grad=True);px=x[0]
            h=model.b*px+model.c*px**3;hp=model.b+3*model.c*px**2
            pos=torch.stack([px,x[1]-h]);vel=torch.stack([x[2],x[3]-hp*x[2]])
            q=pos**2+torch.tensor(model.gamma)*(vel+torch.tensor(model.k)*pos)**2
            v=sum(sum(float(w)*(q[i]/model.scales[i])**j for j,w in enumerate(model.weights[i],1)) for i in range(2))
            grad=torch.autograd.grad(v,x,create_graph=True)[0]
            hess=torch.stack([torch.autograd.grad(grad[i],x,retain_graph=True)[0][i] for i in range(4)])
            drift=torch.tensor(np.r_[state[2:],model(state)],dtype=torch.float64)
            noise=torch.tensor([0.,0.,*np.square(self.problem['acceleration_noise'])],dtype=torch.float64)
            g=(grad*drift).sum()+(hess*noise).sum()/2
            np.testing.assert_allclose(model.values(state),[float(v),float(g)],rtol=1e-11,atol=1e-10)

    def test_generator_majorant_matches_independent_rational_bounds(self):
        proof=Proof(self.candidate);rng=np.random.default_rng(3)
        for _ in range(8):
            lo=rng.uniform(.01,.7,2);hi=lo+.1;cell=np.stack([lo,hi],-1)
            vl,_,g,_=self.pair.energy_bounds(torch.tensor(cell[None]),torch.tensor(np.array(self.candidate['energy_caps'],float)))
            exact_l,exact_g,_=proof.generator([(F(str(a)),F(str(b))) for a,b in cell])
            np.testing.assert_allclose(float(vl[0]),float(exact_l),rtol=1e-12)
            if exact_g is not None: self.assertGreaterEqual(float(exact_g)+1e-8,float(g[0]))

    def test_controller_is_bounded_and_identity_inside_saturation_core(self):
        model=Controller(self.candidate);rng=np.random.default_rng(4);x=rng.uniform(-20,20,(500,4))
        u=model(x);self.assertTrue(np.all(abs(u)<=np.array(self.problem['control_limit'])))
        raw=model.raw_control(x);inside=np.abs(raw)<=.95*np.array(self.problem['control_limit'])
        np.testing.assert_allclose(u[inside],raw[inside],rtol=0,atol=0)

    def test_partition_replay_covers_root_and_rejects_invalid_cuts(self):
        cells=[np.array([[0.,1.],[0.,1.]])];history=[]
        for step in range(20):
            index=step%len(cells);axis=int(np.argmax(np.ptp(cells[index],axis=1)));split(cells,history,index,axis)
        exact=replay([[(F(0),F(1))]*2],history)
        self.assertEqual(sum((b[0][1]-b[0][0])*(b[1][1]-b[1][0]) for b in exact),1)
        with self.assertRaises(ValueError):replay([[(F(0),F(1))]*2],[[0,0,'2']])
        for value in [F(0),F(2),F(1,10**30)]: self.assertGreaterEqual(sqrt_upper(value)**2,value)

    def test_incomplete_or_negative_certificate_is_rejected(self):
        c=copy.deepcopy(self.candidate);c['energy_caps']=['.001','.001']
        with self.assertRaises(ValueError):verify(c)
        c=copy.deepcopy(self.candidate);c['model']['weights'][0][0]='-1'
        with self.assertRaises(ValueError):verify(c)

    def test_generator_is_linear_in_fitted_weights(self):
        cells=torch.tensor([[[.02,.03],[.01,.04]],[[.1,.2],[.1,.2]]],dtype=torch.float64)
        caps=torch.tensor(np.array(self.candidate['energy_caps'],float))
        with torch.no_grad():
            before=self.pair.weights.clone();G=features(self.pair,cells,caps)
            np.testing.assert_allclose(G@before.numpy().ravel(),self.pair.energy_bounds(cells,caps)[2].numpy(),rtol=1e-12)
            torch.testing.assert_close(self.pair.weights,before)

    def test_linear_solver_enforces_nonnegative_coefficients(self):
        result=fit_weights(np.array([[-1.,0.],[0.,-1.],[1.,1.]]),np.array([-2.,-3.,6.]))
        self.assertTrue(result.success);np.testing.assert_allclose(result.x,[2,3],atol=1e-8)

    def test_rollout_integrates_original_sde(self):
        model=Controller(self.candidate);rng=np.random.default_rng(7);initial=np.array(self.problem['initial'])
        x=rng.uniform(initial[:,0],initial[:,1]);expected=x+.01*np.r_[x[2:],model(x)]+.1*np.r_[0,0,np.array(self.problem['acceleration_noise'])*rng.standard_normal(2)]
        result=rollout(model,seed=7,dt=.01,seconds=.01)
        np.testing.assert_allclose(result['samples'][-1][1:5],expected)

    def test_animation_uses_deployed_controls_and_never_claims_stale_sat(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);(out/'candidate.json').write_text(json.dumps(self.candidate))
            (out/'verification.json').write_text(json.dumps({'status':'SAT','candidate_sha256':'stale'}))
            runs=animate(out,count=2,dt=.01,seconds=.02)
            data=json.loads((out/'rollouts.json').read_text());self.assertEqual(data['verification'],'Unverified')
            html=(out/'rollouts_animation.html').read_text();self.assertIn('const data=',html)
            model=Controller(self.candidate)
            for run in runs:
                rows=np.array(run['samples']);np.testing.assert_allclose(rows[:,5:7],model(rows[:,1:5]))


if __name__=='__main__':unittest.main()
