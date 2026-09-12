import json
import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch

from nova_3d_xv15_syn.interval import I
from nova_3d_xv15_syn.model import Model, drift
from nova_3d_xv15_syn.main import Partition
from nova_3d_xv15_syn.verify import Checker, lower, upper, verify
from nova_3d_xv15_syn.animation import Evaluator, simulate, animate


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.p=json.loads((Path(__file__).parent/'problem.json').read_text())
        cls.model=Model(cls.p)

    def artifact(self):
        return dict(controller_type='smooth_affine_softmin',problem=self.p,model=self.model.export(),partition=[])

    def test_exact_benchmark(self):
        self.assertEqual(self.p['domain'],[[.5,100],[-20,20],[0,90]])
        self.assertEqual(self.p['initial'],[[28,32],[8.5,10.5],[58,62]])
        self.assertEqual(self.p['goal'],[[65,85],[-2,10],[25,35]])
        self.assertEqual(self.p['beta_ra'],5)
        self.assertEqual(len(self.p['unsafe']),6)
        self.assertEqual(self.p['unsafe'],[
            [[.5,100],[19,20],[0,90]],[[.5,100],[-20,-19],[0,90]],
            [[.5,100],[-20,20],[0,1]],[[.5,100],[-20,20],[89,90]],
            [[.5,1],[-20,20],[0,90]],[[99.5,100],[-20,20],[0,90]]])
        self.assertEqual(self.p['control_limits'],[[.1,1.8],[-16,16],[-5,5]])
        # Independent copy of the radian-coordinate equations, including Ito units.
        x=np.array([42.,9.,57.]);u=np.array([.8,3.,-2.]);deg=np.pi/180
        v,gamma,beta=x*np.array([1,deg,deg]);T,alpha,delta=u*np.array([5900*9.81,deg,deg])
        a=alpha/deg;r=beta*2/np.pi
        cl=(.0849*a+.3482)*(1-r)+(.0646*a+.2709)*r
        cd=(.00042143*a*a+.003*a+.0218)*(1-r)+(.000473216*a*a+.00343*a+.2165)*r
        L=.5*1.225*v*v*15.7*cl;D=.5*1.225*v*v*15.7*cd
        reference=np.array([(T*np.cos(alpha+beta)-D-5900*9.81*np.sin(gamma))/5900,
                            (T*np.sin(alpha+beta)+L-5900*9.81*np.cos(gamma))/(5900*v),delta])/np.array([1,deg,deg])
        actual=np.array([float(t.lo) for t in drift(I(x),[I(t) for t in u],self.p)])
        np.testing.assert_allclose(actual,reference,atol=1e-12)
        np.testing.assert_allclose(np.array([.5,.1*deg,.1*deg])/np.array([1,deg,deg]),self.p['diffusion'])

    def test_control_limits_and_smoothness(self):
        m=self.model
        x=torch.tensor([75.0001,4.0001,30.0001],dtype=torch.float64,requires_grad=True)
        policy=lambda x:torch.stack([u.lo for u in m.policy(I(x))])
        jac=torch.autograd.functional.jacobian(policy,x)
        self.assertTrue(torch.isfinite(jac).all())
        step=1e-6;eye=torch.eye(3,dtype=torch.float64)
        numeric=torch.stack([(policy(x+step*d)-policy(x-step*d))/(2*step) for d in eye],-1)
        torch.testing.assert_close(jac,numeric,atol=1e-5,rtol=1e-5)
        for u,(lo,hi) in zip(m.policy(I(x)),self.p['control_limits']):
            self.assertGreaterEqual(float(u.lo),lo)
            self.assertLessEqual(float(u.hi),hi)

    def test_derivatives_and_generator(self):
        m=self.model;x=torch.tensor([45.,7.,51.],dtype=torch.float64,requires_grad=True)
        z=(x-m.center)/m.scale
        v=(torch.expm1(z[:,None]*m.rates).square()*m.weights/m.normalizers).sum()
        grad=torch.autograd.grad(v,x,create_graph=True)[0]
        hess=torch.stack([torch.autograd.grad(grad[i],x,retain_graph=True)[0][i] for i in range(3)])
        _,g,h=m.features(I(x))
        torch.testing.assert_close(grad,(g.lo*m.weights).sum(-1))
        torch.testing.assert_close(hess,(h.lo*m.weights).sum(-1))
        f=torch.stack([t.lo for t in drift(I(x),m.policy(I(x)),self.p)])
        gv=(grad*f+.5*torch.tensor(self.p['diffusion']).double()**2*hess).sum()
        torch.testing.assert_close(m.values(I(x))[1].lo,gv,atol=1e-9,rtol=1e-7)

    def test_bounds_contain_samples(self):
        rng=np.random.default_rng(5);checker=Checker(self.artifact())
        for _ in range(6):
            c=rng.uniform([2,-15,3],[95,15,85]);box=np.stack([c-.2,c+.2],-1)
            tv,tg=self.model.bounds(torch.tensor(box))
            rv,rg=checker.evaluate(checker.box(box.tolist()))
            x=rng.uniform(box[:,0],box[:,1],size=(40,3));v,g=self.model.values(I(x))
            for samples,t,r in [(v.lo,tv,rv),(g.lo,tg,rg)]:
                self.assertGreaterEqual(float(samples.min()),float(t.lo)-1e-10)
                self.assertLessEqual(float(samples.max()),float(t.hi)+1e-10)
                self.assertGreaterEqual(float(samples.min()),float(lower(r))-1e-10)
                self.assertLessEqual(float(samples.max()),float(upper(r))+1e-10)

    def test_independent_point_values(self):
        c=Checker(self.artifact());x=[53.125,6.5,48.75]
        v,g=c.evaluate(c.box([[t,t] for t in x]));tv,tg=self.model.bounds(I(x))
        self.assertAlmostEqual(float(lower(v)),float(tv.lo),places=12)
        self.assertAlmostEqual(float(upper(g)),float(tg.hi),places=12)

    def test_partition_covers_full_domain(self):
        p=Partition(self.p);p.align();a=self.artifact();a['partition']=p.history;c=Checker(a)
        volume=sum(np.prod([float(hi-lo) for lo,hi in b]) for b in c.cells)
        self.assertAlmostEqual(volume,99.5*40*90)
        self.assertEqual(len(c.cells),125)
        a['partition']=[[0,0,200]]
        with self.assertRaises(ValueError): Checker(a)

    def test_reject_invalid_certificate(self):
        a=self.artifact();a['model']['weights'][0][0]=-1
        with self.assertRaises(ValueError): Checker(a)
        a=self.artifact();a['model']['weights']=[[0]*6 for _ in range(3)]
        report=verify(a)
        self.assertEqual(report['status'],'UNKNOWN')
        self.assertGreater(report['failed_cells'],0)
        a=self.artifact();a['model']['action_upper'][0]=2
        with self.assertRaises(ValueError): Checker(a)
        a=self.artifact();a['witness_actions']=[4]
        with self.assertRaises(ValueError): Checker(a)
        a=self.artifact();a['model']['temperature']=0
        with self.assertRaises(ValueError): Checker(a)

    def test_joint_gradients(self):
        m=Model(self.p);box=torch.tensor([[[28.,32.],[8.5,10.5],[58.,62.]]],dtype=torch.float64)
        v,g=m.bounds(box);(v.hi.sum()+g.hi.sum()).backward()
        self.assertGreater(float(m.weights.grad.abs().sum()),0)
        self.assertGreater(float(m.alpha_weights.grad.abs().sum()),0)

    def test_smoothing_penalty(self):
        rng=np.random.default_rng(4);x=rng.uniform([1,-19,1],[99,19,89],(200,3))
        m=self.model;_,corners=m.action_bounds(I(x));_,actual=m.values(I(x))
        gap=actual.lo-corners.lo.amin(-1)
        self.assertGreaterEqual(float(gap.min()),-1e-9)
        self.assertLessEqual(float(gap.max()),float(2*m.temperature*np.log(2))+1e-9)

    def test_numpy_rollout_matches_controller(self):
        e=Evaluator(self.artifact())
        for x in [np.array([30.,9.5,60.]),np.array([75.001,4.001,30.001]),np.array([60.,-3.,45.])]:
            v,g,u,f=e(x);tv,tg=self.model.values(I(x))
            np.testing.assert_allclose(u,[float(t.lo) for t in self.model.policy(I(x))],atol=1e-10)
            np.testing.assert_allclose([v,g],[float(tv.lo),float(tg.lo)],atol=1e-10)
        a=self.artifact();r=simulate(a,seconds=.01,dt=.01,seed=3)
        x=np.mean(self.p['initial'],axis=1);f=e(x)[3]
        expected=x+.01*f+.1*np.array(self.p['diffusion'])*np.random.default_rng(3).standard_normal(3)
        np.testing.assert_allclose(r['samples'][-1][1:4],expected)
        self.assertEqual(r,simulate(a,seconds=.01,dt=.01,seed=3))

    def test_animation_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder);(output/'candidate.json').write_text(json.dumps(self.artifact()))
            animate(output,seconds=.002,dt=.001)
            html=(output/'aircraft_animation.html').read_text()
            self.assertIn('requestAnimationFrame',html)
            self.assertNotIn('__ROLLOUT_DATA__',html)
            self.assertTrue((output/'rollout.csv').is_file())


if __name__=='__main__': unittest.main()
