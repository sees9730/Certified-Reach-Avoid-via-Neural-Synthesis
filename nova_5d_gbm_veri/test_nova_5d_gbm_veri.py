"""Run with python -m unittest nova_5d_gbm_veri.test_nova_5d_gbm_veri -v; no files outside nova_5d_gbm_veri are used."""
import contextlib
from fractions import Fraction as Q
import io
import json
from math import prod
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from nova_5d_gbm_veri.main import Bounds, train, verify

ROOT = Path(__file__).parent


def one_d(drift="-x1", diffusion="0"):
    return dict(domain=[[-6,6]],drift=[drift],diffusion=[[diffusion]],
                initial=[[[-1,1]]],unsafe=[[[5,6]]],goal=[[[-1,1]]],beta_ra="20",epsilon="1/10000")


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem = json.loads((ROOT/"5d_gbm.json").read_text())
        cls.engine = Bounds(cls.problem)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.weights,cls.report,cls.log = train(cls.engine,seconds=30)

    def test_exact_benchmark_data_and_successful_bound_training(self):
        p = self.problem
        self.assertEqual(p["domain"],[[-100,100]]*5)
        self.assertEqual(p["initial"],[[[45,55],[-55,-45],[45,55],[45,55],[45,55]]])
        self.assertEqual(p["unsafe"],[[[-100,-80],[-100,100],[-100,-80],[-100,-80],[-100,-80]]])
        self.assertEqual(p["goal"],[[[-25,25]]*5])
        self.assertEqual(p["drift"],["-3/2*x1+x2","-x1-3/2*x2+x3","-x2-3/2*x3+x4","-x3-3/2*x4+x5","-x4-3/2*x5"])
        self.assertEqual(p["diffusion"],[[f"x{i+1}/5" if i==j else "0" for j in range(5)] for i in range(5)])
        self.assertEqual((Q(p["beta_ra"]),Q(p["epsilon"])),(20,Q(1,10000)))
        self.assertEqual(self.report["status"],"SAT")
        self.assertTrue(all(cell["passed"] for cell in self.report["cells"]))
        self.assertGreater(self.log[0]["loss"],0)
        self.assertEqual(self.log[-1]["loss"],0)
        self.assertTrue(self.engine.history)
        self.assertNotEqual(self.weights,[str(Q(0.005))]*len(self.weights))

    def test_replay_checks_all_cells_and_preserves_domain_volume(self):
        fresh = Bounds(self.problem)
        fresh.replay(self.engine.history)
        self.assertEqual(verify(fresh,self.weights),self.report)
        self.assertEqual(sum(prod(b-a for a,b in c) for c in fresh.cells["generator"]),Q(200)**5)
        for bad in ([["generator",0,0,"100"]], [["generator",999,0,"0"]], [["generator",0,5,"0"]]):
            with self.assertRaises(ValueError): Bounds(one_d(),1).replay(bad)

    def test_bad_weights_do_not_reuse_success(self):
        for factor in (Q(0),Q(1,100),Q(100)):
            self.assertEqual(verify(self.engine,[str(Q(w)*factor) for w in self.weights])["status"],"UNKNOWN")
        with self.assertRaises(ValueError): verify(self.engine,[-1]*len(self.weights))

    def test_gv_guard_and_exact_margin_boundary(self):
        engine = Bounds(one_d("-x1/20000"),1)
        engine.split("generator",0,0,Q(-1)); engine.split("generator",1,0,Q(1))
        report = verify(engine,["1","0"])
        self.assertEqual(report["status"],"SAT")
        self.assertEqual(Q(report["summary"]["GV_active_upper"]),-engine.epsilon)
        # GV is positive in part of the domain, but only where V >= beta.
        engine = Bounds(one_d("x1*(x1**2-25)"),1)
        for cut in range(-23,24): engine.split("generator",len(engine.cells["generator"])-1,0,Q(cut,4))
        report = verify(engine,["1","0"])
        self.assertEqual(report["status"],"SAT")
        self.assertTrue(any(r["region"]=="generator" and Q(r["GV_upper"])>0 and
                            r["reason"]=="V_lower >= beta_ra" for r in report["cells"]))

    def test_changed_nonlinear_sde_and_regions_train_without_code_changes(self):
        problem = one_d("-x1-x1**3","1/10")
        problem.update(initial=[[["-0.1","0.1"]]],unsafe=[[[-3,-2]],[[2,3]]],goal=[[["-0.5","0.5"]]])
        with contextlib.redirect_stdout(io.StringIO()):
            weights,report,_ = train(Bounds(problem,2),epochs=200,refine_every=5,seconds=10)
        self.assertEqual(report["status"],"SAT")
        self.assertTrue(all(Q(w)>=0 for w in weights))

    def test_infeasible_generator_returns_unknown_on_budget_exhaustion(self):
        problem = one_d("0")
        problem["initial"]=[[[1,"1.1"]]]
        problem["goal"]=[[["-0.5","0.5"]]]
        with contextlib.redirect_stdout(io.StringIO()):
            _,report,_ = train(Bounds(problem,1),epochs=10,refine_every=1,max_cells=20,seconds=5)
        self.assertEqual(report["status"],"UNKNOWN")

    def test_full_diffusion_and_polynomial_interval_bounds(self):
        p = dict(domain=[[-1,1]]*2,drift=["0","0"],diffusion=[["1","2"],["3","4"]],
                 initial=[[[-1,1]]*2],unsafe=[[[-1,1]]*2],goal=[[[-1,1]]*2],beta_ra="20",epsilon="0.0001")
        engine = Bounds(p,1)
        bounds = engine.bounds(engine.domain)
        self.assertEqual(bounds[-1][1],(Q(52),Q(52)))  # G((z1+z2)^2) = 52, including mixed Hessian.
        for polys,b in zip(engine.polynomials,bounds):
            for terms,(lo,hi) in zip(polys,b):
                for x in ((Q(-1),Q(1)),(Q(1,3),Q(-2,3)),(Q(0),Q(0))):
                    value = sum(c*prod(v**k for v,k in zip(x,powers)) for powers,c in terms)
                    self.assertLessEqual(lo,value); self.assertLessEqual(value,hi)

    def test_invalid_domain_and_executable_expressions_rejected(self):
        for edits in (dict(domain=[[-1,1]]),dict(drift=["__import__('os').getcwd()"]),
                      dict(diffusion=[]),dict(epsilon=0)):
            with self.assertRaises(ValueError): Bounds({**one_d(),**edits},1)

    def test_plot_samples_match_exact_model_and_stay_in_domain(self):
        import numpy as np
        from nova_5d_gbm_veri.plotting import sample
        engine = Bounds(one_d(),1)
        grid,(V,GV) = sample(engine,["1","0"],(0,),[Q(0)],13)
        np.testing.assert_allclose(V,grid[0]**2)
        np.testing.assert_allclose(GV,-2*grid[0]**2)
        self.assertEqual((grid[0].min(),grid[0].max()),(-6,6))
        for axes,anchor in (((0,0),[Q(0)]),((0,),[Q(7)])):
            with self.assertRaises(ValueError): sample(engine,["1","0"],axes,anchor,13)

    def test_plot_exports_actual_region_intersections(self):
        from nova_5d_gbm_veri.plotting import plot
        p = dict(domain=[[-5,5]]*3,drift=["-x1","-x2","-x3"],diffusion=[["0"]]*3,
                 initial=[[[1,2]]*3],unsafe=[[[-4,-3]]*3],goal=[[["-0.5","0.5"]]*3],beta_ra="20",epsilon="0.0001")
        engine = Bounds(p,1)
        with tempfile.TemporaryDirectory() as folder:
            pdf = plot(engine,["1"]+["0"]*(len(engine.names)-1),folder,"UNKNOWN",
                       axes=["x3","x1"],fixed=["x2=3/2"],points=21)
            self.assertGreater(pdf.stat().st_size,1000)
            self.assertGreater((pdf.parent/"custom.png").stat().st_size,1000)
            data = json.loads((pdf.parent/"slices.json").read_text())[0]
            self.assertEqual(data["axes"],["x3","x1"])
            self.assertEqual(data["regions"],["initial"])
            self.assertEqual(data["domain"],p["domain"])

    def test_standalone_folder_can_reverify_without_training(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/"nova_5d_gbm_veri"
            shutil.copytree(ROOT,target,ignore=shutil.ignore_patterns("results","__pycache__"))
            output = target/"results"; output.mkdir()
            artifact = dict(status="SAT",problem=self.problem,degree=self.engine.degree,weights=self.weights,
                            features=self.engine.names,center=list(map(str,self.engine.center)),
                            scale=list(map(str,self.engine.scale)),splits=self.engine.history)
            path = output/"certificate.json"; path.write_text(json.dumps(artifact))
            env = dict(os.environ); env.pop("PYTHONPATH",None)
            run = subprocess.run([sys.executable,str(target/"main.py"),"verify","--no-plots"],cwd=folder,env=env,
                                 capture_output=True,text=True,timeout=20)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            artifact["weights"]=["0"]*len(self.weights); path.write_text(json.dumps(artifact))
            run = subprocess.run([sys.executable,str(target/"main.py"),"verify","--no-plots"],cwd=folder,env=env,
                                 capture_output=True,text=True,timeout=20)
            self.assertEqual(run.returncode,1,run.stdout+run.stderr)


if __name__=="__main__": unittest.main()
