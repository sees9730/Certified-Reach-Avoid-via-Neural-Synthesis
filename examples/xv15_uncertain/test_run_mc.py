"""Monte Carlo physics, event, reproducibility, checkpoint, and reporting checks."""
from copy import deepcopy
from itertools import product
import json

import numpy as np
import pytest
import torch
from torch import nn

from examples.xv15_uncertain.model import XV15Aero, XV15EqMLPControl, find_goal_equilibrium, load_config, load_region_arrays
from examples.xv15_uncertain.run_mc import (
    ATTACKS, MODES, ExportedPPOControl, adversarial_score, aggregate_results,
    discover_checkpoints, load_controller, main, rollout_mc, select_parameters,
)
from examples.xv15_uncertain.postprocess_mc import main as postprocess_main


class ConstantControl(nn.Module):
    def forward(self, x):
        return torch.tensor([5900 * 9.81, 0., 0.]).expand(len(x), -1)


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_realized_physics_matches_independent_models():
    config = load_config()
    x = torch.tensor([[30., .15, 1.], [75., .05, .5]], dtype=torch.float64)
    u = ConstantControl()(x).double()
    rho, mass = torch.tensor([1.16375, 1.28625]), torch.tensor([5841., 5959.])
    actual = XV15Aero(config)(x, u, density=rho, mass=mass)
    for i in range(2):
        fixed = deepcopy(config)
        fixed['dynamics'].update(density=float(rho[i]), mass=float(mass[i]))
        torch.testing.assert_close(actual[i:i+1], XV15Aero(fixed)(x[i:i+1], u[i:i+1]))


@pytest.mark.parametrize('attack', ATTACKS)
def test_adversary_selects_best_joint_corner_and_preserves_uniform_draw(attack):
    config = load_config()
    aero = XV15Aero(config)
    arrays = {k: torch.tensor(v, dtype=torch.float64) for k, v in load_region_arrays(config).items()}
    x = torch.tensor([[30., .15, 1.], [70., -.05, .5]], dtype=torch.float64)
    u = ConstantControl()(x).double()
    density, mass = (1.16375, 1.28625), (5841., 5959.)
    draw = torch.tensor([1.2, 1.25], dtype=torch.float64)
    kwargs = dict(density_interval=density, mass_interval=mass,
                  density_draw=draw, mass_draw=torch.tensor([5900., 5910.]), dt=.01, attack=attack)
    rho, m = select_parameters(aero, x, u, arrays, density_mode='adversarial', mass_mode='adversarial', **kwargs)
    scores = torch.stack([adversarial_score(x+.01*aero(x,u,density=r,mass=w),arrays,attack)
                          for r,w in product(density,mass)])
    selected = adversarial_score(x+.01*aero(x,u,density=rho,mass=m),arrays,attack)
    torch.testing.assert_close(selected, scores.max(dim=0).values)
    assert all(float(v) in density for v in rho)
    assert all(float(v) in mass for v in m)
    rho, m = select_parameters(aero, x, u, arrays, density_mode='uniform', mass_mode='adversarial', **kwargs)
    torch.testing.assert_close(rho, draw)


@pytest.mark.parametrize('refresh', ['step', 'episode'])
def test_uniform_parameters_and_repeatability(refresh):
    config = load_config()
    config['uncertainty']['mass_kg'] = [5841., 5959.]
    kwargs = dict(n_mc=4, t_max=.05, dt=.01, trace_dt=.01, n_paths=4,
                  density_mode='uniform', mass_mode='uniform', uniform_refresh=refresh)
    first = rollout_mc(ConstantControl(), config, **kwargs)
    second = rollout_mc(ConstantControl(), config, **kwargs)
    torch.testing.assert_close(first['final_states'], second['final_states'], rtol=0, atol=0)
    for path in first['paths']:
        parameters = path['parameters']
        assert ((parameters[:,0] >= 1.16375) & (parameters[:,0] <= 1.28625)).all()
        assert ((parameters[:,1] >= 5841) & (parameters[:,1] <= 5959)).all()
        if refresh == 'episode':
            torch.testing.assert_close(parameters, parameters[0].expand_as(parameters))
        else:
            assert parameters[:,0].unique().numel() > 1
            assert parameters[:,1].unique().numel() > 1


def test_random_streams_are_independent_of_modes_and_early_termination(monkeypatch):
    monkeypatch.setattr(XV15Aero, 'forward', lambda self,x,u,**kwargs: u)
    class ZeroControl(nn.Module):
        def forward(self,x):
            return torch.zeros_like(x)
    class FirstPathExits(ZeroControl):
        first = True
        def forward(self,x):
            u = super().forward(x)
            if self.first:
                u[0,0] = 1e6
                self.first = False
            return u
    config = load_config()
    kwargs = dict(n_mc=4, t_max=.03, dt=.01, n_paths=0)
    baseline = rollout_mc(ZeroControl(), config, **kwargs)
    for density_mode, mass_mode in product(MODES, repeat=2):
        result = rollout_mc(ZeroControl(), config, density_mode=density_mode, mass_mode=mass_mode, **kwargs)
        torch.testing.assert_close(result['final_states'], baseline['final_states'], rtol=0, atol=0)
    result = rollout_mc(FirstPathExits(), config, **kwargs)
    assert result['stop_reasons'][0] == 'domain_exit'
    torch.testing.assert_close(result['final_states'][1:], baseline['final_states'][1:], rtol=0, atol=0)


@pytest.mark.parametrize('unsafe', [False, True])
def test_final_partial_step_counts_goal_and_safety_has_priority(monkeypatch, unsafe):
    monkeypatch.setattr(XV15Aero, 'forward', lambda self,x,u,**kwargs:
                        torch.tensor([1.,0.,0.],dtype=x.dtype).expand_as(x))
    config = load_config()
    regions = config['regions_mps_deg_deg']
    regions['init_range'][0] = [30., 30.001]
    regions['goal_range'] = [[30.024,31.],[-19.,19.],[1.,89.]]
    if unsafe:
        regions['unsafe_min_vel'] = deepcopy(regions['goal_range'])
    result = rollout_mc(ConstantControl(),config,n_mc=3,t_max=.025,dt=.01,
                        trace_dt=.01,stochastic=False)
    assert result['outcomes'] == ['fail' if unsafe else 'success'] * 3
    assert result['stop_reasons'] == ['unsafe' if unsafe else 'goal'] * 3
    torch.testing.assert_close(result['stop_times'],torch.full((3,),.025,dtype=torch.float64))
    assert float(result['paths'][0]['times'][-1]) == .025


def test_nonfinite_states_are_failures(monkeypatch):
    monkeypatch.setattr(XV15Aero,'forward',lambda self,x,u,**kwargs: torch.full_like(x,float('nan')))
    result = rollout_mc(ConstantControl(),load_config(),n_mc=2,t_max=.1,dt=.01)
    assert result['stop_reasons'] == ['nonfinite']*2
    assert result['outcomes'] == ['fail']*2


def make_checkpoint(directory):
    directory.mkdir(parents=True)
    config=load_config()
    aero=XV15Aero(config)
    x_eq,u_eq=find_goal_equilibrium(config,aero)
    controller=XV15EqMLPControl(config,x_eq,u_eq,[100.,.35,1.57])
    torch.save(dict(V_state_dict={},control_state_dict=controller.state_dict()),directory/'eval_bundle.pth')
    (directory/'run_config.json').write_text(json.dumps(dict(example=config)))
    return controller


def test_checkpoint_discovery_loading_cli_and_postprocessing(tmp_path):
    base=tmp_path/'controllers'
    original=make_checkpoint(base/'seed10'/'outputs')
    make_checkpoint(base/'seed2'/'outputs')
    checkpoints=discover_checkpoints(base)
    assert [seed for seed,_ in checkpoints] == ['seed2','seed10']
    restored,kind=load_controller(base/'seed10'/'outputs'/'eval_bundle.pth')
    x=torch.randn(5,3)
    torch.testing.assert_close(original(x),restored(x))
    assert kind == 'neural_certificate'
    output=tmp_path/'mc'
    cache=main(['--controller',f'test={base}','--n-mc','2','--t-max','.015',
                '--n-paths','1','--no-plots','--output-dir',str(output)])
    assert len(cache['runs']) == 28  # 14 scenarios/attacks, two training seeds
    assert len(aggregate_results(cache['runs'])) == 9
    saved=torch.load(output/'mc_cache.pth',weights_only=True)
    assert saved['format'] == 'xv15_mc_v1'
    postprocess_main(['--cache',str(output/'mc_cache.pth'),'--output-dir',str(tmp_path/'plots'),'--max-paths','1'])
    assert (tmp_path/'plots'/'success_rate_summary.pdf').stat().st_size > 1000
    assert (tmp_path/'plots'/'density_zero__mass_zero'/'parameters.pdf').is_file()


def test_aggregation_takes_worst_attack_per_seed_before_averaging():
    runs=[]
    for seed,rates in [('seed0',[.2,.8]),('seed1',[.9,.3])]:
        for attack,rate in zip(ATTACKS,rates):
            runs.append(dict(label='test',training_seed=seed,density_mode='adversarial',mass_mode='uniform',
                attack=attack,stats=dict(p_success=rate,p_fail=1-rate,p_timeout=0,
                                         hit_time_s_mean=2.,normalized_effort_mean=3.)))
    row=aggregate_results(runs)[0]
    assert row['p_success'] == pytest.approx(.25)
    assert row['selected_attacks'] == {'seed0':'lookahead','seed1':'nearest_unsafe'}


def test_ppo_export_normalization_clipping_and_loading(tmp_path):
    state=dict(obs_center=torch.tensor([50.,0.,.8]),obs_scale=torch.tensor([50.,.4,.8]),
               action_low=torch.tensor([1.,-2.,-3.]),action_high=torch.tensor([10.,2.,3.]))
    policy=nn.Sequential(nn.Linear(3,4),nn.Tanh())
    action=nn.Linear(4,3)
    state.update({f'policy_net.{k}':v for k,v in policy.state_dict().items()})
    state.update({f'action_net.{k}':v for k,v in action.state_dict().items()})
    payload=dict(format='xv15_sb3_ppo_actor_v1',hidden_dims=[4],control_state_dict=state)
    checkpoint=tmp_path/'rl_controller.pth'
    torch.save(payload,checkpoint)
    controller,_=load_controller(checkpoint)
    x=torch.randn(20,3)*100
    a=action(policy((x-state['obs_center'])/state['obs_scale'])).clamp(-1,1)
    expected=state['action_low']+(a+1)/2*(state['action_high']-state['action_low'])
    torch.testing.assert_close(controller(x),expected)
