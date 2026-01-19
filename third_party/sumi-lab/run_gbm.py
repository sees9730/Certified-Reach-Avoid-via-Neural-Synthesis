import csv
import time
import torch
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle
import controlled_sde
import stochastic_rsa as rsa

import os
import sys
from contextlib import contextmanager

class Tee:
    """Write to both terminal and a file."""
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

@contextmanager
def tee_stdout_stderr(log_path: str, mode: str = "a"):
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, mode) as log_f:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = Tee(old_out, log_f)
        sys.stderr = Tee(old_err, log_f)
        try:
            yield
        finally:
            sys.stdout = old_out
            sys.stderr = old_err

torch.set_default_dtype(torch.float32)
torch.use_deterministic_algorithms(True)

device = torch.device("cpu")

# initialize the controlled SDE
sde = controlled_sde.GBM()
net = rsa.CertificateModule(device=device)

# set the boundaries of the sets
global_bounds = np.array([[[-100.0, -100.0], [100.0, 100.0]]])
initial_bounds = np.array([[[45, -55], [55, -45]]])
target_bounds = np.array([[[-25.0, -25.0], [25.0, 25.0]]])
unsafe_bounds = np.array([
    [[-100.0, -100.0], [-80.0, 100.0]],
])

# create the sets
interest_set = rsa.AABBSet(global_bounds, device)
initial_set = rsa.AABBSet(initial_bounds, device)
target_set = rsa.AABBSet(target_bounds, device)
unsafe_set = rsa.AABBSet(unsafe_bounds, device)
reach_avoid_probability = 0.95

# create the specification
spec = rsa.Specification(
    interest_set,
    initial_set,
    unsafe_set,
    target_set,
    reach_avoid_probability=reach_avoid_probability,
    stay_probability=None,
    require_stay=False
)

log_file = "run_gbm_log.txt"
N_trials = 5
# Note: verifier_mesh_size=100, all passed 54 seconds (on average)
# Note: verifier_mesh_size = 200, 
#   4/5 passed but 1/5 (seed 50058) killed due to running out of memoery

with tee_stdout_stderr(log_file, mode="w"):
    print("=== GBM Training Run ===")
    print("Start time:", time.strftime("%Y-%m-%d %H:%M:%S"))

    overall_t0 = time.perf_counter()

    npr.seed(1)
    seeds = npr.randint(1, 1e5, size=(N_trials,))
    for seed in seeds:
        torch.manual_seed(seed)
        npr.seed(seed)

        # create the certificate
        net = rsa.CertificateModule(device=device)
        certificate = rsa.SupermartingaleCertificate(sde, spec, net, device)

        # train the certificate
        print(f"\n--- Seed {seed} ---")
        t0 = time.perf_counter()
        result = certificate.train(verify_every_n=1000,
                                verifier_mesh_size=100,
                                zeta=1.0,
                                regularizer_lambda=1e-1,
                                verification_slack=2
                                )
        t = time.perf_counter() - t0

        print(f"Seed {seed} training time: {t:.3f} s")
        print(f"Train returned: {result}")

        row = (seed, t, result[0], result[1])
        with open('gbm_3d.csv', 'a', newline='') as file:
            writer = csv.writer(file, dialect='excel')
            writer.writerow(row)

    overall_t = time.perf_counter() - overall_t0
    print("\n=== Done ===")
    print(f"Total wall time: {overall_t:.3f} s")
    print("End time:", time.strftime("%Y-%m-%d %H:%M:%S"))


# # Initialize the batch of starting states
# x0 = torch.tile(
#     torch.tensor([[50.0, -50.0]], device=device),
#     dims=(4, 1)
# )
# ts = torch.linspace(0, 100.0, 1000, device=device)

# #
# sample_paths = sde.sample(x0, ts, method="euler").squeeze()

# # Plot
# fig, ax1 = plt.subplots(1, 1)

# with torch.no_grad():
#     print(global_bounds[0, 0, 0])
#     grid = torch.stack(
#         torch.meshgrid(
#             torch.linspace(global_bounds[0, 0, 0],
#                            global_bounds[0, 1, 0], 101),
#             torch.linspace(global_bounds[0, 0, 1],
#                            global_bounds[0, 1, 1], 101),
#             indexing='xy'
#         )
#     )
#     grid = grid.reshape(2, -1).T
#     out = certificate.net(grid).detach().numpy().reshape(101, 101)
#     scaling_factor = certificate.net(
#         initial_set.sample(1000)
#     ).detach().numpy().max()
#     out /= scaling_factor
#     min_level = int(np.floor(np.log10(out.min()) * 5))
#     max_level = int(np.ceil(np.log10(out.max()) * 5)) + 1
#     c = ax1.contourf(
#         np.linspace(global_bounds[0, 0, 0], global_bounds[0, 1, 0], 101),
#         np.linspace(global_bounds[0, 0, 1], global_bounds[0, 1, 1], 101),
#         out,
#         norm=colors.LogNorm(),
#         levels=[10 ** (n / 5) for n in range(min_level, max_level, 1)]
#     )

# fig.colorbar(c, ax=ax1)

# ax1.set_xlim(global_bounds[0, :, 0])
# ax1.set_ylim(global_bounds[0, :, 1])
# for i in range(initial_bounds.shape[0]):
#     ax1.add_patch(Rectangle(initial_bounds[i, 0, :], *(initial_bounds[i, 1, :] - initial_bounds[i, 0, :]),
#                             edgecolor='yellow',
#                             facecolor='none',
#                             lw=2))
# for i in range(target_bounds.shape[0]):
#     ax1.add_patch(Rectangle(target_bounds[i, 0, :], *(target_bounds[i, 1, :] - target_bounds[i, 0, :]),
#                             edgecolor='limegreen',
#                             facecolor='none',
#                             lw=2))
# for i in range(unsafe_bounds.shape[0]):
#     ax1.add_patch(Rectangle(unsafe_bounds[i, 0, :], *(unsafe_bounds[i, 1, :] - unsafe_bounds[i, 0, :]),
#                             edgecolor='red',
#                             facecolor='none',
#                             lw=2))

# path_data = sample_paths.numpy()
# ax1.plot(path_data[:, :, 0], path_data[:, :, 1],
#          color="white", lw=1, alpha=0.5
#          )

# plt.show()

# # sde.render(sample_paths, ts)
# sde.close()
