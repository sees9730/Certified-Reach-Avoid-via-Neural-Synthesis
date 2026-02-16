# Neural Certificate

<!-- Add your demo gif here -->
![Demo](examples/synthesis/xv15aircraft_syn/results_opt/animation_synthesis_opt.gif)

Learning neural certificates and controllers for stochastic systems with hard constraints. This is the code repository for our paper **"Training with Hard Constraints: Learning Neural Certificates and Controllers for SDEs"** submitted to Neus 2026.

We use interval bound propagation and adaptive partitioning to train neural networks that certify reach-avoid specifications for stochastic differential equations (SDEs). The framework supports both verification (certifying existing systems) and synthesis (learning provably safe controllers).

## Getting Started

**Python Version:** 3.11.14

### Installation

```bash
pip install -r requirements.txt
```

### Running Examples

```bash
cd examples/synthesis/inv_pend_syn_unsafe  # or any other example
python3 main.py
```

The training will start and outputs will be saved in that example's `outputs/` folder.

## What You'll Find

Each example directory contains:
- `main.py` - The main script to run
- `outputs/eval_bundle.pth` - Saved model and training objects
- `outputs/terminal_log.txt` - Training logs
- `results/` - Summary visualizations from the last training epoch
- `test` and/or `opt` python files - To test the SAT certificate and/or controller

## Examples

The repository includes two types of examples:

- **Synthesis** (`examples/synthesis/`) - Control synthesis examples
- **Verification** (`examples/verification/`) - Verification examples

It also includes the AAAI Neural Continuous-Time Supermartingale Certificates code by Neustroev et al. under `third_party/sumi-lab`