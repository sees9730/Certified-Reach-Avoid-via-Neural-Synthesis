# Neural Certificate

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

#### Option 1: Local Python Environment

```bash
cd examples/synthesis/inv_pend_syn_unsafe  # or any other example
python3 main.py
```

The training will start and outputs will be saved in that example's `outputs/` folder.

#### Option 2: Using Docker

First, build the Docker image:

```bash
docker build -t crans .
```

Then run an example from the project **root** directory:

```bash
docker run -v $(pwd):/app \
  -w /app/examples/verification/2D_gbm_veri \
  crans python main.py
```

Replace `examples/verification/2D_gbm_veri` with any example path. The `-w` flag sets the working directory inside the container, ensuring outputs are saved in the correct example folder.

### Docker + dReal Solver (Separate Workflow)

Use this workflow when you want the dReal-enabled container build and run process.

Build image:

```bash
docker buildx build --platform linux/amd64 --load -f Dockerfile.dreal4master -t cra-dreal-src .
```

Run any Python script in background:

```bash
docker rm -f myrun 2>/dev/null || true
docker run -d --name myrun --platform linux/amd64 \
  -v "$PWD":/workspace -w /workspace \
  cra-dreal-src \
  python3 -u path/to/script.py --arg1 ...
```

Example (`inv_pend_syn`):

```bash
docker rm -f invpend_run 2>/dev/null || true
docker run -d --name invpend_run --platform linux/amd64 \
  -v "$PWD":/workspace -w /workspace \
  cra-dreal-src \
  python3 -u examples/synthesis/inv_pend_syn/main.py --train 1
```

View logs:

```bash
docker logs -f invpend_run
```

Graceful stop:

```bash
docker stop -t 10 invpend_run
```

Force stop if stuck:

```bash
docker kill invpend_run
```

Remove container (to reuse name on rerun):

```bash
docker rm invpend_run
```

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
