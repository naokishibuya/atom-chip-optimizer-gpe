# 🧲 Atom Chip Optimizer — Surrogate-Assisted GPE

[![arXiv](https://img.shields.io/badge/arXiv-2609.13762-b31b1b.svg)](https://arxiv.org/abs/2609.13762)
[![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2609.13762-informational)](https://doi.org/10.48550/arXiv.2609.13762)

> Successor to [atom-chip-optimizer](https://github.com/naokishibuya/atom-chip-optimizer)
> ([arXiv:2508.11712](https://arxiv.org/abs/2508.11712)), extending the
> inverse-optimization framework with Gaussian phase-space screening,
> Thomas–Fermi scaling estimates for grid sizing, and three-dimensional
> Gross–Pitaevskii characterization of condensate transport.

`atom-chip-optimizer-gpe` designs wire-current schedules for BEC transport on an atom chip and characterizes the condensate dynamics. The workflow combines regularized inverse optimization, Gaussian phase-space screening, surrogate-assisted grid sizing, and three-dimensional GPE simulation of selected schedules.

## 📄 Paper

**Surrogate-Assisted three-dimensional Gross–Pitaevskii Characterization of Bose–Einstein condensate transport on an atom chip**  
Naoki Shibuya (2026)

- Read on arXiv: https://arxiv.org/abs/2609.13762
- PDF: https://arxiv.org/pdf/2609.13762

### Cite this work

```bibtex
@misc{shibuya2026surrogate,
  title         = {Surrogate-assisted three-dimensional {Gross--Pitaevskii} characterization of {Bose--Einstein} condensate transport on an atom chip},
  author        = {Naoki Shibuya},
  year          = {2026},
  eprint        = {2609.13762},
  archivePrefix = {arXiv},
  primaryClass  = {cond-mat.quant-gas},
  doi           = {10.48550/arXiv.2609.13762}
}
```

## 📝 Prerequisites

- Install [uv](https://docs.astral.sh/uv/) and Python 3.12 or later.
- A CUDA GPU is required for `chip init`, `chip plan`, and `chip eval`.
- `chip scan` and `chip plot` also run on CPU. Set `JAX_PLATFORMS=cpu` to prevent them from initializing the GPU.

## ⚙️ Installation

Clone the repository and install the dependencies recorded in `uv.lock`:

```bash
git clone https://github.com/naokishibuya/atom-chip-optimizer-gpe.git
cd atom-chip-optimizer-gpe
uv sync --locked
```

## ✨ Included data

The `data/` directory contains the chip configuration, planned current schedules, surrogate scans, and saved GPE and linear evaluation results used for the paper. Numerical arrays and metadata are included; generated figures and logs are omitted.

To regenerate the figures from the saved results, run from the repository root:

```bash
JAX_PLATFORMS=cpu uv run chip plot data
```

Figures are written to `figures/` directories within `data/`. This command does not rerun transport simulations; it also regenerates the sweep-level `fidelity.csv` summaries. To plot one evaluated case instead:

```bash
JAX_PLATFORMS=cpu uv run chip plot data/sweep/reg-0.005/N-1000/T-2.0
```

Use a separate experiment directory for new calculations so that the bundled results remain unchanged.

## 🚀 Run a new experiment

From the repository root, create an experiment directory and initialize it:

```bash
mkdir -p runs/exp-01
cd runs/exp-01

uv run chip init
uv run chip plan                                # defaults to 7 regularizations × 6 durations; customizable
uv run chip scan                                # compare candidates using the surrogate
uv run chip eval plan/reg-0.001 --N 1e3 --T 1.5   # evaluate one chosen schedule (GPE)
uv run chip plot                                # generate figures from saved data
```

- `plan`: generate current schedules for the requested regularizations and transport durations.
- `scan`: compare the saved schedules using surrogate endpoint-fidelity estimates.
- `eval`: run a GPE simulation for the specified schedules and atom numbers, automatically choosing the grid and propagation steps.
- `plot`: generate figures from saved results without rerunning simulations.

See `uv run chip <command> --help` for defaults, accepted syntax (space-separated values, `START:STOP:STEP` ranges, scientific notation), and units.

## 🗂️ Data Directory Structure

Both `data/` and newly created experiment directories use this structure.

```
data/
├── chip.json
├── info.txt
├── plan/reg-<reg>/T-<T>/
│   ├── trajectory.npz
│   └── trajectory.json
├── scan/
│   ├── harmonic.npz
│   ├── harmonic.json
│   └── N-<N>/
│       ├── scan.npz
│       └── scan.json
└── sweep/reg-<reg>/
    ├── fidelity.csv
    └── N-<N>/
        ├── info.txt
        └── T-<T>/
            └── {gpe,linear}/
                ├── eval.npz
                └── eval.json
```

- NPZ files store numerical arrays, while JSON files store configuration, metadata, and result summaries.
- Experiment commands write logs alongside their outputs, and `chip plot` generates figures in `figures/` subdirectories.

## 🙏 Credit

This work extends the author's MSc thesis at the University of Sussex. The original atom-chip MATLAB trap model was provided by Dr Fedja Oručević and the Quantum Systems and Devices laboratory at the University of Sussex.
