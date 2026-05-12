# KINNs vs. MLP-PINNs: Can B-Splines Overcome Spectral Bias on Stiff PDEs?

Comparison of Kolmogorov-Arnold Network PINNs (KINNs) against MLP-based PINNs on stiff PDEs. Main experiments cover 1D Convection and 2D Helmholtz across multiple difficulty levels.

## Key Findings

- KINNs achieve **2–3× lower L₂ error** using only **600 vs. 50,049 parameters** on smooth physics
- Both architectures fail at high stiffness (β ≥ 30) under fixed-weight training — the backbone alone does not fix gradient pathology
- KINNs introduce a **B-spline grid under-resolution failure mode** unique to KINNs — grid must satisfy **gs ≥ 4n** for oscillatory PDEs; no such constraint exists for MLPs
- **Adaptive gradient weighting** is the single most effective intervention, recovering boundary condition enforcement within **2,000 epochs** for both architectures
- At β = 50, MLP-adaptive reaches L₂ = 0.021 while KINN-adaptive remains at 0.70 regardless of grid size

## Results

**1D Convection**

| β | MLP fixed | MLP adaptive | KINN best |
|---|-----------|--------------|-----------|
| 1  | 0.0010 | 0.0011 | 0.0003 |
| 10 | 0.0015 | 0.0028 | 0.0012 |
| 30 | 0.0026 | 0.0035 | 0.6088 |
| 50 | 0.8562 | 0.0207 | 0.7808 |

**2D Helmholtz (adaptive weighting)**

| n | MLP adaptive | KINN gs=10 | KINN gs=30 |
|---|-------------|------------|------------|
| 1 | 2.1 × 10⁻⁵ | 3.1 × 10⁻⁴ | 1.3 × 10⁻³ |
| 4 | 0.0017 | 0.1735 | 0.0302 |
| 8 | 0.0385 | 0.6735 | 0.5135 |

## Repo Structure

```
├── train.py          # Models, training loop, all experiment runners
├── plot.py           # Figure generation from saved CSV results
├── results/
│   ├── *.csv         # Experiment results
│   └── figures/      # Generated plots
└── README.md
```

## Setup

```bash
pip install torch numpy pandas matplotlib tqdm efficient-kan
```

Trained on Google Colab with GPU. `DriveCheckpoint` saves checkpoints to Google Drive automatically — falls back gracefully when run locally.

## Running

**Training** — requires GPU, takes several hours:
```bash
python train.py
```

**Plotting** — CSVs are already in `results/` so figures can be regenerated without retraining:
```bash
python plot.py
```

> `plot_solutions()` in `plot.py` requires `MLP` and `KINN` classes from `train.py` to be in scope. All other plot functions work standalone from the CSVs.

## Model Details

**MLP-PINN** — 4 hidden layers, 128 neurons each, tanh activation, Xavier initialization. 50,049 parameters.

**KINN** — Layout [2, 5, 5, 1], spline order k=3. Grid sizes gs ∈ {5, 10, 20, 50} for Convection, gs ∈ {10, 30} for Helmholtz. 400–2,200 parameters.

Three mandatory KINN modifications (Wang et al. 2024):
1. Input normalized to [−1, 1] — B-splines are only valid within the grid range
2. tanh after each hidden layer — prevents second-order autograd derivatives from collapsing to zero
3. Spline order fixed at k=3 — accuracy does not improve beyond this

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Interior collocation points | 50,000 |
| IC / BC collocation points | 500 each |
| Validation grid | 256 × 256 |
| Adam epochs | 10,000 |
| Adam learning rate | 1 × 10⁻³ |
| LR decay (exponential) | γ = 0.9995 |
| L-BFGS outer steps | 200 |
| L-BFGS max inner iterations | 50 |
| Gradient clip (Adam) | ‖·‖₂ ≤ 1.0 |
| Early stopping patience | 5,000 epochs |

## References

- Raissi et al. (2019) — Physics-Informed Neural Networks, *J. Comput. Phys.*
- Liu et al. (2024) — KAN: Kolmogorov-Arnold Networks, *arXiv:2404.19756*
- Wang et al. (2024) — Kolmogorov Arnold Informed Neural Network, *arXiv:2406.11045*
- Wang et al. (2021) — Gradient flow pathologies in PINNs, *SIAM J. Sci. Comput.*
- Krishnapriyan et al. (2021) — Failure modes in PINNs, *NeurIPS*
- Rahaman et al. (2019) — Spectral bias of neural networks, *ICML*
