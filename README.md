# shadow-hamiltonian-learning

**Classical shadows → Hamiltonian learning: randomised, derandomised and matchgate shadows, exact readout-error inversion, JAX-differentiable Gibbs-state inversion, streaming EKF tracking and shadow process tomography.**

[![CI](https://github.com/Jaspersands/shadow-hamiltonian-learning/actions/workflows/ci.yml/badge.svg)](https://github.com/Jaspersands/shadow-hamiltonian-learning/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Cirq](https://img.shields.io/badge/shadows-Cirq-teal.svg)](https://quantumai.google/cirq)
[![JAX](https://img.shields.io/badge/inversion-JAX-red.svg)](https://github.com/google/jax)

**[▶ Interactive demo](web/index.html)** — the browser builds a Gibbs state, samples Pauli shadows (random or derandomised, with optional readout error), estimates observables, inverts (J, h) by gradient descent and tracks a drifting coupling with an EKF. Every number is computed on the page.

---

## What it does

| Module | Capability |
|---|---|
| `cirq_shadows` | Randomised Pauli shadows from a Cirq circuit (single simulation, then O(2ⁿ) per snapshot), `sample_shadows_from_density_matrix` (fixed or random bases, readout errors), `gibbs_state` |
| `reconstruction` | Vectorised unbiased estimators (3ᵏ-weighted for random bases; plain means for derandomised bases), median-of-means, reduced density matrices |
| `derandomized` | **Huang–Kueng–Preskill derandomisation** (Algorithm 1 with the confidence-bound cost) — covers every target and beats random shadows at equal budget |
| `spam_mitigation` | Confusion-matrix inversion at the single-shot level: replaces (−1)ᵇ by g(b) = ((Mᵀ)⁻¹(1, −1))_b, unbiased for **asymmetric** readout errors |
| `differentiable_inversion` | `DifferentiableHamiltonianLearner`: Gibbs observables of H = Σ J_ij σ_i·σ_j + Σ h_i Z_i, loss with **JAX gradients through `eigh`**, L-BFGS-B, NumPy fallback, `identifiability()` (Jacobian SVD) |
| `kalman_tracker` | EKF over (J, h) whose observation model is the nonlinear Gibbs expectation; consumes single snapshots and tracks drift |
| `fermionic_shadows` | Matchgate shadows (Wan et al. 2022): random signed-permutation Clifford matchgates, Majorana pair projectors, unbiased ⟨iγ_uγ_v⟩ with λ₁ = 1/(2n−1), full 1-RDM |
| `process_tomography` | Shadow QPT: Pauli-eigenstate inputs → Pauli transfer matrix → Choi matrix, process and average gate fidelities |
| `cli` | `shadow-learn learn | derand-compare | qpt | track | benchmark` (all `--json`) |

## Quickstart

```python
import numpy as np
from shadow_learning import (
    DifferentiableHamiltonianLearner, gibbs_state, sample_shadows_from_density_matrix,
    DerandomizedShadowSelector, estimate_many_observables, ReadoutErrorModel,
    StreamingKalmanHamiltonianTracker, FermionicMatchgateShadows, slater_state, ShadowProcessTomographer,
)

# 1. A Gibbs state of a 2-qubit Heisenberg chain, and shadows of it
learner = DifferentiableHamiltonianLearner(n_qubits=2, beta=0.6, seed=1)
J = np.array([[0, 0.4], [0.4, 0]]); h = np.array([0.5, -0.3])
rho = gibbs_state(learner.build_hamiltonian_matrix(J, h), beta=0.6)
rng = np.random.default_rng(0)
snaps = sample_shadows_from_density_matrix(rho, 8000, rng)

# 2. Differentiable inversion (JAX through eigh + L-BFGS-B)
res = learner.learn_from_shadows(snaps, true_j_matrix=J, true_h_vector=h)
print(res.recovered_j_matrix[0, 1], res.recovered_h_vector, res.method)   # 0.400 [0.49 -0.33] jax-lbfgs
print(learner.identifiability(J, h)["condition_number"])                   # which directions the data resolves

# 3. Derandomised shadows for a fixed observable set
targets = [learner.pauli_string(n) for n in learner.observable_names]
bases = DerandomizedShadowSelector(targets, n_qubits=2).select_measurement_bases(1000)
der = sample_shadows_from_density_matrix(rho, 1000, rng, bases=bases)
print(estimate_many_observables(der, targets, derandomized=True))

# 4. Asymmetric readout errors, exactly de-biased at the single-shot level
model = ReadoutErrorModel(p01=0.02, p10=0.10)
noisy = sample_shadows_from_density_matrix(rho, 6000, rng, readout_error=model)
print(estimate_many_observables(noisy, ["ZZ"], readout_model=model))

# 5. EKF over a stream of single snapshots
tracker = StreamingKalmanHamiltonianTracker(n_qubits=2, beta=0.6, process_noise_std=0.005)
for snap in snaps[:2000]:
    est = tracker.process_snapshot(snap)
print(est["J"][0, 1], est["std"])

# 6. Matchgate shadows → 1-RDM, and shadow process tomography
fs = FermionicMatchgateShadows(n_modes=4, seed=3)
print(np.round(np.diag(fs.estimate_1rdm(fs.sample(slater_state(4, [1, 3]), 4000))).real, 2))  # ≈ [0 1 0 1]
tomo = ShadowProcessTomographer(n_qubits=1, seed=0)
X = np.array([[0, 1], [1, 0]], dtype=complex)
r = tomo.run(lambda rho_in: X @ rho_in @ X, n_snapshots_per_input=1500)
print(tomo.process_fidelity(r["ptm"], X))                                   # ≈ 1
```

```bash
shadow-learn learn --qubits 3 --shots 6000 --derandomized --spam 0.02 0.08
shadow-learn derand-compare --qubits 3 --shots 300
shadow-learn qpt --channel depolarizing --p 0.2
shadow-learn benchmark --quick
```

## Honest notes

* **Identifiability is physics, not a bug.** A singlet-dominated Gibbs state (large βJ) screens a uniform field, so h₀ + h₁ is nearly unlearnable from Pauli shadows while h₀ − h₁ and J are fine. `identifiability()` reports the Jacobian singular values so you can see it before fitting; the tutorial shows the hot (β = 0.3) vs cold (β = 1.5) case.
* The v0.2 "derandomiser" returned the same basis for every shot (two of three targets were never measured), the "differentiable" learner used Nelder–Mead, the SPAM rule was only correct for symmetric errors, the matchgate and process-tomography modules were stubs, and the benchmark learned a Hamiltonian from a Bell state. All of that is fixed and tested in v0.3.
* Matchgate shadows use the Clifford-matchgate (signed permutation) ensemble; only the degree-2 Majorana sector (1-RDM) is estimated. 2-RDMs need the λ₂ = C(n,2)/C(2n,4) sector and are not implemented.

## Install & test

```bash
pip install -e ".[dev]"
pytest -v tests/                              # 45 tests
python benchmarks/run_shadow_benchmark.py     # budget scaling → derandomisation → SPAM → learning → EKF → 1-RDM → QPT
```

Cirq is only needed for `ClassicalShadowsProtocol` (circuit inputs); everything else is NumPy, with JAX optional for gradients.

## Tutorial

[`notebooks/04_derandomized_classical_shadow_tomography.ipynb`](notebooks/04_derandomized_classical_shadow_tomography.ipynb) — executed outputs included.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Apache-2.0 — Jasper Sands.
