# shadow-hamiltonian-learning

Classical shadows for learning and tracking the Hamiltonian of a small quantum device.

[![CI](https://github.com/Jaspersands/shadow-hamiltonian-learning/actions/workflows/ci.yml/badge.svg)](https://github.com/Jaspersands/shadow-hamiltonian-learning/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

A classical shadow turns single-shot measurements in random bases into unbiased estimates of many observables at once. The package covers:
- sampling shadows and choosing bases deliberately to save shots
- correcting asymmetric readout errors exactly
- recovering couplings from the observables of a thermal state
- tracking drifting parameters from a stream of snapshots
- fermionic (matchgate) shadows, and process tomography constrained to physical channels

[Interactive page](web/index.html): shadows of a thermal Heisenberg chain in the browser, with the inversion, basis comparison and Kalman filter running in a background worker.

## Modules

| Module | Contents |
|---|---|
| `cirq_shadows` | Pauli shadows from a Cirq circuit or a density matrix (random or given bases, optional readout errors), thermal states |
| `reconstruction` | Unbiased estimators for random and chosen bases, median of means, reduced density matrices |
| `derandomized` | Huang–Kueng–Preskill derandomisation with log-space weights |
| `spam_mitigation` | Per-shot readout correction g(b) = ((Mᵀ)⁻¹(1, −1))_b, unbiased for asymmetric errors |
| `differentiable_inversion` | Recovers (J, h) of H = Σ J_ij σ_i·σ_j + Σ h_i Z_i with exact gradients, plus an identifiability analysis |
| `kalman_tracker` | Extended Kalman filter over (J, h) that consumes single snapshots |
| `fermionic_shadows` | Matchgate shadows: 1- and 2-RDMs, a dense sampler for general states and a covariance-matrix sampler for Gaussian states |
| `process_tomography` | Shadow process tomography: Pauli transfer matrix, Choi matrix, CPTP projection, fidelities |
| `cli` | `shadow-learn learn | derand-compare | qpt | track | benchmark` |

## Details that matter

- **Degenerate spectra.** Any SU(2)-symmetric chain has exactly degenerate multiplets, and differentiating an eigendecomposition returns NaN there because the eigenvector derivative divides by $w_i - w_j$. The gradient here is the Fréchet derivative of $e^{-\beta H}$ in the eigenbasis. The divided difference $(e^{-\beta w_i} - e^{-\beta w_j})/(w_i - w_j)$ is replaced by its limit $-\beta e^{-\beta w_i}$ when levels coincide, so it stays exact. An optional JAX backend differentiates through `expm` and agrees to $10^{-15}$. Restarts whose loss is not finite are discarded rather than allowed to win a comparison against NaN.
- **Identifiability.** A cold, singlet-dominated thermal state screens a uniform field, so h₀ + h₁ is nearly invisible while h₀ − h₁ and J are well determined. `identifiability()` reports the Jacobian's singular values before any fitting.
- **The Kalman filter.** One snapshot gives correlated ±1 outcomes for all compatible observables. The update uses their exact covariance ⟨o_a o_b⟩ − ⟨o_a⟩⟨o_b⟩ and the Joseph-form covariance update. It is a Gaussian approximation to a discrete measurement; the tests check that the reported uncertainties match the actual errors.
- **Matchgate shadows.** A product of 2k Majoranas is determined by a snapshot when it is a union of measured pairs, which happens with probability $\binom{n}{k}/\binom{2n}{2k}$. That gives unbiased 1-RDM and 2-RDM estimates, and with them the energy of any two-body fermionic Hamiltonian. For Gaussian states the sampler works on the $2n \times 2n$ covariance matrix with exact rank-2 measurement updates, so 24 modes take about a second and no $2^n$ object is built.
- **Physical process estimates.** Linear inversion from finite data gives a Choi matrix with negative eigenvalues and fidelities above 1. By default the estimate is projected onto CPTP channels with Dykstra's algorithm, which can only reduce its distance to the true channel. Fidelities from the projection are biased low at small budgets; the raw estimate is returned too.

## Examples

```python
import numpy as np
from shadow_learning import (
    DifferentiableHamiltonianLearner, gibbs_state, sample_shadows_from_density_matrix,
    DerandomizedShadowSelector, estimate_many_observables, ReadoutErrorModel, ShadowProcessTomographer,
)
from shadow_learning.fermionic_shadows import FermionicGaussianState, FermionicMatchgateShadows

# Shadows of a thermal state, then inversion
learner = DifferentiableHamiltonianLearner(n_qubits=2, beta=0.6, seed=1)
J = np.array([[0, 0.4], [0.4, 0]]); h = np.array([0.5, -0.3])
rho = gibbs_state(learner.build_hamiltonian_matrix(J, h), beta=0.6)
snaps = sample_shadows_from_density_matrix(rho, 8000, np.random.default_rng(0))
res = learner.learn_from_shadows(snaps, true_j_matrix=J, true_h_vector=h)
print(res.recovered_j_matrix[0, 1], res.recovered_h_vector)          # about 0.42, [0.55 −0.32] (truth 0.4, [0.5 −0.3])

# Chosen bases for a fixed set of observables
targets = [learner.pauli_string(n) for n in learner.observable_names]
bases = DerandomizedShadowSelector(targets, n_qubits=2).select_measurement_bases(1000)
print(estimate_many_observables(sample_shadows_from_density_matrix(rho, 1000, np.random.default_rng(1), bases=bases),
                                targets, derandomized=True))

# Asymmetric readout errors, corrected per shot
model = ReadoutErrorModel(p01=0.02, p10=0.10)
noisy = sample_shadows_from_density_matrix(rho, 6000, np.random.default_rng(2), readout_error=model)
print(estimate_many_observables(noisy, ["ZZ"], readout_model=model))

# A 24-mode free-fermion state from its covariance matrix
g = FermionicGaussianState.ground_state_of_quadratic(np.diag(np.ones(23), 1) + np.diag(np.ones(23), -1), 12)
fs = FermionicMatchgateShadows(n_modes=24, seed=0)
print(np.abs(fs.estimate_1rdm(fs.sample(g, 3000)) - g.one_rdm()).max())

# Process tomography, projected onto physical channels
X = np.array([[0, 1], [1, 0]], dtype=complex)
out = ShadowProcessTomographer(n_qubits=1, seed=0).run(lambda r: X @ r @ X, n_snapshots_per_input=1500)
print(np.linalg.eigvalsh(out["choi_raw"]).min(), np.linalg.eigvalsh(out["choi"]).min())
```

```bash
shadow-learn learn --qubits 3 --shots 6000 --derandomized --spam 0.02 0.08
shadow-learn qpt --channel depolarizing --p 0.2
shadow-learn benchmark --quick
```

## Install

```bash
pip install -e ".[dev]"          # the core needs NumPy and SciPy; Cirq and JAX are optional extras
pytest tests/                    # 59 tests
python benchmarks/run_shadow_benchmark.py
```

The notebook [`notebooks/04_derandomized_classical_shadow_tomography.ipynb`](notebooks/04_derandomized_classical_shadow_tomography.ipynb) has executed outputs. Changes are listed in [CHANGELOG.md](CHANGELOG.md).

Apache-2.0 · Jasper Sands
