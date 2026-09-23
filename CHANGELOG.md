# Changelog

## 0.4.0 — 2026-09-23

Changes from an adversarial review of 0.3.

### Fixed
- Gradients were taken through `jnp.linalg.eigh` and were NaN on any degenerate spectrum (every SU(2)-symmetric chain). The default backend is now the exact Fréchet derivative of e^{−βH} with degenerate-limit divided differences; `backend="jax"` differentiates through `expm`.
- A restart that returned NaN froze the result at the initial guess, because `x < NaN` is always false. Non-finite restarts are skipped, non-finite parameters are rejected by the objective, and a `RuntimeError` is raised if every restart fails.
- Process tomography returned unphysical Choi matrices (negative eigenvalues, F_pro > 1). Estimates are projected onto CPTP channels with Dykstra's algorithm by default; the raw estimate is kept as `ptm_raw` / `choi_raw`. Added `ptm_from_choi`, `is_physical`, `project_cptp`.
- The EKF updated sequentially with independent-noise assumptions and a non-Joseph covariance update. It now makes one joint update per snapshot with the exact outcome covariance and the Joseph form, and uses the exact Jacobian.

### Added
- Matchgate shadows: `FermionicGaussianState` and an O(n³)-per-snapshot covariance sampler (no 2ⁿ objects), degree-4 Majorana estimators, `estimate_2rdm`, exact references via Wick's theorem, `two_body_energy`.

### Changed
- Core dependencies are NumPy and SciPy; Cirq and JAX are optional extras; PennyLane and matplotlib removed from requirements.
- Tests: 45 → 59.

## 0.3.0 — 2026-09-21

### Fixed (correctness)
- `DerandomizedShadowSelector` now implements Huang–Kueng–Preskill Algorithm 1; v0.2 returned the same basis for every shot (for targets ZZ, XX, YY only XX was ever measured).
- `DifferentiableHamiltonianLearner` is differentiable: JAX gradients through `eigh` + L-BFGS-B (v0.2: Nelder–Mead), seeded, with a NumPy fallback.
- SPAM mitigation inverts the confusion matrix at the single-shot level and is unbiased for asymmetric readout errors; the multiplicative rule (exact only for symmetric errors) is kept as `mitigate_pauli_expectation`.
- Benchmark and CLI learn from shadows of the Gibbs state of the Hamiltonian being learned (v0.2 used a Bell state and printed "True: −1.00").
- `FermionicMatchgateShadows` and `ShadowProcessTomographer` were stubs; both are real (see Added).
- `StreamingKalmanHamiltonianTracker` is an EKF over the Gibbs observable model; v0.2 fed the parameter itself in as the measurement.
- Fast sampler: one simulation per state instead of one per snapshot; verified identical in distribution.

### Added
- `sample_shadows_from_density_matrix` (fixed bases, readout errors), `gibbs_state`, `basis_rotation_matrix`.
- Vectorised estimators, `snapshots_to_arrays`, `derandomized=` and `readout_model=` options, `hits_per_observable`.
- `learn_from_expectations`, `identifiability()` (Jacobian SVD), `HamiltonianLearningResult.method / h_error`.
- Matchgate shadows: Majoranas, signed-permutation sampling, `estimate_majorana_pair`, `estimate_1rdm`, `exact_1rdm`, `slater_state`.
- Shadow QPT: `run(channel)`, `pauli_transfer_matrix_of_unitary`, `choi_matrix`, `process_fidelity`, `average_gate_fidelity`.
- CLI `learn --derandomized --spam`, `derand-compare`, `qpt`, `track`, `benchmark`, `--json`.
- 45 tests (from 14); browser demo computes shadows, inversion and tracking live.

## 0.2.0
- Derandomised selector, Kalman tracker, fermionic matchgates, process tomography, CLI, CI, notebook.

## 0.1.0
- Initial release.
