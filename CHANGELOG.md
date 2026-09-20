# Changelog

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
