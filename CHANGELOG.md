# Changelog

## 0.5.0 — 2026-10-19

Learning a Rydberg Hamiltonian from computational-basis snapshots of quench dynamics, the readout available on an analog neutral-atom device.

### Added
- `rydberg` module:
  - The model and its constants:
    - Ω, Δ, C6 and optional per-site detuning offsets.
    - Aquila specifications, `AquilaSpecs`: `AQUILA` from Wurtz et al. (2023), the reference, and `AQUILA_BRAKET` from the current Braket device properties, which have larger readout errors and shorter coherence times. `DeviceNoise.aquila()` and `DeviceNoise.braket()` give the matching noise models.
    - C6 for ⁸⁷Rb 70S from Bloqade.
    - Chain, ring and square geometries snapped to the 0.01 μm position grid, and a check against Aquila's layout rules.
  - Pulses:
    - The default pulse is the shortest one Braket accepts: Ω ramps linearly from 0 over 0.05 μs, stays flat and ramps back to 0 at the probe time, with constant Δ. A probe time is a total pulse duration, and atoms are read out after Ω is back at 0.
    - `waveform` returns the Braket time series; `waveform_violations` checks Ω and Δ ranges, the Ω slew limit, the minimum time step, the 1 ns grid and the 4 μs limit.
    - `ramp_time=0` gives an instantaneous quench for reference.
  - Simulation:
    - Exact dynamics, with the ramps as midpoint time slices, and exact gradients through the slices that stay finite at degenerate levels.
    - Sparse evolution for 11 to 14 atoms.
    - Readout errors, per-shot position jitter, quasi-static drive noise, a phenomenological relaxation model, and empty sites (`p_vacancy`).
  - Data: `QuenchData.from_braket` converts `post_sequence` to bits, drops sites requested empty (`filling`) and failed shots (`statuses`), and drops shots whose `pre_sequence` differs from the requested filling.
  - Information and design:
    - Bitstring and correlator Fisher information, Cramér–Rao bounds and identifiability. Inverses are taken in relative units, since C6 is about 10⁶ times Ω.
    - Shots for a target precision.
    - Probe-time design by coordinate exchange.
  - Fitting and tracking:
    - Maximum-likelihood and least-squares fits, both with time continuation that starts from Ω alone on the earliest probe time.
    - The likelihood fit also tries seeded extra starts, the opposite sign of Δ, and every neighbouring maximum along C6 within ±25 %. It reports a goodness-of-fit test (`goodness_of_fit`), other optima that fit almost as well, and `warnings`; `success` is false if any check fails.
    - `RydbergDriftTracker`, a per-shot extended Kalman filter.
- `benchmarks/make_figures.py` and `figures/`. These cover identifiability and exact symmetries, estimators against the Cramér–Rao bound, shots against probe times and their sensitivity, the noise budget, fit convergence, systematic errors from ramps, position jitter, empty sites and readout, and drift tracking. The numbers are in `figures/findings.json`.
- `shadow-learn rydberg` subcommand, with `--ramp` and `--device {datasheet,braket}`, and warnings when a fit fails a check or the data have position jitter. Benchmark section 8 covers the Rydberg quench.
- `kalman_tracker.joseph_update`, shared by both trackers.

### Fixed
- `benchmark.py` used an f-string with nested quotes that only parses on Python 3.12. It failed to import on 3.10 and 3.11.

### Changed
- README leads with what the data can and cannot determine, and has a References section.
- The `slow` pytest marker is registered, and the Monte Carlo tests carry it.
- The Jacobian tests compare each column of θ_a ∂q/∂θ_a with a relative finite difference, with a tolerance per column, so the C6 column (entries near 10⁻⁷ in absolute units) is tested.
- Tests: 100, up from 59.

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
