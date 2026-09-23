"""
Benchmark: estimator accuracy vs budget, derandomisation gain, SPAM mitigation,
honest Gibbs-state Hamiltonian learning, EKF tracking, matchgate 1-RDM, QPT.
Exposed as ``shadow-learn benchmark``.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict

import numpy as np

from .cirq_shadows import gibbs_state, sample_shadows_from_density_matrix
from .reconstruction import estimate_many_observables, estimate_pauli_string_expectation
from .differentiable_inversion import DifferentiableHamiltonianLearner
from .derandomized import DerandomizedShadowSelector
from .spam_mitigation import ReadoutErrorModel


def run_shadow_benchmark(as_json: bool = False, quick: bool = False) -> int:
    out: Dict[str, Any] = {}
    log = [] if as_json else None

    def say(msg: str = "") -> None:
        (log.append(msg) if log is not None else print(msg))

    say("=" * 72)
    say("CLASSICAL SHADOWS → HAMILTONIAN LEARNING BENCHMARK (v0.4)")
    say("=" * 72)
    rng = np.random.default_rng(42)

    say("\n1. GHZ₃ observable error vs snapshot budget (random Pauli shadows)")
    psi = np.zeros(8, dtype=complex); psi[0] = psi[7] = 1 / np.sqrt(2)
    rho_ghz = np.outer(psi, psi.conj())
    rows = {}
    for S in ([50, 200, 800] if quick else [50, 200, 800, 3200]):
        t0 = time.time()
        snaps = sample_shadows_from_density_matrix(rho_ghz, S, rng)
        e = estimate_many_observables(snaps, ["ZZI", "IZZ", "XXX", "ZII"])
        rows[S] = e
        say(f"   S = {S:5d}: <ZZI> = {e['ZZI']:+.3f}  <IZZ> = {e['IZZ']:+.3f}  <XXX> = {e['XXX']:+.3f}  <ZII> = {e['ZII']:+.3f}  (truth 1, 1, 1, 0)  {1e3*(time.time()-t0):.0f} ms")
    out["ghz"] = rows

    say("\n2. Derandomised vs random shadows, 3-qubit Gibbs state, 6 bond observables, 300 shots")
    learner3 = DifferentiableHamiltonianLearner(n_qubits=3, beta=1.0)
    J3 = np.array([[0, 0.7, 0], [0.7, 0, -0.4], [0, -0.4, 0]]); h3 = np.array([0.2, 0.0, -0.3])
    rho3 = gibbs_state(learner3.build_hamiltonian_matrix(J3, h3), 1.0)
    targets = ["XXI", "YYI", "ZZI", "IXX", "IYY", "IZZ"]
    truth = {P: float(np.real(np.trace(learner3.pauli_matrix(P) @ rho3))) for P in targets}
    bases = DerandomizedShadowSelector(targets, 3).select_measurement_bases(300)
    rr, rd = [], []
    for _ in range(6 if quick else 15):
        er = estimate_many_observables(sample_shadows_from_density_matrix(rho3, 300, rng), targets)
        ed = estimate_many_observables(sample_shadows_from_density_matrix(rho3, 300, rng, bases=bases), targets, derandomized=True)
        rr.append(np.sqrt(np.mean([(er[P] - truth[P]) ** 2 for P in targets]))); rd.append(np.sqrt(np.mean([(ed[P] - truth[P]) ** 2 for P in targets])))
    say(f"   RMSE random {np.mean(rr):.4f} | derandomised {np.mean(rd):.4f} | gain ×{np.mean(rr)/np.mean(rd):.2f}")
    out["derandomisation"] = {"random": float(np.mean(rr)), "derandomized": float(np.mean(rd))}

    say("\n3. SPAM: asymmetric readout p01 = 0.02, p10 = 0.10 on |+⟩, 6000 shots")
    plus = np.array([1, 1]) / np.sqrt(2); rho_p = np.outer(plus, plus)
    model = ReadoutErrorModel(0.02, 0.10)
    snaps = sample_shadows_from_density_matrix(rho_p, 6000, rng, readout_error=model)
    raw = estimate_pauli_string_expectation(snaps, "X"); mit = estimate_pauli_string_expectation(snaps, "X", readout_model=model)
    say(f"   <X> raw {raw:.3f} | symmetric rule {raw/model.readout_fidelity:.3f} (overshoots) | single-shot inversion {mit:.3f} (truth 1)")
    out["spam"] = {"raw": raw, "mitigated": mit}

    say("\n4. Hamiltonian learning from Gibbs-state shadows (2 qubits, β = 0.6, 8000 shots) — L-BFGS, exact gradients")
    learner = DifferentiableHamiltonianLearner(n_qubits=2, beta=0.6, seed=1)
    J = np.array([[0, 0.4], [0.4, 0]]); h = np.array([0.5, -0.3])
    rho = gibbs_state(learner.build_hamiltonian_matrix(J, h), 0.6)
    t0 = time.time()
    res = learner.learn_from_shadows(sample_shadows_from_density_matrix(rho, 8000, rng), true_j_matrix=J, true_h_vector=h)
    say(f"   J01 {res.recovered_j_matrix[0,1]:+.3f} (0.400)  h {np.round(res.recovered_h_vector, 3)} ({h})  ‖ΔJ‖ = {res.frobenius_error:.3f}  {res.iterations} it, {time.time()-t0:.2f} s, {res.method}")
    ident = learner.identifiability(J, h)
    say(f"   identifiability: singular values {np.round(ident['singular_values'], 3)}, condition number {ident['condition_number']:.1f}")
    out["learning"] = {"J01": float(res.recovered_j_matrix[0, 1]), "h": res.recovered_h_vector.tolist(), "iterations": res.iterations}
    L4 = DifferentiableHamiltonianLearner(n_qubits=4, beta=1.0, seed=0)
    J4 = np.diag([0.5, 0.5, 0.5], 1); J4 = J4 + J4.T; h4 = np.zeros(4)
    r4 = L4.learn_from_expectations(L4.compute_thermal_observables(J4, h4), J4, h4)
    gap = np.min(np.diff(np.linalg.eigvalsh(L4.build_hamiltonian_matrix(J4, h4))))
    say(f"   isotropic 4-site chain (h = 0, {"exactly degenerate levels" if gap < 1e-12 else f"min level spacing {gap:.1e}"}): ‖ΔJ‖ = {r4.frobenius_error:.1e}, all gradients finite")
    out["learning_degenerate"] = {"frobenius_error": r4.frobenius_error}

    say("\n5. EKF tracking a step 0.3 → 0.8 in J01 from 3000 single snapshots")
    from .kalman_tracker import StreamingKalmanHamiltonianTracker
    tr = StreamingKalmanHamiltonianTracker(n_qubits=2, beta=1.5, process_noise_std=0.005, seed=0)
    t0 = time.time(); est = []
    for k in range(3000):
        jv = 0.3 if k < 1500 else 0.8
        if k % 250 == 0:
            rho_k = gibbs_state(tr.learner.build_hamiltonian_matrix(np.array([[0, jv], [jv, 0]]), np.array([0.2, -0.15])), 1.5)
        tr.process_snapshot(sample_shadows_from_density_matrix(rho_k, 1, rng)[0]); est.append(tr.x[0])
    say(f"   estimate at shot 1400: {est[1399]:+.3f} (0.3) | at 3000: {est[-1]:+.3f} (0.8)  ({time.time()-t0:.1f} s)")
    out["ekf"] = {"at_1400": est[1399], "final": est[-1]}

    say("\n6. Matchgate shadows")
    from .fermionic_shadows import FermionicGaussianState, FermionicMatchgateShadows, slater_state
    n_s = 2000 if quick else 4000
    fs = FermionicMatchgateShadows(n_modes=4, seed=3)
    psi = (slater_state(4, [1, 3]) + slater_state(4, [0, 2])) / np.sqrt(2)
    t0 = time.time(); snaps = fs.sample(psi, 4 * n_s)
    e1 = np.abs(fs.estimate_1rdm(snaps) - fs.exact_1rdm(psi)).max(); e2 = np.abs(fs.estimate_2rdm(snaps) - fs.exact_2rdm(psi)).max()
    say(f"   non-Gaussian 4-mode state, {4*n_s} snapshots: max |ΔD1| = {e1:.3f}, max |ΔD2| = {e2:.3f}  ({time.time()-t0:.1f} s)")
    hq = rng.normal(size=(24, 24)); g24 = FermionicGaussianState.ground_state_of_quadratic(hq + hq.T, 12)
    fs24 = FermionicMatchgateShadows(n_modes=24, seed=0)
    t0 = time.time(); D24 = fs24.estimate_1rdm(fs24.sample(g24, n_s))
    say(f"   24-mode Gaussian state (covariance sampler, no 2^24 object), {n_s} snapshots: max |ΔD1| = {np.abs(D24 - g24.one_rdm()).max():.3f}  ({time.time()-t0:.1f} s)")
    out["matchgate"] = {"err_1rdm": e1, "err_2rdm": e2, "err_1rdm_24_modes": float(np.abs(D24 - g24.one_rdm()).max())}

    say("\n7. Shadow QPT of a depolarising channel p = 0.2 (1 qubit, 1500 snapshots per input)")
    from .process_tomography import ShadowProcessTomographer
    tomo = ShadowProcessTomographer(1, seed=1)
    r = tomo.run(lambda x: 0.8 * x + 0.2 * np.trace(x) * np.eye(2) / 2, n_snapshots_per_input=1500)
    say(f"   PTM diagonal {np.round(np.diag(r['ptm']), 3)} (1, 0.8, 0.8, 0.8)  F_avg vs identity {tomo.average_gate_fidelity(r['ptm'], np.eye(2)):.3f} (exact {(2*0.85+1)/3:.3f})")
    CZ = np.diag([1, 1, 1, -1]).astype(complex)
    tomo2 = ShadowProcessTomographer(2, seed=0)
    r2 = tomo2.run(lambda x: 0.9 * CZ @ x @ CZ + 0.1 * np.trace(x) * np.eye(4) / 4, n_snapshots_per_input=300)
    say(f"   2-qubit noisy CZ, 300 snapshots/input: linear inversion min eig(J) = {np.linalg.eigvalsh(r2['choi_raw']).min():+.3f} (unphysical), "
        f"CPTP projection min eig = {max(np.linalg.eigvalsh(r2['choi']).min(), 0.0):.3f}")
    out["qpt"] = {"ptm_diag": np.diag(r["ptm"]).tolist(), "raw_min_eig": float(np.linalg.eigvalsh(r2["choi_raw"]).min())}
    say("=" * 72)
    if as_json:
        print(json.dumps(out, indent=2, default=float))
    return 0
