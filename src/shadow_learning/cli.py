"""
Command-line interface: ``shadow-learn <command>``.

learn            sample Gibbs-state shadows of a random Heisenberg chain and invert (J, h)
derand-compare   randomised vs derandomised estimation error at equal shot budget
qpt              shadow process tomography of a test channel
track            EKF tracking of a drifting coupling from a snapshot stream
rydberg          learn (Ω, Δ, C6) of a Rydberg array from simulated Z-basis quench snapshots
benchmark        the full benchmark suite
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

import numpy as np

from .cirq_shadows import gibbs_state, sample_shadows_from_density_matrix
from .differentiable_inversion import DifferentiableHamiltonianLearner
from .derandomized import DerandomizedShadowSelector
from .reconstruction import estimate_many_observables
from .spam_mitigation import ReadoutErrorModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shadow-learn", description="Classical shadows, Hamiltonian learning, derandomisation, SPAM mitigation, QPT, EKF tracking, Rydberg quench learning")
    sub = parser.add_subparsers(dest="command")

    l = sub.add_parser("learn", help="Gibbs-state shadows → (J, h) inversion")
    l.add_argument("--qubits", type=int, default=2)
    l.add_argument("--shots", type=int, default=4000)
    l.add_argument("--beta", type=float, default=0.8)
    l.add_argument("--derandomized", action="store_true")
    l.add_argument("--spam", type=float, nargs=2, metavar=("P01", "P10"), help="inject and mitigate readout errors")
    l.add_argument("--seed", type=int, default=0)
    l.add_argument("--json", action="store_true")

    d = sub.add_parser("derand-compare", help="Random vs derandomised RMSE")
    d.add_argument("--qubits", type=int, default=3)
    d.add_argument("--shots", type=int, default=300)
    d.add_argument("--trials", type=int, default=10)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--json", action="store_true")

    q = sub.add_parser("qpt", help="Shadow process tomography")
    q.add_argument("--channel", choices=["identity", "x", "hadamard", "depolarizing", "cnot"], default="depolarizing")
    q.add_argument("--p", type=float, default=0.2, help="depolarising strength")
    q.add_argument("--shots", type=int, default=1500, help="snapshots per input state")
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--json", action="store_true")

    t = sub.add_parser("track", help="EKF over a drifting coupling")
    t.add_argument("--shots", type=int, default=3000)
    t.add_argument("--beta", type=float, default=1.0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--json", action="store_true")

    r = sub.add_parser("rydberg", help="Rydberg (Ω, Δ, C6) from Z-basis quench snapshots")
    r.add_argument("--atoms", type=int, default=6)
    r.add_argument("--geometry", choices=["chain", "ring"], default="chain")
    r.add_argument("--spacing", type=float, default=8.0, help="nearest-neighbour distance in μm")
    r.add_argument("--omega", type=float, default=12.0, help="Rabi frequency in rad/μs")
    r.add_argument("--delta", type=float, default=5.0, help="detuning in rad/μs")
    r.add_argument("--times", type=int, default=8, help="number of probe times, evenly spaced up to --t-max")
    r.add_argument("--t-max", type=float, default=4.0, help="latest probe time in μs (Aquila on Braket: 4)")
    r.add_argument("--shots", type=int, default=200, help="shots per probe time")
    r.add_argument("--jitter", type=float, default=0.0, help="per-shot atom position jitter σ in μm per in-plane coordinate (not modelled by the fit)")
    r.add_argument("--ramp", type=float, default=0.05, help="Ω ramp at each end of the pulse in μs (0: instantaneous quench, not programmable)")
    r.add_argument("--no-readout", action="store_true", help="switch off Aquila's readout errors")
    r.add_argument("--device", choices=["datasheet", "braket"], default="datasheet",
                   help="Aquila figures: the 2023 datasheet (default) or the current Braket device properties")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--json", action="store_true")
    r.add_argument("--quick", action="store_true", help="4 atoms, 4 probe times, 100 shots each")

    b = sub.add_parser("benchmark", help="Run the benchmark suite")
    b.add_argument("--json", action="store_true")
    b.add_argument("--quick", action="store_true")
    return parser


def _emit(payload: Dict[str, Any], as_json: bool, lines: List[str]) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    else:
        print("\n".join(lines))


def random_chain(n: int, rng: np.random.Generator):
    J = np.zeros((n, n))
    for i in range(n - 1):
        J[i, i + 1] = J[i + 1, i] = rng.uniform(0.3, 0.9) * rng.choice([-1, 1])
    h = rng.uniform(-0.5, 0.5, size=n)
    return J, h


def cmd_learn(args) -> int:
    rng = np.random.default_rng(args.seed)
    J, h = random_chain(args.qubits, rng)
    learner = DifferentiableHamiltonianLearner(n_qubits=args.qubits, beta=args.beta, seed=args.seed)
    rho = gibbs_state(learner.build_hamiltonian_matrix(J, h), args.beta)
    readout = ReadoutErrorModel(*args.spam) if args.spam else None
    bases = None
    if args.derandomized:
        targets = [learner.pauli_string(n) for n in learner.observable_names]
        bases = DerandomizedShadowSelector(targets, args.qubits).select_measurement_bases(args.shots)
    snaps = sample_shadows_from_density_matrix(rho, args.shots, rng, readout_error=readout, bases=bases)
    res = learner.learn_from_shadows(snaps, true_j_matrix=J, true_h_vector=h, derandomized=args.derandomized, readout_model=readout)
    ident = learner.identifiability(J, h)
    payload = {"n_qubits": args.qubits, "shots": args.shots, "beta": args.beta, "derandomized": args.derandomized, "spam": args.spam,
               "true_J": J, "true_h": h, "recovered_J": res.recovered_j_matrix, "recovered_h": res.recovered_h_vector,
               "frobenius_error_J": res.frobenius_error, "h_error": res.h_error, "final_loss": res.final_loss,
               "iterations": res.iterations, "method": res.method, "condition_number": ident["condition_number"]}
    lines = [f"[*] {args.qubits}-qubit Heisenberg chain, β = {args.beta}, {args.shots} {'derandomised' if args.derandomized else 'random'} shadows{' + SPAM mitigation' if readout else ''}"]
    for i in range(args.qubits - 1):
        lines.append(f"    J_{i}{i+1}: true {J[i, i+1]:+.3f}  recovered {res.recovered_j_matrix[i, i+1]:+.3f}")
    for i in range(args.qubits):
        lines.append(f"    h_{i}:  true {h[i]:+.3f}  recovered {res.recovered_h_vector[i]:+.3f}")
    lines.append(f"[+] ‖ΔJ‖_F = {res.frobenius_error:.3f}  ‖Δh‖ = {res.h_error:.3f}  loss = {res.final_loss:.2e}  ({res.method}, {res.iterations} it)  condition number {ident['condition_number']:.1f}")
    _emit(payload, args.json, lines)
    return 0


def cmd_derand(args) -> int:
    rng = np.random.default_rng(args.seed)
    J, h = random_chain(args.qubits, rng)
    learner = DifferentiableHamiltonianLearner(n_qubits=args.qubits, beta=1.0)
    rho = gibbs_state(learner.build_hamiltonian_matrix(J, h), 1.0)
    targets = [learner.pauli_string(n) for n in learner.observable_names if not n.startswith("Z_")]
    truth = {P: float(np.real(np.trace(learner.pauli_matrix(P) @ rho))) for P in targets}
    bases = DerandomizedShadowSelector(targets, args.qubits).select_measurement_bases(args.shots)
    rr, rd = [], []
    for _ in range(args.trials):
        e_r = estimate_many_observables(sample_shadows_from_density_matrix(rho, args.shots, rng), targets)
        e_d = estimate_many_observables(sample_shadows_from_density_matrix(rho, args.shots, rng, bases=bases), targets, derandomized=True)
        rr.append(np.sqrt(np.mean([(e_r[P] - truth[P]) ** 2 for P in targets])))
        rd.append(np.sqrt(np.mean([(e_d[P] - truth[P]) ** 2 for P in targets])))
    payload = {"targets": targets, "shots": args.shots, "trials": args.trials, "rmse_random": float(np.mean(rr)), "rmse_derandomized": float(np.mean(rd)), "gain": float(np.mean(rr) / np.mean(rd))}
    _emit(payload, args.json, [f"[*] {len(targets)} bond observables on {args.qubits} qubits, {args.shots} shots × {args.trials} trials",
                               f"[+] RMSE random = {np.mean(rr):.4f}   derandomised = {np.mean(rd):.4f}   gain ×{np.mean(rr)/np.mean(rd):.2f}"])
    return 0


def cmd_qpt(args) -> int:
    from .process_tomography import ShadowProcessTomographer, pauli_transfer_matrix_of_unitary
    X = np.array([[0, 1], [1, 0]], dtype=complex); Hd = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
    if args.channel == "depolarizing":
        n, target = 1, np.eye(2)
        chan = lambda r: (1 - args.p) * r + args.p * np.trace(r) * np.eye(2) / 2
    else:
        u = {"identity": np.eye(2), "x": X, "hadamard": Hd, "cnot": cnot}[args.channel]
        n, target = int(round(np.log2(u.shape[0]))), u
        chan = lambda r, u=u: u @ r @ u.conj().T
    tomo = ShadowProcessTomographer(n_qubits=n, seed=args.seed)
    res = tomo.run(chan, n_snapshots_per_input=args.shots)
    fp, fa = tomo.process_fidelity(res["ptm"], target), tomo.average_gate_fidelity(res["ptm"], target)
    payload = {"channel": args.channel, "n_qubits": n, "ptm": res["ptm"], "process_fidelity": fp, "average_gate_fidelity": fa, "choi_trace": float(np.real(np.trace(res["choi"])))}
    lines = [f"[*] shadow QPT of '{args.channel}' on {n} qubit(s), {args.shots} snapshots × {6**n} input states",
             f"[+] F_pro vs target = {fp:.4f}   F_avg = {fa:.4f}   Tr(Choi) = {payload['choi_trace']:.3f}", "    PTM:", np.array2string(res["ptm"], precision=2, suppress_small=True)]
    _emit(payload, args.json, lines)
    return 0


def cmd_track(args) -> int:
    from .kalman_tracker import StreamingKalmanHamiltonianTracker
    rng = np.random.default_rng(args.seed)
    tracker = StreamingKalmanHamiltonianTracker(n_qubits=2, beta=args.beta, process_noise_std=0.005, seed=args.seed)
    h = np.array([0.2, -0.15]); est = []; truth = []
    for k in range(args.shots):
        jv = 0.3 if k < args.shots // 2 else 0.8
        if k % 250 == 0:
            rho = gibbs_state(tracker.learner.build_hamiltonian_matrix(np.array([[0, jv], [jv, 0]]), h), args.beta)
        tracker.process_snapshot(sample_shadows_from_density_matrix(rho, 1, rng)[0])
        est.append(tracker.x[0]); truth.append(jv)
    est = np.array(est); truth = np.array(truth)
    payload = {"shots": args.shots, "final_estimate": float(est[-1]), "final_truth": float(truth[-1]), "rms_error_last_500": float(np.sqrt(np.mean((est[-500:] - truth[-500:]) ** 2)))}
    _emit(payload, args.json, [f"[*] EKF over {args.shots} single snapshots; J steps 0.3 → 0.8 at shot {args.shots//2}",
                               f"[+] final estimate {est[-1]:+.3f} (truth {truth[-1]:+.3f}); RMS error over the last 500 shots {payload['rms_error_last_500']:.3f}"])
    return 0


def _finite_or_none(x):
    """JSON has no infinity: an unreachable shot count is reported as null."""
    return float(x) if np.isfinite(x) else None


def cmd_rydberg(args) -> int:
    from .rydberg import AQUILA, AQUILA_BRAKET, RydbergQuenchModel, blockade_radius, chain, geometry_violations, ring
    n, n_times, shots = (4, 4, 100) if args.quick else (args.atoms, args.times, args.shots)
    specs = AQUILA_BRAKET if args.device == "braket" else AQUILA
    pos = chain(n, args.spacing) if args.geometry == "chain" else ring(n, args.spacing)
    model = RydbergQuenchModel(pos, ramp_time=args.ramp)
    theta = model.theta(args.omega, args.delta)
    readout = None if args.no_readout else specs.readout_model()
    times = np.linspace(args.t_max / n_times, args.t_max, n_times)
    rng = np.random.default_rng(args.seed)
    data = model.sample(theta, times, shots, rng, readout=readout, position_jitter=args.jitter)
    start = theta * (1.0 + 0.08 * rng.uniform(-1, 1, size=3))          # nominal values a few per cent off
    fit = model.fit_mle(data, start, readout=readout)
    crb = model.cramer_rao(theta, times, shots, readout)
    need = model.shots_for_precision(theta, times, 0.01, readout)
    grid = np.arange(max(0.1, 2 * args.ramp), args.t_max + 1e-9, 0.05)
    best = model.optimal_probe_times(theta, 1, grid, 0.01, readout)
    flip = float(np.abs(model.probabilities(theta, times, readout) - model.probabilities(theta * np.array([-1, 1, 1]), times, readout)).max())
    names = ("omega", "delta", "c6")
    payload = {"geometry": args.geometry, "atoms": n, "spacing_um": args.spacing, "probe_times_us": times, "shots_per_time": shots,
               "device_figures": args.device, "readout": None if readout is None else [readout.p01, readout.p10],
               "position_jitter_um": args.jitter, "layout_violations": geometry_violations(pos, specs),
               "blockade_radius_um": blockade_radius(args.omega, args.delta),
               "ramp_us": args.ramp, "waveform_violations": model.waveform_violations(theta, times, specs),
               "true": dict(zip(names, theta.tolist())), "estimate": fit.as_dict(), "std": fit.std,
               "fit_success": fit.success and not args.jitter, "fit_warnings": fit.warnings, "goodness_of_fit_p": fit.gof_pvalue,
               "crb_relative_std": dict(zip(names, crb["relative_std"].tolist())),
               "shots_for_1pct_uniform": _finite_or_none(need["total_shots"]),
               "best_single_time_us": float(best["times"][0]) if np.isfinite(best["total_shots"]) else None,
               "shots_for_1pct_best_single_time": _finite_or_none(best["total_shots"]),
               "max_prob_change_under_omega_sign_flip": flip}
    unit = {"omega": "rad/μs", "delta": "rad/μs", "c6": "rad μm^6/μs"}
    lines = [f"[*] {n}-atom {args.geometry}, spacing {args.spacing} μm (blockade radius {payload['blockade_radius_um']:.2f} μm), "
             f"{n_times} pulses of up to {args.t_max} μs with {args.ramp} μs Ω ramps, {shots} shots each"
             + ("" if readout is None else f", readout errors {readout.p01:.2f}/{readout.p10:.2f} ({args.device} figures)")
             + (f", position jitter {args.jitter} μm" if args.jitter else "")]
    if payload["layout_violations"]:
        lines.append("    layout outside Aquila's rules: " + "; ".join(payload["layout_violations"]))
    if payload["waveform_violations"]:
        lines.append("    pulse outside Aquila's rules: " + "; ".join(payload["waveform_violations"]))
    for k, nm in enumerate(names):
        rel = fit.std[nm] / abs(theta[k])
        lines.append(f"    {nm:5s}: true {theta[k]:.6g}  estimate {fit.theta[k]:.6g} ± {fit.std[nm]:.3g} {unit[nm]}  ({100 * rel:.2f} %, CRB {100 * crb['relative_std'][k]:.2f} %)")
    lines.append(f"    goodness of fit p = {fit.gof_pvalue:.2f}")
    for w in fit.warnings:
        lines.append(f"[!] {w}")
    if args.jitter:
        lines.append("[!] the data have position jitter and the fit does not model it: the estimates can be biased by "
                     "several per cent, depending on the geometry and the pulse, and the goodness-of-fit test need not notice")

    def shots_text(total):
        return f"{total:.0f}" if np.isfinite(total) else "not reached (the parameters are not all identifiable)"
    lines.append(f"[+] shots for 1 % on all three with these probe times: {shots_text(need['total_shots'])}")
    if np.isfinite(best["total_shots"]):
        lines.append(f"    with all shots at one probe time: {best['total_shots']:.0f}, at t = {best['times'][0]:.2f} μs")
        lines.append("    (with a single probe time the likelihood has many local maxima, and the optimum is sharp: "
                     "keep a few early times to anchor the fit)")
    else:
        lines.append("    with all shots at one probe time: not identifiable from one probe time")
    lines.append(f"    flipping the sign of Ω changes no outcome probability by more than {flip:.1e}")
    _emit(payload, args.json, lines)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    if args.command == "benchmark":
        from .benchmark import run_shadow_benchmark
        return run_shadow_benchmark(as_json=args.json, quick=args.quick)
    return {"learn": cmd_learn, "derand-compare": cmd_derand, "qpt": cmd_qpt, "track": cmd_track,
            "rydberg": cmd_rydberg}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
