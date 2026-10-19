"""
Dry run of the whole pipeline on Braket's local AHS simulator: the programs written
by build.py are run with LocalSimulator("braket_ahs") at the same shot count,
saved in the same result format that analyze.py reads for Aquila, and fitted.

The local simulator integrates the Schroedinger equation for exactly the programmed
Hamiltonian, with C6 = 5.42e-24 rad m^6/s (5.42e6 rad um^6/us) and no noise: no
readout errors, no empty sites, no position errors, no drive noise or decoherence.
The fit therefore uses no readout or noise model, and should recover Omega = 12,
Delta = 5 and the simulator's C6 within the predicted statistical errors.

With ``--repeats N`` the experiment is also repeated N more times (fresh shots,
no bootstrap) to compare the spread of the estimates with the Cramer-Rao bound.

Needs amazon-braket-sdk (the .venv-braket environment):

    .venv-braket/bin/python experiments/aquila/dry_run.py --repeats 20
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze  # noqa: E402
import build    # noqa: E402

DRYRUN_DIR = HERE / "dryrun"


def simulator_c6() -> float:
    """The local simulator's C6 in rad um^6/us."""
    from braket.analog_hamiltonian_simulator.rydberg.constants import RYDBERG_INTERACTION_COEF
    return float(RYDBERG_INTERACTION_COEF) * 1e36 / 1e6


def run_local(programs_dir: Path, shots: Optional[int] = None, steps: int = 1000) -> List[Dict[str, object]]:
    """Run every program in the manifest on the local simulator; returns result records."""
    from braket.devices import LocalSimulator
    with open(Path(programs_dir) / "manifest.json") as fh:
        manifest = json.load(fh)
    device = LocalSimulator("braket_ahs")
    records = []
    for entry in manifest["programs"]:
        with open(Path(programs_dir) / entry["file"]) as fh:
            ir = json.load(fh)
        n = int(shots if shots is not None else entry["shots"])
        result = device.run(build.sdk_program(ir), shots=n, steps=steps).result()
        records.append(analyze.result_record(result, entry["probe_time_us"], ir, entry["file"], shots_requested=n))
    return records


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run the experiment on the local AHS simulator and fit it.")
    ap.add_argument("--programs", type=Path, default=build.PROGRAMS_DIR)
    ap.add_argument("--out", type=Path, default=DRYRUN_DIR)
    ap.add_argument("--shots", type=int, default=None, help="override the manifest's shots per program")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=0, help="extra independent runs to check the error bars")
    ap.add_argument("--seed", type=int, default=7, help="seed for numpy's global generator, which the simulator samples with")
    args = ap.parse_args(argv)

    np.random.seed(args.seed)                    # the local simulator draws shots with np.random.multinomial
    truth = {"omega": build.OMEGA, "delta": build.DELTA, "c6": simulator_c6()}
    t0 = time.time()
    records = run_local(args.programs, args.shots)
    t_sim = time.time() - t0
    analyze.save_records(records, args.out / "results")
    labels = {"source": "Braket LocalSimulator('braket_ahs'), amazon-braket-sdk "
              + importlib.metadata.version("amazon-braket-sdk"),
              "truth": f"Omega = {truth['omega']} rad/us, Delta = {truth['delta']} rad/us, "
                       f"C6 = {truth['c6']:.6g} rad um^6/us (simulator constant)"}
    t0 = time.time()
    report, text = analyze.run_analysis(records, readout_name="none", noise_name="none", n_boot=args.bootstrap,
                                        truth=truth, labels=labels)
    t_fit = time.time() - t0
    pred = report["predicted_relative_std_at_estimate"]
    text += ("\n\npredicted relative errors (Cramer-Rao, no readout errors, kept shots): "
             + ", ".join(f"{nm} {100 * v:.2f} %" for nm, v in pred.items()))
    text += f"\nsimulation {t_sim:.1f} s, analysis {t_fit:.1f} s"

    if args.repeats > 0:
        est, cr = [], []
        for _ in range(args.repeats):
            recs = run_local(args.programs, args.shots)
            model, theta0, data, _ = analyze.records_to_data(recs)
            fit, _ = analyze.fit_with_restarts(model, data, theta0, restarts=False)
            est.append(fit.theta[:3])
            cr.append([fit.std[nm] for nm in build.FIT_PARAMS])
        est, cr = np.array(est), np.array(cr)
        tv = np.array([truth[nm] for nm in build.FIT_PARAMS])
        z = (est - tv) / cr
        rep = {"n": int(args.repeats), "mean_relative_error": list(map(float, (est.mean(0) / tv - 1))),
               "rms_relative_error": list(map(float, np.sqrt(np.mean((est / tv - 1) ** 2, axis=0)))),
               "mean_relative_cramer_rao": list(map(float, np.mean(cr / est, axis=0))),
               "mean_z2": list(map(float, np.mean(z ** 2, axis=0))),
               "fraction_within_2sigma": list(map(float, np.mean(np.abs(z) < 2, axis=0)))}
        report["repeats"] = rep
        text += f"\n\n{args.repeats} further independent runs (no bootstrap):\n"
        text += f"{'':6s} {'mean error':>11s} {'RMS error':>10s} {'mean CR':>8s} {'mean z^2':>9s} {'|z| < 2':>8s}\n"
        for k, nm in enumerate(("Omega", "Delta", "C6")):
            text += (f"{nm:6s} {100 * rep['mean_relative_error'][k]:+10.3f}% {100 * rep['rms_relative_error'][k]:9.3f}% "
                     f"{100 * rep['mean_relative_cramer_rao'][k]:7.3f}% {rep['mean_z2'][k]:9.2f} "
                     f"{100 * rep['fraction_within_2sigma'][k]:7.0f}%\n")
        text = text.rstrip("\n")

    print(text)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "summary.json", "w") as fh:
        json.dump(report, fh, indent=1, default=analyze._json_default)
    with open(args.out / "summary.txt", "w") as fh:
        fh.write(text + "\n")
    ok = all(abs(report["parameters"][nm]["z"]) < 3 for nm in build.FIT_PARAMS)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
