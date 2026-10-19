"""
Fit Omega, Delta and C6 to the Aquila (or local simulator) results of the reference experiment.

Results come either from the task ARNs that submit.py saved (fetched with
AwsQuantumTask, which needs amazon-braket-sdk and AWS credentials, and cached as
JSON), or from a directory of result JSON files written earlier by this script or by
dry_run.py (no SDK needed).

For each task the shots are converted with QuenchData.from_braket, using the
program's requested filling and each shot's status: failed shots are dropped, and so
are shots whose pre-sequence image differs from the requested filling (a missing atom
reads as Rydberg). The fit is fit_mle with its continuation and search, started from
the programmed values and from points displaced by the device's documented Omega and
Delta errors. Errors are the Cramer-Rao bound at the estimate and a nonparametric
bootstrap over shots. The goodness-of-fit test and any warnings from the fit are
printed, and C6 is compared with the package's value, 2 pi x 862690 rad um^6/us.

    python experiments/aquila/analyze.py --tasks experiments/aquila/tasks.json
    python experiments/aquila/analyze.py --results experiments/aquila/results
"""

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build  # noqa: E402  (also puts the package on the path when it is not installed)

from shadow_learning.rydberg import (  # noqa: E402
    AQUILA,
    AQUILA_BRAKET,
    C6_RB87,
    DeviceNoise,
    QuenchData,
    RydbergFit,
    RydbergQuenchModel,
)
from shadow_learning.spam_mitigation import ReadoutErrorModel  # noqa: E402

FIT_PARAMS = build.FIT_PARAMS
RESULTS_DIR = HERE / "results"
TASKS_FILE = HERE / "tasks.json"


# --------------------------------------------------------------------------- #
# result records: one JSON file per task
# --------------------------------------------------------------------------- #
def _status_str(status) -> str:
    return str(getattr(status, "value", status))


def _seq(x) -> Optional[List[int]]:
    return None if x is None else [int(v) for v in np.asarray(x).ravel()]


def result_record(result, probe_time_us: float, program: Dict[str, object], program_file: str = "",
                  task_id: Optional[str] = None, shots_requested: Optional[int] = None) -> Dict[str, object]:
    """
    A plain record of one task's result. ``result`` is an SDK
    AnalogHamiltonianSimulationQuantumTaskResult or anything with ``measurements``
    whose items have ``status``, ``pre_sequence`` and ``post_sequence``.
    """
    meta = getattr(result, "task_metadata", None)
    return {
        "task_id": task_id or (getattr(meta, "id", None) if meta is not None else None),
        "device": getattr(meta, "deviceId", None) if meta is not None else None,
        "ended_at": getattr(meta, "endedAt", None) if meta is not None else None,
        "probe_time_us": float(probe_time_us),
        "program_file": program_file,
        "program": program,
        "shots_requested": shots_requested if shots_requested is not None else len(result.measurements),
        "measurements": [{"status": _status_str(m.status), "pre_sequence": _seq(m.pre_sequence),
                          "post_sequence": _seq(m.post_sequence)} for m in result.measurements],
    }


def record_filename(record: Dict[str, object], k: int) -> str:
    return f"result_{k:02d}_T{record['probe_time_us']:.3f}us.json"


def save_records(records: Sequence[Dict[str, object]], outdir: Path) -> List[Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for k, rec in enumerate(records):
        p = outdir / record_filename(rec, k)
        with open(p, "w") as fh:
            json.dump(rec, fh, separators=(",", ":"))
            fh.write("\n")
        paths.append(p)
    return paths


def load_records(indir: Path) -> List[Dict[str, object]]:
    paths = sorted(Path(indir).glob("result_*.json"))
    if not paths:
        raise FileNotFoundError(f"no result_*.json files in {indir}")
    out = []
    for p in paths:
        with open(p) as fh:
            out.append(json.load(fh))
    return out


def load_tasks_file(path: Path = TASKS_FILE) -> Dict[str, object]:
    with open(path) as fh:
        return json.load(fh)


def fetch_task_results(tasks_file: Path = TASKS_FILE, outdir: Path = RESULTS_DIR,
                       wait: bool = False) -> List[Dict[str, object]]:
    """
    Download the results of the tasks in ``tasks_file`` with AwsQuantumTask (needs
    amazon-braket-sdk and credentials) and save them as result records. Tasks that
    have not completed are reported; with ``wait`` the call blocks until they do.
    """
    from braket.aws import AwsQuantumTask
    tasks = load_tasks_file(tasks_file)
    progdir = Path(tasks_file).parent / tasks.get("programs_dir", "programs")
    records, pending = [], []
    for entry in tasks["tasks"]:
        task = AwsQuantumTask(entry["task_arn"])
        state = task.state()
        if state != "COMPLETED" and not wait:
            pending.append((entry["task_arn"], state))
            continue
        result = task.result()
        if result is None:
            pending.append((entry["task_arn"], task.state()))
            continue
        with open(progdir / entry["program_file"]) as fh:
            program = json.load(fh)
        records.append(result_record(result, entry["probe_time_us"], program, entry["program_file"],
                                     task_id=entry["task_arn"], shots_requested=entry["shots"]))
    if pending:
        msg = "\n".join(f"  {arn}: {state}" for arn, state in pending)
        raise RuntimeError(f"{len(pending)} task(s) not completed yet; run again later or pass --wait:\n{msg}")
    save_records(records, outdir)
    return records


# --------------------------------------------------------------------------- #
# records to data
# --------------------------------------------------------------------------- #
def records_to_data(records: Sequence[Dict[str, object]], postselect: bool = True
                    ) -> Tuple[RydbergQuenchModel, np.ndarray, QuenchData, Dict[str, object]]:
    """
    The model, programmed parameters (with the package's C6), data and per-time
    bookkeeping for a set of result records. All records must share the register,
    Omega, Delta and ramp; the probe time is each program's duration.
    """
    summaries = [build.program_summary(r["program"]) for r in records]
    ref = summaries[0]
    for s, r in zip(summaries, records):
        for key in ("positions_um", "filling", "omega", "delta", "ramp_us"):
            if not np.allclose(np.asarray(s[key], dtype=float), np.asarray(ref[key], dtype=float)):
                raise ValueError(f"records differ in {key}; analyse one experiment at a time")
        if abs(s["duration_us"] - float(r["probe_time_us"])) > 1e-9:
            raise ValueError(f"record says T = {r['probe_time_us']} us but its program lasts {s['duration_us']} us")
    order = np.argsort([s["duration_us"] for s in summaries], kind="stable")
    recs = [records[k] for k in order]
    times = [summaries[k]["duration_us"] for k in order]
    filling = np.asarray(ref["filling"], dtype=int)
    pres = [[m["pre_sequence"] for m in r["measurements"]] for r in recs]
    posts = [[m["post_sequence"] for m in r["measurements"]] for r in recs]
    stats = [[m["status"] for m in r["measurements"]] for r in recs]
    data = QuenchData.from_braket(times, pres, posts, postselect=postselect, filling=filling, statuses=stats)
    pos = np.asarray(ref["positions_um"])[filling == 1]
    model = RydbergQuenchModel(pos, ramp_time=ref["ramp_us"])
    theta0 = model.theta(ref["omega"], ref["delta"], C6_RB87)
    info = {
        "probe_times_us": [float(t) for t in times],
        "shots_returned": [len(r["measurements"]) for r in recs],
        "shots_success": [sum(1 for s in st if str(s).lower() == "success") for st in stats],
        "shots_kept": [int(n) for n in data.shots],
        "task_ids": [r.get("task_id") for r in recs],
        "programmed": {"omega": ref["omega"], "delta": ref["delta"]},
    }
    return model, theta0, data, info


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #
def readout_model(name: str, paradigm: Optional[Dict[str, object]] = None) -> Optional[ReadoutErrorModel]:
    """
    'none'; 'paper' (0.01, 0.08); 'braket' (0.05, 0.10, the documented Braket
    figures); 'device' (the ground and Rydberg detection errors in ``paradigm``,
    the properties submit.py saved at submission); 'auto' is 'device' when
    ``paradigm`` is given, else 'braket'.
    """
    if name == "auto":
        name = "device" if paradigm else "braket"
    if name == "none":
        return None
    if name == "paper":
        return AQUILA.readout_model()
    if name == "braket":
        return AQUILA_BRAKET.readout_model()
    if name == "device":
        g = paradigm["performance"]["rydberg"]["rydbergGlobal"]
        return ReadoutErrorModel(p01=float(Decimal(str(g["groundDetectionError"]))),
                                 p10=float(Decimal(str(g["rydbergDetectionError"]))))
    raise ValueError(f"unknown readout model {name!r}")


def noise_model(name: str) -> Optional[DeviceNoise]:
    """'none'; 'drive' (shot-to-shot Omega and Delta noise); 'braket' (drive noise and relaxation at 7 us)."""
    return {"none": None, "drive": DeviceNoise.braket(t_relax=None), "braket": DeviceNoise.braket()}[name]


def restart_points(theta0: np.ndarray) -> List[np.ndarray]:
    """
    Starting points: the programmed values, and Omega and Delta displaced by the
    documented device errors (3 % relative in Omega, 1 rad/us in Delta), so that a
    miscalibrated drive does not leave the fit in the basin of a wrong maximum.
    """
    pts = [np.asarray(theta0, dtype=float).copy()]
    for so, sd in ((+1, +1), (-1, -1)):
        th = pts[0].copy()
        th[0] *= 1.0 + so * 0.03
        th[1] += sd * 1.0
        pts.append(th)
    return pts


def fit_with_restarts(model: RydbergQuenchModel, data: QuenchData, theta0: np.ndarray, readout=None, noise=None,
                      n_starts: int = 3, restarts: bool = True) -> Tuple[RydbergFit, List[Dict[str, object]]]:
    """fit_mle from each restart point; returns the best fit and a summary of every run."""
    runs, best = [], None
    for th_s in (restart_points(theta0) if restarts else [np.asarray(theta0, dtype=float)]):
        f = model.fit_mle(data, th_s, fit=FIT_PARAMS, readout=readout, noise=noise, n_starts=n_starts)
        runs.append({"start": [float(x) for x in th_s[:3]], "theta": [float(x) for x in f.theta[:3]],
                     "neg_log_likelihood": float(f.objective), "success": bool(f.success)})
        if best is None or f.objective < best.objective - 1e-9:
            best = f
    return best, runs


def bootstrap(model: RydbergQuenchModel, data: QuenchData, fit: RydbergFit, n_boot: int, seed: int = 2026,
              readout=None, noise=None) -> np.ndarray:
    """
    Nonparametric bootstrap: resample the kept shots with replacement at each probe
    time, refit locally from the estimate (no continuation or search), and return
    the (n_boot, 3) estimates of Omega, Delta and C6.
    """
    rng = np.random.default_rng(seed)
    out = np.empty((int(n_boot), len(FIT_PARAMS)))
    for b in range(int(n_boot)):
        bits = [x[rng.integers(0, len(x), len(x))] for x in data.bits]
        fb = model.fit_mle(QuenchData(times=data.times, bits=bits), fit.theta, fit=FIT_PARAMS, readout=readout,
                           noise=noise, search=False, continuation=False)
        out[b] = fb.theta[:3]
    return out


def analyse(model: RydbergQuenchModel, theta0: np.ndarray, data: QuenchData, readout=None, noise=None,
            n_boot: int = 200, n_starts: int = 3, restarts: bool = True, seed: int = 2026,
            truth: Optional[Dict[str, float]] = None, c6_reference: float = C6_RB87,
            labels: Optional[Dict[str, str]] = None) -> Dict[str, object]:
    """Fit, bootstrap and compare; returns a JSON-ready report."""
    fit, runs = fit_with_restarts(model, data, theta0, readout, noise, n_starts, restarts)
    est = {nm: float(fit.theta[k]) for k, nm in enumerate(FIT_PARAMS)}
    cr = {nm: float(v) for nm, v in fit.std.items()}
    boot = bootstrap(model, data, fit, n_boot, seed, readout, noise) if n_boot > 0 else None
    bs = {nm: float(np.std(boot[:, k], ddof=1)) for k, nm in enumerate(FIT_PARAMS)} if boot is not None else None
    params = {}
    for k, nm in enumerate(FIT_PARAMS):
        p = {"estimate": est[nm], "cramer_rao_std": cr[nm], "relative_cramer_rao_std": cr[nm] / abs(est[nm])}
        if bs is not None:
            p["bootstrap_std"] = bs[nm]
            p["bootstrap_interval_68"] = [float(np.percentile(boot[:, k], 16)), float(np.percentile(boot[:, k], 84))]
        if truth is not None and nm in truth:
            err = max(cr[nm], bs[nm]) if bs is not None else cr[nm]
            p["truth"] = float(truth[nm])
            p["relative_error"] = est[nm] / truth[nm] - 1.0
            p["z"] = (est[nm] - truth[nm]) / err
        params[nm] = p
    err_c6 = max(cr["c6"], bs["c6"]) if bs is not None else cr["c6"]
    report = {
        "labels": labels or {},
        "parameters": params,
        "c6_comparison": {"reference": c6_reference, "reference_note": "2 pi x 862690 rad um^6/us (shadow_learning.C6_RB87)",
                          "ratio_minus_one": est["c6"] / c6_reference - 1.0, "z_statistical": (est["c6"] - c6_reference) / err_c6},
        "goodness_of_fit": {"chi2": float(fit.gof_chi2), "dof": int(fit.gof_dof), "p_value": float(fit.gof_pvalue)},
        "fit_success": bool(fit.success), "warnings": list(fit.warnings),
        "alternatives": [{"theta": [float(x) for x in th[:3]], "delta_log_likelihood": float(df)} for th, df in fit.alternatives],
        "restarts": runs, "n_bootstrap": int(n_boot),
        "readout": None if readout is None else {"p_ground_as_rydberg": float(readout.p01), "p_rydberg_as_ground": float(readout.p10)},
        "noise": None if noise is None else {"sigma_delta": noise.sigma_delta, "sigma_omega_rel": noise.sigma_omega_rel,
                                             "t_relax": noise.t_relax},
    }
    return report


def predicted_errors(model: RydbergQuenchModel, theta: np.ndarray, data: QuenchData, readout=None, noise=None) -> Dict[str, float]:
    """Cramer-Rao relative standard deviations at ``theta`` for the shots actually kept."""
    r = model.cramer_rao(theta, data.times, data.shots, readout, noise, wrt=[0, 1, 2])
    return dict(zip(FIT_PARAMS, map(float, r["relative_std"])))


def format_report(report: Dict[str, object], info: Optional[Dict[str, object]] = None) -> str:
    units = {"omega": "rad/us", "delta": "rad/us", "c6": "rad um^6/us"}
    names = {"omega": "Omega", "delta": "Delta", "c6": "C6"}
    lines = []
    for k, v in (report.get("labels") or {}).items():
        lines.append(f"{k}: {v}")
    if info:
        lines.append("probe time (us)  returned  success  kept")
        for t, a, b, c in zip(info["probe_times_us"], info["shots_returned"], info["shots_success"], info["shots_kept"]):
            lines.append(f"  {t:5.2f}          {a:5d}    {b:5d}  {c:5d}")
    ro, nz = report["readout"], report["noise"]
    lines.append("readout errors in the fit: " + ("none" if ro is None else
                 f"ground read as Rydberg {ro['p_ground_as_rydberg']:g}, Rydberg read as ground {ro['p_rydberg_as_ground']:g}"))
    lines.append("drive noise and relaxation in the fit: " + ("none" if nz is None else
                 f"sigma_Delta {nz['sigma_delta']:g} rad/us, sigma_Omega {100 * nz['sigma_omega_rel']:g} %, "
                 f"t_relax {nz['t_relax'] if nz['t_relax'] else 'off'}"))
    lines.append("")
    hdr = f"{'':6s} {'estimate':>14s} {'Cramer-Rao':>12s} {'bootstrap':>12s} {'rel. error':>10s}"
    if any("truth" in p for p in report["parameters"].values()):
        hdr += f" {'truth':>14s} {'est/truth-1':>11s} {'z':>6s}"
    lines.append(hdr)
    for nm, p in report["parameters"].items():
        fmt = "{:14.6g}" if nm == "c6" else "{:14.5f}"
        row = f"{names[nm]:6s} " + fmt.format(p["estimate"]) + f" {p['cramer_rao_std']:12.4g}"
        row += f" {p['bootstrap_std']:12.4g}" if "bootstrap_std" in p else f" {'-':>12s}"
        row += f" {100 * p['relative_cramer_rao_std']:9.2f}%"
        if "truth" in p:
            row += " " + fmt.format(p["truth"]) + f" {100 * p['relative_error']:+10.2f}% {p['z']:+6.2f}"
        lines.append(row + f"   {units[nm]}")
    c = report["c6_comparison"]
    lines.append("")
    lines.append(f"C6 against {c['reference']:.6g} rad um^6/us ({c['reference_note']}): "
                 f"{100 * c['ratio_minus_one']:+.2f} %, {c['z_statistical']:+.2f} statistical standard deviations")
    g = report["goodness_of_fit"]
    lines.append(f"goodness of fit (readout-corrected <n_i>, <n_i n_j>): chi2 = {g['chi2']:.1f} for {g['dof']} dof, p = {g['p_value']:.3g}")
    sd = np.array([report["parameters"][nm]["cramer_rao_std"] for nm in FIT_PARAMS])
    best = np.array([report["parameters"][nm]["estimate"] for nm in FIT_PARAMS])
    n_off = sum(1 for r in report["restarts"] if np.max(np.abs(np.array(r["theta"]) - best) / sd) > 0.1)
    lines.append(f"restarts: {len(report['restarts'])} starting points, "
                 + ("all reached the same estimate (within 0.1 standard deviation)" if n_off == 0 else
                    f"{n_off} ended elsewhere; the best is reported"))
    lines.append(f"bootstrap: {report['n_bootstrap']} resamples of the kept shots, local refits from the estimate")
    if report["warnings"]:
        lines.append("fit warnings:")
        lines += [f"  {w}" for w in report["warnings"]]
    else:
        lines.append("fit warnings: none")
    return "\n".join(lines)


def run_analysis(records: Sequence[Dict[str, object]], readout_name: str = "auto", noise_name: str = "none",
                 n_boot: int = 200, paradigm: Optional[Dict[str, object]] = None, compare_noise: bool = False,
                 truth: Optional[Dict[str, float]] = None, labels: Optional[Dict[str, str]] = None,
                 n_starts: int = 3) -> Tuple[Dict[str, object], str]:
    model, theta0, data, info = records_to_data(records)
    ro, nz = readout_model(readout_name, paradigm), noise_model(noise_name)
    report = analyse(model, theta0, data, ro, nz, n_boot=n_boot, truth=truth, labels=labels, n_starts=n_starts)
    report["data"] = info
    report["predicted_relative_std_at_estimate"] = predicted_errors(
        model, np.array([report["parameters"][nm]["estimate"] for nm in FIT_PARAMS]), data, ro, nz)
    text = format_report(report, info)
    if compare_noise:
        report["noise_comparison"] = {}
        lines = ["", "The same data under other noise models (no bootstrap):"]
        for nm in ("none", "drive", "braket"):
            if nm == noise_name:
                continue
            r2 = analyse(model, theta0, data, ro, noise_model(nm), n_boot=0, restarts=False, n_starts=n_starts)
            report["noise_comparison"][nm] = r2
            p = r2["parameters"]
            lines.append(f"  noise '{nm}': Omega {p['omega']['estimate']:.4f}, Delta {p['delta']['estimate']:.4f}, "
                         f"C6 {p['c6']['estimate']:.6g} ({100 * r2['c6_comparison']['ratio_minus_one']:+.2f} % from reference), "
                         f"goodness of fit p = {r2['goodness_of_fit']['p_value']:.3g}")
        text += "\n" + "\n".join(lines)
    return report, text


def _json_default(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Fit Omega, Delta and C6 to the experiment's results.")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--tasks", type=Path, help="tasks file written by submit.py; results are fetched and cached")
    src.add_argument("--results", type=Path, help="directory of result_*.json files")
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="where fetched results are cached")
    ap.add_argument("--wait", action="store_true", help="block until every task has completed")
    ap.add_argument("--readout", default="auto", choices=["auto", "device", "braket", "paper", "none"])
    ap.add_argument("--noise", default="none", choices=["none", "drive", "braket"])
    ap.add_argument("--compare-noise", action="store_true", help="also fit under the other noise models")
    ap.add_argument("--bootstrap", type=int, default=200, help="bootstrap resamples (0 to skip)")
    ap.add_argument("--out", type=Path, default=None, help="write the report here (.json and .txt)")
    args = ap.parse_args(argv)

    paradigm = None
    if args.tasks is not None:
        tasks = load_tasks_file(args.tasks)
        paradigm = tasks.get("paradigm_at_submission")
        records = fetch_task_results(args.tasks, args.results_dir, wait=args.wait)
    else:
        indir = args.results or args.results_dir
        records = load_records(indir)
        if TASKS_FILE.exists() and args.results is None:
            paradigm = load_tasks_file(TASKS_FILE).get("paradigm_at_submission")
    report, text = run_analysis(records, args.readout, args.noise, args.bootstrap, paradigm, args.compare_noise)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out.with_suffix(".json"), "w") as fh:
            json.dump(report, fh, indent=1, default=_json_default)
        with open(args.out.with_suffix(".txt"), "w") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
