"""
Figures and numbers for the README.

    python benchmarks/make_figures.py            # about 40 minutes
    python benchmarks/make_figures.py --quick    # fewer Monte Carlo trials, for a smoke test

Writes figures/*.png and figures/findings.json, which hold the numbers quoted
in the README's Rydberg section (the example and command-line outputs there come
from running them). Needs matplotlib in addition to the core dependencies. All
data are simulated; seeds are fixed.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

from shadow_learning import DifferentiableHamiltonianLearner  # noqa: E402
from shadow_learning.rydberg import (  # noqa: E402
    AQUILA,
    AQUILA_BRAKET,
    C6_RB87,
    DeviceNoise,
    RydbergDriftTracker,
    RydbergQuenchModel,
    _relative_inverse,
    blockade_radius,
    chain,
    ring,
    simulate_drifting_stream,
)

QUICK = "--quick" in sys.argv
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

# one colour per parameter, used in every figure (validated for colour-vision deficiency)
COL = {"omega": "#2f6db3", "delta": "#c25a30", "c6": "#2a9a73"}
LAB = {"omega": "Ω", "delta": "Δ", "c6": "C₆"}
NAMES = ("omega", "delta", "c6")
INK, GRAY, LIGHT = "#222222", "#777777", "#bbbbbb"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "normal",
    "axes.titlelocation": "left", "axes.labelsize": 9, "axes.edgecolor": "#999999", "axes.linewidth": 0.7,
    "axes.labelcolor": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e8e8e8", "grid.linewidth": 0.6,
    "xtick.color": "#555555", "ytick.color": "#555555", "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.minor.size": 0, "ytick.minor.size": 0,
    "legend.frameon": False, "legend.fontsize": 8.5, "lines.linewidth": 1.6, "savefig.dpi": 150,
})

T0 = time.time()
F: dict = {"config": {}}


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f} s] {msg}", flush=True)


def save(fig, name: str) -> None:
    path = OUT / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    log(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.0f} KB)")


def rel(x, th):
    return (np.asarray(x)[..., :3] - th[:3]) / th[:3]


def rel_cov(Fm, d):
    """Covariance of θ/d from the Fisher information, inverted in relative units."""
    return _relative_inverse(Fm, d) / np.outer(d, d)


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
N_ATOMS, SPACING, OMEGA, DELTA = 6, 8.0, 12.0, 5.0
RD = AQUILA.readout_model()
model = RydbergQuenchModel(chain(N_ATOMS, SPACING))
theta = model.theta(OMEGA, DELTA)
TIMES = np.linspace(0.5, 4.0, 8)
GRID = np.round(np.arange(0.1, AQUILA.t_max + 1e-9, 0.05), 10)      # probe times are total pulse durations, ≥ 2 ramps
NOISE = DeviceNoise.aquila()
F["config"] = {"atoms": N_ATOMS, "geometry": "chain", "spacing_um": SPACING, "omega": OMEGA, "delta": DELTA, "c6": C6_RB87,
               "readout_p_ground_as_rydberg": RD.p01, "readout_p_rydberg_as_ground": RD.p10, "probe_times_us": TIMES.tolist(),
               "nearest_neighbour_V": C6_RB87 / SPACING**6, "blockade_radius_um": blockade_radius(OMEGA, DELTA),
               "ramp_us": model.ramp_time, "ramp_slices": model.ramp_slices, "waveform_violations": model.waveform_violations(theta, TIMES),
               "example_waveform_4us": model.waveform(theta, 4.0),
               "device_noise": {"sigma_delta": NOISE.sigma_delta, "sigma_omega_rel": NOISE.sigma_omega_rel, "t_relax": NOISE.t_relax}}
log(f"{N_ATOMS}-atom chain at {SPACING} μm, Ω = {OMEGA}, Δ = {DELTA} rad/μs, V_nn = {C6_RB87 / SPACING**6:.1f} rad/μs")

# --------------------------------------------------------------------------- #
# (a) identifiability
# --------------------------------------------------------------------------- #
P0 = model.probabilities(theta, TIMES, RD)
sym = {
    "omega_sign": float(np.abs(model.probabilities(theta * [-1, 1, 1], TIMES, RD) - P0).max()),
    "delta_and_c6_sign": float(np.abs(model.probabilities(theta * [1, -1, -1], TIMES, RD) - P0).max()),
    "delta_sign_alone": float(np.abs(model.probabilities(theta * [1, -1, 1], TIMES, RD) - P0).max()),
    "length_scale_1.05": float(np.abs(model.with_positions(1.05 * model.positions).probabilities(theta * [1, 1, 1.05**6], TIMES, RD) - P0).max()),
}
idn = model.identifiability(theta, TIMES, RD)
cov1 = rel_cov(model.fisher_information(theta, TIMES, 1.0, RD), theta)
sd1 = np.sqrt(np.diag(cov1))
F["identifiability"] = {
    "symmetry_max_probability_change": sym,
    "relative_fisher_eigenvalues_per_shot_per_time": idn["eigenvalues"].tolist(),
    "worst_direction_relative_units": dict(zip(NAMES, idn["directions"][:, 0].tolist())),
    "relative_crb_one_shot_per_time": dict(zip(NAMES, idn["relative_crb_per_shot"].tolist())),
    "correlation": (cov1 / np.outer(sd1, sd1)).round(3).tolist(),
}
log(f"symmetries {sym}; worst direction {np.round(idn['directions'][:, 0], 3)}")

# estimator Monte Carlo against the CRB (uniform design). Continuation alone (search=False) converges on this
# chain from ±8 % starts (see the convergence table below), so the slower multi-start search is skipped here.
budgets = [50, 200, 800]
R_MLE = 12 if QUICK else 200                # about 5 % sampling error on each RMSE
mc = {}
for S in budgets:
    rng = np.random.default_rng(S)
    mle, gls = [], []
    for r in range(R_MLE):
        data = model.sample(theta, TIMES, S, np.random.default_rng(10_000 * S + r), readout=RD)
        start = theta * (1 + 0.08 * rng.uniform(-1, 1, 3))
        mle.append(model.fit_mle(data, start, readout=RD, search=False).theta)
        gls.append(model.fit_correlators(data, start, readout=RD).theta)
    crb = model.cramer_rao(theta, TIMES, S, RD)["relative_std"]
    ccrb = np.sqrt(np.diag(rel_cov(model.correlator_fisher_information(theta, TIMES, S, RD, wrt=[0, 1, 2]), theta)))
    mc[S] = {"total_shots": S * len(TIMES), "crb": crb.tolist(), "correlator_crb": ccrb.tolist(),
             "mle_rmse": np.sqrt(np.mean(rel(mle, theta) ** 2, axis=0)).tolist(), "mle_bias": np.mean(rel(mle, theta), axis=0).tolist(),
             "gls_rmse": np.sqrt(np.mean(rel(gls, theta) ** 2, axis=0)).tolist(), "trials": R_MLE}
    log(f"S = {S}/time: CRB {np.round(100 * crb, 2)} %  MLE {np.round(100 * np.array(mc[S]['mle_rmse']), 2)} %  "
        f"GLS {np.round(100 * np.array(mc[S]['gls_rmse']), 2)} % (correlator CRB {np.round(100 * ccrb, 2)} %)")
F["estimators"] = mc

# C6 against spacing, for a pair and for the chain: with Ω and Δ fitted too, and with C6 alone
spacings = np.linspace(4.0, 14.0, 21 if QUICK else 41)
scan = {"spacing_um": spacings.tolist(), "anti_blockade_resonance_um": float((C6_RB87 / (2 * DELTA)) ** (1 / 6))}
for label, n in (("pair", 2), ("chain6", N_ATOMS)):
    out, alone = [], []
    for a in spacings:
        m = RydbergQuenchModel(chain(n, a))
        out.append(m.shots_for_precision(theta, TIMES, 0.01, RD)["per_parameter"][2])
        alone.append(m.shots_for_precision(theta, TIMES, 0.01, RD, wrt=[2])["total_shots"])
    scan[label] = out
    scan[label + "_c6_alone"] = alone
    scan[label + "_at"] = {f"{a:g}": {"c6_with_omega_delta": float(np.interp(a, spacings, out)),
                                      "c6_alone": float(np.interp(a, spacings, alone))} for a in (8.0, 9.0, 10.0, 11.0, 12.0)}
F["spacing_scan_shots_for_1pct_c6"] = scan
log(f"spacing scan: pair {scan['pair_at']}; chain {scan['chain6_at']}")

# Heisenberg thermal state: which field directions the observables see
betas = np.linspace(0.1, 3.0, 30)
Jh, hh = np.array([[0, 0.75], [0.75, 0]]), np.array([0.2, -0.15])
heis = {"beta": betas.tolist(), "uniform_field": [], "staggered_field": [], "coupling": []}
for b in betas:
    L = DifferentiableHamiltonianLearner(2, beta=b)
    jac = L.jacobian(L.pack(Jh, hh))
    heis["uniform_field"].append(float(np.linalg.norm(jac @ np.array([0, 1, 1]) / np.sqrt(2))))
    heis["staggered_field"].append(float(np.linalg.norm(jac @ np.array([0, 1, -1]) / np.sqrt(2))))
    heis["coupling"].append(float(np.linalg.norm(jac @ np.array([1, 0, 0]))))
for b in (0.3, 1.5):
    info = DifferentiableHamiltonianLearner(2, beta=b).identifiability(Jh, hh)
    heis[f"beta_{b}"] = {"singular_values": info["singular_values"].tolist(), "condition_number": info["condition_number"],
                         "least_identifiable": info["least_identifiable"].tolist()}
F["heisenberg"] = heis

# --------------------------------------------------------------------------- #
# (b) probe times
# --------------------------------------------------------------------------- #
D = theta[:3]
design = {}
for label, noise in (("closed", None), ("aquila_noise", NOISE)):
    Fk = model.fisher_information(theta, GRID, 1.0, RD, noise, wrt=[0, 1, 2], per_time=True) * np.outer(D, D)[None]
    diag = np.array([np.diag(f) for f in Fk])
    uniform = {K: model.shots_for_precision(theta, np.linspace(AQUILA.t_max / K, AQUILA.t_max, K), 0.01, RD, noise)["total_shots"]
               for K in range(1, 13)}
    opt = {K: model.optimal_probe_times(theta, K, GRID, 0.01, RD, noise, per_time_fisher=Fk) for K in range(1, 7)}
    design[label] = {"grid_us": GRID.tolist(), "relative_fisher_diagonal": diag.tolist(),
                     "peak_time_us": dict(zip(NAMES, GRID[np.argmax(diag, axis=0)].tolist())),
                     "uniform_shots": uniform, "optimal_shots": {K: o["total_shots"] for K, o in opt.items()},
                     "optimal_times": {K: o["times"].tolist() for K, o in opt.items()},
                     "uniform8_0.5_to_4_shots": model.shots_for_precision(theta, TIMES, 0.01, RD, noise)["total_shots"]}
    log(f"{label}: uniform-8 {design[label]['uniform8_0.5_to_4_shots']:.0f} shots, best single time "
        f"{opt[1]['times'][0]:.2f} μs → {opt[1]['total_shots']:.0f} shots; per-parameter peaks {design[label]['peak_time_us']}")
t_star = design["closed"]["optimal_times"][1][0]

# which part of the noise model costs the shots (uniform 8 probe times)
A = AQUILA
noise_cases = {"closed": None,
               "drive_noise_only": DeviceNoise(A.delta_shot_to_shot, A.omega_shot_to_shot_rel, None),
               "relaxation_only_t2_echo": DeviceNoise(0.0, 0.0, A.t2_echo),
               "drive_noise_and_t2_echo": DeviceNoise.aquila(),
               "drive_noise_and_2x_t2_echo": DeviceNoise.aquila(t_relax=2 * A.t2_echo),
               "drive_noise_and_t2_rabi": DeviceNoise.aquila(t_relax=A.t2_rabi)}
design["noise_attribution_uniform8"] = {k: model.shots_for_precision(theta, TIMES, 0.01, RD, nz)["total_shots"]
                                        for k, nz in noise_cases.items()}
log("noise attribution: " + ", ".join(f"{k} {v:.0f}" for k, v in design["noise_attribution_uniform8"].items()))

# the same with the current Braket device properties (larger readout errors, T2 echo 7 μs), the datasheet staying the reference
B = AQUILA_BRAKET
RD_B = B.readout_model()
braket_cases = {"closed": None,
                "drive_noise_only": DeviceNoise(B.delta_shot_to_shot, B.omega_shot_to_shot_rel, None),
                "relaxation_only_t2_echo": DeviceNoise(0.0, 0.0, B.t2_echo),
                "drive_noise_and_t2_echo": DeviceNoise.braket(),
                "drive_noise_and_2x_t2_echo": DeviceNoise.braket(t_relax=2 * B.t2_echo),
                "drive_noise_and_t2_echo_omega_noise_2pct": DeviceNoise(B.delta_shot_to_shot, 0.02, B.t2_echo)}
Fk_B = model.fisher_information(theta, GRID, 1.0, RD_B, wrt=[0, 1, 2], per_time=True) * np.outer(D, D)[None]
opt_B = model.optimal_probe_times(theta, 1, GRID, 0.01, RD_B, per_time_fisher=Fk_B)
Fk_Bn = model.fisher_information(theta, GRID, 1.0, RD_B, DeviceNoise.braket(), wrt=[0, 1, 2], per_time=True) * np.outer(D, D)[None]
opt_Bn = model.optimal_probe_times(theta, 1, GRID, 0.01, RD_B, DeviceNoise.braket(), per_time_fisher=Fk_Bn)
design["braket_figures"] = {
    "readout_p_ground_as_rydberg": RD_B.p01, "readout_p_rydberg_as_ground": RD_B.p10, "t2_echo": B.t2_echo, "t2_rabi": B.t2_rabi,
    "uniform8_by_noise_model": {k: model.shots_for_precision(theta, TIMES, 0.01, RD_B, nz)["total_shots"] for k, nz in braket_cases.items()},
    "single_time_closed": {"time_us": float(opt_B["times"][0]), "shots": opt_B["total_shots"]},
    "single_time_with_noise": {"time_us": float(opt_Bn["times"][0]), "shots": opt_Bn["total_shots"]},
    "anchored_closed": float(np.max(np.diag(rel_cov(model.fisher_information(theta, [0.75, 1.5, 2.25, 3.0, t_star],
                                                                              [0.05, 0.05, 0.05, 0.05, 0.80], RD_B, wrt=[0, 1, 2]), D)))) / 1e-4}
log("Braket figures: " + ", ".join(f"{k} {v:.0f}" for k, v in design["braket_figures"]["uniform8_by_noise_model"].items())
    + f"; single time {opt_B['times'][0]:.2f} μs {opt_B['total_shots']:.0f}, with noise {opt_Bn['times'][0]:.2f} μs {opt_Bn['total_shots']:.0f}")


# the single-time optimum is sharp: neighbouring times, timing errors, and designing with slightly wrong nominal values
def single_time_shots(t, th=theta):
    Ft = model.fisher_information(th, [t], 1.0, RD, wrt=[0, 1, 2])
    return float(np.max(np.diag(rel_cov(Ft, th[:3])))) / 1e-4


fine = np.round(np.arange(t_star - 0.3, t_star + 0.3 + 1e-9, 0.01), 10)
misdesign = {}
for a, nm in enumerate(NAMES):
    for e in (-0.03, -0.02, 0.02, 0.03):
        th_n = theta.copy(); th_n[a] *= 1 + e
        Fn = model.fisher_information(th_n, GRID, 1.0, RD, wrt=[0, 1, 2], per_time=True) * np.outer(th_n[:3], th_n[:3])[None]
        t_n = float(model.optimal_probe_times(th_n, 1, GRID, 0.01, RD, per_time_fisher=Fn)["times"][0])
        misdesign[f"{nm}{100 * e:+.0f}%"] = {"chosen_time_us": t_n, "true_shots": single_time_shots(t_n)}
design["single_time_sensitivity"] = {
    "t_star_us": t_star, "shots_at_t_star": single_time_shots(t_star),
    "neighbours": {f"{t_star + d:.2f}": single_time_shots(t_star + d) for d in (-0.1, -0.05, 0.05, 0.1)},
    "fine_grid_us": fine.tolist(), "fine_grid_shots": [single_time_shots(t) for t in fine],
    "designed_with_wrong_nominal_values": misdesign,
    "worst_misdesign_shots": max(v["true_shots"] for v in misdesign.values())}
log(f"single time {t_star:.2f} μs: {design['single_time_sensitivity']['shots_at_t_star']:.0f} shots; neighbours "
    f"{ {k: round(v) for k, v in design['single_time_sensitivity']['neighbours'].items()} }; "
    f"worst with 2–3 % wrong nominal values {design['single_time_sensitivity']['worst_misdesign_shots']:.0f}")

# a single late time minimises the bound; a few early anchors make the fit find the right maximum
anchor_t = np.array([0.75, 1.5, 2.25, 3.0, t_star])
anchor_f = np.array([0.05, 0.05, 0.05, 0.05, 0.80])
Fa = model.fisher_information(theta, anchor_t, anchor_f, RD, wrt=[0, 1, 2])
anchored_shots = float(np.max(np.diag(rel_cov(Fa, D)))) / 1e-4
designs = {"single": (np.array([t_star]), np.array([1.0]), design["closed"]["optimal_shots"][1]),
           "anchored": (anchor_t, anchor_f, anchored_shots)}
R_CONV = 10 if QUICK else 30
conv = {}
for label, (tt, ff, need) in designs.items():
    rng = np.random.default_rng(3)
    ok, flagged, wrong_unflagged, errs = 0, 0, 0, []
    for r in range(R_CONV):
        shots = np.maximum(1, np.round(2 * need * ff)).astype(int)      # twice the 1 % budget
        data = model.sample(theta, tt, shots, np.random.default_rng(20_000 + r), readout=RD)
        fit = model.fit_mle(data, theta * (1 + 0.1 * rng.uniform(-1, 1, 3)), readout=RD)
        ref = model.fit_mle(data, theta, readout=RD, search=False, continuation=False)      # the optimum next to the truth
        ok_r = bool(fit.objective <= ref.objective + 1e-3)
        ok += int(ok_r)
        flagged += int(not fit.success)
        wrong_unflagged += int((not ok_r) and fit.success)
        errs.append(rel(fit.theta, theta))
    conv[label] = {"times_us": tt.tolist(), "shot_fractions": ff.tolist(), "shots_for_1pct": float(need),
                   "shots_used": "twice the 1 % budget", "converged_from_10pct": ok, "flagged": flagged,
                   "wrong_and_not_flagged": wrong_unflagged, "trials": R_CONV,
                   "rmse_at_twice_budget": np.sqrt(np.mean(np.square(errs), axis=0)).tolist()}
    log(f"{label} design: {need:.0f} shots for 1 %, converged {ok}/{R_CONV} from ±10 % starts, {flagged} flagged, "
        f"{wrong_unflagged} wrong without a warning")
design["convergence"] = conv

# convergence of the fit itself: exact expected counts (200 per time, 8 times), starts within ±8 %
conv_cases = {"chain6@8": chain(6, 8.0), "ring6@7": ring(6, 7.0), "chain4@7": chain(4, 7.0), "chain3@6.5": chain(3, 6.5),
              "chain6@6.5": chain(6, 6.5), "pair@8": chain(2, 8.0), "pair@6.5": chain(2, 6.5)}
R_START = 4 if QUICK else 15
table = {}
for label, pos in conv_cases.items():
    m = RydbergQuenchModel(pos)
    counts = 200 * m.probabilities(theta, TIMES, RD)
    row = {"trials": R_START, "period_c6_rel": float(2 * np.pi / (C6_RB87 / m.distances.min() ** 6 * TIMES.max()))}
    for mode, search in (("continuation", False), ("default", None)):
        right = flagged = wrong_unflagged = 0
        alts = []
        for r in range(R_START):
            start = theta * (1 + 0.08 * np.random.default_rng(100 + r).uniform(-1, 1, 3))
            fit = m.fit_mle(None, start, readout=RD, times=TIMES, counts=counts, search=search)
            ok_r = bool(np.abs(fit.theta / theta - 1).max() < 1e-3)
            right += ok_r; flagged += int(not fit.success); wrong_unflagged += int((not ok_r) and fit.success)
            if ok_r and not alts:
                alts = sorted((float(a[2] / theta[2] - 1), gap) for a, gap in fit.alternatives)
        row[mode] = {"found_truth": right, "flagged": flagged, "wrong_and_not_flagged": wrong_unflagged,
                     "alternatives_c6_offset_and_log_likelihood_gap": alts}
    table[label] = row
    log(f"convergence {label}: {row}")
design["fit_convergence_exact_counts"] = table

# the same with sampled data (200 shots per time): how often a fit is more than 4 reported σ off, and whether it says so
R_SAMP = 4 if QUICK else 16
sampled = {}
for label in ("chain6@8", "chain4@7", "ring6@7"):
    m = RydbergQuenchModel(conv_cases[label])
    flagged = far = far_unflagged = ambiguous = 0
    preference = []
    for r in range(R_SAMP):
        rng = np.random.default_rng(1000 + r)
        data = m.sample(theta, TIMES, 200, rng, readout=RD)
        fit = m.fit_mle(data, theta * (1 + 0.08 * rng.uniform(-1, 1, 3)), readout=RD)
        sd = np.array([fit.std[nm] for nm in NAMES])
        is_far = bool(np.any(np.abs(fit.theta[:3] - theta[:3]) > 4 * sd))
        flagged += int(not fit.success); ambiguous += int(bool(fit.alternatives))
        far += int(is_far); far_unflagged += int(is_far and fit.success)
        if is_far and fit.success:               # how much better the data fit the far maximum than the one next to the truth
            preference.append(float(m.fit_mle(data, theta, readout=RD, search=False, continuation=False).objective - fit.objective))
    sampled[label] = {"trials": R_SAMP, "flagged": flagged, "flagged_ambiguous": ambiguous,
                      "more_than_4sigma_off": far, "more_than_4sigma_off_not_flagged": far_unflagged,
                      "not_flagged_far_fits_preferred_by_log_likelihood": preference}
    log(f"sampled data {label}: {sampled[label]}")
design["fit_convergence_sampled_200_per_time"] = sampled
F["design"] = design

# --------------------------------------------------------------------------- #
# (c) robustness: pulse shape, position errors, empty sites and readout
# --------------------------------------------------------------------------- #
# Large-shot fits (expected counts) start at the truth, so they report the shift of the nearest optimum;
# "from_offset_start" repeats a few of them from a start 3 % off with the default search. "The maximum next to the
# truth" is always a local fit from the truth to all probe times at once (continuation=False): continuation from the
# truth can still walk to a neighbouring maximum.
robust = {}
P_ramp = model.probabilities(theta, TIMES, RD) * 1e3
inst = model.with_ramp(0.0)
f_total = inst.fit_mle(None, theta, readout=RD, times=TIMES, counts=P_ramp, search=False)
f_area = inst.fit_mle(None, theta, readout=RD, times=TIMES - model.ramp_time, counts=P_ramp, search=False)
robust["ramps_ignored"] = {
    "ramp_us": model.ramp_time,
    "instantaneous_model_at_total_duration": rel(f_total.theta, theta).tolist(),
    "instantaneous_model_at_equal_area_duration": rel(f_area.theta, theta).tolist(),
    "gof_p_total_duration_1000_per_time": f_total.gof_pvalue,
    "shots_for_1pct_with_ramps": design["closed"]["uniform8_0.5_to_4_shots"],
    "shots_for_1pct_instantaneous": inst.shots_for_precision(theta, TIMES, 0.01, RD)["total_shots"],
    "slice_convergence_max_prob_error": {str(m_s): float(np.abs(model.with_ramp(model.ramp_time, m_s).probabilities(theta, TIMES)
                                                                - model.with_ramp(model.ramp_time, 256).probabilities(theta, TIMES)).max())
                                         for m_s in (2, 4, 8, 16)}}
# Braket lists a Rabi correction of 0.92 for 50 ns ramps (0.97 at 75 ns, 1.0 at 100 ns): the effective on-resonance Rabi
# frequency of a triangular pulse relative to the specified one. Modelled as ramps that deliver 92 % of their nominal area,
# a fit with nominal ramps is off by this much:
short = model.with_ramp(model.ramp_time)
short._ramp_shape = 0.92 * short._ramp_shape
f_corr = model.fit_mle(None, theta, readout=RD, times=TIMES, counts=short.probabilities(theta, TIMES, RD) * 1e3, search=False)
robust["ramps_ignored"]["ramp_area_0.92_fitted_with_nominal_ramps"] = rel(f_corr.theta, theta).tolist()
log(f"ramps ignored: {np.round(100 * rel(f_total.theta, theta), 2)} % at total duration, "
    f"{np.round(100 * rel(f_area.theta, theta), 3)} % at the equal-area duration")

n_draws = 800 if QUICK else 4000


def jitter_bias(m, seed, label):
    """Large-shot fits to the jitter-averaged distribution: the best maximum the fit finds, and the one next to the truth."""
    Pa, batches = m.averaged_probabilities(theta, TIMES, AQUILA.position_jitter, n_draws, np.random.default_rng(seed), RD,
                                           return_batches=8)
    best = m.fit_mle(None, theta, readout=RD, times=TIMES, counts=Pa * 1e3)
    off = m.fit_mle(None, theta * [1.03, 0.97, 1.03], readout=RD, times=TIMES, counts=Pa * 1e3)
    near = m.fit_mle(None, theta, readout=RD, times=TIMES, counts=Pa * 1e3, search=False, continuation=False)
    b_best = [rel(m.fit_mle(None, theta, readout=RD, times=TIMES, counts=B * 1e3).theta, theta) for B in batches]
    b_near = [rel(m.fit_mle(None, theta, readout=RD, times=TIMES, counts=B * 1e3, search=False, continuation=False).theta, theta)
              for B in batches]
    out = {"sigma_um_per_coordinate": AQUILA.position_jitter, "draws": n_draws,
           "spacing_over_blockade_radius": float(m.distances.min() / blockade_radius(OMEGA, DELTA)),
           "bias": rel(best.theta, theta).tolist(), "mc_error": (np.std(b_best, axis=0) / np.sqrt(len(b_best))).tolist(),
           "gof_p_1000_per_time": best.gof_pvalue, "bias_from_offset_start": rel(off.theta, theta).tolist(),
           "nearest_maximum": rel(near.theta, theta).tolist(), "nearest_mc_error": (np.std(b_near, axis=0) / np.sqrt(len(b_near))).tolist(),
           "nearest_gof_p_1000_per_time": near.gof_pvalue,
           "nearest_minus_best_log_likelihood_1000_per_time": float(near.objective - best.objective)}
    log(f"{label} jitter: best {np.round(100 * rel(best.theta, theta), 2)} % (gof p {best.gof_pvalue:.2g}, offset start "
        f"{np.round(100 * rel(off.theta, theta), 2)} %), nearest {np.round(100 * rel(near.theta, theta), 2)} % "
        f"(gof p {near.gof_pvalue:.2g}, {near.objective - best.objective:.1f} worse)")
    return out


robust["thermal_jitter"] = jitter_bias(model, 7, "chain")
robust["thermal_jitter_ring6_at_7um"] = jitter_bias(RydbergQuenchModel(ring(6, 7.0)), 8, "ring")
robust["thermal_jitter_instantaneous_quench"] = {
    "chain": rel(inst.fit_mle(None, theta, readout=RD, times=TIMES, search=False,
                              counts=inst.averaged_probabilities(theta, TIMES, AQUILA.position_jitter, n_draws,
                                                                 np.random.default_rng(7), RD) * 1e3).theta, theta).tolist()}
log(f"chain jitter, instantaneous quench: {np.round(100 * np.array(robust['thermal_jitter_instantaneous_quench']['chain']), 2)} %")


def with_single_vacancies(m, th, times, p):
    """Expected outcome distribution when each site is empty with probability p (to first order in p) and nobody post-selects."""
    n = m.n
    P = (1 - p) ** n * m.probabilities(th, times, RD)
    weights = 1 << np.arange(n - 2, -1, -1)
    for i in range(n):
        keep = [j for j in range(n) if j != i]
        Ps = m.with_positions(m.positions[keep]).probabilities(th, times, RD)
        sel = np.flatnonzero(m.bits[:, i] == 1)                       # an empty site reads as Rydberg
        full = np.zeros_like(P)
        full[:, sel] = Ps[:, m.bits[sel][:, keep].astype(int) @ weights]
        P += p * (1 - p) ** (n - 1) * full
    return P / P.sum(axis=1, keepdims=True)


Pv = with_single_vacancies(model, theta, TIMES, AQUILA.p_vacancy)
fv = model.fit_mle(None, theta, readout=RD, times=TIMES, counts=Pv * 1e3, search=False)
robust["empty_sites_not_postselected"] = {"p_vacancy": AQUILA.p_vacancy, "fraction_of_shots_with_an_empty_site": 1 - (1 - AQUILA.p_vacancy) ** N_ATOMS,
                                          "bias": rel(fv.theta, theta).tolist()}
log(f"empty sites kept: {100 * (1 - (1 - AQUILA.p_vacancy) ** N_ATOMS):.1f} % of shots, bias {np.round(100 * rel(fv.theta, theta), 3)} %")
rng = np.random.default_rng(11)
sys_err = []
for _ in range(20 if QUICK else 80):
    pos = model.positions + AQUILA.position_systematic * rng.standard_normal(model.positions.shape)
    Pk = model.with_positions(pos).probabilities(theta, TIMES, RD)
    sys_err.append(rel(model.fit_mle(None, theta, readout=RD, times=TIMES, counts=Pk * 1e3, search=False).theta, theta))
sys_err = np.array(sys_err)
robust["pattern_error"] = {"sigma_um": AQUILA.position_systematic, "rms": np.sqrt(np.mean(sys_err**2, axis=0)).tolist(),
                           "mean": sys_err.mean(axis=0).tolist(), "patterns": len(sys_err)}
log(f"pattern error RMS {np.round(100 * np.array(robust['pattern_error']['rms']), 2)} %")
Pr = P0 * 1e3
robust["readout_ignored_mle"] = {"bias": rel(model.fit_mle(None, theta, readout=None, times=TIMES, counts=Pr, search=False).theta, theta).tolist()}
robust["readout_ignored_correlators"] = {"bias": rel(model.fit_correlators(None, theta, readout=None, times=TIMES, counts=Pr).theta, theta).tolist()}
robust["readout_modelled_mle"] = {"bias": rel(model.fit_mle(None, theta * 1.02, readout=RD, times=TIMES, counts=Pr, search=False).theta, theta).tolist()}
robust["readout_corrected_correlators"] = {"bias": rel(model.fit_correlators(None, theta * 1.02, readout=RD, times=TIMES, counts=Pr).theta, theta).tolist()}
other = {}
for label, tt in (("single_late_time", np.array([t_star])), ("early_0.25_to_1", np.linspace(0.25, 1.0, 4))):
    Pd = model.probabilities(theta, tt, RD) * 1e3
    other[label] = {"times_us": tt.tolist(),
                    "mle": rel(model.fit_mle(None, theta, readout=None, times=tt, counts=Pd, search=False).theta, theta).tolist(),
                    "correlators": rel(model.fit_correlators(None, theta, readout=None, times=tt, counts=Pd).theta, theta).tolist()}
fu = model.fit_mle(None, theta, readout=None, times=TIMES, counts=Pr, search=False)
Qu = model.probabilities(fu.theta, TIMES)
other["misfit_nats_per_shot_uniform8"] = float(np.sum(P0 * np.log(P0 / np.clip(Qu, 1e-300, None))) / len(TIMES))
robust["readout_ignored_other_designs"] = other
s_clean = model.shots_for_precision(theta, TIMES, 0.01, None)["total_shots"]
robust["shots_for_1pct"] = {"no_readout_errors": s_clean, "aquila_readout": design["closed"]["uniform8_0.5_to_4_shots"],
                            "aquila_readout_and_noise": design["aquila_noise"]["uniform8_0.5_to_4_shots"]}
for k in ("readout_ignored_mle", "readout_ignored_correlators", "readout_modelled_mle", "readout_corrected_correlators"):
    log(f"{k}: {np.round(100 * np.array(robust[k]['bias']), 3)} %")
F["robustness"] = robust

# --------------------------------------------------------------------------- #
# (d) drift tracking
# --------------------------------------------------------------------------- #
Fd = model.fisher_information(theta, GRID, 1.0, RD, wrt=[1], per_time=True)[:, 0, 0]
probe = np.sort(GRID[np.argsort(Fd)[::-1][:2]])
Q = 2e-3
z_end, err_end, sd_end = [], [], []
R_KF = 20 if QUICK else 100
for r in range(R_KF):
    rng = np.random.default_rng(30_000 + r)
    traj = DELTA + 0.05 * rng.standard_normal() + np.concatenate([[0.0], np.cumsum(Q * rng.standard_normal(399))])
    times, bits = simulate_drifting_stream(model, theta, traj, probe, rng, readout=RD)
    tr = RydbergDriftTracker(model, theta, readout=RD, process_noise_std=Q, prior_std=0.05)
    h = tr.track_stream(times, bits)
    z_end.append((h["estimate"][-1, 0] - traj[-1]) / h["std"][-1, 0])
    err_end.append(h["estimate"][-1, 0] - traj[-1])
    sd_end.append(h["std"][-1, 0])
z_end = np.array(z_end)
n_trace = 4000 if QUICK else 20000
Q_TRACE = Q                         # the long trace is a random walk with the filter's own process noise
rng = np.random.default_rng(99)
traj = DELTA + np.concatenate([[0.0], np.cumsum(Q_TRACE * rng.standard_normal(n_trace - 1))])
times, bits = simulate_drifting_stream(model, theta, traj, probe, rng, readout=RD)
start = theta.copy(); start[1] = traj[0] + 0.05 * rng.standard_normal()
tr = RydbergDriftTracker(model, start, readout=RD, process_noise_std=Q_TRACE, prior_std=0.05)
h = tr.track_stream(times, bits)
est, sd = h["estimate"][:, 0], h["std"][:, 0]
zt = (est - traj) / sd
F["drift"] = {"probe_times_us": probe.tolist(), "process_noise_per_shot": Q, "runs": R_KF, "shots_per_run": 400,
              "end_mean_z2": float(np.mean(z_end**2)), "end_coverage_1sigma": float(np.mean(np.abs(z_end) < 1)),
              "end_coverage_2sigma": float(np.mean(np.abs(z_end) < 2)), "end_rmse": float(np.sqrt(np.mean(np.square(err_end)))),
              "end_mean_reported_std": float(np.mean(sd_end)),
              "end_mean_z2_standard_error": float(np.std(z_end**2) / np.sqrt(len(z_end))),
              "trace_shots": n_trace, "trace_process_noise_per_shot": Q_TRACE, "trace_rmse": float(np.sqrt(np.mean((est - traj)[500:] ** 2))),
              "trace_mean_std": float(np.mean(sd[500:])), "trace_coverage_2sigma": float(np.mean(np.abs(zt[500:]) < 2))}
log(f"drift: end-of-run mean z² {F['drift']['end_mean_z2']:.2f}, 1σ {F['drift']['end_coverage_1sigma']:.2f}, "
    f"2σ {F['drift']['end_coverage_2sigma']:.2f}; trace RMSE {F['drift']['trace_rmse']:.3f} (σ {F['drift']['trace_mean_std']:.3f})")

# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
# 1. Cramér–Rao bound against shots, with the two estimators
fig, ax = plt.subplots(figsize=(5.6, 3.6))
S_tot = np.logspace(np.log10(200), np.log10(2e4), 50)
crb1 = model.cramer_rao(theta, TIMES, 1.0, RD)["relative_std"] * np.sqrt(len(TIMES))
for k, nm in enumerate(NAMES):
    ax.loglog(S_tot, 100 * crb1[k] / np.sqrt(S_tot), color=COL[nm], lw=1.6)
    ax.text(S_tot[-1] * 1.06, 100 * crb1[k] / np.sqrt(S_tot[-1]), LAB[nm], color=INK, va="center", fontsize=9.5)
    xs = [mc[S]["total_shots"] for S in budgets]
    ax.plot(xs, [100 * mc[S]["mle_rmse"][k] for S in budgets], "o", ms=5.5, color=COL[nm], mec="white", mew=0.8, zorder=3)
    ax.plot(xs, [100 * mc[S]["gls_rmse"][k] for S in budgets], "s", ms=5, mfc="white", mec=COL[nm], mew=1.2, zorder=3)
ax.axhline(1.0, color=GRAY, lw=0.9, ls=(0, (4, 3)))
ax.text(230, 1.08, "1 %", color=GRAY, fontsize=8.5)
ax.plot([], [], "-", color=GRAY, label="Cramér–Rao bound")
ax.plot([], [], "o", color=GRAY, mec="white", label="maximum likelihood, bitstrings")
ax.plot([], [], "s", mfc="white", mec=GRAY, mew=1.2, label="least squares, ⟨nᵢ⟩ and ⟨nᵢnⱼ⟩")
ax.legend(loc="lower left")
ax.set_xlabel("total shots (8 probe times, 0.5–4 μs)")
ax.set_ylabel("relative error (%)")
ax.set_xlim(180, 3.2e4)
ax.set_title(f"{N_ATOMS}-atom chain, {SPACING:g} μm, Ω = {OMEGA:g}, Δ = {DELTA:g} rad/μs, Aquila readout errors", color=INK)
save(fig, "crb_vs_shots.png")

# 2. information against probe time, and shots against the number of probe times
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.3), gridspec_kw={"width_ratios": [1.25, 1]})
diag = np.array(design["closed"]["relative_fisher_diagonal"])
diag_n = np.array(design["aquila_noise"]["relative_fisher_diagonal"])
for k, nm in enumerate(NAMES):
    a1.semilogy(GRID, diag[:, k], color=COL[nm], lw=1.5)
    a1.semilogy(GRID, diag_n[:, k], color=COL[nm], lw=0.9, alpha=0.55)
    a1.text(GRID[-1] + 0.06, diag[-1, k], LAB[nm], color=INK, va="center", fontsize=9.5)
a1.axvline(t_star, color=GRAY, lw=0.8)
a1.text(t_star - 0.05, 2.6e3, f"{t_star:.2f} μs", color=GRAY, ha="right", va="center", fontsize=8.5)
a1.plot([], [], color=GRAY, lw=1.5, label="closed system")
a1.plot([], [], color=GRAY, lw=0.9, alpha=0.55, label="with drive noise and T₂ relaxation")
a1.legend(loc="lower right", bbox_to_anchor=(0.84, 0.0))
a1.set_ylim(1e-1, 5e3)
a1.set_xlabel("probe time t (μs)")
a1.set_ylabel("information per shot, θ² F(t)")
a1.set_title("Information about each parameter, one shot at time t", color=INK)
Ks = np.arange(1, 13)
for label, mk, lw, alpha in (("closed", "o", 1.4, 1.0), ("aquila_noise", "o", 1.0, 0.5)):
    d = design[label]
    a2.semilogy(Ks, [d["uniform_shots"][K] for K in Ks], "-", marker=mk, ms=4, color=INK, lw=lw, alpha=alpha, mfc="white")
    Ko = np.arange(1, 7)
    a2.semilogy(Ko, [d["optimal_shots"][K] for K in Ko], "-", marker="s", ms=4, color=COL["delta"], lw=lw, alpha=alpha)
a2.text(12.3, design["closed"]["uniform_shots"][12], "evenly\nspaced", fontsize=8.5, color=INK, va="center")
a2.text(6.3, design["closed"]["optimal_shots"][6], "chosen", fontsize=8.5, color=INK, va="center")
a2.text(12.3, design["aquila_noise"]["uniform_shots"][12], "evenly spaced,\nwith noise", fontsize=8.5, color=GRAY, va="center")
a2.text(6.3, design["aquila_noise"]["optimal_shots"][6], "chosen, with noise", fontsize=8.5, color=GRAY, va="center")
a2.yaxis.set_major_locator(FixedLocator([300, 500, 1000, 2000, 5000, 10000]))
a2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
a2.yaxis.set_minor_formatter(NullFormatter())
a2.set_xlim(0.5, 14.2)
a2.set_xlabel("number of probe times")
a2.set_ylabel("shots for 1 % on Ω, Δ and C₆")
a2.set_title("Placement matters more than number", color=INK)
fig.tight_layout(w_pad=2.5)
save(fig, "fisher_vs_time.png")

# 3. bias from position errors and readout
rows = [("thermal jitter, σ = 0.2 μm", robust["thermal_jitter"]["bias"], robust["thermal_jitter"]["mc_error"]),
        ("same, 6-atom ring at 7 μm", robust["thermal_jitter_ring6_at_7um"]["bias"], robust["thermal_jitter_ring6_at_7um"]["mc_error"]),
        ("pattern error, σ = 0.05 μm (RMS)", robust["pattern_error"]["rms"], None),
        ("Ω ramps ignored", robust["ramps_ignored"]["instantaneous_model_at_total_duration"], None),
        ("empty sites kept (no post-selection)", robust["empty_sites_not_postselected"]["bias"], None),
        ("readout errors ignored, likelihood", robust["readout_ignored_mle"]["bias"], None),
        ("readout errors ignored, correlators", robust["readout_ignored_correlators"]["bias"], None),
        ("readout modelled or corrected", robust["readout_modelled_mle"]["bias"], None)]
fig, ax = plt.subplots(figsize=(6.4, 4.6))
ax.grid(False)
ax.grid(True, axis="x", color="#e8e8e8", lw=0.6)
ax.axvspan(-1, 1, color="#f2f2f2", zorder=0, lw=0)
h = 0.24
for i, (label, vals, errs) in enumerate(rows):
    y = len(rows) - 1 - i
    for k, nm in enumerate(NAMES):
        v = 100 * vals[k]
        ax.barh(y + (1 - k) * h, v, height=h - 0.03, color=COL[nm], zorder=2)
        if errs is not None:
            ax.errorbar(v, y + (1 - k) * h, xerr=100 * 2 * errs[k], color=INK, lw=0.8, capsize=2, zorder=3)
ax.axvline(0, color="#999999", lw=0.7, zorder=1)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows][::-1])
ax.tick_params(axis="y", length=0)
ax.set_xlabel("error in the large-shot limit (%)   shaded: ±1 %")
ax.text(0.12, 0, "zero (exact)", color=GRAY, va="center", fontsize=8.5)
ax.legend(handles=[Patch(color=COL[nm], label=LAB[nm]) for nm in NAMES], loc="lower left", ncol=3)
ax.set_title("Systematic errors against the 1 % target", color=INK)
save(fig, "bias_jitter_readout.png")

# 4. drift tracking
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.2), gridspec_kw={"width_ratios": [2.2, 1]})
x = np.arange(n_trace)
a1.fill_between(x, est - 2 * sd, est + 2 * sd, color=COL["delta"], alpha=0.16, lw=0)
a1.plot(x, traj, color=INK, lw=1.0, label="true Δ")
a1.plot(x, est, color=COL["delta"], lw=1.0, label="filter estimate, ±2σ band")
a1.set_xlabel("shot")
a1.set_ylabel("Δ (rad/μs)")
a1.legend(loc="lower left")
a1.set_title(f"Tracking a drifting detuning, one shot at a time (t = {probe[0]:.2f}, {probe[1]:.2f} μs)", color=INK)
bins = np.linspace(-4, 4, 17)
a2.hist(z_end, bins=bins, density=True, color=LIGHT, edgecolor="white", lw=1.0)
zz = np.linspace(-4, 4, 200)
a2.plot(zz, np.exp(-zz**2 / 2) / np.sqrt(2 * np.pi), color=INK, lw=1.2)
a2.set_xlabel("error / reported σ, end of run")
a2.set_yticks([])
a2.spines["left"].set_visible(False)
a2.grid(False)
a2.set_title(f"{R_KF} runs: mean z² = {F['drift']['end_mean_z2']:.2f}", color=INK)
fig.tight_layout(w_pad=2.0)
save(fig, "drift_tracking.png")

# 5. what the data can and cannot see
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.3))
a1.semilogy(betas, heis["coupling"], color=INK, lw=1.5)
a1.semilogy(betas, heis["staggered_field"], color=GRAY, lw=1.5)
a1.semilogy(betas, heis["uniform_field"], color=COL["delta"], lw=1.8)
a1.text(3.05, heis["coupling"][-1] * 0.82, "J", color=INK, va="center")
a1.text(3.05, heis["staggered_field"][-1] * 1.15, "h₀ − h₁", color=INK, va="center")
a1.text(3.05, heis["uniform_field"][-1], "h₀ + h₁", color=INK, va="center")
a1.set_xlim(0, 3.5)
a1.set_xlabel("inverse temperature β")
a1.set_ylabel("sensitivity of the observables")
a1.set_title("Thermal Heisenberg pair: a singlet hides a uniform field", color=INK)
for label, color, name in (("pair", GRAY, "2 atoms"), ("chain6", COL["c6"], f"{N_ATOMS}-atom chain")):
    a2.semilogy(spacings, scan[label], color=color, lw=1.6)
    a2.semilogy(spacings, scan[label + "_c6_alone"], color=color, lw=1.0, ls=(0, (3, 2)))
    j = int(np.argmin(np.abs(spacings - 12.6)))
    a2.text(spacings[j], scan[label][j] * (1.6 if label == "pair" else 0.45), name, color=INK, fontsize=8.5)
a2.plot([], [], color=INK, lw=1.6, label="Ω and Δ fitted too")
a2.plot([], [], color=INK, lw=1.0, ls=(0, (3, 2)), label="C₆ alone")
a2.legend(loc="upper right")
rb = blockade_radius(OMEGA, DELTA)
a2.axvline(rb, color=GRAY, lw=0.8)
a2.text(rb - 0.15, 1e11, f"R_b = {rb:.1f} μm", color=GRAY, fontsize=8.5, ha="right")
a2.set_xlabel("nearest-neighbour spacing (μm)")
a2.set_ylabel("shots for 1 % on C₆")
a2.set_title("Rydberg quench: deep blockade hides C₆", color=INK)
fig.tight_layout(w_pad=2.5)
save(fig, "identifiability.png")


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


(OUT / "findings.json").write_text(json.dumps(_jsonable(F), indent=1))
log(f"wrote figures/findings.json; total {time.time() - T0:.0f} s")
