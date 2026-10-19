"""
Build the Aquila programs for the reference Rydberg Hamiltonian-learning experiment.

The experiment is the reference configuration of the README section "Learning a
Rydberg Hamiltonian from snapshots": six atoms in a chain at 8 um, Omega = 12 rad/us,
Delta = 5 rad/us, 0.05 us linear Omega ramps, and eight pulses with total durations
evenly spaced from 0.5 to 4 us. Each pulse is one Braket AnalogHamiltonianSimulation
program, submitted as one task.

This script

1. writes the eight programs as Braket IR JSON (programs/*.json), which is exactly
   what submit.py sends;
2. checks every program against the package's own rules (geometry_violations,
   waveform_violations, for both the 2023 datasheet and the current Braket figures),
   against the Aquila capabilities documented by Braket (aquila_paradigm_documented.json),
   and, when amazon-braket-sdk is installed, against the SDK: the IR schema, the
   local simulator's program validator with Aquila's limits, and the SDK's
   discretisation to the device resolution, which must leave every program unchanged;
3. chooses the shots per program from the package's Cramer-Rao bound, and writes
   everything to programs/manifest.json.

Nothing here contacts AWS. Run from the repository root:

    python experiments/aquila/build.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
try:
    import shadow_learning  # noqa: F401
except ImportError:                                  # running from a checkout without `pip install -e .`
    sys.path.insert(0, str(HERE.parents[1] / "src"))

from shadow_learning.rydberg import (                # noqa: E402
    AQUILA,
    AQUILA_BRAKET,
    C6_RB87,
    DeviceNoise,
    RydbergQuenchModel,
    chain,
    geometry_violations,
)

# --------------------------------------------------------------------------- #
# the experiment
# --------------------------------------------------------------------------- #
N_ATOMS = 6
SPACING_UM = 8.0
OMEGA = 12.0                                         # rad/us
DELTA = 5.0                                          # rad/us
RAMP_US = 0.05
PROBE_TIMES_US = tuple(float(t) for t in np.round(np.linspace(0.5, 4.0, 8), 3))

#: Precision goal: Cramer-Rao relative standard deviation of C6, with Omega and Delta fitted too.
TARGET_REL_C6 = 0.01
#: Lower limit on shots per program: the goodness-of-fit chi-square needs a few hundred shots per
#: probe time, and the fit's search was validated at 200 shots per time (README).
MIN_SHOTS_PER_PROGRAM = 200
SHOT_STEP = 50

DEVICE_ARN = "arn:aws:braket:us-east-1::device/qpu/quera/Aquila"
REGION = "us-east-1"

#: On-demand pricing for QuEra Aquila from https://aws.amazon.com/braket/pricing/, read on 2026-10-19.
PRICE_PER_TASK_USD = 0.30
PRICE_PER_SHOT_USD = 0.01
PRICING_SOURCE = "https://aws.amazon.com/braket/pricing/"
PRICING_CHECKED = "2026-10-19"

PARADIGM_FILE = HERE / "aquila_paradigm_documented.json"
PROGRAMS_DIR = HERE / "programs"

FIT_PARAMS = ("omega", "delta", "c6")


# --------------------------------------------------------------------------- #
# model and shot count
# --------------------------------------------------------------------------- #
def reference_model() -> RydbergQuenchModel:
    return RydbergQuenchModel(chain(N_ATOMS, SPACING_UM), ramp_time=RAMP_US)


def nominal_theta(model: Optional[RydbergQuenchModel] = None) -> np.ndarray:
    model = model or reference_model()
    return model.theta(OMEGA, DELTA, C6_RB87)


def noise_models() -> Dict[str, tuple]:
    """The device models the shot count is computed under: (readout, noise)."""
    ro = AQUILA_BRAKET.readout_model()
    return {
        "braket readout only": (ro, None),
        "braket readout, drive noise": (ro, DeviceNoise.braket(t_relax=None)),
        "braket readout, drive noise, relaxation at T2 echo (7 us)": (ro, DeviceNoise.braket()),
    }


def choose_shots(model: Optional[RydbergQuenchModel] = None, theta: Optional[np.ndarray] = None,
                 times: Sequence[float] = PROBE_TIMES_US, target: float = TARGET_REL_C6,
                 floor: int = MIN_SHOTS_PER_PROGRAM, step: int = SHOT_STEP,
                 kept_fraction: Optional[float] = None) -> Dict[str, object]:
    """
    Shots per program from the Cramer-Rao bound of the full bitstring distribution,
    with Omega, Delta and C6 all fitted and the shots split equally over the probe
    times. The count is the one for which the relative standard deviation of C6
    reaches ``target`` under the most pessimistic of :func:`noise_models`, divided by
    the expected fraction of shots kept after postselection on a full array
    (Braket's typical filling error, 0.008 per site), raised to ``floor`` and rounded
    up to a multiple of ``step``. Also returns the predicted relative errors of all
    three parameters at that count under each model.
    """
    model = model or reference_model()
    theta = nominal_theta(model) if theta is None else np.asarray(theta, dtype=float)
    times = np.asarray(times, dtype=float)
    if kept_fraction is None:
        kept_fraction = (1.0 - AQUILA_BRAKET.p_vacancy) ** model.n
    per_model = {}
    needed = 0.0
    for name, (ro, nz) in noise_models().items():
        rel1 = model.cramer_rao(theta, times, 1.0, ro, nz)["relative_std"]       # one shot per probe time
        n_c6 = (rel1[2] / target) ** 2
        per_model[name] = {"relative_std_one_shot_per_time": dict(zip(FIT_PARAMS, map(float, rel1))),
                           "shots_per_time_for_target_c6": float(n_c6)}
        needed = max(needed, n_c6)
    raw = needed / kept_fraction
    shots = int(step * math.ceil(max(raw, floor) / step - 1e-12))
    kept = shots * kept_fraction
    for name, (ro, nz) in noise_models().items():
        rel1 = np.array(list(per_model[name]["relative_std_one_shot_per_time"].values()))
        per_model[name]["predicted_relative_std_at_chosen_shots"] = dict(zip(FIT_PARAMS, map(float, rel1 / np.sqrt(kept))))
    return {"shots_per_program": shots, "target_relative_std_c6": target, "expected_kept_fraction": float(kept_fraction),
            "shots_per_time_needed_before_postselection": float(raw), "floor": floor, "step": step, "models": per_model}


def cost_estimate(n_tasks: int, shots_per_task: int, price_task: float = PRICE_PER_TASK_USD,
                  price_shot: float = PRICE_PER_SHOT_USD) -> Dict[str, float]:
    total_shots = int(n_tasks) * int(shots_per_task)
    task_fees = n_tasks * price_task
    shot_fees = total_shots * price_shot
    return {"tasks": int(n_tasks), "shots_per_task": int(shots_per_task), "total_shots": total_shots,
            "task_fees_usd": round(task_fees, 2), "shot_fees_usd": round(shot_fees, 2),
            "total_usd": round(task_fees + shot_fees, 2), "price_per_task_usd": price_task,
            "price_per_shot_usd": price_shot, "pricing_source": PRICING_SOURCE, "pricing_checked": PRICING_CHECKED}


# --------------------------------------------------------------------------- #
# program IR, without the SDK
# --------------------------------------------------------------------------- #
def _si(value: float, unit: str, decimals: int, what: str) -> str:
    """
    An exact decimal string in SI units for a value given in package units (um, us,
    rad/us). ``decimals`` is the grid in package units (2 for 0.01 um, 3 for 1 ns);
    an off-grid value raises instead of being rounded silently.
    """
    v = float(value)
    if abs(v * 10**decimals - round(v * 10**decimals)) > 1e-6:
        raise ValueError(f"{what} = {v!r} is not a multiple of 1e-{decimals} in package units")
    d = Decimal(repr(round(v, decimals))) * Decimal({"um": "1e-6", "us": "1e-6", "rad/us": "1e6", "rad": "1"}[unit])
    d = d.normalize()
    return "0" if d == 0 else format(d, "f")


def program_ir(positions_um: np.ndarray, omega: float, delta: float, duration_us: float, ramp_us: float = RAMP_US,
               filling: Optional[Sequence[int]] = None) -> Dict[str, object]:
    """
    One Braket AHS program (``braket.ir.ahs.program`` v1) as a plain dict with exact
    decimal strings in SI units, for the package's pulse: Omega ramps linearly from 0
    over ``ramp_us``, stays flat and returns to 0 at ``duration_us``; Delta is
    constant and the phase is 0. The time series come from
    :meth:`RydbergQuenchModel.waveform`, so the program and the fitted model describe
    the same pulse.
    """
    pos = np.atleast_2d(np.asarray(positions_um, dtype=float))
    model = RydbergQuenchModel(pos, ramp_time=ramp_us)
    wf = model.waveform(model.theta(omega, delta), duration_us)
    filling = [1] * len(pos) if filling is None else [int(f) for f in filling]

    def series(field: str, unit: str, value_decimals: int) -> Dict[str, object]:
        ts = wf[field]
        return {"time_series": {"values": [_si(v, unit, value_decimals, f"{field} value") for v in ts["values"]],
                                "times": [_si(t, "us", 3, f"{field} time") for t in ts["times"]]},
                "pattern": "uniform"}

    return {
        "braketSchemaHeader": {"name": "braket.ir.ahs.program", "version": "1"},
        "setup": {"ahs_register": {"sites": [[_si(x, "um", 2, "x"), _si(y, "um", 2, "y")] for x, y in pos[:, :2]],
                                   "filling": filling}},
        "hamiltonian": {"drivingFields": [{"amplitude": series("amplitude", "rad/us", 4),
                                           "phase": series("phase", "rad", 7),
                                           "detuning": series("detuning", "rad/us", 6)}],
                        "localDetuning": []},
    }


def reference_programs(times: Sequence[float] = PROBE_TIMES_US) -> List[Dict[str, object]]:
    pos = chain(N_ATOMS, SPACING_UM)
    return [program_ir(pos, OMEGA, DELTA, t, RAMP_US) for t in times]


def program_summary(ir: Dict[str, object]) -> Dict[str, object]:
    """Positions (um), filling, Omega and Delta (rad/us), duration and ramp (us) read back from an IR dict."""
    reg = ir["setup"]["ahs_register"]
    drive = ir["hamiltonian"]["drivingFields"][0]
    amp = drive["amplitude"]["time_series"]
    det = drive["detuning"]["time_series"]
    a_t = [float(Decimal(str(t))) * 1e6 for t in amp["times"]]
    a_v = [float(Decimal(str(v))) * 1e-6 for v in amp["values"]]
    d_v = [float(Decimal(str(v))) * 1e-6 for v in det["values"]]
    return {"positions_um": [[float(Decimal(str(c))) * 1e6 for c in s] for s in reg["sites"]],
            "filling": [int(f) for f in reg["filling"]],
            "omega": max(a_v), "delta": d_v[0], "detuning_constant": len(set(d_v)) == 1,
            "duration_us": round(a_t[-1], 9), "ramp_us": round(a_t[1] - a_t[0], 9)}


# --------------------------------------------------------------------------- #
# checks against the documented device capabilities, without the SDK
# --------------------------------------------------------------------------- #
def load_paradigm(path: Path = PARADIGM_FILE) -> Dict[str, object]:
    with open(path) as fh:
        return json.load(fh)["paradigm"]


def _D(x) -> Decimal:
    return Decimal(str(x))


def _on_grid(x: Decimal, res: Decimal) -> bool:
    return (x / res) == (x / res).to_integral_value()


def capability_violations(ir: Dict[str, object], paradigm: Dict[str, object]) -> List[str]:
    """
    Rules from the QuEra paradigm properties that ``ir`` breaks, in exact decimal
    arithmetic: number of sites, field of view, radial and row spacing, the position
    grid; for each driving field the value ranges and resolutions, the time grid, the
    minimum time step, the maximum duration, the slew rates, Omega starting and ending
    at 0, the phase starting at 0, a uniform pattern, and a common end time; no local
    detuning. ``paradigm`` may be the documented dict or the SDK object's ``.dict()``
    from a live device. An empty list means no rule here fails.
    """
    out: List[str] = []
    lat, ryd = paradigm["lattice"], paradigm["rydberg"]["rydbergGlobal"]
    geo, area = lat["geometry"], lat["area"]
    reg = ir["setup"]["ahs_register"]
    sites = [(_D(x), _D(y)) for x, y in reg["sites"]]
    filling = list(reg["filling"])
    if len(filling) != len(sites):
        out.append("filling and sites differ in length")
    if len(sites) > int(geo["numberSitesMax"]):
        out.append(f"{len(sites)} sites, more than {geo['numberSitesMax']}")
    res = _D(geo["positionResolution"])
    for i, (x, y) in enumerate(sites):
        if not (_on_grid(x, res) and _on_grid(y, res)):
            out.append(f"site {i} ({x}, {y}) is off the {res} m position grid")
    rmin = _D(geo["spacingRadialMin"])
    for i in range(len(sites)):
        for j in range(i + 1, len(sites)):
            d2 = (sites[i][0] - sites[j][0]) ** 2 + (sites[i][1] - sites[j][1]) ** 2
            if d2 < rmin * rmin:
                out.append(f"sites {i} and {j} are closer than {rmin} m")
    if sites:
        xs, ys = [s[0] for s in sites], [s[1] for s in sites]
        if max(xs) - min(xs) > _D(area["width"]):
            out.append(f"pattern width {max(xs) - min(xs)} m exceeds {area['width']} m")
        if max(ys) - min(ys) > _D(area["height"]):
            out.append(f"pattern height {max(ys) - min(ys)} m exceeds {area['height']} m")
        rows = sorted(set(ys))
        vmin = _D(geo["spacingVerticalMin"])
        for a, b in zip(rows, rows[1:]):
            if b - a < vmin:
                out.append(f"rows at y = {a} and {b} m are closer than {vmin} m")

    ham = ir["hamiltonian"]
    if ham.get("localDetuning"):
        out.append("the program uses local detuning, which this experiment does not need and which needs separate access")
    fields = ham["drivingFields"]
    if len(fields) != 1:
        out.append(f"{len(fields)} driving fields; Aquila takes exactly one")
    t_res, t_min_step = _D(ryd["timeResolution"]), _D(ryd["timeDeltaMin"])
    t_max, t_min = _D(ryd["timeMax"]), _D(ryd["timeMin"])
    spec = {"amplitude": (ryd["rabiFrequencyRange"], ryd["rabiFrequencyResolution"], ryd["rabiFrequencySlewRateMax"]),
            "detuning": (ryd["detuningRange"], ryd["detuningResolution"], ryd["detuningSlewRateMax"]),
            "phase": (ryd["phaseRange"], ryd["phaseResolution"], None)}
    end_times = set()
    for f in fields:
        for name, (rng, vres, slew) in spec.items():
            pf = f[name]
            if pf.get("pattern") != "uniform":
                out.append(f"{name}: pattern must be 'uniform'")
            ts = pf["time_series"]
            t = [_D(x) for x in ts["times"]]
            v = [_D(x) for x in ts["values"]]
            if len(t) != len(v) or len(t) < 2:
                out.append(f"{name}: times and values must have the same length, at least 2")
                continue
            if t[0] != 0:
                out.append(f"{name}: first time point must be 0")
            end_times.add(t[-1])
            if t[-1] > t_max or t[-1] < t_min:
                out.append(f"{name}: duration {t[-1]} s outside [{t_min}, {t_max}] s")
            for a, b in zip(t, t[1:]):
                if b - a < t_min_step:
                    out.append(f"{name}: time step {b - a} s is shorter than {t_min_step} s")
            for x in t:
                if not _on_grid(x, t_res):
                    out.append(f"{name}: time {x} s is off the {t_res} s grid")
            lo, hi = _D(rng[0]), _D(rng[1])
            vres = _D(vres)
            for x in v:
                if x < lo or x > hi:
                    out.append(f"{name}: value {x} outside [{lo}, {hi}]")
                if not _on_grid(x, vres):
                    out.append(f"{name}: value {x} is off the {vres} resolution")
            if slew is not None:
                smax = _D(slew)
                for (ta, va), (tb, vb) in zip(zip(t, v), zip(t[1:], v[1:])):
                    if tb > ta and abs(vb - va) / (tb - ta) > smax:
                        out.append(f"{name}: slew rate {abs(vb - va) / (tb - ta):.4E} exceeds {smax:.4E}")
            if name == "amplitude" and (v[0] != 0 or v[-1] != 0):
                out.append("amplitude must start and end at 0")
            if name == "phase" and v[0] != 0:
                out.append("phase must start at 0")
    if len(end_times) > 1:
        out.append(f"time series end at different times: {sorted(end_times)}")
    return out


def package_violations(ir: Dict[str, object]) -> Dict[str, List[str]]:
    """The package's own geometry and waveform checks, with the 2023 datasheet and the current Braket figures."""
    s = program_summary(ir)
    pos = np.array(s["positions_um"])[np.array(s["filling"]) == 1]
    model = RydbergQuenchModel(pos, ramp_time=s["ramp_us"])
    th = model.theta(s["omega"], s["delta"])
    out = {}
    for label, specs in (("datasheet 2023", AQUILA), ("braket", AQUILA_BRAKET)):
        v = geometry_violations(np.array(s["positions_um"]), specs) + model.waveform_violations(th, [s["duration_us"]], specs)
        if not s["detuning_constant"]:
            v.append("detuning is not constant, which the fitted model assumes")
        out[label] = v
    return out


# --------------------------------------------------------------------------- #
# checks with the SDK (optional)
# --------------------------------------------------------------------------- #
def braket_available() -> bool:
    try:
        import braket.ahs  # noqa: F401
        return True
    except ImportError:
        return False


def sdk_program(ir: Dict[str, object]):
    """The SDK's AnalogHamiltonianSimulation for an IR dict (needs amazon-braket-sdk)."""
    from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
    from braket.ir.ahs.program_v1 import Program
    return AnalogHamiltonianSimulation.from_ir(Program.parse_obj(ir))


def sdk_native_program(positions_um, omega: float, delta: float, duration_us: float, ramp_us: float = RAMP_US):
    """The same pulse built with the SDK's own constructors, as an independent cross-check of :func:`program_ir`."""
    from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
    from braket.ahs.atom_arrangement import AtomArrangement
    from braket.ahs.driving_field import DrivingField
    from braket.timings.time_series import TimeSeries
    us, mhz = Decimal("1e-6"), Decimal("1e6")
    reg = AtomArrangement()
    for x, y in np.atleast_2d(positions_um)[:, :2]:
        reg.add((Decimal(repr(round(float(x), 2))) * us, Decimal(repr(round(float(y), 2))) * us))
    T, r = Decimal(repr(round(duration_us, 3))) * us, Decimal(repr(round(ramp_us, 3))) * us
    om = Decimal(repr(float(omega))) * mhz
    amp = TimeSeries()
    for t, v in ((0, 0), (r, om), (T - r, om), (T, 0)):
        amp.put(t, v)
    det = TimeSeries().put(0, Decimal(repr(float(delta))) * mhz).put(T, Decimal(repr(float(delta))) * mhz)
    phase = TimeSeries().put(0, 0).put(T, 0)
    return AnalogHamiltonianSimulation(register=reg, hamiltonian=DrivingField(amplitude=amp, detuning=det, phase=phase))


def _capabilities_constants(paradigm: Dict[str, object]):
    """Aquila's limits in the form the SDK's AHS program validator takes."""
    from braket.analog_hamiltonian_simulator.rydberg.validators.capabilities_constants import CapabilitiesConstants
    ryd, lat = paradigm["rydberg"], paradigm["lattice"]
    g = ryd["rydbergGlobal"]
    loc = ryd.get("rydbergLocal") or {"detuningRange": ["-125000000.0", "0.0"], "siteCoefficientRange": ["0", "1"]}
    return CapabilitiesConstants(
        BOUNDING_BOX_SIZE_X=_D(lat["area"]["width"]), BOUNDING_BOX_SIZE_Y=_D(lat["area"]["height"]),
        MAX_TIME=_D(g["timeMax"]), MIN_DISTANCE=_D(lat["geometry"]["spacingRadialMin"]),
        GLOBAL_AMPLITUDE_VALUE_MIN=_D(g["rabiFrequencyRange"][0]), GLOBAL_AMPLITUDE_VALUE_MAX=_D(g["rabiFrequencyRange"][1]),
        GLOBAL_DETUNING_VALUE_MIN=_D(g["detuningRange"][0]), GLOBAL_DETUNING_VALUE_MAX=_D(g["detuningRange"][1]),
        LOCAL_MAGNITUDE_SEQUENCE_VALUE_MIN=_D(loc["detuningRange"][0]), LOCAL_MAGNITUDE_SEQUENCE_VALUE_MAX=_D(loc["detuningRange"][1]),
        MAGNITUDE_PATTERN_VALUE_MIN=_D(loc["siteCoefficientRange"][0]), MAGNITUDE_PATTERN_VALUE_MAX=_D(loc["siteCoefficientRange"][1]),
        MAX_NET_DETUNING=_D(g["detuningRange"][1]) * 2,
    )


def sdk_violations(ir: Dict[str, object], paradigm: Dict[str, object]) -> List[str]:
    """
    Checks with amazon-braket-sdk: the IR parses as a ``braket.ir.ahs.program``; it
    equals the program built with the SDK's own constructors; the SDK's AHS program
    validator accepts it with Aquila's limits; and discretising it to the device's
    resolutions (``AnalogHamiltonianSimulation.discretize``'s rounding, applied with
    the documented paradigm properties) leaves it unchanged.
    """
    from braket.ahs.discretization_types import DiscretizationProperties
    from braket.analog_hamiltonian_simulator.rydberg.validators.ir_validator import validate_program
    from braket.device_schema.quera.quera_ahs_paradigm_properties_v1 import QueraAhsParadigmProperties
    out: List[str] = []
    try:
        prog = sdk_program(ir)
    except Exception as exc:                          # pydantic validation error
        return [f"IR does not parse as braket.ir.ahs.program: {exc}"]
    s = program_summary(ir)
    native = sdk_native_program(np.array(s["positions_um"]), s["omega"], s["delta"], s["duration_us"], s["ramp_us"])
    if native.to_ir() != prog.to_ir():
        out.append("the IR differs from the program built with the SDK's constructors")
    with warnings.catch_warnings(record=True) as caught:      # the validator warns, rather than fails, on some limits
        warnings.simplefilter("always")
        try:
            validate_program(prog.to_ir(), _capabilities_constants(paradigm))
        except Exception as exc:
            out.append(f"SDK program validator: {exc}")
    out += [f"SDK program validator warning: {w.message}" for w in caught]
    props = QueraAhsParadigmProperties.parse_obj(paradigm)
    disc = DiscretizationProperties(props.lattice, props.rydberg)
    d_prog = type(prog)(register=prog.register.discretize(disc), hamiltonian=prog.hamiltonian.discretize(disc))
    if d_prog.to_ir() != prog.to_ir():
        out.append("discretisation to the device resolution changes the program")
    return out


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def write_json(path: Path, obj) -> None:
    """JSON with two-space indents and short lists of numbers or strings kept on one line."""
    text = json.dumps(obj, indent=2)
    text = re.sub(r"\[\s*((?:(?:\"[^\"\n]*\"|-?[\d.eE+-]+|true|false|null)\s*,\s*)*(?:\"[^\"\n]*\"|-?[\d.eE+-]+|true|false|null))\s*\]",
                  lambda m: "[" + re.sub(r"\s*,\s*", ", ", m.group(1)) + "]", text)
    with open(path, "w") as fh:
        fh.write(text + "\n")


def program_filename(k: int, t: float) -> str:
    return f"program_{k:02d}_T{t:.3f}us.json"


def build(outdir: Path = PROGRAMS_DIR, times: Sequence[float] = PROBE_TIMES_US, shots: Optional[int] = None,
          paradigm_file: Path = PARADIGM_FILE, verbose: bool = True) -> Dict[str, object]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paradigm = load_paradigm(paradigm_file)
    model = reference_model()
    shot_choice = choose_shots(model, times=times)
    shots = int(shot_choice["shots_per_program"] if shots is None else shots)
    lo, hi = 1, 1000                                     # Aquila shotsRange in the Braket device documentation
    if not lo <= shots <= hi:
        raise ValueError(f"{shots} shots per task is outside Aquila's range [{lo}, {hi}]")
    use_sdk = braket_available()
    entries, failures = [], []
    for k, (t, ir) in enumerate(zip(times, reference_programs(times))):
        checks = {"package": package_violations(ir), "documented_capabilities": capability_violations(ir, paradigm)}
        if use_sdk:
            checks["sdk"] = sdk_violations(ir, paradigm)
        bad = [f"{group}/{sub}: {m}" if isinstance(v, dict) else f"{group}: {m}"
               for group, v in checks.items()
               for sub, msgs in (v.items() if isinstance(v, dict) else [("", v)]) for m in msgs]
        failures += [f"T = {t} us: {m}" for m in bad]
        name = program_filename(k, t)
        write_json(outdir / name, ir)
        entries.append({"file": name, "probe_time_us": float(t), "shots": shots, "checks": checks})
    cost = cost_estimate(len(entries), shots)
    manifest = {
        "experiment": {"atoms": N_ATOMS, "geometry": "chain", "spacing_um": SPACING_UM, "omega_rad_per_us": OMEGA,
                       "delta_rad_per_us": DELTA, "c6_reference_rad_um6_per_us": C6_RB87, "ramp_us": RAMP_US,
                       "probe_times_us": list(times), "fit": list(FIT_PARAMS)},
        "device_arn": DEVICE_ARN, "region": REGION,
        "shot_choice": shot_choice, "cost_estimate": cost,
        "checked_with_sdk": use_sdk, "paradigm_file": paradigm_file.name,
        "programs": entries, "all_checks_passed": not failures,
    }
    write_json(outdir / "manifest.json", manifest)
    if verbose:
        print(format_build_summary(manifest, failures))
    return manifest


def format_build_summary(manifest: Dict[str, object], failures: Sequence[str] = ()) -> str:
    e, sc, c = manifest["experiment"], manifest["shot_choice"], manifest["cost_estimate"]
    lines = [
        f"{e['atoms']}-atom chain at {e['spacing_um']} um, Omega = {e['omega_rad_per_us']} rad/us, "
        f"Delta = {e['delta_rad_per_us']} rad/us, {e['ramp_us']} us ramps",
        f"{len(manifest['programs'])} programs, durations {', '.join(f'{t:g}' for t in e['probe_times_us'])} us",
        "",
        f"Shots per program: {sc['shots_per_program']}",
        f"  Cramer-Rao goal: {100 * sc['target_relative_std_c6']:g} % on C6 with Omega and Delta fitted",
        f"  needed per probe time, most pessimistic model, before postselection: {sc['shots_per_time_needed_before_postselection']:.0f}"
        f" (expected kept fraction {sc['expected_kept_fraction']:.3f}); floor {sc['floor']}, rounded to {sc['step']}",
        "  predicted relative standard deviation at this count (Omega, Delta, C6):",
    ]
    for name, m in sc["models"].items():
        p = m["predicted_relative_std_at_chosen_shots"]
        lines.append(f"    {name}: {100 * p['omega']:.2f} %, {100 * p['delta']:.2f} %, {100 * p['c6']:.2f} %")
    lines += [
        "",
        f"Cost: {c['tasks']} tasks x ${c['price_per_task_usd']:.2f} + {c['total_shots']} shots x ${c['price_per_shot_usd']:.2f}"
        f" = ${c['total_usd']:.2f} (pricing from {c['pricing_source']}, read {c['pricing_checked']})",
        "",
        "Checks: package rules (datasheet and Braket figures), documented Aquila capabilities"
        + (", SDK schema, validator and discretisation" if manifest["checked_with_sdk"] else
           "; amazon-braket-sdk not installed, so the SDK checks were skipped"),
    ]
    lines += ["  all passed"] if not failures else ["  FAILED:"] + [f"    {m}" for m in failures]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=PROGRAMS_DIR, help="directory for the program JSON files and manifest")
    ap.add_argument("--shots", type=int, default=None, help="override the Cramer-Rao shot count per program")
    args = ap.parse_args(argv)
    manifest = build(args.out, shots=args.shots)
    return 0 if manifest["all_checks_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
