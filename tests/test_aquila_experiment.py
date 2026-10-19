"""
Tests for the Aquila experiment scripts in experiments/aquila: program building and
checks, shot count and cost, availability windows, result conversion and the fit.
The tests that need amazon-braket-sdk are skipped when it is not installed.
"""

import copy
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pytest

AQUILA_DIR = Path(__file__).resolve().parents[1] / "experiments" / "aquila"
sys.path.insert(0, str(AQUILA_DIR))
import analyze  # noqa: E402
import build    # noqa: E402
import submit   # noqa: E402

from shadow_learning.rydberg import AQUILA_BRAKET, C6_RB87, RydbergQuenchModel, chain  # noqa: E402

assert Path(build.__file__).parent == AQUILA_DIR

PARADIGM = build.load_paradigm()


def _set(ir, path, value):
    ir = copy.deepcopy(ir)
    node = ir
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return ir


AMP = ("hamiltonian", "drivingFields", 0, "amplitude", "time_series")


# --------------------------------------------------------------------------- #
# programs and checks
# --------------------------------------------------------------------------- #
def test_reference_programs_pass_every_check():
    programs = build.reference_programs()
    assert len(programs) == 8
    for t, ir in zip(build.PROBE_TIMES_US, programs):
        assert build.capability_violations(ir, PARADIGM) == []
        assert build.package_violations(ir) == {"datasheet 2023": [], "braket": []}
        s = build.program_summary(ir)
        assert s["omega"] == pytest.approx(12.0) and s["delta"] == pytest.approx(5.0)
        assert s["duration_us"] == pytest.approx(t) and s["ramp_us"] == pytest.approx(0.05)
        assert np.allclose(s["positions_um"], chain(6, 8.0))
        amp = ir["hamiltonian"]["drivingFields"][0]["amplitude"]["time_series"]
        assert amp["values"] == ["0", "12000000", "12000000", "0"]
        assert amp["times"][1] == "0.00000005"


def test_committed_programs_match_the_builder():
    manifest, programs = submit.load_plan(build.PROGRAMS_DIR)
    assert programs == build.reference_programs()
    assert manifest["shot_choice"]["shots_per_program"] == build.choose_shots()["shots_per_program"]


@pytest.mark.parametrize("path, value, expected", [
    (AMP + ("values",), ["0", "16000000", "16000000", "0"], "outside"),
    (AMP, {"values": ["0", "15000000", "15000000", "0"], "times": ["0", "0.00000003", "0.00000397", "0.000004"]},
     "slew rate"),
    (AMP + ("values",), ["100", "12000000", "12000000", "0"], "start and end at 0"),
    (AMP + ("values",), ["0", "12000100", "12000100", "0"], "resolution"),
    (AMP + ("times",), ["0", "0.0000000505", "0.00000395", "0.000004"], "grid"),
    (AMP + ("times",), ["0", "0.00000003", "0.00000395", "0.000004"], "shorter than"),
    (("hamiltonian", "drivingFields", 0, "phase", "time_series", "values"), ["0.5", "0"], "phase must start at 0"),
    (("hamiltonian", "drivingFields", 0, "detuning", "time_series", "values"), ["130000000", "130000000"], "outside"),
    (("hamiltonian", "drivingFields", 0, "detuning", "time_series", "times"), ["0", "0.0000039"], "different times"),
    (("setup", "ahs_register", "sites", 1), ["0.0000035", "0"], "closer than"),
    (("setup", "ahs_register", "sites", 1), ["0.000008005", "0"], "position grid"),
    (("setup", "ahs_register", "sites", 5), ["0.00008", "0"], "width"),
    (("setup", "ahs_register", "sites", 5), ["0.00004", "0.000002"], "rows"),
])
def test_capability_checks_catch_violations(path, value, expected):
    ir = _set(build.reference_programs()[-1], path, value)
    problems = build.capability_violations(ir, PARADIGM)
    assert any(expected in p for p in problems), problems


def test_too_long_and_too_many_sites():
    long = build.program_ir(chain(6, 8.0), 12.0, 5.0, 4.5)
    assert any("duration" in p for p in build.capability_violations(long, PARADIGM))
    assert any("exceeds 4.0" in p for p in build.package_violations(long)["braket"])
    many = build.program_ir(chain(2, 8.0), 12.0, 5.0, 1.0)
    many["setup"]["ahs_register"]["sites"] = [[str(i * 4e-6 % 7.2e-5), str(4e-6 * (i // 18))] for i in range(257)]
    many["setup"]["ahs_register"]["filling"] = [1] * 257
    assert any("more than 256" in p for p in build.capability_violations(many, PARADIGM))


def test_off_grid_inputs_raise_instead_of_rounding():
    with pytest.raises(ValueError):
        build.program_ir(chain(6, 8.0), 12.0, 5.0, 1.0005)
    with pytest.raises(ValueError):
        build.program_ir(chain(6, 8.0005), 12.0, 5.0, 1.0)


# --------------------------------------------------------------------------- #
# shots and cost
# --------------------------------------------------------------------------- #
def test_shot_choice_meets_the_goal():
    sc = build.choose_shots()
    n = sc["shots_per_program"]
    assert n % build.SHOT_STEP == 0 and n >= build.MIN_SHOTS_PER_PROGRAM
    for m in sc["models"].values():
        assert m["predicted_relative_std_at_chosen_shots"]["c6"] <= build.TARGET_REL_C6 + 1e-12
    raw = build.choose_shots(floor=0, step=1)
    assert raw["shots_per_program"] == int(np.ceil(raw["shots_per_time_needed_before_postselection"]))
    # the pessimistic model needs more than the closed one, and the bound scales as 1/sqrt(shots)
    need = [m["shots_per_time_for_target_c6"] for m in sc["models"].values()]
    assert need[-1] == max(need)
    half = build.choose_shots(target=0.5 * build.TARGET_REL_C6, floor=0, step=1)
    assert half["shots_per_time_needed_before_postselection"] == pytest.approx(4 * raw["shots_per_time_needed_before_postselection"])


def test_cost_estimate():
    c = build.cost_estimate(8, 200)
    assert c["total_shots"] == 1600
    assert c["total_usd"] == pytest.approx(8 * 0.30 + 1600 * 0.01)
    assert c["total_usd"] == pytest.approx(18.40)
    manifest = {"programs": [{"shots": 200}] * 8}
    assert submit.plan_cost(manifest)["total_usd"] == pytest.approx(18.40)


# --------------------------------------------------------------------------- #
# availability windows
# --------------------------------------------------------------------------- #
def test_window_status():
    t = dt.time
    windows = [("Monday", t(1, 0), t(23, 59, 59)), ("Tuesday", t(0, 0), t(12, 0)), ("Saturday", t(22, 0), t(2, 0))]
    utc = dt.timezone.utc
    mon_noon = dt.datetime(2026, 10, 19, 12, 0, tzinfo=utc)             # a Monday
    inside, end, nxt = submit.window_status(windows, mon_noon)
    assert inside and end == dt.datetime(2026, 10, 19, 23, 59, 59, tzinfo=utc)
    tue_eve = dt.datetime(2026, 10, 20, 15, 0, tzinfo=utc)
    inside, end, nxt = submit.window_status(windows, tue_eve)
    assert not inside and nxt == dt.datetime(2026, 10, 24, 22, 0, tzinfo=utc)
    sun_1am = dt.datetime(2026, 10, 25, 1, 0, tzinfo=utc)               # inside Saturday's window past midnight
    inside, end, nxt = submit.window_status(windows, sun_1am)
    assert inside and end == dt.datetime(2026, 10, 25, 2, 0, tzinfo=utc)
    assert submit.window_status([("Everyday", t(9), t(10))], mon_noon)[2] == dt.datetime(2026, 10, 20, 9, 0, tzinfo=utc)
    assert submit.window_status([("Weekdays", t(0), t(23, 59))], mon_noon)[0]
    assert not submit.window_status([("Weekend", t(0), t(23, 59))], mon_noon)[0]


# --------------------------------------------------------------------------- #
# results to data
# --------------------------------------------------------------------------- #
class _Shot:
    def __init__(self, status, pre, post):
        self.status, self.pre_sequence, self.post_sequence = status, pre, post


class _Result:
    def __init__(self, shots):
        self.measurements = shots
        self.task_metadata = None


def test_records_to_data_postselects_and_orders():
    sites = chain(4, 8.0)
    filling = [1, 1, 0, 1]
    shots = [_Shot("Success", [1, 1, 0, 1], [1, 0, 1, 1]),     # kept: bits on the filled sites are 1 - [1, 0, 1]
             _Shot("Success", [1, 1, 1, 1], [1, 1, 1, 1]),     # stray atom in the empty site
             _Shot("Success", [1, 0, 0, 1], [1, 0, 0, 1]),     # missing atom
             _Shot("Failure", None, None),
             _Shot("Partial Success", [1, 1, 0, 1], [1, 1, 0, 1])]
    recs = []
    for t in (1.0, 0.5):                                       # out of order on purpose
        ir = build.program_ir(sites, 12.0, 5.0, t, filling=filling)
        recs.append(analyze.result_record(_Result(shots), t, ir, f"p{t}.json"))
    recs = json.loads(json.dumps(recs))                        # the records must survive a JSON round trip
    model, theta0, data, info = analyze.records_to_data(recs)
    assert model.n == 3 and np.allclose(model.positions, sites[[0, 1, 3]])
    assert np.allclose(theta0, [12.0, 5.0, C6_RB87])
    assert list(data.times) == [0.5, 1.0]
    for b in data.bits:
        assert b.tolist() == [[0, 1, 0]]
    assert info["shots_returned"] == [5, 5] and info["shots_success"] == [3, 3] and info["shots_kept"] == [1, 1]
    bad = copy.deepcopy(recs)
    bad[0]["probe_time_us"] = 2.0
    with pytest.raises(ValueError):
        analyze.records_to_data(bad)


def test_save_and_load_records(tmp_path):
    ir = build.program_ir(chain(2, 8.0), 12.0, 5.0, 1.0)
    rec = analyze.result_record(_Result([_Shot("Success", [1, 1], [0, 1])]), 1.0, ir, "x.json", task_id="arn:test")
    analyze.save_records([rec], tmp_path)
    assert analyze.load_records(tmp_path) == [json.loads(json.dumps(rec))]


def test_readout_models():
    assert analyze.readout_model("none") is None
    ro = analyze.readout_model("auto", PARADIGM)
    assert (ro.p01, ro.p10) == pytest.approx((0.05, 0.10))
    ro = analyze.readout_model("auto", None)
    assert (ro.p01, ro.p10) == pytest.approx((AQUILA_BRAKET.p_ground_as_rydberg, AQUILA_BRAKET.p_rydberg_as_ground))
    assert analyze.readout_model("paper").p10 == pytest.approx(0.08)


def _sampled_records(shots, seed, readout=None, p_vacancy=0.0):
    """Package-simulated shots in the Braket result layout (post = 1 - bits, pre = filling image)."""
    model = build.reference_model()
    data = model.sample(build.nominal_theta(model), build.PROBE_TIMES_US, shots, np.random.default_rng(seed),
                        readout=readout, p_vacancy=p_vacancy)
    recs = []
    for k, (t, ir) in enumerate(zip(build.PROBE_TIMES_US, build.reference_programs())):
        pre = data.filled[k] if data.filled is not None else np.ones_like(data.bits[k])
        res = _Result([_Shot("Success", p.tolist(), (1 - b).tolist()) for p, b in zip(pre, data.bits[k])])
        recs.append(analyze.result_record(res, t, ir, f"p{k}.json"))
    return recs


@pytest.mark.slow
def test_analysis_recovers_parameters_with_readout_errors():
    recs = _sampled_records(200, seed=11, readout=AQUILA_BRAKET.readout_model(), p_vacancy=0.01)
    report, text = analyze.run_analysis(recs, readout_name="braket", noise_name="none", n_boot=20,
                                        truth={"omega": 12.0, "delta": 5.0, "c6": C6_RB87})
    assert report["fit_success"], report["warnings"]
    assert report["goodness_of_fit"]["p_value"] > 1e-3
    assert all(n < 200 for n in report["data"]["shots_kept"])          # empty sites were dropped
    for nm in ("omega", "delta", "c6"):
        p = report["parameters"][nm]
        assert abs(p["z"]) < 4
        assert 0.5 < p["bootstrap_std"] / p["cramer_rao_std"] < 2.0
    assert "goodness of fit" in text and "C6 against" in text


# --------------------------------------------------------------------------- #
# with amazon-braket-sdk
# --------------------------------------------------------------------------- #
def test_sdk_checks_accept_the_programs_and_catch_errors():
    pytest.importorskip("braket.ahs")
    for ir in build.reference_programs():
        assert build.sdk_violations(ir, PARADIGM) == []
    off = _set(build.reference_programs()[-1], AMP + ("times",), ["0", "0.0000000505", "0.00000395", "0.000004"])
    assert any("discretisation" in p for p in build.sdk_violations(off, PARADIGM))
    big = _set(build.reference_programs()[-1], AMP + ("values",), ["0", "25000000", "25000000", "0"])
    assert any("validator" in p for p in build.sdk_violations(big, PARADIGM))


def test_submit_with_a_stand_in_device(tmp_path):
    pytest.importorskip("braket.ahs")
    from types import SimpleNamespace
    from braket.device_schema import DeviceActionType
    from braket.device_schema.quera.quera_ahs_paradigm_properties_v1 import QueraAhsParadigmProperties

    created = []

    class Device:
        status = "ONLINE"
        is_available = True
        properties = SimpleNamespace(
            action={DeviceActionType.AHS: SimpleNamespace(actionType=DeviceActionType.AHS)},
            paradigm=QueraAhsParadigmProperties.parse_obj(PARADIGM),
            service=SimpleNamespace(shotsRange=(1, 1000), deviceCost=SimpleNamespace(price=0.01, unit="shot"),
                                    executionWindows=[]))

        def run(self, program, shots):
            created.append((program, shots))
            return SimpleNamespace(id=f"arn:aws:braket:us-east-1:000000000000:quantum-task/test-{len(created)}")

    manifest, programs = submit.load_plan(build.PROGRAMS_DIR)
    problems, cost = submit.device_checks(Device(), programs, manifest)
    assert problems == [] and cost == {"price": 0.01, "unit": "shot"}
    out = tmp_path / "tasks.json"
    rec = submit.submit(Device(), programs, manifest, out, {"region": "us-east-1", "programs_dir": build.PROGRAMS_DIR,
                                                            "cost": submit.plan_cost(manifest), "status": "ONLINE",
                                                            "paradigm": PARADIGM})
    saved = json.loads(out.read_text())
    assert saved == json.loads(json.dumps(rec))
    assert [t["probe_time_us"] for t in saved["tasks"]] == list(build.PROBE_TIMES_US)
    assert all(s == manifest["shot_choice"]["shots_per_program"] for _, s in created)
    assert (out.parent / saved["programs_dir"]).resolve() == build.PROGRAMS_DIR.resolve()
    assert created[0][0].to_ir() == build.sdk_program(programs[0]).to_ir()


def test_local_simulator_round_trip(tmp_path):
    pytest.importorskip("braket.devices")
    import dry_run
    build.build(tmp_path, times=(0.5, 4.0), shots=30, verbose=False)
    recs = dry_run.run_local(tmp_path, steps=200)
    assert [len(r["measurements"]) for r in recs] == [30, 30]
    model, theta0, data, info = analyze.records_to_data(recs)
    assert info["shots_kept"] == [30, 30]
    assert dry_run.simulator_c6() == pytest.approx(5.42e6)
    # the local simulator and the package model describe the same pulse: compare <n_i> at both times
    recs = dry_run.run_local(tmp_path, shots=2000, steps=400)
    _, _, data, _ = analyze.records_to_data(recs)
    p = model.probabilities(model.theta(12.0, 5.0, dry_run.simulator_c6()), data.times)
    n_model = p @ model.bits.astype(float)
    for k in range(len(data.times)):
        n_emp = data.bits[k].mean(axis=0)
        sigma = np.sqrt(n_model[k] * (1 - n_model[k]) / data.shots[k]) + 1e-3
        assert np.all(np.abs(n_emp - n_model[k]) < 5 * sigma), (n_emp, n_model[k])
