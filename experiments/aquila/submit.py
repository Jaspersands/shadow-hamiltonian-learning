"""
Submit the reference experiment to QuEra Aquila through Amazon Braket.

    python experiments/aquila/submit.py              # print the plan; nothing is sent
    python experiments/aquila/submit.py --check      # also query the device (read-only, free)
    python experiments/aquila/submit.py --confirm    # check, then submit the tasks

The script reads the programs and manifest written by build.py, checks them again,
and prints the device ARN, the number of tasks, the total shots and the cost estimate.
With --check or --confirm it then connects to Braket with your own AWS credentials
(set up with `aws configure`; this script never reads or stores them), checks that
Aquila is ONLINE and inside an availability window, and checks the programs against
the capabilities the device reports now. Only with --confirm are tasks created, and
then every task ARN is written to the tasks file as soon as it exists, together with
the device properties at submission (readout error rates included) for analyze.py.

Needs amazon-braket-sdk for --check and --confirm (the .venv-braket environment).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build  # noqa: E402

TASKS_FILE = HERE / "tasks.json"
DEFAULT_MAX_COST_USD = 25.0
DAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def load_plan(programs_dir: Path = build.PROGRAMS_DIR) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """The manifest and the program IR dicts, re-checked against the documented capabilities and the package rules."""
    programs_dir = Path(programs_dir)
    with open(programs_dir / "manifest.json") as fh:
        manifest = json.load(fh)
    paradigm = build.load_paradigm(HERE / manifest.get("paradigm_file", build.PARADIGM_FILE.name))
    programs, problems = [], []
    for entry in manifest["programs"]:
        with open(programs_dir / entry["file"]) as fh:
            ir = json.load(fh)
        s = build.program_summary(ir)
        if abs(s["duration_us"] - entry["probe_time_us"]) > 1e-9:
            problems.append(f"{entry['file']}: duration {s['duration_us']} us, manifest says {entry['probe_time_us']} us")
        problems += [f"{entry['file']}: {m}" for m in build.capability_violations(ir, paradigm)]
        problems += [f"{entry['file']} ({k}): {m}" for k, v in build.package_violations(ir).items() for m in v]
        if not 1 <= int(entry["shots"]) <= 1000:
            problems.append(f"{entry['file']}: {entry['shots']} shots is outside Aquila's 1 to 1000")
        programs.append(ir)
    if not manifest.get("all_checks_passed", False):
        problems.append("build.py reported failed checks; run it again and read its output")
    if problems:
        raise SystemExit("The programs do not pass the checks:\n  " + "\n  ".join(problems))
    return manifest, programs


def plan_cost(manifest: Dict[str, object], price_task: float = build.PRICE_PER_TASK_USD,
              price_shot: float = build.PRICE_PER_SHOT_USD) -> Dict[str, object]:
    shots = [int(e["shots"]) for e in manifest["programs"]]
    total = sum(shots)
    task_fees, shot_fees = len(shots) * price_task, total * price_shot
    return {"tasks": len(shots), "total_shots": total, "shots_per_task": sorted(set(shots)),
            "task_fees_usd": round(task_fees, 2), "shot_fees_usd": round(shot_fees, 2),
            "total_usd": round(task_fees + shot_fees, 2), "price_per_task_usd": price_task, "price_per_shot_usd": price_shot}


def format_plan(manifest: Dict[str, object], cost: Dict[str, object], region: str) -> str:
    per = ", ".join(map(str, cost["shots_per_task"]))
    return "\n".join([
        f"device:      {manifest['device_arn']}",
        f"region:      {region}",
        f"tasks:       {cost['tasks']} (one per probe time: "
        + ", ".join(f"{e['probe_time_us']:g}" for e in manifest["programs"]) + " us)",
        f"shots:       {per} per task, {cost['total_shots']} in total",
        f"cost:        {cost['tasks']} x ${cost['price_per_task_usd']:.2f} + {cost['total_shots']} x ${cost['price_per_shot_usd']:.2f}"
        f" = ${cost['total_usd']:.2f}",
        f"             (on-demand prices from {build.PRICING_SOURCE}, read {build.PRICING_CHECKED};"
        " plus a few cents of S3 storage for the results)",
    ])


# --------------------------------------------------------------------------- #
# availability windows
# --------------------------------------------------------------------------- #
def _day_matches(day: str, weekday: int) -> bool:
    day = day.strip().lower()
    if day == "everyday":
        return True
    if day == "weekdays":
        return weekday < 5
    if day in ("weekend", "weekends"):
        return weekday >= 5
    return DAY_NAMES[weekday].lower() == day


def window_status(windows: Sequence[Tuple[str, dt.time, dt.time]], now: dt.datetime) -> Tuple[bool, Optional[dt.datetime], Optional[dt.datetime]]:
    """
    Whether ``now`` (UTC) lies in one of the execution windows (day, start, end in
    UTC; a window whose end is before its start runs past midnight), the end of the
    current window, and the start of the next one. Mirrors AwsDevice.is_available,
    which is the check that decides; this one is for printing the next opening.
    """
    now = now.astimezone(dt.timezone.utc).replace(tzinfo=None)
    inside, end_now, starts = False, None, []
    for offset in range(-1, 8):
        day = (now + dt.timedelta(days=offset)).date()
        for name, t0, t1 in windows:
            if not _day_matches(name, day.weekday()):
                continue
            start = dt.datetime.combine(day, t0)
            end = dt.datetime.combine(day, t1)
            if end <= start:
                end += dt.timedelta(days=1)
            if start <= now < end:
                inside = True
                end_now = max(end_now, end) if end_now else end
            elif start > now:
                starts.append(start)
    nxt = min(starts) if starts else None
    utc = dt.timezone.utc
    return inside, (end_now.replace(tzinfo=utc) if end_now else None), (nxt.replace(tzinfo=utc) if nxt else None)


def device_windows(device) -> List[Tuple[str, dt.time, dt.time]]:
    out = []
    for w in device.properties.service.executionWindows:
        day = getattr(w.executionDay, "value", w.executionDay)
        out.append((str(day), w.windowStartHour, w.windowEndHour))
    return out


# --------------------------------------------------------------------------- #
# device checks and submission (need the SDK)
# --------------------------------------------------------------------------- #
def connect(arn: str, region: str):
    try:
        import boto3
        from braket.aws import AwsDevice, AwsSession
    except ImportError:
        raise SystemExit("amazon-braket-sdk is not installed here; use the .venv-braket environment (see README)")
    try:
        session = AwsSession(boto_session=boto3.Session(region_name=region))
        return AwsDevice(arn, aws_session=session)
    except Exception as exc:                      # NoCredentialsError, AccessDenied, region mismatch, ...
        raise SystemExit(f"Could not reach the device: {type(exc).__name__}: {exc}\n"
                         "Set up credentials with `aws configure` and make sure Braket is enabled in "
                         f"{region} and your account has access to QuEra Aquila.")


def device_paradigm(device) -> Dict[str, object]:
    """The device's paradigm properties as plain JSON (decimals as strings)."""
    return json.loads(json.dumps(device.properties.paradigm.dict(), default=str))


def device_checks(device, programs: Sequence[Dict[str, object]], manifest: Dict[str, object]) -> Tuple[List[str], Dict[str, object]]:
    """Check the programs against what the device reports now; returns problems and the device cost figure."""
    problems = []
    paradigm = device_paradigm(device)
    for entry, ir in zip(manifest["programs"], programs):
        problems += [f"{entry['file']}: {m}" for m in build.capability_violations(ir, paradigm)]
        prog = build.sdk_program(ir)
        try:
            if prog.discretize(device).to_ir() != prog.to_ir():
                problems.append(f"{entry['file']}: discretisation with the device's resolutions changes the program")
        except Exception as exc:
            problems.append(f"{entry['file']}: discretize failed: {exc}")
    service = device.properties.service
    lo, hi = service.shotsRange
    for entry in manifest["programs"]:
        if not lo <= int(entry["shots"]) <= hi:
            problems.append(f"{entry['file']}: {entry['shots']} shots outside the device's range [{lo}, {hi}]")
    cost = getattr(service, "deviceCost", None)
    dev_cost = {"price": float(cost.price), "unit": str(cost.unit)} if cost is not None else None
    return problems, dev_cost


def submit(device, programs: Sequence[Dict[str, object]], manifest: Dict[str, object], out: Path,
           extra: Dict[str, object]) -> Dict[str, object]:
    """Create one task per program, writing the tasks file after each so that no ARN is lost."""
    record = {"device_arn": manifest["device_arn"], "region": extra["region"],
              "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "programs_dir": os.path.relpath(Path(extra["programs_dir"]).resolve(), out.parent.resolve()),
              "cost_estimate": extra["cost"], "device_status": extra["status"],
              "paradigm_at_submission": extra["paradigm"], "tasks": []}
    for entry, ir in zip(manifest["programs"], programs):
        try:
            task = device.run(build.sdk_program(ir), shots=int(entry["shots"]))
        except Exception as exc:
            _write(out, record)
            raise SystemExit(f"Submission of {entry['file']} failed ({type(exc).__name__}: {exc}). "
                             f"{len(record['tasks'])} task(s) were created and are listed in {out}.")
        record["tasks"].append({"task_arn": task.id, "program_file": entry["file"],
                                "probe_time_us": entry["probe_time_us"], "shots": int(entry["shots"])})
        _write(out, record)
        print(f"  created {task.id}  (T = {entry['probe_time_us']:g} us, {entry['shots']} shots)")
    return record


def _write(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
        fh.write("\n")
    tmp.replace(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Submit the reference experiment to QuEra Aquila (see README).")
    ap.add_argument("--confirm", action="store_true", help="actually create the tasks (costs money)")
    ap.add_argument("--check", action="store_true", help="query the device and check the programs, without submitting")
    ap.add_argument("--programs", type=Path, default=build.PROGRAMS_DIR)
    ap.add_argument("--device-arn", default=build.DEVICE_ARN)
    ap.add_argument("--region", default=build.REGION)
    ap.add_argument("--out", type=Path, default=TASKS_FILE, help="where to save the task ARNs")
    ap.add_argument("--max-cost", type=float, default=DEFAULT_MAX_COST_USD,
                    help=f"refuse to submit if the estimate exceeds this many USD (default {DEFAULT_MAX_COST_USD:g})")
    ap.add_argument("--allow-outside-window", action="store_true",
                    help="submit even when Aquila is outside an availability window; the tasks then wait in the queue")
    args = ap.parse_args(argv)

    manifest, programs = load_plan(args.programs)
    manifest = dict(manifest, device_arn=args.device_arn)
    cost = plan_cost(manifest)
    print(format_plan(manifest, cost, args.region))
    print()
    if not (args.check or args.confirm):
        print("Nothing was submitted. Run with --check to query the device, or --confirm to submit.")
        return 0
    if args.confirm and args.out.exists():
        raise SystemExit(f"{args.out} already exists, so tasks may have been submitted before. "
                         "Move it away, or pass --out with a new file name.")

    device = connect(args.device_arn, args.region)
    status = device.status
    windows = device_windows(device)
    now = dt.datetime.now(dt.timezone.utc)
    inside, ends, nxt = window_status(windows, now)
    print(f"device status: {status}")
    print("availability windows (UTC): " + "; ".join(f"{d} {a.strftime('%H:%M')}-{b.strftime('%H:%M')}" for d, a, b in windows))
    available = bool(device.is_available)
    print(f"now {now.strftime('%a %Y-%m-%d %H:%M')} UTC: " + ("inside a window" + (f" until {ends:%a %H:%M} UTC" if ends else "")
          if available else "not available" + (f"; next window opens {nxt:%a %Y-%m-%d %H:%M} UTC" if nxt else "")))

    problems, dev_cost = device_checks(device, programs, manifest)
    if dev_cost:
        print(f"device-reported price: ${dev_cost['price']:g} per {dev_cost['unit']}")
        if dev_cost["unit"] == "shot" and abs(dev_cost["price"] - build.PRICE_PER_SHOT_USD) > 1e-9:
            alt = plan_cost(manifest, price_shot=dev_cost["price"])
            print(f"  this differs from the pricing page; with it the estimate is ${alt['total_usd']:.2f}")
            cost = alt if alt["total_usd"] > cost["total_usd"] else cost
    if problems:
        print("The programs fail the device's current capabilities:\n  " + "\n  ".join(problems))
        return 1
    print("programs pass the device's current capabilities")
    if not args.confirm:
        print("\nNothing was submitted (--check). Run with --confirm to submit.")
        return 0

    if status != "ONLINE":
        raise SystemExit(f"Aquila is {status}, not ONLINE. Nothing was submitted.")
    if not available and not args.allow_outside_window:
        raise SystemExit("Aquila is outside its availability windows. Nothing was submitted. Run again inside a window, "
                         "or pass --allow-outside-window to queue the tasks (they may then run hours apart).")
    if cost["total_usd"] > args.max_cost:
        raise SystemExit(f"The estimate ${cost['total_usd']:.2f} exceeds --max-cost ${args.max_cost:.2f}. Nothing was submitted.")

    print(f"\nSubmitting {cost['tasks']} tasks to {args.device_arn} ...")
    record = submit(device, programs, manifest, args.out,
                    {"region": args.region, "programs_dir": args.programs, "cost": cost, "status": status,
                     "paradigm": device_paradigm(device)})
    print(f"\n{len(record['tasks'])} task ARNs saved to {args.out}")
    print("When the tasks have completed:  python experiments/aquila/analyze.py --tasks "
          f"{args.out} --out experiments/aquila/aquila_fit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
