# Rydberg Hamiltonian learning on QuEra Aquila

This folder holds a ready-to-submit version of the reference experiment from the main README's section "Learning a Rydberg Hamiltonian from snapshots", for QuEra's Aquila through Amazon Braket, and a dry run of the whole pipeline on Braket's local simulator. Nothing has been submitted to AWS.

The experiment: six atoms in a chain at 8 μm, Ω = 12 rad/μs, Δ = 5 rad/μs, linear Ω ramps of 0.05 μs, and eight pulses with total durations 0.5, 1.0, …, 4.0 μs. Each pulse is one Braket task. All atoms are read out after the pulse, and Ω, Δ and C6 are fitted to the bitstrings by maximum likelihood.

| file | what it does |
|---|---|
| `build.py` | writes the eight programs as Braket IR JSON to `programs/`, checks them, chooses the shot count, writes `programs/manifest.json` |
| `submit.py` | prints the plan and cost; with `--check` queries the device; with `--confirm` submits and saves the task ARNs to `tasks.json` |
| `analyze.py` | fetches the results (or reads saved ones), fits, and reports Ω, Δ, C6 with errors and the goodness of fit |
| `dry_run.py` | runs the same programs on `LocalSimulator("braket_ahs")` and analyses them |
| `aquila_paradigm_documented.json` | Aquila's capabilities as documented by Braket, used by the offline checks |
| `dryrun/` | the dry-run results and summary |

## Shot count and cost

The shots per task come from the package's Cramér–Rao bound, with Ω, Δ and C6 all fitted and the shots split equally over the eight durations. The goal is a relative standard deviation of 1 % on C6. Three device models are considered, all with Braket's current readout errors (a ground atom read as Rydberg with probability 0.05, a Rydberg atom read as ground with probability 0.10):

| model | shots per task for 1 % on C6 | predicted at 200 shots per task: Ω, Δ, C6 |
|---|---|---|
| readout errors only | 26 | 0.22 %, 1.24 %, 0.37 % |
| plus shot-to-shot drive noise | 30 | 0.25 %, 1.35 %, 0.40 % |
| plus relaxation at T₂ echo (7 μs) | 154 | 0.56 %, 3.02 %, 0.90 % |

`build.py` takes the most pessimistic model, 154 shots, divides by the fraction of shots expected to survive postselection on a fully loaded array (0.992⁶ = 0.953, from Braket's typical filling error of 0.008 per site), which gives 162, raises this to a floor of 200, and rounds to a multiple of 50. The floor is there because the goodness-of-fit test needs a few hundred shots per duration, and the fit's search for the right likelihood maximum was validated at 200 shots per duration (main README). The result is 200 shots per task, 1,600 in total.

The cost is

    8 tasks × $0.30 + 1,600 shots × $0.01 = $2.40 + $16.00 = $18.40

using the on-demand prices for QuEra Aquila on the Amazon Braket pricing page (https://aws.amazon.com/braket/pricing/), read on 2026-10-19: $0.30 per task and $0.01 per shot. The page lists no free tier for QPUs. The results are written to an S3 bucket in your account, which costs a few cents at most. `submit.py --check` also prints the per-shot price the device itself reports, and uses it if it is higher.

More shots would lower the statistical errors further, but not the systematic ones below, which are larger for C6.

## What you need to do first

These steps need your own account and decisions; the scripts do not do them.

1. An AWS account. In the Amazon Braket console (https://console.aws.amazon.com/braket/), in the region us-east-1 (N. Virginia), where Aquila is, choose Get Started and follow the dashboard: create the service-linked role and enable access to third-party devices. Enabling third-party devices means accepting terms about data transfer to the device providers; that is done once per account.
2. A user or role with the `AmazonBraketFullAccess` IAM policy, or a narrower policy that allows Braket tasks and S3 writes to the results bucket. The SDK creates a bucket named `amazon-braket-us-east-1-<account id>` on first use.
3. Credentials on this machine: run `aws configure` and enter your access key, secret key and region `us-east-1`. The scripts never read, print or store credentials; boto3 finds them.
4. Aquila runs tasks only inside its availability windows, listed in UTC on the Devices page of the Braket console and in the device properties. They change from time to time; `submit.py --check` prints the current ones and whether the device is available now. A task submitted outside a window waits in the queue.

## Commands

From the repository root:

    python3 -m venv .venv-braket
    .venv-braket/bin/pip install -e ".[braket]"

    .venv-braket/bin/python experiments/aquila/build.py          # programs, checks, shots, cost
    .venv-braket/bin/python experiments/aquila/dry_run.py        # optional: local simulator, about 1 minute
    .venv-braket/bin/python experiments/aquila/submit.py         # prints the plan; sends nothing
    .venv-braket/bin/python experiments/aquila/submit.py --check     # queries the device; sends nothing
    .venv-braket/bin/python experiments/aquila/submit.py --confirm   # submits the 8 tasks

`submit.py --confirm` refuses to submit if Aquila is not ONLINE, if it is outside an availability window (override with `--allow-outside-window`), if the estimate exceeds `--max-cost` (25 USD by default), if the programs fail the capabilities the device reports at that moment, or if `tasks.json` already exists. Running all eight tasks inside one window matters: tasks that wait in the queue may run hours apart, under a different calibration.

To analyse once the tasks have completed:

    .venv-braket/bin/python experiments/aquila/analyze.py --tasks experiments/aquila/tasks.json \
        --compare-noise --out experiments/aquila/aquila_fit

This downloads the results to `results/` (one JSON file per task), so later runs can use `--results experiments/aquila/results` without contacting AWS. `--wait` blocks until every task has completed. The analysis itself needs only the package, not the SDK. It takes about one minute, plus about three for `--compare-noise`.

## What build.py checks

Every program is checked against:

- the package's rules, `geometry_violations` and `waveform_violations`, with both the 2023 datasheet figures (`AQUILA`) and the current Braket ones (`AQUILA_BRAKET`);
- the documented Aquila capabilities, in exact decimal arithmetic: at most 256 sites in a 75 × 76 μm field, 4 μm radial and row spacing, the 0.01 μm position grid; Ω between 0 and 15.8 rad/μs on a 400 rad/s grid, starting and ending at 0, with slew rate at most 400 rad/μs²; |Δ| ≤ 125 rad/μs on a 0.2 rad/s grid; phase starting at 0; times on the 1 ns grid, steps of at least 0.05 μs, durations up to 4 μs; uniform patterns, one driving field, no local detuning;
- the SDK, when installed: the JSON parses as a `braket.ir.ahs.program`, it equals the same pulse built with the SDK's own constructors, the SDK's AHS program validator accepts it with Aquila's limits (its warnings count as failures), and discretising it to the device resolutions leaves it unchanged.

The documented capabilities were copied from the device properties printed in Braket's example notebook `01_Introduction_to_Aquila` (amazon-braket-examples, main branch, read 2026-10-19). `submit.py` checks the programs again against the properties the device reports at submission, and saves those properties in `tasks.json`; `analyze.py` takes the readout error rates from there.

## Dry run on the local simulator

`dry_run.py` runs the programs written by `build.py`, with the same 200 shots each, on `LocalSimulator("braket_ahs")` (amazon-braket-sdk 1.127.3), saves the results in the same format as Aquila's, and runs the same analysis. The simulator was fast enough that nothing had to be reduced: 4 s for the eight tasks, 48 s for the analysis with 200 bootstrap resamples. The simulator's C6 is 5.42 × 10⁶ rad μm⁶/μs, 0.008 % below the package's value; the fit is compared with the simulator's.

| | estimate | Cramér–Rao | bootstrap | truth | error | error / σ |
|---|---|---|---|---|---|---|
| Ω (rad/μs) | 12.0057 | 0.0146 | 0.0141 | 12 | +0.05 % | +0.39 |
| Δ (rad/μs) | 5.0105 | 0.0348 | 0.0300 | 5 | +0.21 % | +0.30 |
| C6 (rad μm⁶/μs) | 5.4245 × 10⁶ | 1.11 × 10⁴ | 1.07 × 10⁴ | 5.42 × 10⁶ | +0.08 % | +0.40 |

The goodness-of-fit χ² is 174.5 for 165 degrees of freedom (p = 0.29), the three starting points reached the same estimate, and the fit raised no warnings. To check the error bars as well as one estimate, the experiment was repeated 20 more times with fresh shots:

| | mean error | RMS error | mean Cramér–Rao σ | mean (error/σ)² | within 2σ |
|---|---|---|---|---|---|
| Ω | −0.006 % | 0.105 % | 0.122 % | 0.74 | 20 of 20 |
| Δ | +0.070 % | 0.643 % | 0.699 % | 0.85 | 18 of 20 |
| C6 | −0.006 % | 0.188 % | 0.204 % | 0.85 | 20 of 20 |

So the pipeline recovers the simulator's parameters within the predicted errors, and the predicted errors match the scatter. The full output is in `dryrun/summary.txt` and `dryrun/summary.json`.

The local simulator has no noise model: no readout errors, empty sites, position errors, drive noise or decoherence. The dry run therefore tests the programs, the plumbing from Braket results to fitted parameters, and the statistics, not the device model. That is also why the dry-run errors are smaller than the ones expected on Aquila, by a factor of two to five.

## What to expect from Aquila

The analysis drops failed shots and shots whose pre-sequence image differs from the requested filling. With a typical filling error of 0.008 per site, about 5 % of shots are dropped; Braket's worst-case figure of 0.05 would drop 26 %. The fit models the readout errors with the rates the device reported at submission, and is started from the programmed values and from points shifted by the documented drive errors (3 % in Ω, 1 rad/μs in Δ).

Statistical errors at 200 shots per task should be about 0.2 to 0.6 % in Ω, 1.2 to 3 % in Δ and 0.4 to 0.9 % in C6, depending on how much decoherence there is. The fitted Ω and Δ will differ from the programmed 12 and 5 rad/μs: Braket lists a detuning error of 1 rad/μs and a total Ω error of 3 %, and the 50 ns ramps deliver about 92 % of their nominal area. That is what the experiment measures.

The C6 comparison with 2π × 862,690 rad μm⁶/μs is limited by systematic errors larger than the statistical ones:

- The data determine C6 / s⁶ only, where s is the array's length scale. A 0.1 % error in the 8 μm spacing is a 0.6 % error in C6. The fit calibrates C6 and the length scale together.
- Braket lists a site position error of 0.1 μm. The main README finds a 1.2 % RMS error in C6 for a 0.05 μm pattern error on this chain; at 0.1 μm, roughly twice that is likely. This was not computed directly.
- Thermal position jitter of 0.2 μm biases this chain's fit by about −0.6 % in C6 and −1.4 % in Δ (main README). The fit does not model jitter.
- The 50 ns ramps' Rabi correction of 0.92, which the model does not apply, moves all three parameters by about −0.14 %.
- Atoms lost during the sequence (0.5 % typical) read as Rydberg; this is only partly covered by the readout model.

A C6 within about 3 % of the reference would therefore be consistent with it; a closer agreement would partly be luck. The goodness-of-fit test may reject the closed model at 200 shots per task if drive noise or decoherence matter; the fit then carries a warning, and `--compare-noise` shows how the estimates and the test change with shot-to-shot drive noise and with relaxation at 7 μs. The relaxation model is a phenomenological per-atom bit flip, not a master equation, and its result should be read as a range rather than a correction.

Aquila's calibration drifts. The fitted Ω and Δ describe the window in which the tasks ran; a repeat on another day will differ by more than the statistical errors. `RydbergDriftTracker` in the package follows a drifting Δ shot by shot, if that becomes of interest.

## Limitations

- Nothing here has run on Aquila. The AWS-facing code (device status and windows, checks against the reported capabilities, `discretize` against the device, `device.run`, fetching with `AwsQuantumTask`) follows the SDK's documented interfaces and is tested only with a stand-in device object, since there are no credentials on this machine.
- The offline capability checks use figures copied from Braket's example notebook. They are checked again against what the device reports before submission.
- Prices and availability windows change. The cost above is from the pricing page on 2026-10-19.
- The local simulator is noiseless, so the dry run says nothing about how well the model describes the device.
