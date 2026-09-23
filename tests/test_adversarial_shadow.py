"""
Adversarial / stress tests: degenerate inputs and invariants.
"""

import numpy as np
import pytest

from shadow_learning.cirq_shadows import ShadowSnapshot, sample_shadows_from_density_matrix, gibbs_state
from shadow_learning.reconstruction import estimate_pauli_string_expectation, estimate_pauli_expectation_median_of_means
from shadow_learning.spam_mitigation import ReadoutErrorModel, SPAMNoiseMitigator
from shadow_learning.derandomized import DerandomizedShadowSelector
from shadow_learning.differentiable_inversion import DifferentiableHamiltonianLearner
from shadow_learning.cli import main as cli_main
from shadow_learning import __version__


def test_version():
    assert __version__ == "0.3.0"


def test_empty_and_minimal_snapshots():
    assert estimate_pauli_string_expectation([], "ZZ") == 0.0
    snap = ShadowSnapshot(bases=("Z", "Z"), bits=(0, 0))
    assert estimate_pauli_string_expectation([snap], "ZZ") == 9.0
    assert estimate_pauli_string_expectation([snap], "ZZ", derandomized=True) == 1.0


def test_identity_pauli_string_is_one():
    snaps = [ShadowSnapshot(("X", "Y"), (1, 0)), ShadowSnapshot(("Z", "X"), (0, 1))]
    assert estimate_pauli_string_expectation(snaps, "II") == 1.0


def test_median_of_means_edge_batches():
    snaps = [ShadowSnapshot(("Z", "Z"), (0, 0)), ShadowSnapshot(("Z", "Z"), (0, 1))]
    assert not np.isnan(estimate_pauli_expectation_median_of_means(snaps, "ZZ", n_batches=10))


def test_singular_readout_guardrail():
    bad = ReadoutErrorModel(p01=0.5, p10=0.5)
    assert bad.singular
    assert np.isfinite(SPAMNoiseMitigator(bad).mitigate_pauli_expectation(0.5, 2))
    assert np.isfinite(bad.corrected_eigenvalue(1))


def test_derandomizer_rejects_bad_observables():
    with pytest.raises(ValueError):
        DerandomizedShadowSelector(["XZ"], n_qubits=3)
    with pytest.raises(ValueError):
        DerandomizedShadowSelector(["XQ"], n_qubits=2)


def test_gibbs_sampler_handles_pure_and_mixed_extremes():
    H = np.diag([0.0, 5.0, 5.0, 5.0]).astype(complex)
    snaps = sample_shadows_from_density_matrix(gibbs_state(H, 100.0), 300, np.random.default_rng(0))
    assert abs(estimate_pauli_string_expectation(snaps, "ZZ") - 1.0) < 0.3
    snaps = sample_shadows_from_density_matrix(gibbs_state(H, 0.0), 300, np.random.default_rng(0))
    assert abs(estimate_pauli_string_expectation(snaps, "ZZ")) < 0.5


def test_learner_single_qubit_has_no_couplings():
    learner = DifferentiableHamiltonianLearner(n_qubits=1, beta=1.0)
    assert learner.n_params == 1 and learner.observable_names == ["Z_0"]
    obs = learner.compute_thermal_observables(np.zeros((1, 1)), np.array([0.7]))
    assert np.isclose(obs["Z_0"], -np.tanh(0.7))


def test_cli_learn_and_qpt(capsys):
    import json
    assert cli_main(["learn", "--qubits", "2", "--shots", "600", "--beta", "0.6", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["method"] in ("analytic-lbfgs", "jax-lbfgs") and "recovered_J" in out
    assert cli_main(["qpt", "--channel", "x", "--shots", "300", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["process_fidelity"] > 0.7
    assert cli_main(["derand-compare", "--qubits", "2", "--shots", "100", "--trials", "2", "--json"]) == 0
    assert "gain" in json.loads(capsys.readouterr().out)
