"""
Tests for shadow estimators, HKP derandomisation, and SPAM mitigation.
"""

import numpy as np
import pytest

from shadow_learning.cirq_shadows import ShadowSnapshot, sample_shadows_from_density_matrix, gibbs_state
from shadow_learning.reconstruction import (
    estimate_pauli_string_expectation,
    estimate_pauli_expectation_median_of_means,
    estimate_many_observables,
    snapshots_to_arrays,
    ShadowObservableEstimator,
    _estimate_loop_reference,
)
from shadow_learning.derandomized import DerandomizedShadowSelector, hits_per_observable
from shadow_learning.spam_mitigation import ReadoutErrorModel, SPAMNoiseMitigator


def bell_rho():
    psi = np.array([1, 0, 0, 1]) / np.sqrt(2)
    return np.outer(psi, psi.conj())


def test_single_snapshot_and_identity_values():
    snap = ShadowSnapshot(bases=("Z", "Z"), bits=(0, 0))
    assert estimate_pauli_string_expectation([snap], "ZZ") == 9.0
    assert estimate_pauli_string_expectation([snap], "II") == 1.0
    assert estimate_pauli_string_expectation([], "ZZ") == 0.0
    assert estimate_pauli_string_expectation([snap], "XZ") == 0.0


def test_vectorised_estimator_matches_loop_reference():
    rng = np.random.default_rng(0)
    snaps = sample_shadows_from_density_matrix(bell_rho(), 500, rng)
    for P in ("ZZ", "XX", "YY", "ZI", "IX", "XY"):
        assert np.isclose(estimate_pauli_string_expectation(snaps, P), _estimate_loop_reference(snaps, P))
    bases, bits = snapshots_to_arrays(snaps)
    assert bases.shape == (500, 2) and bits.shape == (500, 2) and bases.dtype == np.uint8


def test_median_of_means_and_many_observables():
    rng = np.random.default_rng(1)
    snaps = sample_shadows_from_density_matrix(bell_rho(), 3000, rng)
    mom = estimate_pauli_expectation_median_of_means(snaps, "ZZ", n_batches=10)
    assert abs(mom - 1.0) < 0.2
    many = estimate_many_observables(snaps, ["ZZ", "XX", "YY"], use_median_of_means=True)
    assert set(many) == {"ZZ", "XX", "YY"}
    tiny = [ShadowSnapshot(("Z", "Z"), (0, 0)), ShadowSnapshot(("Z", "Z"), (0, 1))]
    assert not np.isnan(estimate_pauli_expectation_median_of_means(tiny, "ZZ", n_batches=10))


def test_reduced_density_matrix_reconstruction():
    rho1 = np.diag([0.0, 1.0]).astype(complex)
    snaps = sample_shadows_from_density_matrix(rho1, 3000, np.random.default_rng(2))
    est = ShadowObservableEstimator(snaps)
    rec = est.reconstruct_reduced_density_matrix([0])
    assert abs(np.trace(rec) - 1.0) < 0.05 and abs(rec[1, 1] - 1.0) < 0.12


# ----------------------------------------------------------------------------- #
# Derandomisation (Huang–Kueng–Preskill 2021)
# ----------------------------------------------------------------------------- #
def test_derandomized_bases_cover_every_target():
    targets = ["ZZ", "XX", "YY"]
    sel = DerandomizedShadowSelector(target_observables=targets, n_qubits=2)
    bases = sel.select_measurement_bases(n_snapshots=90)
    assert len(bases) == 90 and all(len(b) == 2 for b in bases)
    assert len(set(bases)) >= 3, "v0.2 returned the same basis for every shot"
    hits = hits_per_observable(bases, targets)
    for obs in targets:
        assert hits[obs] >= 0.25 * 90


def test_derandomized_is_deterministic_and_handles_identity_padding():
    sel = DerandomizedShadowSelector(["ZI", "IZ", "ZZ"], n_qubits=2)
    a, b = sel.select_measurement_bases(20), sel.select_measurement_bases(20)
    assert a == b
    assert all(x == ("Z", "Z") for x in a)        # Z on both qubits serves all three at once


def test_derandomized_beats_random_at_equal_budget():
    # Gibbs state of a 3-qubit Heisenberg chain; estimate XX/YY/ZZ on both bonds.
    from shadow_learning.differentiable_inversion import DifferentiableHamiltonianLearner
    learner = DifferentiableHamiltonianLearner(n_qubits=3, beta=1.0)
    J = np.array([[0, 0.7, 0], [0.7, 0, -0.4], [0, -0.4, 0]])
    h = np.array([0.2, 0.0, -0.3])
    rho = gibbs_state(learner.build_hamiltonian_matrix(J, h), beta=1.0)
    targets = ["XXI", "YYI", "ZZI", "IXX", "IYY", "IZZ"]
    truth = {P: float(np.real(np.trace(learner.pauli_matrix(P) @ rho))) for P in targets}
    sel = DerandomizedShadowSelector(targets, n_qubits=3)
    bases = sel.select_measurement_bases(300)
    rmse_rand, rmse_der = [], []
    for trial in range(12):
        rng = np.random.default_rng(100 + trial)
        s_rand = sample_shadows_from_density_matrix(rho, 300, rng)
        s_der = sample_shadows_from_density_matrix(rho, 300, rng, bases=bases)
        e_rand = estimate_many_observables(s_rand, targets)
        e_der = estimate_many_observables(s_der, targets, derandomized=True)
        rmse_rand.append(np.sqrt(np.mean([(e_rand[P] - truth[P]) ** 2 for P in targets])))
        rmse_der.append(np.sqrt(np.mean([(e_der[P] - truth[P]) ** 2 for P in targets])))
    assert np.mean(rmse_der) < np.mean(rmse_rand)
    assert np.mean(rmse_der) < 0.15


# ----------------------------------------------------------------------------- #
# SPAM
# ----------------------------------------------------------------------------- #
def test_readout_model_corrected_eigenvalues_are_unbiased():
    m = ReadoutErrorModel(p01=0.02, p10=0.10)
    g = m.corrected_eigenvalue
    # E_meas[g(b')] must equal (-1)^b for true bit b
    for true_bit, target in ((0, 1.0), (1, -1.0)):
        p_meas_1 = m.p01 if true_bit == 0 else 1 - m.p10
        expect = (1 - p_meas_1) * g(0) + p_meas_1 * g(1)
        assert np.isclose(expect, target)


def test_spam_mitigation_asymmetric_readout():
    plus = np.array([1, 1]) / np.sqrt(2)
    rho = np.outer(plus, plus.conj())
    model = ReadoutErrorModel(p01=0.02, p10=0.10)
    snaps = sample_shadows_from_density_matrix(rho, 6000, np.random.default_rng(7), readout_error=model)
    raw = estimate_pauli_string_expectation(snaps, "X")
    mitigated = estimate_pauli_string_expectation(snaps, "X", readout_model=model)
    assert abs(raw - 0.96) < 0.04                      # biased: (1-p01-p10)·1 + (p10-p01) = 0.96
    assert abs(mitigated - 1.0) < 0.06                 # exact single-shot inversion
    mit = SPAMNoiseMitigator(model)
    assert np.isclose(mit.estimate(snaps, "X"), mitigated)
    assert mit.mitigate_pauli_expectation(raw, pauli_weight=1) > 1.05   # symmetric rule overshoots for asymmetric errors


def test_spam_symmetric_helper_and_singular_guard():
    sym = SPAMNoiseMitigator(ReadoutErrorModel(p01=0.05, p10=0.05))
    assert np.isclose(sym.mitigate_pauli_expectation(0.9, pauli_weight=1), 1.0, atol=1e-9)
    bad = SPAMNoiseMitigator(ReadoutErrorModel(p01=0.5, p10=0.5))
    v = bad.mitigate_pauli_expectation(0.5, pauli_weight=2)
    assert np.isfinite(v)
    assert np.isfinite(bad.model.corrected_eigenvalue(0))
