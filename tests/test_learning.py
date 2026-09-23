"""
Tests for differentiable Hamiltonian inversion and the streaming EKF.
"""

import numpy as np
import pytest

from shadow_learning.cirq_shadows import gibbs_state, sample_shadows_from_density_matrix
from shadow_learning import differentiable_inversion as di
from shadow_learning.differentiable_inversion import (
    DifferentiableHamiltonianLearner,
    HamiltonianLearningResult,
    learn_hamiltonian_from_shadows,
)
from shadow_learning.kalman_tracker import StreamingKalmanHamiltonianTracker


TRUE_J = np.array([[0.0, 0.75], [0.75, 0.0]])
TRUE_H = np.array([0.20, -0.15])


def test_pack_unpack_roundtrip_and_observable_names():
    learner = DifferentiableHamiltonianLearner(n_qubits=3, beta=1.0)
    J = np.array([[0, 0.1, 0.2], [0.1, 0, 0.3], [0.2, 0.3, 0]])
    h = np.array([0.4, 0.5, 0.6])
    p = learner.pack(J, h)
    assert p.shape == (6,)
    J2, h2 = learner.unpack(p)
    assert np.allclose(J, J2) and np.allclose(h, h2)
    assert "Z_0" in learner.observable_names and "XX_0_2" in learner.observable_names
    assert learner.pauli_string("XX_0_2") == "XIX"


def test_thermal_observables_match_direct_trace():
    learner = DifferentiableHamiltonianLearner(n_qubits=2, beta=1.5)
    obs = learner.compute_thermal_observables(TRUE_J, TRUE_H)
    rho = gibbs_state(learner.build_hamiltonian_matrix(TRUE_J, TRUE_H), 1.5)
    for name, val in obs.items():
        assert np.isclose(val, np.real(np.trace(learner.pauli_matrix(learner.pauli_string(name)) @ rho)), atol=1e-10)


@pytest.mark.skipif(not di.HAS_JAX, reason="JAX not installed")
def test_jax_gradient_matches_finite_difference():
    learner = DifferentiableHamiltonianLearner(n_qubits=2, beta=1.5)
    targets = learner.compute_thermal_observables(TRUE_J, TRUE_H)
    p = np.array([0.3, 0.1, 0.05])
    val, grad = learner.loss_and_grad(p, targets)
    fd = np.zeros_like(p)
    for i in range(3):
        pp = p.copy(); pp[i] += 1e-6
        pm = p.copy(); pm[i] -= 1e-6
        fd[i] = (learner.loss_and_grad(pp, targets)[0] - learner.loss_and_grad(pm, targets)[0]) / 2e-6
    assert np.allclose(grad, fd, rtol=1e-4, atol=1e-7)


def test_learn_from_exact_expectations_recovers_parameters():
    learner = DifferentiableHamiltonianLearner(n_qubits=2, beta=1.5, seed=0)
    targets = learner.compute_thermal_observables(TRUE_J, TRUE_H)
    res = learner.learn_from_expectations(targets, true_j_matrix=TRUE_J, true_h_vector=TRUE_H)
    assert isinstance(res, HamiltonianLearningResult)
    assert abs(res.recovered_j_matrix[0, 1] - 0.75) < 1e-3
    assert np.allclose(res.recovered_h_vector, TRUE_H, atol=2e-3)
    assert res.frobenius_error < 2e-3 and res.final_loss < 1e-6
    assert res.method == "analytic-lbfgs"


def test_learn_from_gibbs_shadows_end_to_end():
    learner = DifferentiableHamiltonianLearner(n_qubits=2, beta=1.5, seed=1)
    rho = gibbs_state(learner.build_hamiltonian_matrix(TRUE_J, TRUE_H), 1.5)
    snaps = sample_shadows_from_density_matrix(rho, 6000, np.random.default_rng(11))
    res = learner.learn_from_shadows(snaps, true_j_matrix=TRUE_J, true_h_vector=TRUE_H)
    assert abs(res.recovered_j_matrix[0, 1] - 0.75) < 0.08
    # The ~94 %-singlet Gibbs state screens a uniform field: only h0 − h1 is
    # well identified here (see test_identifiability_flags_uniform_field).
    assert abs((res.recovered_h_vector[0] - res.recovered_h_vector[1]) - (TRUE_H[0] - TRUE_H[1])) < 0.1
    assert res.iterations > 0 and len(res.loss_history) == res.iterations
    helper = learn_hamiltonian_from_shadows(snaps, n_qubits=2, beta=1.5, seed=1)
    assert abs(helper.recovered_j_matrix[0, 1] - res.recovered_j_matrix[0, 1]) < 1e-6


def test_identifiability_flags_uniform_field():
    learner = DifferentiableHamiltonianLearner(n_qubits=2, beta=1.5)
    info = learner.identifiability(TRUE_J, TRUE_H)
    sv = info["singular_values"]
    assert sv[0] / sv[-1] > 10                      # ill-conditioned in this regime
    direction = np.abs(info["least_identifiable"])
    # least identifiable direction ≈ uniform field (0, 1, 1)/√2 in (J01, h0, h1) coordinates
    assert direction[0] < 0.3 and abs(direction[1] - direction[2]) < 0.2 and direction[1] > 0.5
    # a hotter state (β = 0.3) is far better conditioned
    hot = DifferentiableHamiltonianLearner(n_qubits=2, beta=0.3).identifiability(TRUE_J, TRUE_H)
    assert hot["condition_number"] < info["condition_number"]


def test_shadow_inversion_in_a_well_conditioned_regime():
    learner = DifferentiableHamiltonianLearner(n_qubits=2, beta=0.6, seed=2)
    J = np.array([[0.0, 0.4], [0.4, 0.0]]); h = np.array([0.5, -0.3])
    rho = gibbs_state(learner.build_hamiltonian_matrix(J, h), 0.6)
    snaps = sample_shadows_from_density_matrix(rho, 8000, np.random.default_rng(4))
    res = learner.learn_from_shadows(snaps, true_j_matrix=J, true_h_vector=h)
    assert abs(res.recovered_j_matrix[0, 1] - 0.4) < 0.08
    assert np.allclose(res.recovered_h_vector, h, atol=0.12)


def test_analytic_path_needs_no_jax(monkeypatch):
    monkeypatch.setattr(di, "HAS_JAX", False)
    learner = DifferentiableHamiltonianLearner(n_qubits=2, beta=1.0, seed=0)
    targets = learner.compute_thermal_observables(TRUE_J, TRUE_H)
    res = learner.learn_from_expectations(targets)
    assert res.method == "analytic-lbfgs"
    assert abs(res.recovered_j_matrix[0, 1] - 0.75) < 1e-3
    with pytest.raises(ImportError):
        learner.loss_and_grad(learner.pack(TRUE_J, TRUE_H), targets, backend="jax")


def test_ekf_tracks_a_step_change_in_coupling():
    tracker = StreamingKalmanHamiltonianTracker(n_qubits=2, beta=1.5, process_noise_std=0.005, seed=0)
    rng = np.random.default_rng(3)
    learner = tracker.learner
    j_values = [0.3] * 1500 + [0.8] * 1500
    estimates = []
    for k, jv in enumerate(j_values):
        if k % 500 == 0:
            rho = gibbs_state(learner.build_hamiltonian_matrix(np.array([[0, jv], [jv, 0]]), TRUE_H), 1.5)
        snap = sample_shadows_from_density_matrix(rho, 1, rng)[0]
        est = tracker.process_snapshot(snap)
        estimates.append(est["J"][0, 1])
    assert abs(estimates[1400] - 0.3) < 0.15
    assert abs(estimates[-1] - 0.8) < 0.15
    assert tracker.n_updates > 0
    hist = tracker.history()
    assert hist["J_01"].shape == (3000,)
