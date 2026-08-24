"""
Unit tests for Classical Shadow Hamiltonian Learning.
"""

import pytest
import numpy as np
import cirq
from shadow_learning.cirq_shadows import (
    ClassicalShadowsProtocol,
    ShadowSnapshot,
    measure_random_pauli_shadows,
)
from shadow_learning.reconstruction import (
    estimate_pauli_string_expectation,
    estimate_many_observables,
    ShadowObservableEstimator,
)
from shadow_learning.differentiable_inversion import (
    DifferentiableHamiltonianLearner,
    learn_hamiltonian_from_shadows,
)
from shadow_learning.spam_mitigation import (
    ReadoutErrorModel,
    SPAMNoiseMitigator,
)


def test_shadow_protocol_snapshot_generation():
    q0 = cirq.LineQubit(0)
    c = cirq.Circuit([cirq.H(q0)])
    snaps = measure_random_pauli_shadows(c, n_qubits=1, n_snapshots=20, seed=123)
    assert len(snaps) == 20
    assert snaps[0].bases[0] in ["X", "Y", "Z"]
    assert snaps[0].bits[0] in [0, 1]


def test_pauli_expectation_unbiasedness():
    # State |0>: <Z> = 1.0, <X> = 0.0
    q0 = cirq.LineQubit(0)
    c = cirq.Circuit() # state |0>
    snaps = measure_random_pauli_shadows(c, n_qubits=1, n_snapshots=500, seed=42)

    exp_z = estimate_pauli_string_expectation(snaps, "Z")
    exp_x = estimate_pauli_string_expectation(snaps, "X")

    assert abs(exp_z - 1.0) < 0.15, f"Expected <Z> ~ 1.0, got {exp_z}"
    assert abs(exp_x - 0.0) < 0.15, f"Expected <X> ~ 0.0, got {exp_x}"


def test_bell_state_shadow_correlations():
    # Bell state (|00> + |11>) / sqrt(2)
    # <ZZ> = 1.0, <XX> = 1.0, <YY> = -1.0
    q = cirq.LineQubit.range(2)
    c = cirq.Circuit([cirq.H(q[0]), cirq.CNOT(q[0], q[1])])
    snaps = measure_random_pauli_shadows(c, n_qubits=2, n_snapshots=1200, seed=42)

    obs = estimate_many_observables(snaps, ["ZZ", "XX", "YY"])
    assert abs(obs["ZZ"] - 1.0) < 0.20
    assert abs(obs["XX"] - 1.0) < 0.20
    assert abs(obs["YY"] - (-1.0)) < 0.20


def test_density_matrix_reconstruction():
    q = cirq.LineQubit.range(1)
    c = cirq.Circuit([cirq.X(q[0])]) # state |1> -> rho = [[0, 0], [0, 1]]
    snaps = measure_random_pauli_shadows(c, n_qubits=1, n_snapshots=600, seed=42)

    estimator = ShadowObservableEstimator(snaps)
    rho_rec = estimator.reconstruct_reduced_density_matrix([0])

    assert rho_rec.shape == (2, 2)
    assert abs(np.trace(rho_rec) - 1.0) < 0.05
    assert abs(rho_rec[1, 1] - 1.0) < 0.15


def test_hamiltonian_learner_dynamics():
    learner = DifferentiableHamiltonianLearner(n_qubits=2, beta=1.0)
    j_mat = np.array([[0.0, 0.5], [0.5, 0.0]])
    h_vec = np.array([0.1, -0.1])

    obs = learner.compute_thermal_observables(j_mat, h_vec)
    assert len(obs) > 0
    assert "XX_0_1" in obs
    assert "Z_0" in obs


def test_spam_mitigation():
    readout = ReadoutErrorModel(p01=0.05, p10=0.05)
    mitigator = SPAMNoiseMitigator(readout)
    raw = 0.90
    mitigated = mitigator.mitigate_pauli_expectation(raw, pauli_weight=1)
    assert mitigated > raw
    assert abs(mitigated - 1.0) < 0.01
