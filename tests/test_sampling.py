"""
Tests for shadow sampling: the single-simulation fast path, density-matrix
sampling, Gibbs states and readout-error injection.
"""

import numpy as np
import pytest
from scipy.stats import chi2

cirq = pytest.importorskip("cirq")

from shadow_learning.cirq_shadows import (
    ClassicalShadowsProtocol,
    ShadowSnapshot,
    measure_random_pauli_shadows,
    sample_shadows_from_density_matrix,
    gibbs_state,
    basis_rotation_matrix,
)
from shadow_learning.reconstruction import estimate_many_observables, estimate_pauli_string_expectation
from shadow_learning.spam_mitigation import ReadoutErrorModel


def bell_circuit():
    q = cirq.LineQubit.range(2)
    return cirq.Circuit([cirq.H(q[0]), cirq.CNOT(q[0], q[1])])


def test_snapshot_dataclass():
    s = ShadowSnapshot(bases=("X", "Z"), bits=(1, 0))
    assert s.bases == ("X", "Z") and s.bits == (1, 0)


def test_fast_and_slow_samplers_agree_on_bell_state():
    c = bell_circuit()
    fast = ClassicalShadowsProtocol(2, seed=1).measure_state(c, n_snapshots=3000, fast=True)
    slow = ClassicalShadowsProtocol(2, seed=1).measure_state(c, n_snapshots=600, fast=False)
    for snaps, tol in ((fast, 0.12), (slow, 0.3)):
        obs = estimate_many_observables(snaps, ["ZZ", "XX", "YY", "ZI"])
        assert abs(obs["ZZ"] - 1.0) < tol and abs(obs["XX"] - 1.0) < tol and abs(obs["YY"] + 1.0) < tol
        assert abs(obs["ZI"]) < tol

    # chi-square on the joint (basis, bit) histogram: both samplers must follow the same law
    def hist(snaps):
        counts = {}
        for s in snaps:
            key = (s.bases, s.bits)
            counts[key] = counts.get(key, 0) + 1
        return counts

    hf, hs = hist(fast), hist(slow)
    keys = sorted(set(hf) | set(hs))
    nf, ns = len(fast), len(slow)
    stat = 0.0
    for k in keys:
        a, b = hf.get(k, 0), hs.get(k, 0)
        expected = (a + b) / (nf + ns)
        if expected > 0:
            stat += (a - nf * expected) ** 2 / (nf * expected) + (b - ns * expected) ** 2 / (ns * expected)
    p_value = 1 - chi2.cdf(stat, df=len(keys) - 1)
    assert p_value > 0.001


def test_basis_rotation_matrices():
    # measuring in X after H, in Y after S†·H: eigenstates map to |0>,|1>
    plus = np.array([1, 1]) / np.sqrt(2)
    plus_i = np.array([1, 1j]) / np.sqrt(2)
    assert np.isclose(abs((basis_rotation_matrix("X") @ plus)[0]), 1.0)
    assert np.isclose(abs((basis_rotation_matrix("Y") @ plus_i)[0]), 1.0)
    assert np.allclose(basis_rotation_matrix("Z"), np.eye(2))


def test_sample_from_density_matrix_pure_state():
    psi = np.array([1, 0, 0, 1]) / np.sqrt(2)
    rho = np.outer(psi, psi.conj())
    snaps = sample_shadows_from_density_matrix(rho, 3000, np.random.default_rng(3))
    obs = estimate_many_observables(snaps, ["ZZ", "XX", "YY"])
    assert abs(obs["ZZ"] - 1) < 0.12 and abs(obs["XX"] - 1) < 0.12 and abs(obs["YY"] + 1) < 0.12


def test_gibbs_state_limits():
    H = np.diag([0.0, 1.0, 2.0, 3.0]).astype(complex)
    rho_hot = gibbs_state(H, beta=0.0)
    assert np.allclose(rho_hot, np.eye(4) / 4)
    rho_cold = gibbs_state(H, beta=200.0)
    assert np.isclose(rho_cold[0, 0].real, 1.0, atol=1e-6)
    rho = gibbs_state(H, beta=1.0)
    assert np.isclose(np.trace(rho).real, 1.0) and np.all(np.linalg.eigvalsh(rho) >= -1e-12)
    # Boltzmann weights
    assert np.isclose(rho[1, 1].real / rho[0, 0].real, np.exp(-1.0))


def test_readout_error_biases_z_expectation():
    rho = np.diag([1.0, 0.0]).astype(complex)                      # |0>
    model = ReadoutErrorModel(p01=0.10, p10=0.0)
    snaps = sample_shadows_from_density_matrix(rho, 6000, np.random.default_rng(5), readout_error=model)
    raw = estimate_pauli_string_expectation(snaps, "Z")
    assert abs(raw - 0.8) < 0.06                                   # <Z> = 1 - 2 p01 before mitigation


def test_measure_random_pauli_shadows_helper_seeded():
    a = measure_random_pauli_shadows(bell_circuit(), n_qubits=2, n_snapshots=50, seed=9)
    b = measure_random_pauli_shadows(bell_circuit(), n_qubits=2, n_snapshots=50, seed=9)
    assert a == b and len(a) == 50
