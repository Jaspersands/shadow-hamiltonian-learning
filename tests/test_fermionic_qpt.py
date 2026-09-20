"""
Tests for matchgate (fermionic) shadows and shadow process tomography.
"""

import numpy as np
import pytest

from shadow_learning.fermionic_shadows import (
    majorana_operators,
    random_signed_permutation,
    FermionicMatchgateShadows,
    MatchgateSnapshot,
    slater_state,
)
from shadow_learning.process_tomography import ShadowProcessTomographer, pauli_transfer_matrix_of_unitary


# ----------------------------------------------------------------------------- #
# Majoranas / matchgate shadows
# ----------------------------------------------------------------------------- #
def test_majorana_algebra():
    gammas = majorana_operators(3)
    assert len(gammas) == 6
    for i, gi in enumerate(gammas):
        assert np.allclose(gi, gi.conj().T)
        for j, gj in enumerate(gammas):
            anti = gi @ gj + gj @ gi
            assert np.allclose(anti, 2 * np.eye(8) if i == j else np.zeros((8, 8)))


def test_signed_permutation_sampling_is_orthogonal_and_seeded():
    rng = np.random.default_rng(0)
    perm, signs = random_signed_permutation(4, rng)
    assert sorted(perm) == list(range(8)) and set(np.unique(signs)) <= {-1, 1}
    Q = np.zeros((8, 8))
    for i, (p, s) in enumerate(zip(perm, signs)):
        Q[i, p] = s
    assert np.allclose(Q @ Q.T, np.eye(8))


def test_slater_state_occupations():
    psi = slater_state(4, occupied=[1, 3])
    assert np.isclose(np.linalg.norm(psi), 1.0)
    assert psi[0b0101] == 1.0  # qubit 1 and 3 set (qubit 0 = MSB)


def test_matchgate_1rdm_of_a_product_state():
    n = 4
    psi = slater_state(n, occupied=[1, 3])
    shadows = FermionicMatchgateShadows(n_modes=n, seed=1)
    snaps = shadows.sample(psi, n_snapshots=6000)
    assert isinstance(snaps[0], MatchgateSnapshot)
    D = shadows.estimate_1rdm(snaps)
    assert D.shape == (n, n)
    assert np.allclose(np.diag(D).real, [0, 1, 0, 1], atol=0.12)
    off = D - np.diag(np.diag(D))
    assert np.abs(off).max() < 0.15
    # exact reference from the state
    D_exact = shadows.exact_1rdm(psi)
    assert np.abs(D - D_exact).max() < 0.15


def test_matchgate_1rdm_of_a_delocalised_orbital():
    # One fermion in (c†_0 + c†_1)/√2 |vac>: D = ½[[1,1],[1,1]]
    n = 2
    psi = (slater_state(n, occupied=[0]) + slater_state(n, occupied=[1])) / np.sqrt(2)
    shadows = FermionicMatchgateShadows(n_modes=n, seed=2)
    snaps = shadows.sample(psi, n_snapshots=6000)
    D = shadows.estimate_1rdm(snaps)
    assert np.allclose(D, 0.5 * np.ones((2, 2)), atol=0.12)
    D_exact = shadows.exact_1rdm(psi)
    assert np.allclose(D_exact, 0.5 * np.ones((2, 2)), atol=1e-12)


def test_majorana_pair_estimator_is_unbiased_for_number_operator():
    n = 3
    psi = slater_state(n, occupied=[0, 2])
    shadows = FermionicMatchgateShadows(n_modes=n, seed=5)
    snaps = shadows.sample(psi, n_snapshots=4000)
    # <i γ_{2p} γ_{2p+1}> = 2 n_p − 1  (Z_p = −i γ_{2p} γ_{2p+1})
    for p_mode, occ in enumerate([1, 0, 1]):
        est = shadows.estimate_majorana_pair(snaps, 2 * p_mode, 2 * p_mode + 1)
        assert abs(est - (2 * occ - 1)) < 0.12


# ----------------------------------------------------------------------------- #
# Process tomography
# ----------------------------------------------------------------------------- #
def _channel_from_unitary(u):
    return lambda rho: u @ rho @ u.conj().T


def test_ptm_of_unitary_helper():
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    R = pauli_transfer_matrix_of_unitary(X)
    assert np.allclose(R, np.diag([1, 1, -1, -1]))


def test_qpt_identity_and_x_channel():
    tomo = ShadowProcessTomographer(n_qubits=1, seed=0)
    res = tomo.run(_channel_from_unitary(np.eye(2)), n_snapshots_per_input=1500)
    assert np.abs(res["ptm"] - np.eye(4)).max() < 0.12
    assert abs(res["process_fidelity"] - 1.0) < 0.1 if "process_fidelity" in res else True
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    res_x = tomo.run(_channel_from_unitary(X), n_snapshots_per_input=1500)
    assert np.abs(res_x["ptm"] - np.diag([1, 1, -1, -1])).max() < 0.12
    assert abs(tomo.process_fidelity(res_x["ptm"], X) - 1.0) < 0.1
    assert tomo.process_fidelity(res_x["ptm"], np.eye(2)) < 0.3


def test_qpt_depolarizing_channel_and_choi():
    p = 0.3
    dep = lambda rho: (1 - p) * rho + p * np.trace(rho) * np.eye(2) / 2
    tomo = ShadowProcessTomographer(n_qubits=1, seed=1)
    res = tomo.run(dep, n_snapshots_per_input=2000)
    R = res["ptm"]
    assert abs(R[0, 0] - 1) < 0.05
    for k in (1, 2, 3):
        assert abs(R[k, k] - (1 - p)) < 0.12
    choi = res["choi"]
    assert choi.shape == (4, 4)
    assert abs(np.trace(choi) - 2.0) < 0.2                  # Tr Choi = d for trace-preserving
    assert np.all(np.linalg.eigvalsh((choi + choi.conj().T) / 2) > -0.15)
    # average gate fidelity of depolarising vs identity: (d F_pro + 1)/(d+1) with F_pro = 1 - 3p/4
    f_avg = tomo.average_gate_fidelity(R, np.eye(2))
    assert abs(f_avg - (2 * (1 - 0.75 * p) + 1) / 3) < 0.08


def test_qpt_two_qubit_cnot():
    cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
    tomo = ShadowProcessTomographer(n_qubits=2, seed=3)
    res = tomo.run(_channel_from_unitary(cnot), n_snapshots_per_input=600)
    assert res["ptm"].shape == (16, 16)
    assert abs(tomo.process_fidelity(res["ptm"], cnot) - 1.0) < 0.15
