"""
Tests for Project 4 Extensions: Derandomized shadows, Streaming Kalman tracker, Fermionic matchgates, and QPT.
"""

import pytest
import numpy as np
import cirq
from shadow_learning.derandomized import DerandomizedShadowSelector
from shadow_learning.kalman_tracker import StreamingKalmanHamiltonianTracker
from shadow_learning.fermionic_shadows import FermionicMatchgateShadows
from shadow_learning.process_tomography import ShadowProcessTomographer
from shadow_learning.cirq_shadows import measure_random_pauli_shadows


def test_derandomized_basis_selection():
    target_obs = ["ZZ", "XX", "YY", "ZI", "IZ"]
    selector = DerandomizedShadowSelector(target_observables=target_obs, n_qubits=2)
    bases = selector.select_measurement_bases(n_snapshots=10)
    assert len(bases) == 10
    for b in bases:
        assert len(b) == 2
        assert b[0] in ["X", "Y", "Z"]
        assert b[1] in ["X", "Y", "Z"]


def test_streaming_kalman_tracker():
    tracker = StreamingKalmanHamiltonianTracker(n_qubits=2, initial_j_guess=np.array([0.5]))
    
    # Send stream of noisy measurements of J_01 = 0.8
    for _ in range(25):
        sample = 0.8 + np.random.normal(0, 0.2)
        state = tracker.process_shadow_snapshot("XX", sample, coupling_idx=0)

    assert abs(state[0] - 0.8) < 0.25


def test_fermionic_matchgate_shadows():
    fshadows = FermionicMatchgateShadows(n_modes=2)
    # Synthetic Majorana correlation matrix sample
    sample_mat = np.random.uniform(-1, 1, size=(4, 4))
    rdm_elem = fshadows.estimate_1rdm_element(0, 1, [sample_mat])
    assert isinstance(rdm_elem, complex)


def test_shadow_process_tomographer():
    tomographer = ShadowProcessTomographer(n_qubits=1)
    q = cirq.LineQubit(0)
    
    # Test on Identity channel
    c_0 = cirq.Circuit() # |0>
    c_p = cirq.Circuit([cirq.H(q)]) # |+>
    
    snaps_0 = measure_random_pauli_shadows(c_0, n_qubits=1, n_snapshots=100, seed=42)
    snaps_p = measure_random_pauli_shadows(c_p, n_qubits=1, n_snapshots=100, seed=42)
    
    fid = tomographer.estimate_process_fidelity(snaps_0, snaps_p)
    assert 0.0 <= fid <= 1.0
