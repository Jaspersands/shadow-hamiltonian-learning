"""
Adversarial and Stress Test Suite for shadow_learning.
"""

import pytest
import numpy as np
import cirq
from shadow_learning.cirq_shadows import ShadowSnapshot, measure_random_pauli_shadows
from shadow_learning.reconstruction import (
    estimate_pauli_string_expectation,
    estimate_pauli_expectation_median_of_means,
)
from shadow_learning.spam_mitigation import ReadoutErrorModel, SPAMNoiseMitigator


def test_adversarial_empty_and_minimal_snapshots():
    """Verify safe handling of empty and single-snapshot lists."""
    assert estimate_pauli_string_expectation([], "ZZ") == 0.0
    
    snap = ShadowSnapshot(bases=("Z", "Z"), bits=(0, 0))
    val = estimate_pauli_string_expectation([snap], "ZZ")
    assert val == 9.0 # 3 * 3 * 1.0 = 9.0 (unbiased single-shot realization)


def test_adversarial_identity_pauli_string():
    """Verify that identity operator 'II' always returns 1.0."""
    snaps = [
        ShadowSnapshot(bases=("X", "Y"), bits=(1, 0)),
        ShadowSnapshot(bases=("Z", "X"), bits=(0, 1)),
    ]
    val = estimate_pauli_string_expectation(snaps, "II")
    assert val == 1.0


def test_adversarial_median_of_means_edge_batches():
    """Verify median of means when batch count > snapshot count."""
    snaps = [
        ShadowSnapshot(bases=("Z", "Z"), bits=(0, 0)),
        ShadowSnapshot(bases=("Z", "Z"), bits=(0, 1)),
    ]
    # Request 10 batches for 2 snapshots -> should fallback gracefully
    val = estimate_pauli_expectation_median_of_means(snaps, "ZZ", n_batches=10)
    assert not np.isnan(val)


def test_adversarial_spam_singular_readout_guardrail():
    """Verify that extreme readout error (e.g. 50% random flip) does not explode to infinity."""
    bad_readout = ReadoutErrorModel(p01=0.50, p10=0.50) # completely random measurement
    mitigator = SPAMNoiseMitigator(bad_readout)
    val = mitigator.mitigate_pauli_expectation(0.5, pauli_weight=2)
    assert not np.isnan(val)
    assert not np.isinf(val)
