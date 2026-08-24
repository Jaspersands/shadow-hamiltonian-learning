"""
Classical Shadow Observable Estimation and State Reconstruction.

Implements:
1. Fast O(k) Pauli string expectation estimation without constructing full 2^N state matrices.
2. Median-of-Means estimator for variance bounding.
3. Density matrix snapshot reconstruction.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Sequence, Dict, Any, Optional
from .cirq_shadows import ShadowSnapshot


def estimate_pauli_string_expectation(
    snapshots: Sequence[ShadowSnapshot],
    pauli_string: str, # e.g. "XXII", "ZIZI", "YY"
) -> float:
    """
    Computes unbiased expectation value of a multi-qubit Pauli observable:
        <P> = (1 / S) * sum_{s=1}^S prod_{i: P_i != I} (3 * (-1)^{b_i} * delta_{W_i, P_i})
    """
    s_count = len(snapshots)
    if s_count == 0:
        return 0.0

    target_chars = list(pauli_string)
    non_trivial_indices = [i for i, c in enumerate(target_chars) if c in ["X", "Y", "Z"]]

    if not non_trivial_indices:
        return 1.0 # Identity operator has expectation 1.0

    total_val = 0.0
    for snap in snapshots:
        # Check if measurement bases match observable
        match = True
        val = 1.0
        for idx in non_trivial_indices:
            target_p = target_chars[idx]
            measured_p = snap.bases[idx]
            if measured_p != target_p:
                match = False
                break
            bit = snap.bits[idx]
            val *= 3.0 * (-1.0 if bit == 1 else 1.0)

        if match:
            total_val += val

    return float(total_val / s_count)


def estimate_pauli_expectation_median_of_means(
    snapshots: Sequence[ShadowSnapshot],
    pauli_string: str,
    n_batches: int = 5,
) -> float:
    """
    Computes Median-of-Means estimator for robust variance reduction.
    """
    s_count = len(snapshots)
    if s_count < n_batches or n_batches <= 1:
        return estimate_pauli_string_expectation(snapshots, pauli_string)

    batch_size = s_count // n_batches
    batch_means = []

    for b in range(n_batches):
        batch = snapshots[b * batch_size : (b + 1) * batch_size]
        mean_val = estimate_pauli_string_expectation(batch, pauli_string)
        batch_means.append(mean_val)

    return float(np.median(batch_means))


def estimate_many_observables(
    snapshots: Sequence[ShadowSnapshot],
    observables: Sequence[str],
    use_median_of_means: bool = False,
    n_batches: int = 5,
) -> Dict[str, float]:
    """
    Simultaneously estimates M Pauli observables from the same shadow dataset.
    Computational complexity: O(M * S).
    """
    results = {}
    for obs in observables:
        if use_median_of_means:
            results[obs] = estimate_pauli_expectation_median_of_means(
                snapshots, obs, n_batches=n_batches
            )
        else:
            results[obs] = estimate_pauli_string_expectation(snapshots, obs)
    return results


class ShadowObservableEstimator:
    """
    High-throughput observable estimation engine for shadow data.
    """

    def __init__(self, snapshots: Sequence[ShadowSnapshot]):
        self.snapshots = list(snapshots)
        self.n_snapshots = len(self.snapshots)
        self.n_qubits = len(self.snapshots[0].bases) if self.snapshots else 0

    def predict(self, observables: Sequence[str]) -> Dict[str, float]:
        return estimate_many_observables(self.snapshots, observables)

    def reconstruct_reduced_density_matrix(self, qubits: Sequence[int]) -> np.ndarray:
        """
        Reconstructs the k-qubit reduced density matrix rho_{qubits} from shadow snapshots.
        """
        k = len(qubits)
        dim = 2**k
        rho = np.zeros((dim, dim), dtype=np.complex128)

        # Single-qubit reconstructed state matrices
        id2 = np.eye(2, dtype=np.complex128)
        paulis = {
            "X": [
                np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128),
                np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=np.complex128),
            ],
            "Y": [
                np.array([[0.5, -0.5j], [0.5j, 0.5]], dtype=np.complex128),
                np.array([[0.5, 0.5j], [-0.5j, 0.5]], dtype=np.complex128),
            ],
            "Z": [
                np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128),
                np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128),
            ],
        }

        for snap in self.snapshots:
            # Build tensor product of inverted single-qubit states
            local_rho = None
            for q_idx in qubits:
                base = snap.bases[q_idx]
                bit = snap.bits[q_idx]
                state_proj = paulis[base][bit]
                inverted = 3.0 * state_proj - id2

                if local_rho is None:
                    local_rho = inverted
                else:
                    local_rho = np.kron(local_rho, inverted)

            rho += local_rho

        rho = rho / self.n_snapshots
        return rho
