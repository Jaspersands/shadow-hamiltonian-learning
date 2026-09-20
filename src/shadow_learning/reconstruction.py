"""
Observable estimation from Pauli shadows.

For randomised shadows the unbiased single-snapshot estimator of a weight-k
Pauli string P is  ∏_{i ∈ supp P} 3·(−1)^{b_i}·δ(W_i, P_i); its average over
snapshots estimates ⟨P⟩. For *derandomised* shadows the bases are fixed, so
the estimator is the plain mean of ∏(−1)^{b_i} over the snapshots whose bases
match P (each such snapshot is an ordinary projective measurement of P).
Readout errors are handled by replacing (−1)^{b} with the confusion-matrix
corrected eigenvalue g(b) (see :mod:`spam_mitigation`).

All estimators are vectorised over snapshots.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .cirq_shadows import ShadowSnapshot

_BASIS_CODE = {"X": 0, "Y": 1, "Z": 2, "I": 3}


def snapshots_to_arrays(snapshots: Sequence[ShadowSnapshot]) -> Tuple[np.ndarray, np.ndarray]:
    """(bases uint8 [S, n] with X=0,Y=1,Z=2 ; bits uint8 [S, n])."""
    if len(snapshots) == 0:
        return np.zeros((0, 0), dtype=np.uint8), np.zeros((0, 0), dtype=np.uint8)
    bases = np.array([[_BASIS_CODE[b] for b in s.bases] for s in snapshots], dtype=np.uint8)
    bits = np.array([list(s.bits) for s in snapshots], dtype=np.uint8)
    return bases, bits


def _pauli_support(pauli_string: str) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.array([i for i, c in enumerate(pauli_string) if c in "XYZ"], dtype=int)
    codes = np.array([_BASIS_CODE[pauli_string[i]] for i in idx], dtype=np.uint8)
    return idx, codes


def _eigenvalues(bits: np.ndarray, readout_model=None) -> np.ndarray:
    """(−1)^b, or the readout-corrected eigenvalue g(b)."""
    if readout_model is None:
        return 1.0 - 2.0 * bits.astype(float)
    g0, g1 = readout_model.corrected_eigenvalue(0), readout_model.corrected_eigenvalue(1)
    return np.where(bits == 0, g0, g1)


def estimate_pauli_string_expectation(
    snapshots: Sequence[ShadowSnapshot],
    pauli_string: str,
    derandomized: bool = False,
    readout_model=None,
) -> float:
    """
    ⟨P⟩ from shadows. ``derandomized=True`` averages only over matching snapshots
    without the 3^k weight (fixed bases); ``readout_model`` de-biases readout errors.
    """
    if len(snapshots) == 0:
        return 0.0
    idx, codes = _pauli_support(pauli_string)
    if idx.size == 0:
        return 1.0
    bases, bits = snapshots_to_arrays(snapshots)
    match = np.all(bases[:, idx] == codes[None, :], axis=1)
    if not match.any():
        return 0.0
    vals = np.prod(_eigenvalues(bits[match][:, idx], readout_model), axis=1)
    if derandomized:
        return float(vals.mean())
    return float(3.0 ** idx.size * vals.sum() / len(snapshots))


def _estimate_loop_reference(snapshots: Sequence[ShadowSnapshot], pauli_string: str) -> float:
    """Slow reference implementation kept for tests."""
    if not snapshots:
        return 0.0
    support = [i for i, c in enumerate(pauli_string) if c in "XYZ"]
    if not support:
        return 1.0
    total = 0.0
    for s in snapshots:
        val = 1.0
        for i in support:
            if s.bases[i] != pauli_string[i]:
                val = 0.0
                break
            val *= 3.0 * (-1.0 if s.bits[i] else 1.0)
        total += val
    return total / len(snapshots)


def estimate_pauli_expectation_median_of_means(
    snapshots: Sequence[ShadowSnapshot], pauli_string: str, n_batches: int = 5, **kwargs
) -> float:
    """Median of per-batch means: robust to heavy tails at small budgets."""
    s_count = len(snapshots)
    if s_count < n_batches or n_batches <= 1:
        return estimate_pauli_string_expectation(snapshots, pauli_string, **kwargs)
    size = s_count // n_batches
    means = [estimate_pauli_string_expectation(snapshots[b * size:(b + 1) * size], pauli_string, **kwargs) for b in range(n_batches)]
    return float(np.median(means))


def estimate_many_observables(
    snapshots: Sequence[ShadowSnapshot],
    observables: Sequence[str],
    use_median_of_means: bool = False,
    n_batches: int = 5,
    derandomized: bool = False,
    readout_model=None,
) -> Dict[str, float]:
    kw = {"derandomized": derandomized, "readout_model": readout_model}
    out = {}
    for obs in observables:
        out[obs] = (estimate_pauli_expectation_median_of_means(snapshots, obs, n_batches=n_batches, **kw)
                    if use_median_of_means else estimate_pauli_string_expectation(snapshots, obs, **kw))
    return out


class ShadowObservableEstimator:
    """Estimation engine over a fixed shadow data set."""

    _PROJ = {
        "X": [np.array([[0.5, 0.5], [0.5, 0.5]]), np.array([[0.5, -0.5], [-0.5, 0.5]])],
        "Y": [np.array([[0.5, -0.5j], [0.5j, 0.5]]), np.array([[0.5, 0.5j], [-0.5j, 0.5]])],
        "Z": [np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([[0.0, 0.0], [0.0, 1.0]])],
    }

    def __init__(self, snapshots: Sequence[ShadowSnapshot]):
        self.snapshots = list(snapshots)
        self.n_snapshots = len(self.snapshots)
        self.n_qubits = len(self.snapshots[0].bases) if self.snapshots else 0

    def predict(self, observables: Sequence[str], **kwargs) -> Dict[str, float]:
        return estimate_many_observables(self.snapshots, observables, **kwargs)

    def reconstruct_reduced_density_matrix(self, qubits: Sequence[int]) -> np.ndarray:
        """ρ̂ on ``qubits`` as the average of ⊗_i (3|s_i⟩⟨s_i| − 1)."""
        k = len(qubits)
        rho = np.zeros((2**k, 2**k), dtype=np.complex128)
        id2 = np.eye(2)
        for snap in self.snapshots:
            local = None
            for q in qubits:
                inv = 3.0 * self._PROJ[snap.bases[q]][snap.bits[q]] - id2
                local = inv if local is None else np.kron(local, inv)
            rho += local
        return rho / max(self.n_snapshots, 1)
