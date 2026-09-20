"""
Readout (measurement) error mitigation for Pauli shadows.

A single-qubit confusion matrix M[measured, true] with
p(1|0) = p01 and p(0|1) = p10 maps the true outcome distribution to the
measured one. Because each Pauli estimator is a product of single-qubit
eigenvalues (−1)^{b}, exact de-biasing only requires replacing (−1)^{b} by

    g = (Mᵀ)⁻¹ · (+1, −1),

i.e. g(b') = Σ_b (M⁻¹)[b, b'] (−1)^b, which satisfies E[g(b')] = (−1)^b for every
true bit b — including *asymmetric* errors. The multiplicative rule
⟨P⟩/(1 − p01 − p10)^k used in v0.2 is exact only for symmetric errors
(p01 = p10) and is retained for that case.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


class ReadoutErrorModel:
    """Asymmetric single-qubit readout errors p01 = p(1|0), p10 = p(0|1)."""

    def __init__(self, p01: float = 0.03, p10: float = 0.05):
        self.p01 = float(p01)
        self.p10 = float(p10)
        # M[measured, true]
        self.matrix = np.array([[1.0 - self.p01, self.p10], [self.p01, 1.0 - self.p10]], dtype=float)
        det = 1.0 - self.p01 - self.p10
        self.singular = abs(det) < 1e-4
        self.inv_matrix = np.eye(2) if self.singular else np.linalg.inv(self.matrix)
        # corrected eigenvalues g(b') = Σ_b inv[b, b'] (−1)^b
        self._g = (self.inv_matrix.T @ np.array([1.0, -1.0])) if not self.singular else np.array([1.0, -1.0])

    @property
    def readout_fidelity(self) -> float:
        return 1.0 - self.p01 - self.p10

    def corrected_eigenvalue(self, measured_bit: int) -> float:
        """Unbiased estimator of (−1)^{true bit} given the measured bit."""
        return float(self._g[int(measured_bit)])

    def apply_readout_noise(self, bit: int, rng: np.random.Generator) -> int:
        if bit == 0:
            return 1 if rng.random() < self.p01 else 0
        return 0 if rng.random() < self.p10 else 1


class SPAMNoiseMitigator:
    """Applies :class:`ReadoutErrorModel` corrections to shadow estimates."""

    def __init__(self, readout_model: Optional[ReadoutErrorModel] = None):
        self.model = readout_model or ReadoutErrorModel()

    def estimate(self, snapshots: Sequence, pauli_string: str, derandomized: bool = False) -> float:
        """Exact (asymmetric-safe) mitigated estimate of ⟨P⟩ at the single-shot level."""
        from .reconstruction import estimate_pauli_string_expectation

        return estimate_pauli_string_expectation(snapshots, pauli_string, derandomized=derandomized, readout_model=self.model)

    def mitigate_pauli_expectation(self, raw_expectation: float, pauli_weight: int) -> float:
        """
        Multiplicative correction ⟨P⟩ / (1 − p01 − p10)^k — exact for symmetric
        readout errors; for asymmetric errors use :meth:`estimate`.
        """
        f = self.model.readout_fidelity
        if f <= 0.01:
            return float(raw_expectation)
        return float(raw_expectation / f**pauli_weight)
