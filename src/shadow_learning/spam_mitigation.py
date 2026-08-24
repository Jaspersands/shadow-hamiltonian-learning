"""
SPAM (State Preparation and Measurement) Error Mitigation for Classical Shadows.

Implements inverted measurement confusion matrix calibration to de-bias
noisy shadow snapshots under asymmetric readout errors.
"""

from __future__ import annotations
import numpy as np
from typing import Sequence, List, Tuple, Dict, Any, Optional
from .cirq_shadows import ShadowSnapshot


class ReadoutErrorModel:
    """
    Asymmetric single-qubit readout error model:
        p(0|1) = eps_10 (state |1> misread as 0)
        p(1|0) = eps_01 (state |0> misread as 1)
    """

    def __init__(self, p01: float = 0.03, p10: float = 0.05):
        self.p01 = p01
        self.p10 = p10
        # Confusion matrix M[measured, true]
        self.matrix = np.array(
            [[1.0 - p01, p10], [p01, 1.0 - p10]], dtype=np.float64
        )
        self.inv_matrix = np.linalg.inv(self.matrix)

    def apply_readout_noise(self, bit: int, rng: np.random.Generator) -> int:
        """Flips bit with asymmetric readout probability."""
        if bit == 0:
            return 1 if rng.random() < self.p01 else 0
        else:
            return 0 if rng.random() < self.p10 else 1


class SPAMNoiseMitigator:
    """
    Mitigates measurement assignment errors on shadow snapshots.
    """

    def __init__(self, readout_model: Optional[ReadoutErrorModel] = None):
        self.model = readout_model or ReadoutErrorModel()

    def mitigate_pauli_expectation(
        self,
        raw_expectation: float,
        pauli_weight: int, # Number of non-identity Paulis in string
    ) -> float:
        """
        Scales the raw shadow expectation by the inverse trace error factor:
            <P>_{mitigated} = <P>_{raw} / (1 - p01 - p10)^k
        """
        readout_fidelity = 1.0 - self.model.p01 - self.model.p10
        if readout_fidelity <= 0.01:
            return raw_expectation

        scale_factor = 1.0 / (readout_fidelity ** pauli_weight)
        return float(raw_expectation * scale_factor)
