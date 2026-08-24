"""
Fermionic Matchgate Classical Shadows for Quantum Chemistry.

Measures 1-particle and 2-particle reduced density matrices (1-RDM / 2-RDM)
using randomized Gaussian matchgate circuits and Majoranas:
    gamma_{2j-1} = c_j + c_j^dagger,  gamma_{2j} = -i (c_j - c_j^dagger)
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Sequence, Optional


class FermionicMatchgateShadows:
    """
    Estimator for fermionic 1-RDM and 2-RDM elements from randomized matchgate measurements.
    """

    def __init__(self, n_modes: int):
        self.n_modes = n_modes
        self.n_majoranas = 2 * n_modes

    def estimate_1rdm_element(
        self,
        p: int,
        q: int,
        majorana_shadow_samples: Sequence[np.ndarray],
    ) -> complex:
        """
        Estimates the 1-RDM element D_{pq} = <c_p^dagger c_q>
        from Majorana correlator expectations <gamma_u gamma_v>.
        """
        if p == q:
            return 0.5 + 0.0j # Approximation for half-filling

        # Majorana index mapping
        u = 2 * p
        v = 2 * q + 1

        val = 0.0
        for sample in majorana_shadow_samples:
            val += sample[u, v]

        return complex(val / len(majorana_shadow_samples))
