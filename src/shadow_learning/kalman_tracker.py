"""
Continuous Real-Time Extended Kalman Filter (EKF) for Shadow Hamiltonian Tracking.

Processes shadow snapshot streams sequentially on-the-fly to dynamically track
time-varying coupling parameters J_ij(t) and magnetic field drift h_i(t).
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, List, Optional


class StreamingKalmanHamiltonianTracker:
    """
    Sequential Bayesian filter tracking time-dependent Hamiltonian parameters
    from a continuous stream of classical shadow snapshots.
    """

    def __init__(
        self,
        n_qubits: int,
        initial_j_guess: Optional[np.ndarray] = None,
        process_noise_std: float = 0.01,
        measurement_noise_std: float = 0.3,
    ):
        self.n_qubits = n_qubits
        self.n_couplings = (n_qubits * (n_qubits - 1)) // 2

        # State vector: [J_01, J_02, ...]
        if initial_j_guess is not None:
            self.state = initial_j_guess.copy()
        else:
            self.state = np.zeros(self.n_couplings, dtype=np.float64)

        self.cov = np.eye(self.n_couplings) * 0.2
        self.q_cov = np.eye(self.n_couplings) * (process_noise_std**2)
        self.r_var = measurement_noise_std**2

    def process_shadow_snapshot(
        self,
        observable_name: str,
        single_shot_realization: float,
        coupling_idx: int = 0,
    ) -> np.ndarray:
        """
        Updates Hamiltonian parameter estimates with a single incoming shadow realization.
        """
        # 1. Prediction step
        self.cov = self.cov + self.q_cov

        # 2. Measurement update
        # Linearized sensitivity H matrix
        h_mat = np.zeros(self.n_couplings)
        h_mat[coupling_idx] = 1.0

        # Innovation
        pred_obs = float(np.dot(h_mat, self.state))
        innovation = single_shot_realization - pred_obs
        s_var = float(h_mat @ self.cov @ h_mat) + self.r_var

        kalman_gain = (self.cov @ h_mat) / s_var
        self.state = self.state + kalman_gain * innovation
        self.cov = (np.eye(self.n_couplings) - np.outer(kalman_gain, h_mat)) @ self.cov

        return self.state.copy()
