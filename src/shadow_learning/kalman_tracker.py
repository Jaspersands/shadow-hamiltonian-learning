"""
Streaming extended Kalman filter over Hamiltonian parameters.

State x = (J_ij…, h_i…) follows a random walk (process noise σ_p per snapshot).
Every incoming Pauli snapshot yields, for each model observable o whose
support is compatible with the snapshot's bases, one ±1 outcome
z_o = ∏(−1)^{b_i} with E[z_o] = h_o(x) = Tr(o ρ_β(x)).

Observation model
-----------------
The outcomes are discrete, so the filter is a moment-matched (Gaussian
assumed-density) approximation rather than an exact Bayesian update: each
snapshot is treated as one vector observation z ∈ {±1}^m with the exact first
two moments under ρ_β(x),

    E[z_a] = ⟨o_a⟩,     Cov(z_a, z_b) = ⟨o_a o_b⟩ − ⟨o_a⟩⟨o_b⟩,

where o_a o_b is again a Pauli string (the outcomes of one snapshot share bits
and are correlated; treating them as independent would over-count
information). h is linearised with the exact Gibbs Jacobian and the covariance
is updated in Joseph form, P ← (I − KH) P (I − KH)ᵀ + K R Kᵀ, which keeps P
symmetric positive semi-definite in floating point. The approximation is good
once many snapshots have been absorbed (the posterior is then close to
Gaussian by the central limit theorem); the tests check calibration of the
reported standard deviations against the actual error.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .cirq_shadows import ShadowSnapshot
from .differentiable_inversion import DifferentiableHamiltonianLearner


class StreamingKalmanHamiltonianTracker:
    def __init__(
        self,
        n_qubits: int,
        beta: float = 1.0,
        initial_params: Optional[np.ndarray] = None,
        process_noise_std: float = 0.01,
        prior_std: float = 0.5,
        min_measurement_var: float = 1e-3,
        seed: int = 0,
    ):
        self.learner = DifferentiableHamiltonianLearner(n_qubits=n_qubits, beta=beta, seed=seed)
        self.n_params = self.learner.n_params
        self.x = np.zeros(self.n_params) if initial_params is None else np.asarray(initial_params, dtype=float).copy()
        self.P = np.eye(self.n_params) * float(prior_std) ** 2
        self.Q = np.eye(self.n_params) * float(process_noise_std) ** 2
        self.min_r = float(min_measurement_var)
        self.names = [f"J_{i}{j}" for i, j in self.learner.pairs] + [f"h_{i}" for i in range(n_qubits)]
        self._hist: List[np.ndarray] = []
        self.n_updates = 0
        self._obs_strings = {n: self.learner.pauli_string(n) for n in self.learner.observable_names}

    # ----------------------------------------------------------------- #
    def predicted_observables(self, x: Optional[np.ndarray] = None) -> np.ndarray:
        return self.learner.gibbs_observables(self.x if x is None else x)

    def _jacobian(self) -> np.ndarray:
        return self.learner.gibbs_observables_and_jacobian(self.x)[1]

    def _outcome_covariance(self, idx: List[int], preds: np.ndarray) -> np.ndarray:
        """Cov(z_a, z_b) = ⟨o_a o_b⟩ − ⟨o_a⟩⟨o_b⟩ in the current Gibbs state."""
        rho = self._rho()
        mats = self.learner._obs_mats
        m = len(idx)
        R = np.empty((m, m))
        for a in range(m):
            for b in range(a, m):
                second = float(np.real(np.trace(mats[idx[a]] @ mats[idx[b]] @ rho)))
                R[a, b] = R[b, a] = second - preds[idx[a]] * preds[idx[b]]
        return R + self.min_r * np.eye(m)

    def _rho(self) -> np.ndarray:
        w, V = np.linalg.eigh(self.learner.hamiltonian_from_params(self.x))
        e = np.exp(-self.learner.beta * (w - w[0]))
        return (V * (e / e.sum())) @ V.conj().T

    def process_snapshot(self, snapshot: ShadowSnapshot) -> Dict[str, np.ndarray]:
        """Predict, then one joint update with every observable this snapshot measures."""
        self.P = self.P + self.Q
        idx, z = [], []
        for k, name in enumerate(self.learner.observable_names):
            s = self._obs_strings[name]
            if all(c == "I" or c == b for c, b in zip(s, snapshot.bases)):
                sign = 1.0
                for c, bit in zip(s, snapshot.bits):
                    if c != "I" and bit:
                        sign = -sign
                idx.append(k)
                z.append(sign)
        if idx:
            preds, Jac = self.learner.gibbs_observables_and_jacobian(self.x)
            Hm = Jac[idx]                                   # (m, n_params)
            R = self._outcome_covariance(idx, preds)
            S = Hm @ self.P @ Hm.T + R
            K = np.linalg.solve(S, Hm @ self.P).T           # P Hᵀ S⁻¹ (S, P symmetric)
            self.x = self.x + K @ (np.asarray(z) - preds[idx])
            A = np.eye(self.n_params) - K @ Hm
            self.P = A @ self.P @ A.T + K @ R @ K.T          # Joseph form
            self.P = 0.5 * (self.P + self.P.T)
            self.n_updates += len(idx)
        self._hist.append(self.x.copy())
        return self.estimate()

    def process_shadow_snapshot(self, observable_name: str, single_shot_realization: float, coupling_idx: int = 0):
        """v0.2-compatible shim: treat ``single_shot_realization`` as a ±1 outcome of the named observable."""
        s = self._obs_strings.get(observable_name)
        if s is None:
            s = self.learner.pauli_string(observable_name) if "_" in observable_name else observable_name
        bases = tuple(c if c != "I" else "Z" for c in s)
        bits = tuple(1 if (c != "I" and single_shot_realization < 0) else 0 for c in s)
        return self.process_snapshot(ShadowSnapshot(bases=bases, bits=bits))["params"]

    def track(self, snapshots) -> Dict[str, np.ndarray]:
        for s in snapshots:
            self.process_snapshot(s)
        return self.history()

    def estimate(self) -> Dict[str, np.ndarray]:
        J, h = self.learner.unpack(self.x)
        return {"J": J, "h": h, "params": self.x.copy(), "std": np.sqrt(np.diag(self.P))}

    def history(self) -> Dict[str, np.ndarray]:
        arr = np.array(self._hist) if self._hist else np.zeros((0, self.n_params))
        return {name: arr[:, k] for k, name in enumerate(self.names)}
