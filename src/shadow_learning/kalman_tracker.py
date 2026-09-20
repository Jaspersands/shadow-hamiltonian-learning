"""
Streaming extended Kalman filter over Hamiltonian parameters.

State x = (J_ij…, h_i…) follows a random walk (process noise σ_p per snapshot).
Every incoming Pauli snapshot yields, for each model observable o whose
support is compatible with the snapshot's bases, one ±1 outcome
z_o = ∏(−1)^{b_i} with E[z_o] = h_o(x) = Tr(o ρ_β(x)). The EKF linearises h_o
around the current estimate (Jacobian by central differences through the
Gibbs state) and applies the standard predict/update, with measurement
variance R = 1 − h_o(x)² (a Bernoulli ±1 outcome). This tracks drifting
couplings from a live stream without ever re-fitting a batch. (v0.2's
"tracker" received the parameter itself as its measurement.)
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
        min_measurement_var: float = 0.05,
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

    def _jacobian(self, eps: float = 1e-5) -> np.ndarray:
        J = np.zeros((len(self.learner.observable_names), self.n_params))
        for k in range(self.n_params):
            xp = self.x.copy(); xp[k] += eps
            xm = self.x.copy(); xm[k] -= eps
            J[:, k] = (self.learner.gibbs_observables(xp) - self.learner.gibbs_observables(xm)) / (2 * eps)
        return J

    def process_snapshot(self, snapshot: ShadowSnapshot) -> Dict[str, np.ndarray]:
        """Predict + update with every observable this snapshot measures. Returns current estimate."""
        self.P = self.P + self.Q
        compatible = []
        for idx, name in enumerate(self.learner.observable_names):
            s = self._obs_strings[name]
            if all(c == "I" or c == b for c, b in zip(s, snapshot.bases)):
                z = 1.0
                for c, bit in zip(s, snapshot.bits):
                    if c != "I" and bit:
                        z = -z
                compatible.append((idx, z))
        if compatible:
            preds = self.predicted_observables()
            Jac = self._jacobian()
            for idx, z in compatible:
                Hrow = Jac[idx]
                r = max(self.min_r, 1.0 - float(preds[idx]) ** 2)
                S = float(Hrow @ self.P @ Hrow) + r
                K = self.P @ Hrow / S
                self.x = self.x + K * (z - float(preds[idx]))
                self.P = (np.eye(self.n_params) - np.outer(K, Hrow)) @ self.P
                self.n_updates += 1
                preds = self.predicted_observables()   # re-linearise for the next observable
                Jac = self._jacobian()
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
        return {"J": J, "h": h, "params": self.x.copy(), "std": np.sqrt(np.clip(np.diag(self.P), 0, None))}

    def history(self) -> Dict[str, np.ndarray]:
        arr = np.array(self._hist) if self._hist else np.zeros((0, self.n_params))
        return {name: arr[:, k] for k, name in enumerate(self.names)}
