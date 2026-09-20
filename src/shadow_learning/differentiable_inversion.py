"""
Differentiable Hamiltonian learning from shadow expectation values.

Model: an isotropic Heisenberg Hamiltonian with local fields,

    H(J, h) = Σ_{i<j} J_ij (X_iX_j + Y_iY_j + Z_iZ_j) + Σ_i h_i Z_i,

whose Gibbs state ρ_β = e^{−βH}/Z produces expectation values of the Pauli
observables {Z_i, X_iX_j, Y_iY_j, Z_iZ_j}. Given shadow estimates of those
observables, (J, h) is recovered by minimising

    L(J, h) = Σ_o (⟨o⟩_ρ(J,h) − ô)² + λ ‖(J, h)‖²

with **analytic gradients through the eigendecomposition** (JAX ``eigh``) and
L-BFGS-B. Without JAX the same loss is minimised with central-difference
gradients. (v0.2 used derivative-free Nelder–Mead.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.optimize

from .cirq_shadows import ShadowSnapshot, gibbs_state
from .reconstruction import estimate_pauli_string_expectation

try:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    HAS_JAX = True
except ImportError:  # pragma: no cover
    HAS_JAX = False
    jax = None
    jnp = np

_P = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


@dataclass
class HamiltonianLearningResult:
    recovered_j_matrix: np.ndarray
    recovered_h_vector: np.ndarray
    true_j_matrix: Optional[np.ndarray]
    true_h_vector: Optional[np.ndarray]
    frobenius_error: float
    loss_history: List[float]
    iterations: int
    matched_observables: Dict[str, float]
    predicted_observables: Dict[str, float]
    method: str = "jax-lbfgs"
    h_error: float = 0.0

    @property
    def final_loss(self) -> float:
        return self.loss_history[-1] if self.loss_history else 0.0


class DifferentiableHamiltonianLearner:
    """
    Parameters
    ----------
    n_qubits : register size
    beta : inverse temperature of the Gibbs state
    l2_reg : ridge penalty λ on the parameter vector
    seed : initial-guess RNG seed
    """

    def __init__(self, n_qubits: int, beta: float = 1.0, l2_reg: float = 1e-8, seed: int = 0, l1_reg: Optional[float] = None):
        self.n_qubits = int(n_qubits)
        self.beta = float(beta)
        self.l2_reg = float(l1_reg if l1_reg is not None else l2_reg)
        self.rng = np.random.default_rng(seed)
        self.pairs = [(i, j) for i in range(self.n_qubits) for j in range(i + 1, self.n_qubits)]
        self.n_params = len(self.pairs) + self.n_qubits
        self.observable_names: List[str] = [f"Z_{i}" for i in range(self.n_qubits)]
        for i, j in self.pairs:
            self.observable_names += [f"XX_{i}_{j}", f"YY_{i}_{j}", f"ZZ_{i}_{j}"]
        self._obs_mats = np.stack([self.pauli_matrix(self.pauli_string(n)) for n in self.observable_names])
        self._coupling_mats = np.stack([sum(self.pauli_matrix(self._pair_string(i, j, p)) for p in "XYZ") for i, j in self.pairs]) if self.pairs else np.zeros((0, 2**self.n_qubits, 2**self.n_qubits), dtype=complex)
        self._field_mats = np.stack([self.pauli_matrix(self.pauli_string(f"Z_{i}")) for i in range(self.n_qubits)])
        self.pauli_basis_matrices = {n: self._obs_mats[k] for k, n in enumerate(self.observable_names)}  # v0.2 name
        self._jax_value_and_grad = self._build_jax() if HAS_JAX else None

    # ----------------------------------------------------------------- #
    # naming / matrices
    # ----------------------------------------------------------------- #
    def _pair_string(self, i: int, j: int, p: str) -> str:
        s = ["I"] * self.n_qubits
        s[i] = p; s[j] = p
        return "".join(s)

    def pauli_string(self, name: str) -> str:
        if name.startswith("Z_"):
            i = int(name.split("_")[1])
            s = ["I"] * self.n_qubits; s[i] = "Z"
            return "".join(s)
        p, i, j = name.split("_")
        return self._pair_string(int(i), int(j), p[0])

    def pauli_matrix(self, pauli_string: str) -> np.ndarray:
        return reduce(np.kron, [_P[c] for c in pauli_string])

    def pack(self, j_matrix: np.ndarray, h_vec: np.ndarray) -> np.ndarray:
        return np.concatenate([[float(j_matrix[i, j]) for i, j in self.pairs], np.asarray(h_vec, dtype=float)])

    def unpack(self, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        J = np.zeros((self.n_qubits, self.n_qubits))
        for k, (i, j) in enumerate(self.pairs):
            J[i, j] = J[j, i] = p[k]
        return J, np.asarray(p[len(self.pairs):], dtype=float)

    def build_hamiltonian_matrix(self, j_matrix: np.ndarray, h_vec: np.ndarray) -> np.ndarray:
        p = self.pack(j_matrix, h_vec)
        H = np.zeros((2**self.n_qubits, 2**self.n_qubits), dtype=np.complex128)
        for k in range(len(self.pairs)):
            H += p[k] * self._coupling_mats[k]
        for i in range(self.n_qubits):
            H += p[len(self.pairs) + i] * self._field_mats[i]
        return H

    # ----------------------------------------------------------------- #
    # forward model
    # ----------------------------------------------------------------- #
    def gibbs_observables(self, p: np.ndarray) -> np.ndarray:
        J, h = self.unpack(p)
        rho = gibbs_state(self.build_hamiltonian_matrix(J, h), self.beta)
        return np.real(np.einsum("kij,ji->k", self._obs_mats, rho))

    def jacobian(self, p: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """∂⟨o_k⟩/∂p_l of the Gibbs observables (central differences)."""
        p = np.asarray(p, dtype=float)
        J = np.zeros((len(self.observable_names), len(p)))
        for l in range(len(p)):
            pp = p.copy(); pp[l] += eps
            pm = p.copy(); pm[l] -= eps
            J[:, l] = (self.gibbs_observables(pp) - self.gibbs_observables(pm)) / (2 * eps)
        return J

    def identifiability(self, j_matrix: np.ndarray, h_vec: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Singular-value analysis of the observable Jacobian at (J, h): which
        parameter directions the chosen observables can resolve. Small singular
        values mark (nearly) unlearnable combinations — e.g. a uniform field on a
        singlet-dominated Gibbs state, which the total-spin-zero state screens.
        """
        Jac = self.jacobian(self.pack(j_matrix, h_vec))
        u, sv, vt = np.linalg.svd(Jac, full_matrices=False)
        return {"singular_values": sv, "directions": vt, "least_identifiable": vt[-1], "condition_number": float(sv[0] / max(sv[-1], 1e-300))}

    def compute_thermal_observables(self, j_matrix: np.ndarray, h_vec: np.ndarray) -> Dict[str, float]:
        vals = self.gibbs_observables(self.pack(j_matrix, h_vec))
        return {n: float(v) for n, v in zip(self.observable_names, vals)}

    def _build_jax(self):
        coup = jnp.asarray(self._coupling_mats)
        fields = jnp.asarray(self._field_mats)
        obs = jnp.asarray(self._obs_mats)
        beta, n_pairs, lam = self.beta, len(self.pairs), self.l2_reg

        def loss(p, targets):
            H = jnp.tensordot(p[:n_pairs], coup, axes=(0, 0)) + jnp.tensordot(p[n_pairs:], fields, axes=(0, 0))
            w, v = jnp.linalg.eigh(H)
            weights = jnp.exp(-beta * (w - jnp.min(w)))
            weights = weights / jnp.sum(weights)
            rho = (v * weights) @ jnp.conj(v).T
            preds = jnp.real(jnp.einsum("kij,ji->k", obs, rho))
            return jnp.sum((preds - targets) ** 2) + lam * jnp.sum(p**2)

        return jax.jit(jax.value_and_grad(loss))

    def loss_and_grad(self, p: np.ndarray, targets) -> Tuple[float, np.ndarray]:
        t = np.asarray([targets[n] for n in self.observable_names], dtype=float) if isinstance(targets, dict) else np.asarray(targets, dtype=float)
        p = np.asarray(p, dtype=float)
        if HAS_JAX and self._jax_value_and_grad is not None:
            v, g = self._jax_value_and_grad(jnp.asarray(p), jnp.asarray(t))
            return float(v), np.asarray(g, dtype=float)

        def f(q):
            return float(np.sum((self.gibbs_observables(q) - t) ** 2) + self.l2_reg * np.sum(q**2))

        base = f(p)
        g = np.zeros_like(p)
        for i in range(len(p)):
            pp = p.copy(); pp[i] += 1e-6
            pm = p.copy(); pm[i] -= 1e-6
            g[i] = (f(pp) - f(pm)) / 2e-6
        return base, g

    # ----------------------------------------------------------------- #
    # learning
    # ----------------------------------------------------------------- #
    def learn_from_expectations(
        self,
        targets: Dict[str, float],
        true_j_matrix: Optional[np.ndarray] = None,
        true_h_vector: Optional[np.ndarray] = None,
        max_iter: int = 300,
        n_restarts: int = 3,
    ) -> HamiltonianLearningResult:
        """Recover (J, h) from a dict of observable expectation values."""
        t = np.asarray([targets[n] for n in self.observable_names], dtype=float)
        best = None
        for r in range(int(n_restarts)):
            p0 = self.rng.uniform(-0.5, 0.5, size=self.n_params) if r > 0 else np.zeros(self.n_params) + 0.05
            hist: List[float] = []

            def objective(p):
                v, g = self.loss_and_grad(p, t)
                hist.append(v)
                return v, g

            res = scipy.optimize.minimize(objective, p0, method="L-BFGS-B", jac=True,
                                          options={"maxiter": int(max_iter), "ftol": 1e-15, "gtol": 1e-12})
            if best is None or res.fun < best[0].fun:
                best = (res, hist)
            if res.fun < 1e-12:
                break
        res, hist = best
        J, h = self.unpack(res.x)
        preds = self.compute_thermal_observables(J, h)
        return HamiltonianLearningResult(
            recovered_j_matrix=J, recovered_h_vector=h, true_j_matrix=true_j_matrix, true_h_vector=true_h_vector,
            frobenius_error=float(np.linalg.norm(J - true_j_matrix)) if true_j_matrix is not None else 0.0,
            h_error=float(np.linalg.norm(h - true_h_vector)) if true_h_vector is not None else 0.0,
            loss_history=hist, iterations=len(hist), matched_observables=dict(zip(self.observable_names, t.tolist())),
            predicted_observables=preds, method="jax-lbfgs" if (HAS_JAX and self._jax_value_and_grad is not None) else "numpy-lbfgs",
        )

    def learn_from_shadows(
        self,
        snapshots: Sequence[ShadowSnapshot],
        true_j_matrix: Optional[np.ndarray] = None,
        true_h_vector: Optional[np.ndarray] = None,
        max_iter: int = 300,
        derandomized: bool = False,
        readout_model=None,
    ) -> HamiltonianLearningResult:
        """Estimate the model observables from shadows, then invert."""
        targets = {n: estimate_pauli_string_expectation(snapshots, self.pauli_string(n), derandomized=derandomized, readout_model=readout_model)
                   for n in self.observable_names}
        targets = {n: float(np.clip(v, -1.0, 1.0)) for n, v in targets.items()}
        return self.learn_from_expectations(targets, true_j_matrix, true_h_vector, max_iter=max_iter)


def learn_hamiltonian_from_shadows(
    snapshots: Sequence[ShadowSnapshot],
    n_qubits: int,
    true_j_matrix: Optional[np.ndarray] = None,
    true_h_vector: Optional[np.ndarray] = None,
    beta: float = 1.0,
    max_iter: int = 300,
    seed: int = 0,
    derandomized: bool = False,
    readout_model=None,
) -> HamiltonianLearningResult:
    learner = DifferentiableHamiltonianLearner(n_qubits=n_qubits, beta=beta, seed=seed)
    return learner.learn_from_shadows(snapshots, true_j_matrix, true_h_vector, max_iter=max_iter,
                                      derandomized=derandomized, readout_model=readout_model)
