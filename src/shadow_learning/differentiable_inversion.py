"""
Differentiable Hamiltonian learning from shadow expectation values.

Model: an isotropic Heisenberg Hamiltonian with local fields,

    H(J, h) = Σ_{i<j} J_ij (X_iX_j + Y_iY_j + Z_iZ_j) + Σ_i h_i Z_i,

whose Gibbs state ρ_β = e^{−βH}/Z produces expectation values of the Pauli
observables {Z_i, X_iX_j, Y_iY_j, Z_iZ_j}. Given shadow estimates of those
observables, (J, h) is recovered by minimising

    L(J, h) = Σ_o (⟨o⟩_ρ(J,h) − ô)² + λ ‖(J, h)‖²

with L-BFGS-B and exact gradients.

Gradients
---------
Heisenberg spectra are degenerate (SU(2) multiplets, and further accidental
degeneracies at symmetric couplings), so differentiating an eigendecomposition
is not an option: the eigenvector derivative contains 1/(w_i − w_j) and
returns NaN exactly where the model is most symmetric. Two degeneracy-safe
routes are provided and agree to ~1e-10:

* ``backend="analytic"`` (default) — the Fréchet derivative of e^{−βH} in the
  eigenbasis, [V†·d(e^{−βH})·V]_ij = f[w_i, w_j]·[V†·dH·V]_ij, with the divided
  difference f[w_i, w_j] = (e^{−βw_i} − e^{−βw_j})/(w_i − w_j) replaced by its
  limit −β e^{−βw_i} when w_i ≈ w_j. This yields the full observable Jacobian
  in one pass.
* ``backend="jax"`` — reverse-mode autodiff through ``jax.scipy.linalg.expm``
  (the spectral shift that prevents overflow is held outside the gradient).
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
    method: str = "analytic-lbfgs"
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
        self._jax_value_and_grad = None
        # Pauli-basis vectors of the generators ∂H/∂p_l, used by the analytic Jacobian
        self._generators = np.concatenate([self._coupling_mats, self._field_mats], axis=0)

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
    def hamiltonian_from_params(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        return np.tensordot(p, self._generators, axes=(0, 0))

    def gibbs_observables(self, p: np.ndarray) -> np.ndarray:
        """⟨o_k⟩ in the Gibbs state of H(p), for every model observable."""
        w, V = np.linalg.eigh(self.hamiltonian_from_params(p))
        e = np.exp(-self.beta * (w - w[0]))
        rho = (V * (e / e.sum())) @ V.conj().T
        return np.real(np.einsum("kij,ji->k", self._obs_mats, rho))

    def gibbs_observables_and_jacobian(self, p: np.ndarray, degeneracy_tol: float = 1e-9) -> Tuple[np.ndarray, np.ndarray]:
        """
        Observables ⟨o_k⟩ and the exact Jacobian ∂⟨o_k⟩/∂p_l (see module docstring).
        Well defined at degenerate spectra.
        """
        w, V = np.linalg.eigh(self.hamiltonian_from_params(p))
        b = self.beta
        e = np.exp(-b * (w - w[0]))
        Z = e.sum()
        dw = w[:, None] - w[None, :]
        de = e[:, None] - e[None, :]
        close = np.abs(dw) < degeneracy_tol
        D = np.where(close, -b * 0.5 * (e[:, None] + e[None, :]), de / np.where(close, 1.0, dw))
        Vh = V.conj().T
        O_t = np.einsum("ai,kab,bj->kij", V.conj(), self._obs_mats, V, optimize=True)      # V† O V
        G_t = np.einsum("ai,lab,bj->lij", V.conj(), self._generators, V, optimize=True)    # V† ∂H V
        obs = np.real(np.einsum("kii,i->k", O_t, e)) / Z
        dE = D[None] * G_t                                     # V† d(e^{−βH}) V, per parameter
        dtrOE = np.real(np.einsum("kji,lij->kl", O_t, dE, optimize=True))
        dZ = np.real(np.einsum("lii->l", dE))
        jac = (dtrOE - obs[:, None] * dZ[None, :]) / Z
        return obs, jac

    def jacobian(self, p: np.ndarray, eps: Optional[float] = None) -> np.ndarray:
        """∂⟨o_k⟩/∂p_l of the Gibbs observables (exact; ``eps`` is ignored, kept for v0.2 callers)."""
        return self.gibbs_observables_and_jacobian(p)[1]

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
        from jax.scipy.linalg import expm

        gens = jnp.asarray(self._generators)
        obs = jnp.asarray(self._obs_mats)
        beta, lam = self.beta, self.l2_reg
        eye = jnp.eye(gens.shape[1])

        def loss(p, targets):
            H = jnp.tensordot(p, gens, axes=(0, 0))
            shift = jax.lax.stop_gradient(jnp.linalg.eigvalsh(H)[0])   # overflow guard; cancels in ρ
            E = expm(-beta * (H - shift * eye))
            rho = E / jnp.real(jnp.trace(E))
            preds = jnp.real(jnp.einsum("kij,ji->k", obs, rho))
            return jnp.sum((preds - targets) ** 2) + lam * jnp.sum(p**2)

        return jax.jit(jax.value_and_grad(loss))

    def loss_and_grad(self, p: np.ndarray, targets, backend: str = "analytic") -> Tuple[float, np.ndarray]:
        """Loss and its exact gradient. ``backend`` is ``"analytic"`` or ``"jax"``."""
        t = np.asarray([targets[n] for n in self.observable_names], dtype=float) if isinstance(targets, dict) else np.asarray(targets, dtype=float)
        p = np.asarray(p, dtype=float)
        if backend == "jax":
            if not HAS_JAX:
                raise ImportError("backend='jax' needs JAX installed")
            if self._jax_value_and_grad is None:
                self._jax_value_and_grad = self._build_jax()
            v, g = self._jax_value_and_grad(jnp.asarray(p), jnp.asarray(t))
            return float(v), np.asarray(g, dtype=float)
        if backend != "analytic":
            raise ValueError(f"unknown backend {backend!r}")
        obs, jac = self.gibbs_observables_and_jacobian(p)
        r = obs - t
        return float(r @ r + self.l2_reg * p @ p), 2.0 * (jac.T @ r) + 2.0 * self.l2_reg * p

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
        backend: str = "analytic",
    ) -> HamiltonianLearningResult:
        """Recover (J, h) from a dict of observable expectation values."""
        t = np.asarray([targets[n] for n in self.observable_names], dtype=float)
        best = None
        for r in range(int(n_restarts)):
            # restart 0: a small generic point; later restarts: uniform in [−0.5, 0.5]
            p0 = self.rng.uniform(-0.5, 0.5, size=self.n_params) if r > 0 else 0.1 + 0.01 * self.rng.standard_normal(self.n_params)
            hist: List[float] = []

            def objective(p):
                if not np.all(np.isfinite(p)):
                    return np.inf, np.zeros_like(p)
                try:
                    v, g = self.loss_and_grad(p, t, backend=backend)
                except np.linalg.LinAlgError:
                    return np.inf, np.zeros_like(p)
                if not (np.isfinite(v) and np.all(np.isfinite(g))):
                    return np.inf, np.zeros_like(p)
                hist.append(v)
                return v, g

            res = scipy.optimize.minimize(objective, p0, method="L-BFGS-B", jac=True,
                                          options={"maxiter": int(max_iter), "ftol": 1e-15, "gtol": 1e-12})
            if not (np.isfinite(res.fun) and np.all(np.isfinite(res.x))):
                continue                      # never let a non-finite run win (NaN < x is always False)
            if best is None or res.fun < best[0].fun:
                best = (res, hist)
            if res.fun < 1e-12:
                break
        if best is None:
            raise RuntimeError("every restart produced a non-finite loss; check the targets and β")
        res, hist = best
        J, h = self.unpack(res.x)
        preds = self.compute_thermal_observables(J, h)
        return HamiltonianLearningResult(
            recovered_j_matrix=J, recovered_h_vector=h, true_j_matrix=true_j_matrix, true_h_vector=true_h_vector,
            frobenius_error=float(np.linalg.norm(J - true_j_matrix)) if true_j_matrix is not None else 0.0,
            h_error=float(np.linalg.norm(h - true_h_vector)) if true_h_vector is not None else 0.0,
            loss_history=hist, iterations=len(hist), matched_observables=dict(zip(self.observable_names, t.tolist())),
            predicted_observables=preds, method=f"{backend}-lbfgs",
        )

    def learn_from_shadows(
        self,
        snapshots: Sequence[ShadowSnapshot],
        true_j_matrix: Optional[np.ndarray] = None,
        true_h_vector: Optional[np.ndarray] = None,
        max_iter: int = 300,
        derandomized: bool = False,
        readout_model=None,
        backend: str = "analytic",
    ) -> HamiltonianLearningResult:
        """Estimate the model observables from shadows, then invert."""
        targets = {n: estimate_pauli_string_expectation(snapshots, self.pauli_string(n), derandomized=derandomized, readout_model=readout_model)
                   for n in self.observable_names}
        targets = {n: float(np.clip(v, -1.0, 1.0)) for n, v in targets.items()}
        return self.learn_from_expectations(targets, true_j_matrix, true_h_vector, max_iter=max_iter, backend=backend)


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
