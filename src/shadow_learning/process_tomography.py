"""
Shadow quantum process tomography.

Product Pauli eigenstates (|0⟩, |1⟩, |+⟩, |−⟩, |+i⟩, |−i⟩ on every qubit) are
sent through the unknown channel E; Pauli shadows of each output estimate all
⟨P_i⟩. Linearity turns the eigenstate inputs into Pauli inputs
(E(Z) = E(|0⟩⟨0|) − E(|1⟩⟨1|), E(1) = E(|0⟩⟨0|) + E(|1⟩⟨1|), …) and gives the
Pauli transfer matrix

    R_ij = Tr(P_i E(P_j)) / d,

from which the Choi matrix, the process fidelity F_pro = Tr(R_Uᵀ R)/d² against
a target unitary and the average gate fidelity (d F_pro + 1)/(d + 1) follow.

Physicality
-----------
Linear inversion from finite data is unbiased but not physical: shot noise
routinely gives a Choi matrix with negative eigenvalues and F_pro > 1. By
default the estimate is therefore projected onto the CPTP set — the
intersection of the PSD cone {J ⪰ 0} and the affine set {Tr_out J = 1} — in
Frobenius norm, using Dykstra's alternating-projection algorithm (which
converges to the exact projection onto an intersection of convex sets, unlike
plain alternation). Because the true channel lies in that convex set, the
projection can only move the estimate closer to it:
‖Π(Ĵ) − J‖_F ≤ ‖Ĵ − J‖_F. The price is the usual one for constrained
estimators: fidelities computed from Π(Ĵ) are biased low at small budgets
(a pure target sits on the boundary of the set), while the raw estimate is
unbiased but can exceed 1. Both are returned; they agree as the budget grows.
"""

from __future__ import annotations

from functools import reduce
from itertools import product
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .cirq_shadows import sample_shadows_from_density_matrix
from .reconstruction import estimate_pauli_string_expectation

_P = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}
_EIGENSTATES = {  # label: (state vector, Pauli, eigenvalue)
    "0": (np.array([1, 0]), "Z", +1), "1": (np.array([0, 1]), "Z", -1),
    "+": (np.array([1, 1]) / np.sqrt(2), "X", +1), "-": (np.array([1, -1]) / np.sqrt(2), "X", -1),
    "+i": (np.array([1, 1j]) / np.sqrt(2), "Y", +1), "-i": (np.array([1, -1j]) / np.sqrt(2), "Y", -1),
}
# coefficient c[P][label]: E(P) = Σ_label c · E(|label⟩⟨label|)
_COEFF = {
    "I": {"0": 1.0, "1": 1.0}, "Z": {"0": 1.0, "1": -1.0},
    "X": {"+": 1.0, "-": -1.0}, "Y": {"+i": 1.0, "-i": -1.0},
}


def pauli_strings(n_qubits: int) -> List[str]:
    return ["".join(p) for p in product("IXYZ", repeat=n_qubits)]


def pauli_matrix(s: str) -> np.ndarray:
    return reduce(np.kron, [_P[c] for c in s])


def pauli_transfer_matrix_of_unitary(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.complex128)
    n = int(round(np.log2(u.shape[0])))
    d = 2**n
    labels = pauli_strings(n)
    mats = [pauli_matrix(s) for s in labels]
    R = np.zeros((d * d, d * d))
    for i, Pi in enumerate(mats):
        for j, Pj in enumerate(mats):
            R[i, j] = np.real(np.trace(Pi @ u @ Pj @ u.conj().T)) / d
    return R


class ShadowProcessTomographer:
    def __init__(self, n_qubits: int = 1, seed: Optional[int] = None):
        self.n_qubits = int(n_qubits)
        self.dim = 2**self.n_qubits
        self.rng = np.random.default_rng(seed)
        self.labels = pauli_strings(self.n_qubits)
        self._mats = [pauli_matrix(s) for s in self.labels]

    # ----------------------------------------------------------------- #
    def run(self, channel: Callable[[np.ndarray], np.ndarray], n_snapshots_per_input: int = 1000,
            physical: bool = True) -> Dict[str, np.ndarray]:
        """
        ``channel`` maps a density matrix to a density matrix. Returns
        ``{"ptm", "choi", "ptm_raw", "choi_raw", "inputs", "output_expectations"}``;
        with ``physical=True`` (default) ``ptm``/``choi`` are the CPTP projection
        and ``*_raw`` the linear-inversion estimate.
        """
        labels_1q = list(_EIGENSTATES)
        out_exp: Dict[tuple, np.ndarray] = {}
        for combo in product(labels_1q, repeat=self.n_qubits):
            psi = reduce(np.kron, [_EIGENSTATES[l][0] for l in combo]).astype(np.complex128)
            rho_out = channel(np.outer(psi, psi.conj()))
            snaps = sample_shadows_from_density_matrix(rho_out, n_snapshots_per_input, self.rng)
            out_exp[combo] = np.array([estimate_pauli_string_expectation(snaps, s) for s in self.labels])
        R = np.zeros((self.dim**2, self.dim**2))
        for j, pj in enumerate(self.labels):
            # E(P_j) = Σ_combo (∏_q c[P_j[q]][combo[q]]) E(ρ_combo)
            for combo, vec in out_exp.items():
                coeff = 1.0
                for q in range(self.n_qubits):
                    coeff *= _COEFF[pj[q]].get(combo[q], 0.0)
                    if coeff == 0.0:
                        break
                if coeff != 0.0:
                    R[:, j] += coeff * vec
        # out_exp holds Tr(P_i E(ρ)); E(P_j) = Σ c·E(ρ) gives Tr(P_i E(P_j)) = Σ c·Tr(P_i E(ρ)); R needs the 1/d.
        R /= self.dim
        J_raw = self.choi_matrix(R)
        J = project_cptp(J_raw) if physical else J_raw
        return {"ptm": self.ptm_from_choi(J) if physical else R, "choi": J, "ptm_raw": R, "choi_raw": J_raw,
                "inputs": list(out_exp.keys()), "output_expectations": out_exp}

    def ptm_from_choi(self, J: np.ndarray) -> np.ndarray:
        """R_ij = Tr(P_i E(P_j))/d with E(X)_ab = Σ_ij X_ij J[(i,a),(j,b)]."""
        d = self.dim
        Jr = np.asarray(J).reshape(d, d, d, d)          # [i_in, a_out, j_in, b_out]
        R = np.zeros((d * d, d * d))
        for j, Pj in enumerate(self._mats):
            out = np.einsum("ij,iajb->ab", Pj, Jr)
            for i, Pi in enumerate(self._mats):
                R[i, j] = np.real(np.trace(Pi @ out)) / d
        return R

    # ----------------------------------------------------------------- #
    def choi_matrix(self, R: np.ndarray) -> np.ndarray:
        """J(E) = Σ_{ij} |i⟩⟨j| ⊗ E(|i⟩⟨j|)  (Tr J = d for trace-preserving E)."""
        d = self.dim
        # E(P_k) = Σ_m R[m, k] P_m
        E_P = [sum(R[m, k] * self._mats[m] for m in range(d * d)) for k in range(d * d)]
        J = np.zeros((d * d, d * d), dtype=np.complex128)
        for i in range(d):
            for j in range(d):
                Eij = np.zeros((d, d), dtype=np.complex128)
                Eij[i, j] = 1.0
                # |i⟩⟨j| = Σ_k a_k P_k with a_k = Tr(P_k† |i⟩⟨j|)/d = conj(P_k[i,j])... use Tr(P_k Eij)/d
                out = sum((np.trace(self._mats[k] @ Eij) / d) * E_P[k] for k in range(d * d))
                J += np.kron(Eij, out)
        return J

    def is_physical(self, J: np.ndarray, tol: float = 1e-8) -> bool:
        """CP (J ⪰ 0) and TP (Tr_out J = 1) to within ``tol``."""
        d = self.dim
        tp = np.einsum("iaja->ij", np.asarray(J).reshape(d, d, d, d))
        return bool(np.linalg.eigvalsh(0.5 * (J + J.conj().T)).min() >= -tol and np.allclose(tp, np.eye(d), atol=tol))

    def process_fidelity(self, R: np.ndarray, target_unitary: np.ndarray) -> float:
        R_u = pauli_transfer_matrix_of_unitary(target_unitary)
        return float(np.trace(R_u.T @ R) / self.dim**2)

    def average_gate_fidelity(self, R: np.ndarray, target_unitary: np.ndarray) -> float:
        return float((self.dim * self.process_fidelity(R, target_unitary) + 1.0) / (self.dim + 1.0))

    # v0.2 shim
    def estimate_process_fidelity(self, shadows_input_0, shadows_input_plus) -> float:
        z = estimate_pauli_string_expectation(shadows_input_0, "Z" * self.n_qubits)
        x = estimate_pauli_string_expectation(shadows_input_plus, "X" * self.n_qubits)
        return float(np.clip(0.5 * (z + x), 0.0, 1.0))


# --------------------------------------------------------------------------- #
# CPTP projection
# --------------------------------------------------------------------------- #
def _project_psd(J: np.ndarray) -> np.ndarray:
    H = 0.5 * (J + J.conj().T)
    w, V = np.linalg.eigh(H)
    return (V * np.clip(w, 0.0, None)) @ V.conj().T


def _project_tp(J: np.ndarray, d: int) -> np.ndarray:
    """Frobenius projection onto {J : Tr_out J = 1_d} (Choi ordering in ⊗ out)."""
    Jr = J.reshape(d, d, d, d)
    T = np.einsum("iaja->ij", Jr)                         # Tr_out J
    corr = (T - np.eye(d)) / d
    return J - np.kron(corr, np.eye(d))


def project_cptp(J: np.ndarray, max_iter: int = 5000, tol: float = 1e-12) -> np.ndarray:
    """
    Frobenius-nearest CPTP Choi matrix to ``J`` (Dykstra's algorithm on the
    PSD cone ∩ trace-preserving affine set).
    """
    J = np.asarray(J, dtype=np.complex128)
    d = int(round(np.sqrt(J.shape[0])))
    x = 0.5 * (J + J.conj().T)
    p = np.zeros_like(x)
    q = np.zeros_like(x)
    for _ in range(int(max_iter)):
        y = _project_tp(x + p, d)
        p = x + p - y
        x_new = _project_psd(y + q)
        q = y + q - x_new
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    # the PSD iterate is exactly CP; remove the residual TP error (≲ tol) without leaving the cone
    return _project_tp(x, d) if np.linalg.eigvalsh(_project_tp(x, d)).min() >= -1e-12 else x
