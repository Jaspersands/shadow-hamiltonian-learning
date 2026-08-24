"""
Differentiable Hamiltonian Learning from Shadow Expectation Values.

Recovers coupling matrices J_ij and magnetic fields h_i by differentiating through
thermal Gibbs states or time-evolution propagators to match shadow observables.
"""

from __future__ import annotations
import numpy as np
import scipy.linalg
import scipy.optimize
from dataclasses import dataclass
from typing import List, Tuple, Dict, Sequence, Optional, Any

from .cirq_shadows import ShadowSnapshot
from .reconstruction import estimate_many_observables


@dataclass
class HamiltonianLearningResult:
    """Stores the reconstructed Hamiltonian parameters and optimization convergence."""
    recovered_j_matrix: np.ndarray # Shape (N, N)
    recovered_h_vector: np.ndarray # Shape (N,)
    true_j_matrix: Optional[np.ndarray]
    true_h_vector: Optional[np.ndarray]
    frobenius_error: float
    loss_history: List[float]
    iterations: int
    matched_observables: Dict[str, float]
    predicted_observables: Dict[str, float]


class DifferentiableHamiltonianLearner:
    """
    Inverts classical shadow snapshot data to discover unknown Hamiltonian couplings.
    """

    def __init__(
        self,
        n_qubits: int,
        beta: float = 1.0, # Inverse temperature beta = 1 / (k_B * T)
        l1_reg: float = 1e-4,
    ):
        self.n_qubits = n_qubits
        self.beta = beta
        self.l1_reg = l1_reg
        self.pauli_basis_matrices = self._build_pauli_matrices()

    def _build_pauli_matrices(self) -> Dict[str, np.ndarray]:
        sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        id2 = np.eye(2, dtype=np.complex128)

        ops = {}
        # 1-qubit Z observables
        for i in range(self.n_qubits):
            m = [sz if k == i else id2 for k in range(self.n_qubits)]
            curr = m[0]
            for k in range(1, self.n_qubits):
                curr = np.kron(curr, m[k])
            ops[f"Z_{i}"] = curr

        # 2-qubit nearest-neighbor and all-to-all observables
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                for p_name, p_mat in [("X", sx), ("Y", sy), ("Z", sz)]:
                    m = [p_mat if k in [i, j] else id2 for k in range(self.n_qubits)]
                    curr = m[0]
                    for k in range(1, self.n_qubits):
                        curr = np.kron(curr, m[k])
                    ops[f"{p_name}{p_name}_{i}_{j}"] = curr

        return ops

    def build_hamiltonian_matrix(self, j_matrix: np.ndarray, h_vec: np.ndarray) -> np.ndarray:
        """Constructs 2^N x 2^N Hamiltonian from parameter matrices."""
        dim = 2**self.n_qubits
        H = np.zeros((dim, dim), dtype=np.complex128)

        # External fields h_i Z_i
        for i in range(self.n_qubits):
            H += h_vec[i] * self.pauli_basis_matrices[f"Z_{i}"]

        # Couplings J_ij (XX + YY + ZZ)
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                j_val = j_matrix[i, j]
                if abs(j_val) > 1e-8:
                    H += j_val * (
                        self.pauli_basis_matrices[f"XX_{i}_{j}"]
                        + self.pauli_basis_matrices[f"YY_{i}_{j}"]
                        + self.pauli_basis_matrices[f"ZZ_{i}_{j}"]
                    )

        return H

    def compute_thermal_observables(
        self, j_matrix: np.ndarray, h_vec: np.ndarray
    ) -> Dict[str, float]:
        """Calculates exact Gibbs state expectation values Tr(O exp(-beta H)) / Z."""
        H = self.build_hamiltonian_matrix(j_matrix, h_vec)
        evals, evecs = np.linalg.eigh(H)

        # Thermal weights
        exp_weights = np.exp(-self.beta * (evals - np.min(evals)))
        z = np.sum(exp_weights)
        probs = exp_weights / z

        # Density matrix rho = sum_k p_k |v_k><v_k|
        rho = (evecs * probs) @ evecs.conj().T

        preds = {}
        for name, op in self.pauli_basis_matrices.items():
            preds[name] = float(np.real(np.trace(op @ rho)))

        return preds

    def learn_from_shadows(
        self,
        snapshots: Sequence[ShadowSnapshot],
        true_j_matrix: Optional[np.ndarray] = None,
        true_h_vector: Optional[np.ndarray] = None,
        max_iter: int = 100,
    ) -> HamiltonianLearningResult:
        """
        Executes differentiable gradient descent to recover J_ij and h_i from shadow snapshots.
        """
        # 1. Estimate shadow expectation values
        shadow_targets = {}
        obs_names = list(self.pauli_basis_matrices.keys())

        # Translate observable names into Pauli string format
        for name in obs_names:
            p_str = ["I"] * self.n_qubits
            if name.startswith("Z_"):
                i = int(name.split("_")[1])
                p_str[i] = "Z"
            elif name.startswith("XX_"):
                _, i, j = name.split("_")
                p_str[int(i)] = "X"
                p_str[int(j)] = "X"
            elif name.startswith("YY_"):
                _, i, j = name.split("_")
                p_str[int(i)] = "Y"
                p_str[int(j)] = "Y"
            elif name.startswith("ZZ_"):
                _, i, j = name.split("_")
                p_str[int(i)] = "Z"
                p_str[int(j)] = "Z"

            p_string = "".join(p_str)
            from .reconstruction import estimate_pauli_string_expectation
            shadow_targets[name] = estimate_pauli_string_expectation(snapshots, p_string)

        # 2. Setup optimization vector: [J_{01}, J_{02}, ..., J_{(N-1)N}, h_0, ..., h_{N-1}]
        n_couplings = (self.n_qubits * (self.n_qubits - 1)) // 2
        n_params = n_couplings + self.n_qubits

        def unpack_params(p_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            j_mat = np.zeros((self.n_qubits, self.n_qubits))
            idx = 0
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    j_mat[i, j] = p_vec[idx]
                    j_mat[j, i] = p_vec[idx]
                    idx += 1
            h_v = p_vec[n_couplings:]
            return j_mat, h_v

        loss_history = []

        def loss_fn(p_vec: np.ndarray) -> float:
            j_mat, h_v = unpack_params(p_vec)
            preds = self.compute_thermal_observables(j_mat, h_v)

            mse = 0.0
            for name, target_val in shadow_targets.items():
                mse += (preds[name] - target_val) ** 2

            l1 = self.l1_reg * float(np.sum(np.abs(p_vec)))
            total = mse + l1
            loss_history.append(total)
            return total

        # Run L-BFGS-B / Powell
        p0 = np.random.uniform(0.1, 0.5, size=n_params)
        res = scipy.optimize.minimize(
            loss_fn,
            p0,
            method="Nelder-Mead",
            options={"maxiter": max_iter, "xatol": 1e-4, "fatol": 1e-4},
        )

        rec_j, rec_h = unpack_params(res.x)
        final_preds = self.compute_thermal_observables(rec_j, rec_h)

        frob_err = 0.0
        if true_j_matrix is not None:
            frob_err = float(np.linalg.norm(rec_j - true_j_matrix))

        return HamiltonianLearningResult(
            recovered_j_matrix=rec_j,
            recovered_h_vector=rec_h,
            true_j_matrix=true_j_matrix,
            true_h_vector=true_h_vector,
            frobenius_error=frob_err,
            loss_history=loss_history,
            iterations=len(loss_history),
            matched_observables=shadow_targets,
            predicted_observables=final_preds,
        )


def learn_hamiltonian_from_shadows(
    snapshots: Sequence[ShadowSnapshot],
    n_qubits: int,
    true_j_matrix: Optional[np.ndarray] = None,
    true_h_vector: Optional[np.ndarray] = None,
    beta: float = 1.0,
    max_iter: int = 100,
) -> HamiltonianLearningResult:
    """Convenience helper to run Hamiltonian recovery from shadow snapshots."""
    learner = DifferentiableHamiltonianLearner(n_qubits=n_qubits, beta=beta)
    return learner.learn_from_shadows(
        snapshots, true_j_matrix=true_j_matrix, true_h_vector=true_h_vector, max_iter=max_iter
    )
