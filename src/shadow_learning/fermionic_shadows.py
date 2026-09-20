"""
Matchgate (fermionic) classical shadows — Wan, Huggins, Lee & Babbush (2022).

Jordan–Wigner Majoranas  γ_{2p} = Z_0⋯Z_{p−1} X_p,  γ_{2p+1} = Z_0⋯Z_{p−1} Y_p,
so that c_p = (γ_{2p} + iγ_{2p+1})/2 and n_p = ½(1 + iγ_{2p}γ_{2p+1}).

A random *Clifford matchgate* U_Q acts as U_Q† γ_μ U_Q = s_μ γ_{π(μ)} for a
uniformly random signed permutation Q = (π, s). Measuring U_Q ρ U_Q† in the
computational basis therefore measures the commuting Hermitian operators

    M_j = −i s_{2j} s_{2j+1} γ_{π(2j)} γ_{π(2j+1)},   eigenvalues ±1 = (−1)^{b_j},

which partition the Majoranas into n random pairs. The inverse shadow channel
rescales the degree-2k Majorana sector by 1/λ_k with
λ_k = C(n, k)/C(2n, 2k); for the 1-RDM only λ₁ = 1/(2n − 1) is needed:

    ⟨iγ_uγ_v⟩ ≈ (2n − 1) · (−1)^{b_j} s_{2j}s_{2j+1} · (−1 if (π(2j),π(2j+1)) = (u,v); +1 if (v,u); 0 otherwise).

Sampling uses the pair-projector recursion Π_b = ∏_j (1 + (−1)^{b_j} M_j)/2 on
the state vector, so a snapshot costs n matrix–vector products.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_I = np.eye(2, dtype=np.complex128)


@dataclass(frozen=True)
class MatchgateSnapshot:
    perm: Tuple[int, ...]    # π
    signs: Tuple[int, ...]   # s
    bits: Tuple[int, ...]    # b


def majorana_operators(n_modes: int) -> List[np.ndarray]:
    """[γ_0, γ_1, …, γ_{2n−1}] as dense 2ⁿ×2ⁿ matrices (Jordan–Wigner)."""
    out = []
    for p in range(n_modes):
        pre = [_Z] * p
        post = [_I] * (n_modes - p - 1)
        out.append(reduce(np.kron, pre + [_X] + post))
        out.append(reduce(np.kron, pre + [_Y] + post))
    return out


def random_signed_permutation(n_modes: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    perm = rng.permutation(2 * n_modes)
    signs = rng.choice([-1, 1], size=2 * n_modes)
    return perm, signs


def slater_state(n_modes: int, occupied: Sequence[int]) -> np.ndarray:
    """Computational basis state with the given modes occupied (qubit 0 = MSB)."""
    psi = np.zeros(2**n_modes, dtype=np.complex128)
    idx = sum(1 << (n_modes - 1 - p) for p in occupied)
    psi[idx] = 1.0
    return psi


class FermionicMatchgateShadows:
    def __init__(self, n_modes: int, seed: Optional[int] = None):
        self.n_modes = int(n_modes)
        self.n_majoranas = 2 * self.n_modes
        self.rng = np.random.default_rng(seed)
        self.gammas = majorana_operators(self.n_modes)
        self.lambda_1 = 1.0 / (2 * self.n_modes - 1)

    # ----------------------------------------------------------------- #
    def _pair_operator(self, a: int, b: int, sa: int, sb: int) -> np.ndarray:
        """M = −i s_a s_b γ_a γ_b (Hermitian, M² = 1)."""
        return -1j * sa * sb * (self.gammas[a] @ self.gammas[b])

    def sample(self, state: np.ndarray, n_snapshots: int) -> List[MatchgateSnapshot]:
        """Draw matchgate shadows from a state vector (2ⁿ) or density matrix (2ⁿ×2ⁿ)."""
        state = np.asarray(state, dtype=np.complex128)
        pure = state.ndim == 1
        snaps: List[MatchgateSnapshot] = []
        dim = 2**self.n_modes
        for _ in range(int(n_snapshots)):
            perm, signs = random_signed_permutation(self.n_modes, self.rng)
            bits = []
            cur = state.copy()
            for j in range(self.n_modes):
                a, b = int(perm[2 * j]), int(perm[2 * j + 1])
                M = self._pair_operator(a, b, int(signs[2 * j]), int(signs[2 * j + 1]))
                P0 = 0.5 * (np.eye(dim) + M)
                if pure:
                    v0 = P0 @ cur
                    norm = float(np.real(np.vdot(cur, cur)))
                    p0 = float(np.real(np.vdot(v0, v0))) / norm if norm > 0 else 0.5
                else:
                    norm = float(np.real(np.trace(cur)))
                    p0 = float(np.real(np.trace(P0 @ cur))) / norm if norm > 0 else 0.5
                bit = 0 if self.rng.random() < np.clip(p0, 0.0, 1.0) else 1
                proj = P0 if bit == 0 else 0.5 * (np.eye(dim) - M)
                cur = proj @ cur if pure else proj @ cur @ proj
                bits.append(bit)
            snaps.append(MatchgateSnapshot(perm=tuple(int(x) for x in perm), signs=tuple(int(x) for x in signs), bits=tuple(bits)))
        return snaps

    # ----------------------------------------------------------------- #
    def estimate_majorana_pair(self, snapshots: Sequence[MatchgateSnapshot], u: int, v: int) -> float:
        """Estimate ⟨i γ_u γ_v⟩ (u ≠ v)."""
        if u == v:
            return 0.0
        total = 0.0
        for s in snapshots:
            for j in range(self.n_modes):
                a, b = s.perm[2 * j], s.perm[2 * j + 1]
                if (a, b) == (u, v) or (a, b) == (v, u):
                    orient = -1.0 if (a, b) == (u, v) else 1.0
                    total += orient * (1.0 if s.bits[j] == 0 else -1.0) * s.signs[2 * j] * s.signs[2 * j + 1]
                    break
        return float(total / (len(snapshots) * self.lambda_1))

    def estimate_majorana_matrix(self, snapshots: Sequence[MatchgateSnapshot]) -> np.ndarray:
        """Antisymmetric G with G[u, v] ≈ ⟨i γ_u γ_v⟩ from one pass over the data."""
        m = self.n_majoranas
        G = np.zeros((m, m))
        for s in snapshots:
            for j in range(self.n_modes):
                a, b = s.perm[2 * j], s.perm[2 * j + 1]
                val = (1.0 if s.bits[j] == 0 else -1.0) * s.signs[2 * j] * s.signs[2 * j + 1]
                G[a, b] -= val
                G[b, a] += val
        return G / (len(snapshots) * self.lambda_1)

    def estimate_1rdm(self, snapshots: Sequence[MatchgateSnapshot]) -> np.ndarray:
        """D_pq = ⟨c_p† c_q⟩ from the Majorana two-point estimates."""
        G = self.estimate_majorana_matrix(snapshots)
        n = self.n_modes

        def gam(u, v):  # ⟨γ_u γ_v⟩ = δ_uv − i ⟨iγ_uγ_v⟩
            return (1.0 if u == v else 0.0) - 1j * G[u, v]

        D = np.zeros((n, n), dtype=np.complex128)
        for p in range(n):
            for q in range(n):
                D[p, q] = 0.25 * (gam(2 * p, 2 * q) + 1j * gam(2 * p, 2 * q + 1) - 1j * gam(2 * p + 1, 2 * q) + gam(2 * p + 1, 2 * q + 1))
        return D

    # ----------------------------------------------------------------- #
    def exact_1rdm(self, state: np.ndarray) -> np.ndarray:
        """Reference D_pq = ⟨c_p† c_q⟩ computed directly from the state."""
        state = np.asarray(state, dtype=np.complex128)
        rho = np.outer(state, state.conj()) if state.ndim == 1 else state
        n = self.n_modes
        c = [0.5 * (self.gammas[2 * p] + 1j * self.gammas[2 * p + 1]) for p in range(n)]
        D = np.zeros((n, n), dtype=np.complex128)
        for p in range(n):
            for q in range(n):
                D[p, q] = np.trace(c[p].conj().T @ c[q] @ rho)
        return D

    # v0.2 shim
    def estimate_1rdm_element(self, p: int, q: int, snapshots: Sequence[MatchgateSnapshot]) -> complex:
        return complex(self.estimate_1rdm(snapshots)[p, q])
