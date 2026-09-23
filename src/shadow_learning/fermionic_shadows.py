"""
Matchgate (fermionic) classical shadows — Wan, Huggins, Lee & Babbush (2022).

Jordan–Wigner Majoranas  γ_{2p} = Z_0⋯Z_{p−1} X_p,  γ_{2p+1} = Z_0⋯Z_{p−1} Y_p,
so that c_p = (γ_{2p} + iγ_{2p+1})/2 and n_p = ½(1 + iγ_{2p}γ_{2p+1}).

A random *Clifford matchgate* U_Q acts as U_Q† γ_μ U_Q = s_μ γ_{π(μ)} for a
uniformly random signed permutation Q = (π, s). Measuring U_Q ρ U_Q† in the
computational basis therefore measures the commuting Hermitian operators

    M_j = −i s_{2j} s_{2j+1} γ_{π(2j)} γ_{π(2j+1)},   eigenvalues ±1 = (−1)^{b_j},

i.e. it pairs the 2n Majoranas at random and reads out the parity of every
pair. A degree-2k Majorana monomial γ_S is determined by one snapshot exactly
when S is a union of k measured pairs, which happens with probability

    λ_k = (2k−1)!! (2n−2k−1)!! / (2n−1)!! = C(n, k) / C(2n, 2k),

so (single-shot value)/λ_k is an unbiased estimator of ⟨γ_S⟩. k = 1 gives the
1-RDM (λ₁ = 1/(2n−1)); k = 2 gives the 2-RDM (λ₂ = 3/((2n−1)(2n−3))), and with
it the energy of any two-body fermionic Hamiltonian.

Two samplers
------------
* Dense (any state): the pair-projector recursion on a 2ⁿ state vector or
  density matrix. Exponential in n — use it for non-Gaussian states only.
* Gaussian (``FermionicGaussianState``): the state is stored as its Majorana
  covariance Γ_pq = ⟨iγ_pγ_q⟩ (a real antisymmetric 2n×2n matrix). Measuring a
  pair parity iγ_aγ_b = σ has probability (1 + σΓ_ab)/2 and, by Wick's theorem,
  leaves a Gaussian state with the rank-2 update

      Γ'_pq = Γ_pq + σ (Γ_pb Γ_qa − Γ_pa Γ_qb) / (1 + σ Γ_ab),

  rows/columns a, b zeroed and Γ'_ab = σ. A snapshot costs O(n³) and no
  2ⁿ object is ever formed, which is the point of matchgate shadows.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from itertools import combinations
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

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
    """[γ_0, γ_1, …, γ_{2n−1}] as dense 2ⁿ×2ⁿ matrices (Jordan–Wigner). Small n only."""
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
    """Computational basis state vector with the given modes occupied (qubit 0 = MSB)."""
    psi = np.zeros(2**n_modes, dtype=np.complex128)
    idx = sum(1 << (n_modes - 1 - p) for p in occupied)
    psi[idx] = 1.0
    return psi


# --------------------------------------------------------------------------- #
# Gaussian states
# --------------------------------------------------------------------------- #
class FermionicGaussianState:
    """
    A fermionic Gaussian state stored as its Majorana covariance
    Γ_pq = ⟨iγ_pγ_q⟩ (p ≠ q), Γ real antisymmetric with Γᵀ Γ ⪯ 1.
    """

    def __init__(self, covariance: np.ndarray):
        G = np.asarray(covariance, dtype=float)
        if G.shape[0] != G.shape[1] or G.shape[0] % 2:
            raise ValueError("covariance must be 2n × 2n")
        if not np.allclose(G, -G.T, atol=1e-10):
            raise ValueError("covariance must be antisymmetric")
        self.covariance = G
        self.n_modes = G.shape[0] // 2

    @classmethod
    def slater(cls, n_modes: int, occupied: Sequence[int]) -> "FermionicGaussianState":
        """Occupation-number state: ⟨iγ_{2p}γ_{2p+1}⟩ = 2n_p − 1."""
        G = np.zeros((2 * n_modes, 2 * n_modes))
        occ = set(int(p) for p in occupied)
        for p in range(n_modes):
            v = 1.0 if p in occ else -1.0
            G[2 * p, 2 * p + 1], G[2 * p + 1, 2 * p] = v, -v
        return cls(G)

    @classmethod
    def from_one_rdm(cls, D: np.ndarray) -> "FermionicGaussianState":
        """Number-conserving Gaussian state with 1-RDM D_pq = ⟨c_p† c_q⟩."""
        D = np.asarray(D, dtype=np.complex128)
        n = D.shape[0]
        # correlation of a = (c_0..c_{n−1}, c_0†..c_{n−1}†):  ⟨c†c⟩ = D, ⟨c c†⟩ = 1 − Dᵀ
        C = np.zeros((2 * n, 2 * n), dtype=np.complex128)
        C[:n, n:] = np.eye(n) - D.T
        C[n:, :n] = D
        W = np.zeros((2 * n, 2 * n), dtype=np.complex128)   # γ = W a
        for p in range(n):
            W[2 * p, p], W[2 * p, n + p] = 1.0, 1.0
            W[2 * p + 1, p], W[2 * p + 1, n + p] = -1j, 1j
        M = W @ C @ W.T                                     # ⟨γ_u γ_v⟩
        G = np.real(1j * (M - np.eye(2 * n)))
        return cls(0.5 * (G - G.T))

    @classmethod
    def ground_state_of_quadratic(cls, h: np.ndarray, n_particles: int) -> "FermionicGaussianState":
        """Slater determinant filling the ``n_particles`` lowest orbitals of H = Σ h_pq c_p† c_q."""
        h = np.asarray(h, dtype=np.complex128)
        eps, U = np.linalg.eigh(0.5 * (h + h.conj().T))
        occ = U[:, : int(n_particles)]                      # orbital k: b_k† = Σ_p U_pk c_p†
        return cls.from_one_rdm(np.conj(occ) @ occ.T)

    def one_rdm(self) -> np.ndarray:
        """D_pq = ⟨c_p† c_q⟩ from Γ."""
        return _one_rdm_from_pair_oracle(self.n_modes, lambda u, v: self.covariance[u, v])

    def majorana_expectation(self, idx: Sequence[int]) -> complex:
        """⟨γ_{i1}⋯γ_{im}⟩ by Wick's theorem (Pfaffian of the covariance block)."""
        sign, T = _reduce_monomial(idx)
        if not T:
            return complex(sign)
        if len(T) % 2:
            return 0.0
        # ⟨γ_{t1}⋯γ_{t2k}⟩ = Pf(A) with A_uv = ⟨γ_uγ_v⟩ = −iΓ_uv (u < v)
        A = -1j * self.covariance[np.ix_(T, T)]
        return sign * _pfaffian(A)

    def to_state_vector(self) -> np.ndarray:
        """Dense state (small n, testing only): ground state of −Σ Γ_pq iγ_pγ_q / 4."""
        g = majorana_operators(self.n_modes)
        m = 2 * self.n_modes
        H = sum(-0.25 * self.covariance[p, q] * 1j * g[p] @ g[q] for p in range(m) for q in range(m) if p != q)
        w, V = np.linalg.eigh(H)
        return V[:, 0]


def _pfaffian(A: np.ndarray) -> complex:
    """Pfaffian of an antisymmetric matrix (small sizes; recursive expansion)."""
    n = A.shape[0]
    if n == 0:
        return 1.0
    if n % 2:
        return 0.0
    if n == 2:
        return A[0, 1]
    total = 0.0
    for j in range(1, n):
        keep = [k for k in range(1, n) if k != j]
        total += (-1) ** (j + 1) * A[0, j] * _pfaffian(A[np.ix_(keep, keep)])
    return total


def _reduce_monomial(idx: Sequence[int]) -> Tuple[int, Tuple[int, ...]]:
    """γ_{i1}⋯γ_{im} = sign · γ_T with T sorted and distinct (γ² = 1, {γ_u, γ_v} = 2δ)."""
    arr = list(idx)
    sign = 1
    # bubble sort, tracking anticommutation signs, cancelling equal neighbours
    changed = True
    while changed:
        changed = False
        k = 0
        while k < len(arr) - 1:
            if arr[k] == arr[k + 1]:
                del arr[k: k + 2]
                changed = True
                continue
            if arr[k] > arr[k + 1]:
                arr[k], arr[k + 1] = arr[k + 1], arr[k]
                sign = -sign
                changed = True
            k += 1
    return sign, tuple(arr)


def _one_rdm_from_pair_oracle(n: int, gam_i: Callable[[int, int], float]) -> np.ndarray:
    """D_pq = ⟨c_p† c_q⟩ given ⟨iγ_uγ_v⟩ (u ≠ v)."""

    def gam(u, v):  # ⟨γ_u γ_v⟩ = δ_uv − i ⟨iγ_uγ_v⟩
        return (1.0 if u == v else 0.0) - 1j * (gam_i(u, v) if u != v else 0.0)

    D = np.zeros((n, n), dtype=np.complex128)
    for p in range(n):
        for q in range(n):
            D[p, q] = 0.25 * (gam(2 * p, 2 * q) + 1j * gam(2 * p, 2 * q + 1) - 1j * gam(2 * p + 1, 2 * q) + gam(2 * p + 1, 2 * q + 1))
    return D


def _two_rdm_from_monomial_oracle(n: int, expect: Callable[[Tuple[int, ...]], complex]) -> np.ndarray:
    """D2_pqrs = ⟨c_p† c_q† c_r c_s⟩ from an oracle for ⟨γ_T⟩ (T sorted, distinct)."""
    # c_p = Σ_x a[x] γ_{2p+x},  a = (1/2, i/2);   c_p† uses conj(a)
    a = np.array([0.5, 0.5j])
    cache: Dict[Tuple[int, ...], complex] = {(): 1.0}

    def ev(idx):
        sign, T = _reduce_monomial(idx)
        if len(T) % 2:
            return 0.0
        if T not in cache:
            cache[T] = expect(T)
        return sign * cache[T]

    D2 = np.zeros((n, n, n, n), dtype=np.complex128)
    for p in range(n):
        for q in range(n):
            for r in range(n):
                for s in range(n):
                    tot = 0.0
                    for x1 in range(2):
                        for x2 in range(2):
                            for x3 in range(2):
                                for x4 in range(2):
                                    coef = np.conj(a[x1]) * np.conj(a[x2]) * a[x3] * a[x4]
                                    tot += coef * ev((2 * p + x1, 2 * q + x2, 2 * r + x3, 2 * s + x4))
                    D2[p, q, r, s] = tot
    return D2


# --------------------------------------------------------------------------- #
# Shadows
# --------------------------------------------------------------------------- #
StateLike = Union[np.ndarray, FermionicGaussianState]


class FermionicMatchgateShadows:
    def __init__(self, n_modes: int, seed: Optional[int] = None):
        self.n_modes = int(n_modes)
        self.n_majoranas = 2 * self.n_modes
        self.rng = np.random.default_rng(seed)
        self._gammas: Optional[List[np.ndarray]] = None
        n = self.n_modes
        self.lambda_1 = 1.0 / (2 * n - 1)
        self.lambda_2 = 3.0 / ((2 * n - 1) * (2 * n - 3)) if n >= 2 else 0.0

    @property
    def gammas(self) -> List[np.ndarray]:
        """Dense Majorana matrices, built on first use (dense sampler / references only)."""
        if self._gammas is None:
            self._gammas = majorana_operators(self.n_modes)
        return self._gammas

    # ----------------------------------------------------------------- #
    # sampling
    # ----------------------------------------------------------------- #
    def sample(self, state: StateLike, n_snapshots: int) -> List[MatchgateSnapshot]:
        """
        Draw matchgate shadows. ``state`` may be a ``FermionicGaussianState``
        (polynomial sampler) or a 2ⁿ state vector / 2ⁿ×2ⁿ density matrix (dense sampler).
        """
        if isinstance(state, FermionicGaussianState):
            if state.n_modes != self.n_modes:
                raise ValueError("state has the wrong number of modes")
            return self._sample_gaussian(state, n_snapshots)
        return self._sample_dense(np.asarray(state, dtype=np.complex128), n_snapshots)

    def _sample_gaussian(self, state: FermionicGaussianState, n_snapshots: int) -> List[MatchgateSnapshot]:
        snaps: List[MatchgateSnapshot] = []
        for _ in range(int(n_snapshots)):
            perm, signs = random_signed_permutation(self.n_modes, self.rng)
            G = state.covariance.copy()
            bits = []
            for j in range(self.n_modes):
                a, b = int(perm[2 * j]), int(perm[2 * j + 1])
                ss = signs[2 * j] * signs[2 * j + 1]
                p0 = float(np.clip(0.5 * (1.0 - ss * G[a, b]), 0.0, 1.0))   # P(M_j = +1)
                bit = 0 if self.rng.random() < p0 else 1
                m = 1.0 if bit == 0 else -1.0
                sigma = -m * ss                                            # measured value of iγ_aγ_b
                denom = 1.0 + sigma * G[a, b]
                if denom > 1e-14:
                    ga, gb = G[:, a].copy(), G[:, b].copy()
                    G = G + sigma * (np.outer(gb, ga) - np.outer(ga, gb)) / denom
                G[a, :] = 0.0; G[:, a] = 0.0; G[b, :] = 0.0; G[:, b] = 0.0
                G[a, b], G[b, a] = sigma, -sigma
                bits.append(bit)
            snaps.append(MatchgateSnapshot(perm=tuple(int(x) for x in perm), signs=tuple(int(x) for x in signs), bits=tuple(bits)))
        return snaps

    def _pair_operator(self, a: int, b: int, sa: int, sb: int) -> np.ndarray:
        """M = −i s_a s_b γ_a γ_b (Hermitian, M² = 1)."""
        return -1j * sa * sb * (self.gammas[a] @ self.gammas[b])

    def _sample_dense(self, state: np.ndarray, n_snapshots: int) -> List[MatchgateSnapshot]:
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
    # estimators
    # ----------------------------------------------------------------- #
    @staticmethod
    def _pair_values(s: MatchgateSnapshot, n: int):
        """[(a, b, σ)] — measured pairs with the single-shot value σ of iγ_aγ_b."""
        out = []
        for j in range(n):
            a, b = s.perm[2 * j], s.perm[2 * j + 1]
            m = 1.0 if s.bits[j] == 0 else -1.0
            out.append((a, b, -m * s.signs[2 * j] * s.signs[2 * j + 1]))
        return out

    def estimate_majorana_pair(self, snapshots: Sequence[MatchgateSnapshot], u: int, v: int) -> float:
        """Estimate ⟨i γ_u γ_v⟩ (u ≠ v)."""
        if u == v:
            return 0.0
        return float(self.estimate_majorana_matrix(snapshots)[u, v])

    def estimate_majorana_matrix(self, snapshots: Sequence[MatchgateSnapshot]) -> np.ndarray:
        """Antisymmetric G with G[u, v] ≈ ⟨i γ_u γ_v⟩ from one pass over the data."""
        m = self.n_majoranas
        G = np.zeros((m, m))
        for s in snapshots:
            for a, b, sigma in self._pair_values(s, self.n_modes):
                G[a, b] += sigma
                G[b, a] -= sigma
        return G / (len(snapshots) * self.lambda_1)

    def estimate_majorana_quartets(self, snapshots: Sequence[MatchgateSnapshot]) -> Dict[Tuple[int, int, int, int], float]:
        """
        ⟨γ_a γ_b γ_c γ_d⟩ (a < b < c < d) for every quartet seen in the data;
        unseen quartets are estimated as 0. Cost O(N n²).
        """
        acc: Dict[Tuple[int, int, int, int], float] = {}
        for s in snapshots:
            pv = self._pair_values(s, self.n_modes)
            for (a, b, s1), (c, d, s2) in combinations(pv, 2):
                # γ_aγ_b = −iσ₁ and γ_cγ_d = −iσ₂ on this snapshot  ⇒  γ_aγ_bγ_cγ_d = −σ₁σ₂
                sign, T = _reduce_monomial((a, b, c, d))
                acc[T] = acc.get(T, 0.0) + sign * (-s1 * s2)
        scale = 1.0 / (len(snapshots) * self.lambda_2)
        return {k: v * scale for k, v in acc.items()}

    def estimate_1rdm(self, snapshots: Sequence[MatchgateSnapshot]) -> np.ndarray:
        """D_pq = ⟨c_p† c_q⟩ from the Majorana two-point estimates."""
        G = self.estimate_majorana_matrix(snapshots)
        return _one_rdm_from_pair_oracle(self.n_modes, lambda u, v: G[u, v])

    def estimate_2rdm(self, snapshots: Sequence[MatchgateSnapshot]) -> np.ndarray:
        """D2_pqrs = ⟨c_p† c_q† c_r c_s⟩ from the degree-2 and degree-4 Majorana estimates."""
        G = self.estimate_majorana_matrix(snapshots)
        Q = self.estimate_majorana_quartets(snapshots)

        def expect(T):
            if len(T) == 2:
                return -1j * G[T[0], T[1]]
            if len(T) == 4:
                return Q.get(T, 0.0)
            return 0.0

        return _two_rdm_from_monomial_oracle(self.n_modes, expect)

    # ----------------------------------------------------------------- #
    # exact references
    # ----------------------------------------------------------------- #
    def exact_1rdm(self, state: StateLike) -> np.ndarray:
        """Reference D_pq = ⟨c_p† c_q⟩ (Wick for Gaussian states, dense otherwise)."""
        if isinstance(state, FermionicGaussianState):
            return state.one_rdm()
        rho = self._rho(state)
        c = self._annihilators()
        n = self.n_modes
        return np.array([[np.trace(c[p].conj().T @ c[q] @ rho) for q in range(n)] for p in range(n)])

    def exact_2rdm(self, state: StateLike) -> np.ndarray:
        """Reference D2_pqrs = ⟨c_p† c_q† c_r c_s⟩."""
        if isinstance(state, FermionicGaussianState):
            return _two_rdm_from_monomial_oracle(self.n_modes, lambda T: state.majorana_expectation(T))
        rho = self._rho(state)
        c = self._annihilators()
        n = self.n_modes
        D2 = np.zeros((n, n, n, n), dtype=np.complex128)
        for p in range(n):
            for q in range(n):
                pq = c[p].conj().T @ c[q].conj().T
                for r in range(n):
                    for s in range(n):
                        D2[p, q, r, s] = np.trace(pq @ c[r] @ c[s] @ rho)
        return D2

    def _annihilators(self) -> List[np.ndarray]:
        return [0.5 * (self.gammas[2 * p] + 1j * self.gammas[2 * p + 1]) for p in range(self.n_modes)]

    @staticmethod
    def _rho(state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.complex128)
        return np.outer(state, state.conj()) if state.ndim == 1 else state

    # v0.2 shim
    def estimate_1rdm_element(self, p: int, q: int, snapshots: Sequence[MatchgateSnapshot]) -> complex:
        return complex(self.estimate_1rdm(snapshots)[p, q])


def two_body_energy(h1: np.ndarray, h2: np.ndarray, D1: np.ndarray, D2: np.ndarray) -> float:
    """E = Σ h1_pq D1_pq + Σ h2_pqrs D2_pqrs for H = Σ h1_pq c_p†c_q + Σ h2_pqrs c_p†c_q†c_r c_s."""
    return float(np.real(np.einsum("pq,pq->", h1, D1) + np.einsum("pqrs,pqrs->", h2, D2)))
