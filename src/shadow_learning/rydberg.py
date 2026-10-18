"""
Learning a Rydberg Hamiltonian from computational-basis snapshots of quench dynamics.

Model
-----
A neutral-atom array under a global drive with zero phase evolves under

    H/ħ = (Ω/2) Σ_i X_i − Δ Σ_i n_i − Σ_i δ_i n_i + Σ_{i<j} V_ij n_i n_j,    V_ij = C6 / |r_i − r_j|^6,

with n = |r⟩⟨r| = (1 − Z)/2, ground state |g⟩ = |0⟩ and Rydberg state |r⟩ = |1⟩. The
per-site offsets δ_i are optional (``local_detuning=True``). Units: Ω, Δ, δ_i and
V in rad/μs, positions in μm, times in μs, C6 in rad μm^6/μs. This is the
Hamiltonian of Wurtz et al. (2023), section 1.3, and of the Amazon Braket AHS
documentation with local detuning switched off. Qubit 0 is the most significant
bit of a basis-state index, as elsewhere in the package.

Protocol
--------
Prepare |g…g⟩ and apply one pulse of total duration T: Ω(t) rises linearly
from 0 to Ω over ``ramp_time`` (0.05 μs by default), stays flat, and falls
back to 0 at T, while Δ is constant and the phase is 0. This is the shortest
waveform Braket accepts, since Ω must start and end at 0. Every atom is read
out in the computational basis at T, after the drive is off, so a probe time is
a total pulse duration. Ramps are simulated by midpoint time slicing
(``ramp_slices`` per ramp); ``ramp_time=0`` gives an instantaneous quench for
reference. Each shot returns one bitstring. In this module bit 1 means "read as
Rydberg"; Aquila's ``post_sequence`` uses 1 for an atom present (ground), so
convert with ``bits = 1 − post_sequence``, and drop failed shots and shots
whose ``pre_sequence`` differs from the requested filling
(:meth:`QuenchData.from_braket`).

Exact symmetries of the data
----------------------------
From |g…g⟩, Z-basis outcome probabilities are identical under

* a change of drive phase, Ω → e^{iφ}Ω: conjugation by Π_i e^{−iφ n_i} fixes
  |g…g⟩ and the Z-basis projectors. For a real drive this is Ω → −Ω (φ = π);
* H → −H, i.e. (Ω, Δ, C6, δ_i) → −(Ω, Δ, C6, δ_i): H is real, so
  ⟨z|e^{+iHt}|0⟩ = ⟨z|e^{−iHt}|0⟩*, and the same holds slice by slice for the
  ramped pulse. Combined with the first, (Δ, C6, δ_i) → −(Δ, C6, δ_i) at fixed
  Ω: the sign of Δ is fixed only relative to the sign of the interaction,
  which is repulsive for nS–nS pairs;
* (Ω, Δ, C6, δ_i) → a·(Ω, Δ, C6, δ_i) with every time, ramps included,
  divided by a: the data fix rates times durations, so absolute values rest on
  the device clock, and an error in the programmed durations moves all
  parameters together;
* C6 → C6·s^6 together with r → s·r: the data determine C6 / s^6 only, so
  calibrating C6 and calibrating the array's length scale are one measurement;
* with local detuning, Δ → Δ + c together with δ_i → δ_i − c: only Δ + δ_i
  enters, so fix Δ or the mean offset.

Everything else is a question of how much information the data carry, which
:meth:`RydbergQuenchModel.fisher_information` and
:meth:`RydbergQuenchModel.identifiability` answer before any fitting.

Gradients
---------
H is linear in (Ω, Δ, C6, δ_i), so ∂ψ(t)/∂θ_a follows from one eigendecomposition
H = V diag(w) Vᵀ:  Vᵀ (∂e^{−iHt}/∂θ_a) V = G̃_a ∘ F(t), with G̃_a = Vᵀ G_a V and
F_mn = −i t e^{−i(w_m + w_n)t/2} sinc((w_m − w_n)t/2). The sinc form is exact and
finite at degenerate levels, which translation-symmetric arrays have. The
ramped pulse is a product of such constant slices, and the derivative is
propagated forward through them (Ω enters a ramp slice scaled by its shape).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.linalg
import scipy.optimize
import scipy.sparse
import scipy.sparse.linalg
import scipy.stats

from .cirq_shadows import ShadowSnapshot
from .kalman_tracker import joseph_update
from .spam_mitigation import ReadoutErrorModel

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
#: Rb-87 C6 used by Bloqade (``bloqade.analog.constants.RB_C6 = 2π × 862690``), rad μm^6/μs.
#: Wurtz et al. (2023) quote 5,420,503 rad μm^6/μs for the 70S state, 1e-5 away.
C6_RB87 = 2.0 * np.pi * 862690.0


@dataclass(frozen=True)
class AquilaSpecs:
    """
    Operating limits and error rates of QuEra's Aquila. Unless marked otherwise
    the values are from the datasheet in section 1.5 of Wurtz et al.,
    arXiv:2306.11727 (2023). The waveform rules marked "Braket" are from the
    device capabilities printed in the Amazon Braket example notebook
    ``01_Introduction_to_Aquila`` (amazon-braket-examples), which also states
    that Ω(t) must start and end at 0 and the phase must start at 0. Rates in
    rad/μs, lengths in μm, times in μs.
    """

    omega_max: float = 15.8                 # maximum Rabi frequency
    omega_slew_max: float = 250.0           # |dΩ/dt| in rad/μs² (datasheet; Braket currently lists 400)
    delta_max: float = 125.0                # |Δ| ≤ delta_max
    delta_slew_max: float = 6000.0          # |dΔ/dt| in rad/μs² (Braket)
    t_max: float = 4.0                      # maximum duration of the drive
    time_delta_min: float = 0.05            # minimum spacing of waveform time points (Braket timeDeltaMin)
    time_resolution: float = 0.001          # waveform time points are multiples of 1 ns (Braket)
    min_spacing: float = 4.0                # minimum distance between sites
    min_row_spacing: float = 4.0            # minimum vertical spacing between rows
    width: float = 75.0                     # maximum pattern width
    height: float = 76.0                    # maximum pattern height
    position_resolution: float = 0.01       # site coordinates are multiples of 0.01 μm (Braket positionResolution)
    position_systematic: float = 0.05       # δx, δy: systematic pattern-dependent site error
    position_jitter: float = 0.2            # σx, σy: random position error from thermal motion, per in-plane coordinate
    p_vacancy: float = 0.007                # ε_fill: probability that a site specified as filled is empty
    p_ground_as_rydberg: float = 0.01       # ε_det-gnd
    p_rydberg_as_ground: float = 0.08       # ε_det-ryd
    delta_shot_to_shot: float = 0.18        # RMS shot-to-shot detuning variation
    omega_shot_to_shot_rel: float = 0.008   # RMS relative shot-to-shot Rabi variation
    delta_inhomogeneity: float = 0.37       # RMS detuning inhomogeneity over the user region
    omega_inhomogeneity_rel: float = 0.02   # RMS relative Rabi inhomogeneity
    delta_systematic: float = 0.63          # systematic error in global detuning
    t2_star: float = 5.8
    t2_echo: float = 11.4
    t2_rabi: float = 7.5
    c6: float = C6_RB87

    def readout_model(self) -> ReadoutErrorModel:
        """Asymmetric readout errors as a :class:`ReadoutErrorModel` (bit 1 = Rydberg)."""
        return ReadoutErrorModel(p01=self.p_ground_as_rydberg, p10=self.p_rydberg_as_ground)

    def minimum_ramp(self, omega: float) -> float:
        """Shortest allowed linear Ω ramp from 0 to ``omega``: the slew limit or the minimum time step, on the 1 ns grid."""
        r = max(abs(float(omega)) / self.omega_slew_max, self.time_delta_min)
        return float(np.ceil(r / self.time_resolution - 1e-9) * self.time_resolution)

    @classmethod
    def braket(cls) -> "AquilaSpecs":
        """
        Aquila's current figures from the Braket device properties printed in the
        ``01_Introduction_to_Aquila`` example notebook, where they differ from the
        2023 datasheet: Ω slew 400 rad/μs², readout errors 0.05 (ground read as
        Rydberg) and 0.10 (Rydberg read as ground), T2* 4.75 μs, T2 echo and
        T2 Rabi 7 μs (ensemble values), filling error 0.008, site position error
        0.1 μm, detuning error and detuning inhomogeneity 1 rad/μs. Braket gives
        no shot-to-shot drive figures, so those keep the datasheet values.
        """
        return cls(omega_slew_max=400.0, position_systematic=0.1, p_vacancy=0.008, p_ground_as_rydberg=0.05,
                   p_rydberg_as_ground=0.10, delta_inhomogeneity=1.0, delta_systematic=1.0, t2_star=4.75, t2_echo=7.0,
                   t2_rabi=7.0)


#: The 2023 datasheet (Wurtz et al.), the reference for every number in the README.
AQUILA = AquilaSpecs()
#: The current Braket device properties (see :meth:`AquilaSpecs.braket`).
AQUILA_BRAKET = AquilaSpecs.braket()

#: Default Ω ramp: 0.05 μs, the minimum time step, which respects the 250 rad/μs² slew limit for Ω ≤ 12.5 rad/μs.
DEFAULT_RAMP = 0.05


def blockade_radius(omega: float, delta: float = 0.0, c6: float = C6_RB87) -> float:
    """R_b = (C6 / √(Ω² + Δ²))^{1/6} (Wurtz et al. 2023, section 3)."""
    return float((c6 / np.hypot(omega, delta)) ** (1.0 / 6.0))


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def snap(positions: np.ndarray, resolution: float = AQUILA.position_resolution) -> np.ndarray:
    """Round coordinates to the device's position grid (0.01 μm on Aquila)."""
    return np.round(np.asarray(positions, dtype=float) / resolution) * resolution


def chain(n_atoms: int, spacing: float) -> np.ndarray:
    """A straight chain along x, positions in μm."""
    return np.column_stack([np.arange(int(n_atoms)) * float(spacing), np.zeros(int(n_atoms))])


def ring(n_atoms: int, spacing: float, resolution: Optional[float] = AQUILA.position_resolution) -> np.ndarray:
    """
    A regular ring with nearest-neighbour distance ``spacing``. Coordinates are
    rounded to ``resolution`` (pass ``None`` for the exact polygon), so a
    distance can differ from ``spacing`` by up to about 0.01 μm.
    """
    n = int(n_atoms)
    radius = float(spacing) / (2.0 * np.sin(np.pi / n))
    ang = 2.0 * np.pi * np.arange(n) / n
    pos = radius * np.column_stack([np.cos(ang), np.sin(ang)])
    pos = pos - pos.min(axis=0)
    return pos if resolution is None else snap(pos, resolution)


def square_lattice(nx: int, ny: int, spacing: float) -> np.ndarray:
    """An nx × ny square lattice, row by row."""
    ix, iy = np.meshgrid(np.arange(int(nx)), np.arange(int(ny)))
    return float(spacing) * np.column_stack([ix.ravel(), iy.ravel()]).astype(float)


def geometry_violations(positions: np.ndarray, specs: AquilaSpecs = AQUILA, tol: float = 1e-9) -> List[str]:
    """
    Aquila layout rules broken by ``positions``: sites closer than the minimum
    spacing, a pattern larger than the user region, rows (distinct y values)
    closer than the minimum row spacing, and coordinates off the 0.01 μm
    position grid. An empty list means none of these checks failed; Braket
    validates further details on submission. Rings and other off-row layouts
    often fail the row rule.
    """
    pos = np.atleast_2d(np.asarray(positions, dtype=float))
    out: List[str] = []
    off = np.abs(pos / specs.position_resolution - np.round(pos / specs.position_resolution)) * specs.position_resolution
    if np.any(off > 1e-6):
        out.append(f"coordinates are not multiples of {specs.position_resolution} μm (largest offset {float(off.max()):.4f} μm); see snap()")
    n = len(pos)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(pos[i] - pos[j]))
            if d < specs.min_spacing - tol:
                out.append(f"sites {i} and {j} are {d:.2f} μm apart (minimum {specs.min_spacing} μm)")
    ext = pos.max(axis=0) - pos.min(axis=0)
    if ext[0] > specs.width + tol:
        out.append(f"pattern width {ext[0]:.1f} μm exceeds {specs.width} μm")
    if pos.shape[1] > 1:
        if ext[1] > specs.height + tol:
            out.append(f"pattern height {ext[1]:.1f} μm exceeds {specs.height} μm")
        ys = np.unique(np.round(pos[:, 1], 9))
        gaps = np.diff(ys)
        if np.any(gaps < specs.min_row_spacing - tol):
            out.append(f"rows {float(gaps.min()):.2f} μm apart (minimum {specs.min_row_spacing} μm)")
    return out


# --------------------------------------------------------------------------- #
# noise and data containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DeviceNoise:
    """
    Noise beyond readout errors, in two parts.

    * Shot-to-shot Gaussian fluctuations of the global drive, constant within a
      shot: Δ → Δ + σ_Δ ξ and Ω → Ω (1 + σ_Ω ξ'). Outcome probabilities are
      averaged over ξ, ξ' with Gauss–Hermite quadrature of ``n_nodes`` points each.
    * Optional phenomenological relaxation (``t_relax`` in μs): at probe time t
      each atom's outcome is replaced by a fair coin with probability
      1 − e^{−t/t_relax}, so every single-atom contrast decays as e^{−t/t_relax}.
      This stands in for decoherence that the closed-system model leaves out; it
      is not a master equation.
    """

    sigma_delta: float = 0.0
    sigma_omega_rel: float = 0.0
    t_relax: Optional[float] = None
    n_nodes: int = 5

    @classmethod
    def aquila(cls, t_relax: Optional[float] = AQUILA.t2_echo, n_nodes: int = 5) -> "DeviceNoise":
        """Datasheet shot-to-shot drive noise, with relaxation at T2_echo (incoherent processes only) by default."""
        return cls(AQUILA.delta_shot_to_shot, AQUILA.omega_shot_to_shot_rel, t_relax, n_nodes)

    @classmethod
    def braket(cls, t_relax: Optional[float] = AQUILA_BRAKET.t2_echo, n_nodes: int = 5) -> "DeviceNoise":
        """As :meth:`aquila`, with relaxation at the current Braket T2_echo (7 μs) by default."""
        return cls(AQUILA_BRAKET.delta_shot_to_shot, AQUILA_BRAKET.omega_shot_to_shot_rel, t_relax, n_nodes)

    @property
    def quasi_static(self) -> bool:
        return self.sigma_delta != 0.0 or self.sigma_omega_rel != 0.0

    @property
    def trivial(self) -> bool:
        return not self.quasi_static and not self.t_relax

    def nodes(self) -> List[Tuple[float, float, float]]:
        """(relative Ω offset, Δ offset, weight) for every quadrature node."""
        if not self.quasi_static:
            return [(0.0, 0.0, 1.0)]
        x, w = np.polynomial.hermite_e.hermegauss(int(self.n_nodes))
        w = w / w.sum()
        xd, wd = (x, w) if self.sigma_delta > 0 else (np.zeros(1), np.ones(1))
        xo, wo = (x, w) if self.sigma_omega_rel > 0 else (np.zeros(1), np.ones(1))
        return [(self.sigma_omega_rel * a, self.sigma_delta * b, float(u * v)) for a, u in zip(xo, wo) for b, v in zip(xd, wd)]

    def flip_probability(self, times: np.ndarray) -> np.ndarray:
        """Per-atom probability that relaxation has randomised the outcome by each probe time, halved (a coin can land right)."""
        times = np.atleast_1d(np.asarray(times, dtype=float))
        if not self.t_relax:
            return np.zeros_like(times)
        return 0.5 * (1.0 - np.exp(-times / float(self.t_relax)))


def _shot_succeeded(status) -> bool:
    """True for a Braket shot status of "Success" (a string or a ``ShotStatus`` member)."""
    return str(getattr(status, "value", status)).strip().lower() == "success"


def _shot_rows(rows: Sequence, keep: Sequence[bool], n_sites: int) -> np.ndarray:
    """The kept rows as an (S, n_sites) integer array."""
    kept = [np.asarray(r, dtype=np.int64) for r, g in zip(rows, keep) if g]
    return np.stack(kept) if kept else np.zeros((0, n_sites), dtype=np.int64)


@dataclass
class QuenchData:
    """
    Shots at each probe time: ``bits[k]`` is an (S_k, n) uint8 array, 1 = read
    as Rydberg. ``filled[k]``, when present, is the matching pre-sequence image
    (1 = site occupied before the drive). An empty site reads as Rydberg in the
    post-sequence image, so shots with a missing atom should be dropped with
    :meth:`postselect` before fitting, as Wurtz et al. (2023) advise.
    """

    times: np.ndarray
    bits: List[np.ndarray]
    filled: Optional[List[np.ndarray]] = None

    @classmethod
    def from_braket(cls, times: Sequence[float], pre_sequences: Sequence, post_sequences: Sequence,
                    postselect: bool = True, filling: Optional[Sequence[int]] = None,
                    statuses: Optional[Sequence[Sequence]] = None) -> "QuenchData":
        """
        Build from Braket results. ``pre_sequences`` and ``post_sequences`` hold
        one entry per probe time, each with one row per shot and one column per
        register site. ``post_sequence`` uses 1 for an atom found in the ground
        state, so bits = 1 − post_sequence.

        ``filling`` is the requested filling of the register, 1 or 0 per site
        (``ahs_register.filling`` in the program). Sites requested empty are
        dropped, so the bits line up with a model built from the filled sites
        alone. Without it every site is taken as requested filled.

        ``statuses``, one sequence per probe time, are the shots' ``status``
        values. Shots whose status is not "Success" are dropped, and their rows
        may be None.

        With ``postselect`` (the default) only shots whose pre-sequence matches
        the requested filling are kept: every requested site occupied and, with
        ``filling``, every other site empty. Braket's Aquila notebook and
        Wurtz et al. (2023) both advise this comparison.
        """
        times = np.atleast_1d(np.asarray(times, dtype=float))
        if not len(pre_sequences) == len(post_sequences) == len(times):
            raise ValueError("need one pre_sequence and one post_sequence array per probe time")
        if statuses is not None and len(statuses) != len(times):
            raise ValueError("need one sequence of shot statuses per probe time")
        want = None if filling is None else np.asarray(filling, dtype=np.int64)
        bits, filled = [], []
        for k, (pre, post) in enumerate(zip(pre_sequences, post_sequences)):
            ok = [True] * len(post) if statuses is None else [_shot_succeeded(x) for x in statuses[k]]
            if len(ok) != len(post) or len(pre) != len(post):
                raise ValueError(f"probe time {k}: pre_sequence, post_sequence and statuses differ in length")
            n_sites = len(want) if want is not None else next((len(r) for r, g in zip(post, ok) if g), 0)
            pre_k, post_k = _shot_rows(pre, ok, n_sites), _shot_rows(post, ok, n_sites)
            if want is not None:
                if postselect:                   # a stray atom in a site requested empty would take part in the dynamics
                    clean = np.all(pre_k[:, want == 0] == 0, axis=1)
                    pre_k, post_k = pre_k[clean], post_k[clean]
                pre_k, post_k = pre_k[:, want == 1], post_k[:, want == 1]
            bits.append((1 - post_k).astype(np.uint8))
            filled.append(pre_k.astype(np.uint8))
        data = cls(times=times, bits=bits, filled=filled)
        return data.postselect() if postselect else data

    def postselect(self) -> "QuenchData":
        """Keep only the shots in which every site was filled (no-op without ``filled``)."""
        if self.filled is None:
            return self
        keep = [np.all(f == 1, axis=1) for f in self.filled]
        return QuenchData(times=self.times, bits=[b[k] for b, k in zip(self.bits, keep)],
                          filled=[f[k] for f, k in zip(self.filled, keep)])

    def filled_fraction(self) -> np.ndarray:
        """Fraction of shots at each probe time with every site filled."""
        if self.filled is None:
            return np.ones(len(self.bits))
        return np.array([float(np.mean(np.all(f == 1, axis=1))) for f in self.filled])

    @property
    def n_atoms(self) -> int:
        return int(self.bits[0].shape[1])

    @property
    def shots(self) -> np.ndarray:
        return np.array([len(b) for b in self.bits], dtype=int)

    def counts(self) -> np.ndarray:
        """(K, 2^n) outcome counts."""
        n = self.n_atoms
        weights = 1 << (n - 1 - np.arange(n))
        return np.stack([np.bincount(b.astype(np.int64) @ weights, minlength=1 << n) for b in self.bits]).astype(float)

    def snapshots(self, k: int) -> List[ShadowSnapshot]:
        """Shots at probe time k as Z-basis :class:`ShadowSnapshot` objects, for the generic estimators."""
        n = self.n_atoms
        return [ShadowSnapshot(bases=("Z",) * n, bits=tuple(int(v) for v in row)) for row in self.bits[k]]


def apply_readout(p: np.ndarray, readout) -> np.ndarray:
    """
    Apply the same single-atom confusion matrix M[measured, true] to every atom
    of distributions over 2^n outcomes (last axis). ``readout`` is a
    :class:`ReadoutErrorModel`, a 2 × 2 matrix, or ``None``.
    """
    if readout is None:
        return p
    M = readout.matrix if hasattr(readout, "matrix") else np.asarray(readout, dtype=float)
    dim = p.shape[-1]
    n = int(round(np.log2(dim)))
    lead = p.shape[:-1]
    x = p.reshape(lead + (2,) * n)
    L = len(lead)
    for q in range(n):
        ax = L + q
        x = np.moveaxis(np.moveaxis(x, ax, -1) @ M.T, -1, ax)
    return x.reshape(p.shape)


def _relax(p: np.ndarray, times: np.ndarray, noise: Optional["DeviceNoise"]) -> np.ndarray:
    """Phenomenological relaxation of :class:`DeviceNoise`, applied per probe time (axis 0 of ``p``)."""
    if noise is None or not noise.t_relax:
        return p
    out = np.empty_like(p)
    for k, f in enumerate(noise.flip_probability(times)):
        out[k] = apply_readout(p[k], np.array([[1.0 - f, f], [f, 1.0 - f]]))
    return out


def _flip_bits(bits: np.ndarray, readout: Optional[ReadoutErrorModel], rng: np.random.Generator) -> np.ndarray:
    if readout is None:
        return bits
    p_flip = np.where(bits == 0, readout.p01, readout.p10)
    return (bits ^ (rng.random(bits.shape) < p_flip)).astype(np.uint8)


@dataclass
class RydbergFit:
    """
    Result of :meth:`RydbergQuenchModel.fit_mle` or :meth:`RydbergQuenchModel.fit_correlators`.

    ``success`` is true only if the optimiser converged, the goodness-of-fit
    test does not reject the model (``gof_pvalue`` ≥ 10⁻³), and no other local
    maximum found by the search fits almost as well. ``warnings`` says which
    check failed. ``alternatives`` lists other local optima within
    ``ambiguity`` (log-likelihood units) of the best one, as (θ, Δ objective).
    """

    theta: np.ndarray
    names: Tuple[str, ...]
    fitted: Tuple[str, ...]
    covariance: np.ndarray        # over the fitted parameters, inverse Fisher information at the estimate
    objective: float
    n_evaluations: int
    method: str
    success: bool
    converged: bool = True
    gof_chi2: float = float("nan")
    gof_dof: int = 0
    gof_pvalue: float = float("nan")
    alternatives: List[Tuple[np.ndarray, float]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def std(self) -> Dict[str, float]:
        return {nm: float(np.sqrt(max(self.covariance[k, k], 0.0))) for k, nm in enumerate(self.fitted)}

    def as_dict(self) -> Dict[str, float]:
        return {nm: float(v) for nm, v in zip(self.names, self.theta)}


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
class RydbergQuenchModel:
    """
    Dynamics of a small Rydberg array from |g…g⟩ under one Aquila-style pulse.

    Parameters
    ----------
    positions : (n, 2) or (n, 3) array of atom positions in μm
    local_detuning : add one detuning offset δ_i per site to the parameters
    ramp_time : duration of the linear Ω ramps at each end of the pulse, in μs.
        The default 0.05 μs is Braket's minimum time step and keeps |dΩ/dt|
        within Aquila's 250 rad/μs² for Ω ≤ 12.5 rad/μs. ``ramp_time=0`` gives
        an instantaneous quench, which cannot be programmed on the device
        (Ω must start and end at 0) but is useful as a reference.
    ramp_slices : piecewise-constant slices per ramp (midpoint rule)

    The pulse for probe time T is Ω(t) = Ω·s(t) with s rising linearly from 0
    to 1 over ``ramp_time``, flat, and falling back to 0 at T; Δ(t) = Δ is
    constant and the phase is 0. Atoms are read out at T, after Ω has returned
    to 0, so a probe time is the total pulse duration and must be at least
    2·ramp_time. :meth:`waveform` returns the Braket time series.

    The parameter vector is θ = (Ω, Δ, C6[, δ_0 … δ_{n−1}]); names are in
    ``param_names``. Arrays up to ``DENSE_MAX_ATOMS`` atoms use dense
    eigendecompositions (exact gradients, Fisher information, fitting); larger
    arrays, up to 14 atoms, can be simulated and sampled with sparse
    ``expm_multiply`` propagation.
    """

    DENSE_MAX_ATOMS = 10
    MAX_ATOMS = 14

    def __init__(self, positions: np.ndarray, local_detuning: bool = False, ramp_time: float = DEFAULT_RAMP,
                 ramp_slices: int = 8):
        pos = np.atleast_2d(np.asarray(positions, dtype=float))
        if pos.ndim != 2 or pos.shape[1] not in (1, 2, 3):
            raise ValueError("positions must be an (n, d) array with d in {1, 2, 3}")
        if pos.shape[0] > self.MAX_ATOMS:
            raise ValueError(f"at most {self.MAX_ATOMS} atoms")
        if ramp_time < 0 or int(ramp_slices) < 1:
            raise ValueError("ramp_time must be non-negative and ramp_slices positive")
        self.ramp_time = float(ramp_time)
        self.ramp_slices = int(ramp_slices)
        self._ramp_shape = (np.arange(self.ramp_slices) + 0.5) / self.ramp_slices if self.ramp_time > 0 else np.zeros(0)
        self.positions = pos
        self.n = int(pos.shape[0])
        self.dim = 1 << self.n
        self.local_detuning = bool(local_detuning)
        names = ["omega", "delta", "c6"]
        if self.local_detuning:
            names += [f"delta_{i}" for i in range(self.n)]
        self.param_names: Tuple[str, ...] = tuple(names)
        self.n_params = len(names)
        idx = np.arange(self.dim)
        self.bits = ((idx[:, None] >> (self.n - 1 - np.arange(self.n))[None, :]) & 1).astype(np.uint8)
        self.pairs = [(i, j) for i in range(self.n) for j in range(i + 1, self.n)]
        b = self.bits.astype(float)
        self._pair_bits = np.stack([b[:, i] * b[:, j] for i, j in self.pairs], axis=1) if self.pairs else np.zeros((self.dim, 0))
        self.distances = self._pair_distances(pos)
        self._g_delta = -b.sum(axis=1)
        self._g_c6 = self._pair_bits @ self.distances ** -6.0 if self.pairs else np.zeros(self.dim)
        rows, cols = [], []
        for q in range(self.n):
            rows.append(idx)
            cols.append(idx ^ (1 << (self.n - 1 - q)))
        self._x_half = scipy.sparse.csr_matrix((np.full(self.n * self.dim, 0.5), (np.concatenate(rows), np.concatenate(cols))),
                                               shape=(self.dim, self.dim))
        self._x_dense: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    def _pair_distances(self, pos: np.ndarray) -> np.ndarray:
        if not self.pairs:
            return np.zeros(0)
        i, j = np.array(self.pairs).T
        return np.linalg.norm(pos[i] - pos[j], axis=1)

    def with_positions(self, positions: np.ndarray) -> "RydbergQuenchModel":
        return RydbergQuenchModel(positions, local_detuning=self.local_detuning, ramp_time=self.ramp_time,
                                  ramp_slices=self.ramp_slices)

    def with_ramp(self, ramp_time: float, ramp_slices: Optional[int] = None) -> "RydbergQuenchModel":
        """The same array with a different pulse shape (``ramp_time=0`` for an instantaneous quench)."""
        return RydbergQuenchModel(self.positions, local_detuning=self.local_detuning, ramp_time=ramp_time,
                                  ramp_slices=self.ramp_slices if ramp_slices is None else ramp_slices)

    def waveform(self, theta: np.ndarray, duration: float) -> Dict[str, Dict[str, List[float]]]:
        """
        Piecewise-linear time series for one probe time, in μs and rad/μs, in
        the layout of a Braket ``DrivingField`` (multiply by 1e-6 and 1e6 for SI).
        """
        th = np.asarray(theta, dtype=float)
        T, r = float(duration), self.ramp_time
        om, de = float(th[0]), float(th[1])
        if r > 0 and T - 2 * r > 1e-12:
            amp = {"times": [0.0, r, T - r, T], "values": [0.0, om, om, 0.0]}
        elif r > 0:
            amp = {"times": [0.0, r, T], "values": [0.0, om, 0.0]}
        else:
            amp = {"times": [0.0, T], "values": [om, om]}
        return {"amplitude": amp, "detuning": {"times": [0.0, T], "values": [de, de]},
                "phase": {"times": [0.0, T], "values": [0.0, 0.0]}}

    def waveform_violations(self, theta: np.ndarray, times: Sequence[float], specs: AquilaSpecs = AQUILA) -> List[str]:
        """
        Braket/Aquila rules the pulses for ``times`` would break: Ω and Δ
        ranges, the Ω slew limit, Ω starting and ending at 0, the minimum time
        step, the 1 ns time grid and the maximum duration. Empty if none of
        these checks fails.
        """
        th = np.asarray(theta, dtype=float)
        out: List[str] = []
        r = self.ramp_time
        tol = 1e-9
        if not 0.0 <= th[0] <= specs.omega_max + tol:
            out.append(f"Ω = {th[0]:g} rad/μs outside [0, {specs.omega_max}]")
        if abs(th[1]) > specs.delta_max + tol:
            out.append(f"|Δ| = {abs(th[1]):g} rad/μs exceeds {specs.delta_max}")
        if r == 0:
            out.append("Ω must start and end at 0: an instantaneous quench cannot be programmed")
        else:
            if abs(th[0]) / r > specs.omega_slew_max * (1 + 1e-9):
                out.append(f"Ω ramp slope {abs(th[0]) / r:.0f} rad/μs² exceeds {specs.omega_slew_max:g}")
            if r < specs.time_delta_min - tol:
                out.append(f"ramp of {r:g} μs is shorter than the {specs.time_delta_min} μs minimum time step")
        for T in np.atleast_1d(np.asarray(times, dtype=float)):
            if T > specs.t_max + tol:
                out.append(f"duration {T:g} μs exceeds {specs.t_max} μs")
            plateau = T - 2 * r
            if plateau < -tol or (r > 0 and tol < plateau < specs.time_delta_min - tol):
                out.append(f"duration {T:g} μs: flat part {plateau:.3f} μs must be 0 or at least {specs.time_delta_min} μs")
            for x in self.waveform(th, T)["amplitude"]["times"]:
                if abs(x / specs.time_resolution - round(x / specs.time_resolution)) > 1e-6:
                    out.append(f"time point {x:g} μs is not a multiple of {specs.time_resolution * 1e3:g} ns")
                    break
        return out

    def theta(self, omega: float, delta: float, c6: float = C6_RB87, delta_local: Optional[Sequence[float]] = None) -> np.ndarray:
        """Assemble a parameter vector."""
        th = [float(omega), float(delta), float(c6)]
        if self.local_detuning:
            th += list(np.zeros(self.n) if delta_local is None else np.asarray(delta_local, dtype=float))
        elif delta_local is not None:
            raise ValueError("model was built without local_detuning")
        return np.array(th)

    def interaction_matrix(self, c6: float = C6_RB87) -> np.ndarray:
        V = np.zeros((self.n, self.n))
        for (i, j), r in zip(self.pairs, self.distances):
            V[i, j] = V[j, i] = c6 / r**6
        return V

    def _diag(self, theta: np.ndarray, g_c6: Optional[np.ndarray] = None) -> np.ndarray:
        th = np.asarray(theta, dtype=float)
        d = th[1] * self._g_delta + th[2] * (self._g_c6 if g_c6 is None else g_c6)
        if self.local_detuning:
            d = d - self.bits @ th[3:]
        return d

    def _xd(self) -> np.ndarray:
        if self._x_dense is None:
            self._x_dense = self._x_half.toarray()
        return self._x_dense

    def hamiltonian(self, theta: np.ndarray, sparse: bool = False):
        """H(θ) as a dense array or a CSR matrix."""
        H = float(theta[0]) * self._x_half + scipy.sparse.diags(self._diag(theta))
        return H.tocsr() if sparse else H.toarray()

    def _dense_h(self, theta: np.ndarray, g_c6: Optional[np.ndarray] = None) -> np.ndarray:
        H = float(theta[0]) * self._xd()
        H[np.diag_indices(self.dim)] += self._diag(theta, g_c6)
        return H

    def _generator_diag(self, a: int) -> np.ndarray:
        if a == 1:
            return self._g_delta
        if a == 2:
            return self._g_c6
        return -self.bits[:, a - 3].astype(float)

    def _check_dense(self) -> None:
        if self.n > self.DENSE_MAX_ATOMS:
            raise ValueError(f"exact gradients and fitting need at most {self.DENSE_MAX_ATOMS} atoms")

    # ------------------------------------------------------------------ #
    # forward model
    # ------------------------------------------------------------------ #
    def _check_times(self, times: np.ndarray) -> None:
        if np.any(times < 0):
            raise ValueError("probe times must be non-negative")
        if self.ramp_time > 0 and np.any(times < 2 * self.ramp_time - 1e-12):
            raise ValueError(f"with {self.ramp_time} μs ramps the shortest probe time is {2 * self.ramp_time} μs")

    def amplitudes(self, theta: np.ndarray, times: Sequence[float], g_c6: Optional[np.ndarray] = None) -> np.ndarray:
        """ψ(T_k) at the end of the pulse of total duration T_k, shape (K, 2^n)."""
        times = np.atleast_1d(np.asarray(times, dtype=float))
        self._check_times(times)
        if self.n <= self.DENSE_MAX_ATOMS:
            return self._evolve(np.asarray(theta, dtype=float), times, None, g_c6)[0]
        th = np.asarray(theta, dtype=float)
        D = scipy.sparse.diags(self._diag(th, g_c6))
        psi = np.zeros(self.dim, dtype=complex)
        psi[0] = 1.0
        dt = self.ramp_time / self.ramp_slices
        for sh in self._ramp_shape:
            psi = scipy.sparse.linalg.expm_multiply(-1j * dt * (th[0] * sh * self._x_half + D).tocsc(), psi)
        H = (th[0] * self._x_half + D).tocsc().astype(complex)
        out = np.empty((len(times), self.dim), dtype=complex)
        tau = times - 2 * self.ramp_time
        t_prev = 0.0
        for k in np.argsort(tau):
            if tau[k] > t_prev:
                psi = scipy.sparse.linalg.expm_multiply(-1j * (tau[k] - t_prev) * H, psi)
                t_prev = tau[k]
            out[k] = psi
        for sh in self._ramp_shape[::-1]:
            Hs = (th[0] * sh * self._x_half + D).tocsc()
            out = scipy.sparse.linalg.expm_multiply(-1j * dt * Hs, out.T).T
        return out

    def _generators(self, V: np.ndarray, wrt: Sequence[int], omega_scale: float) -> List[np.ndarray]:
        """∂H/∂θ_a in the eigenbasis V, with the Ω generator scaled by the pulse shape."""
        out = []
        for a in wrt:
            if a == 0:
                out.append(omega_scale * (V.T @ (self._xd() @ V)))
            else:
                out.append(V.T @ (self._generator_diag(a)[:, None] * V))
        return out

    @staticmethod
    def _duhamel(w: np.ndarray, t: float) -> np.ndarray:
        """F_mn(t) = −i t e^{−i(w_m + w_n)t/2} sinc((w_m − w_n)t/2): ∂e^{−iHt} = V (G̃ ∘ F) Vᵀ for H real symmetric."""
        wbar = 0.5 * (w[:, None] + w[None, :])
        dw = w[:, None] - w[None, :]
        return (-1j * t) * np.exp(-1j * wbar * t) * np.sinc(dw * (t / (2.0 * np.pi)))

    def _slice(self, th: np.ndarray, shape: float, dt: float, wrt: Optional[Sequence[int]], g_c6: Optional[np.ndarray]):
        """Eigendecomposition of one ramp slice, its propagator phases and, if needed, its Duhamel-weighted generators."""
        H = (th[0] * shape) * self._xd()
        H[np.diag_indices(self.dim)] += self._diag(th, g_c6)
        w, V = np.linalg.eigh(H)
        GF = None
        if wrt is not None:
            Fm = self._duhamel(w, dt)
            GF = [G * Fm for G in self._generators(V, wrt, shape)]
        return np.exp(-1j * w * dt), V, GF

    def _apply_slice(self, sl, psi: np.ndarray, dpsi: Optional[np.ndarray]):
        """Apply one slice to a batch of states (dim, B) and their derivatives (dim, B, p)."""
        e, V, GF = sl
        c = V.T @ psi
        psi_new = V @ (e[:, None] * c)
        if dpsi is None:
            return psi_new, None
        B, P = dpsi.shape[1], dpsi.shape[2]
        cd = (V.T @ dpsi.reshape(self.dim, B * P)).reshape(self.dim, B, P)
        dpsi_new = (V @ (e[:, None] * cd.reshape(self.dim, B * P))).reshape(self.dim, B, P)
        for j, M in enumerate(GF):
            dpsi_new[:, :, j] += V @ (M @ c)
        return psi_new, dpsi_new

    def _evolve(self, th: np.ndarray, times: np.ndarray, wrt: Optional[Sequence[int]] = None,
                g_c6: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Final states ψ(T_k) (K, dim) and, if ``wrt`` is given, ∂ψ/∂θ_a (K, dim, p),
        by exact propagation through the ramp slices and the flat part. Every
        slice Hamiltonian is real symmetric, so the ramp down, which applies the
        same slices in reverse order, is the transpose of the ramp up and reuses
        its eigendecompositions.
        """
        P = 0 if wrt is None else len(wrt)
        psi = np.zeros((self.dim, 1), dtype=complex)
        psi[0, 0] = 1.0
        dpsi = np.zeros((self.dim, 1, P), dtype=complex) if wrt is not None else None
        dt = self.ramp_time / self.ramp_slices
        slices = [self._slice(th, sh, dt, wrt, g_c6) for sh in self._ramp_shape]
        for sl in slices:
            psi, dpsi = self._apply_slice(sl, psi, dpsi)
        # flat part, for every probe time at once
        H = th[0] * self._xd()
        H[np.diag_indices(self.dim)] += self._diag(th, g_c6)
        w, V = np.linalg.eigh(H)
        tau = times - 2 * self.ramp_time
        c = (V.T @ psi)[:, 0]
        E = np.exp(-1j * np.outer(w, tau))                       # (dim, K)
        out = V @ (E * c[:, None])                               # (dim, K)
        dout = None
        if wrt is not None:
            K = len(times)
            dout = np.empty((self.dim, K, P), dtype=complex)
            cd = V.T @ dpsi[:, 0, :]                             # (dim, P)
            Gt = self._generators(V, wrt, 1.0)
            for k, t in enumerate(tau):
                dout[:, k, :] = V @ (E[:, k][:, None] * cd)
                if t > 0:
                    Fm = self._duhamel(w, t)
                    for j, G in enumerate(Gt):
                        dout[:, k, j] += V @ ((G * Fm) @ c)
        for sl in slices[::-1]:
            out, dout = self._apply_slice(sl, out, dout)
        return out.T, (None if dout is None else np.transpose(dout, (1, 0, 2)))

    def probabilities(self, theta: np.ndarray, times: Sequence[float], readout: Optional[ReadoutErrorModel] = None,
                      noise: Optional[DeviceNoise] = None) -> np.ndarray:
        """Outcome distributions (K, 2^n) with ``noise`` (see :class:`DeviceNoise`) and readout errors applied."""
        th = np.asarray(theta, dtype=float)
        times = np.atleast_1d(np.asarray(times, dtype=float))
        p = 0.0
        for eo, ed, wgt in (noise.nodes() if noise is not None else [(0.0, 0.0, 1.0)]):
            t2 = th.copy(); t2[0] *= 1.0 + eo; t2[1] += ed
            a = self.amplitudes(t2, times)
            p = p + wgt * (a.real**2 + a.imag**2)
        return apply_readout(_relax(p, times, noise), readout)

    def _closed_jacobian(self, theta: np.ndarray, times: np.ndarray, wrt: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        self._check_times(times)
        psi, dpsi = self._evolve(np.asarray(theta, dtype=float), times, wrt)
        p = psi.real**2 + psi.imag**2
        dp = 2.0 * np.real(np.conj(psi)[:, :, None] * dpsi)
        return p, dp

    def probabilities_and_jacobian(self, theta: np.ndarray, times: Sequence[float], readout: Optional[ReadoutErrorModel] = None,
                                   noise: Optional[DeviceNoise] = None, wrt: Optional[Sequence[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Outcome distributions q (K, 2^n) and their exact derivatives ∂q/∂θ_a,
        shape (K, 2^n, len(wrt)); ``wrt`` lists parameter indices (default all).
        """
        self._check_dense()
        th = np.asarray(theta, dtype=float)
        times = np.atleast_1d(np.asarray(times, dtype=float))
        wrt = list(range(self.n_params)) if wrt is None else [int(a) for a in wrt]
        if noise is None or not noise.quasi_static:
            p, dp = self._closed_jacobian(th, times, wrt)
        else:
            p = np.zeros((len(times), self.dim)); dp = np.zeros((len(times), self.dim, len(wrt)))
            for eo, ed, wgt in noise.nodes():
                t2 = th.copy(); t2[0] *= 1.0 + eo; t2[1] += ed
                pj, dpj = self._closed_jacobian(t2, times, wrt)
                if 0 in wrt:
                    dpj[..., wrt.index(0)] *= 1.0 + eo
                p += wgt * pj
                dp += wgt * dpj
        if noise is not None and noise.t_relax:
            p = _relax(p, times, noise)
            dp = np.moveaxis(_relax(np.moveaxis(dp, -1, 1), times, noise), 1, -1)
        if readout is not None:
            p = apply_readout(p, readout)
            dp = np.moveaxis(apply_readout(np.moveaxis(dp, -1, 1), readout), 1, -1)
        return p, dp

    def averaged_probabilities(self, theta: np.ndarray, times: Sequence[float], position_jitter: float, n_draws: int,
                               rng: np.random.Generator, readout: Optional[ReadoutErrorModel] = None,
                               return_batches: int = 0):
        """
        Outcome distributions averaged over independent Gaussian displacements of
        every atom (σ = ``position_jitter`` μm per in-plane coordinate, frozen
        during a shot), estimated from ``n_draws`` random configurations. With
        ``return_batches = B > 0`` also returns B batch means for an error bar.
        """
        th = np.asarray(theta, dtype=float)
        times = np.atleast_1d(np.asarray(times, dtype=float))
        acc = np.zeros((len(times), self.dim))
        batches = []
        bsize = max(1, n_draws // return_batches) if return_batches else n_draws
        part = np.zeros_like(acc); count = 0
        for d in range(int(n_draws)):
            g = self._displaced_g_c6(position_jitter, rng)
            a = self.amplitudes(th, times, g_c6=g)
            pd = a.real**2 + a.imag**2
            acc += pd
            part += pd; count += 1
            if return_batches and count == bsize:
                batches.append(part / count); part = np.zeros_like(acc); count = 0
        mean = apply_readout(acc / n_draws, readout)
        if return_batches:
            return mean, [apply_readout(b, readout) for b in batches]
        return mean

    def _displaced_g_c6(self, sigma: float, rng: np.random.Generator) -> np.ndarray:
        disp = self.positions.copy()
        disp[:, :2] += sigma * rng.standard_normal((self.n, min(2, self.positions.shape[1])))
        return self._pair_bits @ self._pair_distances(disp) ** -6.0

    # ------------------------------------------------------------------ #
    # sampling
    # ------------------------------------------------------------------ #
    def sample(self, theta: np.ndarray, times: Sequence[float], shots: Union[int, Sequence[int]], rng: np.random.Generator,
               readout: Optional[ReadoutErrorModel] = None, position_jitter: float = 0.0,
               noise: Optional[DeviceNoise] = None, p_vacancy: float = 0.0) -> QuenchData:
        """
        Simulated Z-basis shots. ``position_jitter`` displaces every atom by an
        independent Gaussian (σ in μm per in-plane coordinate) drawn afresh for each
        shot; ``noise`` draws Ω and Δ afresh for each shot and applies its
        relaxation. Readout errors flip each bit independently.

        With ``p_vacancy`` > 0 each site is empty with that probability in each
        shot (Aquila's ε_fill is 0.007). The remaining atoms evolve as the
        smaller array, an empty site reads as Rydberg (bit 1), and the returned
        data carry the ``filled`` images so that :meth:`QuenchData.postselect`
        can drop those shots.
        """
        th = np.asarray(theta, dtype=float)
        times = np.atleast_1d(np.asarray(times, dtype=float))
        S = np.broadcast_to(np.asarray(shots, dtype=int), times.shape)
        out: List[np.ndarray] = []
        quasi = noise is not None and noise.quasi_static
        flips = noise.flip_probability(times) if noise is not None else np.zeros(len(times))
        if position_jitter <= 0 and not quasi:
            P = self.probabilities(th, times)
            idx_k = []
            for k in range(len(times)):
                pk = np.clip(P[k], 0.0, None); pk /= pk.sum()
                idx_k.append(rng.choice(self.dim, size=int(S[k]), p=pk))
        else:
            idx_k = []
            for k, t in enumerate(times):
                idx = np.empty(int(S[k]), dtype=np.int64)
                for s in range(int(S[k])):
                    t2 = th.copy()
                    if quasi:
                        t2[0] *= 1.0 + noise.sigma_omega_rel * rng.standard_normal()
                        t2[1] += noise.sigma_delta * rng.standard_normal()
                    g = self._displaced_g_c6(position_jitter, rng) if position_jitter > 0 else None
                    a = self.amplitudes(t2, [t], g_c6=g)[0]
                    pk = a.real**2 + a.imag**2
                    idx[s] = rng.choice(self.dim, p=pk / pk.sum())
                idx_k.append(idx)
        for k, idx in enumerate(idx_k):
            b = self.bits[idx]
            if flips[k] > 0:
                b = (b ^ (rng.random(b.shape) < flips[k])).astype(np.uint8)
            out.append(_flip_bits(b, readout, rng))
        if p_vacancy <= 0:
            return QuenchData(times=times, bits=out)
        filled: List[np.ndarray] = []
        for k in range(len(times)):
            f = (rng.random(out[k].shape) >= p_vacancy).astype(np.uint8)
            groups: Dict[Tuple[int, ...], List[int]] = {}
            for s in np.flatnonzero(~f.all(axis=1)):
                groups.setdefault(tuple(int(v) for v in f[s]), []).append(int(s))
            for pattern, rows in groups.items():
                present = np.flatnonzero(pattern)
                b = np.ones((len(rows), self.n), dtype=np.uint8)
                if len(present):
                    sub = self.with_positions(self.positions[present])
                    th_sub = np.concatenate([th[:3], th[3 + present]]) if self.local_detuning else th
                    b[:, present] = sub.sample(th_sub, [times[k]], len(rows), rng, readout, position_jitter, noise).bits[0]
                out[k][rows] = b
            filled.append(f)
        return QuenchData(times=times, bits=out, filled=filled)

    # ------------------------------------------------------------------ #
    # correlators and readout correction
    # ------------------------------------------------------------------ #
    def correlator_table(self, readout: Optional[ReadoutErrorModel] = None, pairs: Optional[Sequence[Tuple[int, int]]] = None) -> np.ndarray:
        """
        Per-shot estimators of [n_i, n_i n_j] for every possible measured
        bitstring, shape (2^n, n + n_pairs). With ``readout`` each (−1)^b is
        replaced by the corrected eigenvalue g(b) of :class:`ReadoutErrorModel`,
        so n_i → (1 − g_i)/2 and n_i n_j → (1 − g_i)(1 − g_j)/4, which are
        unbiased because the readout errors of different atoms are independent.
        """
        pairs = self.pairs if pairs is None else list(pairs)
        if readout is None:
            occ = self.bits.astype(float)
        else:
            g = np.where(self.bits == 0, readout.corrected_eigenvalue(0), readout.corrected_eigenvalue(1))
            occ = 0.5 * (1.0 - g)
        cols = [occ] + ([np.stack([occ[:, i] * occ[:, j] for i, j in pairs], axis=1)] if pairs else [])
        return np.concatenate(cols, axis=1)

    def correlators(self, data: QuenchData, readout: Optional[ReadoutErrorModel] = None,
                    pairs: Optional[Sequence[Tuple[int, int]]] = None) -> np.ndarray:
        """Readout-corrected estimates of [⟨n_i⟩, ⟨n_i n_j⟩] at each probe time, shape (K, n + n_pairs)."""
        Y = self.correlator_table(readout, pairs)
        c = data.counts()
        return (c @ Y) / c.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------ #
    # information
    # ------------------------------------------------------------------ #
    def fisher_information(self, theta: np.ndarray, times: Sequence[float], shots: Union[float, Sequence[float]] = 1.0,
                           readout: Optional[ReadoutErrorModel] = None, noise: Optional[DeviceNoise] = None,
                           wrt: Optional[Sequence[int]] = None, per_time: bool = False) -> np.ndarray:
        """
        Fisher information of the full bitstring distribution,
        F_ab = Σ_k S_k Σ_z ∂_a q_kz ∂_b q_kz / q_kz, in the units of θ.
        ``per_time=True`` returns the (K, m, m) contributions.
        """
        q, dq = self.probabilities_and_jacobian(theta, times, readout, noise, wrt)
        mask = q > 1e-14
        inv = np.where(mask, 1.0 / np.where(mask, q, 1.0), 0.0)
        Fk = np.einsum("kza,kz,kzb->kab", dq, inv, dq)
        S = np.broadcast_to(np.asarray(shots, dtype=float), (len(Fk),))
        Fk = Fk * S[:, None, None]
        return Fk if per_time else Fk.sum(axis=0)

    def _scales(self, theta: np.ndarray, wrt: Sequence[int]) -> np.ndarray:
        """Units for relative errors: |θ_a| for Ω, Δ, C6 (1 rad/μs if Δ = 0) and 1 rad/μs for local offsets."""
        th = np.abs(np.asarray(theta, dtype=float))
        return np.array([th[a] if (a < 3 and th[a] > 0) else 1.0 for a in wrt])

    def cramer_rao(self, theta: np.ndarray, times: Sequence[float], shots: Union[float, Sequence[float]],
                   readout: Optional[ReadoutErrorModel] = None, noise: Optional[DeviceNoise] = None,
                   wrt: Optional[Sequence[int]] = None) -> Dict[str, np.ndarray]:
        """
        Cramér–Rao bound for the parameters ``wrt`` (default all) given ``shots``
        per probe time. Relative errors use |θ_a| (or 1 rad/μs for a zero offset).
        """
        wrt = list(range(self.n_params)) if wrt is None else list(wrt)
        F = self.fisher_information(theta, times, shots, readout, noise, wrt)
        cov = _relative_inverse(F, self._scales(theta, wrt))
        std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        return {"names": tuple(self.param_names[a] for a in wrt), "covariance": cov, "std": std,
                "relative_std": std / self._scales(theta, wrt), "fisher": F}

    def identifiability(self, theta: np.ndarray, times: Sequence[float], readout: Optional[ReadoutErrorModel] = None,
                        noise: Optional[DeviceNoise] = None, wrt: Optional[Sequence[int]] = None) -> Dict[str, np.ndarray]:
        """
        Eigen-analysis of the per-shot Fisher information in relative units
        (one shot at each probe time, θ_a ∂/∂θ_a). Small eigenvalues mark parameter
        combinations the data barely constrain; ``directions[:, 0]`` is the worst.
        """
        wrt = list(range(self.n_params)) if wrt is None else list(wrt)
        D = self._scales(theta, wrt)
        F = self.fisher_information(theta, times, 1.0, readout, noise, wrt) * np.outer(D, D)
        ev, vec = np.linalg.eigh(F)
        return {"names": tuple(self.param_names[a] for a in wrt), "eigenvalues": ev, "directions": vec,
                "condition_number": float(ev[-1] / max(ev[0], 1e-300)),
                "relative_crb_per_shot": np.sqrt(np.clip(np.diag(_safe_inverse(F)), 0.0, None))}

    def shots_for_precision(self, theta: np.ndarray, times: Sequence[float], target: float = 0.01,
                            readout: Optional[ReadoutErrorModel] = None, noise: Optional[DeviceNoise] = None,
                            wrt: Optional[Sequence[int]] = None) -> Dict[str, object]:
        """
        Total shots, split equally over the probe times, for which the
        Cramér–Rao relative standard deviation of every parameter in ``wrt``
        reaches ``target``.
        """
        wrt = list(range(3)) if wrt is None else list(wrt)
        times = np.atleast_1d(np.asarray(times, dtype=float))
        D = self._scales(theta, wrt)
        Fbar = self.fisher_information(theta, times, 1.0 / len(times), readout, noise, wrt) * np.outer(D, D)
        var = np.diag(_safe_inverse(Fbar))
        per = var / target**2
        return {"names": tuple(self.param_names[a] for a in wrt), "total_shots": float(np.max(per)),
                "per_parameter": per, "shots_per_time": float(np.max(per)) / len(times)}

    def optimal_probe_times(self, theta: np.ndarray, n_times: int, t_grid: Sequence[float], target: float = 0.01,
                            readout: Optional[ReadoutErrorModel] = None, noise: Optional[DeviceNoise] = None,
                            wrt: Optional[Sequence[int]] = None, max_sweeps: int = 30,
                            per_time_fisher: Optional[np.ndarray] = None) -> Dict[str, object]:
        """
        Choose ``n_times`` probe times from ``t_grid`` (shots split equally) to
        minimise the worst relative Cramér–Rao variance over ``wrt``, by
        coordinate exchange started from an even spread. Returns the times, the
        total shots for ``target`` and the per-time Fisher information used.
        """
        wrt = list(range(3)) if wrt is None else list(wrt)
        grid = np.asarray(t_grid, dtype=float)
        D = self._scales(theta, wrt)
        Fk = per_time_fisher if per_time_fisher is not None else (
            self.fisher_information(theta, grid, 1.0, readout, noise, wrt, per_time=True) * np.outer(D, D)[None])

        def worst(design):
            return float(np.max(np.diag(_safe_inverse(Fk[list(design)].mean(axis=0)))))

        design = list(np.unique(np.linspace(0, len(grid) - 1, int(n_times) + 2).round().astype(int)[1:-1]))
        while len(design) < n_times:
            design.append(len(grid) - 1)
        best = worst(design)
        for _ in range(int(max_sweeps)):
            improved = False
            for j in range(len(design)):
                trial = list(design)
                scores = []
                for g in range(len(grid)):
                    trial[j] = g
                    scores.append(worst(trial))
                g_best = int(np.argmin(scores))
                if scores[g_best] < best * (1 - 1e-12):
                    design[j] = g_best; best = scores[g_best]; improved = True
            if not improved:
                break
        design = sorted(design)
        return {"times": grid[design], "indices": np.array(design), "worst_relative_variance_per_shot": best,
                "total_shots": best / target**2, "per_time_fisher": Fk}

    # ------------------------------------------------------------------ #
    # estimation
    # ------------------------------------------------------------------ #
    def _fit_setup(self, theta0, fit):
        th0 = np.asarray(theta0, dtype=float).copy()
        idx = [self.param_names.index(nm) for nm in fit]
        scale = self._scales(th0, idx)
        lower = np.array([1e-9 if self.param_names[a] in ("omega", "c6") else -np.inf for a in idx])
        return th0, idx, scale, lower

    def fit_mle(self, data: Optional[QuenchData], theta0: np.ndarray, fit: Sequence[str] = ("omega", "delta", "c6"),
                readout: Optional[ReadoutErrorModel] = None, noise: Optional[DeviceNoise] = None,
                times: Optional[Sequence[float]] = None, counts: Optional[np.ndarray] = None,
                continuation: bool = True, max_iter: int = 300, n_polish: int = 4, search: Optional[bool] = None,
                n_starts: int = 3, start_spread: float = 0.08, ambiguity: float = 3.0, max_move: float = 0.3,
                comb_range: float = 0.25) -> RydbergFit:
        """
        Maximum-likelihood fit of the parameters named in ``fit`` (the rest stay
        at ``theta0``) to the bitstring counts, with readout errors and drive
        noise inside the likelihood. Ω and C6 are constrained positive, which
        removes the two exact sign symmetries. ``counts`` may be expected
        (non-integer) counts, in which case the fit returns the parameters that
        minimise the KL divergence to that distribution (the large-shot limit).

        The likelihood of a late probe time oscillates in θ and has local maxima
        a few per cent from the global one, mostly along C6. Three steps deal
        with this:

        1. Continuation: probe times are added one at a time in increasing
           order, each stage starting from the previous estimate. The first
           stage fits Ω alone, from the best point of a grid over ±30 %, and
           the stages before the last keep every parameter within
           ±``max_move`` (relative) of the start.
        2. Search (default on for up to 8 atoms): ``n_starts`` more
           continuation runs from seeded starts within ±``start_spread`` of
           ``theta0``; then refinements from the best estimate with the sign of
           Δ flipped, and with C6 moved by every whole multiple within
           ±``comb_range`` of 2π/(V_max T_max), the step at which the phase of
           the most strongly interacting pair wraps, which is where the
           neighbouring maxima along C6 sit. That is 2·comb_range/step extra
           fits: with the default 0.25 and probe times up to 4 μs, about 22 at
           6.5 μm and 110 at 5 μm, so lower ``comb_range`` for close spacings
           if time matters. The best optimum is returned; other distinct
           optima within ``ambiguity`` log-likelihood units are returned in
           ``alternatives`` and make ``success`` false, because the data cannot
           tell them apart.
        3. Goodness of fit: a χ² test of the readout-corrected ⟨n_i⟩ and
           ⟨n_i n_j⟩ against the fitted model (:meth:`goodness_of_fit`).
           p < 10⁻³ makes ``success`` false: either the model is wrong or the
           fit is stuck.

        A final Fisher-scoring polish is applied, and the covariance is the
        inverse Fisher information at the estimate, computed in relative units.
        Continuation and the search were validated on the reference chain and a
        few other small arrays (see the README); they are heuristics, not a
        guarantee of the global maximum.
        """
        if data is not None:
            times, counts = data.times, data.counts()
        times = np.atleast_1d(np.asarray(times, dtype=float))
        counts = np.asarray(counts, dtype=float)
        th0, idx, scale, lower = self._fit_setup(theta0, fit)
        bounds = [(lo / s if np.isfinite(lo) else None, None) for lo, s in zip(lower, scale)]
        lb = np.array([b[0] if b[0] is not None else -np.inf for b in bounds])
        n_eval = [0]
        shots = counts.sum(axis=1)
        everything = np.ones(len(times), dtype=bool)

        def full(u):
            th = th0.copy(); th[idx] = u * scale
            return th

        def nll_grad(u, sel):
            n_eval[0] += 1
            q, dq = self.probabilities_and_jacobian(full(u), times[sel], readout, noise, wrt=idx)
            qc = np.clip(q, 1e-300, None)
            c = counts[sel]
            return float(-(c * np.log(qc)).sum()), -np.einsum("kz,kza->a", c / qc, dq) * scale

        def minimise(u, sel, strict=True, box=None):
            def objective(v):
                if not np.all(np.isfinite(v)):
                    return np.inf, np.zeros_like(v)
                return nll_grad(v, sel)
            tol = {"ftol": 1e-15, "gtol": 1e-10} if strict else {"ftol": 1e-11, "gtol": 1e-6}
            bnds = bounds
            if box is not None:                  # keep each parameter within ±max_move·|θ| of the run's start
                lo, hi = box
                bnds = [(max(l, b[0]) if b[0] is not None else l, h) for (l, h), b in zip(zip(lo, hi), bounds)]
            return scipy.optimize.minimize(objective, np.clip(u, [b[0] if b[0] is not None else -np.inf for b in bnds],
                                                              [b[1] if b[1] is not None else np.inf for b in bnds]),
                                           jac=True, method="L-BFGS-B", bounds=bnds, options={"maxiter": int(max_iter), **tol})

        def polish(u):
            f = nll_grad(u, everything)[0]
            for _ in range(int(n_polish)):       # Fisher scoring: quadratic convergence near the optimum
                Fu = self.fisher_information(full(u), times, shots, readout, noise, idx) * np.outer(scale, scale)
                _, g = nll_grad(u, everything)
                u_new = np.maximum(u - np.linalg.lstsq(Fu, g, rcond=None)[0], lb)
                f_new = nll_grad(u_new, everything)[0]
                if f_new <= f:
                    u, f = u_new, f_new
                else:
                    break
            return u, f

        stages = np.unique(times) if continuation else np.array([times.max()])
        k_omega = list(fit).index("omega") if "omega" in fit else None

        def omega_first(u, sel):
            # The earliest probe time mostly fixes Ω, and fitting everything to it alone can run far off. Fit Ω
            # alone, from the best point of a grid over ±30 % (one probe time can be a full Rabi cycle or more).
            k = k_omega

            def obj1(v):
                w = u.copy(); w[k] = v[0]
                if not np.all(np.isfinite(w)):
                    return np.inf, np.zeros(1)
                f1, g1 = nll_grad(w, sel)
                return f1, g1[k:k + 1]
            grid = u[k] * np.linspace(0.7, 1.3, 61)
            grid = grid[grid > lb[k]]
            v0 = grid[int(np.argmin([obj1(np.array([g]))[0] for g in grid]))]
            r1 = scipy.optimize.minimize(obj1, np.array([v0]), jac=True, method="L-BFGS-B", bounds=[bounds[k]],
                                         options={"maxiter": int(max_iter), "ftol": 1e-11, "gtol": 1e-6})
            w = u.copy(); w[k] = r1.x[0]
            return w

        def continue_from(u):
            res = None
            width = max_move * np.maximum(np.abs(u), 1e-12)
            box = (u - width, u + width) if max_move and max_move > 0 else None
            for j, horizon in enumerate(stages):
                sel = times <= horizon + 1e-12
                last = j == len(stages) - 1
                if j == 0 and len(stages) > 1 and k_omega is not None and len(idx) > 1:
                    u = omega_first(u, sel)
                    continue
                res = minimise(u, sel, strict=last, box=None if last else box)
                u = res.x
            return (u, float(res.fun), bool(res.success))

        optima = [continue_from(th0[idx] / scale)]
        do_search = (self.n <= 8) if search is None else bool(search)
        if do_search and n_starts > 0:
            # further continuation runs from starts spread around theta0 (seeded, so fits are reproducible)
            rs = np.random.default_rng(12345)
            for _ in range(int(n_starts)):
                u_s = th0[idx] / scale * (1.0 + start_spread * rs.uniform(-1.0, 1.0, len(idx)))
                optima.append(continue_from(np.maximum(u_s, lb + 1e-9)))
            optima.sort(key=lambda o: o[1])
        if do_search:
            th_c = full(optima[0][0])
            starts = []
            if "delta" in fit:                   # Δ enters early times only at second order, so its sign can come out wrong
                th_f = th_c.copy(); th_f[1] = -th_f[1]
                starts.append(th_f)
            if "c6" in fit and len(self.distances):
                # neighbouring maxima along C6 sit where the strongest pair's phase V_max·T_max has wrapped by 2π
                period = 2.0 * np.pi / (th_c[2] / self.distances.min() ** 6 * times.max())
                for k in range(1, int(np.floor(comb_range / period + 1e-9)) + 1):        # every step out to ±comb_range
                    for sgn in (1, -1):
                        if 1.0 + sgn * k * period > 0.0:
                            th_k = th_c.copy(); th_k[2] *= 1.0 + sgn * k * period
                            starts.append(th_k)
            for th_s in starts:
                u_s = np.maximum(th_s[idx] / scale, lb + 1e-9)
                r = minimise(u_s, everything, strict=False)
                optima.append((r.x, float(r.fun), bool(r.success)))
        optima.sort(key=lambda o: o[1])
        u, f, _ = optima[0]
        u, f = polish(u)
        # converged if the Newton decrement gᵀF⁻¹g at the polished optimum is negligible (in log-likelihood units)
        _, g = nll_grad(u, everything)
        Fu = self.fisher_information(full(u), times, shots, readout, noise, idx) * np.outer(scale, scale)
        free = (u > lb + 1e-9) | (g < 0)                 # a parameter held at its bound by a positive gradient is fine
        dec = float(g[free] @ np.linalg.lstsq(Fu[np.ix_(free, free)], g[free], rcond=None)[0]) if np.any(free) else 0.0
        conv_best = bool(np.isfinite(f) and dec < 1e-3)
        th = full(u)
        cov = _relative_inverse(self.fisher_information(th, times, shots, readout, noise, idx), scale)
        sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        alternatives = []
        for u_o, f_o, _ in optima[1:]:
            if f_o - f > ambiguity:
                continue
            th_o = full(u_o)
            width = np.where(np.isfinite(sd) & (sd > 0), sd, 1e-3 * np.abs(th[idx]) + 1e-12)
            if np.max(np.abs(th_o[idx] - th[idx]) / width) < 2.0:
                continue                         # the same optimum
            if any(np.max(np.abs(th_o[idx] - a_th[idx]) / width) < 2.0 for a_th, _ in alternatives):
                continue
            alternatives.append((th_o, float(f_o - f)))
        chi2, dof, pval = self.goodness_of_fit(th, times, counts, readout, noise, n_fitted=len(idx))
        warnings = []
        if not np.isfinite(f) or not conv_best:
            warnings.append("the optimiser did not report convergence")
        if np.isfinite(pval) and pval < 1e-3:
            warnings.append(f"goodness of fit p = {pval:.1e} (χ² = {chi2:.0f} for {dof} degrees of freedom): "
                            "the model does not describe these data, or the fit is stuck in a local maximum")
        for th_a, df in alternatives:
            diff = ", ".join(f"{self.param_names[a]} {100 * (th_a[a] / th[a] - 1):+.1f} %" for a in idx if th[a] != 0)
            warnings.append(f"another optimum ({diff} from this estimate) has a log-likelihood only {df:.2f} lower: "
                            "the data do not distinguish them, and the reported errors understate the uncertainty")
        return RydbergFit(theta=th, names=self.param_names, fitted=tuple(fit), covariance=cov, objective=f,
                          n_evaluations=n_eval[0], method="mle", success=not warnings, converged=bool(conv_best),
                          gof_chi2=chi2, gof_dof=dof, gof_pvalue=pval, alternatives=alternatives, warnings=warnings)

    def goodness_of_fit(self, theta: np.ndarray, times: Sequence[float], counts: np.ndarray,
                        readout: Optional[ReadoutErrorModel] = None, noise: Optional[DeviceNoise] = None,
                        n_fitted: int = 3, pairs: Optional[Sequence[Tuple[int, int]]] = None) -> Tuple[float, int, float]:
        """
        χ² test of the readout-corrected one- and two-site correlators against
        the model at ``theta``: Σ_k S_k (ȳ_k − μ_k)ᵀ C_k⁻¹ (ȳ_k − μ_k), with the
        per-shot covariance C_k under the model, against a χ² distribution with
        K(n + n_pairs) − ``n_fitted`` degrees of freedom. Returns (χ², dof, p).
        The χ² approximation needs a few hundred shots per probe time.
        """
        times = np.atleast_1d(np.asarray(times, dtype=float))
        counts = np.asarray(counts, dtype=float)
        S = counts.sum(axis=1)
        Yc = self.correlator_table(readout, pairs)
        ybar = (counts @ Yc) / S[:, None]
        q = self.probabilities(theta, times, readout, noise)
        chi2 = 0.0
        for k in range(len(times)):
            mu = Yc.T @ q[k]
            C = (Yc.T * q[k]) @ Yc - np.outer(mu, mu)
            C += np.eye(len(C)) * (1e-12 + 1e-9 * np.max(np.diag(C)))
            r = ybar[k] - mu
            chi2 += float(S[k] * r @ np.linalg.solve(C, r))
        dof = int(len(times) * Yc.shape[1] - n_fitted)
        return chi2, dof, float(scipy.stats.chi2.sf(chi2, dof)) if dof > 0 else float("nan")

    def fit_correlators(self, data: Optional[QuenchData], theta0: np.ndarray, fit: Sequence[str] = ("omega", "delta", "c6"),
                        readout: Optional[ReadoutErrorModel] = None, noise: Optional[DeviceNoise] = None,
                        pairs: Optional[Sequence[Tuple[int, int]]] = None, continuation: bool = True,
                        n_rounds: int = 2, times: Optional[Sequence[float]] = None,
                        counts: Optional[np.ndarray] = None) -> RydbergFit:
        """
        Generalised least squares on the readout-corrected one- and two-site
        correlators ⟨n_i(t_k)⟩ and ⟨n_i n_j(t_k)⟩ (:meth:`correlator_table`). The
        weights are the exact covariance of the corrected per-shot estimators
        under the model at the current estimate. Probe times are added in
        increasing order as in :meth:`fit_mle`, starting with Ω alone from the
        earliest time; there is no multi-start search. The last stage is repeated
        ``n_rounds`` times so the weights settle. The returned covariance is
        (Jᵀ C⁻¹ J)⁻¹. As in :meth:`fit_mle`, ``times`` and (possibly expected)
        ``counts`` can be passed instead of ``data``.
        """
        if data is not None:
            times, counts = data.times, data.counts()
        times = np.atleast_1d(np.asarray(times, dtype=float))
        counts = np.asarray(counts, dtype=float)
        S = counts.sum(axis=1)
        Yc = self.correlator_table(readout, pairs)
        Y0 = self.correlator_table(None, pairs)
        ybar = (counts @ Yc) / S[:, None]
        th0, idx, scale, lower = self._fit_setup(theta0, fit)
        lb = np.array([lo / s if np.isfinite(lo) else -np.inf for lo, s in zip(lower, scale)])
        theta = th0.copy()
        n_eval = [0]
        stages = list(np.unique(times)) if continuation else [times.max()]
        if continuation and len(stages) > 1 and "omega" in fit and len(idx) > 1:
            theta = self._omega_from_earliest_time(theta, times, counts, readout, noise)    # as in fit_mle
            stages = stages[1:]
        stages += [stages[-1]] * (max(int(n_rounds), 1) - 1)
        res = None
        for horizon in stages:
            ks = np.flatnonzero(times <= horizon + 1e-12)
            q = self.probabilities(theta, times[ks], readout, noise)
            Ls = []
            for j, k in enumerate(ks):
                mu = Yc.T @ q[j]
                C = ((Yc.T * q[j]) @ Yc - np.outer(mu, mu)) / S[k]
                C += np.eye(len(C)) * (1e-12 + 1e-9 * np.max(np.diag(C)))
                Ls.append(np.linalg.cholesky(C))
            cache: Dict[bytes, Tuple[np.ndarray, np.ndarray]] = {}

            def model_at(u, ks=ks, cache=cache):
                key = np.asarray(u).tobytes()
                if key not in cache:
                    n_eval[0] += 1
                    th = th0.copy(); th[idx] = u * scale
                    cache.clear()
                    cache[key] = self.probabilities_and_jacobian(th, times[ks], None, noise, wrt=idx)
                return cache[key]

            def resid(u, ks=ks, Ls=Ls):
                p, _ = model_at(u)
                return np.concatenate([scipy.linalg.solve_triangular(Ls[j], ybar[k] - Y0.T @ p[j], lower=True) for j, k in enumerate(ks)])

            def jac(u, ks=ks, Ls=Ls):
                _, dp = model_at(u)
                return np.concatenate([-scipy.linalg.solve_triangular(Ls[j], Y0.T @ dp[j], lower=True) for j in range(len(ks))]) * scale

            u0 = np.clip(theta[idx] / scale, lb + 1e-12, None)
            res = scipy.optimize.least_squares(resid, u0, jac=jac, bounds=(lb, np.full(len(idx), np.inf)),
                                               method="trf", x_scale="jac", xtol=1e-12, ftol=1e-12, gtol=1e-12)
            theta = th0.copy(); theta[idx] = res.x * scale
        Jw = jac(res.x) / scale
        cov = _relative_inverse(Jw.T @ Jw, scale)
        chi2 = float(2.0 * res.cost)
        dof = int(len(times) * Yc.shape[1] - len(idx))
        pval = float(scipy.stats.chi2.sf(chi2, dof)) if dof > 0 else float("nan")
        warnings = []
        if not res.success:
            warnings.append("the optimiser did not report convergence")
        if np.isfinite(pval) and pval < 1e-3:
            warnings.append(f"goodness of fit p = {pval:.1e} (χ² = {chi2:.0f} for {dof} degrees of freedom): "
                            "the model does not describe these data, or the fit is stuck in a local minimum")
        return RydbergFit(theta=theta, names=self.param_names, fitted=tuple(fit), covariance=cov,
                          objective=chi2, n_evaluations=n_eval[0], method="correlator-gls",
                          success=not warnings, converged=bool(res.success), gof_chi2=chi2, gof_dof=dof,
                          gof_pvalue=pval, warnings=warnings)

    def _omega_from_earliest_time(self, theta, times, counts, readout, noise) -> np.ndarray:
        """Ω alone from the earliest probe time's likelihood: the best point of a grid over ±30 %, then refined."""
        t1 = float(np.min(times))
        c = counts[np.isclose(times, t1)].sum(axis=0)

        def nll(om):
            th = theta.copy(); th[0] = om
            q = np.clip(self.probabilities(th, [t1], readout, noise)[0], 1e-300, None)
            return -float(c @ np.log(q))
        grid = theta[0] * np.linspace(0.7, 1.3, 61)
        j = int(np.argmin([nll(g) for g in grid]))
        step = grid[1] - grid[0]
        r = scipy.optimize.minimize_scalar(nll, bounds=(max(grid[j] - step, 1e-9), grid[j] + step), method="bounded",
                                           options={"xatol": 1e-9 * abs(theta[0])})
        th = theta.copy(); th[0] = float(r.x)
        return th

    def correlator_fisher_information(self, theta: np.ndarray, times: Sequence[float], shots: Union[float, Sequence[float]] = 1.0,
                                      readout: Optional[ReadoutErrorModel] = None, noise: Optional[DeviceNoise] = None,
                                      wrt: Optional[Sequence[int]] = None,
                                      pairs: Optional[Sequence[Tuple[int, int]]] = None) -> np.ndarray:
        """
        Information retained by the readout-corrected correlators alone,
        Σ_k S_k J_kᵀ C_k⁻¹ J_k, the inverse of the asymptotic covariance of
        :meth:`fit_correlators`. Never larger than :meth:`fisher_information`.
        """
        wrt = list(range(self.n_params)) if wrt is None else list(wrt)
        times = np.atleast_1d(np.asarray(times, dtype=float))
        Yc = self.correlator_table(readout, pairs)
        Y0 = self.correlator_table(None, pairs)
        q = self.probabilities(theta, times, readout, noise)
        _, dp = self.probabilities_and_jacobian(theta, times, None, noise, wrt)
        S = np.broadcast_to(np.asarray(shots, dtype=float), (len(times),))
        F = np.zeros((len(wrt), len(wrt)))
        for k in range(len(times)):
            mu = Yc.T @ q[k]
            C = (Yc.T * q[k]) @ Yc - np.outer(mu, mu)
            C += np.eye(len(C)) * (1e-12 + 1e-9 * np.max(np.diag(C)))
            J = Y0.T @ dp[k]
            F += S[k] * J.T @ np.linalg.solve(C, J)
        return F


def _relative_inverse(F: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    Inverse of a Fisher matrix in the units of θ, computed in relative units:
    (D F D)⁻¹ rescaled by D, with D = diag(|θ_a|). Inverting F directly fails
    because Ω ~ 10 and C6 ~ 5·10⁶ make its eigenvalue ratio 10⁻¹² or smaller.
    """
    D = np.asarray(D, dtype=float)
    DD = np.outer(D, D)
    return _safe_inverse(np.asarray(F, dtype=float) * DD) * DD


def _safe_inverse(F: np.ndarray) -> np.ndarray:
    """Inverse of a Fisher matrix; directions with no information get an infinite variance."""
    F = 0.5 * (F + F.T)
    ev, vec = np.linalg.eigh(F)
    tol = max(float(ev[-1]), 1e-300) * 1e-13
    inv_ev = np.where(ev > tol, 1.0 / np.where(ev > tol, ev, 1.0), np.inf)
    with np.errstate(invalid="ignore"):
        out = (vec * inv_ev) @ vec.T
    if np.any(ev <= tol):
        bad = np.abs(vec[:, ev <= tol]).sum(axis=1) > 1e-8
        out[bad, :] = np.inf
        out[:, bad] = np.inf
    return out


# --------------------------------------------------------------------------- #
# drift tracking
# --------------------------------------------------------------------------- #
class RydbergDriftTracker:
    """
    Extended Kalman filter over a subset of (Ω, Δ, C6, δ_i) that consumes one
    quench shot at a time, for example a slowly drifting laser detuning.

    The state follows a random walk with per-shot standard deviation
    ``process_noise_std``. Each shot is treated as one vector observation
    y = (b_i, b_i b_j) of the measured bits and their pair products, with the
    exact mean and covariance under the model at the current estimate, readout
    errors included. The update is the shared Joseph-form step of
    :mod:`shadow_learning.kalman_tracker`; like the Gibbs-state tracker it is a
    Gaussian approximation to a discrete measurement, and the reported standard
    deviation is checked against the actual error in the tests.
    """

    def __init__(self, model: RydbergQuenchModel, theta: np.ndarray, track: Sequence[str] = ("delta",),
                 readout: Optional[ReadoutErrorModel] = None, process_noise_std: Union[float, Sequence[float]] = 0.002,
                 prior_std: Union[float, Sequence[float]] = 0.5, min_measurement_var: float = 1e-6,
                 pairs: Optional[Sequence[Tuple[int, int]]] = None):
        model._check_dense()
        self.model = model
        self.theta = np.asarray(theta, dtype=float).copy()
        self.track = tuple(track)
        self.idx = [model.param_names.index(nm) for nm in self.track]
        m = len(self.idx)
        self.x = self.theta[self.idx].copy()
        self.P = np.diag(np.broadcast_to(np.asarray(prior_std, dtype=float) ** 2, (m,))).astype(float)
        self.Q = np.diag(np.broadcast_to(np.asarray(process_noise_std, dtype=float) ** 2, (m,))).astype(float)
        self.readout = readout
        self.min_r = float(min_measurement_var)
        self._Y = model.correlator_table(None, pairs)
        self._weights = 1 << (model.n - 1 - np.arange(model.n))
        self._x_hist: List[np.ndarray] = []
        self._std_hist: List[np.ndarray] = []

    def process_shot(self, bits: Sequence[int], t: float) -> Dict[str, np.ndarray]:
        """Predict, then update with the bitstring ``bits`` read out at probe time ``t``."""
        self.P = self.P + self.Q
        th = self.theta.copy(); th[self.idx] = self.x
        q, dq = self.model.probabilities_and_jacobian(th, [t], self.readout, wrt=self.idx)
        q, dq = q[0], dq[0]
        Y = self._Y
        mu = Y.T @ q
        H = Y.T @ dq
        R = (Y.T * q) @ Y - np.outer(mu, mu) + self.min_r * np.eye(len(mu))
        z = int(np.asarray(bits, dtype=np.int64) @ self._weights)
        self.x, self.P = joseph_update(self.x, self.P, Y[z] - mu, H, R)
        self._x_hist.append(self.x.copy())
        self._std_hist.append(np.sqrt(np.diag(self.P)))
        return self.estimate()

    def track_stream(self, times: Sequence[float], bits: np.ndarray) -> Dict[str, np.ndarray]:
        for t, b in zip(times, bits):
            self.process_shot(b, t)
        return self.history()

    def estimate(self) -> Dict[str, np.ndarray]:
        return {"names": self.track, "params": self.x.copy(), "std": np.sqrt(np.diag(self.P))}

    def history(self) -> Dict[str, np.ndarray]:
        m = len(self.idx)
        x = np.array(self._x_hist) if self._x_hist else np.zeros((0, m))
        s = np.array(self._std_hist) if self._std_hist else np.zeros((0, m))
        return {"names": self.track, "estimate": x, "std": s}


def simulate_drifting_stream(model: RydbergQuenchModel, theta: np.ndarray, trajectory: np.ndarray, probe_times: Sequence[float],
                             rng: np.random.Generator, readout: Optional[ReadoutErrorModel] = None,
                             param: str = "delta") -> Tuple[np.ndarray, np.ndarray]:
    """
    One shot per entry of ``trajectory`` (the value of ``param`` at that shot),
    cycling through ``probe_times``. Returns (times, bits) with bits (n_shots, n).
    """
    a = model.param_names.index(param)
    probe_times = np.atleast_1d(np.asarray(probe_times, dtype=float))
    traj = np.asarray(trajectory, dtype=float)
    times = probe_times[np.arange(len(traj)) % len(probe_times)]
    idx = np.empty(len(traj), dtype=np.int64)
    th = np.asarray(theta, dtype=float).copy()
    for s, (val, t) in enumerate(zip(traj, times)):
        th[a] = val
        amp = model.amplitudes(th, [t])[0]
        pk = amp.real**2 + amp.imag**2
        idx[s] = rng.choice(model.dim, p=pk / pk.sum())
    return times, _flip_bits(model.bits[idx], readout, rng)


__all__ = [
    "C6_RB87", "AQUILA", "AQUILA_BRAKET", "AquilaSpecs", "DEFAULT_RAMP", "blockade_radius", "snap", "chain", "ring", "square_lattice",
    "geometry_violations",
    "DeviceNoise", "QuenchData", "RydbergFit", "RydbergQuenchModel", "RydbergDriftTracker",
    "simulate_drifting_stream", "apply_readout",
]
