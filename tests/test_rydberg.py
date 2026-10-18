"""
Physics and estimation tests for the Rydberg quench module.
"""

import numpy as np
import pytest
import scipy.linalg

from shadow_learning.reconstruction import estimate_pauli_string_expectation
from shadow_learning.rydberg import (
    AQUILA,
    AQUILA_BRAKET,
    C6_RB87,
    DeviceNoise,
    QuenchData,
    RydbergDriftTracker,
    RydbergQuenchModel,
    blockade_radius,
    chain,
    geometry_violations,
    ring,
    simulate_drifting_stream,
    snap,
    square_lattice,
)

RD = AQUILA.readout_model()
TIMES = np.linspace(0.5, 4.0, 8)


def _column_errors(model, theta, times, dq, readout=None, noise=None, rel=1e-5):
    """
    Compare each column of θ_a ∂q/∂θ_a with a central difference of step rel·|θ_a| (1 when θ_a = 0), and
    return the largest error in each column relative to that column's largest entry. The raw columns differ
    in size by about 10⁶ (C6 is about 5 × 10⁶ rad μm⁶/μs), so one absolute tolerance would not test the C6
    column. The difference itself is off by O(rel²), at most about 10⁻⁶ here.
    """
    out = []
    for a in range(model.n_params):
        s = abs(theta[a]) if theta[a] != 0 else 1.0
        e = np.zeros(model.n_params); e[a] = rel * s
        fd = (model.probabilities(theta + e, times, readout, noise) - model.probabilities(theta - e, times, readout, noise)) / (2 * rel)
        out.append(np.abs(s * dq[..., a] - fd).max() / np.abs(fd).max())
    return np.array(out)


# --------------------------------------------------------------------------- #
# the Hamiltonian and its dynamics
# --------------------------------------------------------------------------- #
def test_hamiltonian_is_hermitian_with_the_blockade_energy():
    r, omega, delta = 6.0, 10.0, 3.0
    m = RydbergQuenchModel(chain(2, r))
    H = m.hamiltonian(m.theta(omega, delta))
    assert np.allclose(H, H.conj().T)
    # basis |gg>, |gr>, |rg>, |rr>: −Δ per excitation, plus C6/r^6 for the doubly excited state
    assert np.allclose(np.diag(H), [0.0, -delta, -delta, -2 * delta + C6_RB87 / r**6])
    assert np.isclose(H[0, 1], omega / 2) and np.isclose(H[0, 2], omega / 2) and np.isclose(H[1, 3], omega / 2)
    assert H[0, 3] == 0 and H[1, 2] == 0                                   # no two-atom flips
    m5 = RydbergQuenchModel(np.random.default_rng(0).uniform(0, 25, (5, 2)))
    th = m5.theta(7.0, -2.0)
    H5 = m5.hamiltonian(th)
    assert np.allclose(H5, H5.T) and np.allclose(m5.hamiltonian(th, sparse=True).toarray(), H5)
    assert np.isclose(m5.interaction_matrix()[0, 1], C6_RB87 / np.linalg.norm(m5.positions[0] - m5.positions[1]) ** 6)


def test_single_atom_rabi_and_two_atoms_inside_the_blockade_oscillate_at_sqrt2_omega():
    omega = 2 * np.pi
    t = np.linspace(0.0, 2.0, 41)
    one = RydbergQuenchModel(np.zeros((1, 2)), ramp_time=0.0)                  # instantaneous quench
    assert np.allclose(one.probabilities(one.theta(omega, 0.0), t)[:, 1], np.sin(omega * t / 2) ** 2, atol=1e-12)
    # 4 μm: V = 1323 rad/μs ≫ Ω, so |rr> is blockaded and |gg> couples to the W state with √2·Ω/2
    pair = RydbergQuenchModel(chain(2, 4.0), ramp_time=0.0)
    P = pair.probabilities(pair.theta(omega, 0.0), t)
    assert np.abs(P[:, 1] + P[:, 2] - np.sin(np.sqrt(2) * omega * t / 2) ** 2).max() < 1e-3
    assert P[:, 3].max() < 1e-3
    # 30 μm: V ≈ 0.007 rad/μs, the atoms flip independently
    far = RydbergQuenchModel(chain(2, 30.0), ramp_time=0.0)
    Pf = far.probabilities(far.theta(omega, 0.0), t)
    assert np.abs(Pf[:, 3] - np.sin(omega * t / 2) ** 4).max() < 1e-4


def test_ramped_pulse_rotates_a_resonant_atom_by_its_pulse_area():
    """On resonance a single atom's slices commute, so the rotation angle is ∫Ω dt = Ω (T − r)."""
    omega, r = 9.0, 0.05
    one = RydbergQuenchModel(np.zeros((1, 2)), ramp_time=r)
    T = np.array([0.1, 0.13, 0.5, 1.7, 4.0])
    assert np.allclose(one.probabilities(one.theta(omega, 0.0), T)[:, 1], np.sin(omega * (T - r) / 2) ** 2, atol=1e-12)
    m = RydbergQuenchModel(ring(5, 7.0))
    th = m.theta(12.0, 5.0)
    exact = m.with_ramp(0.05, 256).probabilities(th, TIMES)                     # midpoint rule converges as slices⁻²
    assert np.abs(m.probabilities(th, TIMES) - exact).max() < 3e-4
    assert np.abs(m.with_ramp(0.0).probabilities(th, TIMES) - exact).max() > 1e-2   # the ramps are not negligible
    with pytest.raises(ValueError):
        m.probabilities(th, [0.08])                                             # shorter than two ramps


def test_zero_ramp_matches_the_matrix_exponential():
    m = RydbergQuenchModel(ring(5, 7.0), local_detuning=True, ramp_time=0.0)
    th = m.theta(11.0, 3.0, delta_local=[0.2, -0.1, 0.0, 0.3, -0.2])
    H = m.hamiltonian(th)
    t = [0.3, 2.2]
    P = m.probabilities(th, t)
    for k, tk in enumerate(t):
        assert np.abs(P[k] - np.abs(scipy.linalg.expm(-1j * H * tk)[:, 0]) ** 2).max() < 1e-12


def test_two_atom_dynamics_match_the_symmetric_three_level_solution():
    """At any distance, |gg>, |W>, |rr> close under H with couplings Ω/√2."""
    omega, delta, r = 9.0, 4.0, 7.0
    V = C6_RB87 / r**6
    h3 = np.array([[0, omega / np.sqrt(2), 0], [omega / np.sqrt(2), -delta, omega / np.sqrt(2)], [0, omega / np.sqrt(2), -2 * delta + V]])
    pair = RydbergQuenchModel(chain(2, r), ramp_time=0.0)
    t = np.array([0.3, 1.1, 2.5, 3.9])
    P = pair.probabilities(pair.theta(omega, delta), t)
    for k, tk in enumerate(t):
        psi = scipy.linalg.expm(-1j * h3 * tk)[:, 0]
        assert np.allclose([P[k, 0], P[k, 1] + P[k, 2], P[k, 3]], np.abs(psi) ** 2, atol=1e-12)


def test_sparse_and_dense_evolution_agree():
    m = RydbergQuenchModel(square_lattice(4, 2, 7.0))
    th = m.theta(11.0, 4.0)
    times = [2.0, 0.5, 1.0]
    dense = m.amplitudes(th, times)
    m.DENSE_MAX_ATOMS = 0                                                     # force expm_multiply
    assert np.abs(m.amplitudes(th, times) - dense).max() < 1e-10
    big = RydbergQuenchModel(chain(11, 8.0))                                  # beyond the dense limit
    data = big.sample(big.theta(11.0, 4.0), [0.5, 1.0], 20, np.random.default_rng(0))
    assert data.bits[0].shape == (20, 11)
    with pytest.raises(ValueError):
        big.fisher_information(big.theta(11.0, 4.0), [1.0])


# --------------------------------------------------------------------------- #
# exact symmetries of Z-basis data
# --------------------------------------------------------------------------- #
def test_omega_sign_and_the_joint_sign_of_delta_and_c6_are_invisible():
    m = RydbergQuenchModel(np.random.default_rng(1).uniform(0, 20, (5, 2)))
    th = m.theta(11.0, 4.0)
    times = np.linspace(0.1, 4.0, 9)
    P = m.probabilities(th, times, RD)
    assert np.abs(m.probabilities(th * [-1, 1, 1], times, RD) - P).max() < 1e-12      # Π Z_i flips Ω only
    assert np.abs(m.probabilities(th * [1, -1, -1], times, RD) - P).max() < 1e-12     # H → −H, then Ω → −Ω
    assert np.abs(m.probabilities(th * [1, -1, 1], times, RD) - P).max() > 1e-2       # the sign of Δ alone is visible
    loc = RydbergQuenchModel(chain(3, 7.0), local_detuning=True)
    tl = loc.theta(11.0, 4.0, delta_local=[0.3, -0.2, 0.1])
    flip = tl.copy(); flip[1:] *= -1                                                  # H → −H also flips every δ_i
    assert np.abs(loc.probabilities(flip, times) - loc.probabilities(tl, times)).max() < 1e-12


def test_rates_and_durations_enter_only_as_products():
    """(Ω, Δ, C6) → a·(Ω, Δ, C6) with every time, ramps included, divided by a."""
    a = 1.3
    m = RydbergQuenchModel(ring(5, 7.5), ramp_time=0.065)
    th = m.theta(12.0, 5.0)
    P = m.probabilities(th, TIMES)
    fast = RydbergQuenchModel(ring(5, 7.5), ramp_time=0.065 / a)
    assert np.abs(fast.probabilities(a * th, TIMES / a) - P).max() < 1e-12


def test_c6_and_the_length_scale_of_the_array_are_one_parameter():
    pos, s = ring(5, 7.5), 1.07
    th = RydbergQuenchModel(pos).theta(12.0, 5.0)
    a = RydbergQuenchModel(pos).probabilities(th, TIMES)
    b = RydbergQuenchModel(s * pos).probabilities(th * [1, 1, s**6], TIMES)
    assert np.abs(a - b).max() < 1e-12


def test_global_detuning_and_mean_local_offset_are_one_parameter():
    m = RydbergQuenchModel(chain(4, 8.0), local_detuning=True)
    th = m.theta(12.0, 5.0, delta_local=[0.3, -0.1, 0.2, -0.4])
    F = m.fisher_information(th, TIMES, 1.0, RD)
    null = np.zeros(m.n_params); null[1] = 1.0; null[3:] = -1.0
    assert np.linalg.norm(F @ null) < 1e-8 * np.linalg.norm(F)


# --------------------------------------------------------------------------- #
# gradients and information
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("noise", [None, DeviceNoise(sigma_delta=0.18, sigma_omega_rel=0.008, t_relax=6.0)])
def test_exact_jacobian_matches_finite_differences_on_a_degenerate_spectrum(noise):
    m = RydbergQuenchModel(ring(6, 7.0, resolution=None))                     # the exact hexagon, not snapped to the grid
    th = m.theta(12.0, 5.0)
    assert np.min(np.diff(np.linalg.eigvalsh(m.hamiltonian(th)))) < 1e-10        # the ring's spectrum is degenerate
    times = [0.3, 1.7, 3.2]
    q, dq = m.probabilities_and_jacobian(th, times, RD, noise)
    assert np.allclose(q, m.probabilities(th, times, RD, noise), atol=1e-14)
    assert _column_errors(m, th, times, dq, RD, noise).max() < 1e-5
    assert _column_errors(m, th, times, dq * [1.0, 1.0, 1.001], RD, noise)[2] > 1e-4     # a C6 column 0.1 % off fails


@pytest.mark.parametrize("positions", [chain(6, 8.0), chain(2, 6.5)])
def test_exact_jacobian_matches_finite_differences_on_a_chain_and_a_pair(positions):
    m = RydbergQuenchModel(positions)
    th = m.theta(12.0, 5.0)
    times = [0.1, 1.0, 2.5, 4.0]
    _, dq = m.probabilities_and_jacobian(th, times, RD)
    assert _column_errors(m, th, times, dq, RD).max() < 1e-5


def test_exact_jacobian_through_the_ramps_with_local_detuning():
    m = RydbergQuenchModel(ring(5, 7.0), local_detuning=True)
    th = m.theta(12.0, 5.0, delta_local=[0.3, -0.2, 0.0, 0.1, -0.4])
    times = [0.1, 0.13, 1.7, 3.2]                                                     # 0.1 μs: a triangle with no flat part
    _, dq = m.probabilities_and_jacobian(th, times, RD)
    assert _column_errors(m, th, times, dq, RD).max() < 1e-5


def test_device_noise_reduces_to_the_closed_model_and_relaxation_scales_single_atom_contrast():
    m = RydbergQuenchModel(np.zeros((1, 2)))
    th = m.theta(9.0, 2.0)
    t = np.array([0.4, 1.3, 2.9])
    assert np.allclose(m.probabilities(th, t, noise=DeviceNoise()), m.probabilities(th, t))
    z0 = m.probabilities(th, t) @ np.array([1.0, -1.0])
    z1 = m.probabilities(th, t, noise=DeviceNoise(t_relax=2.0)) @ np.array([1.0, -1.0])
    assert np.allclose(z1, z0 * np.exp(-t / 2.0))
    nodes = DeviceNoise.aquila().nodes()
    assert np.isclose(sum(w for _, _, w in nodes), 1.0)


def test_readout_errors_and_correlators_never_add_information():
    m = RydbergQuenchModel(chain(5, 8.0))
    th = m.theta(12.0, 5.0)
    F0 = m.fisher_information(th, TIMES, 1.0, None)
    F1 = m.fisher_information(th, TIMES, 1.0, RD)
    Fc = m.correlator_fisher_information(th, TIMES, 1.0, RD)
    D = np.diag(1 / th)
    assert np.linalg.eigvalsh(D @ (F0 - F1) @ D).min() > -1e-9
    assert np.linalg.eigvalsh(D @ (F1 - Fc) @ D).min() > -1e-9


@pytest.mark.parametrize("positions", [chain(2, 6.0), ring(6, 7.0), chain(6, 9.5)])
def test_cramer_rao_is_finite_and_inverts_the_fisher_information(positions):
    """C6 is ~10⁶ times Ω, so the inverse has to be taken in relative units."""
    m = RydbergQuenchModel(positions)
    th = m.theta(12.0, 5.0)
    crb = m.cramer_rao(th, TIMES, 200, RD)
    assert np.all(np.isfinite(crb["std"])) and np.all(crb["std"] > 0)
    F = m.fisher_information(th, TIMES, 200, RD)
    D = np.diag(th)
    assert np.allclose(D @ F @ D @ (np.diag(1 / th) @ crb["covariance"] @ np.diag(1 / th)), np.eye(3), atol=1e-6)


def test_c6_is_hidden_deep_in_the_blockade():
    """For a pair, C6 only enters through the suppressed |rr> amplitude when V ≫ Ω."""
    def rel_crb(r):
        pair = RydbergQuenchModel(chain(2, r))
        th = pair.theta(12.0, 5.0)
        F = pair.fisher_information(th, TIMES, 1.0, RD, wrt=[2])
        return 1.0 / (th[2] * np.sqrt(F[0, 0]))
    assert rel_crb(4.0) > 100 * rel_crb(7.5)
    assert blockade_radius(12.0, 5.0) == pytest.approx((C6_RB87 / 13.0) ** (1 / 6))


def test_shots_for_precision_agrees_with_the_cramer_rao_bound():
    m = RydbergQuenchModel(chain(5, 8.0))
    th = m.theta(12.0, 5.0)
    need = m.shots_for_precision(th, TIMES, 0.01, RD)
    crb = m.cramer_rao(th, TIMES, need["shots_per_time"], RD)
    assert np.isclose(crb["relative_std"].max(), 0.01, rtol=1e-6)
    best = m.optimal_probe_times(th, 1, np.linspace(0.1, 4.0, 40), 0.01, RD)
    assert best["total_shots"] < need["total_shots"]                          # late times carry the information
    assert best["times"][0] > 2.0


# --------------------------------------------------------------------------- #
# readout correction
# --------------------------------------------------------------------------- #
def test_per_shot_readout_correction_is_exactly_unbiased():
    m = RydbergQuenchModel(chain(4, 7.0))
    th = m.theta(13.0, 3.0)
    q = m.probabilities(th, TIMES, RD)          # what the device reports
    p = m.probabilities(th, TIMES)              # what the atoms did
    corrected = q @ m.correlator_table(RD)
    truth = p @ m.correlator_table(None)
    assert np.abs(corrected - truth).max() < 1e-12
    assert np.abs(q @ m.correlator_table(None) - truth).max() > 0.01           # raw counts are biased


@pytest.mark.slow
def test_correction_agrees_with_the_generic_shadow_estimator_and_is_statistically_unbiased():
    m = RydbergQuenchModel(chain(4, 7.0))
    th = m.theta(13.0, 3.0)
    data = m.sample(th, [1.5], 20000, np.random.default_rng(5), readout=RD)
    est = m.correlators(data, RD)[0]
    snaps = data.snapshots(0)
    z0 = estimate_pauli_string_expectation(snaps, "ZIII", derandomized=True, readout_model=RD)
    z1 = estimate_pauli_string_expectation(snaps, "IZII", derandomized=True, readout_model=RD)
    z01 = estimate_pauli_string_expectation(snaps, "ZZII", derandomized=True, readout_model=RD)
    assert np.isclose(est[0], (1 - z0) / 2) and np.isclose(est[4], (1 - z0 - z1 + z01) / 4)    # n_0 and n_0 n_1
    truth = m.probabilities(th, [1.5])[0] @ m.correlator_table(None)
    Y = m.correlator_table(RD)
    q = m.probabilities(th, [1.5], RD)[0]
    sd = np.sqrt((q @ Y**2 - (q @ Y) ** 2) / 20000)
    assert np.all(np.abs(est - truth) < 4.5 * sd)


# --------------------------------------------------------------------------- #
# estimation
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_mle_recovers_the_truth_within_the_cramer_rao_error():
    m = RydbergQuenchModel(chain(6, 8.0))
    th = m.theta(12.0, 5.0)
    data = m.sample(th, TIMES, 300, np.random.default_rng(11), readout=RD)
    fit = m.fit_mle(data, th * [1.08, 0.9, 1.1], readout=RD)
    crb = m.cramer_rao(th, TIMES, 300, RD)
    assert np.all(np.abs(fit.theta - th) < 3.5 * crb["std"])
    assert np.allclose([fit.std[n] for n in fit.fitted], crb["std"], rtol=0.1)   # reported error ≈ CRB


def test_correlator_fit_recovers_the_truth_with_its_own_error_bar():
    m = RydbergQuenchModel(chain(5, 8.0))
    th = m.theta(12.0, 5.0)
    data = m.sample(th, TIMES, 400, np.random.default_rng(2), readout=RD)
    fit = m.fit_correlators(data, th * [0.95, 1.08, 0.92], readout=RD)
    sd = np.array([fit.std[n] for n in fit.fitted])
    assert np.all(np.abs(fit.theta - th) < 3.5 * sd)
    full = m.cramer_rao(th, TIMES, 400, RD)["std"]
    assert np.all(sd >= full * 0.99)                                           # correlators keep less information


@pytest.mark.slow
def test_ignoring_readout_errors_biases_the_fit_and_modelling_them_does_not():
    m = RydbergQuenchModel(chain(5, 8.0))
    th = m.theta(12.0, 5.0)
    expected = m.probabilities(th, TIMES, RD) * 1000.0                         # large-shot limit
    biased = m.fit_mle(None, th, readout=None, times=TIMES, counts=expected)
    exact = m.fit_mle(None, th * 1.03, readout=RD, times=TIMES, counts=expected)
    assert np.abs(exact.theta / th - 1).max() < 1e-6
    assert np.abs(biased.theta / th - 1).max() > 1e-3
    corrected = m.fit_correlators(None, th * 1.03, readout=RD, times=TIMES, counts=expected)
    assert np.abs(corrected.theta / th - 1).max() < 1e-6


@pytest.mark.slow
def test_local_detuning_offsets_are_recovered():
    m = RydbergQuenchModel(chain(4, 8.0), local_detuning=True)
    offsets = np.array([0.4, -0.3, 0.1, -0.2])
    th = m.theta(12.0, 5.0, delta_local=offsets)
    data = m.sample(th, TIMES, 400, np.random.default_rng(4), readout=RD)
    names = ["omega", "c6"] + [f"delta_{i}" for i in range(4)]                  # Δ fixed: it duplicates the mean offset
    start = th.copy(); start[3:] = 0.0
    fit = m.fit_mle(data, start, fit=names, readout=RD)
    sd = np.array([fit.std[f"delta_{i}"] for i in range(4)])
    assert np.all(np.abs(fit.theta[3:] - offsets) < 3.5 * sd) and sd.max() < 0.1


@pytest.mark.parametrize("positions,shots", [(chain(2, 6.0), 400), pytest.param(ring(6, 7.0), 300, marks=pytest.mark.slow)])
def test_fit_errors_are_finite_for_a_pair_and_a_ring(positions, shots):
    m = RydbergQuenchModel(positions)
    th = m.theta(12.0, 5.0)
    data = m.sample(th, TIMES, shots, np.random.default_rng(13), readout=RD)
    fit = m.fit_mle(data, th * [1.03, 0.97, 1.02], readout=RD)
    sd = np.array([fit.std[n] for n in fit.fitted])
    assert np.all(np.isfinite(sd)) and np.all(sd > 0)
    assert abs(fit.theta[0] - th[0]) < 4 * sd[0]
    if len(positions) > 2:                     # a pair deep in the blockade barely sees the sign of Δ, so skip it there
        assert np.all(np.abs(fit.theta - th) < 4 * sd)


def test_goodness_of_fit_flags_a_wrong_model_but_not_the_right_one():
    m = RydbergQuenchModel(chain(4, 8.0))
    th = m.theta(12.0, 5.0)
    data = m.sample(th, TIMES, 400, np.random.default_rng(3), readout=RD)
    _, _, p_true = m.goodness_of_fit(th, TIMES, data.counts(), RD, n_fitted=0)
    _, _, p_wrong = m.goodness_of_fit(th * [1, 1, 1.15], TIMES, data.counts(), RD, n_fitted=0)
    assert p_true > 1e-3 and p_wrong < 1e-6
    stuck = m.fit_mle(data, th * [1, 1.5, 1], fit=("omega", "c6"), readout=RD)       # Δ held 50 % off
    assert not stuck.success and any("goodness of fit" in w for w in stuck.warnings)
    good = m.fit_mle(data, th * [1.04, 0.96, 1.03], readout=RD)
    assert good.success and good.warnings == [] and good.gof_pvalue > 1e-3


def test_close_spacing_fits_report_the_neighbouring_optima_along_c6():
    """At 6.5 μm the C6 phase wraps every 2.2 %, and 1,600 shots cannot tell the neighbouring maxima apart."""
    m = RydbergQuenchModel(chain(3, 6.5))
    th = m.theta(12.0, 5.0)
    data = m.sample(th, TIMES, 200, np.random.default_rng(0), readout=RD)
    fit = m.fit_mle(data, th * [1.02, 0.97, 1.03], readout=RD)
    assert not fit.success and fit.alternatives
    assert any("do not distinguish" in w for w in fit.warnings)


@pytest.mark.slow
def test_position_jitter_sampler_matches_the_averaged_distribution():
    m = RydbergQuenchModel(chain(3, 7.0))
    th = m.theta(12.0, 4.0)
    rng = np.random.default_rng(8)
    data = m.sample(th, [2.5], 6000, rng, position_jitter=0.3)
    avg = m.averaged_probabilities(th, [2.5], 0.3, 3000, rng)[0]
    emp = data.counts()[0] / 6000
    assert np.abs(emp - avg).max() < 4.5 * np.sqrt(avg.max() / 6000) + 0.01
    assert np.abs(avg - m.probabilities(th, [2.5])[0]).max() > 0.01             # jitter changes the statistics


@pytest.mark.slow
def test_mle_attains_the_cramer_rao_bound():
    m = RydbergQuenchModel(chain(5, 8.0))
    th = m.theta(12.0, 5.0)
    shots = 200
    crb = m.cramer_rao(th, TIMES, shots, RD)["std"]
    rng = np.random.default_rng(21)
    est = []
    for r in range(120):
        data = m.sample(th, TIMES, shots, np.random.default_rng(500 + r), readout=RD)
        # continuation alone, which converges on this chain (see the README), to keep the test quick
        est.append(m.fit_mle(data, th * (1 + 0.08 * rng.uniform(-1, 1, 3)), readout=RD, search=False).theta)
    est = np.array(est)
    ratio = np.sqrt(np.mean((est - th) ** 2, axis=0)) / crb
    assert np.all(ratio > 0.8) and np.all(ratio < 1.25)                          # 120 trials: σ ≈ 6.5 % on the RMSE


# --------------------------------------------------------------------------- #
# drift tracking
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_kalman_tracker_follows_detuning_drift_with_calibrated_uncertainty():
    """
    40 independent runs of 400 shots; Δ starts at a draw from the filter's prior
    and then random-walks with the filter's process noise. At the end of each run
    (error / reported σ) should be a standard normal.
    """
    m = RydbergQuenchModel(chain(4, 8.0))
    th = m.theta(12.0, 5.0)
    z, err = [], []
    for r in range(40):
        rng = np.random.default_rng(700 + r)
        traj = 5.0 + 0.05 * rng.standard_normal() + np.concatenate([[0.0], np.cumsum(2e-3 * rng.standard_normal(399))])
        times, bits = simulate_drifting_stream(m, th, traj, [2.6, 3.1], rng, readout=RD)
        tr = RydbergDriftTracker(m, th, readout=RD, process_noise_std=2e-3, prior_std=0.05)
        h = tr.track_stream(times, bits)
        assert np.all(np.isfinite(h["estimate"])) and np.all(h["std"] > 0)
        z.append((h["estimate"][-1, 0] - traj[-1]) / h["std"][-1, 0])
        err.append(h["estimate"][-1, 0] - traj[-1])
    z = np.array(z)
    assert 0.55 < np.mean(z**2) < 1.6
    assert np.mean(np.abs(z) < 2) > 0.85
    assert np.sqrt(np.mean(np.square(err))) < 0.05                               # better than the prior


# --------------------------------------------------------------------------- #
# geometry and CLI
# --------------------------------------------------------------------------- #
def test_geometry_helpers_and_aquila_layout_rules():
    c = chain(4, 6.0)
    assert np.allclose(np.diff(c[:, 0]), 6.0) and np.allclose(c[:, 1], 0)
    r = ring(6, 7.0)
    d = np.linalg.norm(r - np.roll(r, 1, axis=0), axis=1)
    assert np.allclose(d, 7.0, atol=0.01)                                         # snapped to the 0.01 μm grid
    assert np.allclose(np.linalg.norm(np.diff(ring(6, 7.0, resolution=None), axis=0), axis=1), 7.0)
    sq = square_lattice(3, 2, 5.0)
    assert sq.shape == (6, 2) and np.isclose(np.linalg.norm(sq[0] - sq[1]), 5.0)
    assert geometry_violations(c) == [] and geometry_violations(sq) == []
    assert any("apart" in v for v in geometry_violations(chain(3, 3.0)))         # below the 4 μm minimum
    assert geometry_violations(ring(6, 7.0)) == []                                # a snapped hexagon sits on three rows
    assert any("multiples" in v for v in geometry_violations(ring(6, 7.0, resolution=None)))   # y = ±6.062 is off the grid
    assert np.abs(snap([[0.0, 6.0622]]) - [[0.0, 6.06]]).max() < 1e-12
    assert any("rows" in v for v in geometry_violations(ring(8, 7.0)))           # an octagon does not
    assert any("width" in v for v in geometry_violations(chain(12, 7.0)))        # 77 μm > 75 μm


def test_waveforms_follow_the_braket_rules():
    m = RydbergQuenchModel(chain(3, 7.0))
    th = m.theta(12.0, 5.0)
    w = m.waveform(th, 2.0)
    assert np.allclose(w["amplitude"]["times"], [0.0, 0.05, 1.95, 2.0]) and np.allclose(w["amplitude"]["values"], [0, 12, 12, 0])
    assert np.allclose(w["detuning"]["values"], [5.0, 5.0]) and np.allclose(w["phase"]["values"], 0.0)
    assert np.allclose(m.waveform(th, 0.1)["amplitude"]["times"], [0.0, 0.05, 0.1])        # a triangle
    assert m.waveform_violations(th, TIMES) == []
    assert any("start and end at 0" in v for v in m.with_ramp(0.0).waveform_violations(th, TIMES))
    assert any("slope" in v for v in m.waveform_violations(m.theta(15.0, 5.0), TIMES))      # 300 rad/μs² > 250
    assert m.with_ramp(0.06).waveform_violations(m.theta(15.0, 5.0), TIMES) == []
    assert any("flat part" in v for v in m.waveform_violations(th, [0.13]))
    assert any("exceeds 4" in v for v in m.waveform_violations(th, [4.5]))
    assert any("1 ns" in v for v in m.waveform_violations(th, [1.0005]))
    assert AQUILA.minimum_ramp(15.8) == pytest.approx(0.064)                                 # 15.8 / 250, on the 1 ns grid
    # the current Braket figures allow 400 rad/μs², and have larger readout errors and shorter coherence times
    assert m.waveform_violations(m.theta(15.0, 5.0), TIMES, AQUILA_BRAKET) == []
    assert AQUILA_BRAKET.minimum_ramp(15.8) == pytest.approx(0.05)
    rb = AQUILA_BRAKET.readout_model()
    assert (rb.p01, rb.p10) == (0.05, 0.10) and DeviceNoise.braket().t_relax == 7.0 and AQUILA.t2_echo == 11.4


def test_from_braket_converts_bits_and_drops_shots_with_an_empty_site():
    pre = [np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])]
    post = [np.array([[1, 0, 1], [1, 0, 1], [0, 0, 1]])]                     # 1 = atom found in the ground state
    d = QuenchData.from_braket([1.0], pre, post)
    assert d.shots.tolist() == [2] and d.bits[0].tolist() == [[0, 1, 0], [1, 1, 0]]
    raw = QuenchData.from_braket([1.0], pre, post, postselect=False)
    assert raw.shots.tolist() == [3] and np.isclose(raw.filled_fraction()[0], 2 / 3)


def test_from_braket_handles_sites_requested_empty_and_failed_shots():
    # site 1 is requested empty; shot 2 failed and has no images; shot 3 has a stray atom in site 1
    pre = [[[1, 0, 1], [1, 0, 1], None, [1, 1, 1], [1, 0, 1]]]
    post = [[[1, 0, 0], [0, 0, 1], None, [1, 0, 1], [0, 0, 0]]]
    status = [["Success", "Success", "Failure", "Success", "Partial Success"]]
    d = QuenchData.from_braket([1.0], pre, post, filling=[1, 0, 1], statuses=status)
    assert d.n_atoms == 2 and d.shots.tolist() == [2] and d.bits[0].tolist() == [[0, 1], [1, 0]]
    raw = QuenchData.from_braket([1.0], pre, post, postselect=False, filling=[1, 0, 1], statuses=status)
    assert raw.shots.tolist() == [3] and np.isclose(raw.filled_fraction()[0], 1.0)
    with pytest.raises(ValueError):
        QuenchData.from_braket([1.0, 2.0], pre, post)


def test_empty_sites_read_as_rydberg_and_postselection_removes_them():
    m = RydbergQuenchModel(chain(4, 8.0))
    th = m.theta(12.0, 5.0)
    S, p = 4000, 0.05
    data = m.sample(th, [1.0, 3.0], S, np.random.default_rng(9), p_vacancy=p)
    full = (1 - p) ** 4
    assert np.all(np.abs(data.filled_fraction() - full) < 4 * np.sqrt(full * (1 - full) / S))
    for b, f in zip(data.bits, data.filled):
        assert np.all(b[f == 0] == 1)
    kept = data.postselect()
    P = m.probabilities(th, [1.0, 3.0])
    emp = kept.counts() / kept.shots[:, None]
    assert np.all(np.abs(emp - P) < 4.5 * np.sqrt(P * (1 - P) / kept.shots[:, None]) + 2e-3)
    raw = data.counts() / S
    assert np.abs(raw - P).max() > 0.01                                          # without post-selection the counts are biased


def test_cli_rydberg_quick(capsys):
    import json

    from shadow_learning.cli import main

    assert main(["rydberg", "--quick", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["shots_for_1pct_uniform"] > 0 and payload["waveform_violations"] == []
    assert payload["max_prob_change_under_omega_sign_flip"] < 1e-12
