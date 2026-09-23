"""
Regression tests for failure modes found in the v0.3 adversarial audit:
degenerate spectra, non-finite restarts, unphysical process estimates,
EKF covariance health and exponential-cost matchgate sampling.
"""

import numpy as np
import pytest

from shadow_learning import (
    DifferentiableHamiltonianLearner,
    ShadowProcessTomographer,
    StreamingKalmanHamiltonianTracker,
    gibbs_state,
    pauli_transfer_matrix_of_unitary,
    sample_shadows_from_density_matrix,
)
from shadow_learning import differentiable_inversion as di
from shadow_learning.fermionic_shadows import (
    FermionicGaussianState,
    FermionicMatchgateShadows,
    slater_state,
    two_body_energy,
)


def _chain(n, j=0.5):
    J = np.zeros((n, n))
    for i in range(n - 1):
        J[i, i + 1] = J[i + 1, i] = j
    return J


# --------------------------------------------------------------------------- #
# degenerate Heisenberg spectra
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [3, 4])
def test_gradient_is_finite_and_exact_on_isotropic_chain(n):
    """h = 0 makes every level an SU(2) multiplet; eigh-based autodiff returned NaN here."""
    L = DifferentiableHamiltonianLearner(n, beta=1.0)
    J, h = _chain(n), np.zeros(n)
    p = L.pack(J, h)
    t = L.gibbs_observables(p)
    w = np.linalg.eigvalsh(L.hamiltonian_from_params(p))
    assert np.min(np.diff(w)) < 1e-12                      # the spectrum really is degenerate
    obs, jac = L.gibbs_observables_and_jacobian(p)
    eps = 1e-6
    fd = np.stack([(L.gibbs_observables(p + eps * e) - L.gibbs_observables(p - eps * e)) / (2 * eps) for e in np.eye(len(p))], 1)
    assert np.all(np.isfinite(jac)) and np.abs(jac - fd).max() < 1e-7
    for q in (p, np.full_like(p, 0.05)):                  # the truth and the old symmetric start point
        v, g = L.loss_and_grad(q, t)
        assert np.all(np.isfinite(g))


@pytest.mark.skipif(not di.HAS_JAX, reason="needs JAX")
def test_jax_backend_agrees_with_analytic_at_degeneracy():
    L = DifferentiableHamiltonianLearner(3, beta=1.0)
    p = L.pack(_chain(3), np.zeros(3))
    t = L.gibbs_observables(p + 0.1)
    for q in (p, np.full_like(p, 0.05)):
        va, ga = L.loss_and_grad(q, t, backend="analytic")
        vj, gj = L.loss_and_grad(q, t, backend="jax")
        assert abs(va - vj) < 1e-12 and np.abs(ga - gj).max() < 1e-10


@pytest.mark.parametrize("n", [3, 4])
def test_isotropic_chain_is_recovered(n):
    L = DifferentiableHamiltonianLearner(n, beta=1.0, seed=0)
    J, h = _chain(n), np.zeros(n)
    res = L.learn_from_expectations(L.compute_thermal_observables(J, h), J, h)
    assert np.all(np.isfinite(res.loss_history))
    assert res.frobenius_error < 1e-5 and res.h_error < 1e-5


def test_non_finite_restart_never_wins(monkeypatch):
    """A NaN run must be skipped: `x < NaN` is always False, which used to freeze the first result."""
    L = DifferentiableHamiltonianLearner(2, beta=1.0, seed=0)
    J, h = np.array([[0, 0.6], [0.6, 0]]), np.array([0.2, -0.1])
    targets = L.compute_thermal_observables(J, h)
    import scipy.optimize

    real_minimize = scipy.optimize.minimize
    calls = {"n": 0}

    def minimize_nan_first(*args, **kwargs):             # restart 0 "converges" to NaN
        calls["n"] += 1
        res = real_minimize(*args, **kwargs)
        if calls["n"] == 1:
            res.fun = float("nan")
        return res

    monkeypatch.setattr(di.scipy.optimize, "minimize", minimize_nan_first)
    res = L.learn_from_expectations(targets, J, h, n_restarts=3)
    assert calls["n"] >= 2
    assert np.isfinite(res.final_loss) and res.frobenius_error < 1e-4


def test_all_restarts_non_finite_raises(monkeypatch):
    L = DifferentiableHamiltonianLearner(2, beta=1.0, seed=0)
    monkeypatch.setattr(L, "loss_and_grad", lambda p, t, backend="analytic": (float("nan"), np.full(len(p), np.nan)))
    with pytest.raises(RuntimeError):
        L.learn_from_expectations({n: 0.0 for n in L.observable_names})


# --------------------------------------------------------------------------- #
# physical process tomography
# --------------------------------------------------------------------------- #
def _noisy_cz():
    CZ = np.diag([1, 1, 1, -1]).astype(complex)
    return CZ, (lambda r: 0.9 * CZ @ r @ CZ + 0.1 * np.trace(r) * np.eye(4) / 4)


def test_projected_process_estimate_is_cptp_and_closer_to_truth():
    CZ, channel = _noisy_cz()
    R_true = 0.9 * pauli_transfer_matrix_of_unitary(CZ)
    R_true[0, 0] = 1.0
    for seed in range(3):
        tomo = ShadowProcessTomographer(n_qubits=2, seed=seed)
        out = tomo.run(channel, n_snapshots_per_input=300)
        J_true = tomo.choi_matrix(R_true)
        assert not tomo.is_physical(out["choi_raw"])       # shot noise breaks CP at this budget
        assert tomo.is_physical(out["choi"])
        assert np.linalg.norm(out["choi"] - J_true) <= np.linalg.norm(out["choi_raw"] - J_true) + 1e-12
        f = tomo.process_fidelity(out["ptm"], CZ)
        assert 0.0 <= f <= 1.0


def test_choi_and_ptm_round_trip():
    tomo = ShadowProcessTomographer(n_qubits=2, seed=0)
    R = pauli_transfer_matrix_of_unitary(np.linalg.qr(np.random.default_rng(1).normal(size=(4, 4)) + 0j)[0])
    assert np.allclose(tomo.ptm_from_choi(tomo.choi_matrix(R)), R, atol=1e-12)


# --------------------------------------------------------------------------- #
# EKF health
# --------------------------------------------------------------------------- #
def test_ekf_covariance_stays_symmetric_psd_and_calibrated():
    J, h = np.array([[0, 0.5], [0.5, 0]]), np.array([0.3, -0.2])
    errs, stds = [], []
    for seed in range(4):
        tr = StreamingKalmanHamiltonianTracker(n_qubits=2, beta=0.6, process_noise_std=0.0, prior_std=0.5, seed=seed)
        rho = gibbs_state(tr.learner.build_hamiltonian_matrix(J, h), 0.6)
        for snap in sample_shadows_from_density_matrix(rho, 3000, np.random.default_rng(seed)):
            tr.process_snapshot(snap)
        assert np.allclose(tr.P, tr.P.T, atol=1e-14)
        assert np.linalg.eigvalsh(tr.P).min() > 0
        est = tr.estimate()
        errs.append(est["params"] - tr.learner.pack(J, h))
        stds.append(est["std"])
    z = np.abs(np.array(errs)) / np.array(stds)
    assert np.mean(z**2) < 4.0                            # reported σ is the right order (χ²/dof ≈ 1)


# --------------------------------------------------------------------------- #
# matchgate shadows
# --------------------------------------------------------------------------- #
def _random_gaussian(n, n_particles, seed):
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return FermionicGaussianState.ground_state_of_quadratic(h + h.conj().T, n_particles)


def test_gaussian_covariance_matches_dense_state():
    g = _random_gaussian(4, 2, 0)
    fs = FermionicMatchgateShadows(4)
    psi = g.to_state_vector()
    assert np.allclose(g.one_rdm(), fs.exact_1rdm(psi), atol=1e-12)
    assert np.allclose(fs.exact_2rdm(g), fs.exact_2rdm(psi), atol=1e-12)
    assert np.allclose(FermionicGaussianState.slater(4, [1, 3]).one_rdm(), fs.exact_1rdm(slater_state(4, [1, 3])))


def test_gaussian_and_dense_samplers_estimate_the_same_rdms():
    g = _random_gaussian(4, 2, 1)
    D1, D2 = g.one_rdm(), FermionicMatchgateShadows(4).exact_2rdm(g)
    for state, seed in ((g, 1), (g.to_state_vector(), 2)):
        fs = FermionicMatchgateShadows(4, seed=seed)
        snaps = fs.sample(state, 12000)
        assert np.abs(fs.estimate_1rdm(snaps) - D1).max() < 0.05
        assert np.abs(fs.estimate_2rdm(snaps) - D2).max() < 0.05


def test_two_rdm_of_a_non_gaussian_state_and_its_energy():
    n = 4
    psi = (slater_state(n, [1, 3]) + slater_state(n, [0, 2])) / np.sqrt(2)
    fs = FermionicMatchgateShadows(n, seed=3)
    snaps = fs.sample(psi, 15000)
    D1, D2 = fs.estimate_1rdm(snaps), fs.estimate_2rdm(snaps)
    D1x, D2x = fs.exact_1rdm(psi), fs.exact_2rdm(psi)
    assert np.abs(D2 - D2x).max() < 0.05
    # an interacting Hamiltonian: hopping + density–density repulsion
    h1 = -np.eye(n, k=1) - np.eye(n, k=-1)
    h2 = np.zeros((n, n, n, n))
    for p in range(n - 1):                                # U n_p n_{p+1} = U c_p† c_{p+1}† c_{p+1} c_p
        h2[p, p + 1, p + 1, p] = 2.0
    assert abs(two_body_energy(h1, h2, D1, D2) - two_body_energy(h1, h2, D1x, D2x)) < 0.1


def test_gaussian_sampler_is_polynomial():
    """24 modes would need a 2^24-dimensional state; the covariance sampler never builds one."""
    g = _random_gaussian(24, 12, 2)
    fs = FermionicMatchgateShadows(24, seed=0)
    snaps = fs.sample(g, 300)
    assert fs._gammas is None                             # no dense operator was ever formed
    assert len(snaps) == 300 and len(snaps[0].bits) == 24
    D = fs.estimate_1rdm(snaps)
    assert abs(np.trace(D).real - 12) < 3.0                # particle number, within shadow noise
