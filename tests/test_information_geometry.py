"""
Tests for information geometry module (information_geometry.py)

Tests verify:
- Fubini-Study distance properties
- Quantum Fisher Information
- Bures distance
- Entanglement measures
- Bloch sphere geometry
"""

import numpy as np
import pytest
from state import (
    QuantumState, computational_basis, plus_state, minus_state,
    bell_state, ghz_state, maximally_mixed
)
from gates import Ry, Rz, PAULI_X, PAULI_Y, PAULI_Z
from information_geometry import (
    fubini_study_distance,
    quantum_fisher_information,
    quantum_fisher_information_pure,
    bures_distance,
    fidelity,
    entanglement_entropy,
    concurrence,
    tangle,
    BlochCoordinates,
    bloch_geodesic,
    compute_qgt,
    QuantumGeometricTensor
)


class TestFubiniStudyDistance:
    """Test Fubini-Study metric properties."""

    def test_distance_with_self_is_zero(self):
        """d_FS(ψ, ψ) = 0."""
        state = plus_state(1)
        assert fubini_study_distance(state, state) == pytest.approx(0.0)

    def test_orthogonal_states_distance(self):
        """Orthogonal states have d_FS = π/2."""
        state0 = computational_basis(1, 0)
        state1 = computational_basis(1, 1)

        dist = fubini_study_distance(state0, state1)
        assert dist == pytest.approx(np.pi / 2)

    def test_distance_symmetric(self):
        """d_FS(ψ, φ) = d_FS(φ, ψ)."""
        state1 = plus_state(1)
        state2 = computational_basis(1, 0)

        d12 = fubini_study_distance(state1, state2)
        d21 = fubini_study_distance(state2, state1)

        assert d12 == pytest.approx(d21)

    def test_distance_triangle_inequality(self):
        """d(A,C) ≤ d(A,B) + d(B,C)."""
        state_a = computational_basis(1, 0)
        state_b = plus_state(1)
        state_c = computational_basis(1, 1)

        d_ac = fubini_study_distance(state_a, state_c)
        d_ab = fubini_study_distance(state_a, state_b)
        d_bc = fubini_study_distance(state_b, state_c)

        assert d_ac <= d_ab + d_bc + 1e-10

    def test_pure_state_requirement(self):
        """Should raise for mixed states."""
        pure = plus_state(1)
        mixed = maximally_mixed(1)

        with pytest.raises(ValueError):
            fubini_study_distance(pure, mixed)


class TestQuantumFisherInformation:
    """Test QFI calculations."""

    def test_qfi_pure_equals_4_variance(self):
        """For pure states, QFI = 4 Var(H)."""
        state = plus_state(1)
        generator = PAULI_Z

        qfi = quantum_fisher_information_pure(state, generator)

        # Var(Z) for |+⟩ = ⟨Z²⟩ - ⟨Z⟩² = 1 - 0 = 1
        # So QFI = 4
        assert qfi == pytest.approx(4.0)

    def test_qfi_eigenstate_is_zero(self):
        """QFI = 0 when state is eigenstate of generator."""
        state = computational_basis(1, 0)  # |0⟩ is eigenstate of Z
        generator = PAULI_Z

        qfi = quantum_fisher_information_pure(state, generator)
        assert qfi == pytest.approx(0.0, abs=1e-10)

    def test_qfi_non_negative(self):
        """QFI should always be non-negative."""
        for _ in range(10):
            state = QuantumState(1, amplitudes=np.random.randn(2) + 1j * np.random.randn(2))
            generator = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
            generator = (generator + generator.conj().T) / 2  # Make Hermitian

            qfi = quantum_fisher_information_pure(state, generator)
            assert qfi >= -1e-10

    def test_qfi_mixed_state(self):
        """QFI for mixed state should use general formula."""
        mixed = maximally_mixed(1)
        generator = PAULI_Z

        qfi = quantum_fisher_information(mixed.rho, generator)

        # For maximally mixed state, QFI = 0
        assert qfi == pytest.approx(0.0, abs=1e-10)


class TestBuresDistance:
    """Test Bures distance properties."""

    def test_bures_with_self_is_zero(self):
        """d_B(ρ, ρ) = 0."""
        state = plus_state(1)
        # Allow small numerical tolerance for sqrt operations in matrix square root
        assert bures_distance(state.rho, state.rho) == pytest.approx(0.0, abs=1e-6)

    def test_bures_non_negative(self):
        """Bures distance should be non-negative."""
        rho = plus_state(1).rho
        sigma = computational_basis(1, 0).rho

        assert bures_distance(rho, sigma) >= 0

    def test_bures_symmetric(self):
        """d_B(ρ, σ) = d_B(σ, ρ)."""
        rho = plus_state(1).rho
        sigma = maximally_mixed(1).rho

        assert bures_distance(rho, sigma) == pytest.approx(bures_distance(sigma, rho))

    def test_bures_reduces_to_fubini_study_for_pure(self):
        """For pure states, Bures ~ 2sin(d_FS/2)."""
        psi = plus_state(1)
        phi = computational_basis(1, 0)

        d_fs = fubini_study_distance(psi, phi)
        d_bures = bures_distance(psi.rho, phi.rho)

        # d_B = √(2(1 - cos(d_FS))) = 2|sin(d_FS/2)|
        expected = 2 * np.abs(np.sin(d_fs / 2))
        assert d_bures == pytest.approx(expected, rel=0.01)


class TestFidelity:
    """Test quantum fidelity."""

    def test_fidelity_with_self_is_one(self):
        """F(ρ, ρ) = 1."""
        state = bell_state(0)
        assert fidelity(state.rho, state.rho) == pytest.approx(1.0)

    def test_orthogonal_pure_states_fidelity_zero(self):
        """F(|0⟩, |1⟩) = 0."""
        rho = computational_basis(1, 0).rho
        sigma = computational_basis(1, 1).rho

        assert fidelity(rho, sigma) == pytest.approx(0.0)

    def test_fidelity_bounds(self):
        """0 ≤ F ≤ 1."""
        for _ in range(10):
            rho = QuantumState(1, amplitudes=np.random.randn(2) + 1j * np.random.randn(2)).rho
            sigma = QuantumState(1, amplitudes=np.random.randn(2) + 1j * np.random.randn(2)).rho

            f = fidelity(rho, sigma)
            assert 0 - 1e-10 <= f <= 1 + 1e-10


class TestEntanglementMeasures:
    """Test entanglement quantification."""

    def test_bell_state_maximally_entangled(self):
        """Bell states have max entanglement entropy = 1."""
        bell = bell_state(0)
        entropy = entanglement_entropy(bell, [0])

        assert entropy == pytest.approx(1.0)

    def test_product_state_zero_entropy(self):
        """Product states have zero entanglement entropy."""
        product = computational_basis(1, 0).tensor(computational_basis(1, 0))
        entropy = entanglement_entropy(product, [0])

        assert entropy == pytest.approx(0.0, abs=1e-10)

    def test_bell_state_concurrence_one(self):
        """Bell states have concurrence = 1."""
        for which in range(4):
            bell = bell_state(which)
            assert concurrence(bell) == pytest.approx(1.0)

    def test_product_state_concurrence_zero(self):
        """Product states have concurrence = 0."""
        product = computational_basis(1, 0).tensor(computational_basis(1, 0))
        assert concurrence(product) == pytest.approx(0.0, abs=1e-10)

    def test_tangle_is_squared_concurrence(self):
        """τ = C²."""
        state = bell_state(0)
        c = concurrence(state)
        t = tangle(state)

        assert t == pytest.approx(c**2)

    def test_concurrence_requires_two_qubits(self):
        """Concurrence should raise for non-2-qubit states."""
        with pytest.raises(ValueError):
            concurrence(ghz_state(3))


class TestBlochCoordinates:
    """Test Bloch sphere coordinate transformations."""

    def test_zero_state_coordinates(self):
        """|0⟩ at north pole: θ=0."""
        state = computational_basis(1, 0)
        coords = BlochCoordinates.from_state(state)

        assert coords.theta == pytest.approx(0.0)
        assert coords.r == pytest.approx(1.0)

    def test_one_state_coordinates(self):
        """|1⟩ at south pole: θ=π."""
        state = computational_basis(1, 1)
        coords = BlochCoordinates.from_state(state)

        assert coords.theta == pytest.approx(np.pi)
        assert coords.r == pytest.approx(1.0)

    def test_plus_state_coordinates(self):
        """|+⟩ on equator at φ=0."""
        state = plus_state(1)
        coords = BlochCoordinates.from_state(state)

        assert coords.theta == pytest.approx(np.pi / 2)
        assert coords.phi == pytest.approx(0.0)
        assert coords.r == pytest.approx(1.0)

    def test_cartesian_conversion(self):
        """Cartesian coordinates should match state's bloch_vector."""
        state = plus_state(1)
        coords = BlochCoordinates.from_state(state)

        bloch = state.bloch_vector()
        cartesian = coords.cartesian

        assert np.allclose(cartesian, bloch)

    def test_roundtrip_state_to_coords_to_state(self):
        """Converting state → coords → state should preserve state."""
        original = plus_state(1)
        coords = BlochCoordinates.from_state(original)
        reconstructed = coords.to_state()

        assert original.fidelity(reconstructed) == pytest.approx(1.0)

    def test_mixed_state_inside_sphere(self):
        """Mixed states have r < 1."""
        mixed = maximally_mixed(1)
        coords = BlochCoordinates.from_state(mixed)

        assert coords.r < 1.0
        assert coords.r == pytest.approx(0.0)  # Maximally mixed at center


class TestBlochGeodesic:
    """Test geodesic interpolation on Bloch sphere."""

    def test_geodesic_endpoints(self):
        """Geodesic at t=0 and t=1 should give endpoints."""
        start = BlochCoordinates(0, 0, 1)  # North pole
        end = BlochCoordinates(np.pi, 0, 1)  # South pole

        at_zero = bloch_geodesic(start, end, 0.0)
        at_one = bloch_geodesic(start, end, 1.0)

        assert at_zero.theta == pytest.approx(start.theta)
        assert at_one.theta == pytest.approx(end.theta)

    def test_geodesic_midpoint(self):
        """Midpoint of geodesic from N to S pole should be on equator."""
        start = BlochCoordinates(0, 0, 1)  # North pole
        end = BlochCoordinates(np.pi, 0, 1)  # South pole

        midpoint = bloch_geodesic(start, end, 0.5)

        # Should be at θ = π/2 (equator)
        assert midpoint.theta == pytest.approx(np.pi / 2)
        assert midpoint.r == pytest.approx(1.0)

    def test_geodesic_preserves_purity(self):
        """Geodesic between pure states stays on sphere surface."""
        start = BlochCoordinates.from_state(computational_basis(1, 0))
        end = BlochCoordinates.from_state(plus_state(1))

        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            point = bloch_geodesic(start, end, t)
            assert point.r == pytest.approx(1.0)


class TestQuantumGeometricTensor:
    """Test quantum geometric tensor computation."""

    def test_qgt_metric_symmetric(self):
        """Fubini-Study metric should be symmetric."""
        def state_fn(params):
            theta, phi = params
            amplitudes = np.array([
                np.cos(theta / 2),
                np.exp(1j * phi) * np.sin(theta / 2)
            ], dtype=np.complex128)
            return QuantumState(1, amplitudes=amplitudes)

        params = np.array([np.pi / 4, np.pi / 3])
        qgt = compute_qgt(state_fn, params)

        assert np.allclose(qgt.metric, qgt.metric.T)

    def test_qgt_metric_positive_semidefinite(self):
        """Metric tensor should be positive semidefinite."""
        def state_fn(params):
            theta, = params
            ry = Ry(theta)
            state = computational_basis(1, 0)
            return QuantumState(1, amplitudes=ry.matrix @ state.amplitudes)

        params = np.array([np.pi / 4])
        qgt = compute_qgt(state_fn, params)

        eigenvalues = np.linalg.eigvalsh(qgt.metric)
        assert np.all(eigenvalues >= -1e-10)

    def test_qgt_curvature_antisymmetric(self):
        """Berry curvature should be antisymmetric."""
        def state_fn(params):
            theta, phi = params
            amplitudes = np.array([
                np.cos(theta / 2),
                np.exp(1j * phi) * np.sin(theta / 2)
            ], dtype=np.complex128)
            return QuantumState(1, amplitudes=amplitudes)

        params = np.array([np.pi / 4, np.pi / 3])
        qgt = compute_qgt(state_fn, params)

        # Berry curvature F = -2 Im(Q) should be antisymmetric
        assert np.allclose(qgt.curvature, -qgt.curvature.T)


class TestFubiniStudyGeometry:
    """Test relationship between QFI and Fubini-Study metric."""

    def test_qfi_equals_4_times_fisher_rao(self):
        """For pure state evolution, QFI = 4 × classical Fisher info."""
        # For a qubit parameterized as |ψ(θ)⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
        # The Fubini-Study metric is g_θθ = 1/4
        # The QFI is F_Q = 4 × (1/4) = 1

        state = QuantumState(1, amplitudes=np.array([
            np.cos(np.pi / 8),
            np.sin(np.pi / 8)
        ], dtype=np.complex128))

        # Generator is dψ/dθ direction, which for Ry is Y/2
        generator = PAULI_Y / 2

        qfi = quantum_fisher_information_pure(state, generator)

        # QFI for Ry(θ) should be 1 (since d|ψ⟩/dθ has unit norm modulo projection)
        assert qfi == pytest.approx(1.0, rel=0.1)
