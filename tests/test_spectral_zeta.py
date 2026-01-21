"""
Tests for spectral zeta functions (spectral_zeta.py)

Tests verify:
- Spectral zeta of Hamiltonians (Hermitian operators)
- Vectorized Lindbladian superoperator construction
- Spectral zeta of Lindbladians (non-Hermitian)
- Heat kernel trace (Mellin partner)
- Spectral signatures for fingerprinting
- Geodesic-spectral connections

Mathematical foundations:
    ζ_H(s) = Tr(H^{-s}) = Σₙ λₙ^{-s}  (for λₙ > 0)

    Heat kernel relation (Mellin transform):
    ζ_H(s) = (1/Γ(s)) ∫₀^∞ t^{s-1} Tr(e^{-tH}) dt
"""

import numpy as np
import pytest
from state import QuantumState, computational_basis, plus_state
from gates import PAULI_X, PAULI_Y, PAULI_Z
from lindblad import amplitude_damping_ops, phase_damping_ops, depolarizing_ops
from spectral_zeta import (
    spectral_zeta_hamiltonian,
    vectorize_lindbladian,
    lindbladian_spectrum,
    spectral_zeta_lindbladian,
    heat_kernel_trace,
    spectral_signature,
    spectral_distance,
    mellin_from_heat_kernel,
    SpectralFingerprint
)


class TestSpectralZetaHamiltonian:
    """Test spectral zeta for Hermitian Hamiltonians."""

    def test_identity_hamiltonian(self):
        """ζ_I(s) = d for d-dimensional identity (all eigenvalues = 1)."""
        H = np.eye(2)
        # ζ(s) = Σ 1^{-s} = 2
        zeta = spectral_zeta_hamiltonian(H, s=1.0)
        assert zeta == pytest.approx(2.0)

    def test_diagonal_hamiltonian(self):
        """ζ for diagonal H = Σ λᵢ^{-s}."""
        H = np.diag([1.0, 2.0, 4.0])
        # ζ(1) = 1 + 1/2 + 1/4 = 1.75
        zeta = spectral_zeta_hamiltonian(H, s=1.0)
        assert zeta == pytest.approx(1.75)

        # ζ(2) = 1 + 1/4 + 1/16 = 1.3125
        zeta2 = spectral_zeta_hamiltonian(H, s=2.0)
        assert zeta2 == pytest.approx(1.3125)

    def test_pauli_z_shifted(self):
        """Pauli Z has eigenvalues ±1, need shift for positive spectrum."""
        # Shift: H = Z + 2I has eigenvalues 1, 3
        H = PAULI_Z + 2 * np.eye(2)
        zeta = spectral_zeta_hamiltonian(H, s=1.0)
        # ζ(1) = 1/1 + 1/3 = 4/3
        assert zeta == pytest.approx(4/3, rel=1e-6)

    def test_zeta_decreases_with_s(self):
        """For eigenvalues > 1, ζ(s) decreases as s increases."""
        H = np.diag([2.0, 3.0, 5.0])
        zeta_1 = spectral_zeta_hamiltonian(H, s=1.0)
        zeta_2 = spectral_zeta_hamiltonian(H, s=2.0)
        zeta_3 = spectral_zeta_hamiltonian(H, s=3.0)

        assert zeta_1 > zeta_2 > zeta_3

    def test_complex_s_parameter(self):
        """Spectral zeta should work for complex s."""
        H = np.diag([1.0, 2.0])
        s = 1.0 + 0.5j
        zeta = spectral_zeta_hamiltonian(H, s=s)

        # Should be complex
        assert isinstance(zeta, (complex, np.complexfloating))
        # Manual: 1^{-s} + 2^{-s} = 1 + 2^{-1-0.5j}
        expected = 1.0 + 2.0**(-s)
        assert zeta == pytest.approx(expected, rel=1e-6)

    def test_regularization_for_zero_eigenvalues(self):
        """Zero eigenvalues should be regularized (skipped or capped)."""
        H = np.diag([0.0, 1.0, 2.0])
        # Should not raise, should skip or regularize the zero
        zeta = spectral_zeta_hamiltonian(H, s=1.0)
        # Only non-zero eigenvalues: 1 + 1/2 = 1.5
        assert zeta == pytest.approx(1.5, rel=0.1)


class TestVectorizedLindbladian:
    """Test vectorization of Lindblad superoperator."""

    def test_unitary_lindbladian_shape(self):
        """Pure Hamiltonian Lindbladian should be d²×d²."""
        H = PAULI_Z
        L_matrix = vectorize_lindbladian(H, [])

        # For 2×2 system, superoperator is 4×4
        assert L_matrix.shape == (4, 4)

    def test_pure_hamiltonian_is_antisymmetric(self):
        """L = -i[H, ·] should have purely imaginary eigenvalues."""
        H = PAULI_Z
        L_matrix = vectorize_lindbladian(H, [])
        eigenvalues = np.linalg.eigvals(L_matrix)

        # Real parts should be zero (no dissipation)
        real_parts = np.real(eigenvalues)
        assert np.allclose(real_parts, 0, atol=1e-10)

    def test_dissipator_gives_negative_real_eigenvalues(self):
        """Adding dissipation should give eigenvalues with Re(λ) ≤ 0."""
        H = np.zeros((2, 2))
        ops = amplitude_damping_ops(gamma=0.5, n_qubits=1)
        L_matrix = vectorize_lindbladian(H, ops)

        eigenvalues = np.linalg.eigvals(L_matrix)
        real_parts = np.real(eigenvalues)

        # All real parts should be ≤ 0 (Lindbladian is dissipative)
        assert np.all(real_parts <= 1e-10)

    def test_trace_preserving(self):
        """Lindbladian should preserve trace: Tr(L[ρ]) = 0 for all ρ."""
        H = PAULI_Z
        ops = amplitude_damping_ops(gamma=0.1, n_qubits=1)
        L_matrix = vectorize_lindbladian(H, ops)

        # For any ρ, Tr(L[ρ]) should be 0
        # In vectorized form: trace vector · L · vec(ρ) = 0
        # The trace vector is vec(I)
        trace_vec = np.array([1, 0, 0, 1])  # vec(I) for 2×2

        # L^T · trace_vec should be zero
        assert np.allclose(L_matrix.T @ trace_vec, 0, atol=1e-10)


class TestLindbladianSpectrum:
    """Test spectrum extraction from Lindbladian."""

    def test_spectrum_count(self):
        """Lindbladian should have d² eigenvalues."""
        H = PAULI_Z
        ops = amplitude_damping_ops(gamma=0.1, n_qubits=1)
        spectrum = lindbladian_spectrum(H, ops)

        assert len(spectrum) == 4  # 2² for single qubit

    def test_zero_eigenvalue_exists(self):
        """Lindbladian should have at least one zero eigenvalue (steady state)."""
        H = np.zeros((2, 2))
        ops = amplitude_damping_ops(gamma=0.5, n_qubits=1)
        spectrum = lindbladian_spectrum(H, ops)

        # At least one eigenvalue should be (nearly) zero
        min_abs = np.min(np.abs(spectrum))
        assert min_abs < 1e-10

    def test_decay_rates_from_amplitude_damping(self):
        """Amplitude damping should give specific decay rate."""
        gamma = 0.3
        H = np.zeros((2, 2))
        ops = amplitude_damping_ops(gamma=gamma, n_qubits=1)
        spectrum = lindbladian_spectrum(H, ops)

        # Non-zero eigenvalues should have Re(λ) related to gamma
        nonzero = spectrum[np.abs(spectrum) > 1e-10]
        decay_rates = -np.real(nonzero)

        # The decay rate should be proportional to gamma
        assert np.all(decay_rates >= 0)
        assert np.max(decay_rates) > 0.1 * gamma  # At least some decay


class TestSpectralZetaLindbladian:
    """Test spectral zeta for Lindbladian (non-Hermitian)."""

    def test_zeta_lindbladian_exists(self):
        """Should compute without error."""
        H = PAULI_Z
        ops = amplitude_damping_ops(gamma=0.1, n_qubits=1)

        zeta = spectral_zeta_lindbladian(H, ops, s=1.0)
        assert np.isfinite(zeta)

    def test_zeta_lindbladian_is_complex(self):
        """Lindbladian zeta should generally be complex."""
        H = PAULI_Z
        ops = amplitude_damping_ops(gamma=0.1, n_qubits=1)

        zeta = spectral_zeta_lindbladian(H, ops, s=1.0)
        # With imaginary eigenvalue parts, zeta is complex
        assert isinstance(zeta, (complex, np.complexfloating))

    def test_pure_hamiltonian_lindbladian(self):
        """With no dissipation, Lindbladian zeta relates to Hamiltonian."""
        H = np.diag([1.0, 2.0])

        # Pure Hamiltonian Lindbladian has eigenvalues ±i(λ_i - λ_j)
        zeta = spectral_zeta_lindbladian(H, [], s=1.0)
        assert np.isfinite(zeta)


class TestHeatKernelTrace:
    """Test heat kernel trace computation."""

    def test_heat_kernel_hamiltonian_t0(self):
        """At t=0, Tr(e^{-0·H}) = d (dimension)."""
        H = np.diag([1.0, 2.0, 3.0])
        trace = heat_kernel_trace(H, t=0.0)
        assert trace == pytest.approx(3.0)

    def test_heat_kernel_decay(self):
        """Heat kernel trace should decay for positive H."""
        H = np.diag([1.0, 2.0, 3.0])

        t1 = heat_kernel_trace(H, t=0.1)
        t2 = heat_kernel_trace(H, t=1.0)
        t3 = heat_kernel_trace(H, t=10.0)

        assert t1 > t2 > t3

    def test_heat_kernel_identity(self):
        """For H = I, Tr(e^{-tI}) = d·e^{-t}."""
        d = 4
        H = np.eye(d)
        t = 0.5

        trace = heat_kernel_trace(H, t=t)
        expected = d * np.exp(-t)
        assert trace == pytest.approx(expected)

    def test_heat_kernel_lindbladian(self):
        """Heat kernel for Lindbladian should work."""
        H = PAULI_Z
        ops = amplitude_damping_ops(gamma=0.1, n_qubits=1)

        # Should compute without error
        trace = heat_kernel_trace(H, t=1.0, lindblad_ops=ops)
        assert np.isfinite(trace)


class TestMellinFromHeatKernel:
    """Test Mellin transform relation between heat kernel and zeta."""

    def test_mellin_recovers_zeta(self):
        """ζ(s) = (1/Γ(s)) ∫ t^{s-1} K(t) dt should recover spectral zeta."""
        H = np.diag([1.0, 2.0])
        s = 1.5

        # Direct computation
        zeta_direct = spectral_zeta_hamiltonian(H, s)

        # Via Mellin transform of heat kernel
        zeta_mellin = mellin_from_heat_kernel(H, s)

        # Should be close (numerical integration vs exact)
        assert zeta_mellin == pytest.approx(zeta_direct, rel=0.05)

    def test_mellin_balance_point(self):
        """At s=1/2, Mellin transform is unitary (special structure)."""
        H = np.diag([1.0, 2.0, 3.0])
        s = 0.5

        zeta = mellin_from_heat_kernel(H, s)
        assert np.isfinite(zeta)


class TestSpectralSignature:
    """Test spectral fingerprinting."""

    def test_signature_shape(self):
        """Signature should have correct shape."""
        H = PAULI_Z + 2*np.eye(2)
        s_values = np.linspace(0.5, 3.0, 20)

        sig = spectral_signature(H, s_values)
        assert len(sig.values) == 20

    def test_signature_uniqueness(self):
        """Different Hamiltonians should give different signatures."""
        H1 = np.diag([1.0, 2.0])
        H2 = np.diag([1.0, 3.0])
        s_values = np.linspace(0.5, 3.0, 20)

        sig1 = spectral_signature(H1, s_values)
        sig2 = spectral_signature(H2, s_values)

        # Should be different
        assert not np.allclose(sig1.values, sig2.values)

    def test_signature_invariant_to_basis(self):
        """Unitarily equivalent Hamiltonians should have same signature."""
        H1 = np.diag([1.0, 2.0])
        # Rotate by arbitrary unitary
        U = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        H2 = U @ H1 @ U.conj().T

        s_values = np.linspace(0.5, 3.0, 20)
        sig1 = spectral_signature(H1, s_values)
        sig2 = spectral_signature(H2, s_values)

        # Same spectrum → same signature
        assert np.allclose(sig1.values, sig2.values, rtol=1e-6)

    def test_fingerprint_dataclass(self):
        """SpectralFingerprint should contain expected fields."""
        H = PAULI_Z + 2*np.eye(2)
        sig = spectral_signature(H, np.linspace(0.5, 2.0, 10))

        assert hasattr(sig, 'values')
        assert hasattr(sig, 's_values')
        assert hasattr(sig, 'eigenvalues')


class TestSpectralDistance:
    """Test distance between spectral signatures."""

    def test_distance_with_self_is_zero(self):
        """Distance from signature to itself should be zero."""
        H = np.diag([1.0, 2.0, 3.0])
        s_values = np.linspace(0.5, 3.0, 20)
        sig = spectral_signature(H, s_values)

        dist = spectral_distance(sig, sig)
        assert dist == pytest.approx(0.0, abs=1e-10)

    def test_distance_symmetric(self):
        """Distance should be symmetric."""
        H1 = np.diag([1.0, 2.0])
        H2 = np.diag([1.0, 3.0])
        s_values = np.linspace(0.5, 3.0, 20)

        sig1 = spectral_signature(H1, s_values)
        sig2 = spectral_signature(H2, s_values)

        d12 = spectral_distance(sig1, sig2)
        d21 = spectral_distance(sig2, sig1)

        assert d12 == pytest.approx(d21)

    def test_distance_positive(self):
        """Distance should be non-negative."""
        H1 = np.diag([1.0, 2.0])
        H2 = np.diag([1.5, 2.5])
        s_values = np.linspace(0.5, 3.0, 20)

        sig1 = spectral_signature(H1, s_values)
        sig2 = spectral_signature(H2, s_values)

        assert spectral_distance(sig1, sig2) >= 0


class TestGeodesicSpectralConnection:
    """Test connection between geodesic deviation and spectral structure."""

    def test_hamiltonian_vs_lindbladian_spectra(self):
        """Lindbladian spectrum should differ from Hamiltonian when noise present."""
        H = PAULI_Z + 2*np.eye(2)
        ops = amplitude_damping_ops(gamma=0.5, n_qubits=1)

        s_values = np.linspace(0.5, 2.0, 10)
        sig_H = spectral_signature(H, s_values)
        sig_L = spectral_signature(H, s_values, lindblad_ops=ops)

        # Should be different
        dist = spectral_distance(sig_H, sig_L)
        assert dist > 0.01

    def test_noise_strength_affects_spectral_distance(self):
        """Different noise strengths should give different spectral signatures."""
        H = PAULI_Z + 2*np.eye(2)
        s_values = np.linspace(0.5, 2.0, 10)

        sig_H = spectral_signature(H, s_values)

        sig_weak = spectral_signature(H, s_values,
                                       lindblad_ops=amplitude_damping_ops(0.1, 1))
        sig_strong = spectral_signature(H, s_values,
                                         lindblad_ops=amplitude_damping_ops(0.5, 1))

        dist_weak = spectral_distance(sig_H, sig_weak)
        dist_strong = spectral_distance(sig_H, sig_strong)

        # Both should show deviation from ideal
        assert dist_weak > 0
        assert dist_strong > 0

        # Weak and strong should differ from each other
        dist_weak_strong = spectral_distance(sig_weak, sig_strong)
        assert dist_weak_strong > 0


class TestTwoQubitSystems:
    """Test spectral zeta on larger systems."""

    def test_two_qubit_hamiltonian(self):
        """Should handle 2-qubit Hamiltonians."""
        # ZZ interaction
        H = np.kron(PAULI_Z, PAULI_Z) + np.eye(4)  # Shift for positive spectrum

        zeta = spectral_zeta_hamiltonian(H, s=1.0)
        assert np.isfinite(zeta)

    def test_two_qubit_lindbladian(self):
        """Should handle 2-qubit Lindbladians (16×16 superoperator)."""
        H = np.kron(PAULI_Z, np.eye(2))
        ops = amplitude_damping_ops(gamma=0.1, n_qubits=2)

        spectrum = lindbladian_spectrum(H, ops)

        # 4×4 density matrix → 16 eigenvalues
        assert len(spectrum) == 16
