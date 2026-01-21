"""
Tests for natural gradient optimization (natural_gradient.py)

Tests verify:
- QFI matrix computation
- Natural gradient descent convergence
- Comparison with vanilla gradient
- VQE helper functions
"""

import numpy as np
import pytest
from state import QuantumState, computational_basis
from gates import Ry, Rz, apply_gate, PAULI_Z
from natural_gradient import (
    quantum_fisher_information_matrix,
    compute_gradient,
    natural_gradient_step,
    natural_gradient_descent,
    vanilla_gradient_descent,
    make_vqe_state_fn,
    make_vqe_energy_fn,
    simple_ansatz,
    compare_optimizers,
    geodesic_parameter_distance,
    OptimizationResult
)


class TestQFIMatrix:
    """Test Quantum Fisher Information matrix computation."""

    def test_qfi_positive_semidefinite(self):
        """QFI matrix should be positive semidefinite."""
        def state_fn(params):
            theta, phi = params
            amplitudes = np.array([
                np.cos(theta / 2),
                np.exp(1j * phi) * np.sin(theta / 2)
            ], dtype=np.complex128)
            return QuantumState(1, amplitudes=amplitudes)

        params = np.array([np.pi / 4, np.pi / 3])
        qfi = quantum_fisher_information_matrix(state_fn, params)

        eigenvalues = np.linalg.eigvalsh(qfi)
        assert np.all(eigenvalues >= -1e-10)

    def test_qfi_symmetric(self):
        """QFI matrix should be symmetric."""
        def state_fn(params):
            theta, phi = params
            amplitudes = np.array([
                np.cos(theta / 2),
                np.exp(1j * phi) * np.sin(theta / 2)
            ], dtype=np.complex128)
            return QuantumState(1, amplitudes=amplitudes)

        params = np.array([np.pi / 4, np.pi / 3])
        qfi = quantum_fisher_information_matrix(state_fn, params)

        assert np.allclose(qfi, qfi.T)

    def test_qfi_bloch_sphere_metric(self):
        """QFI on Bloch sphere should give expected metric."""
        # For |ψ⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩, the Fubini-Study metric is 1/4
        # So QFI = 4 × (1/4) = 1 for the θ direction
        def state_fn(params):
            theta = params[0]
            amplitudes = np.array([
                np.cos(theta / 2),
                np.sin(theta / 2)
            ], dtype=np.complex128)
            return QuantumState(1, amplitudes=amplitudes)

        params = np.array([np.pi / 4])
        qfi = quantum_fisher_information_matrix(state_fn, params)

        # QFI for single-parameter rotation should be 1
        assert qfi[0, 0] == pytest.approx(1.0, rel=0.1)


class TestGradientComputation:
    """Test gradient computation."""

    def test_gradient_at_minimum(self):
        """Gradient should be near zero at minimum."""
        def quadratic(params):
            return np.sum(params**2)

        params = np.array([0.0, 0.0])
        grad = compute_gradient(quadratic, params)

        assert np.allclose(grad, 0, atol=1e-4)

    def test_gradient_direction(self):
        """Gradient should point uphill."""
        def quadratic(params):
            return np.sum(params**2)

        params = np.array([1.0, 2.0])
        grad = compute_gradient(quadratic, params)

        # For f = x² + y², gradient is [2x, 2y]
        expected = np.array([2.0, 4.0])
        assert np.allclose(grad, expected, rtol=0.01)


class TestNaturalGradientStep:
    """Test single natural gradient update."""

    def test_step_reduces_energy(self):
        """Natural gradient step should reduce energy for quadratic."""
        def energy_fn(params):
            return np.sum(params**2)

        def state_fn(params):
            # Dummy state function
            amplitudes = np.array([np.cos(params[0]), np.sin(params[0])],
                                   dtype=np.complex128)
            return QuantumState(1, amplitudes=amplitudes)

        params = np.array([1.0])
        grad = compute_gradient(energy_fn, params)
        qfi = quantum_fisher_information_matrix(state_fn, params)

        new_params = natural_gradient_step(params, grad, qfi, learning_rate=0.1)

        # Energy should decrease
        assert energy_fn(new_params) < energy_fn(params)

    def test_step_with_identity_qfi(self):
        """With identity QFI, natural gradient = vanilla gradient."""
        params = np.array([1.0, 2.0])
        gradient = np.array([0.5, 1.0])
        qfi = np.eye(2)  # Identity metric

        new_params = natural_gradient_step(params, gradient, qfi, learning_rate=0.1)

        # Should equal params - 0.1 * gradient
        expected = params - 0.1 * gradient
        assert np.allclose(new_params, expected)


class TestNaturalGradientDescent:
    """Test full natural gradient optimization."""

    def test_converges_simple_problem(self):
        """Should converge on simple 1-qubit rotation optimization."""
        # Minimize ⟨Z⟩ for Ry(θ)|0⟩
        # Minimum at θ = π where state is |1⟩
        def state_fn(params):
            theta = params[0]
            ry = Ry(theta)
            amplitudes = ry.matrix @ np.array([1, 0], dtype=np.complex128)
            return QuantumState(1, amplitudes=amplitudes)

        def energy_fn(params):
            state = state_fn(params)
            return np.real(state.expectation(PAULI_Z))

        initial_params = np.array([0.1])  # Start near |0⟩

        result = natural_gradient_descent(
            state_fn, energy_fn, initial_params,
            learning_rate=0.5,
            max_iterations=50
        )

        # Should find minimum at θ ≈ π
        final_theta = result.params[0] % (2 * np.pi)
        assert abs(final_theta - np.pi) < 0.3 or result.energies[-1] < -0.9

    def test_energy_decreases(self):
        """Energy should generally decrease during optimization."""
        def state_fn(params):
            theta = params[0]
            amplitudes = np.array([
                np.cos(theta / 2),
                np.sin(theta / 2)
            ], dtype=np.complex128)
            return QuantumState(1, amplitudes=amplitudes)

        def energy_fn(params):
            state = state_fn(params)
            return np.real(state.expectation(PAULI_Z))

        initial_params = np.array([0.5])

        result = natural_gradient_descent(
            state_fn, energy_fn, initial_params,
            learning_rate=0.3,
            max_iterations=20
        )

        # Final energy should be less than initial
        assert result.energies[-1] <= result.energies[0] + 1e-6

    def test_returns_optimization_result(self):
        """Should return properly structured OptimizationResult."""
        def state_fn(params):
            return QuantumState(1, amplitudes=np.array([1, 0], dtype=np.complex128))

        def energy_fn(params):
            return float(params[0]**2)

        result = natural_gradient_descent(
            state_fn, energy_fn, np.array([1.0]),
            max_iterations=10
        )

        assert isinstance(result, OptimizationResult)
        assert len(result.energies) > 0
        assert len(result.param_history) == len(result.energies)
        assert result.iterations == len(result.energies) - 1


class TestVanillaGradientDescent:
    """Test vanilla gradient descent for comparison."""

    def test_converges_quadratic(self):
        """Should converge on simple quadratic."""
        def energy_fn(params):
            return np.sum(params**2)

        result = vanilla_gradient_descent(
            energy_fn, np.array([1.0, 1.0]),
            learning_rate=0.1,
            max_iterations=100
        )

        # Should reach near zero
        assert result.energies[-1] < 0.01

    def test_returns_optimization_result(self):
        """Should return properly structured result."""
        def energy_fn(params):
            return float(params[0]**2)

        result = vanilla_gradient_descent(
            energy_fn, np.array([1.0]),
            max_iterations=10
        )

        assert isinstance(result, OptimizationResult)


class TestVQEHelpers:
    """Test VQE helper functions."""

    def test_make_vqe_state_fn(self):
        """Should create working state function."""
        state_fn = make_vqe_state_fn(simple_ansatz, n_qubits=2)

        params = np.array([np.pi/4, np.pi/3])
        state = state_fn(params)

        assert state.n_qubits == 2
        assert state.purity == pytest.approx(1.0)

    def test_make_vqe_energy_fn(self):
        """Should create working energy function."""
        H = np.kron(PAULI_Z, np.eye(2))  # Z on first qubit
        energy_fn = make_vqe_energy_fn(simple_ansatz, H, n_qubits=2)

        params = np.array([0.0, 0.0])  # |00⟩
        energy = energy_fn(params)

        # ⟨00|Z⊗I|00⟩ = 1
        assert energy == pytest.approx(1.0, abs=0.1)

    def test_simple_ansatz_creates_valid_state(self):
        """Simple ansatz should create normalized pure state."""
        params = np.array([np.pi/4, np.pi/2, np.pi/3])
        state = simple_ansatz(params, n_qubits=3)

        assert state.n_qubits == 3
        assert state.purity == pytest.approx(1.0)
        assert np.sum(state.probabilities()) == pytest.approx(1.0)


class TestCompareOptimizers:
    """Test optimizer comparison."""

    def test_both_optimizers_run(self):
        """Both optimizers should complete."""
        def state_fn(params):
            amplitudes = np.array([
                np.cos(params[0] / 2),
                np.sin(params[0] / 2)
            ], dtype=np.complex128)
            return QuantumState(1, amplitudes=amplitudes)

        def energy_fn(params):
            state = state_fn(params)
            return np.real(state.expectation(PAULI_Z))

        natural_result, vanilla_result = compare_optimizers(
            state_fn, energy_fn, np.array([0.5]),
            max_iterations=20
        )

        assert len(natural_result.energies) > 0
        assert len(vanilla_result.energies) > 0


class TestGeodesicParameterDistance:
    """Test geodesic distance computation."""

    def test_distance_with_self_zero(self):
        """Distance from point to itself should be zero."""
        def state_fn(params):
            amplitudes = np.array([
                np.cos(params[0] / 2),
                np.sin(params[0] / 2)
            ], dtype=np.complex128)
            return QuantumState(1, amplitudes=amplitudes)

        params = np.array([np.pi/4])
        dist = geodesic_parameter_distance(state_fn, params, params)

        assert dist == pytest.approx(0.0, abs=1e-10)

    def test_distance_symmetric(self):
        """Distance should be symmetric."""
        def state_fn(params):
            amplitudes = np.array([
                np.cos(params[0] / 2),
                np.sin(params[0] / 2)
            ], dtype=np.complex128)
            return QuantumState(1, amplitudes=amplitudes)

        params1 = np.array([np.pi/4])
        params2 = np.array([np.pi/2])

        d12 = geodesic_parameter_distance(state_fn, params1, params2)
        d21 = geodesic_parameter_distance(state_fn, params2, params1)

        assert d12 == pytest.approx(d21)

    def test_distance_positive(self):
        """Distance should be non-negative."""
        def state_fn(params):
            amplitudes = np.array([
                np.cos(params[0] / 2),
                np.sin(params[0] / 2)
            ], dtype=np.complex128)
            return QuantumState(1, amplitudes=amplitudes)

        for _ in range(10):
            params1 = np.random.randn(1)
            params2 = np.random.randn(1)
            dist = geodesic_parameter_distance(state_fn, params1, params2)
            assert dist >= 0


class TestNaturalGradientVsVanilla:
    """Test that natural gradient has advantages over vanilla."""

    def test_natural_gradient_uses_geometry(self):
        """Natural gradient should account for parameter space geometry."""
        # Create a problem where geometry matters
        def state_fn(params):
            # Bloch sphere parameterization
            theta, phi = params
            amplitudes = np.array([
                np.cos(theta / 2),
                np.exp(1j * phi) * np.sin(theta / 2)
            ], dtype=np.complex128)
            return QuantumState(1, amplitudes=amplitudes)

        def energy_fn(params):
            state = state_fn(params)
            return np.real(state.expectation(PAULI_Z))

        # Near the pole (small θ), φ changes don't matter much
        # Natural gradient should account for this via the metric
        params = np.array([0.1, 1.0])  # Small theta, arbitrary phi
        qfi = quantum_fisher_information_matrix(state_fn, params)

        # QFI in φ direction should be small near poles
        # (sin²θ factor in the metric)
        assert qfi[1, 1] < qfi[0, 0]
