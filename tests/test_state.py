"""
Tests for quantum state representation (state.py)

Tests verify:
- State normalization
- Purity calculations
- Measurement statistics
- Density matrix properties
- Partial trace operations
- Standard state constructors
"""

import numpy as np
import pytest
from state import (
    QuantumState, StateType,
    computational_basis, plus_state, minus_state,
    bell_state, ghz_state, w_state,
    maximally_mixed, random_state
)


class TestQuantumStateBasics:
    """Test basic QuantumState creation and properties."""

    def test_default_state_is_zero(self):
        """Default single-qubit state should be |0⟩."""
        state = QuantumState(1)
        assert state.probability(0) == pytest.approx(1.0)
        assert state.probability(1) == pytest.approx(0.0)

    def test_state_is_normalized(self):
        """State should be automatically normalized."""
        # Create unnormalized amplitudes
        amplitudes = np.array([3, 4], dtype=np.complex128)
        state = QuantumState(1, amplitudes=amplitudes)

        # Should be normalized to unit length
        probs = state.probabilities()
        assert np.sum(probs) == pytest.approx(1.0)

    def test_n_qubit_dimensions(self):
        """State dimension should be 2^n."""
        for n in [1, 2, 3, 4]:
            state = QuantumState(n)
            assert state.dim == 2**n
            assert len(state.amplitudes) == 2**n


class TestPurity:
    """Test purity calculations for pure and mixed states."""

    def test_pure_state_purity_is_one(self):
        """Pure states should have purity = 1."""
        state = computational_basis(2, 0)
        assert state.purity == pytest.approx(1.0)

        state = plus_state(1)
        assert state.purity == pytest.approx(1.0)

        state = bell_state(0)
        assert state.purity == pytest.approx(1.0)

    def test_maximally_mixed_purity(self):
        """Maximally mixed state has purity = 1/d."""
        for n in [1, 2, 3]:
            state = maximally_mixed(n)
            expected_purity = 1 / (2**n)
            assert state.purity == pytest.approx(expected_purity)

    def test_mixed_state_purity_less_than_one(self):
        """Mixed states should have purity < 1."""
        state = maximally_mixed(1)
        assert state.purity < 1.0
        assert state.state_type == StateType.MIXED


class TestMeasurement:
    """Test measurement operations."""

    def test_computational_basis_measurement_deterministic(self):
        """Measuring |0⟩ should always yield 0."""
        state = computational_basis(1, 0)
        for _ in range(10):
            outcome, _ = state.measure(collapse=False)
            assert outcome == 0

    def test_plus_state_measurement_statistics(self):
        """Measuring |+⟩ should give 50/50 outcomes over many trials."""
        state = plus_state(1)
        outcomes = []
        for _ in range(1000):
            outcome, _ = state.measure(collapse=False)
            outcomes.append(outcome)

        # Should be roughly 50/50 (allowing for statistical variation)
        zeros = outcomes.count(0)
        assert 400 < zeros < 600  # ~50% with some tolerance

    def test_measurement_collapse(self):
        """Measurement with collapse should yield deterministic re-measurement."""
        state = plus_state(1)
        outcome1, collapsed = state.measure(collapse=True)

        # Re-measuring collapsed state should give same result
        for _ in range(10):
            outcome2, _ = collapsed.measure(collapse=False)
            assert outcome2 == outcome1

    def test_single_qubit_measurement(self):
        """Single qubit measurement should project correctly."""
        state = bell_state(0)  # (|00⟩ + |11⟩)/√2

        outcome, post_state = state.measure_qubit(0, collapse=True)

        # If we measured 0, state should be |00⟩
        # If we measured 1, state should be |11⟩
        if outcome == 0:
            assert post_state.probability(0) == pytest.approx(1.0)  # |00⟩
        else:
            assert post_state.probability(3) == pytest.approx(1.0)  # |11⟩


class TestDensityMatrix:
    """Test density matrix properties."""

    def test_density_matrix_trace_one(self):
        """Density matrix should have trace 1."""
        for state in [computational_basis(1, 0), plus_state(1), bell_state(0)]:
            rho = state.rho
            assert np.trace(rho) == pytest.approx(1.0)

    def test_density_matrix_hermitian(self):
        """Density matrix should be Hermitian."""
        state = plus_state(2)
        rho = state.rho
        assert np.allclose(rho, rho.conj().T)

    def test_density_matrix_positive_semidefinite(self):
        """Density matrix eigenvalues should be non-negative."""
        state = bell_state(0)
        eigenvalues = np.linalg.eigvalsh(state.rho)
        assert np.all(eigenvalues >= -1e-10)


class TestVonNeumannEntropy:
    """Test von Neumann entropy calculations."""

    def test_pure_state_entropy_zero(self):
        """Pure states should have zero entropy."""
        state = computational_basis(2, 0)
        assert state.von_neumann_entropy == pytest.approx(0.0, abs=1e-10)

    def test_maximally_mixed_entropy(self):
        """Maximally mixed state has maximum entropy = log(d)."""
        for n in [1, 2]:
            state = maximally_mixed(n)
            expected_entropy = n  # log2(2^n) = n
            assert state.von_neumann_entropy == pytest.approx(expected_entropy, rel=0.01)


class TestFidelity:
    """Test fidelity between states."""

    def test_fidelity_with_self(self):
        """Fidelity with self should be 1."""
        state = plus_state(1)
        assert state.fidelity(state) == pytest.approx(1.0)

    def test_orthogonal_states_fidelity_zero(self):
        """Orthogonal states should have fidelity 0."""
        state0 = computational_basis(1, 0)
        state1 = computational_basis(1, 1)
        assert state0.fidelity(state1) == pytest.approx(0.0)

    def test_fidelity_symmetric(self):
        """Fidelity should be symmetric: F(ρ,σ) = F(σ,ρ)."""
        state1 = plus_state(1)
        state2 = computational_basis(1, 0)
        assert state1.fidelity(state2) == pytest.approx(state2.fidelity(state1))


class TestPartialTrace:
    """Test partial trace operations."""

    def test_partial_trace_bell_state(self):
        """Tracing out one qubit of Bell state gives maximally mixed."""
        bell = bell_state(0)  # (|00⟩ + |11⟩)/√2

        # Trace out qubit 1, keep qubit 0
        reduced = bell.partial_trace([0])

        # Should be maximally mixed
        assert reduced.purity == pytest.approx(0.5)

    def test_partial_trace_product_state(self):
        """Tracing out one qubit of product state gives pure state."""
        # |+⟩ ⊗ |0⟩
        state = plus_state(1).tensor(computational_basis(1, 0))

        # Trace out qubit 1, keep qubit 0
        reduced = state.partial_trace([0])

        # Should still be pure (it's |+⟩)
        assert reduced.purity == pytest.approx(1.0)


class TestTensorProduct:
    """Test tensor product operations."""

    def test_tensor_product_dimension(self):
        """Tensor product should have correct dimension."""
        state1 = QuantumState(1)
        state2 = QuantumState(2)
        combined = state1.tensor(state2)

        assert combined.n_qubits == 3
        assert combined.dim == 8

    def test_tensor_product_computational_basis(self):
        """Tensor product of basis states should give combined basis state."""
        state0 = computational_basis(1, 0)  # |0⟩
        state1 = computational_basis(1, 1)  # |1⟩

        combined = state0.tensor(state1)  # |01⟩
        assert combined.probability(1) == pytest.approx(1.0)  # |01⟩ = index 1


class TestBlochVector:
    """Test Bloch vector representation."""

    def test_zero_state_bloch(self):
        """|0⟩ should be at north pole (0, 0, 1)."""
        state = computational_basis(1, 0)
        bloch = state.bloch_vector()
        assert np.allclose(bloch, [0, 0, 1])

    def test_one_state_bloch(self):
        """|1⟩ should be at south pole (0, 0, -1)."""
        state = computational_basis(1, 1)
        bloch = state.bloch_vector()
        assert np.allclose(bloch, [0, 0, -1])

    def test_plus_state_bloch(self):
        """|+⟩ should be at (1, 0, 0)."""
        state = plus_state(1)
        bloch = state.bloch_vector()
        assert np.allclose(bloch, [1, 0, 0])

    def test_mixed_state_bloch_inside_sphere(self):
        """Mixed states should have |r| < 1."""
        state = maximally_mixed(1)
        bloch = state.bloch_vector()
        r = np.linalg.norm(bloch)
        assert r < 1.0


class TestStandardStateConstructors:
    """Test standard state construction functions."""

    def test_computational_basis_states(self):
        """Computational basis states should have correct probabilities."""
        for n in [1, 2, 3]:
            for idx in range(2**n):
                state = computational_basis(n, idx)
                assert state.probability(idx) == pytest.approx(1.0)

    def test_bell_states_entangled(self):
        """Bell states should have maximally mixed reduced states."""
        for which in range(4):
            bell = bell_state(which)
            reduced = bell.partial_trace([0])
            assert reduced.purity == pytest.approx(0.5)

    def test_ghz_state_structure(self):
        """GHZ state should be (|00...0⟩ + |11...1⟩)/√2."""
        for n in [2, 3, 4]:
            ghz = ghz_state(n)
            # Should only have non-zero amplitude at |0...0⟩ and |1...1⟩
            assert ghz.probability(0) == pytest.approx(0.5)
            assert ghz.probability(2**n - 1) == pytest.approx(0.5)

    def test_w_state_structure(self):
        """W state should have n equally weighted terms."""
        n = 3
        w = w_state(n)

        # Should have probability 1/n at each single-excitation state
        expected_prob = 1/n
        for i in range(n):
            idx = 1 << (n - 1 - i)
            assert w.probability(idx) == pytest.approx(expected_prob)

    def test_random_state_pure(self):
        """Random state should be pure and normalized."""
        state = random_state(2, seed=42)
        assert state.state_type == StateType.PURE
        assert state.purity == pytest.approx(1.0)

    def test_random_state_reproducible(self):
        """Random state with same seed should be reproducible."""
        state1 = random_state(2, seed=123)
        state2 = random_state(2, seed=123)
        assert np.allclose(state1.amplitudes, state2.amplitudes)


class TestExpectationValue:
    """Test expectation value calculations."""

    def test_pauli_z_expectation(self):
        """⟨0|Z|0⟩ = 1, ⟨1|Z|1⟩ = -1."""
        pauli_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        state0 = computational_basis(1, 0)
        state1 = computational_basis(1, 1)

        assert state0.expectation(pauli_z) == pytest.approx(1.0)
        assert state1.expectation(pauli_z) == pytest.approx(-1.0)

    def test_pauli_x_expectation_plus_state(self):
        """⟨+|X|+⟩ = 1."""
        pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        state = plus_state(1)
        assert state.expectation(pauli_x) == pytest.approx(1.0)
