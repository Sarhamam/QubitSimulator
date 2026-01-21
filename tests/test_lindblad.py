"""
Tests for Lindblad master equation (lindblad.py)

Tests verify:
- Dissipator structure
- Trace preservation
- Positivity preservation
- Known analytical solutions
- Standard noise channels
"""

import numpy as np
import pytest
from state import QuantumState, computational_basis, plus_state, maximally_mixed
from gates import PAULI_X, PAULI_Y, PAULI_Z
from lindblad import (
    LindbladOperator,
    lindblad_dissipator,
    lindblad_rhs,
    evolve_lindblad,
    evolve_unitary,
    geodesic_deviation,
    purity_decay,
    amplitude_damping_ops,
    phase_damping_ops,
    depolarizing_ops,
    thermal_ops,
    steady_state,
    decoherence_rate
)


class TestLindbladOperator:
    """Test LindbladOperator dataclass."""

    def test_create_operator(self):
        """Should create operator with rate."""
        L = np.array([[0, 1], [0, 0]], dtype=np.complex128)
        op = LindbladOperator(L, rate=0.5, name="test")

        assert op.rate == 0.5
        assert op.name == "test"
        assert np.allclose(op.operator, L)

    def test_negative_rate_raises(self):
        """Negative rate should raise error."""
        L = np.array([[0, 1], [0, 0]], dtype=np.complex128)
        with pytest.raises(ValueError):
            LindbladOperator(L, rate=-0.1)


class TestDissipator:
    """Test Lindblad dissipator computation."""

    def test_dissipator_zero_for_zero_operator(self):
        """Zero operator should give zero dissipator."""
        rho = plus_state(1).rho
        L = np.zeros((2, 2), dtype=np.complex128)

        D = lindblad_dissipator(rho, L, gamma=1.0)

        assert np.allclose(D, 0)

    def test_dissipator_trace_zero(self):
        """Dissipator should be traceless (trace-preserving evolution)."""
        rho = plus_state(1).rho
        L = np.array([[0, 1], [0, 0]], dtype=np.complex128)  # σ₋

        D = lindblad_dissipator(rho, L, gamma=1.0)

        assert np.trace(D) == pytest.approx(0.0, abs=1e-12)

    def test_dissipator_hermitian(self):
        """Dissipator applied to Hermitian matrix should be Hermitian."""
        rho = plus_state(1).rho
        L = np.array([[0, 1], [0, 0]], dtype=np.complex128)

        D = lindblad_dissipator(rho, L, gamma=1.0)

        assert np.allclose(D, D.conj().T)


class TestLindbladRHS:
    """Test the full Lindblad right-hand side."""

    def test_pure_unitary_evolution(self):
        """With no Lindblad ops, should be pure unitary evolution."""
        rho = computational_basis(1, 0).rho
        H = PAULI_X  # Rotates around X axis

        drho = lindblad_rhs(rho, H, [])

        # Should equal -i[H, ρ]
        expected = -1j * (H @ rho - rho @ H)
        assert np.allclose(drho, expected)

    def test_rhs_traceless(self):
        """RHS should always be traceless."""
        rho = plus_state(1).rho
        H = PAULI_Z
        ops = amplitude_damping_ops(0.1, n_qubits=1)

        drho = lindblad_rhs(rho, H, ops)

        assert np.trace(drho) == pytest.approx(0.0, abs=1e-12)


class TestEvolution:
    """Test time evolution under Lindblad dynamics."""

    def test_trace_preserved(self):
        """Trace should remain 1 throughout evolution."""
        state = plus_state(1)
        H = PAULI_Z
        ops = amplitude_damping_ops(0.1, n_qubits=1)

        trajectory = evolve_lindblad(state, H, ops, dt=0.01, steps=100)

        for s in trajectory:
            assert np.trace(s.rho) == pytest.approx(1.0, abs=1e-10)

    def test_positivity_preserved(self):
        """Density matrix should remain positive semidefinite."""
        state = plus_state(1)
        H = PAULI_Z
        ops = amplitude_damping_ops(0.1, n_qubits=1)

        trajectory = evolve_lindblad(state, H, ops, dt=0.01, steps=100)

        for s in trajectory:
            eigenvalues = np.linalg.eigvalsh(s.rho)
            assert np.all(eigenvalues >= -1e-10)

    def test_hermiticity_preserved(self):
        """Density matrix should remain Hermitian."""
        state = plus_state(1)
        H = PAULI_Z
        ops = amplitude_damping_ops(0.1, n_qubits=1)

        trajectory = evolve_lindblad(state, H, ops, dt=0.01, steps=100)

        for s in trajectory:
            assert np.allclose(s.rho, s.rho.conj().T)

    def test_purity_decreases_with_noise(self):
        """Purity should decrease under decoherence."""
        state = plus_state(1)  # Pure state, purity = 1
        H = np.zeros((2, 2))  # No Hamiltonian
        ops = depolarizing_ops(0.1, n_qubits=1)

        trajectory = evolve_lindblad(state, H, ops, dt=0.01, steps=100)

        initial_purity = trajectory[0].purity
        final_purity = trajectory[-1].purity

        assert initial_purity == pytest.approx(1.0)
        assert final_purity < initial_purity

    def test_unitary_preserves_purity(self):
        """Pure unitary evolution should preserve purity."""
        state = plus_state(1)
        H = PAULI_Z

        trajectory = evolve_unitary(state, H, dt=0.01, steps=100)

        for s in trajectory:
            assert s.purity == pytest.approx(1.0, abs=1e-6)

    def test_euler_vs_rk4_consistency(self):
        """Euler and RK4 should give similar results for small dt."""
        state = plus_state(1)
        H = PAULI_Z
        ops = amplitude_damping_ops(0.1, n_qubits=1)

        traj_euler = evolve_lindblad(state, H, ops, dt=0.001, steps=100, method='euler')
        traj_rk4 = evolve_lindblad(state, H, ops, dt=0.001, steps=100, method='rk4')

        # Final states should be similar
        assert traj_euler[-1].fidelity(traj_rk4[-1]) > 0.99


class TestAmplitudeDamping:
    """Test amplitude damping (T1 decay)."""

    def test_excited_state_decays_to_ground(self):
        """Amplitude damping should drive |1⟩ → |0⟩."""
        state = computational_basis(1, 1)  # |1⟩
        H = np.zeros((2, 2))
        ops = amplitude_damping_ops(gamma=0.5, n_qubits=1)

        trajectory = evolve_lindblad(state, H, ops, dt=0.1, steps=100)

        # Should end up close to |0⟩
        ground_state = computational_basis(1, 0)
        assert trajectory[-1].fidelity(ground_state) > 0.9

    def test_ground_state_stable(self):
        """Ground state should be stable under amplitude damping."""
        state = computational_basis(1, 0)  # |0⟩
        H = np.zeros((2, 2))
        ops = amplitude_damping_ops(gamma=0.5, n_qubits=1)

        trajectory = evolve_lindblad(state, H, ops, dt=0.1, steps=100)

        # Should remain |0⟩
        ground_state = computational_basis(1, 0)
        assert trajectory[-1].fidelity(ground_state) > 0.99


class TestPhaseDamping:
    """Test phase damping (T2 dephasing)."""

    def test_coherence_decays(self):
        """Phase damping should decay off-diagonal elements."""
        state = plus_state(1)  # Has off-diagonal coherence
        H = np.zeros((2, 2))
        ops = phase_damping_ops(gamma=0.5, n_qubits=1)

        trajectory = evolve_lindblad(state, H, ops, dt=0.1, steps=100)

        # Off-diagonal elements should decay
        initial_coherence = np.abs(trajectory[0].rho[0, 1])
        final_coherence = np.abs(trajectory[-1].rho[0, 1])

        assert final_coherence < initial_coherence

    def test_populations_unchanged(self):
        """Phase damping should not change diagonal elements."""
        state = plus_state(1)  # 50/50 populations
        H = np.zeros((2, 2))
        ops = phase_damping_ops(gamma=0.5, n_qubits=1)

        trajectory = evolve_lindblad(state, H, ops, dt=0.1, steps=100)

        # Populations should remain ~0.5
        assert trajectory[-1].rho[0, 0] == pytest.approx(0.5, abs=0.01)
        assert trajectory[-1].rho[1, 1] == pytest.approx(0.5, abs=0.01)


class TestDepolarizing:
    """Test depolarizing channel."""

    def test_approaches_maximally_mixed(self):
        """Depolarizing channel should drive any state toward maximally mixed."""
        state = computational_basis(1, 0)
        H = np.zeros((2, 2))
        ops = depolarizing_ops(p=0.3, n_qubits=1)

        trajectory = evolve_lindblad(state, H, ops, dt=0.1, steps=200)

        # Should approach maximally mixed (purity → 0.5)
        assert trajectory[-1].purity < 0.7


class TestMultiQubit:
    """Test multi-qubit Lindblad evolution."""

    def test_two_qubit_trace_preserved(self):
        """Two-qubit evolution should preserve trace."""
        from state import bell_state
        state = bell_state(0)
        H = np.kron(PAULI_Z, np.eye(2))  # Z on first qubit
        ops = amplitude_damping_ops(0.1, n_qubits=2)

        trajectory = evolve_lindblad(state, H, ops, dt=0.01, steps=50)

        for s in trajectory:
            assert np.trace(s.rho) == pytest.approx(1.0, abs=1e-10)

    def test_entanglement_decays(self):
        """Noise should cause entanglement decay in Bell states."""
        from state import bell_state
        from information_geometry import concurrence

        state = bell_state(0)  # Maximally entangled
        H = np.zeros((4, 4))
        ops = amplitude_damping_ops(0.1, n_qubits=2)

        trajectory = evolve_lindblad(state, H, ops, dt=0.1, steps=50)

        initial_concurrence = concurrence(trajectory[0])
        final_concurrence = concurrence(trajectory[-1])

        assert initial_concurrence == pytest.approx(1.0)
        assert final_concurrence < initial_concurrence


class TestGeodesicDeviation:
    """Test geodesic deviation computation."""

    def test_no_deviation_without_noise(self):
        """Without noise, fidelity should remain 1."""
        state = plus_state(1)
        H = PAULI_Z

        ideal = evolve_unitary(state, H, dt=0.01, steps=50)
        noisy = evolve_unitary(state, H, dt=0.01, steps=50)

        deviation = geodesic_deviation(ideal, noisy)

        assert np.all(deviation > 0.999)

    def test_deviation_increases_with_noise(self):
        """With noise, deviation should grow over time."""
        state = plus_state(1)
        H = PAULI_Z
        ops = amplitude_damping_ops(0.1, n_qubits=1)

        ideal = evolve_unitary(state, H, dt=0.01, steps=50)
        noisy = evolve_lindblad(state, H, ops, dt=0.01, steps=50)

        deviation = geodesic_deviation(ideal, noisy)

        # Initial fidelity should be 1, final should be less
        assert deviation[0] == pytest.approx(1.0)
        assert deviation[-1] < deviation[0]


class TestPurityDecay:
    """Test purity decay tracking."""

    def test_pure_state_starts_at_one(self):
        """Pure state should have initial purity 1."""
        state = plus_state(1)
        H = PAULI_Z
        ops = amplitude_damping_ops(0.1, n_qubits=1)

        trajectory = evolve_lindblad(state, H, ops, dt=0.01, steps=50)
        purities = purity_decay(trajectory)

        assert purities[0] == pytest.approx(1.0)

    def test_purity_monotonic_decrease(self):
        """Purity should generally decrease under strong noise."""
        state = plus_state(1)
        H = np.zeros((2, 2))
        ops = depolarizing_ops(0.5, n_qubits=1)

        trajectory = evolve_lindblad(state, H, ops, dt=0.01, steps=50)
        purities = purity_decay(trajectory)

        # Should trend downward (allow some fluctuation)
        assert purities[-1] < purities[0]


class TestSteadyState:
    """Test steady state finding."""

    def test_amplitude_damping_steady_state(self):
        """Amplitude damping steady state should be |0⟩."""
        H = np.zeros((2, 2))
        ops = amplitude_damping_ops(0.5, n_qubits=1)

        ss = steady_state(H, ops, tol=1e-8, dt=0.1)

        ground = computational_basis(1, 0)
        assert ss.fidelity(ground) > 0.99

    def test_depolarizing_steady_state(self):
        """Depolarizing steady state should be maximally mixed."""
        H = np.zeros((2, 2))
        ops = depolarizing_ops(0.5, n_qubits=1)

        ss = steady_state(H, ops, tol=1e-8, dt=0.1)

        mixed = maximally_mixed(1)
        assert ss.fidelity(mixed) > 0.99


class TestDecoherenceRate:
    """Test decoherence rate estimation."""

    def test_zero_noise_zero_rate(self):
        """No noise should give zero decoherence rate."""
        state = plus_state(1)
        H = PAULI_Z

        rate = decoherence_rate(H, [], state, dt=0.01, steps=100)

        assert rate == pytest.approx(0.0, abs=1e-6)

    def test_positive_rate_with_noise(self):
        """Noise should give positive decoherence rate."""
        state = plus_state(1)
        H = np.zeros((2, 2))
        ops = depolarizing_ops(0.1, n_qubits=1)

        rate = decoherence_rate(H, ops, state, dt=0.01, steps=100)

        assert rate > 0
