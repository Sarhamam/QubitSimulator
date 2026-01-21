"""
Tests for quantum gates (gates.py)

Tests verify:
- Gate unitarity
- Pauli gate properties
- Rotation gates
- Two-qubit entangling gates
- Gate composition
"""

import numpy as np
import pytest
from gates import (
    Gate, GateType,
    I, X, Y, Z, H, S, Sdg, T, Tdg,
    Rx, Ry, Rz, P, U3,
    CX, CY, CZ_gate, SWAP_gate, iSWAP, SQRTSWAP,
    CRx, CRy, CRz, CP, XX, YY, ZZ,
    CCX, CCZ, CSWAP,
    apply_gate, tensor_gates, controlled,
    decompose_to_u3, get_gate,
    PAULI_I, PAULI_X, PAULI_Y, PAULI_Z
)


class TestGateUnitarity:
    """Test that all gates are unitary."""

    @pytest.mark.parametrize("gate_factory", [
        I, X, Y, Z, H, S, Sdg, T, Tdg,
        CX, CY, CZ_gate, SWAP_gate, iSWAP, SQRTSWAP,
        CCX, CCZ, CSWAP
    ])
    def test_fixed_gate_unitary(self, gate_factory):
        """Fixed gates should be unitary: U†U = I."""
        gate = gate_factory()
        identity = gate.matrix @ gate.matrix.conj().T
        expected = np.eye(gate.matrix.shape[0])
        assert np.allclose(identity, expected)

    @pytest.mark.parametrize("theta", [0, np.pi/4, np.pi/2, np.pi])
    def test_rotation_gates_unitary(self, theta):
        """Rotation gates should be unitary for all angles."""
        for gate in [Rx(theta), Ry(theta), Rz(theta), P(theta)]:
            identity = gate.matrix @ gate.matrix.conj().T
            expected = np.eye(2)
            assert np.allclose(identity, expected)


class TestPauliGates:
    """Test Pauli gate properties."""

    def test_pauli_x_is_not(self):
        """X gate should flip |0⟩ ↔ |1⟩."""
        x = X().matrix
        zero = np.array([1, 0], dtype=np.complex128)
        one = np.array([0, 1], dtype=np.complex128)

        assert np.allclose(x @ zero, one)
        assert np.allclose(x @ one, zero)

    def test_pauli_z_phases(self):
        """Z gate should give +1 phase to |0⟩, -1 to |1⟩."""
        z = Z().matrix
        zero = np.array([1, 0], dtype=np.complex128)
        one = np.array([0, 1], dtype=np.complex128)

        assert np.allclose(z @ zero, zero)
        assert np.allclose(z @ one, -one)

    def test_pauli_squared_is_identity(self):
        """X², Y², Z² = I."""
        for gate_factory in [X, Y, Z]:
            gate = gate_factory()
            squared = gate.matrix @ gate.matrix
            assert np.allclose(squared, np.eye(2))

    def test_pauli_anticommutation(self):
        """Pauli matrices should anticommute: {X,Y} = {Y,Z} = {Z,X} = 0."""
        x, y, z = PAULI_X, PAULI_Y, PAULI_Z

        assert np.allclose(x @ y + y @ x, np.zeros((2, 2)))
        assert np.allclose(y @ z + z @ y, np.zeros((2, 2)))
        assert np.allclose(z @ x + x @ z, np.zeros((2, 2)))

    def test_pauli_product_relations(self):
        """XY = iZ, YZ = iX, ZX = iY."""
        x, y, z = PAULI_X, PAULI_Y, PAULI_Z

        assert np.allclose(x @ y, 1j * z)
        assert np.allclose(y @ z, 1j * x)
        assert np.allclose(z @ x, 1j * y)


class TestHadamardGate:
    """Test Hadamard gate properties."""

    def test_hadamard_creates_superposition(self):
        """H|0⟩ = |+⟩ = (|0⟩ + |1⟩)/√2."""
        h = H().matrix
        zero = np.array([1, 0], dtype=np.complex128)

        result = h @ zero
        expected = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)

        assert np.allclose(result, expected)

    def test_hadamard_squared_is_identity(self):
        """H² = I."""
        h = H().matrix
        assert np.allclose(h @ h, np.eye(2))

    def test_hadamard_transforms_paulis(self):
        """HXH = Z, HZH = X."""
        h = H().matrix

        assert np.allclose(h @ PAULI_X @ h, PAULI_Z)
        assert np.allclose(h @ PAULI_Z @ h, PAULI_X)


class TestPhaseGates:
    """Test phase gates S and T."""

    def test_s_is_sqrt_z(self):
        """S² = Z."""
        s = S().matrix
        z = Z().matrix
        assert np.allclose(s @ s, z)

    def test_t_is_sqrt_s(self):
        """T² = S."""
        t = T().matrix
        s = S().matrix
        assert np.allclose(t @ t, s)

    def test_s_dagger_is_inverse(self):
        """S†S = I."""
        s = S().matrix
        s_dag = Sdg().matrix
        assert np.allclose(s_dag @ s, np.eye(2))

    def test_t_dagger_is_inverse(self):
        """T†T = I."""
        t = T().matrix
        t_dag = Tdg().matrix
        assert np.allclose(t_dag @ t, np.eye(2))


class TestRotationGates:
    """Test parametric rotation gates."""

    def test_rx_at_pi_is_x(self):
        """Rx(π) = -iX."""
        rx_pi = Rx(np.pi).matrix
        expected = -1j * PAULI_X
        assert np.allclose(rx_pi, expected)

    def test_ry_at_pi_is_y(self):
        """Ry(π) = -iY."""
        ry_pi = Ry(np.pi).matrix
        expected = -1j * PAULI_Y
        assert np.allclose(ry_pi, expected)

    def test_rz_at_pi_is_z(self):
        """Rz(π) = -iZ."""
        rz_pi = Rz(np.pi).matrix
        expected = -1j * PAULI_Z
        assert np.allclose(rz_pi, expected)

    def test_rotation_at_zero_is_identity(self):
        """R(0) = I for all rotation gates."""
        for gate in [Rx(0), Ry(0), Rz(0)]:
            assert np.allclose(gate.matrix, np.eye(2))

    def test_rotation_composition(self):
        """Rx(a)Rx(b) = Rx(a+b)."""
        a, b = np.pi/3, np.pi/4
        composed = Rx(a).matrix @ Rx(b).matrix
        direct = Rx(a + b).matrix
        assert np.allclose(composed, direct)


class TestU3Gate:
    """Test general single-qubit U3 gate."""

    def test_u3_special_cases(self):
        """U3 should reduce to known gates at special parameters."""
        # U3(0, 0, 0) = I
        assert np.allclose(U3(0, 0, 0).matrix, np.eye(2))

        # U3(π, 0, π) = X
        u3_x = U3(np.pi, 0, np.pi).matrix
        assert np.allclose(u3_x, PAULI_X)

    def test_u3_decomposition_extracts_parameters(self):
        """decompose_to_u3 should extract valid Euler angle parameters."""
        # Test that decomposition returns valid parameters
        for gate in [H(), X(), Y(), Z()]:
            params = decompose_to_u3(gate)

            # Should return all required keys
            assert 'theta' in params
            assert 'phi' in params
            assert 'lambda' in params
            assert 'global_phase' in params

            # Theta should be in valid range
            assert 0 <= params['theta'] <= np.pi + 1e-10

    def test_u3_decomposition_returns_none_for_multi_qubit(self):
        """decompose_to_u3 should return None for multi-qubit gates."""
        params = decompose_to_u3(CX())
        assert params is None


class TestTwoQubitGates:
    """Test two-qubit gates."""

    def test_cnot_creates_entanglement(self):
        """CNOT on |+0⟩ creates Bell state."""
        # |+⟩ ⊗ |0⟩
        plus_zero = np.array([1, 0, 1, 0], dtype=np.complex128) / np.sqrt(2)

        cnot = CX().matrix
        result = cnot @ plus_zero

        # Should be (|00⟩ + |11⟩)/√2
        expected = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
        assert np.allclose(result, expected)

    def test_cnot_controlled_action(self):
        """CNOT flips target only when control is |1⟩."""
        cnot = CX().matrix

        # |00⟩ → |00⟩
        state_00 = np.array([1, 0, 0, 0], dtype=np.complex128)
        assert np.allclose(cnot @ state_00, state_00)

        # |10⟩ → |11⟩
        state_10 = np.array([0, 0, 1, 0], dtype=np.complex128)
        state_11 = np.array([0, 0, 0, 1], dtype=np.complex128)
        assert np.allclose(cnot @ state_10, state_11)

    def test_swap_exchanges_qubits(self):
        """SWAP should exchange |01⟩ ↔ |10⟩."""
        swap = SWAP_gate().matrix

        state_01 = np.array([0, 1, 0, 0], dtype=np.complex128)
        state_10 = np.array([0, 0, 1, 0], dtype=np.complex128)

        assert np.allclose(swap @ state_01, state_10)
        assert np.allclose(swap @ state_10, state_01)

    def test_cz_symmetric(self):
        """CZ should be symmetric in both qubits."""
        cz = CZ_gate().matrix
        assert np.allclose(cz, cz.T)

    def test_sqrt_swap_squared_is_swap(self):
        """(√SWAP)² = SWAP."""
        sqrt_swap = SQRTSWAP().matrix
        swap = SWAP_gate().matrix
        assert np.allclose(sqrt_swap @ sqrt_swap, swap)


class TestIsingGates:
    """Test Ising interaction gates XX, YY, ZZ."""

    def test_xx_at_zero_is_identity(self):
        """XX(0) = I."""
        assert np.allclose(XX(0).matrix, np.eye(4))

    def test_yy_at_zero_is_identity(self):
        """YY(0) = I."""
        assert np.allclose(YY(0).matrix, np.eye(4))

    def test_zz_at_zero_is_identity(self):
        """ZZ(0) = I."""
        assert np.allclose(ZZ(0).matrix, np.eye(4))

    def test_zz_diagonal(self):
        """ZZ gate should be diagonal."""
        zz = ZZ(np.pi/4).matrix
        # Check only diagonal elements are non-zero
        off_diag = zz - np.diag(np.diag(zz))
        assert np.allclose(off_diag, 0)


class TestMultiQubitGates:
    """Test 3-qubit gates."""

    def test_toffoli_only_flips_when_both_controls_one(self):
        """CCX only flips target when both controls are |1⟩."""
        toffoli = CCX().matrix

        # |110⟩ → |111⟩
        state_110 = np.zeros(8, dtype=np.complex128)
        state_110[6] = 1
        state_111 = np.zeros(8, dtype=np.complex128)
        state_111[7] = 1

        assert np.allclose(toffoli @ state_110, state_111)

        # |100⟩ → |100⟩ (unchanged)
        state_100 = np.zeros(8, dtype=np.complex128)
        state_100[4] = 1
        assert np.allclose(toffoli @ state_100, state_100)

    def test_ccz_only_phases_111(self):
        """CCZ only adds phase to |111⟩."""
        ccz = CCZ().matrix

        # Should be diagonal
        assert np.allclose(ccz - np.diag(np.diag(ccz)), 0)

        # |111⟩ gets -1 phase
        assert ccz[7, 7] == pytest.approx(-1)

        # All others get +1
        for i in range(7):
            assert ccz[i, i] == pytest.approx(1)

    def test_fredkin_swaps_when_control_one(self):
        """CSWAP swaps last two qubits only when control is |1⟩."""
        fredkin = CSWAP().matrix

        # |101⟩ → |110⟩
        state_101 = np.zeros(8, dtype=np.complex128)
        state_101[5] = 1
        state_110 = np.zeros(8, dtype=np.complex128)
        state_110[6] = 1

        assert np.allclose(fredkin @ state_101, state_110)


class TestGateApplication:
    """Test apply_gate utility."""

    def test_apply_single_qubit_gate(self):
        """Apply X to first qubit of |00⟩ should give |10⟩."""
        amplitudes = np.array([1, 0, 0, 0], dtype=np.complex128)
        result = apply_gate(X(), amplitudes, [0], 2)

        expected = np.array([0, 0, 1, 0], dtype=np.complex128)
        assert np.allclose(result, expected)

    def test_apply_gate_to_second_qubit(self):
        """Apply X to second qubit of |00⟩ should give |01⟩."""
        amplitudes = np.array([1, 0, 0, 0], dtype=np.complex128)
        result = apply_gate(X(), amplitudes, [1], 2)

        expected = np.array([0, 1, 0, 0], dtype=np.complex128)
        assert np.allclose(result, expected)


class TestGateComposition:
    """Test gate composition utilities."""

    def test_tensor_gates_dimension(self):
        """Tensor product should have correct dimension."""
        combined = tensor_gates(H(), X())
        assert combined.n_qubits == 2
        assert combined.matrix.shape == (4, 4)

    def test_controlled_gate(self):
        """controlled(X) should be CNOT."""
        cx_from_controlled = controlled(X(), 1)
        cx_direct = CX()
        assert np.allclose(cx_from_controlled.matrix, cx_direct.matrix)


class TestGateLibrary:
    """Test gate library access."""

    def test_get_gate_by_name(self):
        """Should retrieve gates by name."""
        assert np.allclose(get_gate('X').matrix, X().matrix)
        assert np.allclose(get_gate('H').matrix, H().matrix)
        assert np.allclose(get_gate('CNOT').matrix, CX().matrix)

    def test_get_parametric_gate(self):
        """Should retrieve parametric gates with parameters."""
        rx = get_gate('RX', np.pi/2)
        assert rx.params['theta'] == np.pi/2

    def test_invalid_gate_raises(self):
        """Unknown gate name should raise ValueError."""
        with pytest.raises(ValueError):
            get_gate('UNKNOWN_GATE')


class TestGateAdjoint:
    """Test gate adjoint operation."""

    def test_adjoint_is_inverse(self):
        """G†G = I for all gates."""
        for gate in [X(), Y(), Z(), H(), T(), S(), CX()]:
            product = gate.adjoint.matrix @ gate.matrix
            expected = np.eye(gate.matrix.shape[0])
            assert np.allclose(product, expected)

    def test_adjoint_name(self):
        """Adjoint should have dagger in name."""
        gate = T()
        assert '†' in gate.adjoint.name
