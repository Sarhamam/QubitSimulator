"""
Lindblad Master Equation for Open Quantum Systems

Implements the Lindblad (GKSL) master equation for simulating decoherence
and dissipation in quantum systems. This module enables the geodesic
deviation model where noise causes trajectories to deviate from ideal
unitary evolution geodesics.

Mathematical Foundation:
    dρ/dt = -i[H,ρ] + Σₖ (LₖρLₖ† - ½{Lₖ†Lₖ, ρ})

The first term is unitary (Hamiltonian) evolution.
The second term (dissipator) describes irreversible processes.

Geometric Interpretation:
    - Unitary evolution: geodesic motion on state manifold
    - Lindblad evolution: geodesic deviation due to "curvature" from noise
    - Steady states: fixed points of the flow
"""

import numpy as np
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass

from state import QuantumState, StateType


@dataclass
class LindbladOperator:
    """
    A single Lindblad (jump) operator with associated rate.

    The operator L_k appears in the dissipator as:
        D[L](ρ) = γ(LρL† - ½{L†L, ρ})

    Attributes:
        operator: The jump operator matrix
        rate: Decay rate γ (must be non-negative)
        name: Human-readable description
    """
    operator: np.ndarray
    rate: float = 1.0
    name: str = "L"

    def __post_init__(self):
        if self.rate < 0:
            raise ValueError(f"Lindblad rate must be non-negative, got {self.rate}")
        self.operator = np.asarray(self.operator, dtype=np.complex128)


def lindblad_dissipator(rho: np.ndarray, L: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Compute the Lindblad dissipator for a single jump operator.

    D[L](ρ) = γ(LρL† - ½{L†L, ρ})

    Args:
        rho: Density matrix
        L: Jump operator
        gamma: Decay rate

    Returns:
        Dissipator contribution to dρ/dt
    """
    L_dag = L.conj().T
    L_dag_L = L_dag @ L

    # LρL†
    sandwich = L @ rho @ L_dag

    # ½{L†L, ρ} = ½(L†Lρ + ρL†L)
    anticommutator = 0.5 * (L_dag_L @ rho + rho @ L_dag_L)

    return gamma * (sandwich - anticommutator)


def lindblad_rhs(rho: np.ndarray,
                 hamiltonian: np.ndarray,
                 lindblad_ops: List[LindbladOperator]) -> np.ndarray:
    """
    Compute the right-hand side of the Lindblad master equation.

    dρ/dt = -i[H,ρ] + Σₖ D[Lₖ](ρ)

    Args:
        rho: Current density matrix
        hamiltonian: System Hamiltonian
        lindblad_ops: List of Lindblad operators with rates

    Returns:
        Time derivative dρ/dt
    """
    # Unitary part: -i[H, ρ]
    commutator = hamiltonian @ rho - rho @ hamiltonian
    drho_dt = -1j * commutator

    # Dissipative part: Σₖ D[Lₖ](ρ)
    for lop in lindblad_ops:
        drho_dt += lindblad_dissipator(rho, lop.operator, lop.rate)

    return drho_dt


def evolve_lindblad(state: QuantumState,
                    hamiltonian: np.ndarray,
                    lindblad_ops: List[LindbladOperator],
                    dt: float,
                    steps: int,
                    method: str = 'rk4') -> List[QuantumState]:
    """
    Evolve a quantum state under Lindblad dynamics.

    Args:
        state: Initial quantum state
        hamiltonian: System Hamiltonian
        lindblad_ops: List of Lindblad operators
        dt: Time step
        steps: Number of time steps
        method: Integration method ('euler' or 'rk4')

    Returns:
        List of states at each time step (including initial)
    """
    rho = state.rho.copy()
    trajectory = [QuantumState(state.n_qubits, density_matrix=rho.copy())]

    for _ in range(steps):
        if method == 'euler':
            rho = _euler_step(rho, hamiltonian, lindblad_ops, dt)
        elif method == 'rk4':
            rho = _rk4_step(rho, hamiltonian, lindblad_ops, dt)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Ensure trace normalization (numerical stability)
        rho = rho / np.trace(rho)

        trajectory.append(QuantumState(state.n_qubits, density_matrix=rho.copy()))

    return trajectory


def _euler_step(rho: np.ndarray,
                hamiltonian: np.ndarray,
                lindblad_ops: List[LindbladOperator],
                dt: float) -> np.ndarray:
    """Single Euler integration step."""
    drho = lindblad_rhs(rho, hamiltonian, lindblad_ops)
    return rho + dt * drho


def _rk4_step(rho: np.ndarray,
              hamiltonian: np.ndarray,
              lindblad_ops: List[LindbladOperator],
              dt: float) -> np.ndarray:
    """Single 4th-order Runge-Kutta step."""
    k1 = lindblad_rhs(rho, hamiltonian, lindblad_ops)
    k2 = lindblad_rhs(rho + 0.5 * dt * k1, hamiltonian, lindblad_ops)
    k3 = lindblad_rhs(rho + 0.5 * dt * k2, hamiltonian, lindblad_ops)
    k4 = lindblad_rhs(rho + dt * k3, hamiltonian, lindblad_ops)

    return rho + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


def evolve_unitary(state: QuantumState,
                   hamiltonian: np.ndarray,
                   dt: float,
                   steps: int) -> List[QuantumState]:
    """
    Evolve a quantum state under pure unitary (Hamiltonian) dynamics.

    This is the ideal geodesic evolution without decoherence.
    U(t) = exp(-iHt)

    Args:
        state: Initial quantum state
        hamiltonian: System Hamiltonian
        dt: Time step
        steps: Number of time steps

    Returns:
        List of states at each time step
    """
    return evolve_lindblad(state, hamiltonian, [], dt, steps)


def geodesic_deviation(ideal_trajectory: List[QuantumState],
                       noisy_trajectory: List[QuantumState]) -> np.ndarray:
    """
    Compute geodesic deviation between ideal and noisy trajectories.

    Measures how much the noisy evolution deviates from the ideal
    unitary geodesic at each time step.

    Args:
        ideal_trajectory: States from unitary evolution
        noisy_trajectory: States from Lindblad evolution

    Returns:
        Array of fidelities F(ρ_ideal(t), ρ_noisy(t)) at each time
    """
    deviations = []
    for ideal, noisy in zip(ideal_trajectory, noisy_trajectory):
        deviations.append(ideal.fidelity(noisy))
    return np.array(deviations)


def purity_decay(trajectory: List[QuantumState]) -> np.ndarray:
    """
    Track purity decay along a trajectory.

    Purity Tr(ρ²) measures how mixed the state is:
    - Pure states: purity = 1
    - Maximally mixed: purity = 1/d

    Args:
        trajectory: List of states over time

    Returns:
        Array of purities at each time step
    """
    return np.array([state.purity for state in trajectory])


# ============================================================================
# Standard Lindblad Operators (Noise Channels)
# ============================================================================

def amplitude_damping_ops(gamma: float, n_qubits: int = 1) -> List[LindbladOperator]:
    """
    Amplitude damping (T1 decay) operators.

    Models energy relaxation: |1⟩ → |0⟩ with rate γ.

    For single qubit: L = √γ |0⟩⟨1| = √γ σ₋

    Args:
        gamma: Decay rate (1/T1)
        n_qubits: Number of qubits (applies independently to each)

    Returns:
        List of Lindblad operators
    """
    # σ₋ = |0⟩⟨1|
    sigma_minus = np.array([[0, 1], [0, 0]], dtype=np.complex128)

    ops = []
    for i in range(n_qubits):
        # Build operator that acts on qubit i
        L = _single_qubit_op_to_n_qubit(sigma_minus, i, n_qubits)
        ops.append(LindbladOperator(L, gamma, f"σ₋[{i}]"))

    return ops


def phase_damping_ops(gamma: float, n_qubits: int = 1) -> List[LindbladOperator]:
    """
    Phase damping (pure dephasing, T2*) operators.

    Models loss of coherence without energy exchange.
    Causes off-diagonal elements of ρ to decay.

    For single qubit: L = √(γ/2) σz

    Args:
        gamma: Dephasing rate (related to 1/T2)
        n_qubits: Number of qubits

    Returns:
        List of Lindblad operators
    """
    sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    ops = []
    for i in range(n_qubits):
        L = _single_qubit_op_to_n_qubit(sigma_z, i, n_qubits)
        # Factor of sqrt(1/2) gives standard dephasing rate
        ops.append(LindbladOperator(L, gamma / 2, f"σz[{i}]"))

    return ops


def depolarizing_ops(p: float, n_qubits: int = 1) -> List[LindbladOperator]:
    """
    Depolarizing channel operators.

    Applies random Pauli errors with probability p.
    Maps ρ → (1-p)ρ + (p/3)(XρX + YρY + ZρZ)

    Args:
        p: Depolarizing probability per unit time
        n_qubits: Number of qubits

    Returns:
        List of Lindblad operators (3 per qubit: X, Y, Z)
    """
    sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    paulis = [('X', sigma_x), ('Y', sigma_y), ('Z', sigma_z)]
    rate = p / 3  # Each Pauli has rate p/3

    ops = []
    for i in range(n_qubits):
        for name, pauli in paulis:
            L = _single_qubit_op_to_n_qubit(pauli, i, n_qubits)
            ops.append(LindbladOperator(L, rate, f"σ{name}[{i}]"))

    return ops


def thermal_ops(gamma: float, n_bar: float, n_qubits: int = 1) -> List[LindbladOperator]:
    """
    Thermal bath operators.

    Models coupling to a thermal bath with mean occupation n_bar.
    Includes both emission (cooling) and absorption (heating).

    Args:
        gamma: Coupling strength
        n_bar: Mean thermal occupation number
        n_qubits: Number of qubits

    Returns:
        List of Lindblad operators (2 per qubit: emission and absorption)
    """
    sigma_minus = np.array([[0, 1], [0, 0]], dtype=np.complex128)
    sigma_plus = np.array([[0, 0], [1, 0]], dtype=np.complex128)

    ops = []
    for i in range(n_qubits):
        # Emission: √(γ(n_bar + 1)) σ₋
        L_emit = _single_qubit_op_to_n_qubit(sigma_minus, i, n_qubits)
        ops.append(LindbladOperator(L_emit, gamma * (n_bar + 1), f"emit[{i}]"))

        # Absorption: √(γ n_bar) σ₊
        if n_bar > 0:
            L_absorb = _single_qubit_op_to_n_qubit(sigma_plus, i, n_qubits)
            ops.append(LindbladOperator(L_absorb, gamma * n_bar, f"absorb[{i}]"))

    return ops


def _single_qubit_op_to_n_qubit(op: np.ndarray,
                                 target: int,
                                 n_qubits: int) -> np.ndarray:
    """
    Expand a single-qubit operator to act on a specific qubit in an n-qubit system.

    Args:
        op: 2x2 single-qubit operator
        target: Which qubit to act on (0-indexed)
        n_qubits: Total number of qubits

    Returns:
        2^n × 2^n operator
    """
    result = np.array([[1]], dtype=np.complex128)

    for i in range(n_qubits):
        if i == target:
            result = np.kron(result, op)
        else:
            result = np.kron(result, np.eye(2, dtype=np.complex128))

    return result


# ============================================================================
# Analysis Tools
# ============================================================================

def steady_state(hamiltonian: np.ndarray,
                 lindblad_ops: List[LindbladOperator],
                 tol: float = 1e-10,
                 max_iter: int = 10000,
                 dt: float = 0.01) -> QuantumState:
    """
    Find the steady state of Lindblad dynamics.

    The steady state satisfies dρ/dt = 0.

    Uses time evolution until convergence.

    Args:
        hamiltonian: System Hamiltonian
        lindblad_ops: Lindblad operators
        tol: Convergence tolerance
        max_iter: Maximum iterations
        dt: Time step

    Returns:
        Steady state density matrix
    """
    dim = hamiltonian.shape[0]
    n_qubits = int(np.log2(dim))

    # Start from maximally mixed state
    rho = np.eye(dim, dtype=np.complex128) / dim

    for _ in range(max_iter):
        rho_new = _rk4_step(rho, hamiltonian, lindblad_ops, dt)
        rho_new = rho_new / np.trace(rho_new)

        # Check convergence
        diff = np.linalg.norm(rho_new - rho)
        if diff < tol:
            break

        rho = rho_new

    return QuantumState(n_qubits, density_matrix=rho)


def decoherence_rate(hamiltonian: np.ndarray,
                     lindblad_ops: List[LindbladOperator],
                     initial_state: QuantumState,
                     dt: float = 0.01,
                     steps: int = 100) -> float:
    """
    Estimate the decoherence rate from purity decay.

    For weak decoherence: Tr(ρ²) ≈ 1 - Γt

    Args:
        hamiltonian: System Hamiltonian
        lindblad_ops: Lindblad operators
        initial_state: Starting state (should be pure)
        dt: Time step
        steps: Number of time steps

    Returns:
        Estimated decoherence rate Γ
    """
    trajectory = evolve_lindblad(initial_state, hamiltonian, lindblad_ops, dt, steps)
    purities = purity_decay(trajectory)
    times = np.arange(len(purities)) * dt

    # Linear fit: purity ≈ 1 - Γt for small times
    # Use first half of data where linear approximation is valid
    n_fit = len(times) // 2
    if n_fit < 2:
        n_fit = len(times)

    # Fit 1 - purity = Γt
    purity_loss = 1 - purities[:n_fit]
    times_fit = times[:n_fit]

    # Least squares: Γ = (t · (1-p)) / (t · t)
    if np.sum(times_fit**2) > 1e-12:
        gamma = np.sum(times_fit * purity_loss) / np.sum(times_fit**2)
    else:
        gamma = 0.0

    return max(0, gamma)
