"""
Natural Gradient Optimization for Variational Quantum Algorithms

Implements natural gradient descent using the Quantum Fisher Information
matrix as a metric tensor. This follows the geometry of the quantum state
manifold rather than Euclidean parameter space.

Mathematical Foundation:
    Standard gradient:  θ' = θ - η ∇E
    Natural gradient:   θ' = θ - η F⁻¹ ∇E

where F is the Quantum Fisher Information matrix (Fubini-Study metric).

Geometric Interpretation:
    The natural gradient follows the steepest descent direction on the
    quantum state manifold, accounting for the curvature induced by the
    parameterization. This leads to:
    - Faster convergence
    - Parameterization invariance
    - Better optimization landscapes

Connection to NoeticEidos:
    The QFI is 4× the classical Fisher-Rao metric for pure state evolution.
    Natural gradient uses the same geometric structure as dual transport
    on statistical manifolds.
"""

import numpy as np
from typing import Callable, List, Tuple, Optional
from dataclasses import dataclass

from state import QuantumState
from information_geometry import compute_qgt, QuantumGeometricTensor


@dataclass
class OptimizationResult:
    """
    Result of natural gradient optimization.

    Attributes:
        params: Final optimized parameters
        energies: Energy at each iteration
        param_history: Parameters at each iteration
        converged: Whether optimization converged
        iterations: Number of iterations performed
        final_qfi: QFI matrix at final parameters
    """
    params: np.ndarray
    energies: List[float]
    param_history: List[np.ndarray]
    converged: bool
    iterations: int
    final_qfi: Optional[np.ndarray] = None


def quantum_fisher_information_matrix(
    state_fn: Callable[[np.ndarray], QuantumState],
    params: np.ndarray,
    epsilon: float = 1e-5
) -> np.ndarray:
    """
    Compute the Quantum Fisher Information matrix.

    F_μν = 4 Re(⟨∂_μψ|∂_νψ⟩ - ⟨∂_μψ|ψ⟩⟨ψ|∂_νψ⟩)

    This is the real part of the Quantum Geometric Tensor,
    equal to 4× the Fubini-Study metric.

    Args:
        state_fn: Function mapping parameters to quantum state
        params: Current parameter values
        epsilon: Finite difference step size

    Returns:
        QFI matrix (n_params × n_params)
    """
    qgt = compute_qgt(state_fn, params, epsilon)
    # QFI = 4 × Fubini-Study metric
    return 4 * qgt.metric


def compute_gradient(
    energy_fn: Callable[[np.ndarray], float],
    params: np.ndarray,
    epsilon: float = 1e-5
) -> np.ndarray:
    """
    Compute energy gradient using finite differences.

    Args:
        energy_fn: Function mapping parameters to energy
        params: Current parameter values
        epsilon: Finite difference step size

    Returns:
        Gradient vector
    """
    n_params = len(params)
    grad = np.zeros(n_params)

    for i in range(n_params):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += epsilon
        params_minus[i] -= epsilon

        grad[i] = (energy_fn(params_plus) - energy_fn(params_minus)) / (2 * epsilon)

    return grad


def natural_gradient_step(
    params: np.ndarray,
    gradient: np.ndarray,
    qfi_matrix: np.ndarray,
    learning_rate: float,
    regularization: float = 1e-4
) -> np.ndarray:
    """
    Compute a single natural gradient update.

    θ' = θ - η F⁻¹ ∇E

    Args:
        params: Current parameters
        gradient: Energy gradient ∇E
        qfi_matrix: Quantum Fisher Information matrix F
        learning_rate: Step size η
        regularization: Tikhonov regularization for stability

    Returns:
        Updated parameters
    """
    n_params = len(params)

    # Regularize QFI for numerical stability
    qfi_reg = qfi_matrix + regularization * np.eye(n_params)

    # Natural gradient direction: F⁻¹ ∇E
    try:
        natural_grad = np.linalg.solve(qfi_reg, gradient)
    except np.linalg.LinAlgError:
        # Fallback to pseudoinverse if singular
        natural_grad = np.linalg.lstsq(qfi_reg, gradient, rcond=None)[0]

    return params - learning_rate * natural_grad


def natural_gradient_descent(
    state_fn: Callable[[np.ndarray], QuantumState],
    energy_fn: Callable[[np.ndarray], float],
    initial_params: np.ndarray,
    learning_rate: float = 0.1,
    max_iterations: int = 100,
    convergence_tol: float = 1e-6,
    regularization: float = 1e-4,
    epsilon: float = 1e-5,
    verbose: bool = False
) -> OptimizationResult:
    """
    Optimize parameters using natural gradient descent.

    Args:
        state_fn: Function mapping parameters to quantum state
        energy_fn: Function mapping parameters to energy expectation
        initial_params: Starting parameters
        learning_rate: Step size
        max_iterations: Maximum number of iterations
        convergence_tol: Stop when |∇E| < tol
        regularization: Tikhonov regularization for QFI
        epsilon: Finite difference step size
        verbose: Print progress

    Returns:
        OptimizationResult with final parameters and history
    """
    params = initial_params.copy()
    energies = [energy_fn(params)]
    param_history = [params.copy()]
    converged = False

    for iteration in range(max_iterations):
        # Compute gradient
        gradient = compute_gradient(energy_fn, params, epsilon)

        # Check convergence
        grad_norm = np.linalg.norm(gradient)
        if grad_norm < convergence_tol:
            converged = True
            if verbose:
                print(f"Converged at iteration {iteration}: |∇E| = {grad_norm:.2e}")
            break

        # Compute QFI matrix
        qfi = quantum_fisher_information_matrix(state_fn, params, epsilon)

        # Natural gradient step
        params = natural_gradient_step(params, gradient, qfi, learning_rate, regularization)

        # Record history
        energy = energy_fn(params)
        energies.append(energy)
        param_history.append(params.copy())

        if verbose and iteration % 10 == 0:
            print(f"Iter {iteration}: E = {energy:.6f}, |∇E| = {grad_norm:.2e}")

    # Final QFI
    final_qfi = quantum_fisher_information_matrix(state_fn, params, epsilon)

    return OptimizationResult(
        params=params,
        energies=energies,
        param_history=param_history,
        converged=converged,
        iterations=len(energies) - 1,
        final_qfi=final_qfi
    )


def vanilla_gradient_descent(
    energy_fn: Callable[[np.ndarray], float],
    initial_params: np.ndarray,
    learning_rate: float = 0.1,
    max_iterations: int = 100,
    convergence_tol: float = 1e-6,
    epsilon: float = 1e-5,
    verbose: bool = False
) -> OptimizationResult:
    """
    Standard (vanilla) gradient descent for comparison.

    θ' = θ - η ∇E

    Args:
        energy_fn: Function mapping parameters to energy
        initial_params: Starting parameters
        learning_rate: Step size
        max_iterations: Maximum iterations
        convergence_tol: Convergence tolerance
        epsilon: Finite difference step
        verbose: Print progress

    Returns:
        OptimizationResult
    """
    params = initial_params.copy()
    energies = [energy_fn(params)]
    param_history = [params.copy()]
    converged = False

    for iteration in range(max_iterations):
        gradient = compute_gradient(energy_fn, params, epsilon)

        grad_norm = np.linalg.norm(gradient)
        if grad_norm < convergence_tol:
            converged = True
            if verbose:
                print(f"Converged at iteration {iteration}")
            break

        # Standard gradient step
        params = params - learning_rate * gradient

        energy = energy_fn(params)
        energies.append(energy)
        param_history.append(params.copy())

        if verbose and iteration % 10 == 0:
            print(f"Iter {iteration}: E = {energy:.6f}, |∇E| = {grad_norm:.2e}")

    return OptimizationResult(
        params=params,
        energies=energies,
        param_history=param_history,
        converged=converged,
        iterations=len(energies) - 1
    )


# ============================================================================
# VQE Helper Functions
# ============================================================================

def make_vqe_state_fn(
    circuit_fn: Callable[[np.ndarray, int], QuantumState],
    n_qubits: int
) -> Callable[[np.ndarray], QuantumState]:
    """
    Create a parameterized state function for VQE.

    Args:
        circuit_fn: Function (params, n_qubits) -> QuantumState
        n_qubits: Number of qubits

    Returns:
        Function params -> QuantumState
    """
    def state_fn(params: np.ndarray) -> QuantumState:
        return circuit_fn(params, n_qubits)
    return state_fn


def make_vqe_energy_fn(
    circuit_fn: Callable[[np.ndarray, int], QuantumState],
    hamiltonian: np.ndarray,
    n_qubits: int
) -> Callable[[np.ndarray], float]:
    """
    Create an energy function for VQE.

    E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩

    Args:
        circuit_fn: Function (params, n_qubits) -> QuantumState
        hamiltonian: System Hamiltonian
        n_qubits: Number of qubits

    Returns:
        Function params -> energy
    """
    def energy_fn(params: np.ndarray) -> float:
        state = circuit_fn(params, n_qubits)
        return np.real(state.expectation(hamiltonian))
    return energy_fn


def simple_ansatz(params: np.ndarray, n_qubits: int) -> QuantumState:
    """
    Simple hardware-efficient ansatz for testing.

    Applies Ry(θ) rotations to each qubit followed by a layer of CNOTs.

    Args:
        params: Rotation angles (one per qubit)
        n_qubits: Number of qubits

    Returns:
        Parameterized quantum state
    """
    from gates import Ry, CX, apply_gate

    # Start from |0...0⟩
    amplitudes = np.zeros(2**n_qubits, dtype=np.complex128)
    amplitudes[0] = 1.0

    # Apply Ry rotations
    for i, theta in enumerate(params[:n_qubits]):
        ry = Ry(theta)
        amplitudes = apply_gate(ry, amplitudes, [i], n_qubits)

    # Apply CNOT ladder (if more than 1 qubit)
    if n_qubits > 1:
        cx = CX()
        for i in range(n_qubits - 1):
            amplitudes = apply_gate(cx, amplitudes, [i, i+1], n_qubits)

    return QuantumState(n_qubits, amplitudes=amplitudes)


# ============================================================================
# Comparison and Analysis
# ============================================================================

def compare_optimizers(
    state_fn: Callable[[np.ndarray], QuantumState],
    energy_fn: Callable[[np.ndarray], float],
    initial_params: np.ndarray,
    natural_lr: float = 0.1,
    vanilla_lr: float = 0.01,
    max_iterations: int = 100
) -> Tuple[OptimizationResult, OptimizationResult]:
    """
    Compare natural gradient vs vanilla gradient descent.

    Args:
        state_fn: Parameterized state function
        energy_fn: Energy function
        initial_params: Starting parameters
        natural_lr: Learning rate for natural gradient
        vanilla_lr: Learning rate for vanilla gradient
        max_iterations: Maximum iterations

    Returns:
        (natural_result, vanilla_result)
    """
    natural_result = natural_gradient_descent(
        state_fn, energy_fn, initial_params,
        learning_rate=natural_lr,
        max_iterations=max_iterations
    )

    vanilla_result = vanilla_gradient_descent(
        energy_fn, initial_params,
        learning_rate=vanilla_lr,
        max_iterations=max_iterations
    )

    return natural_result, vanilla_result


def geodesic_parameter_distance(
    state_fn: Callable[[np.ndarray], QuantumState],
    params1: np.ndarray,
    params2: np.ndarray
) -> float:
    """
    Compute geodesic distance between parameter points.

    Uses the Fubini-Study metric integrated along a straight line
    in parameter space (approximation for nearby points).

    Args:
        state_fn: Parameterized state function
        params1: First parameter point
        params2: Second parameter point

    Returns:
        Approximate geodesic distance
    """
    from information_geometry import fubini_study_distance

    state1 = state_fn(params1)
    state2 = state_fn(params2)

    return fubini_study_distance(state1, state2)
