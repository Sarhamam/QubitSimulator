"""
Quantum Information Geometry

Bridges quantum mechanics to information geometry, connecting to
Noetic Eidos concepts like Fisher-Rao metrics and spectral analysis.

Key correspondences:
- Fubini-Study metric on CP^n ↔ Fisher-Rao on statistical manifolds
- Quantum geometric tensor → Fisher info + Berry curvature
- Bloch sphere → Statistical 2-simplex
- Entanglement entropy → Mutual information geometry

This module provides geometric tools for analyzing quantum states
and their evolution through the lens of information geometry.
"""

import numpy as np
from typing import Tuple, Optional, List, Callable
from dataclasses import dataclass

from state import QuantumState


# ============================================================================
# Quantum Geometric Tensor
# ============================================================================

@dataclass
class QuantumGeometricTensor:
    """
    The Quantum Geometric Tensor (QGT) at a point in parameter space.
    
    Q_μν = ⟨∂_μ ψ | ∂_ν ψ⟩ - ⟨∂_μ ψ | ψ⟩⟨ψ | ∂_ν ψ⟩
    
    The real part is the Fubini-Study metric (quantum Fisher information).
    The imaginary part is the Berry curvature.
    
    Attributes:
        metric: Fubini-Study metric (real, symmetric)
        curvature: Berry curvature (imaginary, antisymmetric)
        full_tensor: Complete QGT
    """
    metric: np.ndarray       # g_μν = Re(Q_μν)
    curvature: np.ndarray    # F_μν = -2 Im(Q_μν)
    full_tensor: np.ndarray  # Q_μν
    
    @property
    def n_params(self) -> int:
        return self.metric.shape[0]
    
    def geodesic_distance(self, delta_params: np.ndarray) -> float:
        """
        Compute infinitesimal distance in Fubini-Study metric.
        
        ds² = g_μν dθ^μ dθ^ν
        """
        return np.sqrt(delta_params @ self.metric @ delta_params)
    
    def berry_phase(self, loop_params: np.ndarray) -> float:
        """
        Compute Berry phase for a small loop in parameter space.
        
        γ = ∮ A·dl = ∫∫ F dS
        """
        # For small loops, γ ≈ F_μν * area
        # This is a simplification for infinitesimal loops
        area = np.linalg.det(loop_params) if loop_params.shape[0] >= 2 else 0
        return 0.5 * np.sum(self.curvature) * area


def compute_qgt(state_function: Callable[[np.ndarray], QuantumState],
                params: np.ndarray,
                epsilon: float = 1e-5) -> QuantumGeometricTensor:
    """
    Compute the Quantum Geometric Tensor numerically.
    
    Args:
        state_function: Maps parameters θ → |ψ(θ)⟩
        params: Current parameter values
        epsilon: Finite difference step size
        
    Returns:
        QuantumGeometricTensor at the given parameters
    """
    n_params = len(params)
    psi = state_function(params)
    
    # Compute derivatives using finite differences
    derivatives = []
    for i in range(n_params):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += epsilon
        params_minus[i] -= epsilon
        
        psi_plus = state_function(params_plus)
        psi_minus = state_function(params_minus)
        
        # Central difference
        d_psi = (psi_plus.amplitudes - psi_minus.amplitudes) / (2 * epsilon)
        derivatives.append(d_psi)
    
    # Compute QGT components
    Q = np.zeros((n_params, n_params), dtype=np.complex128)
    
    for mu in range(n_params):
        for nu in range(n_params):
            # ⟨∂_μ ψ | ∂_ν ψ⟩
            overlap_deriv = np.vdot(derivatives[mu], derivatives[nu])
            
            # ⟨∂_μ ψ | ψ⟩⟨ψ | ∂_ν ψ⟩
            proj_mu = np.vdot(derivatives[mu], psi.amplitudes)
            proj_nu = np.vdot(psi.amplitudes, derivatives[nu])
            
            Q[mu, nu] = overlap_deriv - proj_mu * proj_nu
    
    # Extract metric and curvature
    metric = np.real(Q)
    curvature = -2 * np.imag(Q)
    
    return QuantumGeometricTensor(
        metric=metric,
        curvature=curvature,
        full_tensor=Q
    )


# ============================================================================
# Fubini-Study Metric
# ============================================================================

def fubini_study_distance(psi: QuantumState, phi: QuantumState) -> float:
    """
    Compute Fubini-Study distance between two pure states.
    
    d_FS(ψ, φ) = arccos(|⟨ψ|φ⟩|)
    
    This is the geodesic distance on the projective Hilbert space CP^n.
    Ranges from 0 (identical) to π/2 (orthogonal).
    """
    if psi.state_type.value != 'pure' or phi.state_type.value != 'pure':
        raise ValueError("Fubini-Study distance requires pure states")
    
    overlap = np.abs(np.vdot(psi.amplitudes, phi.amplitudes))
    overlap = np.clip(overlap, 0, 1)  # Numerical stability
    return np.arccos(overlap)


def fubini_study_metric_bloch(theta: float, phi: float) -> np.ndarray:
    """
    Fubini-Study metric on the Bloch sphere in (θ, φ) coordinates.
    
    For state |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩
    
    ds² = (1/4)[dθ² + sin²(θ)dφ²]
    
    This is (1/4) times the round metric on S².
    """
    return 0.25 * np.array([
        [1, 0],
        [0, np.sin(theta)**2]
    ])


# ============================================================================
# Quantum Fisher Information
# ============================================================================

def quantum_fisher_information(rho: np.ndarray, 
                                generator: np.ndarray) -> float:
    """
    Compute Quantum Fisher Information for a density matrix.
    
    F_Q[ρ, H] = 2 Σ_{k,l} (λ_k - λ_l)²/(λ_k + λ_l) |⟨k|H|l⟩|²
    
    where λ_k, |k⟩ are eigenvalues/eigenvectors of ρ.
    
    The QFI sets the quantum Cramér-Rao bound for parameter estimation:
    Var(θ) ≥ 1/(n F_Q)
    
    Args:
        rho: Density matrix
        generator: Hamiltonian generating the parametrized evolution
        
    Returns:
        Quantum Fisher Information
    """
    # Eigendecomposition of rho
    eigenvalues, eigenvectors = np.linalg.eigh(rho)
    
    # Clean up numerical noise
    eigenvalues = np.maximum(eigenvalues, 0)
    
    dim = len(eigenvalues)
    F_Q = 0.0
    
    for k in range(dim):
        for l in range(dim):
            lam_k, lam_l = eigenvalues[k], eigenvalues[l]
            
            if lam_k + lam_l > 1e-12:  # Avoid division by zero
                # Matrix element ⟨k|H|l⟩
                v_k = eigenvectors[:, k]
                v_l = eigenvectors[:, l]
                H_kl = np.vdot(v_k, generator @ v_l)
                
                # Contribution to QFI
                F_Q += 2 * (lam_k - lam_l)**2 / (lam_k + lam_l) * np.abs(H_kl)**2
    
    return np.real(F_Q)


def quantum_fisher_information_pure(psi: QuantumState, 
                                     generator: np.ndarray) -> float:
    """
    QFI for pure states: F_Q = 4(⟨H²⟩ - ⟨H⟩²) = 4 Var(H)
    """
    if psi.state_type.value != 'pure':
        return quantum_fisher_information(psi.rho, generator)
    
    H_expectation = np.real(psi.expectation(generator))
    H2_expectation = np.real(psi.expectation(generator @ generator))
    
    return 4 * (H2_expectation - H_expectation**2)


# ============================================================================
# Bures Distance and Fidelity Geometry
# ============================================================================

def bures_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    Compute Bures distance between two density matrices.
    
    d_B(ρ, σ) = √(2(1 - √F(ρ,σ)))
    
    where F is the fidelity. This is a proper metric on the space
    of density matrices and reduces to Fubini-Study for pure states.
    """
    F = fidelity(rho, sigma)
    return np.sqrt(2 * (1 - np.sqrt(F)))


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    Compute quantum fidelity F(ρ, σ) = (Tr√(√ρ σ √ρ))²
    """
    sqrt_rho = _matrix_sqrt(rho)
    inner = sqrt_rho @ sigma @ sqrt_rho
    sqrt_inner = _matrix_sqrt(inner)
    return np.real(np.trace(sqrt_inner))**2


def _matrix_sqrt(A: np.ndarray) -> np.ndarray:
    """Matrix square root via eigendecomposition."""
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    eigenvalues = np.maximum(eigenvalues, 0)
    sqrt_eigenvalues = np.sqrt(eigenvalues)
    return eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.conj().T


# ============================================================================
# Entanglement Geometry
# ============================================================================

def entanglement_entropy(state: QuantumState, 
                         subsystem: List[int]) -> float:
    """
    Compute entanglement entropy of a bipartite pure state.
    
    S(A) = -Tr(ρ_A log ρ_A)
    
    where ρ_A = Tr_B(|ψ⟩⟨ψ|) is the reduced density matrix.
    """
    reduced = state.partial_trace(subsystem)
    return reduced.von_neumann_entropy


def concurrence(state: QuantumState) -> float:
    """
    Compute concurrence for a 2-qubit state.
    
    C(ρ) = max(0, λ₁ - λ₂ - λ₃ - λ₄)
    
    where λᵢ are eigenvalues of √(√ρ ρ̃ √ρ) in decreasing order,
    and ρ̃ = (σ_y ⊗ σ_y) ρ* (σ_y ⊗ σ_y).
    
    Concurrence ranges from 0 (separable) to 1 (maximally entangled).
    """
    if state.n_qubits != 2:
        raise ValueError("Concurrence is defined for 2-qubit states")
    
    rho = state.rho
    
    # σ_y ⊗ σ_y
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sigma_yy = np.kron(sigma_y, sigma_y)
    
    # ρ̃ = (σ_y ⊗ σ_y) ρ* (σ_y ⊗ σ_y)
    rho_tilde = sigma_yy @ np.conj(rho) @ sigma_yy
    
    # R = √(√ρ ρ̃ √ρ)
    sqrt_rho = _matrix_sqrt(rho)
    R_squared = sqrt_rho @ rho_tilde @ sqrt_rho
    
    # Eigenvalues of R
    eigenvalues = np.sqrt(np.maximum(np.linalg.eigvalsh(R_squared), 0))
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order
    
    return max(0, eigenvalues[0] - np.sum(eigenvalues[1:]))


def tangle(state: QuantumState) -> float:
    """
    Tangle (squared concurrence) for 2-qubit states.
    
    τ = C²
    """
    return concurrence(state)**2


# ============================================================================
# Bloch Sphere Geometry
# ============================================================================

@dataclass
class BlochCoordinates:
    """Bloch sphere representation of a single-qubit state."""
    theta: float  # Polar angle [0, π]
    phi: float    # Azimuthal angle [0, 2π)
    r: float      # Radius (1 for pure, < 1 for mixed)
    
    @property
    def cartesian(self) -> np.ndarray:
        """Convert to Cartesian coordinates (x, y, z)."""
        return np.array([
            self.r * np.sin(self.theta) * np.cos(self.phi),
            self.r * np.sin(self.theta) * np.sin(self.phi),
            self.r * np.cos(self.theta)
        ])
    
    @classmethod
    def from_state(cls, state: QuantumState) -> 'BlochCoordinates':
        """Create from a single-qubit QuantumState."""
        bloch = state.bloch_vector()
        if bloch is None:
            raise ValueError("Not a single-qubit state")
        
        x, y, z = bloch
        r = np.linalg.norm(bloch)
        
        if r < 1e-10:
            return cls(0, 0, 0)
        
        theta = np.arccos(np.clip(z / r, -1, 1))
        phi = np.arctan2(y, x) % (2 * np.pi)
        
        return cls(theta, phi, r)
    
    def to_state(self) -> QuantumState:
        """Convert back to a QuantumState."""
        if np.abs(self.r - 1) < 1e-10:
            # Pure state
            amplitudes = np.array([
                np.cos(self.theta / 2),
                np.exp(1j * self.phi) * np.sin(self.theta / 2)
            ], dtype=np.complex128)
            return QuantumState(1, amplitudes=amplitudes)
        else:
            # Mixed state
            x, y, z = self.cartesian
            rho = 0.5 * np.array([
                [1 + z, x - 1j*y],
                [x + 1j*y, 1 - z]
            ], dtype=np.complex128)
            return QuantumState(1, density_matrix=rho)


def bloch_geodesic(start: BlochCoordinates, end: BlochCoordinates,
                   t: float) -> BlochCoordinates:
    """
    Geodesic interpolation on the Bloch sphere.
    
    For pure states (r=1), this follows a great circle.
    
    Args:
        start: Starting point
        end: Ending point
        t: Interpolation parameter [0, 1]
        
    Returns:
        Point at parameter t along the geodesic
    """
    if start.r < 1 - 1e-6 or end.r < 1 - 1e-6:
        # For mixed states, use linear interpolation in Cartesian
        p_start = start.cartesian
        p_end = end.cartesian
        p_t = (1 - t) * p_start + t * p_end
        
        r = np.linalg.norm(p_t)
        if r < 1e-10:
            return BlochCoordinates(0, 0, 0)
        
        theta = np.arccos(np.clip(p_t[2] / r, -1, 1))
        phi = np.arctan2(p_t[1], p_t[0]) % (2 * np.pi)
        return BlochCoordinates(theta, phi, r)
    
    # For pure states, use SLERP (spherical linear interpolation)
    p_start = start.cartesian
    p_end = end.cartesian
    
    dot = np.clip(np.dot(p_start, p_end), -1, 1)
    omega = np.arccos(dot)
    
    if np.abs(omega) < 1e-10:
        return start
    
    p_t = (np.sin((1 - t) * omega) * p_start + 
           np.sin(t * omega) * p_end) / np.sin(omega)
    
    theta = np.arccos(np.clip(p_t[2], -1, 1))
    phi = np.arctan2(p_t[1], p_t[0]) % (2 * np.pi)
    
    return BlochCoordinates(theta, phi, 1.0)


# ============================================================================
# Connections to Noetic Eidos
# ============================================================================

def fisher_rao_from_quantum(state: QuantumState) -> np.ndarray:
    """
    Extract classical Fisher-Rao metric from quantum state.
    
    The probability distribution p(k) = |⟨k|ψ⟩|² defines a point
    on the probability simplex. The Fisher-Rao metric at this point is:
    
    g_FR = Σ_k (1/p_k) dp_k ⊗ dp_k
    
    This connects quantum states to the statistical manifold structure
    central to Noetic Eidos.
    """
    probs = state.probabilities()
    dim = len(probs)
    
    # Fisher-Rao metric on the simplex
    # In the coordinates p_1, ..., p_{n-1} (p_n = 1 - sum)
    metric = np.zeros((dim - 1, dim - 1))
    
    for i in range(dim - 1):
        for j in range(dim - 1):
            # Contribution from p_i and p_j
            metric[i, j] = (1 if i == j else 0) / max(probs[i], 1e-12)
            # Contribution from p_n (constraint term)
            metric[i, j] += 1 / max(probs[-1], 1e-12)
    
    return metric


def spectral_decomposition_geometry(state: QuantumState) -> dict:
    """
    Analyze the spectral structure of a density matrix.
    
    Returns eigenvalues and their geometric interpretation,
    connecting to spectral analysis in Noetic Eidos.
    """
    rho = state.rho
    eigenvalues, eigenvectors = np.linalg.eigh(rho)
    
    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Spectral entropy (like Shannon entropy of eigenvalue distribution)
    spectral_entropy = -np.sum(
        eigenvalues * np.log2(eigenvalues + 1e-12)
        for lam in eigenvalues if lam > 1e-12
    )
    
    # Participation ratio (effective rank)
    participation_ratio = 1 / np.sum(eigenvalues**2)
    
    return {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'spectral_entropy': spectral_entropy,
        'participation_ratio': participation_ratio,
        'purity': np.sum(eigenvalues**2),
        'rank': np.sum(eigenvalues > 1e-10)
    }
