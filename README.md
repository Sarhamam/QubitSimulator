# Quantum Simulator

A functional quantum computer simulator with deep connections to information geometry, designed as a bridge between quantum mechanics and the NoeticEidos framework.

```
╔══════════════════════════════════════════════════════════════╗
║        QUANTUM SIMULATOR - NOETIC EIDOS INSPIRED             ║
║                                                              ║
║  A functional quantum computer simulator with:               ║
║  • Pure and mixed state representation                       ║
║  • Universal gate set                                        ║
║  • Circuit construction and execution                        ║
║  • Standard quantum algorithms                               ║
║  • Lindblad dynamics & geodesic deviation model              ║
║  • Natural gradient optimization (VQE)                       ║
║  • Spectral zeta functions (fingerprinting & duality)        ║
║  • Information geometry interpretation                       ║
╚══════════════════════════════════════════════════════════════╝
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Simulator Core                          │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   state.py      │   gates.py      │   circuit.py                │
│   QuantumState  │   Gate          │   Circuit                   │
│   • Amplitudes  │   • Paulis      │   • Operations              │
│   • Density ρ   │   • Rotations   │   • Execution               │
│   • Measurement │   • Entangling  │   • Visualization           │
│   • Partial Tr  │   • Controlled  │   • Analysis                │
├─────────────────┴─────────────────┴─────────────────────────────┤
│                       standard.py                               │
│   Deutsch-Jozsa • Bernstein-Vazirani • Grover • QFT • VQE       │
├─────────────────────────────────────────────────────────────────┤
│                        lindblad.py                              │
│   Master Equation • Noise Channels • Geodesic Deviation         │
├─────────────────────────────────────────────────────────────────┤
│                    natural_gradient.py                          │
│   QFI Matrix • Natural Gradient Descent • VQE Optimization      │
├─────────────────────────────────────────────────────────────────┤
│                   information_geometry.py                       │
│   Fubini-Study • QFI • Bures • Entanglement • Bloch Geodesics   │
├─────────────────────────────────────────────────────────────────┤
│                     spectral_zeta.py                            │
│   ζ_H(s) • ζ_L(s) • Heat Kernel • Spectral Fingerprints         │
├─────────────────────────────────────────────────────────────────┤
│                    bloch_sphere_viz.jsx                         │
│   Interactive 3D visualization with noise/geodesic deviation    │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Core dependencies
pip install numpy pytest

# For visualization (optional)
npm install react  # or use in claude.ai artifacts
```

## Quick Start

```python
from state import QuantumState, bell_state, plus_state
from circuit import Circuit
from gates import H, CX

# Create a Bell state via circuit
qc = Circuit(2)
qc.h(0)
qc.cx(0, 1)
result = qc.run()

print(result.final_state.to_ket_notation())
# Output: +0.707|00⟩ +0.707|11⟩

# Direct state construction
bell = bell_state(0)
print(f"Purity: {bell.purity}")  # 1.0 (pure state)
print(f"Entanglement: {bell.von_neumann_entropy}")  # ~1.0 ebit
```

## Module Reference

### `state.py` — Quantum State Representation

Supports both pure states (amplitudes) and mixed states (density matrices).

```python
from state import (
    QuantumState,           # Core state class
    computational_basis,    # |0⟩, |1⟩, |00⟩, etc.
    plus_state,            # |+⟩ = (|0⟩ + |1⟩)/√2
    bell_state,            # Bell states Φ±, Ψ±
    ghz_state,             # (|00...0⟩ + |11...1⟩)/√2
    w_state,               # Symmetric entanglement
    maximally_mixed,       # ρ = I/d
    random_state           # Haar-random pure state
)

state = plus_state(1)
state.probability(0)       # P(|0⟩) = 0.5
state.measure()            # Collapse with random outcome
state.measure_qubit(0)     # Partial measurement
state.expectation(H)       # ⟨H⟩ = Tr(ρH)
state.bloch_vector()       # [x, y, z] for single qubit
state.partial_trace([0])   # Trace out subsystem
state.tensor(other)        # |ψ⟩ ⊗ |φ⟩
state.fidelity(other)      # F(ρ, σ)
state.rho                  # Density matrix
state.purity               # Tr(ρ²)
state.von_neumann_entropy  # S = -Tr(ρ log ρ)
```

### `gates.py` — Quantum Gates

Universal gate set with geometric interpretations.

```python
from gates import (
    # Single-qubit
    I, X, Y, Z,            # Paulis
    H,                     # Hadamard
    S, Sdg, T, Tdg,        # Phase gates
    Rx, Ry, Rz,            # Parametric rotations
    P,                     # Phase gate P(φ)
    U3,                    # General SU(2)

    # Two-qubit
    CX, CY, CZ_gate,       # Controlled Paulis
    SWAP_gate, iSWAP,      # Swap operations
    CRx, CRy, CRz, CP,     # Controlled rotations
    XX, YY, ZZ,            # Ising interactions

    # Multi-qubit
    CCX, CCZ, CSWAP,       # Toffoli, CCZ, Fredkin

    # Utilities
    get_gate,              # Get gate by name
    apply_gate,            # Apply to state vector
    tensor_gates,          # G₁ ⊗ G₂
    controlled,            # Make controlled version
    decompose_to_u3        # Extract Euler angles
)

# Geometric interpretation
# X, Y, Z: π rotations around Bloch sphere axes
# H: rotation to diagonal axis (|+⟩, |−⟩ basis)
# Rx(θ): rotation by θ around x-axis
# U3(θ,φ,λ) = Rz(φ)Ry(θ)Rz(λ): any single-qubit gate
```

### `circuit.py` — Quantum Circuits

Composable circuit construction with fluent API.

```python
from circuit import Circuit

# Construction
qc = Circuit(n_qubits=3, n_classical=3)

# Single-qubit gates
qc.x(0).y(1).z(2)          # Paulis
qc.h(0).s(1).t(2)          # Clifford+T
qc.rx(np.pi/4, 0)          # Rotations
qc.u3(θ, φ, λ, qubit)      # General gate

# Two-qubit gates
qc.cx(control, target)     # CNOT
qc.cz(0, 1)                # CZ
qc.swap(0, 1)              # SWAP
qc.cp(np.pi/4, 0, 1)       # Controlled phase

# Multi-qubit gates
qc.ccx(0, 1, 2)            # Toffoli

# Control flow
qc.barrier()               # Visualization barrier
qc.reset(0)                # Reset to |0⟩
qc.measure(qubit, cbit)    # Measure to classical
qc.measure_all()           # Measure all qubits

# Execution
result = qc.run()                    # Single shot
result = qc.run(shots=1000)          # Statistics
result = qc.run(initial_state=psi)   # Custom initial

# Analysis
qc.depth                   # Circuit depth
qc.gate_count              # Total gates
qc.gate_counts()           # Per-gate breakdown
qc.inverse()               # Adjoint circuit
qc.compose(other)          # Append circuit

# Visualization
print(qc.draw())
# q0: ─[H]─●─────
# q1: ─────⊕─[M]─
```

### `standard.py` — Quantum Algorithms

```python
from standard import (
    # Algorithms
    deutsch_jozsa, run_deutsch_jozsa,
    bernstein_vazirani, run_bernstein_vazirani,
    grover_search, run_grover,
    qft,
    phase_estimation,

    # VQE Ansätze
    hardware_efficient_ansatz,
    uccsd_ansatz,

    # Utilities
    bell_pair,
    ghz_circuit,
    teleportation_circuit,
    superdense_coding
)

# Example: Grover's search for item 5 in 8-item database
counts = run_grover(n=3, marked=5, shots=100)
# {'101': 94, '000': 1, '011': 2, ...}  # High probability on |101⟩

# Example: Deutsch-Jozsa
result = run_deutsch_jozsa(3, oracle_type='balanced')
# 'balanced'
```

### `lindblad.py` — Open Quantum Systems & Geodesic Deviation

Implements the Lindblad master equation for simulating decoherence.

```python
from lindblad import (
    # Core evolution
    evolve_lindblad,         # Full Lindblad dynamics
    evolve_unitary,          # Pure Hamiltonian (ideal geodesic)
    lindblad_rhs,            # dρ/dt computation

    # Geodesic deviation analysis
    geodesic_deviation,      # Fidelity between ideal/noisy trajectories
    purity_decay,            # Track purity over time
    decoherence_rate,        # Estimate Γ from trajectory

    # Standard noise channels
    amplitude_damping_ops,   # T1 decay: |1⟩ → |0⟩
    phase_damping_ops,       # T2 dephasing: coherence loss
    depolarizing_ops,        # Uniform Pauli errors
    thermal_ops,             # Thermal bath coupling

    # Analysis
    steady_state,            # Find dρ/dt = 0 fixed point
    LindbladOperator         # Custom noise operators
)

# Example: Compare ideal vs noisy evolution
from state import plus_state
import numpy as np

state = plus_state(1)
H = np.array([[1, 0], [0, -1]])  # σ_z Hamiltonian
ops = amplitude_damping_ops(gamma=0.1)

ideal = evolve_unitary(state, H, dt=0.01, steps=100)
noisy = evolve_lindblad(state, H, ops, dt=0.01, steps=100)

deviation = geodesic_deviation(ideal, noisy)
# deviation[t] = fidelity between ideal and noisy at time t
```

### `natural_gradient.py` — Geometric Optimization

Natural gradient descent using the Quantum Fisher Information matrix.

```python
from natural_gradient import (
    # Core optimization
    natural_gradient_descent,    # θ' = θ - η F⁻¹ ∇E
    vanilla_gradient_descent,    # θ' = θ - η ∇E (for comparison)
    natural_gradient_step,       # Single update step

    # QFI computation
    quantum_fisher_information_matrix,  # F_μν
    compute_gradient,                    # ∇E via finite differences

    # VQE helpers
    make_vqe_state_fn,           # Create |ψ(θ)⟩ function
    make_vqe_energy_fn,          # Create ⟨H⟩(θ) function
    simple_ansatz,               # Hardware-efficient ansatz

    # Analysis
    compare_optimizers,          # Side-by-side comparison
    geodesic_parameter_distance  # Distance in state space
)

# Example: VQE with natural gradient
from gates import PAULI_Z
import numpy as np

H = np.kron(PAULI_Z, PAULI_Z)  # Z⊗Z Hamiltonian
state_fn = make_vqe_state_fn(simple_ansatz, n_qubits=2)
energy_fn = make_vqe_energy_fn(simple_ansatz, H, n_qubits=2)

result = natural_gradient_descent(
    state_fn, energy_fn,
    initial_params=np.array([0.5, 0.5]),
    learning_rate=0.3,
    max_iterations=50
)

print(f"Final energy: {result.energies[-1]}")  # Should approach -1
print(f"Converged: {result.converged}")
```

### `information_geometry.py` — Geometric Connections

The bridge to NoeticEidos concepts.

```python
from information_geometry import (
    # Metrics
    fubini_study_distance,       # Geodesic on CP^n
    fubini_study_metric_bloch,   # Metric in (θ,φ) coords
    bures_distance,              # Metric on density matrices

    # Fisher Information
    quantum_fisher_information,       # General QFI
    quantum_fisher_information_pure,  # 4·Var(H) for pure states

    # Geometric Tensor
    compute_qgt,                 # Full QGT
    QuantumGeometricTensor,      # Metric + Berry curvature

    # Entanglement
    entanglement_entropy,        # S(ρ_A)
    concurrence,                 # 2-qubit entanglement
    tangle,                      # C²

    # Bloch Sphere
    BlochCoordinates,            # (θ, φ, r) representation
    bloch_geodesic,              # Interpolation on sphere

    # NoeticEidos Connections
    fisher_rao_from_quantum,     # Classical FR from |⟨k|ψ⟩|²
    spectral_decomposition_geometry  # Eigenstructure analysis
)

# Example: Quantum Fisher Information
from state import plus_state
import numpy as np

psi = plus_state(1)
sigma_z = np.array([[1, 0], [0, -1]])
qfi = quantum_fisher_information_pure(psi, sigma_z)
# 4.0 (maximum sensitivity to Z rotations)
```

### `spectral_zeta.py` — Spectral Zeta Functions

Spectral fingerprinting and geodesic-spectral duality.

```python
from spectral_zeta import (
    # Hamiltonian spectral zeta
    spectral_zeta_hamiltonian,   # ζ_H(s) = Σₙ λₙ^{-s}
    heat_kernel_trace,           # K(t) = Tr(e^{-tH})
    mellin_from_heat_kernel,     # ζ(s) via Mellin transform

    # Lindbladian spectral zeta (vectorized superoperator)
    vectorize_lindbladian,       # d² × d² superoperator matrix
    lindbladian_spectrum,        # Complex eigenvalues
    spectral_zeta_lindbladian,   # ζ_L(s) for non-Hermitian

    # Spectral fingerprinting
    spectral_signature,          # Generate fingerprint over s-values
    spectral_distance,           # Compare fingerprints (L2, FR, JS)
    SpectralFingerprint,         # Fingerprint container

    # Geodesic-spectral duality
    spectral_geodesic_deviation, # Connect state/operator spectra
    decompose_lindbladian_spectrum,  # Structured decomposition

    # Data structures
    LindbladOperator,            # (operator, gamma) pair
    LindbladianDecomposition     # Coherent/dissipative split
)

# Example: Spectral fingerprint of a Hamiltonian
import numpy as np

H = np.array([[1, 0], [0, -1]])  # σ_z
s_values = np.linspace(1.5, 3.0, 20)
fingerprint = spectral_signature(H, s_values)
print(f"Spectral dim: {fingerprint.spectral_dimension}")

# Example: Compare ideal vs noisy spectral signatures
from lindblad import amplitude_damping_ops

ops = amplitude_damping_ops(gamma=0.1)
deviation, sig_H, sig_L = spectral_geodesic_deviation(H, ops, s_values)
print(f"Spectral deviation: {deviation:.4f}")

# Example: Heat kernel trace
K = heat_kernel_trace(H, t=1.0)
print(f"K(1) = {K:.4f}")  # Tr(e^{-H})
```

**Key mathematical relations:**
- ζ_H(s) = Tr(H^{-s}) = Σₙ λₙ^{-s} (Hamiltonian spectral zeta)
- K(t) = Tr(e^{-tH}) (heat kernel trace, Mellin partner of ζ)
- ζ(s) = (1/Γ(s)) ∫ t^{s-1} K(t) dt (Mellin transform relation)
- ζ_L(s) = Σₙ |λₙ|^{-s} e^{-is·arg(λₙ)} (Lindbladian, handles complex spectrum)

## NoeticEidos Connections

### Mathematical Correspondences

| NoeticEidos Concept | Quantum Analog |
|---------------------|----------------|
| Fisher-Rao metric | Fubini-Study metric (= 4× Fisher-Rao) |
| Statistical manifold | Projective Hilbert space CP^n |
| Spectral decomposition | Observable eigenbases |
| Resonance signatures | Hamiltonian spectrum |
| Spectral zeta ζ(s) | ζ_H(s) = Tr(H^{-s}) — operator fingerprint |
| Heat kernel trace | K(t) = Tr(e^{-tH}) — Mellin partner |
| Dual transport | Parallel transport on state manifold |
| κζ dynamics | Unitary evolution / quantum phase |
| Geodesic-spectral duality | State deviation ↔ spectral deviation |

### Noise as Geodesic Deviation

The simulator frames decoherence geometrically:

```
Ideal evolution:    |ψ(t)⟩ = e^(-iHt)|ψ₀⟩           (Hamiltonian geodesic)
Noisy evolution:    dρ/dt = -i[H,ρ] + Σₖ D[Lₖ](ρ)   (Lindblad deviation)

where D[L](ρ) = LρL† - ½{L†L, ρ} is the Lindblad dissipator.
```

**Geometric interpretation:**
- Ideal unitary evolution follows geodesics on the state manifold CP^n
- Lindblad operators act as "curvature perturbations"
- The fidelity decay measures accumulated geodesic deviation
- Purity decay measures how far the state moves from the pure state manifold

```python
# Demonstration
from lindblad import evolve_unitary, evolve_lindblad, geodesic_deviation
from lindblad import amplitude_damping_ops

ideal = evolve_unitary(state, H, dt=0.01, steps=100)
noisy = evolve_lindblad(state, H, amplitude_damping_ops(0.1), dt=0.01, steps=100)

# Track how noisy trajectory deviates from ideal geodesic
deviation = geodesic_deviation(ideal, noisy)
# deviation[0] = 1.0 (initially identical)
# deviation[t] < 1.0 (fidelity decreases over time)
```

The visualization component (`bloch_sphere_viz.jsx`) demonstrates this with:
- Ideal trajectories following great circles on the Bloch sphere
- Noisy trajectories spiraling inward (toward maximally mixed state at center)
- The radius |r| < 1 indicates deviation from the pure state surface

## Running the Demo

```bash
python demo.py
```

This executes all simulator capabilities:
1. Basic state manipulation
2. Gate operations
3. Circuit construction
4. Standard algorithms (Deutsch-Jozsa, Grover, etc.)
5. Information geometry analysis
6. **Geodesic deviation model** (ideal vs noisy evolution)
7. **Noise channel comparison** (T1, T2, depolarizing)
8. **Entanglement decay** under local noise
9. **Natural gradient VQE** optimization
10. **Spectral zeta functions** (fingerprints, heat kernel, geodesic-spectral duality)

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific module tests
python -m pytest tests/test_lindblad.py -v
python -m pytest tests/test_natural_gradient.py -v
python -m pytest tests/test_spectral_zeta.py -v
```

Test coverage: **213 tests** across all modules.

## File Structure

```
quantum_simulator/
├── state.py                 # State representation (vectors, density matrices)
├── gates.py                 # Gate definitions and application
├── circuit.py               # Circuit construction and execution
├── standard.py              # Standard quantum algorithms
├── lindblad.py              # Lindblad master equation, noise channels
├── natural_gradient.py      # Natural gradient optimization, VQE
├── information_geometry.py  # Geometric analysis tools
├── spectral_zeta.py         # Spectral zeta functions, fingerprinting
├── bloch_sphere_viz.jsx     # Interactive React visualization
├── demo.py                  # Demonstration script
├── pytest.ini               # Test configuration
├── tests/
│   ├── test_state.py        # State tests (35 tests)
│   ├── test_gates.py        # Gate tests (63 tests)
│   ├── test_information_geometry.py  # Geometry tests (34 tests)
│   ├── test_lindblad.py     # Lindblad tests (28 tests)
│   ├── test_natural_gradient.py      # Optimization tests (20 tests)
│   └── test_spectral_zeta.py         # Spectral zeta tests (33 tests)
└── README.md                # This file
```

## Quantum Advantage Summary

| Algorithm | Classical | Quantum | Speedup |
|-----------|-----------|---------|---------|
| Deutsch-Jozsa | O(2^(n-1)+1) | O(1) | Exponential |
| Bernstein-Vazirani | O(n) | O(1) | Linear→Constant |
| Simon's Problem | O(2^(n/2)) | O(n) | Exponential |
| Grover's Search | O(N) | O(√N) | Quadratic |
| Shor's Factoring | O(exp(n)) | O(n³) | Exponential |
| QFT | O(n·2^n) | O(n²) | Exponential |

## License

Part of the NoeticEidos research framework.

## References

- Nielsen & Chuang, "Quantum Computation and Quantum Information"
- Bengtsson & Życzkowski, "Geometry of Quantum States"
- Amari & Nagaoka, "Methods of Information Geometry"
- Breuer & Petruccione, "The Theory of Open Quantum Systems"
