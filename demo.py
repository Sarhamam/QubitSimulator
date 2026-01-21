#!/usr/bin/env python3
"""
Quantum Simulator Demo

Tests and demonstrates the quantum simulator capabilities:
1. Basic state manipulation
2. Gate operations
3. Circuit execution
4. Standard algorithms (Deutsch-Jozsa, Grover, etc.)
5. Information geometry connections
"""

import numpy as np

from state import (
    QuantumState, computational_basis, plus_state, bell_state,
    ghz_state, w_state, random_state
)
from circuit import Circuit
from gates import H, X, Y, Z, CX, T, get_gate
from standard import (
    deutsch_jozsa, run_deutsch_jozsa,
    bernstein_vazirani, run_bernstein_vazirani,
    grover_search, run_grover,
    qft, bell_pair, ghz_circuit
)
from information_geometry import (
    fubini_study_distance, quantum_fisher_information_pure,
    BlochCoordinates, entanglement_entropy, concurrence
)
from lindblad import (
    evolve_lindblad, evolve_unitary, geodesic_deviation, purity_decay,
    amplitude_damping_ops, phase_damping_ops, depolarizing_ops,
    steady_state, decoherence_rate, LindbladOperator
)
from natural_gradient import (
    natural_gradient_descent, vanilla_gradient_descent,
    make_vqe_energy_fn, make_vqe_state_fn, simple_ansatz,
    quantum_fisher_information_matrix
)
from spectral_zeta import (
    spectral_zeta_hamiltonian, spectral_zeta_lindbladian,
    spectral_signature, spectral_distance,
    heat_kernel_trace, decompose_lindbladian_spectrum,
    spectral_geodesic_deviation
)


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def demo_basic_states():
    """Demonstrate basic quantum state operations."""
    separator("BASIC QUANTUM STATES")
    
    # Computational basis states
    print("\n1. Computational Basis States:")
    zero = computational_basis(1, 0)
    one = computational_basis(1, 1)
    print(f"   |0⟩: {zero.to_ket_notation()}")
    print(f"   |1⟩: {one.to_ket_notation()}")
    
    # Superposition states
    print("\n2. Superposition States:")
    plus = plus_state(1)
    print(f"   |+⟩ = {plus.to_ket_notation()}")
    print(f"   Probabilities: P(0)={plus.probability(0):.3f}, P(1)={plus.probability(1):.3f}")
    
    # Bell state
    print("\n3. Entangled States:")
    bell = bell_state(0)
    print(f"   |Φ+⟩ = {bell.to_ket_notation()}")
    print(f"   Purity: {bell.purity:.4f}")
    
    # GHZ state
    ghz = ghz_state(3)
    print(f"   |GHZ₃⟩ = {ghz.to_ket_notation()}")
    
    # W state
    w = w_state(3)
    print(f"   |W₃⟩ = {w.to_ket_notation()}")


def demo_measurement():
    """Demonstrate quantum measurement."""
    separator("QUANTUM MEASUREMENT")
    
    # Create superposition and measure multiple times
    print("\n1. Measuring |+⟩ state (100 shots):")
    results = {'0': 0, '1': 0}
    for _ in range(100):
        plus = plus_state(1)
        outcome, _ = plus.measure()
        results[str(outcome)] += 1
    print(f"   Results: |0⟩: {results['0']}%, |1⟩: {results['1']}%")
    
    # Partial measurement
    print("\n2. Partial measurement of Bell state:")
    bell = bell_state(0)
    print(f"   Initial state: {bell.to_ket_notation()}")
    outcome, collapsed = bell.measure_qubit(0)
    print(f"   Measured qubit 0: {outcome}")
    print(f"   Collapsed state: {collapsed.to_ket_notation()}")


def demo_gates():
    """Demonstrate quantum gate operations."""
    separator("QUANTUM GATES")
    
    # Single qubit gates on |0⟩
    print("\n1. Single-Qubit Gates on |0⟩:")
    state = computational_basis(1, 0)
    
    gates = [('X', X()), ('Y', Y()), ('Z', Z()), ('H', H()), ('T', T())]
    for name, gate in gates:
        qc = Circuit(1)
        qc.apply(gate, [0])
        result = qc.run(state)
        print(f"   {name}|0⟩ = {result.final_state.to_ket_notation()}")
    
    # Two-qubit entangling gate
    print("\n2. Creating Entanglement with CNOT:")
    qc = Circuit(2)
    qc.h(0)
    qc.cx(0, 1)
    result = qc.run()
    print(f"   H⊗I then CNOT: {result.final_state.to_ket_notation()}")


def demo_circuits():
    """Demonstrate circuit construction and execution."""
    separator("QUANTUM CIRCUITS")
    
    # Build a simple circuit
    print("\n1. Bell State Circuit:")
    qc = Circuit(2)
    qc.h(0)
    qc.cx(0, 1)
    print(qc.draw())
    
    result = qc.run()
    print(f"   Final state: {result.final_state.to_ket_notation()}")
    
    # More complex circuit
    print("\n2. GHZ State Circuit (3 qubits):")
    qc = Circuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    print(qc.draw())
    
    result = qc.run()
    print(f"   Final state: {result.final_state.to_ket_notation()}")
    print(f"   Circuit depth: {qc.depth}")
    print(f"   Gate count: {qc.gate_count}")
    
    # Circuit with measurement
    print("\n3. Circuit with Measurement (100 shots):")
    qc = Circuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    
    result = qc.run(shots=100)
    print(f"   Measurement outcomes: {result.counts}")


def demo_deutsch_jozsa():
    """Demonstrate Deutsch-Jozsa algorithm."""
    separator("DEUTSCH-JOZSA ALGORITHM")
    
    print("\nDetermines if f:{0,1}ⁿ → {0,1} is constant or balanced in O(1)")
    
    # Test with constant oracle
    print("\n1. Testing CONSTANT function:")
    result = run_deutsch_jozsa(3, oracle_type='constant')
    print(f"   Detected: {result}")
    
    # Test with balanced oracle
    print("\n2. Testing BALANCED function:")
    result = run_deutsch_jozsa(3, oracle_type='balanced', pattern=5)
    print(f"   Detected: {result}")
    
    # Show the circuit
    print("\n3. Circuit for n=2:")
    circuit, _ = deutsch_jozsa(2, oracle_type='balanced')
    print(circuit.draw())


def demo_bernstein_vazirani():
    """Demonstrate Bernstein-Vazirani algorithm."""
    separator("BERNSTEIN-VAZIRANI ALGORITHM")
    
    print("\nFinds secret string s in f(x) = s·x in O(1)")
    
    secrets = [5, 11, 7]  # Binary: 101, 1011, 111
    
    for secret in secrets:
        n = secret.bit_length()
        found = run_bernstein_vazirani(n, secret)
        print(f"   Secret: {secret:0{n}b} ({secret}) → Found: {found:0{n}b} ({found}) ✓" 
              if found == secret else f"   FAILED")


def demo_grover():
    """Demonstrate Grover's search algorithm."""
    separator("GROVER'S SEARCH ALGORITHM")
    
    print("\nSearches unsorted database of N items in O(√N)")
    
    # Small example: 3 qubits (8 items)
    n = 3
    marked = 5  # Binary: 101
    
    print(f"\n1. Searching for item {marked} in database of {2**n} items:")
    
    circuit = grover_search(n, marked)
    print(f"   Circuit depth: {circuit.depth}")
    
    # Run multiple times
    counts = run_grover(n, marked, shots=100)
    print(f"   Results (100 shots): {counts}")
    
    # Check success
    target_key = format(marked, f'0{n}b')
    success_rate = counts.get(target_key, 0) / 100
    print(f"   Success rate for |{target_key}⟩: {success_rate*100:.1f}%")


def demo_qft():
    """Demonstrate Quantum Fourier Transform."""
    separator("QUANTUM FOURIER TRANSFORM")
    
    print("\nQFT transforms computational basis to Fourier basis")
    
    # Apply QFT to |1⟩ state (3 qubits)
    n = 3
    qft_circuit = qft(n)
    
    print(f"\n1. QFT Circuit ({n} qubits):")
    print(f"   Depth: {qft_circuit.depth}")
    print(f"   Gates: {qft_circuit.gate_counts()}")
    
    # Apply to |001⟩
    initial = computational_basis(n, 1)
    result = qft_circuit.run(initial)
    
    print(f"\n2. QFT|001⟩:")
    print(f"   Probabilities: {result.final_state.probabilities().round(3)}")
    
    # Verify inverse
    inv_qft = qft(n, inverse=True)
    recovered = inv_qft.run(result.final_state)
    print(f"\n3. QFT⁻¹ · QFT|001⟩ = {recovered.final_state.to_ket_notation()}")


def demo_information_geometry():
    """Demonstrate information geometry connections."""
    separator("INFORMATION GEOMETRY")
    
    print("\nConnecting quantum mechanics to information geometry")
    
    # Fubini-Study distance
    print("\n1. Fubini-Study Distance:")
    state0 = computational_basis(1, 0)
    state1 = computational_basis(1, 1)
    plus = plus_state(1)
    
    d_01 = fubini_study_distance(state0, state1)
    d_0plus = fubini_study_distance(state0, plus)
    
    print(f"   d(|0⟩, |1⟩) = {d_01:.4f} (= π/2, orthogonal)")
    print(f"   d(|0⟩, |+⟩) = {d_0plus:.4f} (= π/4)")
    
    # Bloch sphere
    print("\n2. Bloch Sphere Coordinates:")
    states = [
        ("|0⟩", state0),
        ("|1⟩", state1),
        ("|+⟩", plus)
    ]
    
    for name, state in states:
        bloch = BlochCoordinates.from_state(state)
        cart = bloch.cartesian
        print(f"   {name}: θ={bloch.theta:.3f}, φ={bloch.phi:.3f} → ({cart[0]:.2f}, {cart[1]:.2f}, {cart[2]:.2f})")
    
    # Quantum Fisher Information
    print("\n3. Quantum Fisher Information:")
    sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    qfi_0 = quantum_fisher_information_pure(state0, sigma_z)
    qfi_plus = quantum_fisher_information_pure(plus, sigma_z)
    print(f"   F_Q(|0⟩, σ_z) = {qfi_0:.4f} (saturates at 0 - eigenstates)")
    print(f"   F_Q(|+⟩, σ_z) = {qfi_plus:.4f} (maximum sensitivity)")
    
    # Entanglement measures
    print("\n4. Entanglement Measures:")
    bell = bell_state(0)
    separable = computational_basis(2, 0)
    
    c_bell = concurrence(bell)
    c_sep = concurrence(separable)
    print(f"   Concurrence(|Φ+⟩) = {c_bell:.4f} (maximally entangled)")
    print(f"   Concurrence(|00⟩) = {c_sep:.4f} (separable)")


def demo_algorithm_comparison():
    """Compare classical vs quantum algorithm complexity."""
    separator("QUANTUM ADVANTAGE SUMMARY")

    print("""
    Algorithm              Classical    Quantum      Speedup
    ─────────────────────────────────────────────────────────
    Deutsch-Jozsa          O(2ⁿ⁻¹+1)    O(1)         Exponential
    Bernstein-Vazirani     O(n)         O(1)         Linear→Constant
    Simon's Problem        O(2^(n/2))   O(n)         Exponential
    Grover's Search        O(N)         O(√N)        Quadratic
    Shor's Factoring       O(exp(n))    O(n³)        Exponential
    QFT                    O(n·2ⁿ)      O(n²)        Exponential
    """)


def demo_geodesic_deviation():
    """
    Demonstrate geodesic deviation model for quantum decoherence.

    Key insight: Ideal unitary evolution follows geodesics on the quantum
    state manifold (projective Hilbert space CP^n). Noise causes the actual
    trajectory to deviate from this geodesic - this deviation is measurable
    and connects to the geometric interpretation of decoherence.
    """
    separator("GEODESIC DEVIATION MODEL")

    print("""
    Geometric Interpretation of Decoherence:
    ─────────────────────────────────────────
    • Unitary evolution: geodesic motion on state manifold CP^n
    • Lindblad noise: causes deviation from ideal geodesic
    • Fidelity decay: measures accumulated geodesic deviation
    • Purity decay: measures how far state moves toward interior
    """)

    # Setup: single qubit rotating around Z-axis
    sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    H = sigma_z  # Hamiltonian = σ_z (rotation around z-axis)

    print("\n1. Comparing Ideal vs Noisy Evolution:")
    print("   System: Single qubit, H = σ_z (Z-rotation)")
    print("   Initial state: |+⟩ = (|0⟩ + |1⟩)/√2")

    state = plus_state(1)
    dt = 0.05
    steps = 60
    total_time = dt * steps

    # Ideal evolution (pure unitary)
    ideal_trajectory = evolve_unitary(state, H, dt, steps)

    # Noisy evolution with amplitude damping (T1 decay)
    gamma_t1 = 0.1
    ops_t1 = amplitude_damping_ops(gamma_t1)
    noisy_trajectory_t1 = evolve_lindblad(state, H, ops_t1, dt, steps)

    # Noisy evolution with phase damping (T2 dephasing)
    gamma_t2 = 0.15
    ops_t2 = phase_damping_ops(gamma_t2)
    noisy_trajectory_t2 = evolve_lindblad(state, H, ops_t2, dt, steps)

    # Compute geodesic deviations
    deviation_t1 = geodesic_deviation(ideal_trajectory, noisy_trajectory_t1)
    deviation_t2 = geodesic_deviation(ideal_trajectory, noisy_trajectory_t2)

    print(f"\n   Total evolution time: T = {total_time:.1f}")
    print(f"   Time steps: {steps}")

    print("\n2. Geodesic Deviation (Fidelity with Ideal):")
    print("   ─────────────────────────────────────────")
    print("   Time    Ideal    T1 Noise    T2 Dephasing")
    print("   ─────────────────────────────────────────")

    for i in [0, steps//4, steps//2, 3*steps//4, steps]:
        t = i * dt
        f_ideal = 1.0  # Fidelity with itself
        f_t1 = deviation_t1[i]
        f_t2 = deviation_t2[i]
        print(f"   {t:5.2f}   {f_ideal:.4f}   {f_t1:.4f}      {f_t2:.4f}")

    # Purity analysis
    print("\n3. Purity Decay (measures mixedness):")
    print("   ─────────────────────────────────────────")
    print("   Time    Ideal    T1 Noise    T2 Dephasing")
    print("   ─────────────────────────────────────────")

    purity_ideal = purity_decay(ideal_trajectory)
    purity_t1 = purity_decay(noisy_trajectory_t1)
    purity_t2 = purity_decay(noisy_trajectory_t2)

    for i in [0, steps//4, steps//2, 3*steps//4, steps]:
        t = i * dt
        print(f"   {t:5.2f}   {purity_ideal[i]:.4f}   {purity_t1[i]:.4f}      {purity_t2[i]:.4f}")

    # Bloch sphere visualization (text-based)
    print("\n4. Bloch Sphere Trajectory (x, y, z coordinates):")
    print("   ─────────────────────────────────────────────────────────")
    print("   Time    Ideal (x,y,z)        T1 Noisy (x,y,z)      r")
    print("   ─────────────────────────────────────────────────────────")

    for i in [0, steps//4, steps//2, 3*steps//4, steps]:
        t = i * dt
        ideal_bloch = ideal_trajectory[i].bloch_vector()
        noisy_bloch = noisy_trajectory_t1[i].bloch_vector()
        r_noisy = np.linalg.norm(noisy_bloch)

        print(f"   {t:5.2f}   ({ideal_bloch[0]:+.2f},{ideal_bloch[1]:+.2f},{ideal_bloch[2]:+.2f})   "
              f"({noisy_bloch[0]:+.2f},{noisy_bloch[1]:+.2f},{noisy_bloch[2]:+.2f})   {r_noisy:.3f}")

    print("""
    Geometric Insight:
    • Ideal evolution: Bloch vector rotates on sphere surface (r = 1)
    • T1 noise: Vector spirals toward ground state |0⟩ at north pole
    • T2 noise: Vector shrinks toward z-axis while rotating
    • The deviation from r = 1 measures distance from pure state manifold
    """)


def demo_noise_channels():
    """Demonstrate different noise channels and their effects."""
    separator("NOISE CHANNEL COMPARISON")

    print("""
    Standard Noise Models in Quantum Computing:
    ───────────────────────────────────────────
    • Amplitude Damping (T1): Energy relaxation, |1⟩ → |0⟩
    • Phase Damping (T2):     Pure dephasing, coherence loss
    • Depolarizing:           Uniform random Pauli errors
    • Thermal:                Coupling to heat bath
    """)

    # Compare steady states
    print("\n1. Steady States under Different Noise:")
    H = np.zeros((2, 2), dtype=np.complex128)  # No Hamiltonian

    channels = [
        ("Amplitude Damping", amplitude_damping_ops(0.5)),
        ("Phase Damping", phase_damping_ops(0.5)),
        ("Depolarizing", depolarizing_ops(0.5))
    ]

    for name, ops in channels:
        ss = steady_state(H, ops, tol=1e-8, dt=0.1)
        bloch = ss.bloch_vector()
        print(f"\n   {name}:")
        print(f"      Steady state purity: {ss.purity:.4f}")
        print(f"      Bloch vector: ({bloch[0]:.3f}, {bloch[1]:.3f}, {bloch[2]:.3f})")

        if name == "Amplitude Damping":
            print("      → Decays to ground state |0⟩")
        elif name == "Phase Damping":
            print("      → Decays to classical mixture on z-axis")
        else:
            print("      → Decays to maximally mixed I/2")

    # Decoherence rates
    print("\n2. Decoherence Rates from Different Initial States:")
    print("   ─────────────────────────────────────────────────")

    ops = depolarizing_ops(0.1)
    initial_states = [
        ("|0⟩", computational_basis(1, 0)),
        ("|1⟩", computational_basis(1, 1)),
        ("|+⟩", plus_state(1))
    ]

    for name, state in initial_states:
        rate = decoherence_rate(H, ops, state, dt=0.01, steps=100)
        print(f"   {name}: Γ ≈ {rate:.4f}")


def demo_entanglement_under_noise():
    """Demonstrate how entanglement decays under local noise."""
    separator("ENTANGLEMENT DECAY")

    print("""
    Entanglement is fragile: local noise on either qubit
    causes the global entanglement to decay.
    """)

    # Start with Bell state
    bell = bell_state(0)
    print(f"\n1. Initial State: |Φ+⟩ = (|00⟩ + |11⟩)/√2")
    print(f"   Initial concurrence: {concurrence(bell):.4f} (maximally entangled)")

    # Evolve under local amplitude damping
    H = np.zeros((4, 4), dtype=np.complex128)
    ops = amplitude_damping_ops(0.1, n_qubits=2)

    trajectory = evolve_lindblad(bell, H, ops, dt=0.1, steps=50)

    print("\n2. Entanglement Decay under Local Amplitude Damping:")
    print("   ─────────────────────────────────────")
    print("   Time    Concurrence    Purity")
    print("   ─────────────────────────────────────")

    for i in [0, 10, 20, 30, 40, 50]:
        t = i * 0.1
        c = concurrence(trajectory[i])
        p = trajectory[i].purity
        print(f"   {t:5.1f}   {c:.4f}         {p:.4f}")

    # ESD (Entanglement Sudden Death)
    print("""
    Note: Entanglement can reach zero in FINITE time
    (Entanglement Sudden Death), even though purity
    decays asymptotically. This is a purely quantum
    phenomenon with no classical analog.
    """)


def demo_natural_gradient_vqe():
    """Demonstrate natural gradient optimization for VQE."""
    separator("NATURAL GRADIENT VQE")

    print("""
    Natural Gradient Descent:
    ─────────────────────────
    • Standard gradient:  θ' = θ - η ∇E
    • Natural gradient:   θ' = θ - η F⁻¹ ∇E

    The QFI matrix F accounts for the geometry of the
    quantum state manifold, leading to faster convergence.
    """)

    # Simple 2-qubit problem: minimize ⟨Z⊗Z⟩
    print("\n1. VQE Problem: Minimize ⟨Z⊗Z⟩ with 2-qubit ansatz")

    H = np.kron(
        np.array([[1, 0], [0, -1]], dtype=np.complex128),
        np.array([[1, 0], [0, -1]], dtype=np.complex128)
    )  # Z⊗Z

    # Ground state of Z⊗Z is |01⟩ or |10⟩ with energy -1
    print("   Target: Ground state energy = -1")
    print("   Ground states: |01⟩, |10⟩")

    state_fn = make_vqe_state_fn(simple_ansatz, n_qubits=2)
    energy_fn = make_vqe_energy_fn(simple_ansatz, H, n_qubits=2)

    # Random starting point
    np.random.seed(42)
    initial_params = np.random.randn(2) * 0.5

    print(f"\n   Initial params: [{initial_params[0]:.3f}, {initial_params[1]:.3f}]")
    print(f"   Initial energy: {energy_fn(initial_params):.4f}")

    # Natural gradient optimization
    print("\n2. Optimization Progress (Natural Gradient):")
    print("   ──────────────────────────────────")

    result = natural_gradient_descent(
        state_fn, energy_fn, initial_params,
        learning_rate=0.3,
        max_iterations=30,
        verbose=False
    )

    print(f"   Iterations: {result.iterations}")
    print(f"   Converged: {result.converged}")
    print(f"   Final energy: {result.energies[-1]:.4f}")
    print(f"   Final params: [{result.params[0]:.3f}, {result.params[1]:.3f}]")

    # Show energy progression
    print("\n   Energy vs Iteration:")
    print("   Iter    Energy")
    print("   ─────────────────")
    for i in [0, 5, 10, 15, 20, min(25, len(result.energies)-1)]:
        if i < len(result.energies):
            print(f"   {i:4d}    {result.energies[i]:+.4f}")

    # Compare with vanilla gradient
    print("\n3. Comparison: Natural vs Vanilla Gradient")

    vanilla_result = vanilla_gradient_descent(
        energy_fn, initial_params,
        learning_rate=0.05,  # Smaller LR needed for stability
        max_iterations=30
    )

    print(f"   Natural gradient final energy: {result.energies[-1]:.4f}")
    print(f"   Vanilla gradient final energy: {vanilla_result.energies[-1]:.4f}")

    print("""
    Geometric Insight:
    • Natural gradient follows steepest descent on state manifold
    • Vanilla gradient can zig-zag due to coordinate artifacts
    • Near the poles of Bloch sphere, natural gradient adapts
      automatically to the changing metric
    """)


def demo_spectral_zeta():
    """Demonstrate spectral zeta functions and geodesic-spectral duality."""
    separator("SPECTRAL ZETA FUNCTIONS")

    print("""
    Spectral Zeta Functions:
    ────────────────────────
    • ζ_H(s) = Σₙ λₙ^{-s} for Hamiltonian eigenvalues λₙ
    • Heat kernel relation: ζ(s) = (1/Γ(s)) ∫ t^{s-1} Tr(e^{-tH}) dt
    • Connects quantum mechanics to NoeticEidos framework
    • Spectral signatures = fingerprints for quantum systems
    """)

    # Simple Hamiltonian example
    print("\n1. Spectral Zeta for Simple Hamiltonians:")

    # Diagonal Hamiltonian
    H1 = np.diag([1.0, 2.0, 4.0])
    zeta_1 = spectral_zeta_hamiltonian(H1, s=1.0)
    print(f"   H = diag(1, 2, 4)")
    print(f"   ζ_H(1) = 1/1 + 1/2 + 1/4 = {zeta_1:.4f}")

    # Shifted Pauli Z
    sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    H2 = sigma_z + 2*np.eye(2)  # Eigenvalues: 1, 3
    zeta_2 = spectral_zeta_hamiltonian(H2, s=1.0)
    print(f"\n   H = σ_z + 2I (eigenvalues: 1, 3)")
    print(f"   ζ_H(1) = 1/1 + 1/3 = {zeta_2:.4f}")

    # Heat kernel comparison
    print("\n2. Heat Kernel Trace (Mellin Partner):")
    print("   K(t) = Tr(e^{-tH}) = Σₙ e^{-tλₙ}")

    for t in [0.0, 0.5, 1.0, 2.0]:
        K = heat_kernel_trace(H1, t)
        print(f"   K({t:.1f}) = {K:.4f}")

    # Spectral signatures
    print("\n3. Spectral Signatures (Fingerprinting):")
    print("   Signatures uniquely identify spectra")

    s_values = np.linspace(0.5, 3.0, 10)

    H_a = np.diag([1.0, 2.0])
    H_b = np.diag([1.0, 3.0])
    H_c = np.diag([1.5, 2.5])

    sig_a = spectral_signature(H_a, s_values)
    sig_b = spectral_signature(H_b, s_values)
    sig_c = spectral_signature(H_c, s_values)

    d_ab = spectral_distance(sig_a, sig_b)
    d_ac = spectral_distance(sig_a, sig_c)
    d_bc = spectral_distance(sig_b, sig_c)

    print(f"\n   H_a = diag(1, 2)")
    print(f"   H_b = diag(1, 3)")
    print(f"   H_c = diag(1.5, 2.5)")
    print(f"\n   Spectral distances:")
    print(f"   d(H_a, H_b) = {d_ab:.4f}")
    print(f"   d(H_a, H_c) = {d_ac:.4f}")
    print(f"   d(H_b, H_c) = {d_bc:.4f}")

    # Lindbladian spectral analysis
    print("\n4. Lindbladian Spectral Analysis:")
    print("   Non-Hermitian spectrum encodes decay and oscillation")

    H = sigma_z
    ops = amplitude_damping_ops(gamma=0.3)
    decomp = decompose_lindbladian_spectrum(H, ops)

    print(f"\n   System: σ_z with amplitude damping (γ=0.3)")
    print(f"   Decay rates: {decomp.decay_rates.round(4)}")
    print(f"   Oscillation frequencies: {decomp.oscillation_freqs.round(4)}")
    print(f"   Spectral gap: {decomp.gap:.4f}")
    print(f"   (Gap determines timescale to reach steady state)")

    # Geodesic-spectral connection
    print("\n5. Geodesic-Spectral Duality:")
    print("   Spectral difference ↔ Geodesic deviation")

    H_rot = sigma_z + 2*np.eye(2)
    s_range = np.linspace(0.5, 2.0, 10)

    # Compare ideal vs noisy
    deviation, sig_ideal, sig_noisy = spectral_geodesic_deviation(
        H_rot, amplitude_damping_ops(0.2), s_range
    )

    print(f"\n   Hamiltonian: σ_z + 2I")
    print(f"   Noise: Amplitude damping (γ=0.2)")
    print(f"   Spectral deviation: {deviation:.4f}")

    # Show how deviation increases with noise
    print("\n   Spectral deviation vs noise strength:")
    print("   γ        Deviation")
    print("   ─────────────────")
    for gamma in [0.05, 0.1, 0.2, 0.3, 0.5]:
        dev, _, _ = spectral_geodesic_deviation(
            H_rot, amplitude_damping_ops(gamma), s_range
        )
        print(f"   {gamma:.2f}     {dev:.4f}")

    print("""
    Geodesic-Spectral Insight:
    ──────────────────────────
    • Ideal evolution (unitary): characterized by ζ_H poles
    • Noisy evolution (Lindblad): characterized by ζ_L poles
    • The spectral distance measures how noise deforms the spectrum
    • This connects state-space geodesic deviation to operator-space
      spectral structure - the bridge to NoeticEidos framework
    """)


def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║        QUANTUM SIMULATOR - NOETIC EIDOS INSPIRED         ║
    ║                                                          ║
    ║  A functional quantum computer simulator with:           ║
    ║  • Pure and mixed state representation                   ║
    ║  • Universal gate set                                    ║
    ║  • Circuit construction and execution                    ║
    ║  • Standard quantum algorithms                           ║
    ║  • Lindblad dynamics & geodesic deviation model          ║
    ║  • Natural gradient optimization (VQE)                   ║
    ║  • Spectral zeta functions & geodesic-spectral duality   ║
    ║  • Information geometry interpretation                   ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    demo_basic_states()
    demo_measurement()
    demo_gates()
    demo_circuits()
    demo_deutsch_jozsa()
    demo_bernstein_vazirani()
    demo_grover()
    demo_qft()
    demo_information_geometry()
    demo_algorithm_comparison()

    # New demos for geodesic deviation and natural gradient
    demo_geodesic_deviation()
    demo_noise_channels()
    demo_entanglement_under_noise()
    demo_natural_gradient_vqe()
    demo_spectral_zeta()

    print("\n" + "="*60)
    print("  ALL DEMOS COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main()
