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
    
    print("\n" + "="*60)
    print("  ALL DEMOS COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main()
