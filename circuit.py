"""
Quantum Circuit

A circuit is a sequence of gate operations that transforms quantum states.
This module provides the Circuit class for building, analyzing, and executing
quantum algorithms.

Design Philosophy:
- Immutable operation sequence (append returns new circuit)
- Lazy execution (state only computed when needed)
- Support for classical control and measurement
- Geometric interpretation hooks (for Noetic Eidos integration)
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import copy

# Import from our modules
from state import QuantumState, computational_basis
from gates import (
    Gate, apply_gate, get_gate, GATE_LIBRARY,
    H, X, Y, Z, CX, T, S, Rx, Ry, Rz, CCX
)


class OperationType(Enum):
    """Types of operations in a circuit."""
    GATE = "gate"
    MEASUREMENT = "measurement"
    BARRIER = "barrier"
    RESET = "reset"
    CONDITIONAL = "conditional"


@dataclass
class Operation:
    """
    A single operation in a quantum circuit.
    
    Attributes:
        op_type: What kind of operation
        gate: The gate (if GATE type)
        qubits: Which qubits are involved
        classical_bits: Which classical bits (for measurement)
        condition: Classical condition for conditional gates
    """
    op_type: OperationType
    qubits: List[int]
    gate: Optional[Gate] = None
    classical_bits: Optional[List[int]] = None
    condition: Optional[Tuple[int, int]] = None  # (bit_index, expected_value)
    
    def __repr__(self) -> str:
        if self.op_type == OperationType.GATE:
            return f"{self.gate.name}({self.qubits})"
        elif self.op_type == OperationType.MEASUREMENT:
            return f"M({self.qubits} -> {self.classical_bits})"
        elif self.op_type == OperationType.BARRIER:
            return f"BARRIER({self.qubits})"
        elif self.op_type == OperationType.RESET:
            return f"RESET({self.qubits})"
        return f"{self.op_type.value}({self.qubits})"


@dataclass
class CircuitResult:
    """
    Result of executing a quantum circuit.
    
    Attributes:
        final_state: The quantum state after execution
        measurements: Dict mapping classical bit indices to measured values
        measurement_history: List of all measurements in order
        shots: Number of times circuit was run (for sampling)
        counts: Measurement outcome counts (for multi-shot execution)
    """
    final_state: QuantumState
    measurements: Dict[int, int] = field(default_factory=dict)
    measurement_history: List[Tuple[int, int]] = field(default_factory=list)
    shots: int = 1
    counts: Optional[Dict[str, int]] = None
    
    def get_probabilities(self) -> Dict[str, float]:
        """Convert counts to probability distribution."""
        if self.counts is None:
            return {}
        total = sum(self.counts.values())
        return {k: v/total for k, v in self.counts.items()}


class Circuit:
    """
    Quantum circuit: a composable sequence of operations.
    
    Usage:
        qc = Circuit(2)  # 2-qubit circuit
        qc.h(0)          # Hadamard on qubit 0
        qc.cx(0, 1)      # CNOT from 0 to 1
        result = qc.run()
    """
    
    def __init__(self, n_qubits: int, n_classical: Optional[int] = None):
        """
        Initialize a quantum circuit.
        
        Args:
            n_qubits: Number of qubits
            n_classical: Number of classical bits (defaults to n_qubits)
        """
        self.n_qubits = n_qubits
        self.n_classical = n_classical if n_classical is not None else n_qubits
        self.operations: List[Operation] = []
        self._depth_cache = None
    
    # ========================================================================
    # Gate Application Methods
    # ========================================================================
    
    def _add_gate(self, gate: Gate, qubits: List[int], 
                  condition: Optional[Tuple[int, int]] = None) -> 'Circuit':
        """Internal method to add a gate operation."""
        for q in qubits:
            if q < 0 or q >= self.n_qubits:
                raise ValueError(f"Qubit {q} out of range [0, {self.n_qubits})")
        
        op = Operation(
            op_type=OperationType.GATE,
            gate=gate,
            qubits=qubits,
            condition=condition
        )
        self.operations.append(op)
        self._depth_cache = None
        return self
    
    # Single-qubit gates
    def i(self, qubit: int) -> 'Circuit':
        """Identity gate."""
        return self._add_gate(get_gate('I'), [qubit])
    
    def x(self, qubit: int) -> 'Circuit':
        """Pauli-X gate."""
        return self._add_gate(get_gate('X'), [qubit])
    
    def y(self, qubit: int) -> 'Circuit':
        """Pauli-Y gate."""
        return self._add_gate(get_gate('Y'), [qubit])
    
    def z(self, qubit: int) -> 'Circuit':
        """Pauli-Z gate."""
        return self._add_gate(get_gate('Z'), [qubit])
    
    def h(self, qubit: int) -> 'Circuit':
        """Hadamard gate."""
        return self._add_gate(get_gate('H'), [qubit])
    
    def s(self, qubit: int) -> 'Circuit':
        """S gate (√Z)."""
        return self._add_gate(get_gate('S'), [qubit])
    
    def sdg(self, qubit: int) -> 'Circuit':
        """S-dagger gate."""
        return self._add_gate(get_gate('SDG'), [qubit])
    
    def t(self, qubit: int) -> 'Circuit':
        """T gate (√S)."""
        return self._add_gate(get_gate('T'), [qubit])
    
    def tdg(self, qubit: int) -> 'Circuit':
        """T-dagger gate."""
        return self._add_gate(get_gate('TDG'), [qubit])
    
    def rx(self, theta: float, qubit: int) -> 'Circuit':
        """X-rotation gate."""
        return self._add_gate(get_gate('RX', theta), [qubit])
    
    def ry(self, theta: float, qubit: int) -> 'Circuit':
        """Y-rotation gate."""
        return self._add_gate(get_gate('RY', theta), [qubit])
    
    def rz(self, theta: float, qubit: int) -> 'Circuit':
        """Z-rotation gate."""
        return self._add_gate(get_gate('RZ', theta), [qubit])
    
    def p(self, phi: float, qubit: int) -> 'Circuit':
        """Phase gate."""
        return self._add_gate(get_gate('P', phi), [qubit])
    
    def u3(self, theta: float, phi: float, lam: float, qubit: int) -> 'Circuit':
        """General single-qubit gate."""
        return self._add_gate(get_gate('U3', theta, phi, lam), [qubit])
    
    # Two-qubit gates
    def cx(self, control: int, target: int) -> 'Circuit':
        """CNOT gate."""
        return self._add_gate(get_gate('CX'), [control, target])
    
    def cnot(self, control: int, target: int) -> 'Circuit':
        """Alias for cx."""
        return self.cx(control, target)
    
    def cy(self, control: int, target: int) -> 'Circuit':
        """Controlled-Y gate."""
        return self._add_gate(get_gate('CY'), [control, target])
    
    def cz(self, control: int, target: int) -> 'Circuit':
        """Controlled-Z gate."""
        return self._add_gate(get_gate('CZ'), [control, target])
    
    def swap(self, qubit1: int, qubit2: int) -> 'Circuit':
        """SWAP gate."""
        return self._add_gate(get_gate('SWAP'), [qubit1, qubit2])
    
    def crx(self, theta: float, control: int, target: int) -> 'Circuit':
        """Controlled X-rotation."""
        return self._add_gate(get_gate('CRX', theta), [control, target])
    
    def cry(self, theta: float, control: int, target: int) -> 'Circuit':
        """Controlled Y-rotation."""
        return self._add_gate(get_gate('CRY', theta), [control, target])
    
    def crz(self, theta: float, control: int, target: int) -> 'Circuit':
        """Controlled Z-rotation."""
        return self._add_gate(get_gate('CRZ', theta), [control, target])
    
    def cp(self, phi: float, control: int, target: int) -> 'Circuit':
        """Controlled phase gate."""
        return self._add_gate(get_gate('CP', phi), [control, target])
    
    # Three-qubit gates
    def ccx(self, control1: int, control2: int, target: int) -> 'Circuit':
        """Toffoli (CCNOT) gate."""
        return self._add_gate(get_gate('CCX'), [control1, control2, target])
    
    def toffoli(self, control1: int, control2: int, target: int) -> 'Circuit':
        """Alias for ccx."""
        return self.ccx(control1, control2, target)
    
    def ccz(self, q0: int, q1: int, q2: int) -> 'Circuit':
        """CCZ gate."""
        return self._add_gate(get_gate('CCZ'), [q0, q1, q2])
    
    def cswap(self, control: int, target1: int, target2: int) -> 'Circuit':
        """Fredkin (controlled-SWAP) gate."""
        return self._add_gate(get_gate('CSWAP'), [control, target1, target2])
    
    # Generic gate application
    def apply(self, gate: Gate, qubits: List[int]) -> 'Circuit':
        """Apply an arbitrary gate."""
        return self._add_gate(gate, qubits)
    
    # ========================================================================
    # Non-Gate Operations
    # ========================================================================
    
    def measure(self, qubit: int, classical_bit: Optional[int] = None) -> 'Circuit':
        """
        Add a measurement operation.
        
        Args:
            qubit: Which qubit to measure
            classical_bit: Which classical bit to store result (defaults to qubit index)
        """
        if classical_bit is None:
            classical_bit = qubit
        
        if classical_bit >= self.n_classical:
            raise ValueError(f"Classical bit {classical_bit} out of range")
        
        op = Operation(
            op_type=OperationType.MEASUREMENT,
            qubits=[qubit],
            classical_bits=[classical_bit]
        )
        self.operations.append(op)
        return self
    
    def measure_all(self) -> 'Circuit':
        """Measure all qubits into corresponding classical bits."""
        for i in range(self.n_qubits):
            self.measure(i, i)
        return self
    
    def barrier(self, qubits: Optional[List[int]] = None) -> 'Circuit':
        """
        Add a barrier (for visualization/optimization boundary).
        
        Barriers don't affect computation but prevent gate reordering.
        """
        if qubits is None:
            qubits = list(range(self.n_qubits))
        
        op = Operation(
            op_type=OperationType.BARRIER,
            qubits=qubits
        )
        self.operations.append(op)
        return self
    
    def reset(self, qubit: int) -> 'Circuit':
        """Reset a qubit to |0⟩ state."""
        op = Operation(
            op_type=OperationType.RESET,
            qubits=[qubit]
        )
        self.operations.append(op)
        return self
    
    # ========================================================================
    # Execution
    # ========================================================================
    
    def run(self, initial_state: Optional[QuantumState] = None,
            shots: int = 1) -> CircuitResult:
        """
        Execute the circuit.
        
        Args:
            initial_state: Starting state (defaults to |00...0⟩)
            shots: Number of times to run (for sampling)
            
        Returns:
            CircuitResult with final state and measurements
        """
        if shots == 1:
            return self._run_single(initial_state)
        else:
            return self._run_shots(initial_state, shots)
    
    def _run_single(self, initial_state: Optional[QuantumState] = None) -> CircuitResult:
        """Execute circuit once."""
        if initial_state is None:
            state = computational_basis(self.n_qubits, 0)
        else:
            state = initial_state.copy()
        
        measurements = {}
        measurement_history = []
        classical_register = [0] * self.n_classical
        
        for op in self.operations:
            if op.op_type == OperationType.GATE:
                # Check condition
                if op.condition is not None:
                    bit_idx, expected = op.condition
                    if classical_register[bit_idx] != expected:
                        continue  # Skip this gate
                
                # Apply the gate
                new_amplitudes = apply_gate(
                    op.gate, state.amplitudes, op.qubits, self.n_qubits
                )
                state = QuantumState(self.n_qubits, amplitudes=new_amplitudes)
            
            elif op.op_type == OperationType.MEASUREMENT:
                qubit = op.qubits[0]
                classical_bit = op.classical_bits[0]
                
                outcome, state = state.measure_qubit(qubit)
                classical_register[classical_bit] = outcome
                measurements[classical_bit] = outcome
                measurement_history.append((qubit, outcome))
            
            elif op.op_type == OperationType.RESET:
                qubit = op.qubits[0]
                # Measure and flip if needed
                outcome, state = state.measure_qubit(qubit)
                if outcome == 1:
                    new_amplitudes = apply_gate(
                        get_gate('X'), state.amplitudes, [qubit], self.n_qubits
                    )
                    state = QuantumState(self.n_qubits, amplitudes=new_amplitudes)
            
            elif op.op_type == OperationType.BARRIER:
                pass  # No effect on execution
        
        return CircuitResult(
            final_state=state,
            measurements=measurements,
            measurement_history=measurement_history,
            shots=1
        )
    
    def _run_shots(self, initial_state: Optional[QuantumState], 
                   shots: int) -> CircuitResult:
        """Execute circuit multiple times for sampling."""
        counts = {}
        last_result = None
        
        for _ in range(shots):
            result = self._run_single(initial_state)
            last_result = result
            
            # Convert measurements to bitstring
            bitstring = ''.join(
                str(result.measurements.get(i, 0)) 
                for i in range(self.n_classical)
            )
            counts[bitstring] = counts.get(bitstring, 0) + 1
        
        return CircuitResult(
            final_state=last_result.final_state,
            measurements=last_result.measurements,
            measurement_history=last_result.measurement_history,
            shots=shots,
            counts=counts
        )
    
    def statevector(self, initial_state: Optional[QuantumState] = None) -> np.ndarray:
        """Get the final statevector (no measurements)."""
        # Create a copy without measurements
        temp_circuit = Circuit(self.n_qubits, self.n_classical)
        for op in self.operations:
            if op.op_type != OperationType.MEASUREMENT:
                temp_circuit.operations.append(op)
        
        result = temp_circuit.run(initial_state)
        return result.final_state.amplitudes
    
    def unitary(self) -> np.ndarray:
        """
        Compute the unitary matrix for the circuit.
        
        Only works for circuits without measurements/resets.
        """
        dim = 2 ** self.n_qubits
        U = np.eye(dim, dtype=np.complex128)
        
        for op in self.operations:
            if op.op_type == OperationType.GATE:
                gate_full = np.eye(dim, dtype=np.complex128)
                from gates.gates import _expand_gate
                gate_full = _expand_gate(op.gate.matrix, op.qubits, self.n_qubits)
                U = gate_full @ U
            elif op.op_type in [OperationType.MEASUREMENT, OperationType.RESET]:
                raise ValueError("Cannot compute unitary for circuit with measurements/resets")
        
        return U
    
    # ========================================================================
    # Circuit Analysis
    # ========================================================================
    
    @property
    def depth(self) -> int:
        """
        Compute circuit depth (longest path through the circuit).
        """
        if self._depth_cache is not None:
            return self._depth_cache
        
        # Track when each qubit becomes available
        qubit_depth = [0] * self.n_qubits
        
        for op in self.operations:
            if op.op_type in [OperationType.GATE, OperationType.MEASUREMENT]:
                # This operation must wait for all involved qubits
                max_depth = max(qubit_depth[q] for q in op.qubits)
                new_depth = max_depth + 1
                for q in op.qubits:
                    qubit_depth[q] = new_depth
        
        self._depth_cache = max(qubit_depth) if qubit_depth else 0
        return self._depth_cache
    
    @property
    def gate_count(self) -> int:
        """Total number of gates."""
        return sum(1 for op in self.operations if op.op_type == OperationType.GATE)
    
    def gate_counts(self) -> Dict[str, int]:
        """Count of each gate type."""
        counts = {}
        for op in self.operations:
            if op.op_type == OperationType.GATE:
                name = op.gate.name
                counts[name] = counts.get(name, 0) + 1
        return counts
    
    # ========================================================================
    # Circuit Composition
    # ========================================================================
    
    def compose(self, other: 'Circuit', qubits: Optional[List[int]] = None) -> 'Circuit':
        """
        Append another circuit to this one.
        
        Args:
            other: Circuit to append
            qubits: Which qubits to map other's qubits to (defaults to 0,1,2,...)
        """
        if qubits is None:
            qubits = list(range(other.n_qubits))
        
        if len(qubits) != other.n_qubits:
            raise ValueError("Qubit mapping length mismatch")
        
        for op in other.operations:
            mapped_qubits = [qubits[q] for q in op.qubits]
            new_op = Operation(
                op_type=op.op_type,
                gate=op.gate,
                qubits=mapped_qubits,
                classical_bits=op.classical_bits,
                condition=op.condition
            )
            self.operations.append(new_op)
        
        self._depth_cache = None
        return self
    
    def inverse(self) -> 'Circuit':
        """Return the inverse circuit (adjoint of all gates, reversed order)."""
        inv = Circuit(self.n_qubits, self.n_classical)
        
        for op in reversed(self.operations):
            if op.op_type == OperationType.GATE:
                inv._add_gate(op.gate.adjoint, op.qubits)
            elif op.op_type in [OperationType.MEASUREMENT, OperationType.RESET]:
                raise ValueError("Cannot invert circuit with measurements/resets")
        
        return inv
    
    def copy(self) -> 'Circuit':
        """Create a deep copy of the circuit."""
        new_circuit = Circuit(self.n_qubits, self.n_classical)
        new_circuit.operations = copy.deepcopy(self.operations)
        return new_circuit
    
    # ========================================================================
    # Visualization
    # ========================================================================
    
    def draw(self, output: str = 'text') -> str:
        """
        Draw the circuit.
        
        Args:
            output: 'text' for ASCII art
        """
        if output == 'text':
            return self._draw_text()
        raise ValueError(f"Unknown output format: {output}")
    
    def _draw_text(self) -> str:
        """ASCII art circuit diagram."""
        lines = []
        
        # Initialize qubit lines
        qubit_lines = [f"q{i}: " for i in range(self.n_qubits)]
        
        for op in self.operations:
            if op.op_type == OperationType.GATE:
                name = op.gate.name
                qubits = op.qubits
                
                if len(qubits) == 1:
                    # Single qubit gate
                    q = qubits[0]
                    gate_str = f"─[{name}]─"
                    qubit_lines[q] += gate_str
                    
                    # Pad other lines
                    for i in range(self.n_qubits):
                        if i != q:
                            qubit_lines[i] += "─" * len(gate_str)
                
                elif len(qubits) == 2:
                    # Two qubit gate
                    control, target = qubits[0], qubits[1]
                    min_q, max_q = min(control, target), max(control, target)
                    
                    gate_width = max(len(name) + 2, 5)
                    
                    for i in range(self.n_qubits):
                        if i == control:
                            if name in ['CX', 'CNOT', 'CY', 'CZ', 'CRX', 'CRY', 'CRZ', 'CP']:
                                qubit_lines[i] += "─●─" + "─" * (gate_width - 3)
                            else:
                                qubit_lines[i] += f"─[{name}]─"[:gate_width]
                        elif i == target:
                            if name in ['CX', 'CNOT']:
                                qubit_lines[i] += "─⊕─" + "─" * (gate_width - 3)
                            elif name in ['CZ']:
                                qubit_lines[i] += "─●─" + "─" * (gate_width - 3)
                            else:
                                qubit_lines[i] += f"─[{name[1:] if name.startswith('C') else name}]─"[:gate_width]
                        elif min_q < i < max_q:
                            qubit_lines[i] += "─│─" + "─" * (gate_width - 3)
                        else:
                            qubit_lines[i] += "─" * gate_width
                
                else:
                    # Multi-qubit gate
                    gate_width = len(name) + 4
                    min_q, max_q = min(qubits), max(qubits)
                    
                    for i in range(self.n_qubits):
                        if i in qubits:
                            if i == qubits[-1]:
                                qubit_lines[i] += f"─[{name}]─"
                            else:
                                qubit_lines[i] += "─●─" + "─" * (gate_width - 3)
                        elif min_q < i < max_q:
                            qubit_lines[i] += "─│─" + "─" * (gate_width - 3)
                        else:
                            qubit_lines[i] += "─" * gate_width
            
            elif op.op_type == OperationType.MEASUREMENT:
                q = op.qubits[0]
                qubit_lines[q] += "─[M]─"
                for i in range(self.n_qubits):
                    if i != q:
                        qubit_lines[i] += "─────"
            
            elif op.op_type == OperationType.BARRIER:
                for i in range(self.n_qubits):
                    qubit_lines[i] += "─║─"
            
            elif op.op_type == OperationType.RESET:
                q = op.qubits[0]
                qubit_lines[q] += "─|0⟩─"
                for i in range(self.n_qubits):
                    if i != q:
                        qubit_lines[i] += "─────"
        
        # Equalize line lengths
        max_len = max(len(line) for line in qubit_lines)
        qubit_lines = [line.ljust(max_len, '─') for line in qubit_lines]
        
        return '\n'.join(qubit_lines)
    
    def __repr__(self) -> str:
        return f"Circuit(qubits={self.n_qubits}, depth={self.depth}, gates={self.gate_count})"
    
    def __str__(self) -> str:
        return self.draw()
