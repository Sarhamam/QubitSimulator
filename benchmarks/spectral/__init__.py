"""
Spectral Analysis Benchmarks

Demonstrates unique spectral analysis capabilities unavailable
in standard quantum simulators like Qiskit or Cirq.

Benchmarks:
- spectral_fingerprints: System identification via zeta functions
- spectral_gap_relaxation: Spectral gap → relaxation time validation
- signature_robustness: Perturbation stability of spectral signatures
- chaos_detection: Integrable vs chaotic via spectral form factor
"""

from .spectral_fingerprints import run_spectral_fingerprints_benchmark
from .spectral_gap_relaxation import run_spectral_gap_benchmark
from .signature_robustness import run_signature_robustness_benchmark
from .chaos_detection import run_chaos_detection_benchmark

__all__ = [
    'run_spectral_fingerprints_benchmark',
    'run_spectral_gap_benchmark',
    'run_signature_robustness_benchmark',
    'run_chaos_detection_benchmark',
]
