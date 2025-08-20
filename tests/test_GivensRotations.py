"""
Optimized tests for Givens rotation gates.

This file contains streamlined tests for different Givens rotation implementations.
Duplicate test loops have been combined for efficiency.
"""
from itertools import combinations
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from nuqulib import *


def test_GivensRotations():
    theta = 2 * (np.pi / 4)

    # Check different implementations - optimized version combining both test cases
    state_vectors = [ ]
    for method in ["magic", "Xanadu", "iSWAP_Rz"]:
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.append(G_gate(theta, method=method), [0, 1])
        state_vector = Statevector.from_instruction(qc)
        state_vectors.append(state_vector)
    
    # Test all pairwise comparisons with phase normalization
    for i, j in combinations(range(len(state_vectors)), 2):
        sv_i = state_vectors[i]
        sv_j = state_vectors[j]
        
        # Normalize for global phase differences
        if np.real(sv_i[1]) < 0:
            sv_i = -sv_i
        if np.real(sv_j[1]) < 0:
            sv_j = -sv_j
        
        tnorm = np.linalg.norm(sv_i - sv_j)
        assert tnorm < 1e-10, f"Norm difference {tnorm} is too large for methods {i} and {j}"

if __name__ == "__main__":    
    # Check the optimized Givens rotation gates (duplicate test loops removed)
    print("Testing Givens rotation implementations...")
    test_GivensRotations()
    print("Givens rotation tests completed successfully!")