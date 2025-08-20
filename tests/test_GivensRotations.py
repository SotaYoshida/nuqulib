from itertools import combinations
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from nuqulib import *


def test_GivensRotations():
    theta = 2 * (np.pi / 4)

    # Check different implementations
    state_vectors = [ ]
    for method in ["magic", "Xanadu", "iSWAP_Rz"]:
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.append(G_gate(theta, method=method), [0, 1])
        state_vector = Statevector.from_instruction(qc)
        state_vectors.append(state_vector)
    print(state_vectors)    
    for i, j in combinations(range(len(state_vectors)), 2):
        sv_i = state_vectors[i]
        sv_j = state_vectors[j]
        print(f"Comparing method {i} with method {j}:")
        np.linalg.norm(sv_i - sv_j)
        print(f"Norm difference: {np.linalg.norm(sv_i - sv_j)}")

    # Check different implementations
    state_vectors = [ ]
    for method in ["magic", "Xanadu", "iSWAP_Rz"]:
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.append(G_gate(theta, method=method), [0, 1])
        state_vector = Statevector.from_instruction(qc)
        state_vectors.append(state_vector)
    for i, j in combinations(range(len(state_vectors)), 2):
        sv_i = state_vectors[i]
        # Since global phase can be arbitrary, we normalize the state vectors to compare
        if np.real(sv_i[1]) < 0:
            sv_i = -sv_i
        sv_j = state_vectors[j]
        if np.real(sv_j[1]) < 0:
            sv_j = -sv_j
        print(f"Comparing method {i} with method {j}:")
        tnorm = np.linalg.norm(sv_i - sv_j)
        assert tnorm < 1e-10, f"Norm difference {tnorm} is too large for methods {i} and {j}"

if __name__ == "__main__":    
    # Check the simple Givens rotation gates
    test_GivensRotations()