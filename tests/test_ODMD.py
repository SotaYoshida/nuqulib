import numpy as np
import pytest
import pennylane as qml
from qiskit_aer.primitives import SamplerV2
from qiskit.quantum_info import Statevector
from nuqulib import *

def test_ansatz_pairing():
    Nq = 4
    Nocc = 2
    Hamil = PairingHamiltonian(Nq, Nocc, 0.33)
    hamiltonian_op = Hamil.encoding()
    evals = np.linalg.eigvalsh(Hamil.Hmat)
    Egs_exact = evals[0]
    print("evals: ", evals)
    # Create a dummy parameters array
    params = np.zeros(100, dtype=float)

    # Create a pairing ansatz circuit and optimize it via Nakanishi-Fujii-Todo (NFT) method
    method_ansatz = "HF+Givens"
    qc, where_is_G_or_cG1 = pair_ansatz_qiskit(params, Nq, Nocc, method=method_ansatz,
                            return_Gdict=True)
    method_measure = "statevector"
    it_max = 10
    ngate = len(where_is_G_or_cG1.keys()) 
    params_NFT, Emin_NFT = optimize_params_with_NFT(it_max, hamiltonian_op, params, 
                                                    Nq, Nocc, ngate, where_is_G_or_cG1,
                                                    method_ansatz, method_measure)

    # Hadamard-test
    Uprep = pair_ansatz_qiskit(params_NFT, Nq, Nocc, method=method_ansatz)

    delta_t = 0.0123
    max_iterations = 30
    trotter_steps = 20

    E_ODMD = ODMD(Uprep, hamiltonian_op, delta_t, max_iterations,
                  trotter_steps, sampler=None, backend=None,
                  ancilla_qubits=[0], target_qubits=list(range(1, Nq + 1)),
                  d=10
    )
