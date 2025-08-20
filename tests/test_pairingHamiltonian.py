import pytest
import pennylane as qml
from nuqulib import *

def test_pairingHamiltonian():
    Norb = 4
    Nocc = 2
    gval = 0.33  

    Hamil = PairingHamiltonian(Norb, Nocc, gval)
    evals = np.linalg.eigvalsh(Hamil.Hmat)
    Egs_exact = evals[0]

    print("basis:", Hamil.basis)
    print([tuple_to_bitstring(tup, Norb) for tup in Hamil.basis])
    print("eps: ", Hamil.epsilon)
    print("Hmat: ", Hamil.Hmat)
    print("evals: ", evals)
    print("Egs_exact: ", Egs_exact)

    assert abs(Egs_exact-1.18985) < 1e-5

def test_ansatz_pairing():
    Nq = 4
    Nocc = 2
    Hamil = PairingHamiltonian(Nq, Nocc, 0.33)
    hamiltonian_op = Hamil.encoding()
    evals = np.linalg.eigvalsh(Hamil.Hmat)
    Egs_exact = evals[0]

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
    assert abs(Emin_NFT - Egs_exact) < 1e-5

if __name__ == "__main__": 
    # Check the PairingHamiltonian class
    test_pairingHamiltonian()

    # Check the pairing ansatz and optimization via NFT method
    test_ansatz_pairing()
