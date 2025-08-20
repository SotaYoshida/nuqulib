import pytest
import pennylane as qml
from nuqulib import *

def test_pairingHamiltonian():
    Norb = 4
    Nocc = 2
    gval = 0.33  

    Hamil = PairingHamiltonian(Norb, Nocc, gval)
    evals, evecs = np.linalg.eigh(Hamil.Hmat)
    evals = np.linalg.eigvalsh(Hamil.Hmat)
    Egs_exact = evals[0]
    E_HF = Hamil.Hmat[0,0]

    print("basis:", Hamil.basis)
    print([tuple_to_bitstring(tup, Norb) for tup in Hamil.basis])
    print("eps: ", Hamil.epsilon)
    print("Hmat: ", Hamil.Hmat)
    print("evals: ", evals)
    print("Egs_exact: ", Egs_exact, " E_HF", E_HF)
    print("gs evec", evecs[:,0])
    print("gs prob", evecs[:,0]**2)

    assert abs(Egs_exact-1.18985) < 1e-5


if __name__ == "__main__":    
    test_pairingHamiltonian()
    print("Test passed successfully.")
