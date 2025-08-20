import os
import pytest
import pennylane as qml
import pennylane.numpy as qnp
from nuqulib import *

chdir = os.path.dirname(os.path.abspath(__file__))
int_dir = os.path.join(chdir, "interaction_file/")

def test_vqe_pshell():
    filename_snt = int_dir + "ckpot.snt"
    Z = 2
    N = 4
    hamil = Hamiltonian(filename_snt, Z, N, verbose=False)

    print(hamil.single_particle_states)
    Hdict = hamil.get_mscheme_H()            

    n_qubits_p = hamil.n_qubits_p
    n_qubits_n = hamil.n_qubits_n
    proton_number = 0
    neutron_number = 2
    params, Emin = vqe_example_pennylane(Hdict, proton_number, neutron_number, n_qubits_p, n_qubits_n, using_chs=["1b", "nn"])
    assert abs(Emin - (-3.90981)) < 1.e-4

if __name__ == "__main__":    
    test_vqe_pshell()