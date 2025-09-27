import os
import pytest
from nuqulib import *

chdir = os.path.dirname(os.path.abspath(__file__))
int_dir = os.path.join(chdir, "interaction_file/")

def test_ODMD():
    using_statevector = True
    sampler = backend = None

    Z = 0; N = 2
    fn_snt = int_dir+"ckpot.snt"
  
    hamil = Hamiltonian(fn_snt, Z, N)
    n_qubits = hamil.n_qubits
    proton_qubits = list(range(0, hamil.n_qubits_p))
    neutron_qubits = list(range(hamil.n_qubits_p, n_qubits))

    Hdict_M = hamil.get_mscheme_H(opform=True)
    H_1b_p, H_1b_n, H_jz_p, H_jz_n, H_pp, H_nn, H_pn, H_3b = hamil.mapping_opform(Hdict_M, "JordanWigner")
    H_mapped = H_1b_p + H_1b_n + H_nn 

    print(f"H_1b {H_1b_n+H_1b_p}")
    print(f"H_1b_p {H_1b_p}")
    print(f"H_1b_n {H_1b_n}")
    print(f"H_nn {H_nn}")

    delta_t = 0.01234
    trotter_rank = 1; trotter_steps = 15
    #trotter_rank = 2; trotter_steps = 3
    max_iterations = 20

    dummy_params = [ ]
    #U_prep = nucl_ansatz(Hdict_M, n_qubits, proton_qubits, neutron_qubits, Z, N, dummy_params, method="HF")
    U_prep = QuantumCircuit(n_qubits)
    U_prep.x([9, 10])

    ancilla_qubits=[0]
    target_qubits=list(range(1,n_qubits+1))

    E0 = ODMD(U_prep, H_mapped, delta_t, max_iterations, 
              trotter_rank, trotter_steps, 
              sampler, backend, 
              ancilla_qubits, target_qubits,
              using_statevector=using_statevector, d=10,
              plot_lambda=False)
    Eexact = -3.910

    assert ( 100*abs(E0 - Eexact)/abs(Eexact) < 5), f"ODMD test failed: {E0} != {Eexact} within 5% tolerance"