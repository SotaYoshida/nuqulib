import os
from nuqulib import *

chdir = os.path.dirname(os.path.abspath(__file__))
int_dir = os.path.join(chdir, "interaction_file/")

def test_ODMD():
    using_statevector = True
    sampler = None

    Z = 2; N = 4
    Eexact = - 3.910

    fn_snt = int_dir+"ckpot.snt"
  
    hamil = Hamiltonian(fn_snt, Z, N)
    n_qubits = hamil.n_qubits

    hamil.get_mscheme_H(opform=True)
    H_1b_p, H_1b_n, H_jz_p, H_jz_n, H_pp, H_nn, H_pn, H_3b = hamil.mapping_opform("JordanWigner")
    H_mapped = H_1b_p + H_1b_n + H_nn 

    delta_t = 0.01234
    trotter_rank = 2; trotter_steps = 7
    max_iterations = 15

    U_prep = QuantumCircuit(n_qubits)
    U_prep.x([9, 10])

    ancilla_qubits=[0]
    target_qubits=list(range(1,n_qubits+1))

    odmd = ODMD(U_prep, H_mapped, delta_t, max_iterations, 
                     trotter_rank, trotter_steps, 
                     sampler,  
                     ancilla_qubits, target_qubits,
                     using_statevector=using_statevector, dim_Hankel=10,
                     plot_lambda=False)

    Ens_ODMD, lams_ODMD = odmd.run()
    print("Ens from ODMD:", "".join([f"{tmp:8.3f} " for tmp in np.sort(Ens_ODMD)]))
    E0 = np.min(Ens_ODMD)

    assert ( 100*abs(E0 - Eexact)/abs(Eexact) < 5), f"ODMD test failed: {E0} != {Eexact} within 5% tolerance"
