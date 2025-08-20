import os
import pytest
import pennylane as qml
import pennylane.numpy as qnp
from nuqulib import *

chdir = os.path.dirname(os.path.abspath(__file__))
int_dir = os.path.join(chdir, "interaction_file/")

def test_O18_zero_seniority():
    Z = 8
    N = 10

    ## Specifying the target system and the Hamiltonian
    A = Z + N
    fn = int_dir + "usdb.msnt"
    Acore = 16
    pow_A = -0.3 
    massop = 1
    Hamil, p_sps, n_sps, Dict_qubits_to_sps, Dict_sps_to_qubits = read_msnt(fn, A, Acore, pow_A=pow_A, massdep=massop)

    ## Setting up the qubits and Hamiltonian in the qubit representation
    Nocc = (A - Acore) // 2
    Nq = len(Dict_sps_to_qubits.keys())
    config_list = generate_config_bitstr_list(Nq, Nocc)
    relevant_pairs = get_possible_configs(n_sps)

    h1b, h2b = get_pairwise_Hamil(Hamil, relevant_pairs)
    evals, evecs = eval_Hflat_eigen(config_list, Nq, Nocc, Hamil, h1b, h2b, Dict_qubits_to_sps)
    ops, coeffs, hamiltonian_op = make_pw_hamil_qiskit(Hamil, h1b, Nq, Nocc, Dict_qubits_to_sps)

    ## Translating the Hamiltonian (Qiskit) to pennylane format
    coeffs_pl, obs_pl = read_QiskitPauli(ops, coeffs)
    Hamil_pl = qml.Hamiltonian(coeffs_pl, obs_pl)
    params = np.random.rand(100)

    dev = qml.device("default.qubit", wires=Nq)

    @qml.qnode(dev)
    def circuit(params_in):
        return pair_ansatz_pennylane(Hamil_pl, params_in, Nq, Nocc, type_of_ansatz="pUCCD")        
    where_is_G_or_cG1 = pair_ansatz_pennylane(Hamil_pl, params, Nq, Nocc, type_of_ansatz="pUCCD", return_Gdict=True)
    ngate = len(where_is_G_or_cG1.keys()) 

    params_pl = qnp.array(params[:ngate], requires_grad=True)     
    optimizer = qml.AdamOptimizer(stepsize=1.e-1)        
    Emin = 10**10
    for it in range(200):
        params_pl, _cost = optimizer.step_and_cost(circuit, params_pl)
        if _cost <  Emin:
            Emin = _cost

    assert abs(Emin - np.min(evals)) < 1e-5, "The minimum energy should be close to the exact diagonalization result."

if __name__ == "__main__":    
    test_O18_zero_seniority()