import os
import numpy as np
import pytest
import pennylane as qml
import pennylane.numpy as qnp
from qiskit.primitives import StatevectorEstimator
from qiskit_aer.primitives import SamplerV2
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

    h1b, h2b = get_pairwise_Hamil(Hamil, n_sps)
    evals, evecs = eval_Hflat_eigen(Nq, Nocc, Hamil, h1b, h2b, Dict_qubits_to_sps)
    hamiltonian_op = make_pw_hamil_qiskit(Hamil, h1b, Nq, Nocc, Dict_qubits_to_sps)
    hamiltonian_op_diag, hamiltonian_op_XXYY = separate_Hamil_terms(hamiltonian_op)

    ## Translating the Hamiltonian (Qiskit) to pennylane format
    ops = [op.to_label() for op in hamiltonian_op.paulis]
    coeffs = np.array([coeff for coeff in hamiltonian_op.coeffs])
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
    params_opt = np.zeros_like(params_pl)
    optimizer = qml.AdamOptimizer(stepsize=1.e-1)        
    Emin = 10**10
    for it in range(200):
        params_pl, _cost = optimizer.step_and_cost(circuit, params_pl)
        if _cost <  Emin:
            Emin = _cost
            params_opt = params_pl
    params_opt = np.array(params_opt)
    assert abs(Emin - np.min(evals)) < 1e-5, "The minimum energy should be close to the exact diagonalization result."
    
    ## Check IBM's Estimator gives the same result
    ansatz = pair_ansatz_qiskit(
        np.array(params_pl),
        Nq,
        Nocc,
        method="pUCCD",        
    )

    Estimator = StatevectorEstimator()
    job = Estimator.run([(ansatz, hamiltonian_op)])
    results = job.result()
    E_meas = results[0].data.evs
    assert abs(E_meas - np.min(evals)) < 1e-5, "The minimum energy should be close to the exact diagonalization result."

    ## Mearure <H> using basis rotations circuits to diagonalize XX+YY
    using_noisy_simulation = False
    postselection_XXYY = True
    adopted = "simFTQC"
    backend = None
    nshot = 10**4
    num_experiment = 1
    sampler = SamplerV2() 
    
    qc_ansatz = pair_ansatz_qiskit(params_opt, Nq, Nocc, method="pUCCD",)
    E_diag = eval_Ediag(adopted, Nq, Nocc, hamiltonian_op_diag, qc_ansatz, backend, sampler, nshot)

    qc_list = circuit_XXYY(qc_ansatz, "simFTQC", Nq, backend=backend)
    E_XXYY_Google = eval_Energy_using_GoogleCircuit(Nq, Nocc, hamiltonian_op_XXYY, 
                                        qc_list, sampler, nshot, num_experiment, 
                                        using_noisy_simulation, postselection_XXYY=postselection_XXYY)
    Energy = E_diag + E_XXYY_Google
    print(f"E_dig {E_diag} XXYY {E_XXYY_Google} => Energy = {Energy}")
    assert abs(Energy - E_meas) < 10/(np.sqrt(nshot)), "The energy should be close to the measured value."


def test_O20_zero_seniority():
    Z = 8
    N = 12

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
    
    h1b, h2b = get_pairwise_Hamil(Hamil, n_sps)
    hamiltonian_op = make_pw_hamil_qiskit(Hamil, h1b, Nq, Nocc, Dict_qubits_to_sps)

    ## Translating the Hamiltonian (Qiskit) to pennylane format
    coeffs_pl, obs_pl = transform_qiskitOps_to_pennylane(hamiltonian_op)
    Hamil_pl = qml.Hamiltonian(coeffs_pl, obs_pl)
    params = np.random.rand(100)
    dev = qml.device("default.qubit", wires=Nq)

    @qml.qnode(dev)
    def circuit(params_in):
        return pair_ansatz_pennylane(Hamil_pl, params_in, Nq, Nocc, type_of_ansatz="pUCCD+all2all")        
    ngate = Nocc * (Nq-Nocc)

    params_pl = qnp.array(params[:ngate], requires_grad=True)     
    optimizer = qml.AdamOptimizer(stepsize=1.e-1)        
    Emin = 10**10
    for it in range(200):
        params_pl, _cost = optimizer.step_and_cost(circuit, params_pl)
        if _cost <  Emin:
            Emin = _cost
    E_DOCI = -23.1461
    print("Emin(O20): ", Emin, "E_DOCI", E_DOCI)
    assert abs(Emin - E_DOCI) < 0.5 # "The minimum energy should be close to the DOCI result."

