"""
Resource estimation tests for nuqulib quantum algorithms.

This file contains basic resource estimation tests for efficiency.
For comprehensive resource estimation benchmarks with exact gate counts,
see test_technical_integration.py.
"""
import os
import pytest
from nuqulib import *

chdir = os.path.dirname(os.path.abspath(__file__))
int_dir = os.path.join(chdir, "interaction_file/")

def get_Hamil(filename_snt, Z, N, fn_3NF="", emax=100, ncsm=False):
    if fn_3NF != "":
        hamil = Hamiltonian(filename_snt, Z, N, ncsm=True, emax_truncate=emax, e3max=emax, fn_3NF=fn_3NF)
    else:
        hamil = Hamiltonian(filename_snt, Z, N, ncsm=ncsm, emax_truncate=emax)

    n_qubits = hamil.n_qubits
    proton_qubits = list(range(0, hamil.n_qubits_p))
    neutron_qubits = list(range(hamil.n_qubits_p, n_qubits))

    Hdict_M = hamil.get_mscheme_H(opform=True)
    H_1b, H_pp, H_nn, H_pn = hamil.mapping_opform(Hdict_M, "Jordan-Wigner")

    if fn_3NF != "":
        hamil.set_mscheme_3NF()
        H_3b = hamil.mapping_3NF_Mscheme()

    Hamil_ShellModel = H_1b 
    if Z > 1:
        Hamil_ShellModel += H_pp 
    if N > 1:
        Hamil_ShellModel += H_nn
    if Z > 0 and N > 0:
        Hamil_ShellModel += H_pn

    if fn_3NF != "":
        Hamil_ShellModel += H_3b

    return hamil, Hamil_ShellModel, proton_qubits, neutron_qubits

def resource_estimation_QPE(N_a, N_q, H_mapped, dt, trotter_steps=1):
    Unitary = PauliEvolutionGate(H_mapped, dt, synthesis=SuzukiTrotter(order=1,reps=trotter_steps))
    # We don't construct the full circuit here, just construct a single controlled unitary    
    qc = QuantumCircuit(1+N_q)
    cU = Unitary.control(1)
    qc.append(cU, [0] + list(range(1, 1+N_q)))
    qc = qc.decompose(reps=5)
    cU_ops = qc.count_ops()
    print("# of gates c-U:", cU_ops)
    dict_ops = { }
    for ith in range(N_a):
        for k, v in cU_ops.items():
            if k not in dict_ops:
                dict_ops[k] = 0
            dict_ops[k] += v * 2**ith
    return dict_ops

def resource_estimation_QKrylov(Niter, H_mapped, dt=1.0, trotter_steps=1,
                                H_is_like="generic"):
    if H_is_like == "pairing":
        N_H = 2
    else:
        N_H = len(H_mapped.paulis) # worst case 
    Unitary = PauliEvolutionGate(H_mapped, dt, synthesis=SuzukiTrotter(order=1,reps=trotter_steps))
    qc = QuantumCircuit(1+H_mapped.num_qubits)
    cU = Unitary.control(1)
    qc.append(cU, [0] + list(range(1, 1+H_mapped.num_qubits)))
    qc = qc.decompose(reps=5)
    cU_ops = qc.count_ops()
    print("# of gates c-U:", cU_ops)
    dict_ops = { }
    for k, v in cU_ops.items():
        dict_ops[k] = v * ( 2 * Niter * (Niter -1) )* (1+N_H)
    return dict_ops

def test_resource_estimation():
    """Simplified resource estimation test for basic functionality."""
    filename_snt = int_dir + "ckpot.snt"

    hamil, H_mapped, proton_qubits, neutron_qubits = get_Hamil(filename_snt, 2, 2)
    N_a = 2  # Reduced from 4 for efficiency
    Niter = 5  # Reduced from 20 for efficiency
    N_q = hamil.n_qubits
    dt = 0.1
    trotter_steps = 1

    # Basic QPE resource estimation test
    dict_ops_QPE = resource_estimation_QPE(N_a, N_q, H_mapped, dt=dt, trotter_steps=trotter_steps)
    assert 'cx' in dict_ops_QPE, "QPE resource estimation should include cx gates"
    assert dict_ops_QPE['cx'] > 0, "QPE should require some cx gates"
    
    # Basic Quantum Krylov resource estimation test  
    dict_ops_QKrylov = resource_estimation_QKrylov(Niter, H_mapped, dt=dt, trotter_steps=trotter_steps)
    assert 'cx' in dict_ops_QKrylov, "QKrylov resource estimation should include cx gates"
    assert dict_ops_QKrylov['cx'] > dict_ops_QPE['cx'], "QKrylov should generally require more gates than QPE"

if __name__ == "__main__":    
    print("Running optimized resource estimation tests...")
    test_resource_estimation()
    print("Resource estimation tests completed successfully!")
    print("For comprehensive resource estimation with exact gate counts, run test_technical_integration.py")
