"""
Technical integration tests for nuqulib.

These tests focus on computational benchmarks, resource estimation,
and complex integration scenarios that are primarily for technical validation
rather than core functionality testing. They may take longer to run.
"""

import os
import pytest
from nuqulib import *

chdir = os.path.dirname(os.path.abspath(__file__))
int_dir = os.path.join(chdir, "interaction_file/")


def get_Hamil(filename_snt, Z, N, fn_3NF="", emax=100, ncsm=False):
    """Helper function for setting up Hamiltonians."""
    hamil = Hamiltonian(filename_snt, Z, N, fn_3NF=fn_3NF, ncsm=ncsm, emax_truncate=emax, verbose=False)
    n_qubits = hamil.n_qubits
    proton_qubits = list(range(0, hamil.n_qubits_p))
    neutron_qubits = list(range(hamil.n_qubits_p, n_qubits))
    
    Hdict_M = hamil.get_mscheme_H(opform=True)
    H_1b, H_pp, H_nn, H_pn = hamil.mapping_opform(Hdict_M, "JordanWigner")
    
    if fn_3NF != "":
        H_3b = hamil.mapping_3NF_Mscheme("JordanWigner")
    else:
        H_3b = None
    
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
    """Resource estimation for Quantum Phase Estimation."""
    # Count gates in Hamiltonian simulation
    dict_ops = H_mapped.count_ops()
    for k, v in dict_ops.items():
        dict_ops[k] = v * trotter_steps * N_a * (2**N_a - 1)
    return dict_ops


def resource_estimation_QKrylov(Niter, H_mapped, dt=1.0, trotter_steps=1, H_is_like="generic"):
    """Resource estimation for Quantum Krylov methods."""
    dict_ops = H_mapped.count_ops()
    
    if H_is_like == "pairing":
        # Simplified counting for pairing-like Hamiltonians
        N_H = len(dict_ops) // 4  # Rough estimate
    else:
        N_H = len(dict_ops)
    
    for k, v in dict_ops.items():
        dict_ops[k] = v * (2 * Niter * (Niter - 1)) * (1 + N_H)
    return dict_ops


def test_resource_estimation():
    """Test resource estimation for quantum algorithms."""
    filename_snt = int_dir + "ckpot.snt"

    hamil, H_mapped, proton_qubits, neutron_qubits = get_Hamil(filename_snt, 2, 2)
    N_a = 4
    Niter = 20
    N_q = hamil.n_qubits
    dt = 0.1
    trotter_steps = 1

    # QPE resource estimation
    dict_ops_QPE = resource_estimation_QPE(N_a, N_q, H_mapped, dt=dt, trotter_steps=trotter_steps)
    print("# of gates (QPE): ", dict_ops_QPE)
    assert dict_ops_QPE['cx'] == 1040220, "cx for QPE may be wrong under Na=4 and p-shell space"
    
    # Quantum Krylov resource estimation
    dict_ops_QKrylov = resource_estimation_QKrylov(Niter, H_mapped, dt=dt, trotter_steps=trotter_steps)
    print("# of gates (QKrylov): ", dict_ops_QKrylov)
    assert dict_ops_QKrylov['cx'] == 52862593440, "cx for QKrylov may be wrong under Nit=20 and p-shell space"

    dict_ops_QKrylov_pairing = resource_estimation_QKrylov(Niter, H_mapped, dt=dt, trotter_steps=trotter_steps, H_is_like="pairing")
    print("# of gates (QKrylov; pairing-like): ", dict_ops_QKrylov_pairing)
    assert dict_ops_QKrylov_pairing['cx'] == 158113440, "cx for QKrylov pairing-like may be wrong under Nit=20 and p-shell space"


def test_comprehensive_O18_zero_seniority():
    """Comprehensive O18 test with full optimization - moved from main tests for efficiency."""
    Z = 8
    N = 10

    # Specifying the target system and the Hamiltonian
    A = Z + N
    fn = int_dir + "usdb.msnt"
    Acore = 16
    pow_A = -0.3 
    massop = 1
    Hamil, p_sps, n_sps, Dict_qubits_to_sps, Dict_sps_to_qubits = read_msnt(fn, A, Acore, pow_A=pow_A, massdep=massop)

    # Setting up the qubits and Hamiltonian in the qubit representation
    Nocc = (A - Acore) // 2
    Nq = len(Dict_sps_to_qubits.keys())
    config_list = generate_config_bitstr_list(Nq, Nocc)
    relevant_pairs = get_possible_configs(n_sps)

    h1b, h2b = get_pairwise_Hamil(Hamil, relevant_pairs)
    evals, evecs = eval_Hflat_eigen(config_list, Nq, Nocc, Hamil, h1b, h2b, Dict_qubits_to_sps)
    ops, coeffs, hamiltonian_op = make_pw_hamil_qiskit(Hamil, h1b, Nq, Nocc, Dict_qubits_to_sps)

    # Translating the Hamiltonian (Qiskit) to pennylane format
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
    # Full optimization for technical validation
    for it in range(200):
        params_pl, _cost = optimizer.step_and_cost(circuit, params_pl)
        if _cost <  Emin:
            Emin = _cost

    assert abs(Emin - np.min(evals)) < 1e-5, "The minimum energy should be close to the exact diagonalization result."
    
    # Check IBM's Estimator gives the same result
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


def test_comprehensive_nuclear_structure():
    """Comprehensive nuclear structure tests moved from main integration tests."""
    
    # Test 16O valence configuration
    filename_snt = int_dir + "ckpot.snt"
    Eref = -113.81425
    
    if not os.path.exists(filename_snt):
        pytest.skip(f"File {filename_snt} does not exist.")

    proton_number = neutron_number = 8
    hamil = Hamiltonian(filename_snt, proton_number, neutron_number, verbose=False)
    n_qubits = hamil.n_qubits
    proton_qubits = list(range(0, hamil.n_qubits_p))
    neutron_qubits = list(range(hamil.n_qubits_p, n_qubits))
    qc = nucl_ansatz(n_qubits, proton_qubits, neutron_qubits, proton_number, neutron_number, [], "HF")

    # measurement of Hamiltonian
    Hdict_M = hamil.get_mscheme_H(opform=True)
    H_1b, H_pp, H_nn, H_pn = hamil.mapping_opform(Hdict_M, "JordanWigner")
    h_mapped = H_1b + H_pp + H_nn + H_pn
    estimator = StatevectorEstimator()
    job = estimator.run([(qc, h_mapped)])
    results = job.result()
    E_meas = results[0].data.evs
    assert abs(E_meas - Eref) < 1e-5, f"Expected energy: {Eref}, got: {E_meas}"

    # Test NCSM calculations
    test_ncsm_2n_emax0()
    test_ncsm_16O_emax1()
    test_ncsm_16O_emax1_NN3NF()


def test_ncsm_2n_emax0(Eref=-5.8015, filename_snt=int_dir+"TwBME-HO_NN-only_N3LO_EM500_srg1.8_hw20_emax0_e2max0.kshell.snt"):
    """Test NCSM calculation for 2-neutron system."""
    if not os.path.exists(filename_snt):
        pytest.skip(f"File {filename_snt} does not exist.")
        
    hamil = Hamiltonian(filename_snt, 0, 2, ncsm=True, verbose=False)
    n_qubits = hamil.n_qubits
    proton_qubits = list(range(0, hamil.n_qubits_p))
    neutron_qubits = list(range(hamil.n_qubits_p, n_qubits))
    qc = nucl_ansatz(n_qubits, proton_qubits, neutron_qubits, 0, 2, [], "HF")

    Hdict_M = hamil.get_mscheme_H(opform=True)
    H_1b, H_pp, H_nn, H_pn = hamil.mapping_opform(Hdict_M, "JordanWigner")
    h_mapped = H_1b + H_nn
    estimator = StatevectorEstimator()
    job = estimator.run([(qc, h_mapped)])
    results = job.result()
    E_meas = results[0].data.evs
    assert abs(E_meas - Eref) < 1e-5, f"Expected energy: {Eref}, got: {E_meas}"


def test_ncsm_16O_emax1(Eref=-148.36879, filename_snt=int_dir+"TwBME-HO_NN-only_N3LO_EM500_srg1.8_hw20_emax1_e2max2.kshell.snt"):
    """Test NCSM calculation for 16O system."""
    if not os.path.exists(filename_snt):
        pytest.skip(f"File {filename_snt} does not exist.")

    hamil = Hamiltonian(filename_snt, 8, 8, ncsm=True, emax_truncate=1, verbose=False)
    n_qubits = hamil.n_qubits
    proton_qubits = list(range(0, hamil.n_qubits_p))
    neutron_qubits = list(range(hamil.n_qubits_p, n_qubits))
    qc = nucl_ansatz(n_qubits, proton_qubits, neutron_qubits, 8, 8, [], "HF")

    Hdict_M = hamil.get_mscheme_H(opform=True)
    H_1b, H_pp, H_nn, H_pn = hamil.mapping_opform(Hdict_M, "JordanWigner")
    h_mapped = H_1b + H_pp + H_nn + H_pn
    estimator = StatevectorEstimator()
    job = estimator.run([(qc, h_mapped)])
    results = job.result()
    E_meas = results[0].data.evs
    assert abs(E_meas - Eref) < 1e-5, f"Expected energy: {Eref}, got: {E_meas}"


def test_ncsm_16O_emax1_NN3NF(Eref=-131.83565, 
                              filename_snt=int_dir+"TwBME-HO_NN-only_N3LO_EM500_srg1.8_hw20_emax1_e2max2.kshell.snt",
                              fn_3NF=int_dir+"ThBME_lnl_ms1_2_1.readable.txt"):
    """Test NCSM calculation with 3-nucleon forces."""
    if not os.path.exists(filename_snt) or not os.path.exists(fn_3NF):
        pytest.skip(f"Required files do not exist: {filename_snt}, {fn_3NF}")

    hamil = Hamiltonian(filename_snt, 8, 8, fn_3NF=fn_3NF, ncsm=True, emax_truncate=1, verbose=False)
    n_qubits = hamil.n_qubits
    proton_qubits = list(range(0, hamil.n_qubits_p))
    neutron_qubits = list(range(hamil.n_qubits_p, n_qubits))
    qc = nucl_ansatz(n_qubits, proton_qubits, neutron_qubits, 8, 8, [], "HF")

    Hdict_M = hamil.get_mscheme_H(opform=True)
    Hamil_NCSM_NN = H_1b + H_pp + H_nn + H_pn = hamil.mapping_opform(Hdict_M, "JordanWigner")
    H_3b = hamil.mapping_3NF_Mscheme("JordanWigner")

    estimator = StatevectorEstimator()
    job = estimator.run([(qc, H_1b), (qc, H_pp), (qc, H_nn), (qc, H_pn), (qc, H_3b)])
    results = job.result()
    E_1b, E_pp, E_nn, E_pn, E_3n = [results[i].data.evs for i in range(len(results))]
    E_total = E_1b + E_pp + E_nn + E_pn + E_3n    
    
    assert abs(E_total - Eref) < 1e-5, f"Expected energy: {Eref}, got: {E_total}"


if __name__ == "__main__":
    # Run technical integration tests
    test_resource_estimation()
    test_comprehensive_O18_zero_seniority()
    test_comprehensive_nuclear_structure()
    print("Technical integration tests passed successfully.")