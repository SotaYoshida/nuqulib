import os
import pytest
import numpy as np
from qiskit.primitives import StatevectorEstimator
from nuqulib import *

chdir = os.path.dirname(os.path.abspath(__file__))
int_dir = os.path.join(chdir, "interaction_file/")

def test_valence_2n(filename_snt=int_dir+"ckpot.snt",
                    Eref=2.3458):
    print("\nTesting naive filling config. of 2n in the 0p-shell")
    if not os.path.exists(filename_snt):
        raise FileNotFoundError(f"File {filename_snt} does not exist.")

    params = np.random.randn(100)
    proton_number = 0
    neutron_number = 2
    hamil = Hamiltonian(filename_snt, proton_number, neutron_number, verbose=False)
    Hdict_M = hamil.get_mscheme_H(opform=True)

    n_qubits = hamil.n_qubits
    proton_qubits = list(range(0, hamil.n_qubits_p))
    neutron_qubits = list(range(hamil.n_qubits_p, n_qubits))
    qc = nucl_ansatz(Hdict_M, n_qubits, proton_qubits, neutron_qubits, proton_number, neutron_number,
                     params, "HF")

    # measurement of Hamiltonian
    H_1b, H_n, H_p, H_jz, H_pp, H_nn, H_pn = hamil.mapping_opform(Hdict_M, "JordanWigner")
    h_mapped = H_1b + H_nn
    estimator = StatevectorEstimator()
    job = estimator.run([(qc, h_mapped)])
    results = job.result()
    E_meas = results[0].data.evs
    print("E_meas: ", E_meas)
    assert abs(E_meas - Eref) < 1e-5, f"Expected energy: {Eref}, got: {E_meas}"
    
def test_valence_pn(filename_snt=int_dir+"ckpot.snt",
                    Eref=-5.0088):
    print("\nTesting naive filling config. of p-n in the 0p-shell")
    if not os.path.exists(filename_snt):
        raise FileNotFoundError(f"File {filename_snt} does not exist.")

    proton_number = neutron_number = 1
    hamil = Hamiltonian(filename_snt, proton_number, neutron_number, verbose=False)
    n_qubits = hamil.n_qubits
    proton_qubits = list(range(0, hamil.n_qubits_p))
    neutron_qubits = list(range(hamil.n_qubits_p, n_qubits))
    Hdict_M = hamil.get_mscheme_H(opform=True)
    qc = nucl_ansatz(Hdict_M, n_qubits, proton_qubits, neutron_qubits, proton_number, neutron_number, [], "HF")

    # measurement of Hamiltonian
    H_1b, H_n, H_p, H_jz, H_pp, H_nn, H_pn = hamil.mapping_opform(Hdict_M, "JordanWigner")
    h_mapped = H_1b + H_pp + H_nn + H_pn
    estimator = StatevectorEstimator()
    job = estimator.run([(qc, h_mapped)])
    results = job.result()
    E_meas = results[0].data.evs
    print("E_meas: ", E_meas)
    assert abs(E_meas - Eref) < 1e-5, f"Expected energy: {Eref}, got: {E_meas}"
    
def test_valence_16O(filename_snt=int_dir+"ckpot.snt",
                     Eref=-113.81425):
    print("\nTesting 16O in the 0p-shell (fully filled)")
    if not os.path.exists(filename_snt):
        raise FileNotFoundError(f"File {filename_snt} does not exist.")

    proton_number = neutron_number = 6
    hamil = Hamiltonian(filename_snt, proton_number, neutron_number, verbose=False)
    n_qubits = hamil.n_qubits
    proton_qubits = list(range(0, hamil.n_qubits_p))
    neutron_qubits = list(range(hamil.n_qubits_p, n_qubits))
    Hdict_M = hamil.get_mscheme_H(opform=True)
    qc = nucl_ansatz(Hdict_M, n_qubits, proton_qubits, neutron_qubits, proton_number, neutron_number, [], "HF")

    # measurement of Hamiltonian
    H_1b, H_n, H_p, H_jz, H_pp, H_nn, H_pn = hamil.mapping_opform(Hdict_M, "JordanWigner")
    h_mapped = H_1b + H_pp + H_nn + H_pn

    estimator = StatevectorEstimator()
    job = estimator.run([(qc, h_mapped)])
    results = job.result()
    E_meas = results[0].data.evs
    print("E_meas: ", E_meas)
    assert abs(E_meas - Eref) < 1e-5, f"Expected energy: {Eref}, got: {E_meas}"
    
def test_ncsm_2n_emax0(Eref=5.90409, filename_snt=int_dir+"TwBME-HO_NN-only_N3LO_EM500_srg1.8_hw20_emax0_e2max0.kshell.snt"):
    print("\nTesting NCSM hamiltonian for 2n in the 0p-shell. Now using EM500 hw20 SRG(1.8) emax=0.")
    Z = 0
    N = 2
    hamil = Hamiltonian(filename_snt, Z, N, ncsm=True, verbose=False)
    n_qubits = hamil.n_qubits
    proton_qubits = list(range(0, hamil.n_qubits_p))
    neutron_qubits = list(range(hamil.n_qubits_p, n_qubits))
    mapping_method = "JordanWigner"
    Hdict_M = hamil.get_mscheme_H(opform=True)
    H_1b, H_n, H_p, H_jz, H_pp, H_nn, H_pn = hamil.mapping_opform(Hdict_M, mapping_method)
    qc = nucl_ansatz(Hdict_M, n_qubits, proton_qubits, neutron_qubits, Z, N, [], "HF")
        
    # measurement of Hamiltonian
    h_mapped = H_1b + H_nn
    estimator = StatevectorEstimator()
    job = estimator.run([(qc, h_mapped)])
    results = job.result()
    E_meas = results[0].data.evs
    print("E_meas: ", E_meas)
    assert abs(E_meas - Eref) < 1e-5, f"Expected energy: {Eref}, got: {E_meas}"

def test_ncsm_4He_emax0(Eref=-22.99767, filename_snt=int_dir+"TwBME-HO_NN-only_N3LO_EM500_srg1.8_hw20_emax0_e2max0.kshell.snt"):
    print("\nTesting NCSM hamiltonian for 4He in the 0p-shell. Now using EM500 hw20 SRG(1.8) emax=0.")
    Z = proton_number = 2
    N = neutron_number = 2
    hamil = Hamiltonian(filename_snt, Z, N, ncsm=True, verbose=False)
    n_qubits = hamil.n_qubits
    proton_qubits = list(range(0, hamil.n_qubits_p))
    neutron_qubits = list(range(hamil.n_qubits_p, n_qubits))

    mapping_method = "JordanWigner"
    Hdict_M = hamil.get_mscheme_H(opform=True)
    H_1b, H_n, H_p, H_jz, H_pp, H_nn, H_pn = hamil.mapping_opform(Hdict_M, mapping_method)
    qc = nucl_ansatz(Hdict_M, n_qubits, proton_qubits, neutron_qubits, proton_number, neutron_number, [], "HF")

    # measurement of Hamiltonian
    h_mapped = H_1b + H_nn + H_pp + H_pn
    estimator = StatevectorEstimator()
    job = estimator.run([(qc, h_mapped)])
    results = job.result()
    E_meas = results[0].data.evs
    print("E_meas: ", E_meas)
    assert abs(E_meas - Eref) < 1e-5, f"Expected energy: {Eref}, got: {E_meas}"

def test_ncsm_16O_emax1(Eref=-148.36879, filename_snt=int_dir+"TwBME-HO_NN-only_N3LO_EM500_srg1.8_hw20_emax1_e2max2.kshell.snt"):
    print("\nTesting NCSM hamiltonian for 16O in the 0p-shell. Now using EM500 hw20 SRG(1.8) emax=1.")
    Z = proton_number = 8
    N = neutron_number = 8
    hamil = Hamiltonian(filename_snt, Z, N, ncsm=True, verbose=False)
    n_qubits = hamil.n_qubits
    proton_qubits = list(range(0, hamil.n_qubits_p))
    neutron_qubits = list(range(hamil.n_qubits_p, n_qubits))

    mapping_method = "JordanWigner"
    Hdict_M = hamil.get_mscheme_H(opform=True)
    H_1b, H_n, H_p, H_jz, H_pp, H_nn, H_pn = hamil.mapping_opform(Hdict_M, mapping_method)

    qc = nucl_ansatz(Hdict_M, n_qubits, proton_qubits, neutron_qubits, proton_number, neutron_number, [], "HF")

    # Decompose it. Otherwise, it will take much time.
    qc = qc.decompose()

    # measurement of Hamiltonian
    h_mapped = H_1b + H_nn + H_pp + H_pn
    estimator = StatevectorEstimator()
    job = estimator.run([(qc, h_mapped)])
    results = job.result()
    E_meas = results[0].data.evs
    print("E_meas: ", E_meas)
    assert abs(E_meas - -148.36879) < 1e-5, f"Expected energy: -22.99767, got: {E_meas}"

def test_ncsm_16O_emax1_NN3NF(Eref=-131.83565, 
                              filename_snt=int_dir+"TwBME-HO_NN-only_N3LO_EM500_srg1.8_hw20_emax1_e2max2.kshell.snt",
                              fn_3NF=int_dir+"ThBME_lnl_ms1_2_1.readable.txt"):
    emax = 1
    Z = proton_number  = N = neutron_number = 8
    hamil = Hamiltonian(filename_snt, Z, N, ncsm=True, verbose=False, emax_truncate=emax,
                        e3max=emax, fn_3NF=fn_3NF)    
    mapping_method = "JordanWigner"
    print("get_mscheme_H...")
    Hdict_M = hamil.get_mscheme_H(opform=True)
    print("mapping_opform...")
    H_1b, H_n, H_p, H_jz, H_pp, H_nn, H_pn = hamil.mapping_opform(Hdict_M, mapping_method)
    print("2NF done!")
    Hamil_NCSM_NN = H_1b 
    if Z >= 2:
        Hamil_NCSM_NN += H_pp
    if N >= 2:
        Hamil_NCSM_NN += H_nn
    if Z >= 1 and N >= 1:
        Hamil_NCSM_NN += H_pn

    print("set_mscheme_3NF...")
    hamil.set_mscheme_3NF()
    print("mapping_3NF_Mscheme...")
    H_3b = hamil.mapping_3NF_Mscheme()

    n_qubits = hamil.n_qubits
    proton_qubits = list(range(0, hamil.n_qubits_p))
    neutron_qubits = list(range(hamil.n_qubits_p, n_qubits))

    qc = nucl_ansatz(Hdict_M, n_qubits, proton_qubits, neutron_qubits, proton_number, neutron_number, [], "HF")
    qc = qc.decompose()

    # measurement of Hamiltonian
    estimator = StatevectorEstimator()
    job = estimator.run([(qc, H_1b), (qc, H_pp), (qc, H_nn), (qc, H_pn), (qc, H_3b)])
    results = job.result()
    E_1b, E_pp, E_nn, E_pn, E_3n = [ results[i].data.evs for i in range(len(results))]
    E_total = E_1b + E_pp + E_nn + E_pn + E_3n    
    print("Etot", E_total, "E(NN)", E_1b + E_pp + E_nn + E_pn)
    print("E_1b: ", E_1b, "<pp>", E_pp, "<nn>", E_nn, "<pn>", E_pn, "<3b>", E_3n)
    print("# of paulis in H2b/H3b:", len(Hamil_NCSM_NN.paulis), len(H_3b.paulis))
    print("")
    assert abs(E_total - Eref) < 1e-5, f"Expected energy: {Eref}, got: {E_total}"

