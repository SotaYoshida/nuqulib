import os
import pytest
import numpy as np
from qiskit.primitives import StatevectorEstimator
from nuqulib import *

chdir = os.path.dirname(os.path.abspath(__file__))
int_dir = os.path.join(chdir, "interaction_file/")

def Estimate_Energy(qc, h_mapped, mapping_method, verbose=True):
    estimator = StatevectorEstimator()
    job = estimator.run([(qc.decompose(), h_mapped)])
    results = job.result()
    E_meas = results[0].data.evs
    if verbose:
        print(f"E_meas({mapping_method}): ", E_meas)
    return E_meas

# def test_valence_2n(filename_snt=int_dir+"ckpot.snt",
#                     Eref=2.3458):
#     print("\nTesting naive filling config. of 2n in the 0p-shell")
#     if not os.path.exists(filename_snt):
#         raise FileNotFoundError(f"File {filename_snt} does not exist.")
#     proton_number = 0
#     neutron_number = 2

#     params = np.random.randn(100)
#     hamil = Hamiltonian(filename_snt, proton_number, neutron_number, verbose=False)
#     Hdict_M = hamil.get_mscheme_H(opform=True)
#     n_qubits = hamil.n_qubits
#     proton_qubits = list(range(0, hamil.n_qubits_p))
#     neutron_qubits = list(range(hamil.n_qubits_p, n_qubits))

#     ### JordanWigner Mapper
#     mapping_method = "JordanWigner"
#     H_1b, H_n, H_p, H_jz, H_pp, H_nn, H_pn = hamil.mapping_opform(Hdict_M, mapping_method)
#     h_mapped = H_1b + H_nn

#     qc = nucl_ansatz(Hdict_M, n_qubits, proton_qubits, neutron_qubits, proton_number, neutron_number,
#                      params, "HF", mapping_method=mapping_method)
#     E_meas_JW = Estimate_Energy(qc, h_mapped, mapping_method)

#     ### HATTMapper
#     mapping_method = "HATTMapper"
#     filepath="./test_mapper_ckpot_2n"

#     Hdict_M = hamil.get_mscheme_H(opform=True)
#     H_1b, H_n, H_p, H_jz, H_pp, H_nn, H_pn = hamil.mapping_opform(Hdict_M, mapping_method, filepath)
#     h_mapped = H_1b + H_nn

#     qc = nucl_ansatz(Hdict_M, n_qubits, proton_qubits, neutron_qubits,
#                      proton_number, neutron_number, params, "HF",
#                      mapping_method=mapping_method, filepath=filepath)

#     E_meas_HATT = Estimate_Energy(qc, h_mapped, mapping_method)
#     assert abs(E_meas_HATT - E_meas_JW) < 1e-5
#     os.system("rm test_mapper_ckpot_2n")



# def test_valence_pn(filename_snt=int_dir+"ckpot.snt",
#                     Eref=-5.0088):
#     print("\nTesting naive filling config. of p-n in the 0p-shell")
#     if not os.path.exists(filename_snt):
#         raise FileNotFoundError(f"File {filename_snt} does not exist.")

#     proton_number = neutron_number = 6
#     params = np.random.randn(100) # dummy

#     hamil = Hamiltonian(filename_snt, proton_number, neutron_number, verbose=False)
#     Hdict_M = hamil.get_mscheme_H(opform=True)
#     n_qubits = hamil.n_qubits
#     proton_qubits = list(range(0, hamil.n_qubits_p))
#     neutron_qubits = list(range(hamil.n_qubits_p, n_qubits))


#     ## Edit H_dict by hand
#     # Hdict_M["Vpn"] = {('+_3 -_3', '+_4 -_4'): -4.252095000000001,
#     #                   ('+_4 -_4', '+_3 -_3'): -4.252095000000001
#     # }
    
#     ### JordanWigner Mapper
#     mapping_method = "JordanWigner"
#     H_1b, H_n, H_p, H_jz, H_pp, H_nn, H_pn = hamil.mapping_opform(Hdict_M, mapping_method)
#     h_mapped_JW = H_1b + H_pp + H_nn + H_pn 
#     qc_JW = nucl_ansatz(Hdict_M, n_qubits, proton_qubits, neutron_qubits, proton_number, neutron_number,
#                      params, "HF", mapping_method=mapping_method)
#     E_meas_JW = Estimate_Energy(qc_JW, h_mapped_JW, mapping_method)

#     # ### HATTMapper
#     # mapping_method = "HATTMapper"
#     # filepath="./test_mapper_ckpot_pn"
#     # H_1b, H_n, H_p, H_jz, H_pp, H_nn, H_pn = hamil.mapping_opform(Hdict_M, mapping_method, filepath)
#     # h_mapped =  H_1b + H_pp + H_nn + H_pn
#     # qc = nucl_ansatz(Hdict_M, n_qubits, proton_qubits, neutron_qubits,
#     #                  proton_number, neutron_number, params, "HF",
#     #                  mapping_method=mapping_method, filepath=filepath)
#     # E_meas_HATT = Estimate_Energy(qc, h_mapped, mapping_method)

#     # # Esum_J = Esum_H = 0.0
#     # # for idx in range(len(h_mapped)):
#     # #     h_JW = h_mapped_JW[idx]
#     # #     h_HATT = h_mapped[idx]
#     # #     e_JW = Estimate_Energy(qc_JW, h_JW, "JordanWigner", verbose=False)
#     # #     e_HATT = Estimate_Energy(qc, h_HATT, "HATTMapper", verbose=False)
#     # #     Esum_J += e_JW
#     # #     #Esum_H += e_HATT
#     # #     close_ = abs(e_JW - e_HATT) < 1e-5
#     # #     if e_JW != 0.0: #not(close_):
#     # #         print(f"Term@{idx}: {close_}")
#     # #         print(f"- JW:   {h_mapped_JW[idx].paulis[0]}  {h_mapped_JW[idx].coeffs[0]} => {e_JW}")
#     # #         print(f"- HATT: {h_mapped[idx].paulis[0]}  {h_mapped[idx].coeffs[0]} => {e_HATT}")
#     # #     if close_:
#     # #         Esum_H += e_HATT
#     # #     else:
#     # #         Esum_H -= e_HATT

#     # #print("h_mapped_JW", h_mapped_JW)
#     # #print("h_mapped", h_mapped)

#     # assert abs(E_meas_HATT - E_meas_JW) < 1e-5
#     assert abs(E_meas_JW - Eref) < 1e-5
#     os.system("rm test_mapper_ckpot_pn")
    

# def test_valence_16O(filename_snt=int_dir+"ckpot.snt",
#                      Eref=-113.81425):
#     print("\nTesting 16O in the 0p-shell (fully filled)")
#     if not os.path.exists(filename_snt):
#         raise FileNotFoundError(f"File {filename_snt} does not exist.")

#     params = np.zeros(100)
#     proton_number = neutron_number = 8
#     hamil = Hamiltonian(filename_snt, proton_number, neutron_number, verbose=False)
#     n_qubits = hamil.n_qubits
#     proton_qubits = list(range(0, hamil.n_qubits_p))
#     neutron_qubits = list(range(hamil.n_qubits_p, n_qubits))
#     Hdict_M = hamil.get_mscheme_H(opform=True)

#     ### JordanWigner Mapper
#     mapping_method = "JordanWigner"
#     H_1b, H_n, H_p, H_jz, H_pp, H_nn, H_pn = hamil.mapping_opform(Hdict_M, mapping_method)
#     h_mapped = H_1b + H_nn + H_pp + H_pn

#     qc = nucl_ansatz(Hdict_M, n_qubits, proton_qubits, neutron_qubits, proton_number, neutron_number,
#                      params, "HF", mapping_method=mapping_method)
#     E_meas_JW = Estimate_Energy(qc, h_mapped, mapping_method)

#     ### HATTMapper
#     mapping_method = "HATTMapper"
#     filepath="./test_mapper_ckpot_16O"
#     Hdict_M = hamil.get_mscheme_H(opform=True)
#     H_1b, H_n, H_p, H_jz, H_pp, H_nn, H_pn = hamil.mapping_opform(Hdict_M, mapping_method, filepath)
#     h_mapped = H_1b + H_nn + H_pp + H_pn
#     qc = nucl_ansatz(Hdict_M, n_qubits, proton_qubits, neutron_qubits,
#                      proton_number, neutron_number, params, "HF",
#                      mapping_method=mapping_method, filepath=filepath)

#     E_meas_HATT = Estimate_Energy(qc, h_mapped, mapping_method)
#     assert abs(E_meas_HATT - E_meas_JW) < 1e-5
#     assert abs(E_meas_JW - Eref) < 1e-5
#     os.system("rm test_mapper_ckpot_16O")


def eval_eachterm(qc, h_mapped, mapping_method, max_eval_num=10):
    for i in range(max_eval_num):
        Estimate_Energy(qc, h_mapped[i], mapping_method)

def test_ncsm_4He_emax0(Eref=50.30169, filename_snt=int_dir+"TwBME-HO_NN-only_N3LO_EM500_srg1.8_hw20_emax0_e2max0.kshell.snt"):
    print("\nTesting NCSM hamiltonian for 4He in emax=0 model space. Now using EM500 hw20 SRG(1.8) emax=1.")
    Z = proton_number = 2
    N = neutron_number = 2

    hamil = Hamiltonian(filename_snt, Z, N, ncsm=True, verbose=False)
    n_qubits = hamil.n_qubits
    proton_qubits = list(range(0, hamil.n_qubits_p))
    neutron_qubits = list(range(hamil.n_qubits_p, n_qubits))
    Hdict_M = hamil.get_mscheme_H(opform=True)

    #print(Hdict_M["Vpn"])

    Hdict_M["Vpn"] = { ('+_0 -_0', '+_0 -_0'): -13.77303517, 
                      ('+_1 -_1', '+_1 -_1'): -13.77303517}

    Hdict_M["Vpn"] = { ('+_0 -_0', '+_0 -_0'): -13.77303517}


    mapping_method = "JordanWigner"
    H_1b, H_n, H_p, H_jz, H_pp, H_nn, H_pn = hamil.mapping_opform(Hdict_M, mapping_method)
    h_mapped = H_pn #+ H_1b + H_nn + H_pp

    ### JordanWigner Mapper
    qc = nucl_ansatz(Hdict_M, n_qubits, proton_qubits, neutron_qubits, proton_number, neutron_number,
                     [ ], "HF", mapping_method=mapping_method)
    E_meas_JW = Estimate_Energy(qc, h_mapped, mapping_method)
    for idx in range(len(h_mapped)):
        h = h_mapped[idx]
        e = Estimate_Energy(qc, h, "JordanWigner", verbose=False)
        if e != 0.0: 
            print(f"Term@{idx}:")
            print(f"- JW:   {h.paulis[0]}  {h.coeffs[0]} => {e}")
    print("---\n")

    ### HATTMapper
    mapping_method = "HATTMapper"
    filepath="./test_mapper_4He"
    H_1b, H_n, H_p, H_jz, H_pp, H_nn, H_pn = hamil.mapping_opform(Hdict_M, mapping_method, filepath)
    h_mapped = H_pn #H_1b + H_nn #+ H_pp #+ 

    qc = nucl_ansatz(Hdict_M, n_qubits, proton_qubits, neutron_qubits,
                     proton_number, neutron_number, [ ], "HF",
                     mapping_method=mapping_method, filepath=filepath)
    E_meas_HATT = Estimate_Energy(qc, h_mapped, mapping_method)

    for idx in range(len(h_mapped)):
        h = h_mapped[idx]
        e = Estimate_Energy(qc, h, "HATTMapper", verbose=False)
        if e != 0.0: 
            print(f"Term@{idx}:")
            print(f"- HT:   {h.paulis[0]}  {h.coeffs[0]} => {e}")

    #eval_eachterm(qc, h_mapped, mapping_method)
    print("---\n")

    #assert abs(E_meas_HATT - E_meas_JW) < 1e-5
    #assert abs(E_meas_JW - Eref) < 1e-5    

if __name__ == "__main__":
    test_ncsm_4He_emax0()