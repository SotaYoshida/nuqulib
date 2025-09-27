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

def eval_eachterm(qc, h_mapped, mapping_method, max_eval_num=10):
    for i in range(max_eval_num):
        Estimate_Energy(qc, h_mapped[i], mapping_method)

def length_checker(H_pn, n_qubits):
    TF = True
    for term in H_pn:
        paulistr = term.paulis[0].to_label()
        coeff = term.coeffs[0]
        if len(paulistr) != n_qubits:
            print("length doesn't match!!", paulistr)
            TF = False
    return TF

# def test_ckpot():
#     filename_snt = int_dir + "ckpot.snt"

#     for (Z, N) in [(0,2), (2,0)]:
#         proton_number = Z
#         neutron_number = N

#         hamil = Hamiltonian(filename_snt, Z+2, N+2)
                            
#         n_qubits = hamil.n_qubits
#         proton_qubits = list(range(0, hamil.n_qubits_p))
#         neutron_qubits = list(range(hamil.n_qubits_p, n_qubits))
#         Hdict_M = hamil.get_mscheme_H(opform=True)

#         mapping_method = "JordanWigner"
#         H_1b_p, H_1b_n, H_jz_p, H_jz_n, H_pp, H_nn, H_pn, H_3b = hamil.mapping_opform(Hdict_M, mapping_method)
#         h_mapped = H_1b_p + H_1b_n + H_pn + H_nn + H_pp 

#         ### JordanWigner Mapper
#         qc = nucl_ansatz(Hdict_M, n_qubits, proton_qubits, neutron_qubits, proton_number, neutron_number,
#                         [ ], "HF", mapping_method=mapping_method)
#         E_meas_JW = Estimate_Energy(qc, h_mapped, mapping_method)
#         for idx in range(len(h_mapped)):
#             h = h_mapped[idx]
#             e = Estimate_Energy(qc, h, "JordanWigner", verbose=False)
#             if e != 0.0: 
#                 print(f"Term@{idx}:")
#                 print(f"- JW:   {h.paulis[0]}  {h.coeffs[0]} => {e}")
#         print("---\n")

#         ### HATTMapper
#         mapping_method = "HATTMapper"
#         filepath="./test_mapper_ckpot"
#         H_1b_p, H_1b_n, H_jz_p, H_jz_n, H_pp, H_nn, H_pn, H_3b = hamil.mapping_opform(Hdict_M, mapping_method, filepath)

#         # length checker
#         print("H_1b_p...", length_checker(H_1b_p, n_qubits))
#         print("H_1b_n...", length_checker(H_1b_n, n_qubits))
#         print("H_pp...", length_checker(H_pp, n_qubits))
#         print("H_nn...", length_checker(H_nn, n_qubits))
#         print("H_pn...", length_checker(H_pn, n_qubits))
                
        
#         h_mapped = H_1b_p + H_1b_n + H_nn + H_pp + H_pn 

#         qc = nucl_ansatz(Hdict_M, n_qubits, proton_qubits, neutron_qubits,
#                         proton_number, neutron_number, [ ], "HF",
#                         mapping_method=mapping_method, filepath=filepath)
#         E_meas_HATT = Estimate_Energy(qc, h_mapped, mapping_method)

#         for idx in range(len(h_mapped)):
#             h = h_mapped[idx]
#             e = Estimate_Energy(qc, h, "HATTMapper", verbose=False)
#             if e != 0.0: 
#                 print(f"Term@{idx}:")
#                 print(f"- HT:   {h.paulis[0]}  {h.coeffs[0]} => {e}")

#         #print(qc.decompose().draw())

#         #eval_eachterm(qc, h_mapped, mapping_method)
#         print("---\n")

#         assert abs(E_meas_HATT - E_meas_JW) < 1e-5


def test_ncsm_4He_emax0(Eref=-22.99767,
                        filename_snt=int_dir+"TwBME-HO_NN-only_N3LO_EM500_srg1.8_hw20_emax0_e2max0.kshell.snt",
                        fn_3NF=None):
                        #fn_3NF=int_dir+"ThBME_srg2.0_ramp40-5-36-7-32-9-28-11-24_N3LO_EM500_c1_-0.81_c3_-3.2_c4_5.4_cD_0.7_cE_-0.06_LNL2_650_500_IS_hw20from30_ms1_2_1.me3j.gz"):
    Z = proton_number = 0
    N = neutron_number = 2

    hamil = Hamiltonian(filename_snt, Z, N, ncsm=True, verbose=False,
                        fn_3NF=fn_3NF, emax_truncate=0, e3max=0)
    n_qubits = hamil.n_qubits
    proton_qubits = list(range(0, hamil.n_qubits_p))
    neutron_qubits = list(range(hamil.n_qubits_p, n_qubits))
    Hdict_M = hamil.get_mscheme_H(opform=True)

    mapping_method = "JordanWigner"
    H_1b_p, H_1b_n, H_jz_p, H_jz_n, H_pp, H_nn, H_pn, H_3b = hamil.mapping_opform(Hdict_M, mapping_method)

    # print(f"H_1b_p {H_1b_p}")
    # print(f"H_1b_n {H_1b_n}")
    # print(f"H_nn {H_nn}")
    # print(f"H_pp {H_pp}")
    # print(f"H_pn {H_pn}")

    h_mapped = H_1b_p + H_1b_n + H_nn + H_pp #+ H_pn
    if Z == 0 and N== 2:
        h_mapped = H_1b_n + H_nn 
    elif Z == 2 and N==0:
        h_mapped = H_1b_p + H_pp 

    if fn_3NF is not None:
        h_mapped += H_3b

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
    H_1b_p, H_1b_n, H_jz_p, H_jz_n, H_pp, H_nn, H_pn, H_3b = hamil.mapping_opform(Hdict_M, mapping_method, filepath)

    print(f"H_1b_p {H_1b_p}")
    print(f"H_1b_n {H_1b_n}")

    #h_mapped = H_1b_p + H_1b_n + H_nn + H_pp + H_pn 
    if Z == 0 and N== 2:
        h_mapped = H_1b_n + H_nn 
    elif Z == 2 and N==0:
        h_mapped = H_1b_p + H_pp 


    if fn_3NF is not None:
        h_mapped += H_3b

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

    #print(qc.decompose().draw())

    #eval_eachterm(qc, h_mapped, mapping_method)
    print("---\n")

    assert abs(E_meas_HATT - E_meas_JW) < 1e-5
    #assert abs(E_meas_JW - Eref) < 1e-5    


# def test_ncsm_160_emax1(Eref=-131.835579,
#                         filename_snt=int_dir+"TwBME-HO_NN-only_N3LO_EM500_srg1.8_hw20_emax1_e2max2.kshell.snt",
#                         fn_3NF=None):
#                         #fn_3NF=int_dir+"ThBME_srg2.0_ramp40-5-36-7-32-9-28-11-24_N3LO_EM500_c1_-0.81_c3_-3.2_c4_5.4_cD_0.7_cE_-0.06_LNL2_650_500_IS_hw20from30_ms1_2_1.me3j.gz"):
#     Z = proton_number = 8
#     N = neutron_number = 8

#     hamil = Hamiltonian(filename_snt, Z, N, ncsm=True, verbose=False,
#                         fn_3NF=fn_3NF, emax_truncate=1, e3max=1)
#     n_qubits = hamil.n_qubits
#     proton_qubits = list(range(0, hamil.n_qubits_p))
#     neutron_qubits = list(range(hamil.n_qubits_p, n_qubits))
#     Hdict_M = hamil.get_mscheme_H(opform=True)

#     mapping_method = "JordanWigner"
#     H_1b_p, H_1b_n, H_jz_p, H_jz_n, H_pp, H_nn, H_pn, H_3b = hamil.mapping_opform(Hdict_M, mapping_method)
#     h_mapped = H_1b_p + H_1b_n + H_pn + H_nn + H_pp 
#     if fn_3NF is not None:
#         h_mapped += H_3b

#     ### JordanWigner Mapper
#     qc = nucl_ansatz(Hdict_M, n_qubits, proton_qubits, neutron_qubits, proton_number, neutron_number,
#                      [ ], "HF", mapping_method=mapping_method)
#     E_meas_JW = Estimate_Energy(qc, h_mapped, mapping_method)
#     # for idx in range(len(h_mapped)):
#     #     h = h_mapped[idx]
#     #     e = Estimate_Energy(qc, h, "JordanWigner", verbose=False)
#     #     if e != 0.0: 
#     #         print(f"Term@{idx}:")
#     #         print(f"- JW:   {h.paulis[0]}  {h.coeffs[0]} => {e}")
#     # print("---\n")

#     ### HATTMapper
#     mapping_method = "HATTMapper"
#     filepath="./test_mapper_16O"
#     H_1b_p, H_1b_n, H_jz_p, H_jz_n, H_pp, H_nn, H_pn, H_3b = hamil.mapping_opform(Hdict_M, mapping_method, filepath)
#     h_mapped = H_1b_p + H_1b_n + H_nn + H_pp + H_pn 
#     if fn_3NF is not None:
#         h_mapped += H_3b

#     qc = nucl_ansatz(Hdict_M, n_qubits, proton_qubits, neutron_qubits,
#                      proton_number, neutron_number, [ ], "HF",
#                      mapping_method=mapping_method, filepath=filepath)
#     E_meas_HATT = Estimate_Energy(qc, h_mapped, mapping_method)

#     for idx in range(len(h_mapped)):
#         h = h_mapped[idx]
#         e = Estimate_Energy(qc, h, "HATTMapper", verbose=False)
#         if e != 0.0: 
#             print(f"Term@{idx}:")
#             print(f"- HT:   {h.paulis[0]}  {h.coeffs[0]} => {e}")

#     #eval_eachterm(qc, h_mapped, mapping_method)
#     print("---\n")

#     assert abs(E_meas_HATT - E_meas_JW) < 1e-5
#     assert abs(E_meas_JW - Eref) < 1e-5    

# def test_ckpot_16O(Eref=-113.814,
#                     filename_snt=int_dir+"ckpot.snt",
#                     fn_3NF=None):
#     Z = 8; proton_number = 6
#     Z = 2; proton_number = 0
    
#     N = 8; neutron_number = 6
#     Z = 8; proton_number = 6; N = 2; neutron_number = 0


#     hamil = Hamiltonian(filename_snt, Z, N)
#     n_qubits = hamil.n_qubits
#     proton_qubits = list(range(0, hamil.n_qubits_p))
#     neutron_qubits = list(range(hamil.n_qubits_p, n_qubits))
#     Hdict_M = hamil.get_mscheme_H(opform=True)

#     mapping_method = "JordanWigner"
#     H_1b_p, H_1b_n, H_jz_p, H_jz_n, H_pp, H_nn, H_pn, H_3b = hamil.mapping_opform(Hdict_M, mapping_method)
#     #h_mapped = H_1b_p + H_1b_n + H_pn + H_nn + H_pp 
#     h_mapped = H_1b_p + H_pp 

#     ### JordanWigner Mapper
#     qc = nucl_ansatz(Hdict_M, n_qubits, proton_qubits, neutron_qubits, proton_number, neutron_number,
#                      [ ], "HF", mapping_method=mapping_method)
#     E_meas_JW = Estimate_Energy(qc, h_mapped, mapping_method)

#     for idx in range(len(h_mapped)):
#         h = h_mapped[idx]
#         e = Estimate_Energy(qc, h, "JordanWigner", verbose=False)
#         if e != 0.0: 
#             print(f"Term@{idx}:")
#             print(f"- JW:   {h.paulis[0]}  {h.coeffs[0]} => {e}")

#     ### HATTMapper
#     mapping_method = "HATTMapper"
#     filepath="./test_mapper_16O"
#     H_1b_p, H_1b_n, H_jz_p, H_jz_n, H_pp, H_nn, H_pn, H_3b = hamil.mapping_opform(Hdict_M, mapping_method, filepath)
#     #h_mapped = H_1b_p + H_1b_n + H_nn + H_pp + H_pn 
#     h_mapped = H_1b_p + H_pp 


#     qc = nucl_ansatz(Hdict_M, n_qubits, proton_qubits, neutron_qubits,
#                      proton_number, neutron_number, [ ], "HF",
#                      mapping_method=mapping_method, filepath=filepath)
#     E_meas_HATT = Estimate_Energy(qc, h_mapped, mapping_method)
#     for idx in range(len(h_mapped)):
#         h = h_mapped[idx]
#         e = Estimate_Energy(qc, h, "HATTMapper", verbose=False)
#         if e != 0.0: 
#             print(f"Term@{idx}:")
#             print(f"- HT:   {h.paulis[0]}  {h.coeffs[0]} => {e}")

#     #eval_eachterm(qc, h_mapped, mapping_method)
#     print("---\n")

#     assert abs(E_meas_HATT - E_meas_JW) < 1e-5
#     assert abs(E_meas_JW - Eref) < 1e-5    
