import numpy as np
from nuqulib import get_Hamiltonian
from commutation_utils import *
import argparse

root_dir = "../"
int_file_path = root_dir + "tests/interaction_file/"
    
which_one_to_use = {
    "p_shell": ["ckpot.snt", None, "Be8", 4, 4, 2, 2, None, None],
    "sd_shell": ["usdb.snt", None, "Ne20", 10, 10, 8, 8, None, None],
    "pf_shell": ["gxpf1a.snt", None, "Ti44", 22, 22, 20, 20, None, None],
    "psd_shell": ["ysox.snt", None, "C12", 6, 6, 2, 2, None, None],
    "NN-only_0": ["TwBME-HO_NN-only_N3LO_EM500_srg1.8_hw20_emax0_e2max0.kshell.snt", None, "4He",   2,  2, 0, 0, 0, None],
    "NN-only_1": ["TwBME-HO_NN-only_N3LO_EM500_srg1.8_hw20_emax1_e2max2.kshell.snt", None, "16O",   8,  8, 0, 0, 1, None],
    "NN-only_2": ["TwBME-HO_NN-only_N3LO_EM500_srg1.8_hw20_emax2_e2max4.kshell.snt", None, "40Ca", 20, 20, 0, 0, 2, None],
    "NN-only_3": ["TwBME-HO_NN-only_N3LO_EM500_srg1.8_hw20_emax3_e2max6.kshell.snt", None, "80Zr", 40, 40, 0, 0, 3, None],
    "NN+3NF_1_2_3": ["TwBME-HO_NN-only_N3LO_EM500_srg1.8_hw20_emax1_e2max2.kshell.snt", f"ThBME_lnl_ms1_2_3.readable.txt",
                     "16O", 8, 8, 0, 0, 1, 3],
    "NN+3NF_2_4_6": ["TwBME-HO_NN-only_N3LO_EM500_srg1.8_hw20_emax2_e2max4.kshell.snt", f"ThBME_srg2.0_ramp40-5-36-7-32-9-28-11-24_N3LO_EM500_c1_-0.81_c3_-3.2_c4_5.4_cD_0.7_cE_-0.06_LNL2_650_500_IS_hw20from30_ms2_4_6.me3j.gz",
                     "40Ca", 20, 20, 0, 0, 2, 6]
}

def get_paulis(target):
    snt_name, fn_3NF, nuc, Z, N, Zc, Nc, emax, e3max = which_one_to_use[target]
    
    fn_snt = int_file_path+snt_name
    fn_3NF = int_file_path+fn_3NF if fn_3NF is not None else None
    ncsm = True if Zc == Nc == 0 else False
    
    if fn_3NF is not None:
        print(f"Using 3NF from {fn_3NF}")
        hamil, Hamil_mapped, proton_qubits, neutron_qubits = get_Hamiltonian(fn_snt, Z, N, fn_3NF=fn_3NF, emax=emax, e3max=e3max, ncsm=ncsm)
        n_qubit = hamil.n_qubits
    else:
        if ncsm:
            print("Using NCSM Hamiltonian without 3NF")
            hamil, Hamil_mapped, proton_qubits, neutron_qubits = get_Hamiltonian(fn_snt, Z, N, fn_3NF=None, emax=emax, ncsm=True)
        else:
            hamil, Hamil_mapped, proton_qubits, neutron_qubits = get_Hamiltonian(fn_snt, Z, N)
    
    Nq = hamil.n_qubits
    paulis = Hamil_mapped.paulis
    coeffs = Hamil_mapped.coeffs
    return Nq, coeffs, paulis

    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", default=1, type=int, help="which target to use")
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    
    keys = ["p_shell","sd_shell","pf_shell","psd_shell","NN-only_0","NN-only_1","NN-only_2","NN+3NF_1_2_3","NN+3NF_2_4_6"]
    target = keys[args.which]
    
    result={}
    Nq, coeffs, paulis = get_paulis(target)

    groups = commuting_groups_chunked(paulis, chunk_size = min(len(paulis),60000), return_type="indices")
    total_noncomm, sum_coeffs = noncomm_cross_group_count_and_coeff_product_sum(paulis, groups, abs(np.real(coeffs)))
    result[target]={"Nq": Nq, "paulis": paulis, "coeffs": coeffs, "groups": groups, "non-commuting": total_noncomm, "sum-coeffs": sum_coeffs}
    
    print(f"Target: {target}")
    print(f"groups: {len(groups)}")
    print(f"# of terms: {len(paulis)} non-commuting pairs: {total_noncomm} sum of coeff products: {sum_coeffs}")
    red_factor = (len(paulis) * (len(paulis)-1) / 2) / total_noncomm if total_noncomm > 0 else float('inf')
    print(f"Reduction factor: {red_factor}")

    if target == "NN-only_0" or target == "p_shell":  # print details for small cases
        print("This is a small test case, printing all details:")
        for i, group in enumerate(groups):
            print(f"Group {i}: size {len(group)}")
            for idx in group:
                print(f"  Pauli: {paulis[idx]}, coeff: {coeffs[idx]}")
        print()
    np.savez("Pauli_count_"+str(target)+".npz",result=result)


if __name__ == "__main__":
    main()
    
