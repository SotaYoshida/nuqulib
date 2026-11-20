import os
import pytest
from nuqulib import *

chdir = os.path.dirname(os.path.abspath(__file__))
int_dir = os.path.join(chdir, "interaction_file/")

THREE_BODY_PREFIX = (
    "ThBME_srg2.0_ramp40-5-36-7-32-9-28-11-24_N3LO_EM500_"
    "c1_-0.81_c3_-3.2_c4_5.4_cD_0.7_cE_-0.06_LNL2_650_500_"
    "IS_hw20from30_"
)


def test_read_3NF_readable_legacy():
    # Keep coverage of the development-era parser. Its mixed-species
    # M-scheme conversion is not a reference for physical results.
    fn_2NF = int_dir + "TwBME-HO_NN-only_N3LO_EM500_srg2.0_hw20_emax1_e2max2.kshell.snt"
    hamil = Hamiltonian(fn_2NF, 8, 8, ncsm=True, emax_truncate=1,
                        e3max=1, fn_3NF=int_dir + "ThBME_lnl_ms1_2_1.readable.txt")
    assert hamil.v3b_pn


def test_read_3NF_me3j():
    fn_2NF = int_dir + "TwBME-HO_NN-only_N3LO_EM500_srg2.0_hw20_emax1_e2max2.kshell.snt"
    fn_3NF = int_dir + THREE_BODY_PREFIX + "ms1_2_1.me3j.gz"
    hamil = Hamiltonian(fn_2NF, 8, 8, ncsm=True, emax_truncate=1,
                        e3max=1, fn_3NF=fn_3NF)
    matrix_elements = hamil.set_mscheme_3NF()

    # The recommended me3j path stores complete matrix elements between
    # canonical triples. No expanded-permutation normalization is applied.
    assert matrix_elements
    for key, value in matrix_elements.items():
        bra, ket = key[:3], key[3:]
        assert bra == tuple(sorted(bra))
        assert ket == tuple(sorted(ket))
        assert sum(hamil.msps[index].tz for index in bra) == sum(
            hamil.msps[index].tz for index in ket
        )
        reverse_key = (*ket, *bra)
        assert abs(value - matrix_elements[reverse_key]) < 1.e-10


def test_truncate_3bme():
    fn_2NF = int_dir + "TwBME-HO_NN-only_N3LO_EM500_srg2.0_hw20_emax1_e2max2.kshell.snt"
    matrices = []
    # Compare a larger file truncated to e3max=0 with the supplied 0s file.
    for space in ("ms1_2_1", "ms0_0_0"):
        fn_3NF = int_dir + THREE_BODY_PREFIX + space + ".me3j.gz"
        hamil = Hamiltonian(fn_2NF, 8, 8, ncsm=True, emax_truncate=1,
                            e3max=0, fn_3NF=fn_3NF)
        matrix_elements = hamil.set_mscheme_3NF()
        assert matrix_elements
        for key in matrix_elements:
            assert sum(hamil.msps[index].e for index in key[:3]) == 0
            assert sum(hamil.msps[index].e for index in key[3:]) == 0
        matrices.append(matrix_elements)
    assert matrices[0].keys() == matrices[1].keys()
    for key, value in matrices[0].items():
        assert abs(value - matrices[1][key]) < 1.e-5


def test_truncate_3bme_e3max1_and_4He_diagonalization():
    e1max = 1
    e3max = 1
    Z = N = 2
    target_parity = 1

    fn_2NF = int_dir + "TwBME-HO_NN-only_N3LO_EM500_srg2.0_hw20_emax1_e2max2.kshell.snt"
    fn_3NF = int_dir + "ThBME_srg2.0_ramp40-5-36-7-32-9-28-11-24_N3LO_EM500_c1_-0.81_c3_-3.2_c4_5.4_cD_0.7_cE_-0.06_LNL2_650_500_IS_hw20from30_ms1_2_1.me3j.gz"
    hamil_2 = Hamiltonian(fn_2NF, Z, N, ncsm=True, emax_truncate=e1max,
                        e3max=e3max, fn_3NF=fn_3NF)
    v3b_Mscheme_2 = hamil_2.set_mscheme_3NF()

    fn_3NF = int_dir + "ThBME_srg2.0_ramp40-5-36-7-32-9-28-11-24_N3LO_EM500_c1_-0.81_c3_-3.2_c4_5.4_cD_0.7_cE_-0.06_LNL2_650_500_IS_hw20from30_ms1_2_2.me3j.gz"
    hamil_1 = Hamiltonian(fn_2NF, Z, N, ncsm=True, emax_truncate=e1max,
                          e3max=e3max, fn_3NF=fn_3NF)
    v3b_Mscheme_1 = hamil_1.set_mscheme_3NF()

    assert v3b_Mscheme_2.keys() == v3b_Mscheme_1.keys()
    for (key, value) in v3b_Mscheme_2.items():
        tf = abs(value - v3b_Mscheme_1[key]) < 1.e-5
        assert tf, f"key: {key}, value: {v3b_Mscheme_1[key]} v3b_2: {v3b_Mscheme_2[key]}"

    # Exact diagonalization
    H_1b_p, H_1b_n, H_jz_p, H_jz_n, H_pp, H_nn, H_pn, H_3b = hamil_1.mapping_opform()
    Hamil_mapped = H_1b_p + H_1b_n + H_jz_p + H_jz_n + H_pp + H_nn + H_pn + H_3b
    obj_Diag = Diagonalize_Hamiltonian(Hamil_mapped, hamil_1, Z, N, target_parity)
    Ens_exact = obj_Diag["evals"][:10]
    Erefs = [
        -23.82578536, -13.56933504, -11.36316128, 9.14615567,
        9.17431653, 9.64945112, 9.95330695, 13.49562418,
        14.45536711, 15.67231950,
    ]
    Ediffs = np.array(Ens_exact) - np.array(Erefs)
    assert np.all(np.abs(Ediffs) < 1.e-5), f"Energy differences from reference values: {Ediffs}"
    print(f"Ens_exact: {Ens_exact}")
