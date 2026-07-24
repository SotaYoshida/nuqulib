import os
import pytest
from nuqulib import *

chdir = os.path.dirname(os.path.abspath(__file__))
int_dir = os.path.join(chdir, "interaction_file/")

def test_read_3NF_files():
    e1max = e3max = 1
    Z = N = 8

    fn_2NF = int_dir + "TwBME-HO_NN-only_N3LO_EM500_srg1.8_hw20_emax1_e2max2.kshell.snt"

    fn_3NF_txt = int_dir + "ThBME_lnl_ms1_2_1.readable.txt"
    hamil_txt = Hamiltonian(fn_2NF, Z, N, ncsm=True, emax_truncate=e1max,
                            e3max=e3max, fn_3NF=fn_3NF_txt)
    v3b_Mscheme_txt = hamil_txt.set_mscheme_3NF()

    fn_3NF_gzip = int_dir + "ThBME_srg2.0_ramp40-5-36-7-32-9-28-11-24_N3LO_EM500_c1_-0.81_c3_-3.2_c4_5.4_cD_0.7_cE_-0.06_LNL2_650_500_IS_hw20from30_ms1_2_1.me3j.gz"
    hamil_gzip = Hamiltonian(fn_2NF, Z, N, ncsm=True, emax_truncate=e1max,
                             e3max=e3max, fn_3NF=fn_3NF_gzip)
    v3b_Mscheme_gzip = hamil_gzip.set_mscheme_3NF()

    # The me3j.gz path now uses the direct antisymmetrized m-scheme
    # recoupling route.  It intentionally has a different normalization from
    # the legacy readable.txt conversion, which expanded mixed-species
    # permutations without spatial recoupling.
    assert v3b_Mscheme_txt
    assert v3b_Mscheme_gzip
    for (key, value) in v3b_Mscheme_gzip.items():
        bra, ket = key[:3], key[3:]
        assert bra == tuple(sorted(bra))
        assert ket == tuple(sorted(ket))
        assert sum(hamil_gzip.msps[index].tz for index in bra) == sum(
            hamil_gzip.msps[index].tz for index in ket
        )
        reverse_key = (*ket, *bra)
        assert abs(value - v3b_Mscheme_gzip[reverse_key]) < 1.e-10

def test_truncate_3bme():
    e1max = 1
    e3max = 0
    Z = N = 8

    fn_2NF = int_dir + "TwBME-HO_NN-only_N3LO_EM500_srg1.8_hw20_emax1_e2max2.kshell.snt"

    fn_3NF = int_dir + "ThBME_lnl_ms1_2_1.readable.txt"
    hamil_2 = Hamiltonian(fn_2NF, Z, N, ncsm=True, emax_truncate=e1max,
                        e3max=e3max, fn_3NF=fn_3NF)
    v3b_Mscheme_2 = hamil_2.set_mscheme_3NF()

    fn_3NF = int_dir + "ThBME_lnl_ms1_2_0.readable.txt"
    hamil_1 = Hamiltonian(fn_2NF, Z, N, ncsm=True, emax_truncate=e1max,
                          e3max=e3max, fn_3NF=fn_3NF)
    v3b_Mscheme_1 = hamil_1.set_mscheme_3NF()

    for (key, value) in v3b_Mscheme_1.items():
        assert abs(value - v3b_Mscheme_2[key]) < 1.e-5

def test_truncate_3bme_e3max1_and_4He_diagonalization():
    e1max = 1
    e3max = 1
    Z = N = 2
    target_parity = 1

    fn_2NF = int_dir + "TwBME-HO_NN-only_N3LO_EM500_srg1.8_hw20_emax1_e2max2.kshell.snt"
    fn_3NF = int_dir + "ThBME_srg2.0_ramp40-5-36-7-32-9-28-11-24_N3LO_EM500_c1_-0.81_c3_-3.2_c4_5.4_cD_0.7_cE_-0.06_LNL2_650_500_IS_hw20from30_ms1_2_1.me3j.gz"
    hamil_2 = Hamiltonian(fn_2NF, Z, N, ncsm=True, emax_truncate=e1max,
                        e3max=e3max, fn_3NF=fn_3NF)
    v3b_Mscheme_2 = hamil_2.set_mscheme_3NF()

    fn_3NF = int_dir + "ThBME_srg2.0_ramp40-5-36-7-32-9-28-11-24_N3LO_EM500_c1_-0.81_c3_-3.2_c4_5.4_cD_0.7_cE_-0.06_LNL2_650_500_IS_hw20from30_ms1_2_2.me3j.gz"
    hamil_1 = Hamiltonian(fn_2NF, Z, N, ncsm=True, emax_truncate=e1max,
                          e3max=e3max, fn_3NF=fn_3NF)
    v3b_Mscheme_1 = hamil_1.set_mscheme_3NF()

    count = 0
    for (key, value) in v3b_Mscheme_2.items():
        tf = abs(value - v3b_Mscheme_1[key]) < 1.e-5
        assert tf, f"key: {key}, value: {v3b_Mscheme_1[key]} v3b_2: {v3b_Mscheme_2[key]}"

    # Exact diagonalization
    H_1b_p, H_1b_n, H_jz_p, H_jz_n, H_pp, H_nn, H_pn, H_3b = hamil_1.mapping_opform()
    Hamil_mapped = H_1b_p + H_1b_n + H_jz_p + H_jz_n + H_pp + H_nn + H_pn + H_3b
    obj_Diag = Diagonalize_Hamiltonian(Hamil_mapped, hamil_1, Z, N, target_parity)
    Ens_exact = obj_Diag["evals"][:10]
    Erefs = [-25.65386984096901, -15.430812854039264,
             -13.162040300702188, 7.450992058125176,
             7.709801976779019, 8.058982941831085,
             8.62959853761564, 12.045818975516676,
             13.358830819917609, 14.266854908670563]
    Ediffs = np.array(Ens_exact) - np.array(Erefs)
    assert np.all(np.abs(Ediffs) < 1.e-5), f"Energy differences from reference values: {Ediffs}"
    print(f"Ens_exact: {Ens_exact}")
