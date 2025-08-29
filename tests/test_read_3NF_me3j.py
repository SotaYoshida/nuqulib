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

    for (key, value) in v3b_Mscheme_gzip.items():
        assert abs(value - v3b_Mscheme_txt[key]) < 1.e-5

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