from nuqulib import *

chdir = os.path.dirname(os.path.abspath(__file__))
int_dir = os.path.join(chdir, "interaction_file/")

def test_pshell_Hamil():
    Z = 2; N = 4
    fn_snt = int_dir+"ckpot.snt"
 
    # using wrapper
    hamil, H_mapped, proton_qubits, neutron_qubits = get_Hamiltonian(fn_snt, Z, N, single_spiecies=2)
    obj_Diag = Diagonalize_Hamiltonian(H_mapped, hamil, Z, N, 1,
                                       hamil.core_p, hamil.core_n, calc_J2=True,
                                       verbose=True)
    J_expect = obj_Diag["Jvals"]
    print("<H>:", obj_Diag["evals"])
    print("<J>:", J_expect)
    Hvals_ref = [-3.90981246, 0.6322095, 4.1172905, 4.2824, 7.92111246]
    assert np.allclose(np.sort(obj_Diag["evals"]), np.sort(Hvals_ref))

def test_NCSM_NN3N_Hamil():

    emax = e3max = 1
    fn_snt = int_dir + f"TwBME-HO_NN-only_N3LO_EM500_srg1.8_hw20_emax{emax}_e2max{emax*2}.kshell.snt"
    fn_3NF = int_dir + "ThBME_lnl_ms1_2_1.readable.txt"    
    
    nuc = "3n"
    Z = 0
    N = 3
    target_parity = -1

    (hamil, 
     H_mapped, 
     proton_qubits, 
     neutron_qubits) = get_Hamiltonian(
        fn_snt, Z, N, 
        fn_3NF=fn_3NF, 
        emax=emax, 
        e3max=e3max,
        ncsm=True, 
        single_spiecies=2
        )

    obj_Diag = Diagonalize_Hamiltonian(H_mapped, hamil, Z, N, target_parity, hamil.core_p, hamil.core_n)
    Ens_exact = obj_Diag["evals"]
    Href = [24.61611688, 28.81430013, 40.29680743, 41.36242406, 41.46721527, 42.64558134, 49.64463354]
    Jref = [1.5, 0.5, 1.5, 2.5, 0.5, 1.5, 1.5]

    assert np.allclose(np.sort(Ens_exact), np.sort(Href))
    assert np.allclose(np.sort(obj_Diag["Jvals"]), np.sort(Jref))