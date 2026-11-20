from qiskit import transpile
from qiskit_aer import AerSimulator
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


def test_pshell_ckpot_lowest_filling(Eref=-18.69821):
    Z = N = 4
    fn_snt = int_dir+"ckpot.snt"
 
    hamil, H_mapped, proton_qubits, neutron_qubits = get_Hamiltonian(fn_snt, Z, N)
    Zc = hamil.core_p
    Nc = hamil.core_n
    Mtot = 0
    return_idxs = True

    Uprep, selected_p, selected_n = lowest_filling_ansatz(hamil, Z-Zc, N-Nc, Mtot, return_idxs)
    print("lowest filling ansatz circuit:")
    print("Selected proton qubits: ", selected_p)
    print("Selected neutron qubits:", selected_n)

    sim = AerSimulator(method='statevector')
    psi = Uprep.remove_final_measurements(inplace=False)
    psi.save_statevector()

    tqc = transpile(psi, sim)
    job = sim.run(tqc)
    result = job.result()
    psi_final = result.get_statevector(tqc)
    E_expect = psi_final.expectation_value(H_mapped).real
    print(f"Energy expectation value of the prepared state: {E_expect:.6f} MeV")
    assert np.isclose(E_expect, Eref, rtol=1e-4)

def test_NCSM_NN3N_Hamil():

    emax = e3max = 1
    fn_snt = int_dir + f"TwBME-HO_NN-only_N3LO_EM500_srg2.0_hw20_emax{emax}_e2max{emax*2}.kshell.snt"
    fn_3NF = int_dir + "ThBME_srg2.0_ramp40-5-36-7-32-9-28-11-24_N3LO_EM500_c1_-0.81_c3_-3.2_c4_5.4_cD_0.7_cE_-0.06_LNL2_650_500_IS_hw20from30_ms1_2_1.me3j.gz"
    
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
    Href = [
        24.61348634, 28.91943721, 40.30246349, 41.36536075,
        41.50407100, 42.68053007, 49.77167448,
    ]
    Jref = [1.5, 0.5, 1.5, 2.5, 0.5, 1.5, 1.5]

    assert np.allclose(np.sort(Ens_exact), np.sort(Href))
    assert np.allclose(np.sort(obj_Diag["Jvals"]), np.sort(Jref))