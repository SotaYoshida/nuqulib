import numpy as np
from nuqulib import *


def test_J2_one_particle_j_half():
    msps = [
        Orbit_nljjztz(0, 0, 1, -1, -1),
        Orbit_nljjztz(0, 0, 1, 1, -1),
    ]
    basis, index = fixed_N_P_M_basis(2, 0, 1, 0, msps, 1, 1)

    J2 = total_angular_momentum_squared_matrix(2, 0, basis, index, msps, 1)

    assert np.allclose(J2, [[0.75]])
    assert np.allclose(angular_momentum_from_J2(np.linalg.eigvalsh(J2)), [0.5])


def test_J2_closed_j_half_pair_is_J_zero():
    msps = [
        Orbit_nljjztz(0, 0, 1, -1, -1),
        Orbit_nljjztz(0, 0, 1, 1, -1),
    ]
    basis, index = fixed_N_P_M_basis(2, 0, 2, 0, msps, 1, 0)

    J2 = total_angular_momentum_squared_matrix(2, 0, basis, index, msps, 0)

    assert np.allclose(J2, [[0.0]])
    assert np.allclose(angular_momentum_from_J2(np.linalg.eigvalsh(J2)), [0.0])


def test_J2_proton_neutron_j_half_couples_to_J_zero_and_one():
    msps = [
        Orbit_nljjztz(0, 0, 1, -1, -1),
        Orbit_nljjztz(0, 0, 1, 1, -1),
        Orbit_nljjztz(0, 0, 1, -1, 1),
        Orbit_nljjztz(0, 0, 1, 1, 1),
    ]
    basis, index = fixed_N_P_M_basis(2, 2, 1, 1, msps, 1, 0)

    J2 = total_angular_momentum_squared_matrix(2, 2, basis, index, msps, 0)
    evals, evecs = np.linalg.eigh(J2)
    J2_expect = expectation_values(J2, evecs)

    assert np.allclose(np.sort(evals), [0.0, 2.0])
    assert np.allclose(np.sort(J2_expect), [0.0, 2.0])
    assert np.allclose(np.sort(angular_momentum_from_J2(J2_expect)), [0.0, 1.0])

def test_for_6He():

    fn_snt = "tests/interaction_file/ckpot.snt"
    Z = 2; N = 4; target_parity = 1; Zc = 2; Nc = 2

    hamil = Hamiltonian(fn_snt, Z, N) 
    hamil.get_mscheme_H(opform=True)
    H_1b_p, H_1b_n, H_jz_p, H_jz_n, H_pp, H_nn, H_pn, H_3b = hamil.mapping_opform()
    Hamil_mapped = 0 * H_1b_p
    if Z-Zc > 0:
        Hamil_mapped += H_1b_p
        if Z-Zc > 1:
            Hamil_mapped += H_pp
    if N-Nc > 0:
        Hamil_mapped += H_1b_n
        if N-Nc > 1:
            Hamil_mapped += H_nn

    if (Z-Zc) * (N-Nc) > 0:
        Hamil_mapped += H_pn 

    Hamil_mapped = Hamil_mapped.simplify()
    obj_Diag = Diagonalize_Hamiltonian(Hamil_mapped, hamil, Z, N, target_parity, Zc, Nc, calc_J2=True)
    J_expect = obj_Diag["Jvals"]
    print("<H>:", obj_Diag["evals"])
    Hvals_ref = [-3.90981246, 0.6322095, 4.1172905, 4.2824, 7.92111246]
    assert np.allclose(np.sort(obj_Diag["evals"]), np.sort(Hvals_ref))

    print("<J>:", J_expect)
    Jvals_ref = [0, 2, 2, 1, 0]
    assert np.allclose(np.sort(J_expect), np.sort(Jvals_ref))
    