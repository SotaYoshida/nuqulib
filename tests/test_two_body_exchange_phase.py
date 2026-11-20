"""Two-body matrix elements must not depend on the ordering of a pair."""

import copy
from itertools import combinations

import numpy as np
import pytest

from nuqulib.nuclear_hamiltonian import Hamiltonian


# (n, l, 2j) of the second orbit, J, and the fermionic exchange phase.
# The first orbit is 0s1/2. Different j values also expose confusion between
# orbital indices and indices into the expanded m-scheme single-particle list.
PAIR_CASES = [
    pytest.param((1, 0, 1), 0, +1, id="s-half-s-half-J0"),
    pytest.param((1, 0, 1), 1, -1, id="s-half-s-half-J1"),
    pytest.param((0, 2, 3), 1, +1, id="s-half-d-three-halves-J1"),
    pytest.param((0, 2, 3), 2, -1, id="s-half-d-three-halves-J2"),
]
ORDERINGS = [
    pytest.param(False, False, id="canonical"),
    pytest.param(True, False, id="bra-swapped"),
    pytest.param(False, True, id="ket-swapped"),
    pytest.param(True, True, id="both-swapped"),
]


def write_pair_interaction(path, second_orbit, J, columns, swap_bra, swap_ket, phase):
    """Write one normalized J-coupled matrix element between distinct orbits."""
    n, l, j2 = second_orbit
    a, b = (2, 1) if swap_bra else (1, 2)
    c, d = (2, 1) if swap_ket else (1, 2)
    value = 1.75 * (phase if swap_bra else 1) * (phase if swap_ket else 1)
    # Seven-column NuHamil inputs have a separate kinetic-energy coefficient.
    ncsm = columns == 7
    one_body_header = "2 10 20.0" if ncsm else "2 0"
    two_body_header = "3 10 20.0" if ncsm else "3 0"
    kinetic = " 0.0" if ncsm else ""
    path.write_text(
        f"2 0 0 0\n1 0 0 1 -1\n2 {n} {l} {j2} -1\n"
        f"{one_body_header}\n1 1 0.0\n2 2 0.0\n{two_body_header}\n"
        f"1 1 1 1 0 0.0{kinetic}\n2 2 2 2 0 0.0{kinetic}\n"
        f"{a} {b} {c} {d} {J} {value}{kinetic}\n"
    )
    return Hamiltonian(str(path), Z=2, N=0, ncsm=ncsm, emax_truncate=2, e3max=0)


def two_particle_spectrum(operators, n_orbitals):
    """Apply a†_a a†_b a_d a_c to ordered two-fermion determinants."""
    basis = list(combinations(range(n_orbitals), 2))
    index = {pair: i for i, pair in enumerate(basis)}
    matrix = np.zeros((len(basis), len(basis)))
    for term, value in operators.items():
        a, b, d, c = [int(op.split("_")[1]) for op in term.split()]
        assert a < b and c < d
        matrix[index[a, b], index[c, d]] += value
    return np.linalg.eigvalsh(matrix)


def assert_pair_spectrum(hamiltonian, operators, J):
    # V times the projector onto this J multiplet has 2J+1 eigenvalues V.
    eigenvalues = two_particle_spectrum(operators, hamiltonian.n_qubits_p)
    expected = np.zeros_like(eigenvalues)
    expected[-(2 * J + 1):] = 1.75
    np.testing.assert_allclose(eigenvalues, expected, atol=1.e-12, rtol=0)


@pytest.mark.parametrize("second_orbit,J,phase", PAIR_CASES)
@pytest.mark.parametrize("swap_bra,swap_ket", ORDERINGS)
@pytest.mark.parametrize("columns", [6, 7])
def test_snt_pair_order_preserves_matrix_element_and_spectrum(
    tmp_path, second_orbit, J, phase, swap_bra, swap_ket, columns
):
    hamil = write_pair_interaction(
        tmp_path / "pair_hw20_emax2_e2max4.snt",
        second_orbit, J, columns, swap_bra, swap_ket, phase,
    )
    assert hamil.v2b[-1] == (0, 1, 0, 1, J, 1.75)
    operators = hamil.get_mscheme_H(opform=True)["Vpp"]
    assert_pair_spectrum(hamil, operators, J)


@pytest.mark.parametrize("second_orbit,J,phase", PAIR_CASES)
@pytest.mark.parametrize("swap_bra,swap_ket", ORDERINGS[1:])
def test_mscheme_conversion_accepts_noncanonical_pair_order(
    tmp_path, second_orbit, J, phase, swap_bra, swap_ket
):
    canonical = write_pair_interaction(
        tmp_path / "pair.snt", second_orbit, J, 6, False, False, phase,
    )
    expected = canonical.get_mscheme_H(opform=True)["Vpp"]
    reordered = copy.deepcopy(canonical)
    a, b = (1, 0) if swap_bra else (0, 1)
    c, d = (1, 0) if swap_ket else (0, 1)
    value = 1.75 * (phase if swap_bra else 1) * (phase if swap_ket else 1)
    reordered.v2b = [(a, b, c, d, J, value)]
    actual = reordered.get_mscheme_H(opform=True)["Vpp"]
    assert actual == pytest.approx(expected, abs=1.e-12)
    assert_pair_spectrum(reordered, actual, J)
