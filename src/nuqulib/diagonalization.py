"""Functions to diagonalize nuclear Hamiltonians


"""
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from .nuclear_hamiltonian import Hamiltonian
from .operators import (
    _state_Mtot,
    _ladder_coefficient,
    _angular_momentum_ladder_terms,
    _apply_fermion_one_body_term,
    _apply_one_body_operator
)

Pauli_I = np.array([[1, 0], [0, 1]], dtype=complex)
Pauli_X = np.array([[0, 1], [1, 0]], dtype=complex)
Pauli_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Pauli_Z = np.array([[1, 0], [0, -1]], dtype=complex)
pauli_dict = {'I': Pauli_I, 'X': Pauli_X, 'Y': Pauli_Y, 'Z': Pauli_Z}

def Diagonalize_Hamiltonian(
        Hamil_mapped: SparsePauliOp,
        hamil: Hamiltonian, 
        Z: int, N: int, 
        target_parity: int, 
        Zc: int = 0, 
        Nc: int = 0,
        calc_J2: bool = True,
        verbose: bool = False,
        use_basis: str = "NPM",
    ) -> dict:
    assert abs(target_parity) == 1, "parity should be ±1"
    n_qubit_p = hamil.n_qubits_p
    n_qubit_n = hamil.n_qubits_n
    Mtot = (Z+N-Zc-Nc) % 2
    if verbose:
        print(f"Z = {Z}, N = {N}, Zc = {Zc}, Nc = {Nc}, Mtot = {Mtot}")
    # For neutron-only cases, we can treat them as "protons" in the code by swapping Z/N and setting N to zero.
    # This allows us to reuse the same basis generation and Hamiltonian construction logic without needing separate code paths for neutron-only systems.
    if hamil.single_spiecies == 2:        
        Z = N
        N = Nc
        n_qubit_p, n_qubit_n = n_qubit_n, n_qubit_p

    # Generate the M-projected basis states and the corresponding Hamiltonian matrix in that basis
    if use_basis == "NPM":
        basis, index = fixed_N_P_M_basis(n_qubit_p, n_qubit_n, Z-Zc, N-Nc,
                                         hamil.msps, target_parity, Mtot, verbose)
    elif use_basis == "NP":
        basis, index = fixed_N_P_basis(n_qubit_p, n_qubit_n, Z-Zc, N-Nc, target_parity, hamil.msps)
    else:
        basis = list(range(1 << (n_qubit_p + n_qubit_n)))
        index = {s: i for i, s in enumerate(basis)}

    if verbose:
        print(f"Total qubits (p+n): {n_qubit_p}, {n_qubit_n}, Protons: {Z}, Neutrons: {N}")
        print(f"basis generated {basis}")
    Hsub = Mprojected_hamiltonian(Hamil_mapped, n_qubit_p, n_qubit_n, basis, index)
    if len(basis) > 4000:
        print("Warning: Large dimension in diagonalization:", len(basis))
        print("One may consider to use Krylov subspace methods.")
    if verbose:
        print("Diagonalizing the Hamiltonian...")
    evals, evecs = np.linalg.eigh(Hsub)
    if verbose and use_basis == "NPM":
        print(f"dim. (N・M・P-projected; M={Mtot}):", len(basis))

    # Sort idxs by their contribution to ground state energy
    idxs_sorted = np.argsort(np.abs(evecs[:, 0]))[::-1]
    result = {
        "basis": basis,
        "Hsub": Hsub,
        "evals": evals,
        "idxs_sorted": idxs_sorted,
        "evecs": evecs,
        "Mtot": Mtot,
    }
    if calc_J2:
        J2sub = total_angular_momentum_squared_matrix(
            n_qubit_p, n_qubit_n, basis, index, hamil.msps, Mtot
        )
        J2_expect = expectation_values(J2sub, evecs)
        J_values = angular_momentum_from_J2(J2_expect)
        result["J2sub"] = J2sub
        result["J2_expect"] = J2_expect
        result["Jvals"] = J_values
    return result


def fixed_N_P_M_basis_neutron(n_qubits, n_particles, msps, parity, M_tot): # M is doubled
    basis = []
    for s in range(1 << n_qubits):
        if s.bit_count() == n_particles:
            Mz = 0
            parity_ = 1
            for i in range(n_qubits):
                bit = (s >> (n_qubits - 1 - i)) & 1
                if bit == 0:
                    continue
                m = msps[-1-i].jz
                Mz += m
                parity_ *= (-1)**(msps[-1-i].l)
            if Mz == M_tot and parity_ == parity:
                basis.append(s)
                # bitstr = format(s, f'0{n_qubits}b')
                # print("M(OK)?:", bitstr)
    index = {s: i for i, s in enumerate(basis)}
    return basis, index


def Mprojected_hamiltonian_single(hamiltonian, n_qubits, basis, index):
    dim = len(basis)
    H = np.zeros((dim, dim), dtype=complex)
    coeffs = hamiltonian.coeffs
    paulis = [ tmp.to_label() for tmp in hamiltonian.paulis ]

    for coeff, pauli in zip(coeffs, paulis):
        for col, state in enumerate(basis):
            new_state, phase = apply_pauli_string_single(pauli, state, n_qubits)

            if new_state in index:
                row = index[new_state]
                H[row, col] += coeff * phase
    H = 0.5 * (H + H.conj().T)
    return H


def Mprojected_hamiltonian(hamiltonian: SparsePauliOp,
                           n_qubits_p: int, n_qubits_n: int, 
                           basis: list[int],
                           index: dict[int, int]) -> np.ndarray:
    """
    Build the M-projected Hamiltonian for a Hamiltonian whose Pauli strings may
    act on protons, neutrons, or both. Treat proton and neutron parts independently
    by padding shorter Pauli labels appropriately.
    """
    dim = len(basis)
    H = np.zeros((dim, dim), dtype=complex)
    coeffs = hamiltonian.coeffs
    paulis = [ tmp.to_label() for tmp in hamiltonian.paulis ]

    for coeff, pauli in zip(coeffs, paulis):
        for col, state in enumerate(basis):
            new_state, phase = apply_pauli_string_pn(pauli, state, n_qubits_p, n_qubits_n)
            if new_state in index:
                row = index[new_state]
                H[row, col] += coeff * phase

    # Enforce Hermiticity
    H = 0.5 * (H + H.conj().T)
    return H


def fixed_N_P_basis(n_qubits_p: int, n_qubits_n: int, Z: int, N: int, target_parity: int, msps: list) -> tuple[list[int], dict[int, int]]:
    """
    Generate basis states with fixed proton number Z and neutron number N, without parity or M projection.
    """
    basis = []
    for s in range(1 << (n_qubits_p + n_qubits_n)):
        if s.bit_count() == Z + N: # A check
            # then, Z check (lower n_qubits_p should have Z hot bits)
            Zcount = 0
            for i in range(n_qubits_n, n_qubits_p + n_qubits_n):
                bit = (s >> (n_qubits_p + n_qubits_n - 1 - i)) & 1
                if bit == 1:
                    Zcount += 1
            parity_ = 1
            for i in range(n_qubits_p + n_qubits_n):
                bit = (s >> (n_qubits_p + n_qubits_n - 1 - i)) & 1
                if bit == 0:
                    continue
                parity_ *= (-1)**(msps[-1-i].l)
            if parity_ == target_parity:
                basis.append(s)

    index = {s: i for i, s in enumerate(basis)}
    return basis, index


def fixed_N_P_M_basis(n_qubits_p: int, n_qubits_n: int, 
                      Z: int, N: int, 
                      msps: list, parity: int,
                      M_tot: int, verbose: bool = False) -> tuple[list[int], dict[int, int]]: 
    """
    Generate basis states with fixed proton number Z, neutron number N,
    parity, and total M. Note that M is assumed to be doubled.
    """
    basis = []
    for s in range(1 << (n_qubits_p + n_qubits_n)):
        if s.bit_count() == Z + N: # A check
            # then, Z check (lower n_qubits_p should have Z hot bits)
            Zcount = 0
            for i in range(n_qubits_n, n_qubits_p + n_qubits_n):
                bit = (s >> (n_qubits_p + n_qubits_n - 1 - i)) & 1
                if bit == 1:
                    Zcount += 1
            if verbose:
                print(f"s {format(s, f'0{n_qubits_p + n_qubits_n}b')} has Zcount {Zcount} (target {Z})") if verbose else None
            if Zcount != Z:
                continue

            Mz = 0
            parity_ = 1
            for i in range(n_qubits_p + n_qubits_n):
                bit = (s >> (n_qubits_p + n_qubits_n - 1 - i)) & 1
                if bit == 0:
                    continue
                m = msps[-1-i].jz
                Mz += m
                parity_ *= (-1)**(msps[-1-i].l)
            if Mz == M_tot and parity_ == parity:
                basis.append(s)

    index = {s: i for i, s in enumerate(basis)}
    return basis, index


def total_angular_momentum_squared_matrix(
    n_qubits_p: int,
    n_qubits_n: int,
    basis: list[int],
    index: dict[int, int],
    msps: list,
    M_tot: int | None = None,
) -> np.ndarray:
    """Build the J^2 matrix in the fixed-M M-scheme basis.

    The single-particle angular momenta in ``msps`` are assumed to use the
    doubled convention: ``j=2J`` and ``jz=2M``.  The returned matrix is in
    physical units, so an eigenstate with angular momentum J has eigenvalue
    J(J+1).
    """
    if n_qubits_p + n_qubits_n != len(msps):
        raise ValueError(
            "n_qubits_p + n_qubits_n must match the number of single-particle states."
        )

    dim = len(basis)
    J2 = np.zeros((dim, dim), dtype=complex)
    if dim == 0:
        return J2

    if M_tot is None:
        M_tot = _state_Mtot(basis[0], msps)
    for state in basis:
        state_Mtot = _state_Mtot(state, msps)
        if state_Mtot != M_tot:
            raise ValueError(
                f"All basis states must have fixed M_tot={M_tot}, but found {state_Mtot}."
            )
    M = M_tot / 2.0
    diagonal_M_part = M * (M + 1.0)

    J_plus_terms = _angular_momentum_ladder_terms(msps, 2)
    J_minus_terms = _angular_momentum_ladder_terms(msps, -2)

    for col, state in enumerate(basis):
        J2[col, col] += diagonal_M_part
        for intermediate_state, amp_plus in _apply_one_body_operator(
            state, J_plus_terms
        ).items():
            for new_state, amp_minus in _apply_one_body_operator(
                intermediate_state, J_minus_terms
            ).items():
                row = index.get(new_state)
                if row is not None:
                    J2[row, col] += amp_minus * amp_plus

    J2 = 0.5 * (J2 + J2.conj().T)
    return J2


def expectation_values(operator_matrix: np.ndarray, states: np.ndarray) -> np.ndarray:
    """Return <psi|O|psi> for one state vector or column-wise state vectors."""
    states = np.asarray(states)
    if states.ndim == 1:
        return np.asarray(np.vdot(states, operator_matrix @ states))
    values = np.einsum("ij,ij->j", states.conj(), operator_matrix @ states)
    return np.real_if_close(values)


def angular_momentum_from_J2(J2_values: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Convert J(J+1) values to the corresponding effective J."""
    J2_values = np.real_if_close(np.asarray(J2_values))
    J2_real = np.real(J2_values)
    J2_real = np.where(np.abs(J2_real) < tol, 0.0, J2_real)
    return 0.5 * (-1.0 + np.sqrt(1.0 + 4.0 * J2_real))


def apply_pauli_string_single(pauli: str, state: int, n_qubits: int) -> tuple[int, complex]:
    phase = 1.0 + 0.0j
    new_state = state

    for i, p in enumerate(pauli):
        bit = (state >> (n_qubits - 1 - i)) & 1
        if p == 'I':
            continue
        elif p == 'Z':
            if bit == 1:
                phase *= -1
        elif p == 'X':
            new_state ^= (1 << (n_qubits - 1 - i))
        elif p == 'Y':
            new_state ^= (1 << (n_qubits - 1 - i))
            phase *= (1j if bit == 0 else -1j)

    return new_state, phase

def apply_pauli_string_pn(pauli: str, state: int, n_qubits_p: int, n_qubits_n: int, debug=False):
    n_state = format(state >> n_qubits_p, f'0{n_qubits_n}b')
    p_state = format(state & ((1 << n_qubits_p) - 1), f'0{n_qubits_p}b')

    pauli_on_n = pauli[:n_qubits_n]
    pauli_on_p = pauli[n_qubits_n:]
    new_state_p, phase_p = apply_pauli_string_single(pauli_on_p, int(p_state, 2), n_qubits_p)
    new_state_n, phase_n = apply_pauli_string_single(pauli_on_n, int(n_state, 2), n_qubits_n)
    
    new_state = (new_state_n << n_qubits_p) | new_state_p
    phase = phase_p * phase_n

    if debug:
        ini_str = format(state, f'0{n_qubits_p + n_qubits_n}b')
        new_str = format(new_state, f'0{n_qubits_p + n_qubits_n}b')
        print(f" {pauli} : {ini_str} -> {new_str} phase: {phase} <= phase_p: {phase_p}, phase_n: {phase_n}")

    return new_state, phase


def selected_ci_sequential(basis_configs, Hmat, sampled_counts, 
                           max_iter=100, add_per_iter=15, 
                           threshold=None, initial_size=None):
    """
    Selected CI: sequentially grow a subspace by adding configurations with the largest coupling to the current subspace.

    Parameters
    ----------
    basis_configs : list[int]
        M-scheme basis (bitstring as int).
    Hmat : np.ndarray
        Full Hamiltonian matrix in basis_configs ordering.
    sampled_counts : dict[str,int] | list[str] | list[int]
        Sampled configurations. If dict, keys are bitstrings ("0101...")
        and counts are their weights. If list, it can be bitstrings or indices.
    max_iter : int
        Number of selection iterations.
    add_per_iter : int | None
        How many configs to add per iteration. None -> add all candidates.
    threshold : float | None
        Threshold on max |H_ij|. If set, add configs with coupling >= threshold.
    initial_size : int | None
        If sampled_counts is dict, use only top-N by count as initial pool.

    Returns
    -------
    selected_indices : list[int]
        Selected subspace indices in basis_configs order (sequentially grown).
    history : list[dict]
        Iteration log with added counts and selected size.
    """
    n = len(basis_configs)
    if isinstance(sampled_counts, dict):
        ordered = sorted(sampled_counts.items(), key=lambda x: x[1], reverse=True)
        if initial_size is not None:
            ordered = ordered[:initial_size]
        selected = [basis_configs.index(int(bitstr, 2)) for bitstr, _ in ordered]
    elif isinstance(sampled_counts, (list, tuple, np.ndarray)):
        if len(sampled_counts) == 0:
            selected = []
        else:
            first = sampled_counts[0]
            if isinstance(first, str):
                selected = [basis_configs.index(int(bitstr, 2)) for bitstr in sampled_counts]
            else:
                max_val = max(sampled_counts)
                if max_val < n:
                    selected = list(sampled_counts)
                else:
                    selected = [basis_configs.index(int(v)) for v in sampled_counts]
    else:
        raise TypeError("sampled_counts must be dict or list-like")

    # unique & keep order
    selected = list(dict.fromkeys(selected))
    selected_mask = np.zeros(n, dtype=bool)
    selected_mask[selected] = True
    all_indices = np.arange(n)
    history = []

    for it in range(max_iter):
        if selected_mask.all():
            break

        if np.any(selected_mask):
            H_sub = np.abs(Hmat[selected_mask][:, ~selected_mask])
            max_couplings = H_sub.max(axis=0)
            candidate_indices = all_indices[~selected_mask]
            if threshold is not None:
                pick_mask = max_couplings >= threshold
                picks = candidate_indices[pick_mask]
                if add_per_iter is not None and len(picks) > add_per_iter:
                    order = np.argsort(max_couplings[pick_mask])[::-1][:add_per_iter]
                    picks = picks[order]
            else:
                k = add_per_iter if add_per_iter is not None else len(candidate_indices)
                order = np.argsort(max_couplings)[::-1][:k]
                picks = candidate_indices[order]
        else:
            diag = np.real(np.diag(Hmat))
            k = add_per_iter if add_per_iter is not None else 1
            picks = np.argsort(diag)[:k]

        if len(picks) == 0:
            break
        for p in picks:
            selected_mask[p] = True
            selected.append(int(p))
        history.append({"iter": it, "added": int(len(picks)), "selected_size": int(selected_mask.sum())})

    return selected, history
