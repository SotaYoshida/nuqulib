"""Functions to diagonalize nuclear Hamiltonians


"""
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from .nuclear_hamiltonian import Hamiltonian

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
        Zc: int = 0, Nc: int = 0):
    assert abs(target_parity) == 1, "parity should be ±1"
    n_qubit_p = hamil.n_qubits_p
    n_qubit_n = hamil.n_qubits_n

    Mtot = (Z+N-Zc-Nc) % 2 
    basis, index = fixed_N_P_M_basis(n_qubit_p, n_qubit_n, Z-Zc, N-Nc,
                                     hamil.msps, target_parity, Mtot)
    Hsub = Mprojected_hamiltonian(Hamil_mapped, n_qubit_p, n_qubit_n, basis, index)
    if len(basis) > 4000:
        print("Warning: Large dimension in diagonalization:", len(basis))
        print("One may consider to use Krylov subspace methods.")
    print("Diagonalizing the Hamiltonian...")
    evals, evecs = np.linalg.eigh(Hsub)
    print(f"dim. (N・M・P-projected; M={Mtot}):", len(basis))
    num_evals = len(evals)
    num_show = min(10, num_evals)
    print(np.sort(evals.real)[:num_show])
    return basis, Hsub, evals


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


def fixed_N_P_M_basis(n_qubits_p, n_qubits_n, 
                    Z, N, msps, parity, M_tot): # M is doubled
    basis = []
    for s in range(1 << (n_qubits_p + n_qubits_n)):
        if s.bit_count() == Z + N: # A check
            # then, Z check (lower n_qubits_p should have Z hot bits)
            Zcount = 0
            for i in range(n_qubits_n, n_qubits_p + n_qubits_n):
                bit = (s >> (n_qubits_p + n_qubits_n - 1 - i)) & 1
                if bit == 1:
                    Zcount += 1
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


def apply_pauli_string_single(pauli, state, n_qubits):
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


def apply_pauli_string_pn(pauli, state, n_qubits_p, n_qubits_n, debug=False):
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


def Mprojected_hamiltonian(hamiltonian: SparsePauliOp,
                           n_qubits_p: int, n_qubits_n: int, 
                           basis: list[int],
                           index: dict[int, int]) -> np.ndarray:
    """
    Build the M-projected Hamiltonian for a Hamiltonian whose Pauli strings may
    act on protons, neutrons, or both. Treat proton and neutron parts independently
    by padding shorter Pauli labels appropriately.
    """
    total = n_qubits_p + n_qubits_n
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

def fixed_N_P_M_basis(n_qubits_p: int, n_qubits_n: int, 
                      Z: int, N: int, 
                      msps: list, parity: int,
                      M_tot: int) -> tuple[list[int], dict[int, int]]: 
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
    Selected CI: これまでsampleした配位に加えて、遷移行列要素(|H_ij|)が大きい配位を逐次的に追加する。

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