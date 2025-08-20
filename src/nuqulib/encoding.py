import numpy as np
from qiskit_nature.second_q.mappers import JordanWignerMapper, BravyiKitaevMapper
from qiskit_nature.second_q.operators import FermionicOp
from qiskit.quantum_info import SparsePauliOp
from pytket.pauli import Pauli, QubitPauliString
from pytket.utils.operators import QubitPauliOperator
from collections import defaultdict
from pytket import Qubit


def mapping_to_Pauli_string(
    Fermionic_op: FermionicOp,
    n_qubits: int, 
    method: str):
    if method == "JordanWigner" or method == "JW" or method == "Jordan-Wigner":
        mapper = JordanWignerMapper()
    elif method == "BravyiKitaev" or method == "BK" or method == "Bravyi-Kitaev":
        mapper = BravyiKitaevMapper()
    else:
        raise ValueError("Invalid method for mapping: " + method)
    qubit_op = mapper.map(Fermionic_op, register_length=n_qubits)
    return qubit_op


def mapping_of_pn_hamiltonians(op_pn, n_qubits_p, n_qubits_n, method):
    op_list = []
    for p_str, n_str in op_pn.keys():
        coeff_overall = op_pn[(p_str, n_str)]
        op_p = mapping_to_Pauli_string(
            FermionicOp({p_str: 1.0}, num_spin_orbitals=n_qubits_p),
            n_qubits_p,
            method=method,
        )
        op_n = mapping_to_Pauli_string(
            FermionicOp({n_str: 1.0}, num_spin_orbitals=n_qubits_n),
            n_qubits_n,
            method=method,
        )
        for idx_p, pauli_p in enumerate(op_p.paulis):
            coeff_p = op_p.coeffs[idx_p]
            for idx_n, pauli_n in enumerate(op_n.paulis):
                coeff_n = op_n.coeffs[idx_n]
                coeff_pn = coeff_p * coeff_n * coeff_overall
                pauli = str(pauli_n) + str(pauli_p)
                op_list.append((pauli, coeff_pn))
    return SparsePauliOp.from_list(op_list)


def check_XXYYterm(hamiltonian_op_XXYY):
    """
    In ordinary Hamiltonians, the XX and YY terms should have the same coefficient.
    This function checks whether the XX and YY terms have the same coefficient.
    """
    ops = hamiltonian_op_XXYY.paulis
    coeffs = hamiltonian_op_XXYY.coeffs
    for idx_op, op in enumerate(ops):
        idx_XX = [idx for idx, p in enumerate(str(op)) if p == "X"]
        if idx_XX == []:
            continue
        coeff_XX = coeffs[idx_op]
        for idx_op2 in range(len(ops)):
            op2 = ops[idx_op2]
            idx_YY = [idx for idx, p in enumerate(str(op2)) if p == "Y"]
            if idx_XX == idx_YY != []:
                coeff_YY = coeffs[idx_op2]
                assert abs(coeff_XX - coeff_YY) < 1e-10, (
                    "The XX and YY terms should have the same coefficient."
                )
                break
    return True


def separate_Hamil_terms(hamiltonian_op: SparsePauliOp):
    """
    Separate the Hamiltonian into the diagonal terms and the XX+YY terms.
    Args:
        hamiltonian_op (SparsePauliOp): Hamiltonian operator
    Returns:
        hamiltonian_op_diag (SparsePauliOp): Hamiltonian operator with diagonal terms
        hamiltonian_op_XXYY (SparsePauliOp): Hamiltonian operator with XX+YY terms

    This can be useful for post-processing the results of the quantum simulation.
    """
    ops_all = hamiltonian_op.paulis
    coeffs_all = hamiltonian_op.coeffs
    ops_diag = []
    coeffs_diag = []
    ops_XXYY = []
    coeffs_XXYY = []
    for idx in range(len(ops_all)):
        op = str(ops_all[idx])
        assert abs(np.imag(coeffs_all[idx])) < 1e-10
        coeff = float(np.real(coeffs_all[idx]))
        if ("X" in op) or ("Y" in op):
            ops_XXYY.append(op)
            coeffs_XXYY.append(coeff)
        else:
            ops_diag.append(op)
            coeffs_diag.append(coeff)

    hamiltonian_op_diag = SparsePauliOp.from_list(list(zip(ops_diag, coeffs_diag)))
    hamiltonian_op_XXYY = SparsePauliOp.from_list(list(zip(ops_XXYY, coeffs_XXYY)))
    check_XXYYterm(hamiltonian_op_XXYY)
    return hamiltonian_op_diag, hamiltonian_op_XXYY


def qps_from_sparsepauliop(paulis):
    """Convert SparsePauliOp tensor of Paulis to pytket QubitPauliString."""
    pauli_sym = {"I": Pauli.I, "X": Pauli.X, "Y": Pauli.Y, "Z": Pauli.Z}
    qlist = []
    plist = []
    for q, p in enumerate(paulis):
        if p != "I":
            qlist.append(Qubit(q))
            plist.append(pauli_sym[p])
    return QubitPauliString(qlist, plist)


def qpo_from_sparsepauliop(sp_op: SparsePauliOp) -> QubitPauliOperator:
    """
    Convert SparsePauliOp QubitOperator to pytket QubitPauliOperator.
    This is code is originally from pytket documentation [here](https://docs.quantinuum.com/systems/trainings/knowledge_articles/Quantinuum_high_energy_physics_experiment.html),
    but we need to modify to reverse the order of the qubits in the QubitPauliOperator.
    """
    tk_op = defaultdict(complex)
    for term, coeff in sp_op.to_list():
        term = term[::-1]
        string = qps_from_sparsepauliop(term)
        tk_op[string] += coeff
    return QubitPauliOperator(tk_op)
