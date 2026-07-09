"""Quantum encoding utilities for nuclear Hamiltonians.

This module provides functions for mapping nuclear Hamiltonians to qubit operators
using e.g., Jordan-Wigner,  and utilities for 
working with Pauli operators in different quantum computing frameworks.
"""

from collections import defaultdict
import numpy as np
import os
import multiprocessing
from multiprocessing import get_context
from pytket.pauli import Pauli, QubitPauliString
from pytket.utils.operators import QubitPauliOperator
from pytket import Qubit
from qiskit_nature.second_q.mappers import JordanWignerMapper, BravyiKitaevMapper
from qiskit_nature.second_q.operators import FermionicOp
from qiskit.quantum_info import SparsePauliOp
import time
from tqdm import tqdm
from .hatt_mapper import HATTMapper

def mapping_to_Pauli_string(
    Fermionic_op: FermionicOp,
    n_qubits: int, 
    init_qubit: int=0,
    method: str="JordanWigner",
    Hamildict_specified: dict={},
    filepath: str|os.PathLike="./",
    verbose: bool=False):
    """Map fermionic operators to Pauli strings using specified encoding.
    
    Args:
        Fermionic_op (FermionicOp): Fermionic operator to be mapped.
        n_qubits (int): Number of qubits for the mapping.
        init_qubit (int): Starting index for qubits in the mapping.
        method (str): Encoding method. Options: "JordanWigner"/"JW"/"Jordan-Wigner"
                     or "BravyiKitaev"/"BK"/"Bravyi-Kitaev".
        Hamildict_specified (dict): Dictionary of Hamiltonian terms in fermionic form.
                     used in the special case, HATTMapper.
        filepath (str|os.PathLike): File path for saving/loading mapper in the case of HATTMapper.
        verbose (bool): If True, print intermediate mapping results for debugging.
    
    Returns:
        SparsePauliOp: Mapped Pauli operator.
        
    Raises:
        ValueError: If invalid encoding method is provided.

    Note:
        BravyiKitaev mapping has never been tested in this module.
    """
    if method == "JordanWigner" or method == "JW" or method == "Jordan-Wigner":
        mapper = JordanWignerMapper()

    elif method == "BravyiKitaev" or method == "BK" or method == "Bravyi-Kitaev":
        mapper = BravyiKitaevMapper()

    elif method == "HATTMapper":
        if os.path.exists(filepath):
            mapper = HATTMapper.load(filepath)
        else:
            mapper = HATTMapper(FermionicOp(Hamildict_specified))
            mapper.save(filepath)
    else:
        raise ValueError("Invalid method for mapping: " + method)
    
    op_list=[]
    qubit_op = mapper.map(Fermionic_op)
    if verbose:
        print(f"qubit_op(before) {qubit_op} Hamildict_specified {Hamildict_specified}")
    for op in qubit_op:
        nqb=len(op.paulis[0].to_label())
        nleft=n_qubits-nqb-init_qubit
        target_pauli = op.paulis[0].to_label()[::-1] # Qiskit ordering
        pauli_str = init_qubit*'I'+target_pauli+nleft*'I'
        pauli_str = pauli_str[::-1] 
        op_list.append(SparsePauliOp(pauli_str,op.coeffs[0]))
    qubit_op = sum(op_list).simplify()
    if verbose:
        print(f"qubit_op(after) {qubit_op}")
    return qubit_op


def task_pn_mapping(op_pn_key, op_pn, n_qubits_p, n_qubits_n, method,
                    Hamildict_specified_p, Hamildict_specified_n,
                    filepath_p, filepath_n):
    """Map one proton-neutron fermionic term to Pauli-label terms.

    Args:
        op_pn_key (tuple[str, str]): ``(proton_op, neutron_op)`` fermionic
            operator strings.
        op_pn (dict): Dictionary containing the coefficient for
            ``op_pn_key``.
        n_qubits_p (int): Number of proton qubits.
        n_qubits_n (int): Number of neutron qubits.
        method (str): Fermion-to-qubit mapping method.
        Hamildict_specified_p (dict): Proton-sector Hamiltonian terms used by
            mappings such as ``HATTMapper``.
        Hamildict_specified_n (dict): Neutron-sector Hamiltonian terms used by
            mappings such as ``HATTMapper``.
        filepath_p (str | os.PathLike): Proton-sector mapper cache path.
        filepath_n (str | os.PathLike): Neutron-sector mapper cache path.

    Returns:
        list[list]: Pauli labels and coefficients generated for this term.
    """
    worker_list = [ ]
    p_str, n_str = op_pn_key
    coeff_overall = op_pn[op_pn_key]
    show_verbose = False

    if method == "HATTMapper":
        n_cre, n_ani = n_str.split(" ")
        n_cre = "+_" + str( int(n_cre.split("_")[1]) )
        n_ani = "-_" + str( int(n_ani.split("_")[1]) )
        op_p = mapping_to_Pauli_string(
            FermionicOp({p_str : 1.0}, num_spin_orbitals=n_qubits_p), \
            n_qubits_p, 0, method, Hamildict_specified_p, filepath_p)
        op_n = mapping_to_Pauli_string(
            FermionicOp({n_str : 1.0}, num_spin_orbitals=n_qubits_n), \
            n_qubits_n, 0, method, Hamildict_specified_n, filepath_n)
    else:
        op_p = mapping_to_Pauli_string(FermionicOp({p_str : 1.0}, num_spin_orbitals=n_qubits_p), \
                                    n_qubits_p, 0, method)
        op_n = mapping_to_Pauli_string(FermionicOp({n_str : 1.0}, num_spin_orbitals=n_qubits_n), \
                                    n_qubits_n, 0, method)

    for idx_p, pauli_p in enumerate(op_p.paulis):
        pauli_p = str(pauli_p)
        coeff_p = op_p.coeffs[idx_p]
        for idx_n, pauli_n in enumerate(op_n.paulis):
            coeff_n = op_n.coeffs[idx_n]
            pauli_n = str(pauli_n)[:n_qubits_n]
            coeff_pn = coeff_p * coeff_n * coeff_overall
            pauli = str(pauli_n) + str(pauli_p)
            worker_list.append([pauli, coeff_pn])

    return worker_list


def mapping_of_pn_hamiltonians(op_pn: dict[tuple[str, str], float],
                               n_qubits_p: int, n_qubits_n: int,
                               method: str,
                               Hamildict_specified_p: dict,
                               Hamildict_specified_n: dict,
                               filepath_p: str|os.PathLike,
                               filepath_n: str|os.PathLike):
    """Map proton-neutron coupled Hamiltonians to Pauli operators.
    
    This function handles the mapping of nuclear Hamiltonians that include
    both proton and neutron sectors with their respective interactions.
    We here assume that proton indices are lower than neutron indices,
    and those are to be coupled like neutron part followed by proton part
    so that one can use them in Qiskit.

    Args:
        op_pn (dict): Dictionary with (proton_str, neutron_str) keys and coefficient values.
        n_qubits_p (int): Number of qubits for proton sector.
        n_qubits_n (int): Number of qubits for neutron sector.
        method (str): Encoding method (e.g., "Jordan-Wigner", "Bravyi-Kitaev").
        Hamildict_specified_p (dict): Dictionary of Hamiltonian terms in fermionic form for protons.
        Hamildict_specified_n (dict): Dictionary of Hamiltonian terms in fermionic form for neutrons.
        filepath_p (str|os.PathLike): File path for saving/loading proton mapper.
        filepath_n (str|os.PathLike): File path for saving/loading neutron mapper.

    Returns:
        SparsePauliOp: Combined Pauli operator representing the full Hamiltonian.
    """
    print(f"Mapping p-n Hamiltonian terms to Pauli strings using {method}...")
    if method != "HATTMapper":
        Hamildict_specified_p = {}
        Hamildict_specified_n = {}
        filepath_p = "./"
        filepath_n = "./"

    nproc = max(multiprocessing.cpu_count() - 2, 1)

    with get_context("fork").Pool(processes=nproc) as pool:
        results = list(tqdm(pool.starmap(
            task_pn_mapping, [(tkey, op_pn, n_qubits_p, n_qubits_n, method,
                               Hamildict_specified_p, Hamildict_specified_n,
                               filepath_p, filepath_n) for tkey in op_pn.keys()]
        )))

    paulis = [ ]
    coeffs = [ ]
    for res in results:
        for pauli, coeff in res: 
            paulis.append(pauli)
            coeffs.append(coeff)
    if len(paulis) == 0:
        return SparsePauliOp.from_list([("I"* (n_qubits_p + n_qubits_n), 0.0)])
    else:
        return SparsePauliOp.from_list(list(zip(paulis, coeffs)))


def check_XXYYterm(hamiltonian_op_XXYY):
    """Check that XX and YY terms have identical coefficients.
    
    In pairing or pair-wise Hamiltonians, the XX and YY terms should have the same coefficient.
    This function validates this constraint for debugging and verification.
    
    Args:
        hamiltonian_op_XXYY (SparsePauliOp): Hamiltonian containing only XX and YY terms.
        
    Returns:
        bool: True if all XX/YY coefficient pairs match.
        
    Raises:
        AssertionError: If XX and YY terms have different coefficients.
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
    """Separate Hamiltonian, pairing or pair-wise ones, into diagonal and XX+YY terms.
    
    This function partitions a Hamiltonian operator into terms that are diagonal
    in the computational basis (I and Z terms) and off-diagonal XX+YY terms.
    This separation is useful for quantum algorithms that treat different
    types of terms separately.
    
    Args:
        hamiltonian_op (SparsePauliOp): Full Hamiltonian operator.
        
    Returns:
        tuple: Tuple containing:
            - hamiltonian_op_diag (SparsePauliOp): Diagonal terms (I, Z).
            - hamiltonian_op_XXYY (SparsePauliOp): XX and YY terms.
            
    Note:
        This function validates that XX and YY terms have matching coefficients
        and assumes real coefficients.
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
    """Convert SparsePauliOp Pauli strings to PyTKET QubitPauliString.
    
    Args:
        paulis (str): Pauli string (e.g., "IXYZ").
        
    Returns:
        QubitPauliString: PyTKET QubitPauliString representation.
        
    Note:
        Identity operators are automatically filtered out as they don't
        contribute to the QubitPauliString representation.
    """
    pauli_sym = {"I": Pauli.I, "X": Pauli.X, "Y": Pauli.Y, "Z": Pauli.Z}
    qlist = []
    plist = []
    for q, p in enumerate(paulis):
        if p != "I":
            qlist.append(Qubit(q))
            plist.append(pauli_sym[p])
    return QubitPauliString(qlist, plist)


def qpo_from_sparsepauliop(sp_op: SparsePauliOp) -> QubitPauliOperator:
    """Convert Qiskit SparsePauliOp to PyTKET QubitPauliOperator.
    
    This function converts Qiskit's SparsePauliOp representation to PyTKET's
    QubitPauliOperator format, with automatic reversal of qubit ordering
    to match PyTKET conventions.
    
    Args:
        sp_op (SparsePauliOp): Qiskit SparsePauliOp to convert.
        
    Returns:
        QubitPauliOperator: PyTKET QubitPauliOperator with reversed qubit order.
        
    Note:
        This code is based on PyTKET documentation but modified to handle
        qubit ordering differences between Qiskit and PyTKET.
        `Original reference <https://docs.quantinuum.com/systems/trainings/knowledge_articles/Quantinuum_high_energy_physics_experiment.html>`_
    """
    tk_op = defaultdict(complex)
    for term, coeff in sp_op.to_list():
        term = term[::-1]
        string = qps_from_sparsepauliop(term)
        tk_op[string] += coeff
    return QubitPauliOperator(tk_op)
