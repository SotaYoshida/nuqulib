"""Utility functions for NuQuLib.

This module provides various utility functions for quantum computing simulations
in nuclear physics, including format conversions between different quantum computing
frameworks and LaTeX formatting for nuclear notation.
"""

import re
import numpy as np
import pennylane as qml
from pytket.extensions.qiskit import qiskit_to_tk
from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit


def latex_nuc(nuc):
    """Convert nuclear notation to LaTeX format.
    
    Args:
        nuc (str): Nuclear notation string (e.g., "6He", "12C").
        
    Returns:
        str: LaTeX formatted string with mass number as superscript.
        
    Example:
        >>> latex_nuc("6He")
        '${}^{6}$He'
    """
    Anum = re.findall(r"\d+", nuc)[0]
    return "${}^{" + Anum + "}$" + nuc.replace(Anum, "")


def read_QiskitPauli(ops_qiskit, coeffs_qiskit):
    """Convert Qiskit Pauli operators to PennyLane format.
    
    Args:
        ops_qiskit (list): List of Qiskit Pauli operator strings.
        coeffs_qiskit (list): List of coefficients corresponding to operators.
        
    Returns:
        tuple: Tuple containing:
            - coeffs (list[float]): List of float coefficients.
            - obs (list): List of PennyLane operators.
    """
    coeffs = []
    obs = []
    for idx in range(len(ops_qiskit)):
        pauli_str = ops_qiskit[idx]
        coeff = coeffs_qiskit[idx]
        coeffs += [float(coeff)]
        obs += [get_operator_from_QiskitStr(pauli_str)]
    return coeffs, obs


def get_operator_from_QiskitStr(pauli_str):
    """Convert a Qiskit Pauli string to a PennyLane operator.
    
    Args:
        pauli_str (str): Pauli string in Qiskit format (e.g., "IXYZ").
        
    Returns:
        qml.operation.Operator: PennyLane operator corresponding to the Pauli string.
        
    Raises:
        ValueError: If the Pauli string contains more than 2 non-identity operators.
        
    Note:
        The function reverses the Pauli string to match PennyLane's qubit ordering.
        Currently supports up to 2 non-identity Pauli operators.
    """
    pauli_str = pauli_str[::-1]
    n_qubits = len(pauli_str)
    nonI_idxs = [i for i in range(n_qubits) if pauli_str[i] != "I"]
    if len(nonI_idxs) == 0:
        return qml.Identity(0)
    elif len(nonI_idxs) == 1:
        idx = nonI_idxs[0]
        if pauli_str[idx] == "X":
            return qml.PauliX(idx)
        elif pauli_str[idx] == "Y":
            return qml.PauliY(idx)
        elif pauli_str[idx] == "Z":
            return qml.PauliZ(idx)
    elif len(nonI_idxs) == 2:
        idx_1, idx_2 = nonI_idxs
        if pauli_str[idx_1] == "X":
            op1 = qml.PauliX(idx_1)
        elif pauli_str[idx_1] == "Y":
            op1 = qml.PauliY(idx_1)
        elif pauli_str[idx_1] == "Z":
            op1 = qml.PauliZ(idx_1)
        if pauli_str[idx_2] == "X":
            op2 = qml.PauliX(idx_2)
        elif pauli_str[idx_2] == "Y":
            op2 = qml.PauliY(idx_2)
        elif pauli_str[idx_2] == "Z":
            op2 = qml.PauliZ(idx_2)
        return op1 @ op2
    else:
        raise ValueError(f"Invalid Pauli string: {pauli_str}")


def transform_pytket_counts_to_qiskit(counts_pytket):
    """Convert PyTKET measurement counts to Qiskit format.
    
    Args:
        counts_pytket (dict): Dictionary of measurement outcomes from PyTKET.
        
    Returns:
        dict: Dictionary with bit strings reversed to match Qiskit ordering.
        
    Note:
        PyTKET and Qiskit use different qubit ordering conventions.
        This function reverses the bit string keys to convert between them.
    """
    counts_pytket = dict(counts_pytket)
    counts_qiskit = {}
    for key, value in counts_pytket.items():
        new_key = key[::-1]
        q_key = "".join(str(i) for i in new_key)
        counts_qiskit[q_key] = value
    return counts_qiskit


def transform_qiskitOps_to_pennylane(ops_qiskit: SparsePauliOp, coeffs_qiskit):
    """Transform Qiskit SparsePauliOp operators to PennyLane format.
    
    Args:
        ops_qiskit (SparsePauliOp): Qiskit sparse Pauli operator.
        coeffs_qiskit (list): List of coefficients for the operators.
        
    Returns:
        tuple: Tuple containing:
            - coeffs (list[float]): List of float coefficients.
            - obs (list): List of PennyLane operators.
    """
    coeffs = []
    obs = []
    for idx in range(len(ops_qiskit)):
        pauli_str = ops_qiskit[idx]
        coeff = coeffs_qiskit[idx]
        coeffs += [float(coeff)]
        obs += [get_operator_from_QiskitStr(pauli_str)]
    return coeffs, obs


def write_out_pytketCircuits_from_Qiskit(
    qc: list[QuantumCircuit], fname: str, circ_names: list[str] = None
):
    """Save a list of Qiskit circuits as PyTKET circuits to a numpy file.
    
    Args:
        qc (list[QuantumCircuit]): List of Qiskit quantum circuits.
        fname (str): Output filename (will be saved as .npy file).
        circ_names (list[str], optional): Names for the circuits. 
            Defaults to empty strings if not provided.
    
    Note:
        The circuits are saved as a dictionary with 'circuit' and 'name' keys
        using numpy's save function with pickle enabled.
    """
    Mydict = {"circuit": [], "name": []}
    if circ_names == None:
        circ_names = ["" for _ in range(len(qc))]
    for i in range(len(qc)):
        tk_circ = qiskit_to_tk(qc[i])
        Mydict["circuit"].append(tk_circ)
        Mydict["name"].append(circ_names[i])
    np.save(fname, Mydict, allow_pickle=True)


def write_out_pytketCircuit_from_Qiskit(
    qc: QuantumCircuit, fname: str, circ_name: str = ""
):
    """Save a single Qiskit circuit as PyTKET circuit to a numpy file.
    
    Args:
        qc (QuantumCircuit): Qiskit quantum circuit to convert and save.
        fname (str): Output filename (will be saved as .npy file).
        circ_name (str, optional): Name for the circuit. Defaults to empty string.
    
    Note:
        The circuit is saved as a dictionary with 'circuit' and 'name' keys
        using numpy's save function with pickle enabled.
    """
    tk_circ = qiskit_to_tk(qc)
    MyDict = {}
    MyDict["circuit"] = tk_circ
    MyDict["name"] = circ_name
    np.save(fname, MyDict, allow_pickle=True)
