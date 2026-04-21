"""Utility functions for NuQuLib.

This module provides various utility functions for NuQuLib,
including format conversions between different quantum computing
frameworks, LaTeX formatting for nuclear notation, etc.
"""

import itertools
import re
import numpy as np
import pennylane as qml
from pytket.extensions.qiskit import qiskit_to_tk
from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit


def latex_nuc(nuc: str) -> str:
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
    coeffs = coeffs_qiskit
    obs = []
    for idx in range(len(ops_qiskit)):
        pauli_str = ops_qiskit[idx]
        coeff = coeffs_qiskit[idx]
        obs += [get_operator_from_QiskitStr(pauli_str)]
    return coeffs, obs


def get_operator_from_QiskitStr(pauli_str:str) -> qml.operation.Operator:
    """Convert a Qiskit Pauli string to a PennyLane operator.
    
    Args:
        pauli_str (str): Pauli string in Qiskit format (e.g., "IXYZ").
        
    Returns:
        qml.operation.Operator: PennyLane operator corresponding to the Pauli string.
        
    Raises:
        ValueError: If the Pauli string contains invalid characters.
        
    Note:
        The function reverses the Pauli string to match PennyLane's qubit ordering.
    """
    pauli_str = pauli_str[::-1]
    n_qubits = len(pauli_str)
    nonI_idxs = [i for i in range(n_qubits) if pauli_str[i] != "I"]
    if len(nonI_idxs) == 0:
        op = qml.Identity(0)
    elif len(nonI_idxs) >= 1:
        idx = nonI_idxs[0]
        if pauli_str[idx] == "X":
            op = qml.PauliX(idx)
        elif pauli_str[idx] == "Y":
            op = qml.PauliY(idx)
        elif pauli_str[idx] == "Z":
            op = qml.PauliZ(idx)
        else:
            raise ValueError(f"Invalid Pauli string: {pauli_str}")
        
        for i in range(1, len(nonI_idxs)):
            idx = nonI_idxs[i]
            if pauli_str[idx] == "X":
                op = op @ qml.PauliX(idx)
            elif pauli_str[idx] == "Y":
                op = op @ qml.PauliY(idx)
            elif pauli_str[idx] == "Z":
                op = op @ qml.PauliZ(idx)
            else:
                raise ValueError(f"Invalid Pauli string: {pauli_str}")
    return op


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


def transform_qiskitOps_to_pennylane(Qiskit_Ops: SparsePauliOp):
    """Transform Qiskit SparsePauliOp operator(s) to PennyLane format.
    
    Args:
        Qiskit_Ops (SparsePauliOp): Qiskit sparse Pauli operator(s).

    Returns:
        tuple: Tuple containing:
            - coeffs (list[float]): List of float coefficients.
            - obs (list): List of PennyLane operators.
    """
    ops_qiskit = Qiskit_Ops.paulis
    coeffs = Qiskit_Ops.coeffs
    obs = []
    for idx in range(len(ops_qiskit)):
        pauli_str = ops_qiskit[idx].to_label()
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


class Orbit_nljjztz:
    """ Single particle state in the model space with n, l, j, jz, tz quanta. """
    def __init__(self, n:int, l:int, j:int, jz:int, tz:int):
        self.n = n
        self.l = l
        self.j = j
        self.jz = jz
        self.tz = tz
        self.e = 2 * n + l


class Orbit_nljtz:
    """ Single particle state in the model space with n, l, j, tz quanta. """
    def __init__(self, n:int, l:int, j:int, tz:int):
        self.n = n
        self.l = l
        self.j = j
        self.tz = tz
        self.e = 2 * n + l

class Orbit_nlj:
    """ Single particle state in the model space with n, l, j quanta. """
    def __init__(self, n:int, l:int, j:int):
        self.n = n
        self.l = l
        self.j = j
        self.e = 2 * n + l


def get_spsidx_from_nljtz(single_particle_states: list, n, l, j, tz):
    for idx, sps in enumerate(single_particle_states):
        n_ = sps.n
        l_ = sps.l
        j_ = sps.j
        tz_ = sps.tz
        if n == n_ and l == l_ and j == j_ and tz == tz_:
            return idx
    raise ValueError(f"Single particle state with n={n}, l={l}, j={j}, tz={tz} not found in the model space.")


def count_msps(emax: int, vemin: int=0, vemax:int = 100):
    """ Count the number of single particle states in the model space up to emax. """
    count = 0
    msps_p = []
    msps_n = []
    for te in range(emax+1):
        if te < vemin or te > vemax:
            continue
        for l in range(te + 1):
            n = (te - l) // 2
            if n < 0 or 2*n + l != te:
                continue
            j_vals = [l - 0.5, l + 0.5] if l > 0 else [0.5]
            for j in j_vals:
                j2 = int(2*j)
                count += j2 + 1
                for jz in np.arange(-j2, j2+1, 2):
                    msps_p.append(Orbit_nljjztz(n, l, j2, jz, -1))
                    msps_n.append(Orbit_nljjztz(n, l, j2, jz, +1))
    return msps_p, msps_n


def t_count(eps_tol=1.e-9, model="ross-selinger", precision=None):
    if eps_tol <= 0:
        raise ValueError(f"precision must be positive, got {eps_tol}")
    if model == "bocharov":
        return int(round(1.149 * np.log2(1.0 / eps_tol) + 9.2))
    if model == "ross-selinger":
        return int(np.ceil(3.0 * np.log2(1.0 / eps_tol)))
    raise ValueError(f"Unknown rotation model: {model}")
