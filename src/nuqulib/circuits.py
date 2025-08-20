"""Quantum circuit implementations for nuclear simulations.

This module provides various quantum circuit implementations including Givens rotations,
controlled gates, and utility functions for Pauli string measurements.
"""

from qiskit import QuantumCircuit
import numpy as np


def cG1(circ: QuantumCircuit, c_qubit: int, i: int, j: int, theta: float):
    """Apply controlled Givens rotation to a quantum circuit.
    
    Implements a controlled version of the Givens rotation gate using
    elementary quantum gates. This is used for e.g. state preparation.
    
    Args:
        circ (QuantumCircuit): Quantum circuit to modify.
        c_qubit (int): Control qubit index.
        i (int): First target qubit index.
        j (int): Second target qubit index.
        theta (float): Rotation angle parameter.
    """
    theta_4 = theta / 4
    circ.cx(i, j)
    circ.ry(theta_4, i)
    circ.cx(j, i)
    circ.ry(-theta_4, i)
    circ.cx(c_qubit, i)
    circ.ry(theta_4, i)
    circ.cx(j, i)
    circ.ry(-theta_4, i)
    circ.cx(c_qubit, i)
    circ.cx(i, j)


def cG1_gate(theta: float, method: str = "magic"):
    """Create a controlled Givens rotation gate.
    
    Args:
        theta (float): Rotation angle parameter.
        method (str, optional): Implementation method. Options: "magic", "Xanadu", 
                               "iSWAP_Rz". Defaults to "magic".
    
    Returns:
        Gate: Controlled Givens rotation gate.
        
    Raises:
        ValueError: If invalid method is provided.
    """
    circ = QuantumCircuit(2)
    if method == "magic":
        Givens_magic(circ, 0, 1, theta)
        circ.name = "cG1(magic)"
    elif method == "Xanadu":
        Givens_Xanadu(circ, 0, 1, theta)
        circ.name = "cG1(Xanadu)"
    elif method == "iSWAP_Rz":
        Givens_iSWAP_Rz(circ, 0, 1, theta)
        circ.name = "cG1(iSWAP_Rz)"
    else:
        raise ValueError(f"Invalid method: {method}")
    circ = circ.to_gate()
    circ = circ.control(1)
    return circ

def apply_sqrt_iSWAP(qc: QuantumCircuit, i: int, j: int):
    """Apply square root of iSWAP gate to quantum circuit.
    
    Args:
        qc (QuantumCircuit): Quantum circuit to modify.
        i (int): First qubit index.
        j (int): Second qubit index.
    """
    sqrt_iSWAP_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 1 / np.sqrt(2), 1j / np.sqrt(2), 0],
            [0, 1j / np.sqrt(2), 1 / np.sqrt(2), 0],
            [0, 0, 0, 1],
        ]
    )
    qc.unitary(sqrt_iSWAP_matrix, [i, j], label="sqrt_iSWAP")


def Givens_iSWAP_Rz(circ: QuantumCircuit, i: int, j: int, theta: float):
    """Givens rotation implementation using iSWAP and Rz gates.
    
    This implementation is based on Google's approach from 
    Science 369, 1084 (2020) - DOI: 10.1126/science.abb9811.
    
    Args:
        circ (QuantumCircuit): Quantum circuit to modify.
        i (int): First qubit index.
        j (int): Second qubit index.
        theta (float): Rotation angle parameter.
    """
    theta = theta / 2
    apply_sqrt_iSWAP(circ, i, j)
    circ.rz(-theta, i)
    circ.rz(theta + np.pi, j)
    apply_sqrt_iSWAP(circ, i, j)
    circ.rz(np.pi, j)


def Givens_magic(circ: QuantumCircuit, i: int, j: int, theta: float):
    """Optimal Givens rotation using magic basis decomposition.
    
    This implementation is based on optimal quantum circuits for general 
    two-qubit gates from Phys. Rev. A 69, 032315 (2004).
    It minimizes the number of CNOT gates required.
    
    Args:
        circ (QuantumCircuit): Quantum circuit to modify.
        i (int): First qubit index.
        j (int): Second qubit index.
        theta (float): Rotation angle parameter.
        
    Reference:
        Used in npj Quantum Information (2023) 9:60
    """
    theta = theta / 2
    circ.s(i)
    circ.s(j)
    circ.h(j)
    circ.cx(j, i)
    circ.ry(theta, i)
    circ.ry(theta, j)
    circ.cx(j, i)
    circ.h(j)
    circ.sdg(i)
    circ.sdg(j)


def Givens_Xanadu(circ: QuantumCircuit, i: int, j: int, theta: float):
    """Givens rotation implementation from Xanadu.
    
    Implementation based on the approach described in 
    Quantum 6, 742 (2022) - doi:10.22331/q-2022-06-20-742.
    
    Args:
        circ (QuantumCircuit): Quantum circuit to modify.
        i (int): First qubit index.
        j (int): Second qubit index.
        theta (float): Rotation angle parameter.
    """
    theta_2 = theta / 2
    circ.cx(i, j)
    circ.ry(theta_2, i)
    circ.cx(j, i)
    circ.ry(-theta_2, i)
    circ.cx(j, i)
    circ.cx(i, j)


def G_gate(theta: float, method: str = "magic"):
    """Create a Givens rotation gate using specified method.
    
    Args:
        theta (float): Rotation angle parameter.
        method (str, optional): Implementation method. Options: "magic", "Xanadu", 
                               "iSWAP_Rz". Defaults to "magic".
    
    Returns:
        Gate: Givens rotation gate.
        
    Raises:
        ValueError: If invalid method is provided.
    """
    circ = QuantumCircuit(2)
    if method == "magic":
        Givens_magic(circ, 0, 1, theta)
        circ.name = "G(magic)"
    elif method == "Xanadu":
        Givens_Xanadu(circ, 0, 1, theta)
        circ.name = "G(Xanadu)"
    elif method == "iSWAP_Rz":
        Givens_iSWAP_Rz(circ, 0, 1, theta)
        circ.name = "G(iSWAP_Rz)"
    else:
        raise ValueError("Invalid method for G_gate")
    return circ.to_gate()


def Givens_2_Xanadu(circ: QuantumCircuit, i: int, j: int, k: int, l: int, theta: float):
    """Two-qubit pair Givens rotation using Xanadu method.
    
    Implements a four-qubit Givens rotation that acts on two qubit pairs
    simultaneously. This is useful for efficient fermionic simulations.
    
    Args:
        circ (QuantumCircuit): Quantum circuit to modify.
        i (int): First qubit of first pair.
        j (int): Second qubit of first pair.
        k (int): First qubit of second pair.
        l (int): Second qubit of second pair.
        theta (float): Rotation angle parameter.
    """
    theta = theta / 8
    circ.cx(k, l)
    circ.cx(i, k)
    circ.h(i)
    circ.h(l)
    circ.cx(i, j)
    circ.cx(k, l)
    circ.ry(-theta, i)
    circ.ry(theta, j)
    circ.cx(i, l)
    circ.h(l)
    circ.cx(l, j)
    circ.ry(-theta, i)
    circ.ry(theta, j)
    circ.cx(k, j)
    circ.cx(k, i)
    circ.ry(theta, i)
    circ.ry(-theta, j)
    circ.cx(l, j)
    circ.h(l)
    circ.cx(i, l)
    circ.ry(theta, i)
    circ.ry(-theta, j)
    circ.h(l)
    circ.cx(i, j)
    circ.cx(k, i)
    circ.h(i)
    circ.cx(i, k)
    circ.cx(k, l)


def G2_gate(theta: float, method: str = "Xanadu"):
    """Create a two-pair Givens rotation gate.
    
    Args:
        theta (float): Rotation angle parameter.
        method (str, optional): Implementation method. Currently only "Xanadu" 
                               is supported. Defaults to "Xanadu".
    
    Returns:
        Gate: Two-pair Givens rotation gate acting on 4 qubits.
        
    Raises:
        ValueError: If invalid method is provided.
    """
    circ = QuantumCircuit(4)
    if method == "Xanadu":
        Givens_2_Xanadu(circ, 0, 1, 2, 3, theta)
        circ.name = "G2(Xanadu)"
    else:
        raise ValueError(f"Invalid method for G2_gate: {method}")
    return circ.to_gate()


def get_idx_ancilla_in_string(n_qubit: int, ancilla: int | None, Qiskit_ordering: bool) -> int:
    """Get ancilla qubit index in measurement string.
    
    Args:
        n_qubit (int): Total number of qubits.
        ancilla (int or None): Ancilla qubit index.
        Qiskit_ordering (bool): Whether to use Qiskit qubit ordering.
    
    Returns:
        int or None: Index of ancilla qubit in the measurement string,
                    or None if no ancilla qubit.
    """
    idx_ancilla = None
    if ancilla != None:
        if Qiskit_ordering:
            idx_ancilla = n_qubit - 1 - ancilla
        else:
            idx_ancilla = ancilla
    return idx_ancilla


def additional_qc(qc_in, pauli_str, register_target, Qiskit_order=True):
    """Add basis rotation gates for Pauli string measurement.
    
    Applies the necessary basis rotation gates (H for X measurement, 
    SdgH for Y measurement) to measure a Pauli string.
    
    Args:
        qc_in (QuantumCircuit): Quantum circuit to modify.
        pauli_str (str): Pauli string to measure.
        register_target (list): List of target qubit indices.
        Qiskit_order (bool, optional): Whether to use Qiskit qubit ordering.
                                      Defaults to True.
    
    Raises:
        ValueError: If invalid Pauli string character is encountered.
    """
    pauli_str = str(pauli_str)
    if Qiskit_order:
        pauli_str = pauli_str[::-1]

    for i in range(len(pauli_str)):
        if pauli_str[i] == "X":
            qc_in.h(register_target[i])
        elif pauli_str[i] == "Y":
            qc_in.sdg(register_target[i])
            qc_in.h(register_target[i])
        elif pauli_str[i] == "Z" or pauli_str[i] == "I":
            pass
        else:
            raise ValueError("Invalid Pauli string: ", pauli_str)


def get_idx_to_measure(pauli_str: str, Qiskit_order: bool = True) -> list[int]:
    """Get qubit indices that need to be measured for a Pauli string.
    
    Returns indices of qubits that have non-identity Pauli operators
    and therefore need to be measured.
    
    Args:
        pauli_str (str): Pauli string.
        Qiskit_order (bool, optional): Whether to use Qiskit qubit ordering.
                                      Defaults to True.
    
    Returns:
        list: List of qubit indices to measure.
    """
    idxs = [idx for idx, p in enumerate(pauli_str) if p != "I"]
    if Qiskit_order:
        idxs = [len(pauli_str) - 1 - idx for idx in idxs]
    return idxs


def expec_Zstring(
    res: dict, idx_relevant: list[int], Qiskit_ordering: bool = True, target_qubits: list[int] = [], ancilla_qubit: int | None = None
):
    """Calculate expectation value of Z-string measurement from results.
    
    Computes the expectation value of a Z-string Pauli operator from measurement results,
    with optional ancilla qubit for some algorithms such as QKrylov.

    Args:
        res (dict): Dictionary of measurement results {bitstring: count/weights}.
                    values can be either raw counts (int) or normalized weights (float)
        idx_relevant (list): List of relevant qubit indices for Z measurements.
        Qiskit_ordering (bool, optional): Whether to use Qiskit ordering. 
                                         Defaults to True.
        target_qubits (list, optional): List of target qubit indices. 
                                       Defaults to [].
        ancilla_qubit (int, optional): Ancilla qubit index for post-selection. 
                                      Defaults to None.
    
    Returns:
        tuple: Tuple containing:
            - exp_val (float): Overall expectation value.
            - exp_val_p0 (float): Expectation value when ancilla=0.
            - exp_val_p1 (float): Expectation value when ancilla=1.
    """
    exp_val = exp_val_p0 = exp_val_p1 = 0.0
    n_shot = sum(res.values())
    n_qubit = len(list(res.keys())[0])
    idx_ancilla = get_idx_ancilla_in_string(n_qubit, ancilla_qubit, Qiskit_ordering)
    for bitstr, count in res.items():
        if ancilla_qubit != None and target_qubits != []:
            bitstr_target = "".join(
                [bitstr[k] for k in range(n_qubit) if k != idx_ancilla]
            )
        else:
            bitstr_target = bitstr
        tmp = 1.0
        for idx in idx_relevant:
            if Qiskit_ordering:
                idx = -1 - idx
            bit = int(bitstr_target[idx])
            tmp *= 1 - 2 * bit
        exp_val += tmp * count

        if ancilla_qubit != None:
            if int(bitstr[idx_ancilla]) == 0:
                exp_val_p0 += tmp * count
            else:
                exp_val_p1 += tmp * count
    exp_val /= n_shot
    exp_val_p0 /= n_shot
    exp_val_p1 /= n_shot
    return exp_val, exp_val_p0, exp_val_p1
