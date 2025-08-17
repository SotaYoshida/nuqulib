from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
import numpy as np

"""
Controlled Givens rotation
"""


def cG1(circ, c_qubit, i, j, theta):
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


def cG1_gate(theta, method="magic"):
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
    circ = circ.to_gate()
    circ = circ.control(1)
    return circ


def apply_sqrt_iSWAP(qc, i, j):
    sqrt_iSWAP_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 1 / np.sqrt(2), 1j / np.sqrt(2), 0],
            [0, 1j / np.sqrt(2), 1 / np.sqrt(2), 0],
            [0, 0, 0, 1],
        ]
    )
    qc.unitary(sqrt_iSWAP_matrix, [i, j], label="sqrt_iSWAP")


def Givens_iSWAP_Rz(circ, i, j, theta):
    """
    Givens rotation used in Google's HF paper, [Science 369, 1084 (2020)](DOI: 10.1126/science.abb9811).
    """
    theta = theta / 2
    apply_sqrt_iSWAP(circ, i, j)
    circ.rz(-theta, i)
    circ.rz(theta + np.pi, j)
    apply_sqrt_iSWAP(circ, i, j)
    circ.rz(np.pi, j)


def Givens_magic(circ, i, j, theta):
    """
    Optimal quantum circuits for general two-qubit gates, based on [Phys. Rev. A 69, 032315 (2004)](https://doi.org/10.1103/PhysRevA.69.032315).
    It was used in e.g. [npj Quantum Information (2023) 9:60](https://www.nature.com/articles/s41534-023-00730-8)
    Maybe optimal in terms of number of CNOTs.
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


def Givens_Xanadu(circ, i, j, theta):
    """
    Implementation given in paper by e.g. [Quantum 6, 742 (2022).](https://quantum-journal.org/papers/q-2022-06-20-742/)
    """
    theta_2 = theta / 2
    circ.cx(i, j)
    circ.ry(theta_2, i)
    circ.cx(j, i)
    circ.ry(-theta_2, i)
    circ.cx(j, i)
    circ.cx(i, j)


def G_gate(theta, method="magic"):
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


def Givens_2_Xanadu(circ, i, j, k, l, theta):
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


def G2_gate(theta, method="Xanadu"):
    circ = QuantumCircuit(4)
    if method == "Xanadu":
        Givens_2_Xanadu(circ, 0, 1, 2, 3, theta)
        circ.name = "G2(Xanadu)"
    else:
        raise ValueError("Invalid method for G2_gate")
    return circ.to_gate()


def get_idx_ancilla_in_string(n_qubit, ancilla, Qiskit_ordering):
    idx_ancilla = None
    if ancilla != None:
        if Qiskit_ordering:
            idx_ancilla = n_qubit - 1 - ancilla
        else:
            idx_ancilla = ancilla
    return idx_ancilla


def additional_qc(qc_in, pauli_str, register_target, Qiskit_order=True):
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


def get_idx_to_measure(pauli_str, Qiskit_order=True):
    idxs = [idx for idx, p in enumerate(pauli_str) if p != "I"]
    if Qiskit_order:
        idxs = [len(pauli_str) - 1 - idx for idx in idxs]
    return idxs


def expec_Zstring(
    res, idx_relevant, Qiskit_ordering=True, target_qubits=[], ancilla_qubit=None
):
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
