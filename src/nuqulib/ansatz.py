"""Quantum ansatz circuits for nuclear simulations.

This module provides various ansatz implementations for nuclear quantum simulations,
including Hartree-Fock states (lowest-filling more precisely),
Givens rotation-based circuits, and pair Unitary Coupled-Cluster Doubles (pUCCD).
"""

from collections.abc import Iterable
import numpy as np
import pennylane as qml
from pennylane import numpy as qnp
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from .circuits import G_gate, cG1_gate

def nucl_ansatz(
    n_qubit: int,
    proton_qubits: Iterable[int],
    neutron_qubits: Iterable[int],
    proton_number: int,
    neutron_number: int,
    params: Iterable[float],
    method: str = "HF",
    return_Gdict: bool = False,
):
    """Construct nuclear ansatz circuit for proton-neutron systems.
    
    Creates quantum circuits for nuclear many-body states with separate
    proton and neutron sectors. Supports Hartree-Fock initial states
    and Givens rotation-based variational ansätze.
    
    Args:
        n_qubit (int): Total number of qubits.
        proton_qubits (Iterable[int]): Indices of proton qubits.
        neutron_qubits (Iterable[int]): Indices of neutron qubits.
        proton_number (int): Number of protons.
        neutron_number (int): Number of neutrons.
        params (Iterable[float]): Variational parameters.
        method (str, optional): Ansatz method. Options: "HF", "HF+Givens". 
                               Defaults to "HF".
        return_Gdict (bool, optional): Whether to return gate type dictionary. 
                                      Defaults to False.
    
    Returns:
        QuantumCircuit or tuple: Quantum circuit representing the ansatz.
                                If return_Gdict=True, returns (circuit, gate_dict).
    """
    n_qubits_p = len(proton_qubits)

    where_is_G_or_cG1 = {}
    if method == "HF": # more precisely, lowest filling
        ansatz = QuantumCircuit(n_qubit)
        for i in range(proton_number):
            ansatz.x(proton_qubits[-1 - i])
        for i in range(neutron_number):
            ansatz.x(neutron_qubits[-1 - i])
        return ansatz
    elif method == "HF+Givens":
        ansatz = QuantumCircuit(n_qubit)
        for i in range(proton_number):
            ansatz.x(proton_qubits[-1 - i])
        for i in range(neutron_number):
            ansatz.x(neutron_qubits[-1 - i])
        ## Givens rotation
        ## on proton qubits
        count = 0
        for turn in range(proton_number):
            i_lowest = n_qubits_p - proton_number + turn
            if turn == 0:
                for G_lower in range(n_qubits_p - proton_number, 0, -1):
                    G_upper = G_lower - 1
                    ansatz.append(G_gate(params[count]), [G_upper, G_lower])
                    where_is_G_or_cG1[count] = "G"
                    count += 1
            else:
                for G_lower in range(i_lowest, 0, -1):
                    G_upper = G_lower - 1
                    for c in range(G_upper - 1, -1, -1):
                        ansatz.append(cG1_gate(params[count]), [c, G_upper, G_lower])
                        where_is_G_or_cG1[count] = "cG1"
                        count += 1

        # on neutron qubits
        n_qubits_p = len(proton_qubits)
        first_neutron_qubit = len(proton_qubits)
        for turn in range(neutron_number):
            i_lowest = n_qubit - neutron_number + turn
            if turn == 0:
                for G_lower in range(i_lowest, n_qubits_p, -1):
                    G_upper = G_lower - 1
                    ansatz.append(G_gate(params[count]), [G_upper, G_lower])
                    where_is_G_or_cG1[count] = "G"
                    count += 1
            else:  # c-G1
                for G_lower in range(i_lowest, n_qubits_p, -1):
                    G_upper = G_lower - 1
                    for c in range(G_upper - 1, -1, -1):
                        if c < first_neutron_qubit:
                            break
                        ansatz.append(cG1_gate(params[count]), [c, G_upper, G_lower])
                        where_is_G_or_cG1[count] = "cG1"
                        count += 1
        if return_Gdict:
            return ansatz, where_is_G_or_cG1
        return ansatz
    else:
        raise ValueError("Invalid method for ansatz: " + method)


def pair_ansatz_qiskit(
    params: Iterable[float],
    Norb: int,
    Nocc: int,
    method: str = "HF",
    return_Gdict: bool = False,
    decent_order: bool = True,
    rotation_XXYY=[],
    idxs_hole_in=[],
):
    """Construct pairing model ansatz circuit using Qiskit.
    
    Creates quantum circuits for pairing Hamiltonian simulations using
    Hartree-Fock states with optional Givens rotation enhancements.
    Designed for pairing Hamiltonian or pair-wise form of shell model,
    such as Hard-core boson formulation focused on seniority-zero states.
    
    Args:
        params (Iterable[float]): Variational parameters for gates, mainly Givens rotations.
        Norb (int): Number of orbitals (qubits).
        Nocc (int): Number of occupied orbitals.
        method (str, optional): Ansatz method. Options: "HF", "HF+Givens". 
                               Defaults to "HF".
        return_Gdict (bool, optional): Whether to return gate type dictionary. 
                                      Defaults to False.
        decent_order (bool, optional): Whether to use descending qubit ordering. 
                                      Defaults to True.
        rotation_XXYY (list, optional): List of XX+YY rotation parameters. 
                                       Defaults to []. This is used to diagonalize XX+YY term in the computational basis.
        idxs_hole_in (list, optional): Indices of hole states. Defaults to [].
    
    Returns:
        QuantumCircuit or tuple: Pairing ansatz circuit. If return_Gdict=True,
                                returns (circuit, gate_dict).
                                
    Note:
        The circuit starts with Hartree-Fock state preparation (X gates on
        occupied orbitals) followed by layer of Givens rotations,
        which can be ragarded as excitations from the reference state.
    """
    where_is_G_or_cG1 = {}
    qc = QuantumCircuit(Norb)
    if method == "HF+Givens":
        for i in range(Nocc):
            if decent_order:
                qc.x(Norb - 1 - i)
            else:
                qc.x(i)
        count = 0
        for turn in range(Nocc):
            if turn == 0:
                if decent_order:
                    for G_lower in range(Nocc, 0, -1):
                        G_upper = G_lower - 1
                        qc.append(G_gate(params[count]), [G_upper, G_lower])
                        where_is_G_or_cG1[count] = "G"
                        count += 1
                else:
                    for G_lower in range(Nocc, Norb):
                        G_upper = G_lower - 1
                        qc.append(G_gate(params[count]), [G_upper, G_lower])
                        where_is_G_or_cG1[count] = "G"
                        count += 1
            else:  # c-G1
                if decent_order:
                    for G_lower in range(Norb - Nocc + turn, 1, -1):
                        G_upper = G_lower - 1
                        for c in range(G_upper - 1, -1, -1):
                            qc.append(cG1_gate(params[count]), [c, G_upper, G_lower])
                            where_is_G_or_cG1[count] = "cG1"
                            count += 1
                else:
                    for G_lower in range(Nocc - turn, Norb - 1):
                        G_upper = G_lower - 1
                        for c in range(G_lower + 1, Norb):
                            qc.append(cG1_gate(params[count]), [c, G_upper, G_lower])
                            where_is_G_or_cG1[count] = "cG1"
                            count += 1
    if method == "pUCCD":
        for i in range(Nocc):
            if decent_order:
                qc.x(Norb - 1 - i)
            else:
                qc.x(i)
        theta_idx = 0
        for cycle in range(Nocc):
            if decent_order:
                for G_lower in range(Norb - Nocc + cycle, cycle, -1):
                    where_is_G_or_cG1[theta_idx] = "G"
                    G_upper = G_lower - 1
                    qc.append(G_gate(params[theta_idx]), [G_upper, G_lower])
                    theta_idx += 1
                    qc.swap(G_upper, G_lower)

            else:
                for G_lower in range(Nocc - cycle, Norb - cycle):
                    where_is_G_or_cG1[theta_idx] = "G"
                    G_upper = G_lower - 1
                    qc.append(G_gate(params[theta_idx]), [G_upper, G_lower])
                    theta_idx += 1
                    qc.swap(G_upper, G_lower)

    if method == "pUCCD+all2all":
        theta_idx = 0
        assert decent_order, "pUCCD+all2all should be decent order"

        if len(idxs_hole_in) > 0:
            for i in idxs_hole_in:
                qc.x(i)
            idxs_virt = [idx for idx in range(Norb) if idx not in idxs_hole_in]
            idxs_virt = sorted(idxs_virt, reverse=True)
            for q_v in idxs_virt:
                for q_h in idxs_hole_in:
                    where_is_G_or_cG1[theta_idx] = "G"
                    qc.append(G_gate(params[theta_idx]), [q_v, q_h])
                    theta_idx += 1
        else:
            idxs_hole = [i for i in range(Norb - Nocc, Norb)]
            if len(idxs_hole_in) > 0:
                idxs_hole = idxs_hole_in
            for i in idxs_hole:
                qc.x(i)
            idxs_virt = [idx for idx in range(Norb) if idx not in idxs_hole]
            idxs_virt = sorted(idxs_virt, reverse=True)
            for q_v in idxs_virt:
                for q_h in idxs_hole:
                    where_is_G_or_cG1[theta_idx] = "G"
                    qc.append(G_gate(params[theta_idx]), [q_v, q_h])
                theta_idx += 1

    ## Additional 2-qubit rotation gates to diagonalize XX+YY term in the computational basis
    if rotation_XXYY != []:  # not empty
        if len(rotation_XXYY) != 2:
            raise ValueError("rotation_XXYY should be a list of two qubits")
        qc.append(G_gate(-np.pi / 2), rotation_XXYY)

    if return_Gdict:
        return qc, where_is_G_or_cG1
    return qc


def pair_ansatz_pennylane(
    Hamil_pl,
    params,
    Nq: int,
    Nocc: int,
    type_of_ansatz: str="HF",
    observable: str="Hamil",
    return_Gdict: str=False,
    random_init_circ: str=False,
    idxs_hole_in=[],
    combination_h_v=[],
):
    """Construct pairing model ansatz circuit using PennyLane.

    This function is a counterpart of `pair_ansatz_qiskit` but uses PennyLane
    and only supports pUCCD ansatze.
    """
    local_where_is_G_or_cG1 = {}
    if type_of_ansatz != "pUCCD+all2all":
        for i in range(Nocc):
            qml.PauliX(Nq - 1 - i)

    theta_idx = 0

    ## pUCCD ansatz in hard core boson formulation
    if type_of_ansatz == "pUCCD":
        for cycle in range(Nocc):
            for i in range(Nq - Nocc + cycle, cycle, -1):
                local_where_is_G_or_cG1[theta_idx] = "G"
                qml.SingleExcitation(params[theta_idx], wires=[i - 1, i])
                theta_idx += 1
                qml.SWAP(wires=[i - 1, i])

    if type_of_ansatz == "pUCCD+all2all":
        """
        For devices with all-to-all connectivity, one can omit the SWAP gates in pUCCD ansatz above.
        """
        idxs_hole = []
        if random_init_circ:
            for i in idxs_hole_in:
                qml.PauliX(i)
            for i in range(len(combination_h_v)):
                q_h, q_v = combination_h_v[i]
                local_where_is_G_or_cG1[theta_idx] = "G"
                qml.SingleExcitation(params[theta_idx], wires=[q_v, q_h])
                theta_idx += 1

        else:
            if len(idxs_hole_in) > 0:
                for i in idxs_hole_in:
                    qml.PauliX(i)
                idxs_virt = [idx for idx in range(Nq) if idx not in idxs_hole_in]
                idxs_virt = sorted(idxs_virt, reverse=True)
                for q_v in idxs_virt:
                    for q_h in idxs_hole_in:
                        local_where_is_G_or_cG1[theta_idx] = "G"
                        qml.SingleExcitation(params[theta_idx], wires=[q_v, q_h])
                        theta_idx += 1
            else:
                for i in range(Nocc):
                    idx = Nq - 1 - i
                    qml.PauliX(idx)
                    idxs_hole.append(idx)
                idxs_hole = sorted(idxs_hole)
                idxs_virt = [idx for idx in range(Nq) if idx not in idxs_hole]
                idxs_virt = sorted(idxs_virt, reverse=True)

                for q_h in idxs_hole:
                    for q_v in idxs_virt:
                        local_where_is_G_or_cG1[theta_idx] = "G"
                        qml.SingleExcitation(params[theta_idx], wires=[q_v, q_h])
                        theta_idx += 1

    if return_Gdict:
        return local_where_is_G_or_cG1
    if observable == "Z" or observable == "ansatz":
        return qnp.array([qml.sample(qml.PauliZ(i)) for i in range(Nq)])
    elif observable == "Hamil":
        return qml.expval(Hamil_pl)
    else:
        raise ValueError(f"Invalid observable: {observable}")


def circuit_XXYY(
    adopted: bool,
    params: list[float],
    Nq: int,
    Nocc: int,
    method_ansatz: str,
    decent_order: bool = True,
    methods_XXYY: str = "Google",
    backend:str | None = None,
    opt_level=3,
    verbose:bool =False,
    idxs_hole_in=[],
):
    """
    Generates quantum circuits to measure X_iX_j+Y_iY_j terms in the Hamiltonian.

    Args:
        adopted (str):
            String that indicates the simulator/device type.
            Acceptable values are "simNISQ"/"simFTQC" and "Real".

    Returns
    -------
    list
        A list of transpiled and/or decomposed quantum circuits with measurements added. If verbose is True,
        a tuple is returned where the first element is the list of final circuits and the second element is the
        list of intermediate circuits prepared for drawing.
    Raises
    ------
    ValueError
        If the provided methods_XXYY is not "Google", which means that this function
        is designed for basis rotation (Givens rotation) to diagonalize XX+YY term in the computational basis.  
    """
    qc_list = []
    if methods_XXYY == "Google":
        qc_list_for_draw = []
        num_cycle = (Nq + 1) // 2
        idx_circuit = 0
        for cycle in range(num_cycle):
            for eo_pair in range(2):
                qc = pair_ansatz_qiskit(
                    params,
                    Nq,
                    Nocc,
                    method=method_ansatz,
                    decent_order=decent_order,
                    idxs_hole_in=idxs_hole_in,
                )
                for c in range(cycle):
                    # even swap
                    for i in range(0, Nq - 1, 2):
                        qc.swap(i, i + 1)
                    # odd swap
                    for i in range(1, Nq - 1, 2):
                        qc.swap(i, i + 1)
                i_start = Nq - 2 - eo_pair if Nq % 2 == 0 else Nq - 3 + eo_pair
                for i in range(i_start, -1, -2):
                    # print("idx_circuit", idx_circuit, "eo_pair", eo_pair, "i", i)
                    qc.append(G_gate(-np.pi / 2), [i, i + 1])

                if verbose:
                    qc_list_for_draw.append(qc.copy())
                qc.measure_all()
                qc = qc.decompose(reps=8)
                if adopted == "simNISQ":
                    qc = transpile(qc, backend, optimization_level=opt_level)
                elif adopted == "Real":
                    pm = generate_preset_pass_manager(backend=backend)
                    qc = pm.run(qc)
                elif adopted == "simFTQC":
                    pass
                else:
                    raise ValueError(f"Invalid adopted: {adopted}")
                qc_list.append(qc)
                idx_circuit += 1
        if verbose:
            return qc_list, qc_list_for_draw
    else:
        raise ValueError(f"Invalid methods_XXYY: {methods_XXYY}")
    return qc_list
