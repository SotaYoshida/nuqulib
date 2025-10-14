"""Quantum ansatz circuits for nuclear simulations.

This module provides various ansatz implementations for nuclear quantum simulations,
including Hartree-Fock states (lowest-filling more precisely),
Givens rotation-based circuits, and pair Unitary Coupled-Cluster Doubles (pUCCD).
"""

from collections.abc import Iterable
import numpy as np
import os
import pennylane as qml
from pennylane import numpy as qnp
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliGate, PauliEvolutionGate
from qiskit_nature.second_q.operators import FermionicOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from .circuits import G_gate, cG1_gate
from .encoding import mapping_to_Pauli_string


def naive_filling_ansatz_old(proton_qubits: Iterable[int],
                         neutron_qubits: Iterable[int],
                         proton_number: int, neutron_number: int):
    n_qubit = len(proton_qubits) + len(neutron_qubits)
    ansatz = QuantumCircuit(n_qubit)
    for i in range(proton_number):
        ansatz.x(proton_qubits[-1 - i])
    for i in range(neutron_number):
        ansatz.x(neutron_qubits[-1 - i])
    return ansatz


def naive_filling_ansatz(
        proton_qubits: Iterable[int],
        neutron_qubits: Iterable[int],
        proton_number: int, neutron_number: int,
        mapping_method="JordanWigner",
        Hamildict_opform: dict = {},
        filepath: str|os.PathLike = "./"
    ):    
    N_dict = Hamildict_opform['SPE']['n']
    P_dict = Hamildict_opform['SPE']['p']
    n_keys = [key.split(" -_")[0] for key in N_dict.keys()]  
    p_keys = [key.split(" -_")[0] for key in P_dict.keys() ]

    n_qubits_p = len(p_keys)
    n_qubits_n = len(n_keys)
    n_qubits = n_qubits_p + n_qubits_n

    ansatz = QuantumCircuit(n_qubits)
    for i in range(proton_number): 
        paulistr = mapping_to_Pauli_string(
            FermionicOp({p_keys[-1-i]: 2.0}, num_spin_orbitals=n_qubits_p), 
            n_qubits, 0, method=mapping_method,Hamildict_specified=P_dict, filepath=filepath+'_p',
            ).paulis[0].to_label()
        pauli_op = PauliGate(paulistr)
        ansatz.append(pauli_op, list(range(pauli_op.num_qubits)))
    for i in range(neutron_number):
        paulistr = mapping_to_Pauli_string(
            FermionicOp({n_keys[-1-i]: 2.0}, num_spin_orbitals=n_qubits_n), 
            n_qubits, n_qubits_p, method=mapping_method,Hamildict_specified=N_dict, filepath=filepath+'_n',
            ).paulis[0].to_label()
        pauli_op = PauliGate(paulistr)
        ansatz.append(pauli_op, list(range(pauli_op.num_qubits)))
    return ansatz


# def nucl_ansatz(
#     Hamildict_opform: dict,
#     n_qubit: int,
#     proton_qubits: Iterable[int],
#     neutron_qubits: Iterable[int],
#     proton_number: int,
#     neutron_number: int,
#     params: Iterable[float],
#     method: str = "HF",
#     mapping_method: str = "JordanWigner",
#     filepath: str|os.PathLike = "./",
#     return_Gdict: bool = False,
# ):
#     """Construct nuclear ansatz circuit for proton-neutron systems.
    
#     Creates quantum circuits for nuclear many-body states with separate
#     proton and neutron sectors. Supports Hartree-Fock initial states
#     and Givens rotation-based variational ansätze.
    
#     Args:
#         n_qubit (int): Total number of qubits.
#         proton_qubits (Iterable[int]): Indices of proton qubits.
#         neutron_qubits (Iterable[int]): Indices of neutron qubits.
#         proton_number (int): Number of protons.
#         neutron_number (int): Number of neutrons.
#         params (Iterable[float]): Variational parameters.
#         method (str, optional): Ansatz method. Options: "HF", "HF+Givens". 
#                                Defaults to "HF".
#         return_Gdict (bool, optional): Whether to return gate type dictionary. 
#                                       Defaults to False.
    
#     Returns:
#         QuantumCircuit or tuple: Quantum circuit representing the ansatz.
#                                 If return_Gdict=True, returns (circuit, gate_dict).
#     """
#     assert len(proton_qubits) >= proton_number, f"Invalid proton_number={proton_number}"
#     assert len(neutron_qubits) >= neutron_number, f"Invalid neutron_number={neutron_number}"

#     n_qubits_p = len(proton_qubits)
#     where_is_G_or_cG1 = {}
#     if method == "HF": # more precisely, lowest filling
#         return naive_filling_ansatz(
#                 proton_qubits=proton_qubits,
#                 neutron_qubits=neutron_qubits,
#                 proton_number=proton_number,
#                 neutron_number=neutron_number,
#                 mapping_method=mapping_method,
#                 Hamildict_opform=Hamildict_opform,
#                 filepath=filepath
#         )
#     elif method == "HF+Givens":
#         ansatz = naive_filling_ansatz(
#             proton_qubits=proton_qubits,
#             neutron_qubits=neutron_qubits,
#             proton_number=proton_number,
#             neutron_number=neutron_number,
#             mapping_method=mapping_method,
#             Hamildict_opform=Hamildict_opform,
#             filepath=filepath
#         )
#         ## Givens rotation
#         ## on proton qubits
#         count = 0
#         for turn in range(proton_number):
#             i_lowest = n_qubits_p - proton_number + turn
#             if turn == 0:
#                 for G_lower in range(n_qubits_p - proton_number, 0, -1):
#                     G_upper = G_lower - 1
#                     ansatz.append(G_gate(params[count]), [G_upper, G_lower])
#                     where_is_G_or_cG1[count] = "G"
#                     count += 1
#             else:
#                 for G_lower in range(i_lowest, 0, -1):
#                     G_upper = G_lower - 1
#                     for c in range(G_upper - 1, -1, -1):
#                         ansatz.append(cG1_gate(params[count]), [c, G_upper, G_lower])
#                         where_is_G_or_cG1[count] = "cG1"
#                         count += 1

#         # on neutron qubits
#         n_qubits_p = len(proton_qubits)
#         first_neutron_qubit = len(proton_qubits)
#         for turn in range(neutron_number):
#             i_lowest = n_qubit - neutron_number + turn
#             if turn == 0:
#                 for G_lower in range(i_lowest, n_qubits_p, -1):
#                     G_upper = G_lower - 1
#                     ansatz.append(G_gate(params[count]), [G_upper, G_lower])
#                     where_is_G_or_cG1[count] = "G"
#                     count += 1
#             else:  # c-G1
#                 for G_lower in range(i_lowest, n_qubits_p, -1):
#                     G_upper = G_lower - 1
#                     for c in range(G_upper - 1, -1, -1):
#                         if c < first_neutron_qubit:
#                             break
#                         ansatz.append(cG1_gate(params[count]), [c, G_upper, G_lower])
#                         where_is_G_or_cG1[count] = "cG1"
#                         count += 1
#         if return_Gdict:
#             return ansatz, where_is_G_or_cG1
#         return ansatz
#     else:
#         raise ValueError("Invalid method for ansatz: " + method)


def nucl_ansatz(
    n_qubits: int, 
    proton_number: int, 
    neutron_number: int, 
    Hamildict_opform: dict, 
    params: Iterable[float], 
    method: str = "HF", 
    mapping_method: str = "JordanWigner", 
    filepath: str|os.PathLike = "./",
    return_Gdict: bool = False,
):
    
    N_dict = Hamildict_opform['SPE']['n']
    P_dict = Hamildict_opform['SPE']['p']
    n_keys = [key[:3] for key in N_dict.keys()]  
    n_keys_a = [key[3:] for key in N_dict.keys()]  
    p_keys = [key[:3] for key in P_dict.keys()]
    p_keys_a = [key[3:] for key in P_dict.keys()]
    n_qubits_p = len(p_keys)
    n_qubits_n = len(n_keys)
    
    
    assert proton_number <= n_qubits_p
    assert neutron_number <= n_qubits_n #TODO fix HF+Givens
    assert n_qubits_p + n_qubits_n == n_qubits

    where_is_G_or_cG1 = { }
    if method == "HF":
        ansatz = QuantumCircuit(n_qubits)
        for i in range(proton_number): 
            pauli_op = PauliGate(mapping_to_Pauli_string(
                FermionicOp({p_keys[-1-i]: 2.0}, num_spin_orbitals=n_qubits_p), 
                n_qubits, 0, method=mapping_method,Hamildict_specified=Hamildict_opform, filepath=filepath+'_p').paulis[0].to_label())
            ansatz.append(pauli_op, list(range(pauli_op.num_qubits)))
        for i in range(neutron_number):
            pauli_op = PauliGate(mapping_to_Pauli_string(
                FermionicOp({n_keys[-1-i]: 2.0}, num_spin_orbitals=n_qubits_n), 
                n_qubits, n_qubits_p, method=mapping_method,Hamildict_specified=Hamildict_opform, filepath=filepath+'_n').paulis[0].to_label())
            ansatz.append(pauli_op, list(range(pauli_op.num_qubits)))
        return ansatz
    elif method == "HF+Givens":
        ansatz = QuantumCircuit(n_qubits)
        for i in range(proton_number):
            pauli_op = PauliGate(mapping_to_Pauli_string(
                FermionicOp({p_keys[-1-i]: 2.0}, num_spin_orbitals=n_qubits_p), 
                n_qubits, 0, method=mapping_method,Hamildict_specified=Hamildict_opform, \
                filepath=filepath+'_p').paulis[0].to_label())
            ansatz.append(pauli_op, list(range(pauli_op.num_qubits)))
        for i in range(neutron_number):
            pauli_op = PauliGate(mapping_to_Pauli_string(
            FermionicOp({n_keys[-1-i]: 2.0}, num_spin_orbitals=n_qubits_n), \
                n_qubits, n_qubits_p, method=mapping_method,Hamildict_specified=Hamildict_opform, \
                filepath=filepath+'_n').paulis[0].to_label())
            ansatz.append(pauli_op, list(range(pauli_op.num_qubits)))
        if mapping_method == "JordanWigner" or mapping_method == "JW" or mapping_method == "Jordan-Wigner" or \
           mapping_method == "BravyiKitaev" or mapping_method == "BK" or mapping_method == "Bravyi-Kitaev":
            ## Givens rotation
            ## on proton qubits 
            count = 0
            for turn in range(proton_number):
                i_lowest = n_qubits_p - proton_number + turn
                if turn == 0:
                    for G_lower in range(n_qubits_p - proton_number, 0, -1):
                        G_upper = G_lower -1 
                        ansatz.append(G_gate(params[count]), [G_upper, G_lower])
                        where_is_G_or_cG1[count] = "G"
                        count += 1
                else:
                    for G_lower in range(i_lowest, 0, -1):
                        G_upper = G_lower -1
                        for c in range(G_upper-1, -1, -1):
                            ansatz.append(cG1_gate(params[count]), [c, G_upper, G_lower])
                            where_is_G_or_cG1[count] = "cG1"
                            count += 1
        
            # on neutron qubits
            # n_qubits_p = len(proton_qubits)        
            first_neutron_qubit =  n_qubits_p#len(proton_qubits)
            for turn in range(neutron_number):
                i_lowest = n_qubits - neutron_number + turn
                if turn == 0:
                    for G_lower in range(i_lowest, n_qubits_p, -1):
                        G_upper = G_lower -1 
                        ansatz.append(G_gate(params[count]), [G_upper, G_lower])
                        where_is_G_or_cG1[count] = "G"
                        count += 1
                else: # c-G1
                    for G_lower in range(i_lowest, n_qubits_p, -1):
                        G_upper = G_lower -1
                        for c in range(G_upper-1, -1, -1):
                            if c < first_neutron_qubit:
                                break
                            ansatz.append(cG1_gate(params[count]), [c, G_upper, G_lower])
                            where_is_G_or_cG1[count] = "cG1"
                            count += 1
        elif mapping_method == "HATTMapper":
            count = 0 
            for i in range(len(p_keys)):
                qi = int(p_keys[i].replace("_",""))
                for j in range(i+1,len(p_keys)):
                    qj = int(p_keys[j].replace("_",""))
                    term1 = mapping_to_Pauli_string(\
                        FermionicOp({p_keys[i]+p_keys_a[j]: 2.0}, num_spin_orbitals=n_qubits_p),\
                        n_qubits, 0, method=mapping_method,Hamildict_specified=Hamildict_opform,filepath=filepath+'_p')
                    term2 = mapping_to_Pauli_string(\
                        FermionicOp({p_keys[j]+p_keys_a[i]: 2.0}, num_spin_orbitals=n_qubits_p),\
                        n_qubits, 0, method=mapping_method,Hamildict_specified=Hamildict_opform,filepath=filepath+'_p')
                    term = term1+term2
                    term = term.simplify()
                    ansatz.append(PauliEvolutionGate(term,time = params[count]),range(n_qubits))
                    count += 1
            for i in range(len(n_keys)):
                qi = int(n_keys[i].replace("_",""))
                for j in range(i+1,len(n_keys)):
                    qj = int(n_keys[j].replace("_",""))
                    term1 = mapping_to_Pauli_string(\
                        FermionicOp({n_keys[i]+n_keys_a[j]: 2.0}, num_spin_orbitals=n_qubits_n),\
                        n_qubits, n_qubits_p, method=mapping_method,Hamildict_specified=Hamildict_opform,filepath=filepath+'_n')
                    term2 = mapping_to_Pauli_string(\
                        FermionicOp({n_keys[j]+n_keys_a[i]: 2.0}, num_spin_orbitals=n_qubits_n),\
                        n_qubits, n_qubits_p, method=mapping_method,Hamildict_specified=Hamildict_opform,filepath=filepath+'_n')
                    term = term1+term2
                    term = term.simplify()
                    ansatz.append(PauliEvolutionGate(term,time = params[count]),range(n_qubits))
                    count += 1
        if return_Gdict:
            return ansatz, where_is_G_or_cG1
        return ansatz
    else:
        raise ValueError("Invalid method for ansatz: "+method)


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
    if "HF" in method:
        for i in range(Nocc):
            if decent_order:
                qc.x(Norb - 1 - i)
            else:
                qc.x(i)
    if method == "HF+Givens":
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
    qc_ansatz: QuantumCircuit,
    adopted: str,
    Nq: int,
    methods_XXYY: str = "Google",
    backend:str | None = None,
    opt_level=3,
    verbose:bool =False,
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
                qc = qc_ansatz.copy()
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
