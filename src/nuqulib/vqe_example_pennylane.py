import numpy as np
import pennylane as qml
import openfermion as of
from pennylane import numpy as pnp  # pennylane numpy (using Torch for arrays)
import matplotlib.pyplot as plt
from .shellmodel_hamiltonian import *


def vqe_example_pennylane(
    Hdict,
    proton_number,
    neutron_number,
    n_qubits_p,
    n_qubits_n,
    using_chs,
    mapping_method="JordanWigner",
):
    n_qubits = n_qubits_p + n_qubits_n
    H_1b_of, H_pp_of, H_nn_of, H_pn_of = define_Hamil_in_OpenFermion(
        Hdict, proton_number, neutron_number
    )

    print("Hnn_of", H_nn_of)

    H = of.ops.FermionOperator()
    for ch in using_chs:
        if ch == "1b":
            H += H_1b_of
        if ch == "pp":
            H += H_pp_of
        if ch == "nn":
            H += H_nn_of
        if ch == "pn":
            H += H_pn_of
    if (
        mapping_method == "JW"
        or mapping_method == "Jordan-Wigner"
        or mapping_method == "JordanWigner"
    ):
        H_mapped = qml.import_operator(
            of.transforms.jordan_wigner(H), format="openfermion"
        )
    elif (
        mapping_method == "BK"
        or mapping_method == "Bravyi-Kitaev"
        or mapping_method == "BravyiKitaev"
    ):
        H_mapped = qml.import_operator(
            of.transforms.bravyi_kitaev(H), format="openfermion"
        )

    dev = qml.device("default.qubit", wires=n_qubits)
    global num_params

    @qml.qnode(dev)
    def ansatz(params, method="HF+Givens"):
        global num_params
        # HF state
        if method == "HF" or method == "HF+Givens":
            for i in range(proton_number):
                qml.PauliX(wires=n_qubits_p - i - 1)
            for i in range(neutron_number):
                qml.PauliX(wires=n_qubits - i - 1)
        # + Givens rotations
        if method == "HF+Givens":
            count = 0
            # for proton part (not implemented yet)
            for turn in range(proton_number):
                i_lowest = n_qubits_p - proton_number + turn
                for G_lower in range(i_lowest, 0, -1):
                    G_upper = G_lower - 1
                    qml.SingleExcitation(params[count], wires=[G_upper, G_lower])
                    count += 1

            # for neutron part
            first_neutron_qubit = n_qubits_p
            for turn in range(neutron_number):
                i_lowest = n_qubits - neutron_number + turn
                if turn == 0:
                    for G_lower in range(i_lowest, n_qubits_p, -1):
                        G_upper = G_lower - 1
                        qml.SingleExcitation(params[count], wires=[G_upper, G_lower])
                        count += 1
                else:  # c-G1
                    for G_lower in range(i_lowest, n_qubits_p, -1):
                        G_upper = G_lower - 1
                        for c in range(G_upper - 1, -1, -1):
                            if c < first_neutron_qubit:
                                break
                            qml.ctrl(qml.SingleExcitation, control=[c])(
                                params[count], wires=[G_upper, G_lower]
                            )
                            count += 1
            num_params = count
        return qml.expval(H_mapped)

    num_params = 100
    params = pnp.array([np.pi / 2 for _ in range(num_params)], requires_grad=True)
    ansatz(params, method="HF+Givens")
    params = params[:num_params]

    ## ****** Optimizing parameters ******
    N_it = 300
    optimizer = qml.AdamOptimizer(stepsize=0.3)

    energies = []
    for _ in range(N_it):
        params, _cost = optimizer.step_and_cost(ansatz, params)
        energies.append(_cost)
    print("Energy:", np.min(energies[-1]))
    print("Parameters:", params)

    params_opt = params.copy()

    plt.rcParams["font.size"] = 14
    plt.figure(figsize=(10, 6))
    plt.plot(energies, label="Energies")
    plt.xlabel("Iterations")
    plt.ylabel("Energy")
    plt.legend()
    plt.savefig("vqe_example_pennylane_" + mapping_method + ".pdf")

    return np.array(params_opt), np.min(energies)


def define_Hamil_in_OpenFermion(Hamil_dict, proton_number=1, neutron_number=1):
    H1b = of.ops.FermionOperator()
    Hpp = of.ops.FermionOperator()
    Hnn = of.ops.FermionOperator()
    Hpn = of.ops.FermionOperator()
    ## One-body terms
    h1b = Hamil_dict["SPE"]
    for key, value in h1b.items():
        i, j = key  # now assuming i = j
        i -= 1
        H1b += of.ops.FermionOperator(f"{i}^ {i}", value)

    # Two-body tems
    if proton_number > 0:
        for tmp in Hamil_dict["Vpp"]:
            a, b, c, d, V = tmp
            a = a - 1
            b = b - 1
            c = c - 1
            d = d - 1
            if [a, b] == [c, d]:
                Hpp += of.ops.FermionOperator(f"{a}^ {b}^ {d} {c}", V)
            else:
                Hpp += of.ops.FermionOperator(f"{a}^ {b}^ {d} {c}", V)
                Hpp += of.ops.FermionOperator(f"{c}^ {d}^ {b} {a}", V)

    if neutron_number > 0:
        for tmp in Hamil_dict["Vnn"]:
            a, b, c, d, V = tmp
            a = a - 1
            b = b - 1
            c = c - 1
            d = d - 1
            if [a, b] == [c, d]:
                Hnn += of.ops.FermionOperator(f"{a}^ {b}^ {d} {c}", V)
            else:
                Hnn += of.ops.FermionOperator(f"{a}^ {b}^ {d} {c}", V)
                Hnn += of.ops.FermionOperator(f"{c}^ {d}^ {b} {a}", V)

    if proton_number * neutron_number > 0:  # not checked yet
        for tmp in Hamil_dict["Vpn"]:
            a, b, c, d, V = tmp
            a = a - 1
            b = b - 1
            c = c - 1
            d = d - 1
            Hpn += of.ops.FermionOperator(f"{a}^ {b}^ {d} {c}", V / 2)
            Hpn += of.ops.FermionOperator(f"{d}^ {c}^ {a} {b}", V / 2)
    return H1b, Hpp, Hnn, Hpn
