import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter
from qiskit.circuit.library import QFT
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
import scipy
from .circuits import get_idx_to_measure, additional_qc, expec_Zstring
from tqdm import tqdm


def circuit_HadamardTest(
    Norb, Uprep, Hamiltonian_op, t, trotter_steps, using_statevector=True
):
    op = PauliEvolutionGate(
        Hamiltonian_op, t, synthesis=SuzukiTrotter(order=1, reps=trotter_steps)
    )
    U = op.definition
    U.name = "$U$"

    qr_Hadamard = QuantumRegister(Norb + 1)
    cr_Hadamard = ClassicalRegister(1)
    qc_Hadamard = QuantumCircuit(qr_Hadamard, cr_Hadamard)

    # State preparation
    qc_Hadamard.append(Uprep, range(Norb))

    # Hadamard on ancilla
    qc_Hadamard.h(Norb)
    # controlled-U
    cU = U.to_gate().control(1)
    qc_Hadamard.append(cU, [Norb] + list(range(Norb)))
    # Hadamard on ancilla and measurement
    qc_Hadamard.h(Norb)

    # for measurement
    qc_1 = qc_Hadamard.copy()
    qc_1.measure(Norb, 0)

    # for statevector
    qc_2 = qc_Hadamard.copy()
    if using_statevector:
        return qc_2.decompose(reps=5)
    else:
        return qc_1.decompose(reps=5)


def circuit_QPE(
    n_ancilla, Norb, Uprep, Hamiltonian_op, time, trotter_steps, measure=False
):
    qc_QPE = QuantumCircuit(n_ancilla + Norb, n_ancilla)
    register_ancilla = range(Norb, Norb + n_ancilla)
    register_target = range(Norb)
    # State preparation
    qc_QPE.append(Uprep, register_target)
    # Hadamard on ancilla
    for qubit in register_ancilla:
        qc_QPE.h(qubit)
    # Controlled-U operations
    U = PauliEvolutionGate(
        Hamiltonian_op, time, synthesis=SuzukiTrotter(order=1, reps=trotter_steps)
    )
    repetitions = 1
    for counting_qubit in register_ancilla:
        for _ in range(repetitions):
            U.label = f"U_(2^{int(np.log2(repetitions))})"
            U_ctrl = U.control(1)
            qc_QPE.append(U_ctrl, [counting_qubit] + list(register_target))
        repetitions *= 2
    # Inverse QFT
    qft_dagger = QFT(n_ancilla, inverse=True)
    qc_QPE.append(qft_dagger, register_ancilla)

    if measure:
        qc_QPE.measure(register_ancilla, range(n_ancilla))
    return qc_QPE


# The following functions are used in QuantumKrylov/ODMD

def make_U_and_cU(
    i, delta_t, trotter_steps, hamiltonian_op, ancilla_qubits, target_qubits, Uprep
):
    Ntar = len(target_qubits)
    time = i * delta_t
    circuit_U = QuantumCircuit(Ntar)
    expiHt = PauliEvolutionGate(
        hamiltonian_op, time, synthesis=SuzukiTrotter(order=1, reps=trotter_steps)
    )
    circuit_U.append(expiHt, range(Ntar))
    qc_U = circuit_U.decompose().decompose()

    qc_cU = QuantumCircuit(Ntar)
    qc_cU.append(Uprep, range(Ntar))
    qc_cU.append(expiHt, range(Ntar))
    qc_cU = qc_cU.decompose().decompose()
    qc_cU = qc_cU.to_gate().control(1)

    return qc_U, qc_cU


def make_overlap_qc(
    Ntar, gate_cUi, gate_cUj, ancilla_qubits, target_qubits, using_statevector
):
    qc_re = QuantumCircuit(1 + Ntar, 1)
    qc_re.h(0)
    qc_re.append(gate_cUi, ancilla_qubits + target_qubits)
    qc_re.x(0)
    qc_re.append(gate_cUj, ancilla_qubits + target_qubits)
    qc_re.h(0)
    if not using_statevector:
        qc_re.measure(0, 0)
    qc_re = qc_re.decompose()

    # Overlap, Im part
    qc_im = QuantumCircuit(1 + Ntar, 1)
    qc_im.h(0)
    qc_im.append(gate_cUi, ancilla_qubits + target_qubits)
    qc_im.x(0)
    qc_im.append(gate_cUj, ancilla_qubits + target_qubits)
    qc_im.sdg(0)
    qc_im.h(0)
    if not using_statevector:
        qc_im.measure(0, 0)
    qc_im = qc_im.decompose()

    return qc_re, qc_im


def measure_overlap(
    num_shot,
    Ntar,
    gate_cUi,
    gate_cUj,
    ancilla_qubits,
    target_qubits,
    sampler,
    backend,
    using_statevector,
    do_simulation=True,
):
    qc_re, qc_im = make_overlap_qc(
        Ntar, gate_cUi, gate_cUj, ancilla_qubits, target_qubits, using_statevector
    )
    if backend != None:
        print("transpile: w/ backend=", backend)
        qc_re = transpile(qc_re, backend=backend, optimization_level=2)
        qc_im = transpile(qc_im, backend=backend, optimization_level=2)
    if do_simulation:
        if using_statevector:
            results = [
                Statevector.from_instruction(qc).probabilities_dict()
                for qc in [qc_re, qc_im]
            ]
            prob_Re = results[0]
            prob_Im = results[1]
        else:
            job = sampler.run([qc_re, qc_im], shots=num_shot)
            results = job.result()
            prob_Re = results[0].data.c.get_counts()
            prob_Im = results[1].data.c.get_counts()

        p0 = np.sum(
            [count for bitstr, count in prob_Re.items() if bitstr[-1] == "0"]
        ) / np.sum(list(prob_Re.values()))
        p1 = np.sum(
            [count for bitstr, count in prob_Re.items() if bitstr[-1] == "1"]
        ) / np.sum(list(prob_Re.values()))
        ReN = p0 - p1

        p0 = np.sum(
            [count for bitstr, count in prob_Im.items() if bitstr[-1] == "0"]
        ) / np.sum(list(prob_Im.values()))
        p1 = np.sum(
            [count for bitstr, count in prob_Im.items() if bitstr[-1] == "1"]
        ) / np.sum(list(prob_Im.values()))
        ImN = p0 - p1

        U_ij = ReN + 1j * ImN
        return U_ij
    else:  # only resource estimation
        print("qc_re:", dict(qc_re.decompose(reps=5).count_ops()))
        print("qc_im:", dict(qc_im.decompose(reps=5).count_ops()))
        return None


def make_cU(Uprep, Ui, Ntar):
    circuit_cUi = QuantumCircuit(Ntar)
    circuit_cUi.append(Uprep, range(Ntar))
    circuit_cUi.append(Ui, range(Ntar))
    circuit_cUi = circuit_cUi.decompose(reps=5)
    return circuit_cUi.to_gate().control(1)


def make_nonD_H_qc(
    op_string,
    Ntar,
    ancilla_qubits,
    target_qubits,
    gate_cUi,
    gate_cUj,
    qcs_re,
    qcs_im,
    using_statevector,
):
    # real part
    qc_reH_nonD = QuantumCircuit(1 + Ntar)
    qc_reH_nonD.h(0)
    qc_reH_nonD.append(gate_cUi, ancilla_qubits + target_qubits)
    qc_reH_nonD.x(0)
    qc_reH_nonD.append(gate_cUj, ancilla_qubits + target_qubits)
    qc_reH_nonD.h(0)
    additional_qc(qc_reH_nonD, op_string, target_qubits)
    if not using_statevector:
        qc_reH_nonD.measure_all()
    qc_reH = qc_reH_nonD.decompose().decompose()
    qcs_re.append(qc_reH)

    # imaginary part
    qc_imH_nonD = QuantumCircuit(1 + Ntar)
    qc_imH_nonD.h(0)
    qc_imH_nonD.append(gate_cUi, ancilla_qubits + target_qubits)
    qc_imH_nonD.x(0)
    qc_imH_nonD.append(gate_cUj, ancilla_qubits + target_qubits)
    qc_imH_nonD.sdg(0)
    qc_imH_nonD.h(0)
    additional_qc(qc_imH_nonD, op_string, target_qubits)
    if not using_statevector:
        qc_imH_nonD.measure_all()
    qc_imH = qc_imH_nonD.decompose().decompose()
    qcs_im.append(qc_imH)

    return None


def QuantumKrylov(
    Uprep: QuantumCircuit,  # circuit to prepare a reference state
    hamiltonian_op: SparsePauliOp,  # Hamiltonian operator
    sampler,
    backend,
    ancilla_qubits,
    target_qubits,
    delta_t=0.01,
    max_iterations=10,
    trotter_steps=1,
    num_shot=10**4,
    using_statevector=False,
    do_simulation=True,
):
    if len(ancilla_qubits) == 0:
        raise ValueError(
            "ancilla_qubits = []! You may need ancilla qubits for the Quantum Krylov method."
        )
    if len(target_qubits) == 0:
        raise ValueError(
            "target_qubits = []! You may need target qubits for the Quantum Krylov method."
        )
    Hamil_coeffs = hamiltonian_op.coeffs
    Hamil_paulis = hamiltonian_op.paulis
    Ntar = len(target_qubits)
    N = np.zeros((max_iterations, max_iterations), dtype=np.complex128)
    H = np.zeros((max_iterations, max_iterations), dtype=np.complex128)
    ws = []
    Unitaries = []

    # To estimate the resource...
    Ui = PauliEvolutionGate(
        hamiltonian_op, delta_t, synthesis=SuzukiTrotter(order=1, reps=trotter_steps)
    )
    gate_cUi = make_cU(Uprep, Ui, Ntar)
    Uj = PauliEvolutionGate(
        hamiltonian_op,
        2 * delta_t,
        synthesis=SuzukiTrotter(order=1, reps=trotter_steps),
    )
    gate_cUj = make_cU(Uprep, Uj, Ntar)

    qc_Ui = QuantumCircuit(1 + Ntar)
    qc_Ui.append(gate_cUi, range(Ntar + 1))
    qc_Ui = qc_Ui.decompose(reps=5)
    print("Circuit for c-UiUp:", dict(qc_Ui.count_ops()))
    U_ij = measure_overlap(
        num_shot,
        Ntar,
        gate_cUi,
        gate_cUj,
        ancilla_qubits,
        target_qubits,
        sampler,
        backend,
        using_statevector,
        do_simulation,
    )
    num_of_Hamil_term = len(hamiltonian_op.paulis)
    print("num of Hamil term: ", num_of_Hamil_term)
    if not (do_simulation):
        return None
    for it in range(max_iterations):
        print("iteration: ", it)
        ## make controlled U = exp(-iHδt * it)
        Ui = PauliEvolutionGate(
            hamiltonian_op,
            it * delta_t,
            synthesis=SuzukiTrotter(order=1, reps=trotter_steps),
        )
        gate_cUi = make_cU(Uprep, Ui, Ntar)
        Unitaries.append(gate_cUi)
        N[it, it] = 1.0
        ## evaluate overlap to previous states
        for j in range(it - 1, -1, -1):
            gate_cUj = Unitaries[j]
            U_ij = measure_overlap(
                num_shot,
                Ntar,
                gate_cUi,
                gate_cUj,
                ancilla_qubits,
                target_qubits,
                sampler,
                backend,
                using_statevector,
                do_simulation,
            )
            N[it, j] = U_ij
            N[j, it] = np.conj(U_ij)
        ### evaluate H_ii no need ancilla qubit
        qcs = []
        for idx_H in range(len(Hamil_paulis)):
            op_string = Hamil_paulis[idx_H].to_label()
            idx_relevant = get_idx_to_measure(op_string)
            qc_reH_D = QuantumCircuit(Ntar)
            qc_reH_D.append(Uprep, range(Ntar))
            qc_reH_D.append(Ui, range(Ntar))
            additional_qc(qc_reH_D, op_string, range(Ntar))
            if not (using_statevector):
                qc_reH_D.measure_all()
            qc_reH_D = qc_reH_D.decompose().decompose()
            qcs.append(qc_reH_D)
        if using_statevector:
            results = [
                Statevector.from_instruction(qc).probabilities_dict() for qc in qcs
            ]
        else:
            job = sampler.run(qcs, shots=num_shot)
            results = job.result()

        Hsum = 0.0
        for idx_H in range(len(Hamil_paulis)):
            op_string = Hamil_paulis[idx_H].to_label()
            idx_relevant = get_idx_to_measure(op_string)
            if using_statevector:
                res = results[idx_H]
            else:
                res = results[idx_H].data.meas.get_counts()
            expval, dummy, dummy_ = expec_Zstring(res, idx_relevant)
            Hsum += Hamil_coeffs[idx_H] * expval
            ##print("operator: ", op_string, "coeff: ", Hamil_coeffs[idx_H], "exp. val: ",  expval)
        # print("H[it, it]", it, Hsum)
        H[it, it] = Hsum

        ### evaluate H_ij ancilla qubit is needed
        for j in range(it - 1, -1, -1):
            gate_cUj = Unitaries[j]
            qcs_re = []
            qcs_im = []
            for idx_H in range(len(Hamil_paulis)):
                op_string = Hamil_paulis[idx_H].to_label()
                make_nonD_H_qc(
                    op_string,
                    Ntar,
                    ancilla_qubits,
                    target_qubits,
                    gate_cUi,
                    gate_cUj,
                    qcs_re,
                    qcs_im,
                    using_statevector,
                )
            # Re part
            if using_statevector:
                results = [
                    Statevector.from_instruction(qc).probabilities_dict()
                    for qc in qcs_re
                ]
            else:
                job = sampler.run(qcs_re, shots=num_shot)
                results = job.result()
            Re_H_ij = 0.0
            for idx_H in range(len(Hamil_paulis)):
                op_string = Hamil_paulis[idx_H].to_label()
                idx_relevant = get_idx_to_measure(op_string)
                if using_statevector:
                    res = results[idx_H]
                else:
                    res = results[idx_H].data.meas.get_counts()
                dummy_e, p0, p1 = expec_Zstring(
                    res,
                    idx_relevant,
                    target_qubits=range(len(target_qubits)),
                    ancilla_qubit=0,
                )
                expval = p0 - p1
                Re_H_ij += Hamil_coeffs[idx_H] * expval
                # print("operator: ", op_string, "coeff: ", Hamil_coeffs[idx_H], "exp. val: ",  expval)
            # print("Re H_{ij}", Re_H_ij)

            # Im part
            if using_statevector:
                results = [
                    Statevector.from_instruction(qc).probabilities_dict()
                    for qc in qcs_im
                ]
            else:
                job = sampler.run(qcs_im, shots=num_shot)
                results = job.result()
            Im_H_ij = 0.0
            for idx_H in range(len(Hamil_paulis)):
                op_string = Hamil_paulis[idx_H].to_label()
                idx_relevant = get_idx_to_measure(op_string)
                if using_statevector:
                    res = results[idx_H]
                else:
                    res = results[idx_H].data.meas.get_counts()
                dummy_e, p0, p1 = expec_Zstring(
                    res,
                    idx_relevant,
                    target_qubits=range(len(target_qubits)),
                    ancilla_qubit=0,
                )
                expval = p0 - p1
                Im_H_ij += Hamil_coeffs[idx_H] * expval
            # print("Im H_{ij}", Im_H_ij)

            H[it, j] = Re_H_ij + 1j * Im_H_ij
            H[j, it] = Re_H_ij - 1j * Im_H_ij

        # solve generalized eigenvalue problem
        Nsub = N[: it + 1, : it + 1]
        Hsub = H[: it + 1, : it + 1]
        lam, v = scipy.linalg.eigh(Nsub)
        # truncate orthogonal basis with small eigenvalues
        cols = [i for i in range(it + 1) if lam[i] >= 1.0e-6]
        r = len(cols)
        Ur = v[:, cols]
        sq_Sigma_inv = np.diag(lam[cols] ** (-0.5))

        X = Ur @ sq_Sigma_inv @ Ur.conj().T
        Xdag = X.conj().T
        tildeH = X @ Hsub @ Xdag
        w, v = scipy.linalg.eigh(tildeH)

        w_r = w[-r:]
        ws += [w_r]

        print("eigs of N: ", lam, "cond", np.linalg.cond(Nsub), "r:", r)
        print(f"w: {w_r}")
        print("")
    return H[: it + 1, : it + 1], N[: it + 1, : it + 1], ws


def construct_X_and_Y(snapshots, d=8):  # d is the number of delay
    N = len(snapshots)
    X = np.zeros((d, N - d), dtype=np.complex128)
    Y = np.zeros((d, N - d), dtype=np.complex128)
    for j in range(N - d):
        for i in range(d):
            idx = j + i
            X[i, j] = snapshots[idx]
            Y[i, j] = snapshots[idx + 1]
    return X, Y


def ODMD(
    Uprep: QuantumCircuit,
    HamiltonianOps: SparsePauliOp,
    delta_t: float,
    max_iterations: int,
    trotter_steps: int,
    sampler,
    backend,
    ancilla_qubits,
    target_qubits,
    num_shot: int = 10**4,
    using_statevector: bool = True,
    d: int = 8,
    tol_SVD: float = 1.0e-8,
    verbose: bool = False,
):
    Ntar = len(target_qubits)

    cU0 = Uprep
    cU0 = cU0.to_gate().control(1)

    snapshots = np.zeros(max_iterations, dtype=np.complex128)
    snapshots[0] = 1.0
    for it in tqdm(range(1, max_iterations)):
        Uj = PauliEvolutionGate(
            HamiltonianOps,
            it * delta_t,
            synthesis=SuzukiTrotter(order=1, reps=trotter_steps),
        )

        gate_cUj = make_cU(Uprep, Uj, Ntar)
        overlap = measure_overlap(
            num_shot,
            Ntar,
            cU0,
            gate_cUj,
            ancilla_qubits,
            target_qubits,
            sampler,
            backend,
            using_statevector,
        )
        snapshots[it] = overlap
    print(
        f"Max iteration: {max_iterations:5d} trotter_steps: {trotter_steps:5d} delta_t: {delta_t:12.8f}"
    )

    # Observable DMD
    if verbose:
        print("snapshots of <U0|Uj>:", snapshots)
    X, Y = construct_X_and_Y(snapshots, d)

    # SVD of X
    U, Sigma, Vh = np.linalg.svd(X, full_matrices=False)
    if verbose:
        print("Sigma", Sigma)

    trank = np.sum(Sigma > tol_SVD)
    Ur = U[:, :trank]
    Sigmar = np.diag(Sigma[:trank])
    Vhr = Vh[:trank, :]

    # A =Y X^+ = Y (V Sigma^-1 Udag)
    A = Y @ Vhr.T.conj() @ np.linalg.inv(Sigmar) @ Ur.T.conj()

    # Check |AX - Y|
    print("Check |AX - Y|", np.linalg.norm(A @ X - Y))

    # Eigen values of A would be exp(-iE_j dt)
    lam, v = np.linalg.eig(A)

    print("lam", lam)
    print("|lam|", np.abs(lam))
    idx_remax = np.argmin(np.abs(np.abs(lam)-1))

    ## argument of eigenvalues
    arg_lam = np.angle(lam)
    # print("angle[/pi]", arg_lam / (np.pi))
    print("E?", list(-(arg_lam) / delta_t))

    arg_in0to2pi = arg_lam % (2 * np.pi)
    idx_E0 = np.argmax(arg_in0to2pi)
    E0 = -arg_lam[idx_E0] / delta_t

    Eremax = -arg_lam[idx_remax] / delta_t

    print(f"E0: {E0:12.8f} E closest to unit circle: {Eremax:12.8f}")
    print("")
    return Eremax
