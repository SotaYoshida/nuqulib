from tabnanny import verbose

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.circuit.library import QFT
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import SuzukiTrotter
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
import scipy
from .circuits import get_idx_to_measure, expec_Zstring
from tqdm import tqdm


def circuit_HadamardTest(
    Norb, Uprep, Hamiltonian_op, t, trotter_steps, using_statevector=True
):
    op = PauliEvolutionGate(
        Hamiltonian_op, t, 
        synthesis=SuzukiTrotter(order=1, reps=trotter_steps)
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
        return qc_2.decompose(reps=3)
    else:
        return qc_1.decompose(reps=3)


def circuit_my_QPE(n_ancilla: int,
    Norb: int, 
    Hamiltonian_op: SparsePauliOp,
    Uprep: QuantumCircuit,
    time: float, 
    measure=False,
    trotter_order: int = 2,
    trotter_steps: int = 1,
    repeat: bool = False
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
    unitU = PauliEvolutionGate(Hamiltonian_op, time,
                               synthesis=SuzukiTrotter(order=trotter_order, reps=trotter_steps))
    for iter, counting_qubit in enumerate(register_ancilla):
        Upow = QuantumCircuit(Norb)
        if repeat:
            for _ in range(2**iter):
                Upow.compose(unitU, inplace=True)
        else:
            Upow = PauliEvolutionGate(Hamiltonian_op, time * (2**iter),
                                      synthesis=SuzukiTrotter(order=trotter_order, reps=trotter_steps))
        cU = Upow.control()
        cU.name = "$U^{2^{" + str(iter) + "}}$"
        qc_QPE.compose(cU, qubits=[counting_qubit] + list(register_target), inplace=True)

    # Inverse QFT
    qft_dagger = QFT(n_ancilla, inverse=True)
    qc_QPE.append(qft_dagger, register_ancilla)

    if measure:
        qc_QPE.measure(register_ancilla, range(n_ancilla))
    return qc_QPE


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
    qc_im = qc_re.copy() # make a copy for Im part
    qc_re.h(0)
    qc_im.sdg(0)
    qc_im.h(0)
    if not using_statevector:
        qc_re.measure(0, 0)
        qc_im.measure(0, 0)
    qc_re = qc_re.decompose()
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
    using_statevector,
    do_simulation=True,
):
    qc_re, qc_im = make_overlap_qc(
        Ntar, gate_cUi, gate_cUj, ancilla_qubits, target_qubits, using_statevector
    )
    #print(f"before transpile...:", qc_re.count_ops())
    qc_re = transpile(qc_re, sampler, optimization_level=2)
    qc_im = transpile(qc_im, sampler, optimization_level=2)
    #print(f"after transpile...:", qc_re.count_ops())
    if do_simulation:
        if using_statevector:
            results = [ ]
            sim = AerSimulator(method='statevector')
            for qc in [qc_re, qc_im]:
                qc_sv = transpile(qc, sim)
                qc_sv.save_statevector()
                job = sim.run(qc_sv)
                result = job.result()
                psi_final = result.get_statevector(qc_sv)
                results.append(psi_final.probabilities_dict())
            prob_Re = results[0]
            prob_Im = results[1]
        else:
            sampler = SamplerV2() if sampler is None else sampler
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
        print("qc_re:", dict(qc_re.decompose(reps=1).count_ops()))
        print("qc_im:", dict(qc_im.decompose(reps=1).count_ops()))
        return None


def make_cU(Uprep, Ui, Ntar):
    circuit_cUi = QuantumCircuit(Ntar)
    circuit_cUi.append(Uprep, range(Ntar))
    circuit_cUi.append(Ui, range(Ntar))
    return circuit_cUi.decompose().to_gate().control(1)


def make_Circ_forNondiagH(term_types,
                          Ntar, ancilla_qubits, target_qubits, 
                          gate_cUi, gate_cUj, qcs_re, qcs_im, using_statevector):
    
    for idx_term in range(len(term_types)):
        term = term_types[idx_term]

        qc = QuantumCircuit(1+Ntar)
        
        # gate for the ancilla qubit
        qc.h(0)
        qc.append(gate_cUi, ancilla_qubits + target_qubits)
        qc.x(0)
        qc.append(gate_cUj, ancilla_qubits + target_qubits)
        qc_im = qc.copy() # copy here
        qc.h(0)
        qc_im.sdg(0)
        qc_im.h(0)

        # gates for target qubits
        if term == "IZ":
            pass
        elif term == "XX":
            qc.h(target_qubits)
            qc_im.h(target_qubits)
        elif term == "YY":
            qc.sdg(target_qubits)
            qc.h(target_qubits)
            qc_im.sdg(target_qubits)
            qc_im.h(target_qubits)
        else:
            pauli_locs = term.split(",")
            for loc in pauli_locs:
                if loc.startswith("X_"):
                    qubit_idx = int(loc[2:]) 
                    qc.h(target_qubits[qubit_idx])
                elif loc.startswith("Y_"):
                    qubit_idx = int(loc[2:])
                    qc.sdg(target_qubits[qubit_idx])
                    qc.h(target_qubits[qubit_idx])
                elif loc.strip() == "":
                    continue
                else:
                    raise ValueError(f"Unexpected term in XYstr: {loc}. Supported formats are 'IZ', 'XX', 'YY' or 'X_i', 'Y_i' for i-th qubit.")

        if not(using_statevector):
            qc.measure_all()
            qc_im.measure_all()

        qc = qc.decompose()
        qcs_re.append(qc)

        qc_im = qc_im.decompose()
        qcs_im.append(qc_im)

    return None

def get_idx_circuit(op_string, term_types):
    idx_circuit = None
    if set(op_string) == {'I', 'Z'} or set(op_string) == {'I'}:                            
        for i in range(len(term_types)):
            if set(term_types[i]) == {'I', 'Z'} or set(term_types[i]) == {'I'}:
                idx_circuit = i
                break
        if idx_circuit is None:
            raise ValueError(f"Corresponding circuit for {op_string} not found in term_types: {term_types}")
    elif set(op_string) == {'X', 'I'} or set(op_string) == {'X'}:
        for i in range(len(term_types)):
            if set(term_types[i]) == {'X', 'I'} or set(term_types[i]) == {'X'}:
                idx_circuit = i
                break
        if idx_circuit is None:
            raise ValueError(f"Corresponding circuit for {op_string} not found in term_types: {term_types}")
    elif set(op_string) == {'Y', 'I'} or set(op_string) == {'Y'}:
        for i in range(len(term_types)):
            if set(term_types[i]) == {'Y', 'I'} or set(term_types[i]) == {'Y'}:
                idx_circuit = i
                break
        if idx_circuit is None:
            raise ValueError(f"Corresponding circuit for {op_string} not found in term_types: {term_types}")
    else:
        raise ValueError(f"Unexpected operator string: {op_string}. Supported types are 'IZ', 'XX', 'YY' for now.")
    return idx_circuit     


def prepare_qc_for_QKrylov(Hamiltonian_op, Uprep, Ui, Ntar, Bosonic, using_statevector=False, verbose=True):
    """
    Prepare quantum circuits for evaluating Hamiltonian terms in QKrylov method.
    This could be also used for VQE-type algorithms where one needs to evaluate the expectation value of Hamiltonian terms.
    """
    qcs = [ ] 
    term_types = [ ]
    idxs_circuit = [ None for _ in range(len(Hamiltonian_op.paulis)) ]
    for idx_H in range(len(Hamiltonian_op.paulis)):
        op_string = Hamiltonian_op.paulis[idx_H].to_label()
        Xloc = [Ntar - 1 - i for i, char in enumerate(op_string) if char == 'X']
        Yloc = [Ntar - 1 - i for i, char in enumerate(op_string) if char == 'Y']
        Xloc = list(set(Xloc))
        Yloc = list(set(Yloc))
        dupricate = False
        # Check whether the term to be measured the circuits already prepared 
        XYstr = trans_XYloc_str(Xloc, Yloc, Bosonic)
        if XYstr not in term_types:
            term_types.append(XYstr)
        else:
            dupricate = True    
        
        idx_G = term_types.index(XYstr)
        idxs_circuit[idx_H] = idx_G

        if dupricate:
            continue
        qc = QuantumCircuit(Ntar)
        qc.append(Uprep, range(Ntar))
        qc.append(Ui, range(Ntar))
        if XYstr == "IZ":
            pass
        elif XYstr == "XX":
            qc.h(range(Ntar))
        elif XYstr == "YY":
            qc.sdg(range(Ntar))
            qc.h(range(Ntar))
        else:
            pauli_locs = XYstr.split(",")
            for loc in pauli_locs:
                if loc.startswith("X_"):
                    qubit_idx = int(loc[2:])
                    qc.h(qubit_idx)
                elif loc.startswith("Y_"):
                    qubit_idx = int(loc[2:])
                    qc.sdg(qubit_idx)
                    qc.h(qubit_idx)
                elif loc.strip() == "":
                    continue
                else:
                    raise ValueError(f"Unexpected term in XYstr: {loc}. Supported formats are 'IZ', 'XX', 'YY' or 'X_i', 'Y_i' for i-th qubit.")
        qcs.append(qc)
    return qcs, term_types, idxs_circuit


def trans_XYloc_str(Xloc, Yloc, Bosonic):
    if len(Xloc) == 0 and len(Yloc) == 0:
        return "IZ"
    elif len(Xloc) > 0 and len(Yloc) == 0:
        if Bosonic:
            txt = "XX"
        else:
            txt = ""
            for Xi in Xloc:
                txt += "X_"+str(Xi)+","
        return txt
    elif len(Xloc) == 0 and len(Yloc) > 0:
        if Bosonic:
            txt = "YY"
        else:
            txt = ""
            for Yi in Yloc:
                txt += "Y_"+str(Yi)+","
        return txt
    elif len(Xloc) > 0 and len(Yloc) > 0:
        if Bosonic:
            raise ValueError(f"Bosonic case does not support terms with both X and Y. But got Xloc: {Xloc} and Yloc: {Yloc}.")
        else:
            txt = ""
            for Xi in Xloc:
                txt += "X_"+str(Xi)+","
            for Yi in Yloc:
                txt += "Y_"+str(Yi)+","
            return txt
    else:
        raise ValueError(f"Unsupported term with Xloc: {Xloc} and Yloc: {Yloc}.")


def reorder_based_on_layout(res, 
                            qlayout):
    """
    Transpilors sometimes change the order of qubits, so we need to reorder the bitstrings according to the mapping given by `qlayout`.
    """
    if qlayout is None:
        return res
    new_res = { }
    q_measured = qlayout.routing_permutation() 
    for bitstring, value in res.items():
        new_bitstring = ["0"] * len(bitstring)
        for idx, bit in enumerate(bitstring):
            phys_q = q_measured[idx]
            new_bitstring[phys_q] = bit
        new_bitstring = "".join(new_bitstring)
        new_res[new_bitstring] = value
    return new_res


def QuantumKrylov(
    Uprep: QuantumCircuit,  # circuit to prepare a reference state
    hamiltonian_op: SparsePauliOp,  # Hamiltonian operator
    sampler,
    ancilla_qubits,
    target_qubits,
    delta_t=0.01,
    max_iterations=10,
    trotter_rank=2,
    trotter_steps=1,
    num_shot=10**4,
    using_statevector=False,
    do_simulation=True,
    Bosonic=True,
    verbose=False
):
    """Function to perform Quantum Krylov subspace method.

    This function implements the Quantum Krylov subspace method for simulating quantum dynamics.

    Note:
      Within the current implementation, we assume that the Hamiltonian is pairing or pair-wise one,
      leading to only I, Z, and XX+YY terms. Under this condition, the number of additional
      quantum circuits needed is only two.
      One consists of ansatz + Hadamard on all target qubits, and the other consists of
      ansatz + Sdg followed by Hadamard gate on all target qubits.
    """
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
        hamiltonian_op, delta_t, synthesis=SuzukiTrotter(order=trotter_rank, reps=trotter_steps)
    )
    gate_cUi = make_cU(Uprep, Ui, Ntar)
    Uj = PauliEvolutionGate(
        hamiltonian_op,
        2 * delta_t,
        synthesis=SuzukiTrotter(order=trotter_rank, reps=trotter_steps),
    )
    gate_cUj = make_cU(Uprep, Uj, Ntar)

    qc_Ui = QuantumCircuit(1 + Ntar)
    qc_Ui.append(gate_cUi, range(Ntar + 1))
    qc_Ui = qc_Ui.decompose()

    num_of_Hamil_term = len(hamiltonian_op.paulis)
    print("num of Hamil term: ", num_of_Hamil_term)
    if not (do_simulation):
        return None

    #term_types = { 0:"IZ", 1:"XX", 2:"YY"}  # general case => "XXYZ..."
    for it in tqdm(range(max_iterations)):
        print("iteration: ", it)
        ## make controlled U = exp(-iHδt * it)
        Ui = PauliEvolutionGate(hamiltonian_op, it*delta_t, 
                                synthesis=SuzukiTrotter(order=trotter_rank,reps=trotter_steps))
        qcs, term_types, idxs_circuit = prepare_qc_for_QKrylov(hamiltonian_op, Uprep, Ui, Ntar, Bosonic, using_statevector=using_statevector, verbose=True)

        gate_cUi = make_cU(Uprep, Ui, Ntar)
        Unitaries.append(gate_cUi)
        N[it, it] = 1.0
        ## evaluate overlap to previous states
        for j in range(it-1, -1, -1):
            gate_cUj = Unitaries[j]
            U_ij = measure_overlap(num_shot, Ntar, gate_cUi, gate_cUj, ancilla_qubits, target_qubits, 
                                   sampler, using_statevector, do_simulation)
            N[it, j] = U_ij
            N[j, it] = np.conj(U_ij)
        ### evaluate H_ii no need ancilla qubit                      
        if using_statevector:
            sim = AerSimulator(method='statevector')
            results = [ ]
            for qc in qcs:
                qc_sv = transpile(qc, sim)
                qc_sv.save_statevector()
                job = sim.run(qc_sv)
                result = job.result()
                psi_final = result.get_statevector(qc_sv)
                results.append([psi_final.probabilities_dict(), qc_sv.layout])
        else:
            job = sampler.run(qcs, shots=num_shot)
            results  = job.result()

        # Comparison with Aer statevector simulator
        Hsum = 0.0
        for idx_H in range(len(Hamil_paulis)):
            op_string = Hamil_paulis[idx_H].to_label()
            idx_relevant = get_idx_to_measure(op_string)
            idx_circuit = idxs_circuit[idx_H]
            if using_statevector:
                res, qlayout = results[idx_circuit]
                res = reorder_based_on_layout(res, qlayout)
            else:
                res = results[idx_circuit].data.meas.get_counts()                    
            expval, dummy, dummy_ = expec_Zstring(res, idx_relevant)
            # if verbose:
            #     print(f"idx_H: {idx_H}, op_string: {op_string},",
            #           f"idx_rel: {idx_relevant}, <op> ", expval,
            #           f"expval: {expval * Hamil_coeffs[idx_H]}")
            Hsum += Hamil_coeffs[idx_H] * expval
        H[it, it] = Hsum
        if verbose:
            print(f"H[diag={it}] = {Hsum}")

        ### evaluate H_ij non-diagonal terms where an ancilla qubit is needed
        for j in range(it-1, -1, -1):
            gate_cUj = Unitaries[j]
            qcs_re = []
            qcs_im = []
            make_Circ_forNondiagH(term_types,
                                  Ntar, ancilla_qubits, target_qubits,
                                  gate_cUi, gate_cUj, qcs_re, qcs_im, 
                                  using_statevector)

            # Re part
            if using_statevector:
                results_Re = []
                results_Im = []
                sim = AerSimulator(method='statevector')
                for qc in qcs_re:
                    qc_sv = transpile(qc, sim)
                    qc_sv.save_statevector()
                    job = sim.run(qc_sv)
                    result = job.result()
                    psi_final = result.get_statevector(qc_sv)
                    results_Re.append([psi_final.probabilities_dict(), qc_sv.layout])
                for qc in qcs_im:
                    qc_sv = transpile(qc, sim)
                    qc_sv.save_statevector()
                    job = sim.run(qc_sv)
                    result = job.result()
                    psi_final = result.get_statevector(qc_sv)
                    results_Im.append([psi_final.probabilities_dict(), qc_sv.layout])
            else:
                job = sampler.run(qcs_re, shots=num_shot)
                results_Re = job.result()
                job = sampler.run(qcs_im, shots=num_shot)
                results_Im = job.result()

            Re_H_ij = Im_H_ij = 0.0
            for idx_H in range(len(Hamil_paulis)):
                op_string = Hamil_paulis[idx_H].to_label()
                idx_relevant = get_idx_to_measure(op_string)
                idx_circuit = idxs_circuit[idx_H]
                if using_statevector:
                    res_Re, layout_Re = results_Re[idx_circuit]
                    res_Re = reorder_based_on_layout(res_Re, layout_Re)
                    res_Im, layout_Im = results_Im[idx_circuit]
                    res_Im = reorder_based_on_layout(res_Im, layout_Im)
                else:
                    res_Re = results_Re[idx_circuit].data.meas.get_counts()
                    res_Im = results_Im[idx_circuit].data.meas.get_counts()
                dummy_e, p0, p1 = expec_Zstring(res_Re, idx_relevant, target_qubits=range(len(target_qubits)), ancilla_qubit=0)
                expval = p0 - p1
                Re_H_ij += Hamil_coeffs[idx_H] * expval

                dummy_e, p0, p1 = expec_Zstring(res_Im, idx_relevant, target_qubits=range(len(target_qubits)), ancilla_qubit=0)
                expval = p0 - p1
                Im_H_ij += Hamil_coeffs[idx_H] * expval

            H[it, j] = Re_H_ij + 1j * Im_H_ij
            H[j, it] = Re_H_ij - 1j * Im_H_ij
            print(f"H[off-diag={it},{j}] = {H[it, j]}")

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


def lambda_plot(lam, Ens):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    for idx, point in enumerate(lam):
        re = np.real(point)
        im = np.imag(point)
        ax.plot(re, im, 'x', label=f"<E>={Ens[idx]:8.3f} MeV")
    circle = patches.Circle((0, 0), 1.0, edgecolor='green', facecolor='none', linewidth=2)
    ax.add_patch(circle)
    ax.legend()
    plt.savefig("lambda_plot.pdf", bbox_inches='tight', pad_inches = 0.05)
    plt.close()


def ODMD(
    Uprep: QuantumCircuit,
    HamiltonianOps: SparsePauliOp,
    delta_t: float,
    max_iterations: int,
    trotter_rank: int,
    trotter_steps: int,
    sampler,
    ancilla_qubits,
    target_qubits,
    num_shot: int = 10**4,
    using_statevector: bool = True,
    dim_Hankel: int = 8,
    tol_SVD: float = 1.0e-8,
    verbose: bool = False,
    plot_lambda=False,
    tol_lambda = 1.e-2,
    energy_shift=0.0
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
            synthesis=SuzukiTrotter(order=trotter_rank, reps=trotter_steps),
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
            using_statevector,
        )
        print(f"overlap @{it:3d}: {overlap}")
        snapshots[it] = overlap
    print(
        f"Max iteration: {max_iterations:5d} trotter_steps: {trotter_steps:5d} delta_t: {delta_t:12.8f}",
        f" tol for lambda = {tol_lambda:8.2e}",
    )

    # Observable DMD
    if verbose:
        print("snapshots of <U0|Uj>:", snapshots)
    if dim_Hankel >= max_iterations:
        dim_Hankel = max_iterations // 2 + 1
        print(f"dim_Hankel is set to {dim_Hankel} since the original value is too large for the number of snapshots.")

    X, Y = construct_X_and_Y(snapshots, dim_Hankel)

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
    lams, v = np.linalg.eig(A)

    idxs = [ ]
    while len(idxs) ==0:
        idxs = [ i for i in range(len(lams)) if np.abs((np.abs(lams[i])-1)) < tol_lambda]
        lam = lams[idxs]
        if len(idxs) == 0:
            print(f"No eigenvalue found within |lambda|=1 +/- {tol_lambda:8.2e}.")
            print("Increasing the tolerance by a factor of 10.")
            tol_lambda *= 10

    print("|lam|", np.abs(lam))
    idx_remax = np.argmin(np.abs(np.abs(lam)-1))

    ## argument of eigenvalues
    arg_lam = np.angle(lam)
    Ens = list(-(arg_lam) / delta_t)
    Ens = [ E + energy_shift for E in Ens]
    print("Ens:", Ens)

    arg_in0to2pi = arg_lam % (2 * np.pi)
    idx_E0 = np.argmax(arg_in0to2pi)
    E0 = -arg_lam[idx_E0] / delta_t
    Eremax = -arg_lam[idx_remax] / delta_t

    if plot_lambda:
        lambda_plot(lam, Ens)
    return Ens
