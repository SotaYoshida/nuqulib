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
from .myutils import t_count
from .circuits import get_idx_to_measure, expec_Zstring
from tqdm import tqdm


def circuit_HadamardTest(
    Norb, Uprep, Hamiltonian_op, t, trotter_steps, using_statevector=True
):
    """Build a Hadamard-test circuit for a time-evolution operator.

    Args:
        Norb (int): Number of target orbitals/qubits.
        Uprep (QuantumCircuit): State-preparation circuit on the target register.
        Hamiltonian_op (SparsePauliOp): Hamiltonian used for Pauli evolution.
        t (float): Evolution time.
        trotter_steps (int): Number of Suzuki-Trotter repetitions.
        using_statevector (bool, optional): If True, omit measurement and return
            a statevector-ready circuit.

    Returns:
        QuantumCircuit: Decomposed Hadamard-test circuit.
    """
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


def T_formula_QFT(N_ancilla, eps_rotation: float=1.e-10):
    """Estimate T-count for the inverse QFT rotation gates.

    Args:
        N_ancilla (int): Number of QPE ancilla qubits.
        eps_rotation (float, optional): Rotation synthesis tolerance.

    Returns:
        float: Estimated T-count.
    """
    Teps = np.ceil( 3 * np.log2(1/eps_rotation))
    return N_ancilla * (N_ancilla - 1) // 2 * 2 * Teps


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
    """Construct a basic quantum phase-estimation circuit.

    Args:
        n_ancilla (int): Number of counting qubits.
        Norb (int): Number of target qubits.
        Hamiltonian_op (SparsePauliOp): Hamiltonian used for time evolution.
        Uprep (QuantumCircuit): State-preparation circuit.
        time (float): Base evolution time.
        measure (bool, optional): If True, measure the ancilla register.
        trotter_order (int, optional): Suzuki-Trotter order.
        trotter_steps (int, optional): Number of Suzuki-Trotter repetitions.
        repeat (bool, optional): If True, build powers by repeated composition
            instead of scaling the evolution time.

    Returns:
        QuantumCircuit: QPE circuit.
    """
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


class myTextBookQPE:
    """Small wrapper around textbook quantum phase estimation."""

    def __init__(self, n_ancilla, Norb, Hamiltonian_op,
                  Uprep, time, trotter_order=2, trotter_steps=1):
        """Store QPE construction and resource-estimation parameters."""
        self.Na = n_ancilla
        self.Norb = Norb
        self.Hamiltonian_op = Hamiltonian_op
        self.Uprep = Uprep
        self.time = time
        self.trotter_order = trotter_order
        self.trotter_steps = trotter_steps


    def construct_circuit(self):
        """Construct the measured textbook QPE circuit."""
        return circuit_my_QPE(
            self.Na,
            self.Norb,
            self.Hamiltonian_op,
            self.Uprep,
            self.time,
            measure=True,
            trotter_order=self.trotter_order,
            trotter_steps=self.trotter_steps,
            repeat=True
        )


    def estimate_resource(self, tol=1.e-10, verbose=False):
        """Estimate the T-count for the stored QPE circuit parameters.

        Args:
            tol (float, optional): Rotation synthesis tolerance.
            verbose (bool, optional): If True, print intermediate estimates.

        Returns:
            None: The estimate is printed to stdout.
        """
        T_epsilon = t_count(tol)
        T_U = self.trotter_steps * self.trotter_order * ( len(self.Hamiltonian_op.paulis) - 1 ) * T_epsilon
        T_cU = 2 * T_U
        N_cU = (2**self.Na - 1)
        TQFT = T_formula_QFT(self.Na)
        Tcount = T_cU * N_cU + TQFT
        print(f"Estimated T-count for textbook QPE: {Tcount} (log10={np.log10(Tcount):.1f})")
        if verbose:
            print(f"  T_epsilon (for rotation synthesis): {T_epsilon}")
            print(f"  T_cU (for one controlled-U including rotation synthesis): {T_cU}")
            print(f"  N_cU (number of controlled-U's): {N_cU}")
            print(f"  TQFT (for inverse QFT): {TQFT}")
        return Tcount


def make_overlap_qc(
    Ntar, gate_cUi, gate_cUj, ancilla_qubits, target_qubits, using_statevector
):
    """Build circuits for real and imaginary parts of an overlap test.

    Args:
        Ntar (int): Number of target qubits.
        gate_cUi: Controlled unitary for the first state.
        gate_cUj: Controlled unitary for the second state.
        ancilla_qubits (list): Ancilla qubits used by the controlled gates.
        target_qubits (list): Target qubits acted on by the controlled gates.
        using_statevector (bool): If False, add measurements.

    Returns:
        tuple[QuantumCircuit, QuantumCircuit]: Circuits for real and imaginary
        overlap components.
    """
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
    """Estimate an overlap from Hadamard-test circuits.

    Args:
        num_shot (int): Number of shots for sampled execution.
        Ntar (int): Number of target qubits.
        gate_cUi: Controlled unitary for the first state.
        gate_cUj: Controlled unitary for the second state.
        ancilla_qubits (list): Ancilla qubits used by the controlled gates.
        target_qubits (list): Target qubits acted on by the controlled gates.
        sampler: Sampler or backend-like object used for transpilation/execution.
        using_statevector (bool): If True, use statevector simulation.
        do_simulation (bool, optional): If False, print resource information
            and skip execution.

    Returns:
        complex | None: Estimated overlap, or None for resource-only mode.
    """
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
    """Create a controlled gate for ``Uprep`` followed by ``Ui``."""
    circuit_cUi = QuantumCircuit(Ntar)
    circuit_cUi.append(Uprep, range(Ntar))
    circuit_cUi.append(Ui, range(Ntar))
    return circuit_cUi.decompose().to_gate().control(1)


def make_Circ_forNondiagH(term_types,
                          Ntar, ancilla_qubits, target_qubits, 
                          gate_cUi, gate_cUj, qcs_re, qcs_im, using_statevector):
    """Append measurement circuits for non-diagonal Hamiltonian term types.

    Args:
        term_types (list[str]): Measurement basis descriptors.
        Ntar (int): Number of target qubits.
        ancilla_qubits (list): Ancilla qubits used by the controlled gates.
        target_qubits (list): Target qubit indices.
        gate_cUi: Controlled unitary for the first state.
        gate_cUj: Controlled unitary for the second state.
        qcs_re (list): Output list receiving real-part circuits.
        qcs_im (list): Output list receiving imaginary-part circuits.
        using_statevector (bool): If False, add measurements.

    Returns:
        None: The output lists are modified in place.
    """
    
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
    """Find the prepared measurement circuit matching a Pauli string type.

    Args:
        op_string (str): Pauli label to classify.
        term_types (list[str]): Available measurement basis descriptors.

    Returns:
        int: Index of the matching circuit type.
    """
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
    """Translate X/Y support lists into a measurement-basis descriptor string.

    Args:
        Xloc (list[int]): Qubit locations with X operators.
        Yloc (list[int]): Qubit locations with Y operators.
        Bosonic (bool): If True, collapse all-X/all-Y terms to ``"XX"`` or
            ``"YY"`` descriptors.

    Returns:
        str: Descriptor consumed by QKrylov measurement-circuit builders.
    """
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
    Transpilers sometimes change the order of qubits, so we need to reorder the bitstrings according to the mapping given by `qlayout`.
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


class QuantumKrylovSolver:
    """

    Design:
      - prepare_* methods: circuit construction only
      - evaluate_* methods: execute circuits and compute matrix elements
      - run: iterative Krylov loop
      - estimate_resource: coarse resource estimate without simulation
    """

    def __init__(
        self,
        Uprep: QuantumCircuit,
        hamiltonian_op: SparsePauliOp,
        sampler,
        ancilla_qubits,
        target_qubits,
        delta_t: float = 0.01,
        max_iterations: int = 10,
        trotter_rank: int = 2,
        trotter_steps: int = 1,
        num_shot: int = 10**4,
        using_statevector: bool = False,
        do_simulation: bool = True,
        Bosonic: bool = False,
        verbose: bool = False,
        tol_eig: float = 1.0e-6,
    ):
        self.Uprep = Uprep
        self.hamiltonian_op = hamiltonian_op
        self.sampler = sampler
        self.ancilla_qubits = ancilla_qubits
        self.target_qubits = target_qubits
        self.delta_t = delta_t
        self.max_iterations = max_iterations
        self.trotter_rank = trotter_rank
        self.trotter_steps = trotter_steps
        self.num_shot = num_shot
        self.using_statevector = using_statevector
        self.do_simulation = do_simulation
        self.Bosonic = Bosonic
        self.verbose = verbose
        self.tol_eig = tol_eig

        if len(self.ancilla_qubits) == 0:
            raise ValueError(
                "ancilla_qubits = []! You may need ancilla qubits for the Quantum Krylov method."
            )
        if len(self.target_qubits) == 0:
            raise ValueError(
                "target_qubits = []! You may need target qubits for the Quantum Krylov method."
            )

        self.Ntar = len(self.target_qubits)
        self.Hamil_coeffs = self.hamiltonian_op.coeffs
        self.Hamil_paulis = self.hamiltonian_op.paulis

        # Cached results
        self.N = None
        self.H = None
        self.ws = None
        self.Unitaries = []

    def prepare_iteration_circuits(self, it):
        """Build all circuits needed at iteration it."""
        Ui = PauliEvolutionGate(
            self.hamiltonian_op,
            it * self.delta_t,
            synthesis=SuzukiTrotter(order=self.trotter_rank, reps=self.trotter_steps),
        )
        qcs, term_types, idxs_circuit = prepare_qc_for_QKrylov(
            self.hamiltonian_op,
            self.Uprep,
            Ui,
            self.Ntar,
            self.Bosonic,
            using_statevector=self.using_statevector,
            verbose=self.verbose,
        )
        gate_cUi = make_cU(self.Uprep, Ui, self.Ntar)
        return gate_cUi, qcs, term_types, idxs_circuit

    def evaluate_diag_element(self, qcs, idxs_circuit):
        """Evaluate diagonal matrix element H[it, it]."""
        if self.using_statevector:
            sim = AerSimulator(method='statevector')
            results = []
            for qc in qcs:
                qc_sv = transpile(qc, sim)
                qc_sv.save_statevector()
                job = sim.run(qc_sv)
                result = job.result()
                psi_final = result.get_statevector(qc_sv)
                results.append([psi_final.probabilities_dict(), qc_sv.layout])
        else:
            job = self.sampler.run(qcs, shots=self.num_shot)
            results = job.result()

        Hsum = 0.0
        for idx_H in range(len(self.Hamil_paulis)):
            op_string = self.Hamil_paulis[idx_H].to_label()
            idx_relevant = get_idx_to_measure(op_string)
            idx_circuit = idxs_circuit[idx_H]
            if self.using_statevector:
                res, qlayout = results[idx_circuit]
                res = reorder_based_on_layout(res, qlayout)
            else:
                res = results[idx_circuit].data.meas.get_counts()
            expval, _, _ = expec_Zstring(res, idx_relevant)
            Hsum += self.Hamil_coeffs[idx_H] * expval
        return Hsum

    def evaluate_offdiag_element(self, gate_cUi, gate_cUj, term_types, idxs_circuit):
        """Evaluate off-diagonal matrix element H[it, j]."""
        qcs_re = []
        qcs_im = []
        make_Circ_forNondiagH(
            term_types,
            self.Ntar,
            self.ancilla_qubits,
            self.target_qubits,
            gate_cUi,
            gate_cUj,
            qcs_re,
            qcs_im,
            self.using_statevector,
        )

        if self.using_statevector:
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
            job = self.sampler.run(qcs_re, shots=self.num_shot)
            results_Re = job.result()
            job = self.sampler.run(qcs_im, shots=self.num_shot)
            results_Im = job.result()

        Re_H_ij = Im_H_ij = 0.0
        for idx_H in range(len(self.Hamil_paulis)):
            op_string = self.Hamil_paulis[idx_H].to_label()
            idx_relevant = get_idx_to_measure(op_string)
            idx_circuit = idxs_circuit[idx_H]
            if self.using_statevector:
                res_Re, layout_Re = results_Re[idx_circuit]
                res_Re = reorder_based_on_layout(res_Re, layout_Re)
                res_Im, layout_Im = results_Im[idx_circuit]
                res_Im = reorder_based_on_layout(res_Im, layout_Im)
            else:
                res_Re = results_Re[idx_circuit].data.meas.get_counts()
                res_Im = results_Im[idx_circuit].data.meas.get_counts()

            _, p0, p1 = expec_Zstring(
                res_Re,
                idx_relevant,
                target_qubits=range(len(self.target_qubits)),
                ancilla_qubit=0,
            )
            Re_H_ij += self.Hamil_coeffs[idx_H] * (p0 - p1)

            _, p0, p1 = expec_Zstring(
                res_Im,
                idx_relevant,
                target_qubits=range(len(self.target_qubits)),
                ancilla_qubit=0,
            )
            Im_H_ij += self.Hamil_coeffs[idx_H] * (p0 - p1)

        return Re_H_ij + 1j * Im_H_ij

    def solve_projected_eigenproblem(self, Nsub, Hsub):
        """Solve projected generalized eigenvalue problem for current Krylov subspace."""
        lam, v = scipy.linalg.eigh(Nsub)
        cols = [i for i in range(len(lam)) if lam[i] >= self.tol_eig]
        r = len(cols)
        if r == 0:
            raise ValueError(
                f"All overlap eigenvalues are below tol_eig={self.tol_eig:.2e}. "
                "Try reducing tol_eig or increasing shot count."
            )

        Ur = v[:, cols]
        sq_Sigma_inv = np.diag(lam[cols] ** (-0.5))
        X = Ur @ sq_Sigma_inv @ Ur.conj().T
        tildeH = X @ Hsub @ X.conj().T
        w, _ = scipy.linalg.eigh(tildeH)
        return lam, r, w[-r:]

    def run(self):
        """Execute Quantum Krylov iterations and return (Hsub, Nsub, ws)."""
        print("num of Hamil term: ", len(self.Hamil_paulis))
        if not self.do_simulation:
            return None

        self.N = np.zeros((self.max_iterations, self.max_iterations), dtype=np.complex128)
        self.H = np.zeros((self.max_iterations, self.max_iterations), dtype=np.complex128)
        self.ws = []
        self.Unitaries = []

        for it in tqdm(range(self.max_iterations)):
            print("iteration: ", it)
            gate_cUi, qcs, term_types, idxs_circuit = self.prepare_iteration_circuits(it)

            self.Unitaries.append(gate_cUi)
            self.N[it, it] = 1.0

            for j in range(it - 1, -1, -1):
                gate_cUj = self.Unitaries[j]
                U_ij = measure_overlap(
                    self.num_shot,
                    self.Ntar,
                    gate_cUi,
                    gate_cUj,
                    self.ancilla_qubits,
                    self.target_qubits,
                    self.sampler,
                    self.using_statevector,
                    self.do_simulation,
                )
                self.N[it, j] = U_ij
                self.N[j, it] = np.conj(U_ij)

            self.H[it, it] = self.evaluate_diag_element(qcs, idxs_circuit)
            if self.verbose:
                print(f"H[diag={it}] = {self.H[it, it]}")

            for j in range(it - 1, -1, -1):
                gate_cUj = self.Unitaries[j]
                Hij = self.evaluate_offdiag_element(gate_cUi, gate_cUj, term_types, idxs_circuit)
                self.H[it, j] = Hij
                self.H[j, it] = np.conj(Hij)
                print(f"H[off-diag={it},{j}] = {Hij}")

            Nsub = self.N[: it + 1, : it + 1]
            Hsub = self.H[: it + 1, : it + 1]
            lam, r, w_r = self.solve_projected_eigenproblem(Nsub, Hsub)

            self.ws.append(w_r)
            print("eigs of N: ", lam, "cond", np.linalg.cond(Nsub), "r:", r)
            print(f"w: {w_r}")
            print("")

        return self.H[: it + 1, : it + 1], self.N[: it + 1, : it + 1], self.ws

    def estimate_resource(self, tol=1.0e-10, model="ross-selinger", 
                          reduction_factor=1,
                          verbose=False):
        """
        Coarse resource estimate without running simulation.
        """
        if self.trotter_rank > 2:
            print(
                "Warning: T-count estimation assumes Suzuki-Trotter rank 1 or 2. "
                f"But got trotter_rank = {self.trotter_rank}."
            )

        num_terms = len(self.Hamil_paulis)
        term_groups = 3 if self.Bosonic else num_terms
        if reduction_factor > 1:
            term_groups = term_groups // reduction_factor
        print(f"term_groups: {term_groups} (reduction_factor: {reduction_factor})")
        Niter = self.max_iterations

        # For diaonal terms, we first apply U and measure them, which costs T_U. 
        # Since we are counting T-count in the unit of 'unit' time-evolution, we have to take a sum over \sum_{i=1}^{Niter} i = Niter*(Niter+1)/2, which gives the factor of Niter^2.
        u_uses = (Niter * (Niter + 1)) // 2
        T_U = 2 * max(num_terms - 1, 0) * self.trotter_steps * self.trotter_rank
        # For off-diagonal terms, we have to apply controlled-U_i and controlled-U_j.
        controlled_u_uses = Niter**3 - Niter
        T_epsilon = t_count(tol, model=model)
        T_cU = 2 * max(num_terms - 1, 0) * self.trotter_steps * self.trotter_rank
        Tcount = self.num_shot * term_groups * (u_uses * T_U + controlled_u_uses * T_cU) * T_epsilon

        if Tcount > 0:
            print(f"T-count (rough): {Tcount} (log10={np.log10(Tcount):.1f})")
        else:
            print("T-count (rough): 0")


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
    verbose=False,
    tol_eig=1.e-6
):
    """Backward-compatible wrapper around QuantumKrylovSolver."""
    solver = QuantumKrylovSolver(
        Uprep=Uprep,
        hamiltonian_op=hamiltonian_op,
        sampler=sampler,
        ancilla_qubits=ancilla_qubits,
        target_qubits=target_qubits,
        delta_t=delta_t,
        max_iterations=max_iterations,
        trotter_rank=trotter_rank,
        trotter_steps=trotter_steps,
        num_shot=num_shot,
        using_statevector=using_statevector,
        do_simulation=do_simulation,
        Bosonic=Bosonic,
        verbose=verbose,
        tol_eig=tol_eig,
    )
    return solver.run()


def lambda_plot(lam, Ens):
    """Save a complex-plane plot of selected ODMD eigenvalues.

    Args:
        lam (Iterable[complex]): Eigenvalues to plot.
        Ens (Iterable[float]): Energies used in legend labels.
    """
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


class ODMD:
    """Online dynamic mode decomposition solver for quantum time snapshots."""

    def __init__(
        self,
        Uprep: QuantumCircuit,
        HamiltonianOps: SparsePauliOp,
        delta_t: float,
        max_iterations: int,
        trotter_rank: int,
        trotter_steps: int,
        sampler,
        ancilla_qubits,
        target_qubits,
        num_shot: int = 1,
        using_statevector: bool = True,
        dim_Hankel: int = 8,
        tol_SVD: float = 1.0e-8,
        verbose: bool = False,
        plot_lambda: bool = False,
        tol_lambda: float = 1.0e-2
    ):
        """Initialize ODMD simulation and post-processing parameters.

        Args:
            Uprep (QuantumCircuit): State-preparation circuit.
            HamiltonianOps (SparsePauliOp): Hamiltonian used for time evolution.
            delta_t (float): Snapshot time spacing.
            max_iterations (int): Number of time snapshots.
            trotter_rank (int): Suzuki-Trotter order/rank.
            trotter_steps (int): Number of Trotter repetitions.
            sampler: Sampler used for sampled overlap measurements.
            ancilla_qubits: Ancilla qubits used in overlap measurements.
            target_qubits: Target qubits evolved by the Hamiltonian.
            num_shot (int, optional): Number of shots for sampled execution.
            using_statevector (bool, optional): If True, use statevector
                overlap simulation.
            dim_Hankel (int, optional): Hankel matrix row dimension.
            tol_SVD (float, optional): Singular-value cutoff.
            verbose (bool, optional): If True, print intermediate arrays.
            plot_lambda (bool, optional): If True, save a lambda plot.
            tol_lambda (float, optional): Initial tolerance for selecting
                eigenvalues close to the unit circle.
        """
        self.Uprep = Uprep
        self.HamiltonianOps = HamiltonianOps
        self.delta_t = delta_t
        self.max_iterations = max_iterations
        self.trotter_rank = trotter_rank
        self.trotter_steps = trotter_steps
        self.sampler = sampler
        self.ancilla_qubits = ancilla_qubits
        self.target_qubits = target_qubits
        self.num_shot = num_shot
        self.using_statevector = using_statevector
        self.dim_Hankel = dim_Hankel
        self.tol_SVD = tol_SVD
        self.verbose = verbose
        self.plot_lambda = plot_lambda
        self.tol_lambda = tol_lambda

        # Cached results (set by run)
        self.snapshots = None
        self.X = None
        self.Y = None
        self.U = None
        self.Sigma = None
        self.Vh = None
        self.trank = None
        self.A = None
        self.fit_error = None
        self.lams = None
        self.evecs = None
        self.selected_indices = None
        self.selected_lams = None
        self.energies = None
        self.tol_lambda_used = None

    def run(self):
        """Run ODMD snapshot generation, fitting, and energy extraction.

        Returns:
            tuple[list[float], np.ndarray]: Extracted energies and selected
            unit-circle eigenvalues.
        """
        Ntar = len(self.target_qubits)

        cU0 = self.Uprep.to_gate().control(1)

        snapshots = np.zeros(self.max_iterations, dtype=np.complex128)
        snapshots[0] = 1.0
        for it in tqdm(range(1, self.max_iterations)):
            Uj = PauliEvolutionGate(
                self.HamiltonianOps,
                it * self.delta_t,
                synthesis=SuzukiTrotter(order=self.trotter_rank, reps=self.trotter_steps),
            )
            gate_cUj = make_cU(self.Uprep, Uj, Ntar)
            overlap = measure_overlap(
                self.num_shot,
                Ntar,
                cU0,
                gate_cUj,
                self.ancilla_qubits,
                self.target_qubits,
                self.sampler,
                self.using_statevector,
            )
            print(f"overlap @{it:3d}: {overlap}")
            snapshots[it] = overlap
        print(
            f"Max iteration: {self.max_iterations:5d} trotter_steps: {self.trotter_steps:5d} delta_t: {self.delta_t:12.8f}",
            f" tol for lambda = {self.tol_lambda:8.2e}",
        )

        if self.verbose:
            print("snapshots of <U0|Uj>:", snapshots)

        dim_Hankel = self.dim_Hankel
        if dim_Hankel >= self.max_iterations:
            dim_Hankel = self.max_iterations // 2 + 1
            print(f"dim_Hankel is set to {dim_Hankel} since the original value is too large for the number of snapshots.")

        X, Y = self.construct_X_and_Y(snapshots, dim_Hankel)

        A = self.construct_A_from_XY(X, Y)

        fit_error = np.linalg.norm(A @ X - Y)
        print("Check |AX - Y|", fit_error)

        # Eigen values of A would be exp(-iE_j dt)
        lams, v = np.linalg.eig(A)

        tol_lambda_used = self.tol_lambda
        selected_indices = []
        while len(selected_indices) == 0:
            selected_indices = [
                i for i in range(len(lams)) if np.abs(np.abs(lams[i]) - 1) < tol_lambda_used
            ]
            if len(selected_indices) == 0:
                print(f"No eigenvalue found within |lambda|=1 +/- {tol_lambda_used:8.2e}.")
                print("Increasing the tolerance by a factor of 10.")
                tol_lambda_used *= 10

        selected_lams = lams[selected_indices]
        arg_lam = np.angle(selected_lams)
        energies = list(-(arg_lam) / self.delta_t)

        # Cache results for post-analysis
        self.snapshots = snapshots
        self.X = X
        self.Y = Y
        self.A = A
        self.fit_error = fit_error
        self.lams = lams
        self.evecs = v
        self.selected_indices = selected_indices
        self.selected_lams = selected_lams
        self.energies = energies
        self.tol_lambda_used = tol_lambda_used
        self.dim_Hankel = dim_Hankel

        return energies, selected_lams
    
    def construct_X_and_Y(self, snapshots, dim_Hankel):
        """Construct shifted Hankel matrices from a snapshot sequence.

        Args:
            snapshots (np.ndarray): Complex time-snapshot sequence.
            dim_Hankel (int): Number of Hankel rows.

        Returns:
            tuple[np.ndarray, np.ndarray]: Shifted Hankel matrices ``X`` and
            ``Y``.
        """
        N = len(snapshots)
        X = np.zeros((dim_Hankel, N - dim_Hankel), dtype=np.complex128)
        Y = np.zeros((dim_Hankel, N - dim_Hankel), dtype=np.complex128)
        for j in range(N - dim_Hankel):
            for i in range(dim_Hankel):
                idx = j + i
                X[i, j] = snapshots[idx]
                Y[i, j] = snapshots[idx + 1]
        return X, Y

    
    def construct_A_from_XY(self, X, Y):            
        """Fit the reduced linear propagator from Hankel matrices.

        Args:
            X (np.ndarray): Source Hankel matrix.
            Y (np.ndarray): Shifted target Hankel matrix.

        Returns:
            np.ndarray: Least-squares propagator ``A = Y X^+`` after SVD
            truncation.
        """
        # SVD of X
        U, Sigma, Vh = np.linalg.svd(X, full_matrices=False)
        if self.verbose:
            print("Sigma", Sigma)

        trank = np.sum(Sigma > self.tol_SVD)
        Ur = U[:, :trank]
        Sigmar = np.diag(Sigma[:trank])
        Vhr = Vh[:trank, :]

        # A = Y X^+ = Y (V Sigma^-1 Udag)
        A = Y @ Vhr.T.conj() @ np.linalg.inv(Sigmar) @ Ur.T.conj()
        return A 

    
    def get_results(self):
        """Return cached ODMD results from the most recent :meth:`run` call."""
        return {
            "energies": self.energies,
            "selected_lams": self.selected_lams,
            "selected_indices": self.selected_indices,
            "all_lams": self.lams,
            "fit_error": self.fit_error,
            "tol_lambda_used": self.tol_lambda_used,
            "trank": self.trank,
            "dim_Hankel": self.dim_Hankel,
        }
    
    def estimate_resource(self, tol=1.e-10, model="ross-selinger", verbose=False):
        """Estimate T-count for the ODMD overlap measurements.

        Args:
            tol (float, optional): Rotation synthesis tolerance.
            model (str, optional): Rotation synthesis estimate model.
            verbose (bool, optional): If True, print intermediate estimates.

        Returns:
            float: Estimated T-count.
        """
        print(f"Estimating resource for ODMD with Niter={self.max_iterations}, eps={tol:.1e}: ")
        if self.trotter_rank > 2:
            print(f"Warning: The T-count estimation is based on the assumption of using Suzuki-Trotter decomposition with rank 1 or 2. But got trotter_rank = {self.trotter_rank}. The estimation may not be accurate in this case.")
        T_epsilon = t_count(tol, model=model)
        T_cU = 2 * (len(self.HamiltonianOps.paulis)-1) * self.trotter_steps * self.trotter_rank  # very rough estimation for the T-count of controlled-U operation
        if verbose:
            print(f"T_cU: {T_cU}, T_epsilon: {T_epsilon}")
        Tcount = self.num_shot * T_cU * T_epsilon * self.max_iterations * (self.max_iterations + 1) 
        print(f"T-count formula: Nshot * T_cU * T_epsilon * max_iterations * (max_iterations + 1) = {Tcount} log10: {np.log10(Tcount):.1f}")
