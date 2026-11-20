import pytest
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_aer.primitives import SamplerV2
from nuqulib import expec_Zstring, pair_ansatz_qiskit, circuit_XXYY

def test_meas_Paulis( ):
    """Test measurement of Pauli operators.
    H = X_0 X_1 and |psi> = a |11> + b|01> + c|10> + d|00>
    """
    hamiltonian_op = SparsePauliOp("XX")

    # Prepare the state
    qc_state = QuantumCircuit(2)
    qc_state.ry(np.random.randn(), 0)
    qc_state.ry(np.random.randn(), 1)

    # Statevector
    sim = AerSimulator(method='statevector')
    qc_sv = transpile(qc_state, sim)
    qc_sv.save_statevector()
    job = sim.run(qc_sv)
    result = job.result()
    state_vector = result.get_statevector(qc_sv)
    
    a, b, c, d = state_vector.data
    print(f"a = {a}, b = {b}, c = {c}, d = {d}")
    E_sv = np.conj(a)*d + np.conj(c)*b + np.conj(b)*c + np.conj(d)*a
    print("From statevector, a^*d + c^*b + b^*c + d^*a  =", E_sv)

    # Measurement of Pauli operator X0X1 with circuit
    qc = qc_state.copy()

    ## Estimator
    estimator = StatevectorEstimator()
    job = estimator.run([(qc, hamiltonian_op,)])
    results = job.result()
    E_meas = results[0].data.evs
    print("From statevector estimator:", E_meas)

    ## Sampler
    qc = qc_state.copy()
    qc.h(0)
    qc.h(1)
    qc.measure_all()

    num_shot = 10**5
    sampler = SamplerV2()
    job = sampler.run([qc], shots=num_shot)
    result = job.result()
    counts = result[0].data.meas.get_counts()
    E_shot = expec_Zstring(counts, [0, 1])[0]
    assert abs(E_sv-E_meas) < 1.e-10
    assert abs(E_sv - E_shot) < 10/np.sqrt(num_shot), "Statevector and shot results do not match"


@pytest.mark.parametrize("using_statevector", [True, False])
@pytest.mark.parametrize(
    ("basis", "labels"),
    [
        ("X_0,", ["IX", "ZX"]),
        ("Y_0,", ["IY", "ZY"]),
        ("X_0,Y_1,", ["YX"]),
        ("XX", ["IX", "XI", "XX"]),
        ("YY", ["IY", "YI", "YY"]),
        ("IZ", ["II", "IZ", "ZI", "ZZ"]),
    ],
)
def test_qkrylov_offdiagonal_measurement_bases(basis, labels, using_statevector):
    """Recover complex transition amplitudes from each shared measurement basis."""
    from qiskit.quantum_info import Pauli, Statevector
    from nuqulib.circuits import get_idx_to_measure
    from nuqulib.quantum_algorithms import make_cU, make_Circ_forNondiagH

    prep = QuantumCircuit(2)
    prep.ry(0.23, 0)
    prep.p(-0.41, 0)
    ui = QuantumCircuit(2)
    ui.ry(0.47, 0)
    ui.rx(-0.91, 1)
    ui.cx(0, 1)
    uj = QuantumCircuit(2)
    uj.rx(0.63, 0)
    uj.ry(1.14, 1)
    uj.cx(1, 0)
    uj.p(0.29, 0)
    initial = Statevector.from_instruction(prep)
    psi_i, psi_j = initial.evolve(ui), initial.evolve(uj)

    circuits_re, circuits_im = [], []
    make_Circ_forNondiagH(
        [basis], 2, [0], [1, 2],
        make_cU(prep, ui, 2), make_cU(prep, uj, 2),
        circuits_re, circuits_im, using_statevector,
    )
    probabilities = []
    for circuit in [circuits_re[0], circuits_im[0]]:
        if not using_statevector:
            circuit = circuit.remove_final_measurements(inplace=False)
        probabilities.append(Statevector.from_instruction(circuit).probabilities_dict())

    for label in labels:
        expected = np.vdot(psi_i.data, Pauli(label).to_matrix() @ psi_j.data)
        components = []
        for distribution in probabilities:
            _, p0, p1 = expec_Zstring(
                distribution, get_idx_to_measure(label),
                target_qubits=[0, 1], ancilla_qubit=0,
            )
            components.append(p0 - p1)
        measured = components[0] + 1j * components[1]
        assert measured == pytest.approx(expected, abs=1e-12)
