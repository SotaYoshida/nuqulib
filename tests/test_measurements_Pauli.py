import pytest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info import Statevector
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
    state_vector = Statevector.from_instruction(qc_state)
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


