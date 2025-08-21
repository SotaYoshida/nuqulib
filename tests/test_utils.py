import pytest
from qiskit.quantum_info import SparsePauliOp
from nuqulib import *

def test_utils( ):
    paulis = ["IIXX", "IYYX"]
    coeffs = [ 1.234, 5.678]

    # Make PennyLane operators
    coeffs_pl, obs_pl = read_QiskitPauli(paulis, coeffs)
    print("coeffs_pl", coeffs_pl)
    print("obs_pl", obs_pl)

    # Make Qiskit operators
    Qiskit_op = SparsePauliOp.from_list(zip(paulis, coeffs))
    print("Qiskit_ops:", Qiskit_op)

    # Then, transformed into pennylane one again
    coeffs_pl2, obs_pl2 = transform_qiskitOps_to_pennylane(Qiskit_op)
    print("coeffs_pl2", coeffs_pl2)
    print("obs_pl2", obs_pl2)

   # Check if the transformation is consistent
    assert list(map(np.real, coeffs_pl)) == list(map(np.real, coeffs_pl2))
    assert obs_pl == obs_pl2

