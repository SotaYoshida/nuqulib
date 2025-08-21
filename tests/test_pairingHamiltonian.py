import numpy as np
import pytest
import pennylane as qml
from qiskit_aer.primitives import SamplerV2
from qiskit.quantum_info import Statevector
from nuqulib import *

def test_pairingHamiltonian():
    Norb = 4
    Nocc = 2
    gval = 0.33  

    Hamil = PairingHamiltonian(Norb, Nocc, gval)
    evals = np.linalg.eigvalsh(Hamil.Hmat)
    Egs_exact = evals[0]

    print("basis:", Hamil.basis)
    print([tuple_to_bitstring(tup, Norb) for tup in Hamil.basis])
    print("eps: ", Hamil.epsilon)
    print("Hmat: ", Hamil.Hmat)
    print("evals: ", evals)
    print("Egs_exact: ", Egs_exact)

    assert abs(Egs_exact-1.18985) < 1e-5

def test_ansatz_pairing():
    Nq = 4
    Nocc = 2
    Hamil = PairingHamiltonian(Nq, Nocc, 0.33)
    hamiltonian_op = Hamil.encoding()
    evals = np.linalg.eigvalsh(Hamil.Hmat)
    Egs_exact = evals[0]

    # Create a dummy parameters array
    params = np.zeros(100, dtype=float)

    # Create a pairing ansatz circuit and optimize it via Nakanishi-Fujii-Todo (NFT) method
    method_ansatz = "HF+Givens"
    qc, where_is_G_or_cG1 = pair_ansatz_qiskit(params, Nq, Nocc, method=method_ansatz,
                            return_Gdict=True)
    method_measure = "statevector"
    it_max = 10
    ngate = len(where_is_G_or_cG1.keys()) 
    params_NFT, Emin_NFT = optimize_params_with_NFT(it_max, hamiltonian_op, params, 
                                                    Nq, Nocc, ngate, where_is_G_or_cG1,
                                                    method_ansatz, method_measure)
    assert abs(Emin_NFT - Egs_exact) < 1e-5

    # Hadamard-test
    qc = pair_ansatz_qiskit(params_NFT, Nq, Nocc, method=method_ansatz)
    dt = 0.1
    trotter_steps = 15
    qc_Htest = circuit_HadamardTest(Nq, qc, hamiltonian_op, dt, trotter_steps, using_statevector=True)

    state_vector = Statevector.from_instruction(qc_Htest)
    sv_arr = np.array(state_vector)

    p0 = p1 = 0
    for k in range(2**(1+Nq)):
        ancilla = k >> Nq
        if ancilla == 0:
            p0 += sv_arr[k] * np.conj(sv_arr[k])
        else:
            p1 += sv_arr[k] * np.conj(sv_arr[k])
    print("p0:", p0, "p1:", p1)
    Et = p0 - p1
    E = np.real( np.arccos(Et) / dt)
    print("E:", np.real(E), "Egs_exact: ", Egs_exact, "diff.", E - Egs_exact)
    assert abs(E - Egs_exact) < 1e-3, f"Expected energy {Egs_exact} but got {E} with difference {abs(E - Egs_exact)}"

def test_QPE():
    Nq = 4
    Nocc = 2
    Hamil = PairingHamiltonian(Nq, Nocc, 0.33)
    hamiltonian_op = Hamil.encoding()
    evals = np.linalg.eigvalsh(Hamil.Hmat)
    Egs_exact = evals[0]

    # Create a dummy parameters array
    params = np.zeros(100, dtype=float)

    # Create a pairing ansatz circuit and optimize it via Nakanishi-Fujii-Todo (NFT) method
    method_ansatz = "HF+Givens"
    qc, where_is_G_or_cG1 = pair_ansatz_qiskit(params, Nq, Nocc, method=method_ansatz,
                            return_Gdict=True)
    method_measure = "statevector"
    it_max = 10
    ngate = len(where_is_G_or_cG1.keys()) 
    params_NFT, Emin_NFT = optimize_params_with_NFT(it_max, hamiltonian_op, params, 
                                                    Nq, Nocc, ngate, where_is_G_or_cG1,
                                                    method_ansatz, method_measure)

    Uprep = pair_ansatz_qiskit(params_NFT, Nq, Nocc, method=method_ansatz)

    Na = 6
    dt = - 1.0 # dt < 0 because Egs > 0
    trotter_steps = 15
    buffer = 10
    print("max(|E|):", (2**Na-1) * 2 *np.pi / -dt)
    qc_QPE = circuit_QPE(Na, Nq, Uprep, hamiltonian_op, dt, trotter_steps, measure=True)
    qc_QPE = qc_QPE.decompose(reps=5)

    n_shot = 128
    sampler = SamplerV2()
    job = sampler.run([qc_QPE], shots=n_shot)
    result = job.result()
    print(f"QPE result: {result[0].data.c}")
    counts = result[0].data.c.get_counts()

    float_from_bitstr = lambda bitstr: int(bitstr, 2)/(2**Na) * 2 * np.pi / -dt

    key_most_freq = list(counts.keys())[int(np.argmax(list(counts.values())))]
    print("Most frequent key:", key_most_freq)
    E_estimated = float_from_bitstr(key_most_freq)
    print(f"E_estimated(QPE): {E_estimated:.8f} Egs_exact: {Egs_exact:.8f} diff. {E_estimated - Egs_exact:.2e}")
    assert abs(E_estimated - Egs_exact) < buffer * 2*np.pi/(2**Na), f"Expected energy {Egs_exact} but got {E_estimated} with difference {abs(E_estimated - Egs_exact)}"

def test_QKrylov():
    Nq = 4
    Nocc = 2
    Hamil = PairingHamiltonian(Nq, Nocc, 0.33)
    hamiltonian_op = Hamil.encoding()
    evals = np.linalg.eigvalsh(Hamil.Hmat)
    Egs = evals[0]

    Uprep = pair_ansatz_qiskit([ ], Nq, Nocc, method="HF")
    ancilla_qubits=[0]
    target_qubits=list(range(1,Nq+1))
    sampler = backend = None

    Hmat, Nmat, Ens = QuantumKrylov(
        Uprep, hamiltonian_op, sampler, backend,
        ancilla_qubits, target_qubits, delta_t=1.0,
        max_iterations=6, trotter_steps=10,
        using_statevector=True)
    
    Eestimated = np.min(Ens[-1])
    assert abs(Egs - Eestimated) < 1.e-3, f"Expected energy {Egs} but got {Eestimated} with difference {abs(Egs - Eestimated)}"
