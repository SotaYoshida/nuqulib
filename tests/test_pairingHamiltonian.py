import numpy as np
import pytest
from qiskit_aer import AerSimulator
from qiskit.circuit.library import PauliEvolutionGate, phase_estimation, UnitaryGate
from nuqulib import *
from qiskit.quantum_info import Operator
from scipy.linalg import expm


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

    sim = AerSimulator(method='statevector')
    qc_sv = transpile(qc_Htest, sim)
    qc_sv.save_statevector()
    job = sim.run(qc_sv)
    result = job.result()
    state_vector = result.get_statevector(qc_sv)

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

    # For check the ansatz
    qc_emeas = QuantumCircuit(Nq)
    qc_emeas.append(Uprep.to_gate(), range(Nq))
    sim = AerSimulator(method='statevector')
    qc_sv = transpile(qc_emeas, sim)
    qc_sv.save_statevector()
    job = sim.run(qc_sv)
    result = job.result()
    state_vector = result.get_statevector(qc_sv)
    E_meas = np.real(state_vector.expectation_value(hamiltonian_op))
    print(f"E_meas (before QPE): {E_meas} Egs_exact: {Egs_exact} diff. {E_meas - Egs_exact}")
    assert abs(E_meas - Egs_exact) < 1e-5, f"Expected energy {Egs_exact} but got {E_meas} with difference {abs(E_meas - Egs_exact)}"

    Na = 6
    dt = - 1.0 # dt < 0 because Egs > 0
    buffer = 10
    print("max(|E|):",  2 *np.pi / -dt)

    Umat = expm(-1j * hamiltonian_op.to_matrix() * dt)
    unitU = UnitaryGate(Operator(Umat))
    pe = phase_estimation(Na, unitU)

    anc = QuantumRegister(Na, 'ancilla')
    tgt = QuantumRegister(Nq, 'target')
    qc_QPE = QuantumCircuit(anc, tgt)
    qc_QPE.append(Uprep.to_gate(), tgt[:])
    qc_QPE.append(pe.to_gate(), anc[:] + tgt[:])

    qc_sv = qc_QPE.remove_final_measurements(inplace=False)
    qc_sv.save_statevector()

    # Run Aer statevector simulator
    sim = AerSimulator(method='statevector')
    tqc = transpile(qc_sv, sim)
    job = sim.run(tqc)
    result = job.result()
    psi_final = result.get_statevector(tqc)

    # Get probabilities on ancilla register only
    #probs = psi_final.probabilities_dict(range(Na+Nq))
    probs = psi_final.probabilities_dict(range(Na))
    list_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    for bitstring_whole, p in list_probs:
        if p < 1e-3:
            continue
        print(f"bitstring whole: {bitstring_whole} prob.: {p:.6f}")

    summarized_probs = {} 
    for bitstring, p in list_probs:
        #bitstring = bitstring[:Na]
        bitstring = bitstring
        if bitstring in summarized_probs:
            summarized_probs[bitstring] += p
        else:
            summarized_probs[bitstring] = p
    list_probs = sorted(summarized_probs.items(), key=lambda x: x[1], reverse=True)
    for bitstring, p in list_probs:
        energy = int(bitstring[::-1], 2)/(2**Na) * 2 * np.pi / -dt
        if p < 1e-3:
            continue
        print(f"bitstring: {bitstring} energy: {energy:.6f}  prob.: {p:.6f}")

    bitstring_most_prob = list_probs[0][0]
    E_estimated = int(bitstring_most_prob[::-1], 2)/(2**Na) * 2 * np.pi / -dt

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
        max_iterations=6, trotter_steps=5,
        using_statevector=True)
    
    Eestimated = np.min(Ens[-1])
    assert abs(Egs - Eestimated) < 1.e-3, f"Expected energy {Egs} but got {Eestimated} with difference {abs(Egs - Eestimated)}"
