from pathlib import Path

import numpy as np
import pytest
from types import SimpleNamespace
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Operator, Statevector

from nuqulib import (
    AngularMomentumProjector,
    Diagonalize_Hamiltonian,
    get_Hamiltonian,
    nucl_ansatz,
)
from nuqulib.angular_momentum_projection import (
    _build_jx_matrix,
    _build_kjx_circuit,
    _decompose_nearest_neighbor_givens,
    _diagonalize_jx,
)

INTERACTION_DIR = Path(__file__).parent / "interaction_file"

@pytest.mark.parametrize("j2val", [1, 3, 5])
def test_kjx_basis_transformation(j2val):
    """Verify Eq. (9) and the circuit representation of KJx."""
    Jx = _build_jx_matrix(j2val)
    KJx_dagger, evals = _diagonalize_jx(Jx)
    KJx = KJx_dagger.T
    Jz = np.diag(evals)

    np.testing.assert_allclose(
        KJx_dagger @ Jx @ KJx,
        Jz,
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        KJx @ Jz @ KJx_dagger,
        Jx,
        atol=1e-12,
        rtol=0.0,
    )

    rotations, residual = _decompose_nearest_neighbor_givens(KJx)
    np.testing.assert_allclose(
        residual,
        np.eye(KJx.shape[0]),
        atol=1e-12,
        rtol=0.0,
    )

    kjx_gate = _build_kjx_circuit(
        num_targets=KJx.shape[0],
        blocks_list=[list(range(KJx.shape[0]))],
        params=[theta for _, theta in rotations],
    )
    full_unitary = Operator(kjx_gate).data
    one_particle_indices = [1 << qubit for qubit in range(KJx.shape[0])]
    circuit_KJx = full_unitary[np.ix_(one_particle_indices, one_particle_indices)]
    np.testing.assert_allclose(circuit_KJx, KJx, atol=1e-12, rtol=0.0)


def test_projector_separates_proton_and_neutron_blocks():
    """Do not mix orbitals that share (n, l, j) but have different tz."""
    msps = [
        SimpleNamespace(n=0, l=0, j=1, jz=-1, tz=-1),
        SimpleNamespace(n=0, l=0, j=1, jz=1, tz=-1),
        SimpleNamespace(n=0, l=0, j=1, jz=-1, tz=1),
        SimpleNamespace(n=0, l=0, j=1, jz=1, tz=1),
    ]
    test_hamiltonian = SimpleNamespace(msps=msps)
    test_projector = AngularMomentumProjector(
        test_hamiltonian,
        AerSimulator(method="statevector"),
    )

    assert test_projector.j_blocks == [[0, 1], [2, 3]]
    assert test_projector.j2_values == [1, 1]


def test_project_j_zero_records_projection_history():
    """Record every Jz/Jx stage and its cumulative success probability."""
    msps = [
        SimpleNamespace(n=0, l=0, j=1, jz=-1, tz=1),
        SimpleNamespace(n=0, l=0, j=1, jz=1, tz=1),
    ]
    test_hamiltonian = SimpleNamespace(msps=msps)
    test_projector = AngularMomentumProjector(
        test_hamiltonian,
        AerSimulator(method="statevector"),
    )
    circuit = QuantumCircuit(3)
    circuit.x(1)

    result = test_projector.project_j_zero(
        circuit=circuit,
        ancilla_qubit=0,
        target_qubits=[1, 2],
        angles=[0.2],
        num_steps=2,
    )

    assert [record.label for record in result.history] == [
        "1:Jz",
        "1:Jx",
        "2:Jz",
        "2:Jx",
    ]
    conditional_probabilities = [
        record.probability for record in result.history
    ]
    cumulative_probabilities = np.cumprod(conditional_probabilities)
    np.testing.assert_allclose(
        [record.cumulative_probability for record in result.history],
        cumulative_probabilities,
    )
    assert result.probability == pytest.approx(cumulative_probabilities[-1])


def test_angle_resolved_history_matches_stage_projection():
    """Angle-resolved history preserves the final projected state and probability."""
    msps = [
        SimpleNamespace(n=0, l=0, j=1, jz=-1, tz=1),
        SimpleNamespace(n=0, l=0, j=1, jz=1, tz=1),
    ]
    test_hamiltonian = SimpleNamespace(msps=msps)
    test_projector = AngularMomentumProjector(
        test_hamiltonian,
        AerSimulator(method="statevector"),
    )
    circuit = QuantumCircuit(3)
    circuit.x(1)
    angles = [0.4, 0.2]

    stage_result = test_projector.project_j_zero(
        circuit=circuit,
        ancilla_qubit=0,
        target_qubits=[1, 2],
        angles=angles,
        num_steps=2,
    )
    angle_result = test_projector.project_j_zero(
        circuit=circuit,
        ancilla_qubit=0,
        target_qubits=[1, 2],
        angles=angles,
        num_steps=2,
        history_granularity="angle",
    )

    assert len(angle_result.history) == 2 * 2 * len(angles)
    assert [record.label for record in angle_result.history[:4]] == [
        "1:Jz[1]",
        "1:Jz[2]",
        "1:Jx[1]",
        "1:Jx[2]",
    ]
    assert angle_result.probability == pytest.approx(stage_result.probability)
    assert Statevector(angle_result.state).equiv(Statevector(stage_result.state))


def test_6He():
    sim = AerSimulator(method='statevector')

    num_projection_steps = 5
    Na = 1
    fn_snt = INTERACTION_DIR / "ckpot.snt"
    Zc = Nc = 2
    nuc = "He6"; Z = 2; N = 4
    target_parity = 1
    num_Jz_projections = 2 

    hamil, H_mapped, proton_qubits, neutron_qubits = get_Hamiltonian(fn_snt, Z, N, single_spiecies=2)
    n_qubit_p = hamil.n_qubits_p
    n_qubit_n = hamil.n_qubits_n
    Nq = n_qubit_p + n_qubit_n

    # Brute force diagonalization for comparison: `Diagonalize_Hamiltonian` in src/nuqulib/diagonalization.py
    obj_Diag = Diagonalize_Hamiltonian(H_mapped, hamil, Z, N, target_parity, Zc, Nc, calc_J2=True)
    print("<H>:", obj_Diag["evals"])
    print("<J>:", obj_Diag["Jvals"])
    obj_Diag = Diagonalize_Hamiltonian(H_mapped, hamil, Z, N, target_parity, Zc, Nc, use_basis=None, calc_J2=False)

    Ens_exact = obj_Diag["evals"]
    evecs_exact = obj_Diag["evecs"]


    # State prep.
    proton_qubits = list(range(n_qubit_p))
    neutron_qubits = list(range(n_qubit_p, Nq))
    print(f"Proton qubits: {proton_qubits}")
    print(f"Neutron qubits: {neutron_qubits}")
    qr_ancilla = range(Na)
    qr_target = range(Na, Na + Nq)

    theta = np.random.randn(Nq**2) * np.pi/4

    Uprep = nucl_ansatz(
        hamil.Hamildict, Nq, proton_qubits, neutron_qubits, Z-Zc, N-Nc,
        params = theta,
        method = "HF+Givens"
    )

    # projector
    projector = AngularMomentumProjector(
        hamiltonian=hamil,
        simulator=sim,
        num_ancillas=Na,
    )

    theta_list_jz = [np.pi / (2 ** (iter_Jz + 1)) for iter_Jz in range(num_Jz_projections)]

    ancilla_qubits = list(range(Na))
    target_qubits = list(range(Na, Na + Nq))

    qc_psi = QuantumCircuit(Na + Nq)
    qc_psi.append(Uprep, target_qubits)

    jzero_result = projector.project_j_zero(
        circuit=qc_psi,
        ancilla_qubit=ancilla_qubits[0],
        target_qubits=target_qubits,
        angles=theta_list_jz,
        num_steps=num_projection_steps,
        postselect_state=0,
        verbose=True,
    )

    target_state = jzero_result.state
    qc_psi = jzero_result.circuit
    print(f"Total post-selection probability = {jzero_result.probability:.12e}")

    print("Post-selected target state:")
    # print_state_amplitudes(target_state.data, Nq, 0, filter_0=False)
    Eof_J0state_by_hand = [-3.909812, 7.921112]
                           
    psi_post = target_state.data
    for nth_exact in range(evecs_exact.shape[1]):
        psi_exact = evecs_exact[:, nth_exact]
        overlap = np.abs(np.vdot(psi_exact, psi_post))**2
        if overlap > 1e-3:
            print(f"Overlap with exact state E = {Ens_exact[nth_exact]:.6f}: {overlap:.6f}")
            # Energy should be close to an exact eigenvalue of the Hamiltonian
            evals_exact = Ens_exact[nth_exact]
            if all(np.abs(np.array(Eof_J0state_by_hand) - evals_exact) > 1e-3):
                raise ValueError(f"Energy of post-selected state {evals_exact:.6f} is not close to any expected eigenvalue {Eof_J0state_by_hand}.")
