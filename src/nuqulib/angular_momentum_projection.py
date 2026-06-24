import numpy as np
from dataclasses import dataclass
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from .nuclear_hamiltonian import Hamiltonian
from .circuits import G_gate

__all__ = [
    "AngularMomentumProjector",
    "ProjectionRecord",
    "ProjectionResult",
    "print_state_amplitudes",
]

def _build_jx_matrix(j2val):
    """Construct the one-particle Jx matrix for a fixed-j shell-model block.

    The basis is ordered by the doubled magnetic quantum number ``2m`` from
    ``-j2val`` to ``j2val`` in steps of two. The input follows NuQuLib's
    convention of storing angular-momentum quantum numbers as integers
    doubled from their physical values.

    Parameters
    ----------
    j2val : int
        Doubled total single-particle angular momentum, ``2j``.

    Returns
    -------
    numpy.ndarray
        Real symmetric matrix representing ``Jx`` in the ``Jz`` eigenbasis.
    """
    mvals = range(-j2val, j2val+1, 2)
    dim = len(mvals)
    Jx_mat = np.zeros((dim, dim))
    for i, m1 in enumerate(mvals):
        for j, m2 in enumerate(mvals):
            if m2 == m1 + 2: # <J+>
                Jx_mat[i, j] = 0.5 * np.sqrt((j2val/2 )* (j2val/2 + 1) - (m1/2) * (m1/2 + 1))
            elif m2 == m1 - 2: # <J->
                Jx_mat[i, j] = 0.5 * np.sqrt((j2val/2 )* (j2val/2 + 1) - (m1/2) * (m1/2 - 1))

    return Jx_mat

def _diagonalize_jx(Jx, target_eigs=None):
    """Diagonalize a real symmetric Jx matrix and return the basis change.

    The returned matrix follows the notation of Eq. (9):

    ``Jx = KJx @ Jz @ KJx.T`` and
    ``Jz = KJx_dagger @ Jx @ KJx``,

    where ``KJx_dagger = KJx.T`` for the real matrices used here.

    Parameters
    ----------
    Jx : array_like
        Real symmetric matrix to diagonalize.
    target_eigs : array_like, optional
        Desired eigenvalue ordering. Each requested value is matched to the
        closest unused eigenvalue returned by ``numpy.linalg.eigh``.

    Returns
    -------
    KJx_dagger : numpy.ndarray
        Orthogonal matrix that transforms ``Jx`` to its diagonal form.
    evals : numpy.ndarray
        Eigenvalues in the selected order; these form the diagonal ``Jz``.

    Notes
    -----
    The sign of one eigenvector is adjusted when necessary so that
    ``det(KJx_dagger) = +1``. This leaves the diagonalization unchanged and
    makes the matrix representable as a product of proper Givens rotations.
    """
    evals, V = np.linalg.eigh(Jx)

    # Optional: reorder eigenvectors to match target_eigs, e.g. mvals
    if target_eigs is not None:
        target_eigs = np.asarray(target_eigs)
        order = []
        used = set()

        for val in target_eigs:
            idx = min(
                [i for i in range(len(evals)) if i not in used],
                key=lambda i: abs(evals[i] - val),
            )
            order.append(idx)
            used.add(idx)

        evals = evals[order]
        V = V[:, order]

    KJx_dagger = V.T

    # Make det(KJx_dagger)=+1 if desired.
    # Flipping one eigenvector sign does not change diagonalization.
    if np.linalg.det(KJx_dagger) < 0:
        KJx_dagger[-1, :] *= -1

    return KJx_dagger, evals


def _givens_rotation_matrix(n, i, theta):
    """Return a real nearest-neighbor Givens rotation.

    The nontrivial two-dimensional block acting on components ``i`` and
    ``i + 1`` is ``[[cos(theta), -sin(theta)],
    [sin(theta), cos(theta)]]``.

    Parameters
    ----------
    n : int
        Dimension of the full matrix.
    i : int
        Index of the first component in the adjacent pair.
    theta : float
        Rotation angle in radians.

    Returns
    -------
    numpy.ndarray
        An ``n x n`` orthogonal Givens rotation matrix.
    """
    G = np.eye(n)
    c = np.cos(theta)
    s = np.sin(theta)

    G[i, i] = c
    G[i, i + 1] = -s
    G[i + 1, i] = s
    G[i + 1, i + 1] = c

    return G


def _decompose_nearest_neighbor_givens(K, atol=1e-12):
    """Eliminate an orthogonal matrix using nearest-neighbor Givens rotations.

    The algorithm finds rotations G_r such that

        G_L ... G_2 G_1 K ≈ diagonal sign matrix.

    If the final sign matrix is ignored or absorbed into phases, then

        K ≈ G_1^{-1} G_2^{-1} ... G_L^{-1}.

    Parameters
    ----------
    K : array_like
        Real square matrix to decompose. For the angular-momentum basis
        change this is ``KJx``.
    atol : float, optional
        Elements with magnitude below this tolerance are treated as zero.

    Returns
    -------
    rotations : list of tuple[int, float]
        Pairs ``(i, theta)`` describing the elimination rotations in the
        order in which they left-multiply ``K``.
    A : numpy.ndarray
        Matrix remaining after elimination. For a proper orthogonal input it
        should be close to the identity matrix.

    Notes
    -----
    To reconstruct ``K`` as a circuit, apply the stored rotations in reverse
    order. The circuit builder performs this reversal explicitly.
    """
    A = np.array(K, dtype=float, copy=True)
    n = A.shape[0]

    rotations = []

    for col in range(n - 1):
        for row in range(n - 1, col, -1):
            i = row - 1

            a = A[i, col]
            b = A[row, col]

            if abs(b) < atol:
                continue

            theta = np.arctan2(-b, a)

            G = _givens_rotation_matrix(n, i, theta)
            A = G @ A

            rotations.append((i, theta))

    return rotations, A

def _build_kjx_circuit(num_targets, blocks_list, params, angle_scale=2.0, verbose=False):
    """Build the second-quantized circuit implementing KJx.

    Parameters
    ----------
    num_targets : int
        Number of target qubits in the orbital register.
    blocks_list : sequence of sequence of int
        Target-qubit indices grouped into fixed-``(n, l, j)`` blocks.
    params : sequence of float
        Nearest-neighbor Givens angles in elimination order, concatenated
        block by block.
    angle_scale : float, optional
        Conversion factor from the one-particle matrix rotation angle to the
        parameter accepted by ``G_gate``. The current gate convention uses 2.
    verbose : bool, optional
        Print each applied orbital rotation when ``True``.

    Returns
    -------
    qiskit.circuit.Gate
        Number-conserving gate whose one-particle representation is ``KJx``.

    Notes
    -----
    The angles are consumed in elimination order but applied in reverse order
    within each block to reconstruct the original basis-change matrix.
    """
    qc = QuantumCircuit(num_targets, name="K_Jx")
    param_idx = 0
    if verbose:
        print(f"blocks_list: {blocks_list}")

    # Reverse the elimination rotations to reconstruct KJx.
    for block_indices in blocks_list:
        n_block = len(block_indices)

        block_rotations = []
        for col in range(n_block - 1):
            for row in range(n_block - 1, col, -1):
                local_i = row - 1
                block_rotations.append((local_i, col, row, params[param_idx]))
                param_idx += 1

        for local_i, col, row, param in reversed(block_rotations):
            # G_gate rotates the one-particle subspace by half its parameter.
            theta = angle_scale * param
            g_gate = G_gate(theta)

            actual_q1 = block_indices[local_i]
            actual_q2 = block_indices[local_i + 1]
            if verbose:
                print(
                    f"Applying Givens rotation with theta={theta:.4f} "
                    f"(col={col}, row={row}) between qubits {actual_q1} and {actual_q2}"
                )
            qc.append(g_gate, [actual_q1, actual_q2])

    return qc.to_gate(label="K_Jx")

def _simulate_statevector(qc, sim):
    """Simulate a circuit and return its statevector.

    Parameters
    ----------
    qc : qiskit.QuantumCircuit
        Circuit to simulate. The input circuit is not modified.
    sim : qiskit.providers.Backend
        Statevector-capable backend, such as ``AerSimulator``.

    Returns
    -------
    qiskit.quantum_info.Statevector
        Simulated statevector of the complete circuit.
    """
    qc_sv = qc.copy()
    qc_sv.save_statevector()
    tqc = transpile(qc_sv, sim, optimization_level=0)
    result = sim.run(tqc).result()
    psi = result.get_statevector(tqc)
    return psi if isinstance(psi, Statevector) else Statevector(psi)

def _postselect_ancilla(psi, Na, target_state=0):
    """Post-select an ancilla state and rebuild the normalized target circuit.

    The ancilla qubits are assumed to occupy the least-significant ``Na``
    Qiskit qubit positions. The selected target amplitudes are normalized and
    loaded into a fresh circuit with the same ancilla/target layout.

    Parameters
    ----------
    psi : array_like or qiskit.quantum_info.Statevector
        Statevector of the ancilla-plus-target system.
    Na : int
        Number of least-significant ancilla qubits.
    target_state : int, optional
        Computational-basis ancilla outcome to retain.

    Returns
    -------
    target_sv : qiskit.quantum_info.Statevector
        Normalized post-selected state of the target register.
    prob : float
        Probability of the selected ancilla outcome.
    qc_post : qiskit.QuantumCircuit
        Fresh circuit preparing ``target_sv`` on the target qubits while the
        ancillas remain in ``|0>``.

    Raises
    ------
    ValueError
        If ``target_state`` is invalid or its probability is numerically zero.
    """
    n_ancilla_states = 2**Na
    psi_array = np.asarray(psi, dtype=complex).reshape(-1, n_ancilla_states)

    if not 0 <= target_state < n_ancilla_states:
        raise ValueError(f"target_state must be in [0, {n_ancilla_states - 1}].")

    target_sv = psi_array[:, target_state].astype(complex).copy()
    prob = float(np.vdot(target_sv, target_sv).real)

    if prob <= 1e-14:
        raise ValueError(f"Post-selection probability is too small: {prob}")

    target_sv /= np.sqrt(prob)
    Nq_local = int(np.log2(target_sv.size))

    qc_post = QuantumCircuit(Na + Nq_local)
    qc_post.initialize(target_sv, range(Na, Na + Nq_local))

    return Statevector(target_sv), prob, qc_post

def _append_jz_filter_step(qc: QuantumCircuit,
                          hamil: Hamiltonian,
                          idx_ancilla_qubit: int,
                          target_qubits: list,
                          t_val: float):
    """Append one ancilla-assisted Jz-filtering step.

    Parameters
    ----------
    qc : qiskit.QuantumCircuit
        Circuit modified in place.
    hamil : Hamiltonian
        NuQuLib Hamiltonian whose single-particle states provide doubled
        magnetic quantum numbers ``2m``.
    idx_ancilla_qubit : int
        Ancilla-qubit index.
    target_qubits : sequence of int
        Circuit qubits corresponding, in order, to ``hamil.msps``.
    t_val : float
        Filter evolution parameter.

    Notes
    -----
    With the doubled-``m`` convention and Qiskit's ``Ry`` definition, the
    applied angles implement the desired ``exp(-i t Jz tensor Ya)`` coupling.
    """
    for p, qubit in enumerate(target_qubits):
        m_p = hamil.msps[p].jz # already doubled
        theta = t_val * m_p
        qc.cry(theta, qubit, idx_ancilla_qubit)

def _apply_jz_projection(qc_psi: QuantumCircuit,
                        hamil: Hamiltonian,
                        idx_ancilla_qubit: int,
                        target_qubits: list,
                        theta_list,
                        sim,
                        Na: int,
                        postselect_state: int = 0,
                        label: str = "Jz",
                        verbose: bool = False):
    """Apply a sequence of post-selected Jz filters.

    After each filter, the ancilla is post-selected and the normalized target
    state is loaded into a fresh circuit before the next angle is applied.

    Parameters
    ----------
    qc_psi : qiskit.QuantumCircuit
        Circuit preparing the current ancilla-plus-target state.
    hamil : Hamiltonian
        NuQuLib Hamiltonian defining the orbital magnetic quantum numbers.
    idx_ancilla_qubit : int
        Ancilla used for every filtering step.
    target_qubits : sequence of int
        Target-register qubits ordered consistently with ``hamil.msps``.
    theta_list : sequence of float
        Filter parameters applied successively.
    sim : qiskit.providers.Backend
        Statevector-capable simulator.
    Na : int
        Number of least-significant ancilla qubits.
    postselect_state : int, optional
        Ancilla computational-basis outcome retained after each step.
    label : str, optional
        Label used in verbose progress messages.
    verbose : bool, optional
        Print the conditional probability at each step when ``True``.

    Returns
    -------
    target_state : qiskit.quantum_info.Statevector
        Final normalized target state.
    p_total : float
        Product of all conditional post-selection probabilities.
    qc_current : qiskit.QuantumCircuit
        Fresh circuit preparing the final post-selected target state.
    """
    qc_current = qc_psi
    p_total = 1.0
    target_state = None

    for iter_idx, theta in enumerate(theta_list):
        qc_work = qc_current.copy()
        _append_jz_filter_step(qc_work, hamil, idx_ancilla_qubit, target_qubits, theta)

        psi_after = _simulate_statevector(qc_work, sim)
        target_state, p_anc0, qc_current = _postselect_ancilla(
            psi_after,
            Na,
            target_state=postselect_state,
        )
        p_total *= p_anc0
        if verbose:
            print(
                f"  {label} projection {iter_idx + 1}/{len(theta_list)}: "
                f"theta={theta:.6f}, p(ancilla={postselect_state})={p_anc0:.12e}"
            )

    return target_state, p_total, qc_current


def _build_kjx_gate_from_blocks(Nq, nljblocks, j2list, angle_scale=2.0):
    """Construct KJx for a collection of fixed-(n, l, j) blocks.

    Parameters
    ----------
    Nq : int
        Number of target qubits.
    nljblocks : sequence of sequence of int
        Qubit indices grouped into fixed-``(n, l, j)`` blocks.
    j2list : sequence of int
        Doubled angular momenta ``2j``, one for each block.
    angle_scale : float, optional
        Conversion factor passed to the underlying Givens-gate builder.

    Returns
    -------
    qiskit.circuit.Gate
        Block-diagonal second-quantized implementation of ``KJx``.
    """
    angles = []
    for idx in range(len(nljblocks)):
        j2 = j2list[idx]
        Jx_mat = _build_jx_matrix(j2)
        KJx_dagger, _ = _diagonalize_jx(Jx_mat)
        KJx = KJx_dagger.T
        rotations, _ = _decompose_nearest_neighbor_givens(KJx)
        angles.extend(theta for i, theta in rotations)
    return _build_kjx_circuit(
        Nq, nljblocks, angles, angle_scale=angle_scale
    )


def _apply_jx_projection(qc_psi: QuantumCircuit,
                        hamil: Hamiltonian,
                        idx_ancilla_qubit: int,
                        target_qubits: list,
                        theta_list,
                        nljblocks,
                        j2list,
                        sim,
                        Na: int,
                        postselect_state: int = 0,
                        kjx_gate=None,
                        label: str = "Jx",
                        verbose: bool = False):
    """Apply Jx filtering through a rotation to the Jz eigenbasis.

    The routine applies ``KJx_dagger``, performs the same post-selected filter
    sequence used for ``Jz``, and finally applies ``KJx`` to return to the
    original basis. This realizes the similarity transformation in Eq. (9).

    Parameters
    ----------
    qc_psi : qiskit.QuantumCircuit
        Circuit preparing the current ancilla-plus-target state.
    hamil : Hamiltonian
        NuQuLib Hamiltonian defining the single-particle basis.
    idx_ancilla_qubit : int
        Ancilla used by the intermediate Jz filtering steps.
    target_qubits : sequence of int
        Target-register qubits.
    theta_list : sequence of float
        Filter parameters applied successively.
    nljblocks : sequence of sequence of int
        Qubit indices grouped into fixed-``(n, l, j)`` blocks.
    j2list : sequence of int
        Doubled angular momenta ``2j`` corresponding to ``nljblocks``.
    sim : qiskit.providers.Backend
        Statevector-capable simulator.
    Na : int
        Number of least-significant ancilla qubits.
    postselect_state : int, optional
        Ancilla outcome retained after each filter.
    kjx_gate : qiskit.circuit.Gate, optional
        Prebuilt ``KJx`` gate.
    label : str, optional
        Prefix used in progress labels.
    verbose : bool, optional
        Print the conditional Jz-filter probabilities when ``True``.

    Returns
    -------
    target_state : qiskit.quantum_info.Statevector
        Final normalized target state in the original basis.
    p_total : float
        Product of the intermediate Jz-filter post-selection probabilities.
    qc_final : qiskit.QuantumCircuit
        Fresh circuit preparing the final target state.
    """
    if kjx_gate is None:
        nljblocks, j2list = _get_j_blocks_and_j2(hamil)
        kjx_gate = _build_kjx_gate_from_blocks(
            len(target_qubits), nljblocks, j2list
        )

    qc_rotated = qc_psi.copy()
    qc_rotated.append(kjx_gate.inverse(), target_qubits)

    _, p_total, qc_rotated = _apply_jz_projection(
        qc_rotated,
        hamil,
        idx_ancilla_qubit,
        target_qubits,
        theta_list,
        sim,
        Na,
        postselect_state=postselect_state,
        label=f"{label}/Jz",
        verbose=verbose,
    )

    qc_final = qc_rotated.copy()
    qc_final.append(kjx_gate, target_qubits)

    psi_final = _simulate_statevector(qc_final, sim)
    target_state, _, qc_final = _postselect_ancilla(
        psi_final,
        Na,
        target_state=postselect_state,
    )

    return target_state, p_total, qc_final

def _get_j_blocks_and_j2(hamil):
    """Group orbitals into fixed-(n, l, j, tz) blocks.

    Parameters
    ----------
    hamil : Hamiltonian
        NuQuLib Hamiltonian containing the ordered single-particle states.

    Returns
    -------
    blocks : list of list of int
        Local target-qubit indices for each ``(n, l, j, tz)`` block.
    j2_list : list of int
        Doubled angular momentum ``2j`` associated with each block.
    """
    blocks = []
    j2_list = []
    block_index_by_key = {}
    for q, msp in enumerate(hamil.msps):
        key = (msp.n, msp.l, msp.j, msp.tz)
        if key not in block_index_by_key:
            block_index_by_key[key] = len(blocks)
            blocks.append([])
            j2_list.append(msp.j)
        blocks[block_index_by_key[key]].append(q)

    return blocks, j2_list


@dataclass
class ProjectionRecord:
    """State and success probability recorded after one projection stage.

    Attributes
    ----------
    step : int
        One-based iteration index of the alternating Jz/Jx projection.
    axis : str
        Projected angular-momentum component, either ``"Jz"`` or ``"Jx"``.
    state : qiskit.quantum_info.Statevector
        Normalized target state after this projection stage.
    probability : float
        Conditional post-selection probability for this stage.
    cumulative_probability : float
        Product of all conditional probabilities up to this stage.
    circuit : qiskit.QuantumCircuit
        Circuit preparing ``state`` on the target register.
    angle_index : int or None
        One-based index of the filter angle when angle-resolved history is
        requested. ``None`` for stage-level history.
    angle : float or None
        Filter angle used for this record. ``None`` for stage-level history.
    """

    step: int
    axis: str
    state: Statevector
    probability: float
    cumulative_probability: float
    circuit: QuantumCircuit
    angle_index: int | None = None
    angle: float | None = None

    @property
    def label(self) -> str:
        """Return a compact label suitable for tables and plots."""
        if self.angle_index is not None:
            return f"{self.step}:{self.axis}[{self.angle_index}]"
        return f"{self.step}:{self.axis}"


@dataclass
class ProjectionResult:
    """Result of a post-selected angular-momentum projection."""

    state: Statevector
    probability: float
    circuit: QuantumCircuit
    history: tuple[ProjectionRecord, ...] = ()


class AngularMomentumProjector:
    """Apply ancilla-assisted Jz, Jx, and iterative J=0 projections.

    The projector caches the orbital blocks and the second-quantized KJx
    gate associated with a NuQuLib Hamiltonian. The current statevector
    implementation assumes that the ancillas are the least-significant
    qubits and that the target register immediately follows them.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        NuQuLib Hamiltonian defining the ordered M-scheme orbitals.
    simulator : qiskit.providers.Backend
        Statevector-capable simulator.
    num_ancillas : int, optional
        Number of least-significant ancilla qubits.
    angle_scale : float, optional
        Conversion factor between one-particle Givens angles and the
        parameter accepted by ``G_gate``.
    """

    def __init__(self, hamiltonian, simulator, num_ancillas=1, angle_scale=2.0):
        if num_ancillas < 1:
            raise ValueError("num_ancillas must be positive.")

        self.hamiltonian = hamiltonian
        self.simulator = simulator
        self.num_ancillas = num_ancillas
        self.angle_scale = angle_scale
        self.num_targets = len(hamiltonian.msps)
        self.j_blocks, self.j2_values = _get_j_blocks_and_j2(hamiltonian)
        self._validate_orbital_ordering()
        self.kjx_gate = _build_kjx_gate_from_blocks(
            self.num_targets,
            self.j_blocks,
            self.j2_values,
            angle_scale=self.angle_scale,
        )

    def project_jz(self, circuit, ancilla_qubit, target_qubits, angles,
                   postselect_state=0, verbose=False):
        """Apply a sequence of post-selected Jz filters."""
        target_qubits = self._validate_register_layout(
            circuit, ancilla_qubit, target_qubits
        )
        state, probability, projected_circuit = _apply_jz_projection(
            circuit,
            self.hamiltonian,
            ancilla_qubit,
            target_qubits,
            angles,
            self.simulator,
            self.num_ancillas,
            postselect_state=postselect_state,
            label="Jz",
            verbose=verbose,
        )
        return ProjectionResult(state, probability, projected_circuit)

    def project_jx(self, circuit, ancilla_qubit, target_qubits, angles,
                   postselect_state=0, verbose=False):
        """Apply Jx projection using KJx_dagger, Jz filters, and KJx."""
        target_qubits = self._validate_register_layout(
            circuit, ancilla_qubit, target_qubits
        )
        state, probability, projected_circuit = _apply_jx_projection(
            circuit,
            self.hamiltonian,
            ancilla_qubit,
            target_qubits,
            angles,
            self.j_blocks,
            self.j2_values,
            self.simulator,
            self.num_ancillas,
            postselect_state=postselect_state,
            kjx_gate=self.kjx_gate,
            label="Jx",
            verbose=verbose,
        )
        if verbose:
            print(f"  Jx projection probability={probability:.12e}")
        return ProjectionResult(state, probability, projected_circuit)

    def project_j_zero(self, circuit, ancilla_qubit, target_qubits, angles,
                       num_steps=1, postselect_state=0, verbose=False,
                       history_granularity="stage"):
        """Iteratively apply Jz and Jx projections to approach J=0.

        The returned :class:`ProjectionResult` contains a ``history`` entry
        after every Jz and Jx stage by default. Set ``history_granularity`` to
        ``"angle"`` to apply and record each filter angle separately. The
        angle-resolved mode implements the same ordered filter product while
        exposing the evolution within each Jz and Jx stage.
        """
        if num_steps < 1:
            raise ValueError("num_steps must be positive.")
        if history_granularity not in {"stage", "angle"}:
            raise ValueError(
                "history_granularity must be either 'stage' or 'angle'."
            )

        angles = tuple(angles)
        if not angles:
            raise ValueError("angles must contain at least one filter angle.")

        current_circuit = circuit
        total_probability = 1.0
        result = None
        history = []

        for step in range(num_steps):
            if verbose:
                print(f"Projection step {step + 1}/{num_steps}")

            if history_granularity == "stage":
                result_jz = self.project_jz(
                    current_circuit,
                    ancilla_qubit,
                    target_qubits,
                    angles,
                    postselect_state=postselect_state,
                    verbose=verbose,
                )
                total_probability *= result_jz.probability
                history.append(
                    ProjectionRecord(
                        step=step + 1,
                        axis="Jz",
                        state=result_jz.state,
                        probability=result_jz.probability,
                        cumulative_probability=total_probability,
                        circuit=result_jz.circuit,
                    )
                )

                result = self.project_jx(
                    result_jz.circuit,
                    ancilla_qubit,
                    target_qubits,
                    angles,
                    postselect_state=postselect_state,
                    verbose=verbose,
                )
                total_probability *= result.probability
                current_circuit = result.circuit
                history.append(
                    ProjectionRecord(
                        step=step + 1,
                        axis="Jx",
                        state=result.state,
                        probability=result.probability,
                        cumulative_probability=total_probability,
                        circuit=result.circuit,
                    )
                )
                continue

            for angle_index, angle in enumerate(angles, start=1):
                result_jz = self.project_jz(
                    current_circuit,
                    ancilla_qubit,
                    target_qubits,
                    [angle],
                    postselect_state=postselect_state,
                    verbose=verbose,
                )
                total_probability *= result_jz.probability
                current_circuit = result_jz.circuit
                history.append(
                    ProjectionRecord(
                        step=step + 1,
                        axis="Jz",
                        state=result_jz.state,
                        probability=result_jz.probability,
                        cumulative_probability=total_probability,
                        circuit=result_jz.circuit,
                        angle_index=angle_index,
                        angle=float(angle),
                    )
                )

            for angle_index, angle in enumerate(angles, start=1):
                result = self.project_jx(
                    current_circuit,
                    ancilla_qubit,
                    target_qubits,
                    [angle],
                    postselect_state=postselect_state,
                    verbose=verbose,
                )
                total_probability *= result.probability
                current_circuit = result.circuit
                history.append(
                    ProjectionRecord(
                        step=step + 1,
                        axis="Jx",
                        state=result.state,
                        probability=result.probability,
                        cumulative_probability=total_probability,
                        circuit=result.circuit,
                        angle_index=angle_index,
                        angle=float(angle),
                    )
                )

        return ProjectionResult(
            result.state,
            total_probability,
            current_circuit,
            history=tuple(history),
        )

    def _validate_orbital_ordering(self):
        """Check that each block is ordered from m=-j to m=+j."""
        for block, j2val in zip(self.j_blocks, self.j2_values):
            observed = [self.hamiltonian.msps[q].jz for q in block]
            expected = list(range(-j2val, j2val + 1, 2))
            if observed != expected:
                raise ValueError(
                    "Each (n, l, j, tz) block must be ordered by increasing "
                    f"2m. Expected {expected}, got {observed}."
                )

    def _validate_register_layout(self, circuit, ancilla_qubit, target_qubits):
        """Validate the register layout required by post-selection."""
        target_qubits = list(target_qubits)
        expected_targets = list(
            range(self.num_ancillas, self.num_ancillas + self.num_targets)
        )
        if target_qubits != expected_targets:
            raise ValueError(
                "target_qubits must immediately follow the least-significant "
                f"ancillas. Expected {expected_targets}, got {target_qubits}."
            )
        if ancilla_qubit not in range(self.num_ancillas):
            raise ValueError("ancilla_qubit must belong to the ancilla register.")
        if circuit.num_qubits != self.num_ancillas + self.num_targets:
            raise ValueError("Circuit size is inconsistent with the projector.")
        return target_qubits


def print_state_amplitudes(statevector, Nq, Na, filter_0=True, tolerance=1e-4):
    """Print significant amplitudes for one selected ancilla sector.

    Parameters
    ----------
    statevector : array_like
        Statevector of the ancilla-plus-target system.
    Nq : int
        Number of target qubits.
    Na : int
        Number of least-significant ancilla qubits.
    filter_0 : bool, optional
        Select ancilla state ``0`` when ``True``. When ``False``, select the
        all-ones ancilla state ``2**Na - 1``.
    tolerance : float, optional
        Suppress amplitudes with magnitude below this threshold.
    """
    sv_system = np.array(statevector).reshape(-1, 2**Na)
    if filter_0:
        sv_system = sv_system[:, 0]  # Select the all-zero ancilla sector.
    else:
        sv_system = sv_system[:, -1]  # Select the all-one ancilla sector.

    for i in range(2**Nq):
        amplitude = sv_system[i]
        if np.abs(amplitude) < tolerance:
            continue
        bitstring = format(i, f'0{Nq}b')
        print(f"|{bitstring}>: {amplitude:.4f}")
