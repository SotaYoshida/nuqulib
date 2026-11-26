"""Ideal-statevector spectral tutorials, without measurement-circuit overhead.

Evolution uses the full encoded sparse Hamiltonian, not FCI eigenpairs.
These helpers do not simulate shots, hardware noise, or product formulas.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply


def spectral_trajectory(hamiltonian, initial, delta_t, count, energy_shift=0.0):
    """Return rows exp[-i(H-shift I) k dt]|initial> on the full qubit space."""
    if delta_t <= 0 or count < 2:
        raise ValueError("delta_t must be positive and count must be at least two")
    h = csr_matrix(hamiltonian, dtype=complex)
    psi = np.asarray(initial, dtype=complex)
    if h.shape != (psi.size, psi.size) or psi.ndim != 1:
        raise ValueError("Hamiltonian and initial-state dimensions disagree")
    if not np.isclose(np.vdot(psi, psi), 1.0):
        raise ValueError("Initial state must be normalized")
    defect = h - h.conj().T
    if defect.nnz and np.max(np.abs(defect.data)) > 1e-10:
        raise ValueError("Hamiltonian must be Hermitian")
    # Remove the trace only inside expm_multiply's numerical algorithm; the
    # requested physical energy shift is applied explicitly below.
    states = expm_multiply(-1j * h, psi, start=0,
                           stop=(count - 1) * delta_t, num=count,
                           traceA=-1j * h.diagonal().sum())
    states *= np.exp(1j * energy_shift * delta_t * np.arange(count))[:, None]
    np.testing.assert_allclose(np.sum(abs(states)**2, axis=1), 1.0, atol=1e-9)
    return states


def statevector_krylov(hamiltonian, states, overlap_cutoff=1e-10):
    """Ritz values from cached states, retaining only independent directions.

    The cutoff is relative to the largest overlap eigenvalue. Diagonalize
    only the retained r-dimensional problem, avoiding spurious null-space
    zeros (especially important for negative nuclear energies).
    """
    if not 0 < overlap_cutoff < 1:
        raise ValueError("overlap_cutoff must lie between zero and one")
    phi = np.asarray(states).T
    overlap = phi.conj().T @ phi
    projected = phi.conj().T @ (hamiltonian @ phi)
    history, ranks = [], []
    for size in range(1, phi.shape[1] + 1):
        s = overlap[:size, :size]
        h = projected[:size, :size]
        values, vectors = np.linalg.eigh((s + s.conj().T) / 2)
        keep = values > overlap_cutoff * values[-1]
        if not np.any(keep):
            raise ValueError("No independent Krylov direction remains")
        whitening = vectors[:, keep] / np.sqrt(values[keep])
        reduced = whitening.conj().T @ h @ whitening
        history.append(np.linalg.eigvalsh((reduced + reduced.conj().T) / 2))
        ranks.append(int(keep.sum()))
    return dict(energies=history, ranks=ranks, overlap=overlap, projected=projected)


def statevector_odmd(snapshots, delta_t, rows, columns=None,
                     energy_shift=0.0, svd_cutoff=1e-10, radius_tolerance=0.05):
    """Fit shifted Hankel matrices; return energies in shift +/- pi/dt.

    svd_cutoff is relative to the largest singular value. Eigenvalues outside
    the requested unit-circle tolerance are rejected without relaxing it.
    """
    signal = np.asarray(snapshots, dtype=complex)
    columns = len(signal) - rows if columns is None else columns
    if delta_t <= 0 or min(rows, columns) < 1 or rows + columns > len(signal):
        raise ValueError("Invalid time step or Hankel shape")
    if not 0 < svd_cutoff < 1 or radius_tolerance < 0:
        raise ValueError("Invalid SVD cutoff or radius tolerance")
    indices = np.arange(rows)[:, None] + np.arange(columns)
    x, y = signal[indices], signal[indices + 1]
    u, singular, vh = np.linalg.svd(x, full_matrices=False)
    keep = singular > svd_cutoff * singular[0]
    if not np.any(keep):
        raise ValueError("Hankel matrix has zero retained rank")
    u, v = u[:, keep], vh[keep].conj().T
    reduced = u.conj().T @ y @ (v / singular[keep])
    eigenvalues = np.linalg.eigvals(reduced)
    accepted = np.abs(np.abs(eigenvalues) - 1) <= radius_tolerance
    energies = energy_shift - np.angle(eigenvalues[accepted]) / delta_t
    fitted = y @ (v / singular[keep]) @ u.conj().T
    return dict(energies=np.sort(energies), eigenvalues=eigenvalues,
                accepted=accepted, rank=int(keep.sum()),
                fit_error=float(np.linalg.norm(fitted @ x - y)),
                shape=(rows, columns))
