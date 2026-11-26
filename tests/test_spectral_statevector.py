"""Independent small-system checks for the ideal spectral helpers."""
import unittest
import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix
from nuqulib.spectral_statevector import (
    spectral_trajectory, statevector_krylov, statevector_odmd,
)


class SpectralStatevectorTests(unittest.TestCase):
    def test_complex_hamiltonian_and_shift(self):
        h = np.array([[-4, 1j], [-1j, -2]], complex)
        psi = np.array([1, 1j]) / np.sqrt(2)
        states = spectral_trajectory(csr_matrix(h), psi, .17, 9, -3)
        for k, state in enumerate(states):
            np.testing.assert_allclose(state, expm(-1j * (h + 3*np.eye(2)) * k*.17) @ psi, atol=1e-12)

    def test_negative_spectrum_with_redundant_krylov_vectors(self):
        h = csr_matrix(np.diag([-7., -3., -1.]))
        psi = np.sqrt([.5, .3, .2])
        states = spectral_trajectory(h, psi, .2, 16)
        result = statevector_krylov(h, states)
        self.assertEqual(result['ranks'][-1], 3)
        np.testing.assert_allclose(result['energies'][-1], [-7, -3, -1], atol=1e-10)
        for energies in result['energies']:
            self.assertGreaterEqual(energies[0], -7 - 1e-10)

    def test_odmd_phase_sign_shift_and_rank(self):
        energies = np.array([-34.5, -28., -21.])
        shift, dt = -25., .2
        signal = np.exp(-1j*np.arange(20)[:,None]*dt*(energies-shift)) @ np.array([.7,.2,.1])
        result = statevector_odmd(signal, dt, 10, energy_shift=shift)
        self.assertEqual(result['rank'], 3)
        np.testing.assert_allclose(result['energies'], energies, atol=1e-10)
        with self.assertRaises(ValueError):
            statevector_odmd(signal, dt, 11, 10)


if __name__ == '__main__':
    unittest.main()
