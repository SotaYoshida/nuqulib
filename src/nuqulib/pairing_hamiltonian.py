"""Pairing Hamiltonian implementation

This module implements the pairing Hamiltonian, a simplified model used in nuclear physics,
condensed matter physics, and quantum computing. The pairing Hamiltonian describes
the interactions between pairs of particles in a many-body system, making it a valuable tool
for studying phenomena like superconductivity and nuclear pairing effects.

This implementation is not optimized for performance but serves as a pedagogical example
of how to construct and manipulate a pairing Hamiltonian in Python.
More detailed and efficient implementations can be found as a Julia package
`PairingHamiltonians.jl <https://github.com/SotaYoshida/PairingHamiltonians.jl>`_
by the same author.
"""

import numpy as np
from itertools import combinations
from qiskit.quantum_info import SparsePauliOp


class PairingHamiltonian:
    """Pairing Hamiltonian model for nuclear many-body systems.
    
    This class implements a pairing Hamiltonian with
    single-particle energies with a fixed spacing
    and a constant pairing interaction strength. 
    
    Args:
        Norb (int): Number of single-particle orbitals.
        Nocc (int): Number of occupied orbitals.
        gval (float): Pairing interaction strength.
        delta_eps (float, optional): Single-particle energy spacing. Defaults to 1.0.
    
    Attributes:
        Norb (int): Number of orbitals.
        Nocc (int): Number of occupied states.
        gval (float): Pairing strength parameter.
        delta_eps (float): Energy level spacing.
        basis (list): List of basis states as orbital occupation tuples.
        epsilon (list): Single-particle energies.
        Hmat (numpy.ndarray): Hamiltonian matrix.
    """
    def __init__(self, Norb, Nocc, gval, delta_eps=1.0):
        self.Norb = Norb
        self.Nocc = Nocc
        self.delta_eps = delta_eps
        self.gval = gval
        self.basis = self.make_basis()
        self.epsilon = self.eval_epsilon()
        self.Hmat = self.eval_Hmat()

    def make_basis(self):
        """Generate basis states for the pairing model.
        
        Creates all possible configurations of Nocc particles in Norb orbitals
        using combinations.
        
        Returns:
            list: List of tuples representing occupied orbital configurations.
        """
        self.basis = []
        for occ in combinations(range(self.Norb), self.Nocc):
            self.basis.append(occ)

        return self.basis

    def eval_epsilon(self):
        """Calculate single-particle energies.
        
        Sets up a linear spacing of single-particle energies with
        epsilon_i = 2 * i * delta_eps for orbital i.
        
        Returns:
            list: List of single-particle energies.
        """
        self.epsilon = [2 * i * self.delta_eps for i in range(self.Norb)]
        return self.epsilon

    def eval_Hmat(self):
        """Construct the Hamiltonian matrix.
        
        Builds the Hamiltonian matrix in the many-body basis including:

        - Single-particle energy terms (diagonal)
        
        - Pairing interaction terms (off-diagonal for single excitations)
        
        Returns:
            numpy.ndarray: Hamiltonian matrix of size (dim, dim) where dim is the basis size.
        """
        dim = len(self.basis)
        self.Hmat = np.zeros((dim, dim))
        for bra_idx, bra in enumerate(self.basis):
            for ket_idx, ket in enumerate(self.basis):
                # Hamming distance
                diff = [i for i in bra if i not in ket]
                same = [i for i in bra if i in ket]
                # for SPE term
                if bra_idx == ket_idx:
                    self.Hmat[bra_idx, ket_idx] += np.sum(
                        [self.epsilon[i] for i in same]
                    )
                    self.Hmat[bra_idx, ket_idx] += -self.gval * len(same)
                # for pairing term
                if len(diff) == 1:
                    self.Hmat[bra_idx, ket_idx] = -self.gval

        return self.Hmat

    def diagonalize(self):
        """Diagonalize the Hamiltonian matrix.
        
        Performs exact diagonalization to find eigenvalues and eigenvectors
        of the Hamiltonian matrix.
        
        Returns:
            tuple: Tuple containing:
                - evals (numpy.ndarray): Eigenvalues in ascending order.
                - evecs (numpy.ndarray): Corresponding eigenvectors as columns.
        """
        self.evals, self.evecs = np.linalg.eigh(self.Hmat)
        print("evals: ", self.evals)
        return self.evals, self.evecs

    def encoding(self):
        """Encode the Hamiltonian as Pauli operators for quantum computing.
        
        Maps the pairing Hamiltonian to a sum of Pauli operators suitable
        for quantum algorithms. The encoding includes:
        
        - Identity terms for constant energy shifts
        - Z terms for single-particle energies
        - XX and YY terms for pairing interactions
        
        Returns:
            SparsePauliOp: Qiskit SparsePauliOp representing the Hamiltonian.
            
        Note:
            The qubit ordering is reversed to match Qiskit conventions.
        """
        SPEs = self.epsilon
        obs = []
        coeffs = []

        # I term
        coeff = 0.0
        op = "I" * self.Norb
        for i in range(self.Norb):
            coeff += 0.5 * (SPEs[i] - self.gval)
        obs += [op]
        coeffs += [coeff]
        # -Zp term
        for i in range(self.Norb):
            op = "I" * self.Norb
            op = op[:i] + "Z" + op[i + 1 :]
            coeff = -0.5 * (SPEs[i] - self.gval)

            op = op[::-1]
            obs += [op]
            coeffs += [coeff]
        # XX+YY term
        for i in range(self.Norb):
            for j in range(i + 1, self.Norb):
                factor = -self.gval / 2
                op = "I" * self.Norb
                op = op[:i] + "X" + op[i + 1 : j] + "X" + op[j + 1 :]
                op = op[::-1]
                obs += [op]
                coeffs += [factor]
                op = "I" * self.Norb
                op = op[::-1]
                op = op[:i] + "Y" + op[i + 1 : j] + "Y" + op[j + 1 :]
                obs += [op]
                coeffs += [factor]

        return SparsePauliOp(obs, coeffs)


def tuple_to_bitstring(tup, Norb, rev=True):
    """Convert an orbital occupation tuple to bitstring notation.
    
    Args:
        tup (tuple): Tuple of occupied orbital indices.
        Norb (int): Total number of orbitals.
        rev (bool, optional): Whether to reverse the bitstring. Defaults to True.
        
    Returns:
        str: Bitstring representation in ket notation (e.g., "|0110>").
        
    Example:
        >>> tuple_to_bitstring((0, 2), 4, rev=True)
        "|0101>"
    """
    bitint = 0
    for i in tup:
        bitint += 2**i
    if rev:
        bitstring = "|" + format(bitint, f"0{Norb}b")[::-1] + ">"
    else:
        bitstring = "|" + format(bitint, f"0{Norb}b") + ">"
    return bitstring
