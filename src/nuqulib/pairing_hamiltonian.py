import numpy as np
from itertools import combinations
from qiskit.quantum_info import SparsePauliOp


class PairingHamiltonian:
    def __init__(self, Norb, Nocc, gval, delta_eps=1.0):
        self.Norb = Norb
        self.Nocc = Nocc
        self.delta_eps = delta_eps
        self.gval = gval
        self.basis = self.make_basis()
        self.epsilon = self.eval_epsilon()
        self.Hmat = self.eval_Hmat()

    def make_basis(self):
        self.basis = []
        for occ in combinations(range(self.Norb), self.Nocc):
            self.basis.append(occ)

        return self.basis

    def eval_epsilon(self):
        self.epsilon = [2 * i * self.delta_eps for i in range(self.Norb)]
        return self.epsilon

    def eval_Hmat(self):
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
        self.evals, self.evecs = np.linalg.eigh(self.Hmat)
        print("evals: ", self.evals)
        return self.evals, self.evecs

    def encoding(self):
        SPEs = self.epsilon
        pauli_list = []
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
    bitint = 0
    for i in tup:
        bitint += 2**i
    if rev:
        bitstring = "|" + format(bitint, f"0{Norb}b")[::-1] + ">"
    else:
        bitstring = "|" + format(bitint, f"0{Norb}b") + ">"
    return bitstring
