import copy
from collections import Counter
import itertools
import multiprocessing
from multiprocessing import get_context
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from sympy.physics.quantum.cg import CG as ClebschGordan
from tqdm import tqdm
from .encoding import *

element = ['NA', 
    'H',  'He', 'Li', 'Be', 'B',  'C',  'N',  'O',  'F',  'Ne', 
    'Na', 'Mg', 'Al', 'Si', 'P',  'S',  'Cl', 'Ar', 'K',  'Ca',
    'Sc', 'Ti', 'V',  'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y',  'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I',  'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W',  'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U',  'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og' ]


class Orbit_nljjztz:
    def __init__(self, n, l, j, jz, tz):
        self.n = n
        self.l = l
        self.j = j
        self.jz = jz
        self.tz = tz
        self.e = 2 * n + l


class Orbit_nljtz:
    def __init__(self, n, l, j, tz):
        self.n = n
        self.l = l
        self.j = j
        self.tz = tz
        self.e = 2 * n + l


def get_spsidx_from_nljtz(single_particle_states: list, n, l, j, tz):
    for idx, sps in enumerate(single_particle_states):
        n_ = sps.n
        l_ = sps.l
        j_ = sps.j
        tz_ = sps.tz
        if n == n_ and l == l_ and j == j_ and tz == tz_:
            return idx
    print("Warning: no such sps in the model space:", n, l, j, tz)
    return None


class Orbit_nlj:
    def __init__(self, n, l, j):
        self.n = n
        self.l = l
        self.j = j
        self.e = 2 * n + l


class JTcoupledOrbitals:
    def __init__(self, emax):
        self.emax = emax
        self.orbitals = {}
        self.dict_sps2JTorbitals = {}

    def add_orbital(self, idx, n, l, j):
        self.orbitals[idx] = Orbit_nlj(n, l, j)

    def get_dict_sps2JTsps(self, sps_defined_in_NN):
        """
        Convert the single particle states (sps) to the JT coupled states.
        The sps are defined when the NN interaction file is loaded.
        """
        print("sps_defined_in_NN", sps_defined_in_NN)
        dict_sps2JTorbitals = {}
        for idx_sps, sps in enumerate(sps_defined_in_NN):
            idx_sps_, n, l, j, tz = sps
            assert idx_sps == idx_sps_, "idx_sps should be the same as idx_sps_"
            for idx_JT, JT in self.orbitals.items():
                n_JT = JT.n
                l_JT = JT.l
                j_JT = JT.j
                if (n == n_JT) and (l == l_JT) and (j == j_JT):
                    dict_sps2JTorbitals[idx_sps] = idx_JT
                    break
            else:
                print(f"idx_sps {idx_sps} not found in JT orbitals")
                print("You should check the model space for NN and 3NF files")
        self.dict_sps2JTorbitals = dict_sps2JTorbitals
        return dict_sps2JTorbitals


class Hamiltonian:
    def __init__(
        self,
        fn_NN,
        Z,
        N,
        fn_3NF=None,
        ncsm=False,
        emax_truncate=20,
        Qiskit_order=True,
        verbose=False,
        e3max=None,
    ):
        self.channel = {-3: "ppp", -1: "ppn", 1: "pnn", 3: "nnn"}
        self.fn_NN = fn_NN
        self.Z = Z
        self.N = N
        self.Anum = Z + N
        self.nucleus = str(self.Anum) + element[Z]
        self.emax = emax_truncate
        self.verbose = verbose
        self.ncsm = ncsm
        self.Qiskit_order = Qiskit_order
        self.hw = None
        self.fn_3NF = fn_3NF
        if ncsm:
            self.hw = self.extract_hw()
        (
            self.nsp_p,
            self.nsp_n,
            self.core_p,
            self.core_n,
            self.single_particle_states,
            self.v1b,
            self.v2b,
        ) = self.read_snt_file(self.fn_NN)
        self.vZ = self.Z - self.core_p
        self.vN = self.N - self.core_n
        self.proton_qubits, self.neutron_qubits, self.msps, self.dict_sps2msps = (
            self.get_mscheme_sps()
        )
        self.n_qubits = len(self.msps)
        self.n_qubits_p = len(self.proton_qubits)
        self.n_qubits_n = len(self.neutron_qubits)
        self.e3max = e3max if e3max != None else emax_truncate
        if (ncsm or fn_3NF != None) and e3max is None:
            print(
                "Warning: e3max is not set. You must specify it. Using emax_truncate value for it for now."
            )
            self.e3max = emax_truncate
        if ncsm:
            self.e1max_file, self.e2max, self.e3max_file = self.guess_emax_from_fn(
                self.fn_3NF
            )

        self.CG_dict = {}
        if fn_3NF != None:
            self.fn_3NF = fn_3NF
            self.JTorbitals = JTcoupledOrbitals(self.emax)
            if "readable.txt" in self.fn_3NF:
                self.v3b_pn = self.read_3NF_readable(verbose=verbose)
            elif "me3j.gz" in self.fn_3NF:
                idx_sps = 0
                for sps in self.single_particle_states:
                    idx_sps += 1
                    n, l, j, tz = sps.n, sps.l, sps.j, sps.tz
                    if tz > 0:
                        continue
                    self.JTorbitals.add_orbital(idx_sps, n, l, j)
                obj = ReadThBME_me3jgz(
                    self.single_particle_states,
                    self.JTorbitals,
                    self.Z,
                    self.N,
                    self.fn_3NF,
                    emax_truncate,
                    self.e1max_file,
                    self.e2max,
                    self.e3max,
                    self.e3max_file,
                    return_pnME_3NF=True,
                )
                self.v3b_pn = obj.pnME_3NF
                obj = None
            else:
                print("fn_3NF is now: ", self.fn_3NF)
                raise ValueError(
                    "fn_3NF should be either readable.text or me3j.gz file format"
                )

            # print("v3b_pn")
            # for key, value in self.v3b_pn.items():
            #     pn_a, pn_b, pn_c, pn_d, pn_e, pn_f, Jab, Jde, Jabc = key
            #     print(f"pnkey: {key} value :{value}")

            self.v3b_Mscheme = {}

    def guess_emax_from_fn(self, fn_3NF):
        e1max_file = e2max_file = e3max_file = None
        if fn_3NF is not None and "_ms" in fn_3NF:
            txt = fn_3NF.split("_ms")[-1].split(".")[0].split("_")
            e1max_file, e2max_file, e3max_file = list(map(int, txt))
        if e1max_file is None or e2max_file is None or e3max_file is None:
            print(
                "Warning: e1max_file, e2max, e3max_file are not set from the file name. ",
                "For now, we used emax_truncate value for them multiplied by 1/2/3",
            )
            e1max_file = self.emax
            e2max_file = self.emax * 2
            e3max_file = self.emax * 3
        return e1max_file, e2max_file, e3max_file

    def extract_hw(self):
        txt = self.fn_NN.split("hw")[-1]
        hw = txt.split("_")[0]
        return float(hw)

    def read_snt_file(self, fn_NN_in):  
        """
        To read (valence-)NN interaction file in snt format.

        For NCSM, one-body matrix elements are purely kinetic terms, :math:`T_{n,n'}`.
        :math:`\hbar\omega *(A-1)/A` factor is to be multiplied.
        """
        with open(fn_NN_in) as f:
            lines = f.readlines()
        single_particle_states = []
        excluded = []
        v1b = []
        v2b = []
        dict_sps = {}
        sector = 0
        count = 0
        for line in lines:
            if line[0] == "#" or line[0] == "!":
                continue
            line = line.split("!")[0]
            tl = line.split()
            if sector == 0:  # reading sps
                if len(tl) == 4:
                    nsp_p = int(tl[0])
                    nsp_n = int(tl[1])
                    core_p = int(tl[2])
                    core_n = int(tl[3])
                    if self.verbose:
                        print(
                            "nsp_p, nsp_n, core_p, core_n", nsp_p, nsp_n, core_p, core_n
                        )
                    continue
                if len(tl) == 5:
                    idx, n, l, j, tz = list(map(int, tl))
                    te = 2 * n + l
                    if self.verbose:
                        print(f"idx, n, l, j, tz, te: {idx, n, l, j, tz, te}")
                    if te > self.emax:
                        excluded.append(idx)
                        continue
                    dict_sps[idx] = count
                    # single_particle_states.append((count, n, l, j, tz))
                    single_particle_states.append(Orbit_nljtz(n, l, j, tz))

                    count += 1
                    assert abs(tz) == 1
                if len(tl) == 2 and tl[1] == "0":  # for valence
                    sector = 1
                    continue
                if len(tl) == 3 and tl[1] == "10":  # for ncsm
                    A, massop, hw = list(map(float, tl))
                    sector = 1
                    continue
            if sector == 1:  # reading 1b
                Sfactor = Vfactor = 1.0
                if self.ncsm:
                    Sfactor = (self.Anum - 1) / self.Anum * self.hw
                """                
                For some empirical shell model interaction, TBME is multiplied by a factor (A/Aref)^p
                the following block is needed to take account of this
                """
                if len(tl) != 3 and not (self.ncsm):
                    if (
                        len(tl) == 2
                    ):  # The first one is to be num_tbme, and the second one is "probably" 0
                        massop = 0  # using raw TBME for all isotopes in the model space
                    else:
                        if (
                            tl[1] != "0"
                        ):  # should be like  518   1  42  -0.35000: num_tbme, massop, Aref, p
                            n_tbme, massop, Aref, p = tl
                            massop = int(massop)
                            Vfactor = (self.Anum / float(Aref)) ** float(p)
                            print(
                                f"massop is set to {massop}. You may need to care about the mass number A={self.Anum} for your system"
                            )

                    sector = 2
                    continue
                if len(tl) == 3:
                    a, b, Tab = tl
                    if (int(a) > nsp_p + nsp_n and b == "10") or (b == "0"):
                        sector = 2
                        continue
                    a = int(a)
                    b = int(b)
                    if a in excluded or b in excluded:
                        continue
                    a = dict_sps[a]
                    b = dict_sps[b]
                    Tab = float(Tab) * Sfactor
                    v1b.append((a, b, Tab))

            if sector == 2:  # reading 2b
                if len(tl) == 7:  # for NuHamil output
                    a, b, c, d, J = list(map(int, tl[:5]))
                    if a in excluded or b in excluded or c in excluded or d in excluded:
                        continue
                    a = dict_sps[a]
                    b = dict_sps[b]
                    c = dict_sps[c]
                    d = dict_sps[d]
                    vint, vkin = list(map(float, tl[5:]))
                    v = vint + vkin * self.hw / self.Anum
                    v2b.append((a, b, c, d, J, v))
                if len(tl) == 6:  # valence snt files used in KSHELL code
                    a, b, c, d, J = list(map(int, tl[:5]))
                    if a in excluded or b in excluded or c in excluded or d in excluded:
                        continue
                    a = dict_sps[a]
                    b = dict_sps[b]
                    c = dict_sps[c]
                    d = dict_sps[d]
                    v = vint = float(tl[5]) * Vfactor
                    v2b.append((a, b, c, d, J, v))
        return nsp_p, nsp_n, core_p, core_n, single_particle_states, v1b, v2b

    def get_mscheme_sps(self):
        mps = []
        dict_sps2msps = {}
        count = 0
        for idx, sps in enumerate(self.single_particle_states):
            n = sps.n
            l = sps.l
            j = sps.j
            tz = sps.tz
            if tz != -1:
                continue
            dict_sps2msps[idx] = []
            for jz in range(-j, j + 2, 2):
                # mps.append((idx, n, l, j, jz, tz))
                mps.append(Orbit_nljjztz(n, l, j, jz, tz))
                dict_sps2msps[idx].append(count)
                count += 1
        num_proton = count
        for idx, sps in enumerate(self.single_particle_states):
            n = sps.n
            l = sps.l
            j = sps.j
            tz = sps.tz
            if tz != 1:
                continue
            dict_sps2msps[idx] = []
            for jz in range(-j, j + 2, 2):
                mps.append(Orbit_nljjztz(n, l, j, jz, tz))
                dict_sps2msps[idx].append(count)
                count += 1
        if self.verbose:
            print("mps", mps)
            print("dict_sps2msps", dict_sps2msps)
        proton_register = range(num_proton)
        neutron_register = range(num_proton, count)
        return proton_register, neutron_register, mps, dict_sps2msps

    def delta(self, morb_a, morb_b):
        n_a = morb_a.n
        l_a = morb_a.l
        j_a = morb_a.j
        jz_a = morb_a.jz
        tz_a = morb_a.tz
        n_b = morb_b.n
        l_b = morb_b.l
        j_b = morb_b.j
        jz_b = morb_b.jz
        tz_b = morb_b.tz
        if n_a == n_b and l_a == l_b and j_a == j_b and tz_a == tz_b:
            return 1
        else:
            return 0

    def op_dict_T1_permutations(self, op_dict, aa, bb, cc, dd, J, v):
        bitstr = "+_" + str(aa) + " +_" + str(bb) + " -_" + str(dd) + " -_" + str(cc)
        if [aa, bb] == [cc, dd]:
            if bitstr in op_dict:
                op_dict[bitstr] += v
            else:
                op_dict[bitstr] = v
        else:
            if bitstr in op_dict:
                op_dict[bitstr] += v
            else:
                op_dict[bitstr] = v

            # exchange bra and ket
            bitstr = (
                "+_" + str(cc) + " +_" + str(dd) + " -_" + str(bb) + " -_" + str(aa)
            )
            if bitstr in op_dict:
                op_dict[bitstr] += v
            else:
                op_dict[bitstr] = v

    def get_mscheme_H(self, opform=False):
        op_dict_1b = {}
        if opform:
            op_dict_pp = {}
            op_dict_nn = {}
            op_dict_pn = {}
        else:
            op_dict_pp = []
            op_dict_nn = []
            op_dict_pn = []

        # for 1-body term
        num_1b_term = 0
        for a, b, Tab in self.v1b:
            # print("1b in MS", a, b, Tab)
            for aa in self.dict_sps2msps[a]:
                morb_a = self.msps[aa]
                # idx_ma, n_a, l_a, j_a, jz_a, tz_a = morb_a
                n_a = morb_a.n
                l_a = morb_a.l
                j_a = morb_a.j
                jz_a = morb_a.jz
                tz_a = morb_a.tz
                for bb in self.dict_sps2msps[b]:
                    morb_b = self.msps[bb]
                    # idx_mb, n_b, l_b, j_b, jz_b, tz_b = morb_b
                    n_b = morb_b.n
                    l_b = morb_b.l
                    j_b = morb_b.j
                    jz_b = morb_b.jz
                    tz_b = morb_b.tz
                    if aa == bb or (
                        l_a == l_b and j_a == j_b and jz_a == jz_b and tz_a == tz_b
                    ):
                        if opform:
                            bitstr = "+_" + str(aa) + " -_" + str(bb)
                            op_dict_1b[bitstr] = Tab
                            num_1b_term += 1
                            if aa != bb:
                                bitstr = "+_" + str(bb) + " -_" + str(aa)
                                op_dict_1b[bitstr] = Tab
                                num_1b_term += 1
                        else:
                            op_dict_1b[(aa + 1, bb + 1)] = Tab
                            num_1b_term += 1
                            if aa != bb:
                                op_dict_1b[(bb + 1, aa + 1)] = Tab
                                num_1b_term += 1

        # for 2-body term
        num_pp = num_nn = num_pn = 0
        for a, b, c, d, J, V in self.v2b:
            # make them canonical order
            flip_ab = (-1) ** ((self.msps[a].j + self.msps[b].j) // 2 - J)
            flip_cd = (-1) ** ((self.msps[c].j + self.msps[d].j) // 2 - J)
            if a > b:
                a, b = b, a
                V *= flip_ab
            if c > d:
                c, d = d, c
                V *= flip_cd

            for aa in self.dict_sps2msps[a]:
                morb_a = self.msps[aa]
                n_a = morb_a.n
                l_a = morb_a.l
                j_a = morb_a.j
                jz_a = morb_a.jz
                tz_a = morb_a.tz
                for bb in self.dict_sps2msps[b]:
                    if aa >= bb:
                        continue
                    morb_b = self.msps[bb]
                    n_b = morb_b.n
                    l_b = morb_b.l
                    j_b = morb_b.j
                    jz_b = morb_b.jz
                    tz_b = morb_b.tz
                    phase_bra = (-1) ** ((j_a + j_b) // 2 - J)
                    Mbra = (jz_a + jz_b) // 2
                    N_ab = 1.0
                    delta_ab = self.delta(morb_a, morb_b)
                    Tz = tz_a + tz_b
                    if abs(Tz) == 2 and delta_ab == 1:
                        N_ab = 2 / np.sqrt(1 + (-1) ** J)
                    for cc in self.dict_sps2msps[c]:
                        morb_c = self.msps[cc]
                        n_c = morb_c.n
                        l_c = morb_c.l
                        j_c = morb_c.j
                        jz_c = morb_c.jz
                        tz_c = morb_c.tz
                        for dd in self.dict_sps2msps[d]:
                            if cc >= dd:
                                continue
                            if Tz != 0 and aa > cc:  # or (aa == cc and bb > dd):
                                continue
                            if Tz == 0 and (set([a, b]) == set([c, d])) and aa > cc:
                                continue

                            morb_d = self.msps[dd]
                            n_d = morb_d.n
                            l_d = morb_d.l
                            j_d = morb_d.j
                            jz_d = morb_d.jz
                            tz_d = morb_d.tz
                            phase_ket = (-1) ** ((j_c + j_d) // 2 - J)
                            Mket = (jz_c + jz_d) // 2
                            if Mbra != Mket:
                                continue

                            N_cd = 1.0
                            delta_cd = self.delta(morb_c, morb_d)
                            if abs(tz_c + tz_d) == 2 and delta_cd == 1:
                                N_cd = 2 / np.sqrt(1 + (-1) ** J)

                            CG1 = self.get_cG(morb_a, morb_b, J)
                            CG2 = self.get_cG(morb_c, morb_d, J)

                            v = V * CG1 * CG2 * N_ab * N_cd
                            if v == 0:
                                continue

                            if Tz == -2:
                                num_pp += 1
                                if opform:
                                    self.op_dict_T1_permutations(
                                        op_dict_pp, aa, bb, cc, dd, J, v
                                    )
                                else:
                                    op_dict_pp.append(
                                        [aa + 1, bb + 1, cc + 1, dd + 1, J, v]
                                    )
                            elif Tz == 2:
                                num_nn += 1
                                if opform:
                                    self.op_dict_T1_permutations(
                                        op_dict_nn, aa, bb, cc, dd, J, v
                                    )
                                else:
                                    op_dict_nn.append(
                                        [aa + 1, bb + 1, cc + 1, dd + 1, J, v]
                                    )
                            else:
                                num_pn += 1
                                # for pn interaction, we need to consider operators separately
                                if opform:
                                    p_str = "+_" + str(aa) + " -_" + str(cc)
                                    n_str = (
                                        "+_"
                                        + str(bb - self.n_qubits_p)
                                        + " -_"
                                        + str(dd - self.n_qubits_p)
                                    )
                                    if (p_str, n_str) in op_dict_pn:
                                        op_dict_pn[(p_str, n_str)] += v / 2
                                    else:
                                        op_dict_pn[(p_str, n_str)] = v / 2
                                    p_str = "+_" + str(cc) + " -_" + str(aa)
                                    n_str = (
                                        "+_"
                                        + str(dd - self.n_qubits_p)
                                        + " -_"
                                        + str(bb - self.n_qubits_p)
                                    )
                                    if (p_str, n_str) in op_dict_pn:
                                        op_dict_pn[(p_str, n_str)] += v / 2
                                    else:
                                        op_dict_pn[(p_str, n_str)] = v / 2
                                else:
                                    op_dict_pn.append(
                                        [aa + 1, bb + 1, cc + 1, dd + 1, J, v]
                                    )
        print(f"# of H_m terms, 1b: {num_1b_term}, 2b pp: {num_pp}, nn: {num_nn}, pn: {num_pn}")
        Hamildict = {
            "SPE": op_dict_1b,
            "Vpp": op_dict_pp,
            "Vnn": op_dict_nn,
            "Vpn": op_dict_pn,
        }
        if opform:
            return Hamildict
        else:
            Hamildict = sum_over_J(Hamildict)
            return Hamildict

    def mapping_opform(self, Hamildict_opform, mapping_method):
        H_1b = mapping_to_Pauli_string(
            FermionicOp(Hamildict_opform["SPE"], num_spin_orbitals=self.n_qubits),
            self.n_qubits,
            method=mapping_method,
        )
        H_pp = mapping_to_Pauli_string(
            FermionicOp(Hamildict_opform["Vpp"], num_spin_orbitals=self.n_qubits),
            self.n_qubits,
            method=mapping_method,
        )
        H_nn = mapping_to_Pauli_string(
            FermionicOp(Hamildict_opform["Vnn"], num_spin_orbitals=self.n_qubits),
            self.n_qubits,
            method=mapping_method,
        )
        H_pn = mapping_of_pn_hamiltonians(
            Hamildict_opform["Vpn"],
            self.n_qubits_p,
            self.n_qubits_n,
            method=mapping_method,
        )
        H_pn = removing_redundant_terms(H_pn)
        return H_1b, H_pp, H_nn, H_pn

    def get_cG(self, sps_i, sps_j, J):
        j = sps_i.j
        jz = sps_i.jz
        j_ = sps_j.j
        jz_ = sps_j.jz
        M = (jz + jz_) // 2
        return float(ClebschGordan(j / 2, jz / 2, j_ / 2, jz_ / 2, J, M).doit())

    def get_midx_from_nljjztz(self, n, l, j, jz, tz):
        for idx, target in enumerate(self.msps):
            n_ = target.n
            l_ = target.l
            j_ = target.j
            jz_ = target.jz
            tz_ = target.tz
            if n == n_ and l == l_ and j == j_ and jz == jz_ and tz == tz_:
                return idx
        print("Warning: no such msps in the model space:", n, l, j, jz, tz)
        return None

    def read_3NF_readable(self, verbose=False):
        """
        Read the NuHamil 3NF file in readable.text fmt.
        This method should be modified if the file is too large.

        In readable.text, V_{3N} is given as a function of a set of quanta,
        {a,b,c, Jab, Tab, d,e,f, Jde, Tde, Jabc, Tabc, Jabc, Tabc}
        where a~f are the single particle states having {n,l,j} quanta.
        Those matrix elements correspond to Eq.(41) in NuHamil paper, `T.Miyagi, EPJA (2023)59:150 <https://doi.org/10.1140/epja/s10050-023-01039-y>`_.

        pnME: proton-neutron matrix elements are obtained through the Clebsch-Gordan coefficients as detailed in Eq.(36) of NuHamil paper.
        """
        with open(self.fn_3NF, "r") as f:
            lines = f.readlines()
        pnME_3NF = {}
        sector = 0
        for line in lines:
            if ("idx," in line) and ("n," in line) and ("l," in line) and ("j" in line):
                sector = 1
                continue
            if (
                ("a," in line)
                and ("b," in line)
                and ("c," in line)
                and ("d" in line)
                and ("Tdef" in line)
            ):
                sector = 2
                continue
            if sector == 1:
                idx, n, l, j = map(int, line.split())
                self.JTorbitals.add_orbital(idx, n, l, j)
            if sector == 2:
                # read the matrix elements
                tl = line.split()
                a, b, c, Jab, Tab, d, e, f, Jde, Tde, Jabc, Tabc, Jdef, Tdef = map(
                    int, tl[:-1]
                )
                ME = float(tl[-1])
                relevant = True
                for target in [a, b, c, d, e, f]:
                    t_e = self.JTorbitals.orbitals[target].e
                    if t_e > self.emax:
                        relevant = False
                        break
                if not relevant:
                    continue
                assert Jdef == Jabc, "Jdef should be the same as Jabc"
                assert Tdef == Tabc, "Tdef should be the same as Tabc"
                if verbose:
                    print(
                        f"a {a} b {b} c {c} Jab {Jab} Tab {Tab} d {d} e {e} f {f} Jde {Jde} Tde {Tde} Jtot {Jabc} Ttot {Tabc}"
                    )
                orb_a = self.JTorbitals.orbitals[a]
                orb_b = self.JTorbitals.orbitals[b]
                orb_c = self.JTorbitals.orbitals[c]
                orb_d = self.JTorbitals.orbitals[d]
                orb_e = self.JTorbitals.orbitals[e]
                orb_f = self.JTorbitals.orbitals[f]
                for tz_a, tz_b, tz_c, tz_d, tz_e, tz_f in itertools.product(
                    [-1, 1], repeat=6
                ):
                    if (tz_a + tz_b + tz_c) != (tz_d + tz_e + tz_f):
                        continue
                    if (tz_a + tz_b) > 2 * Tab:
                        continue
                    if (tz_d + tz_e) > 2 * Tde:
                        continue
                    pn_a = get_spsidx_from_nljtz(
                        self.single_particle_states, orb_a.n, orb_a.l, orb_a.j, tz_a
                    )
                    pn_b = get_spsidx_from_nljtz(
                        self.single_particle_states, orb_b.n, orb_b.l, orb_b.j, tz_b
                    )
                    pn_c = get_spsidx_from_nljtz(
                        self.single_particle_states, orb_c.n, orb_c.l, orb_c.j, tz_c
                    )
                    pn_d = get_spsidx_from_nljtz(
                        self.single_particle_states, orb_d.n, orb_d.l, orb_d.j, tz_d
                    )
                    pn_e = get_spsidx_from_nljtz(
                        self.single_particle_states, orb_e.n, orb_e.l, orb_e.j, tz_e
                    )
                    pn_f = get_spsidx_from_nljtz(
                        self.single_particle_states, orb_f.n, orb_f.l, orb_f.j, tz_f
                    )
                    Tz_bra = tz_a + tz_b + tz_c
                    if Tz_bra == 3 and self.N < 3:
                        continue  # nnn
                    if Tz_bra == 1 and (self.Z == 0 or self.N <= 1):
                        continue  # pnn
                    if Tz_bra == -1 and (self.N == 0 or self.Z <= 1):
                        continue  # npp
                    if Tz_bra == -3 and self.Z < 3:
                        continue  # ppp
                    if abs(Tz_bra) > Tabc:
                        continue
                    pnkey = (pn_a, pn_b, pn_c, pn_d, pn_e, pn_f, Jab, Jde, Jabc)
                    cg_1 = self.get_CGs_from_dict(
                        1, tz_a, 1, tz_b, Tab * 2, (tz_a + tz_b)
                    )
                    cg_2 = self.get_CGs_from_dict(
                        1, tz_d, 1, tz_e, Tde * 2, (tz_d + tz_e)
                    )
                    cg_3 = self.get_CGs_from_dict(
                        Tab * 2, (tz_a + tz_b), 1, tz_c, Tabc, (tz_a + tz_b + tz_c)
                    )
                    cg_4 = self.get_CGs_from_dict(
                        Tde * 2, (tz_d + tz_e), 1, tz_f, Tabc, (tz_d + tz_e + tz_f)
                    )
                    cgfact = cg_1 * cg_2 * cg_3 * cg_4
                    if abs(cgfact) < 1.0e-8:
                        continue
                    part = ME * cgfact
                    if pnkey in pnME_3NF.keys():
                        pnME_3NF[pnkey] += part
                    else:
                        pnME_3NF[pnkey] = part

                    # bra <-> ket permutations must be considered, since they are absent in readable.text format
                    if pnkey != (pn_d, pn_e, pn_f, pn_a, pn_b, pn_c, Jde, Jab, Jabc):
                        pnkey = (pn_d, pn_e, pn_f, pn_a, pn_b, pn_c, Jde, Jab, Jabc)
                        part = ME * cgfact
                        if pnkey in pnME_3NF.keys():
                            pnME_3NF[pnkey] += part
                        else:
                            pnME_3NF[pnkey] = part

        if verbose:
            print("pnME_3NF")
            for key, value in pnME_3NF.items():
                pn_a, pn_b, pn_c, pn_d, pn_e, pn_f, Jab, Jde, Jabc = key
                # print(f"pnkey: {key} value :{value}")
                # if abs(value) < 1.e-8:
                #    continue
                # print(f"pn_a {pn_a} pn_b {pn_b} pn_c {pn_c} pn_d {pn_d} pn_e {pn_e} pn_f {pn_f} Jab {Jab} Jde {Jde} Jabc {Jabc} value {value}")
            print(
                "Total number of 3NF matrix elements in pn J-coupld form",
                len(list(pnME_3NF.keys())),
            )
        return pnME_3NF

    def get_CGs_from_dict(self, j1, m1, j2, m2, J, M):
        tkey = (j1, m1, j2, m2, J, M)
        if tkey in self.CG_dict:
            return self.CG_dict[tkey]
        else:
            CG = float(
                ClebschGordan(j1 / 2, m1 / 2, j2 / 2, m2 / 2, J / 2, M / 2).doit()
            )
            self.CG_dict[tkey] = CG
            return CG

    def set_mscheme_3NF(self, dev_mode=True, verbose=False):
        """
        Q. Why do we need to devide the 3NF matrix elements by 36?
        A. All the permutations of the 3NF matrix elements are counted below.
        It would be more efficient to avoid redundancy in the first place. It is left for the future.

        Q. Why do we need to multiply the 3NF matrix elements by 9 for pnn and npp?
        A. At first, proton and neutron operators commute.
        We are now generating all the permutations not taking into account of this.
        While transforming the pn JT-coupled form to J-coupled form,
        we generate something like <pnn|V|npn>, <pnn|V|nnp>, etc.
        The factor of 9 is to take account of this.
        """
        v3b_Mscheme = {}
        if verbose:
            print("Converting pn J-coupled 3NF to mscheme...")
        print("Setting up 3NF matrix elements in mscheme...")
        for key, ME_3n_pn in tqdm(self.v3b_pn.items()):
            pn_a, pn_b, pn_c, pn_d, pn_e, pn_f, Jab, Jde, Jabc = key
            orb_a = self.single_particle_states[pn_a]
            ja_range = range(-orb_a.j, orb_a.j + 1, 2)
            orb_b = self.single_particle_states[pn_b]
            jb_range = range(-orb_b.j, orb_b.j + 1, 2)
            orb_c = self.single_particle_states[pn_c]
            jc_range = range(-orb_c.j, orb_c.j + 1, 2)
            orb_d = self.single_particle_states[pn_d]
            jd_range = range(-orb_d.j, orb_d.j + 1, 2)
            orb_e = self.single_particle_states[pn_e]
            je_range = range(-orb_e.j, orb_e.j + 1, 2)
            orb_f = self.single_particle_states[pn_f]
            jf_range = range(-orb_f.j, orb_f.j + 1, 2)
            Tz_bra = orb_a.tz + orb_b.tz + orb_c.tz
            Tz_ket = orb_d.tz + orb_e.tz + orb_f.tz
            assert Tz_bra == Tz_ket, "Tz_bra and Tz_ket must be the same"
            ME_3n_pn /= 36
            if abs(Tz_bra) == 3:
                1
            else:
                ME_3n_pn *= 9
                if dev_mode and (
                    orb_a.tz != orb_d.tz or orb_b.tz != orb_e.tz or orb_c.tz != orb_f.tz
                ):
                    continue

            for m_a, m_b, m_c, m_d, m_e, m_f in itertools.product(
                ja_range, jb_range, jc_range, jd_range, je_range, jf_range
            ):
                if (m_a + m_b + m_c) != (m_d + m_e + m_f):
                    continue
                if abs(m_a + m_b) > 2 * Jab:
                    continue
                if abs(m_d + m_e) > 2 * Jde:
                    continue
                if abs(m_a + m_b + m_c) > Jabc:
                    continue
                if abs(m_d + m_e + m_f) > Jabc:
                    continue

                im_a = self.get_midx_from_nljjztz(
                    orb_a.n, orb_a.l, orb_a.j, m_a, orb_a.tz
                )
                im_b = self.get_midx_from_nljjztz(
                    orb_b.n, orb_b.l, orb_b.j, m_b, orb_b.tz
                )
                im_c = self.get_midx_from_nljjztz(
                    orb_c.n, orb_c.l, orb_c.j, m_c, orb_c.tz
                )
                if im_a == im_b or im_b == im_c or im_a == im_c:
                    continue
                im_d = self.get_midx_from_nljjztz(
                    orb_d.n, orb_d.l, orb_d.j, m_d, orb_d.tz
                )
                im_e = self.get_midx_from_nljjztz(
                    orb_e.n, orb_e.l, orb_e.j, m_e, orb_e.tz
                )
                im_f = self.get_midx_from_nljjztz(
                    orb_f.n, orb_f.l, orb_f.j, m_f, orb_f.tz
                )
                if im_d == im_e or im_e == im_f or im_d == im_f:
                    continue
                # print("Tz_bra", Tz_bra, "Tz_ket", Tz_ket, im_a, im_b, im_c, im_d, im_e, im_f)
                cg1 = self.get_CGs_from_dict(
                    orb_a.j, m_a, orb_b.j, m_b, 2 * Jab, (m_a + m_b)
                )
                cg2 = self.get_CGs_from_dict(
                    orb_d.j, m_d, orb_e.j, m_e, 2 * Jde, (m_d + m_e)
                )
                if abs(cg1 * cg2) < 1.0e-16:
                    continue
                cg3 = self.get_CGs_from_dict(
                    Jab * 2, (m_a + m_b), orb_c.j, m_c, Jabc, (m_a + m_b + m_c)
                )
                cg4 = self.get_CGs_from_dict(
                    Jde * 2, (m_d + m_e), orb_f.j, m_f, Jabc, (m_d + m_e + m_f)
                )
                cgfact = cg1 * cg2 * cg3 * cg4
                if abs(cgfact) < 1.0e-8:
                    continue

                mkey = (im_a, im_b, im_c, im_d, im_e, im_f)
                if mkey in v3b_Mscheme.keys():
                    v3b_Mscheme[mkey] += ME_3n_pn * cgfact
                else:
                    v3b_Mscheme[mkey] = ME_3n_pn * cgfact

        def perms(i1, i2, i3):
            return {
                (i1, i2, i3): 1,
                (i2, i3, i1): 1,
                (i3, i1, i2): 1,
                (i2, i1, i3): -1,
                (i1, i3, i2): -1,
                (i3, i2, i1): -1,
            }

        tmp = copy.copy(v3b_Mscheme)
        for key, me in tmp.items():
            i1, i2, i3, i4, i5, i6 = key
            perms_bra = perms(i1, i2, i3)
            perms_ket = perms(i4, i5, i6)
            for i123, sign_123 in perms_bra.items():
                for i456, sign_456 in perms_ket.items():
                    v3b_Mscheme[(*i123, *i456)] = me * sign_123 * sign_456
                    v3b_Mscheme[(*i456, *i123)] = me * sign_123 * sign_456
        if verbose:
            print("After perm.")
            # for key, me in v3b_Mscheme.items():
            for key, me in tmp.items():
                im_a, im_b, im_c, im_d, im_e, im_f = key
                if set([im_a, im_b, im_c]) == set([im_d, im_e, im_f]) and (
                    11 in [im_a, im_b, im_c] or 10 in [im_a, im_b, im_c]
                ):
                    print(
                        f"<{im_a} {im_b} {im_c} |V| {im_d} {im_e} {im_f}> ME3n_M: {v3b_Mscheme[key]}"
                    )

        print(
            "Total number of 3NF matrix elements in M-scheme",
            len(list(v3b_Mscheme.keys())),
        )
        self.v3b_Mscheme = v3b_Mscheme
        return v3b_Mscheme

    def separate_proton_and_neutron(self, im_a, im_b, im_c, im_d, im_e, im_f):
        morb_a = self.msps[im_a]
        morb_b = self.msps[im_b]
        morb_c = self.msps[im_c]
        morb_d = self.msps[im_d]
        morb_e = self.msps[im_e]
        morb_f = self.msps[im_f]
        p_op_str = n_op_str = ""
        ## for bra (+)
        idx_bra = [im_a, im_b, im_c]
        for idx, morb in enumerate([morb_a, morb_b, morb_c]):
            idx_morb = idx_bra[idx]
            if morb.tz == -1:
                p_op_str += f"+_{idx_morb} "
            elif morb.tz == 1:
                n_op_str += f"+_{idx_morb - self.n_qubits_p} "
        ## for ket (-)
        idx_ket = [im_f, im_e, im_d]
        for idx, morb in enumerate([morb_f, morb_e, morb_d]):
            idx_morb = idx_ket[idx]
            if morb.tz == -1:
                p_op_str += f"-_{idx_morb} "
            elif morb.tz == 1:
                n_op_str += f"-_{idx_morb - self.n_qubits_p} "
        return p_op_str.rstrip(), n_op_str.rstrip()

    def mapping_3NF_Mscheme(self, method="Jordan-Wigner"):
        # global op_dict_3b
        global shared_data, locks

        op_dict_3b = {}
        print("Setting up op_dict_3b...")
        for mkey, value in tqdm(self.v3b_Mscheme.items()):
            im_a, im_b, im_c, im_d, im_e, im_f = mkey
            morb_a = self.msps[im_a]
            morb_b = self.msps[im_b]
            morb_c = self.msps[im_c]
            if abs(value) < 1.0e-8:
                continue
            Tz_bra = morb_a.tz + morb_b.tz + morb_c.tz
            if self.verbose:
                print(
                    f"M: <{im_a} {im_b} {im_c} |V| {im_d} {im_e} {im_f}> Chan:{self.channel[Tz_bra]} ME3n_M: {value} "
                )
            p_op_str, n_op_str = self.separate_proton_and_neutron(
                im_a, im_b, im_c, im_d, im_e, im_f
            )
            if self.verbose:
                print(f" => proton_op {p_op_str} x neutron_op {n_op_str}")
            if (p_op_str, n_op_str) in op_dict_3b:
                op_dict_3b[(p_op_str, n_op_str)] += value
            else:
                op_dict_3b[(p_op_str, n_op_str)] = value
        print("Total number of 3NF pn products", len(list(op_dict_3b.keys())))
        op_list = set_op_list_from_op_dict_3b(
            op_dict_3b, self.n_qubits_p, self.n_qubits_n, method=method
        )
        mapped_H3b = removing_redundant_ops(op_list)
        return mapped_H3b


def process_op(args):
    p_str, n_str, coeff_overall, n_qubits_p, n_qubits_n, method = args
    op_list_local = []
    op_p = mapping_to_Pauli_string(
        FermionicOp({p_str: 1.0}, num_spin_orbitals=n_qubits_p),
        n_qubits_p,
        method=method,
    )
    op_n = mapping_to_Pauli_string(
        FermionicOp({n_str: 1.0}, num_spin_orbitals=n_qubits_n),
        n_qubits_n,
        method=method,
    )
    # Precompute the labels to avoid repeated attribute look-ups in the inner loop
    p_labels = [p.to_label() for p in op_p.paulis]
    n_labels = [n.to_label() for n in op_n.paulis]
    for cp, p_label in zip(op_p.coeffs, p_labels):
        for cn, n_label in zip(op_n.coeffs, n_labels):
            op_list_local.append((n_label + p_label, coeff_overall * cp * cn))
    return op_list_local


def set_op_list_from_op_dict_3b(
    op_dict_3b, n_qubits_p, n_qubits_n, method="Jordan-Wigner"
):
    print("Setting up op_list...")
    # Prepare task arguments for each entry in op_dict_3b
    tasks = [
        (p_str, n_str, op_dict_3b[(p_str, n_str)], n_qubits_p, n_qubits_n, method)
        for (p_str, n_str) in op_dict_3b.keys()
    ]

    # Use multiprocessing Pool with a context manager
    nproc = max([multiprocessing.cpu_count() - 2, 1])
    with get_context("fork").Pool(processes=nproc) as pool:
        results = list(tqdm(pool.imap(process_op, tasks), total=len(tasks)))

    # Flatten the list of lists into a single op_list
    op_list = [item for sublist in results for item in sublist]
    print("# of op_list terms", len(op_list))
    return op_list


def worker_cU(inp):
    term, dt, n_qubits = inp
    term = term.to_label()
    op = SparsePauliOp.from_list(list(zip([term], [1.0])))
    u = PauliEvolutionGate(op, dt, synthesis=SuzukiTrotter(order=1, reps=1))
    u = u.control(1)
    qc = QuantumCircuit(n_qubits + 1)
    qc.append(u, [0] + list(range(1, n_qubits + 1)))
    qc = qc.decompose(reps=5)
    counts = qc.count_ops()
    return counts


def worker_U(inp):
    term, dt, n_qubits = inp
    term = term.to_label()
    op = SparsePauliOp.from_list(list(zip([term], [1.0])))
    u = PauliEvolutionGate(op, dt, synthesis=SuzukiTrotter(order=1, reps=1))
    qc = QuantumCircuit(n_qubits)
    qc.append(u, range(n_qubits))
    qc = qc.decompose(reps=5)
    counts = qc.count_ops()
    return counts


def counting_CNOTs(H_mapped, label_int, label_op, dt, n_qubits):
    print(f"Counting CNOTs for {label_int} with {label_op} operator...")
    nproc = multiprocessing.cpu_count() - 2
    Ntasks = len(H_mapped.paulis)
    t = tqdm(total=Ntasks)
    p = get_context("fork").Pool(processes=nproc)
    if label_op == "c-U":
        worker_func = worker_cU
    elif label_op == "U":
        worker_func = worker_U
    else:
        raise ValueError(f"Unknown label_op: {label_op}. Use 'c-U' or 'U'.")

    results_all = {}
    with get_context("fork").Pool(processes=nproc) as p:
        results = []
        for counts in p.imap_unordered(
            worker_func,
            [(term, dt, n_qubits) for term in H_mapped.paulis],
            chunksize=1000,
        ):
            results.append(counts)
            t.update()
    total_counts = Counter()
    for counts in results:
        total_counts.update(counts)
    results_all = dict(total_counts)

    p.close()
    print(f"{label_int}: # of gates in (multi-processing) {label_op}: {results_all}")


def removing_redundant_ops(op_list):
    print("Removing redundant terms... len=", len(op_list), end=" => ")
    new_dict = {}
    for label, c in tqdm(op_list):
        new_dict[label] = new_dict.get(label, 0) + c

    new_ops = list(new_dict.keys())
    new_coeffs = [new_dict[label] for label in new_ops]

    ops = [
        new_ops[i]
        for i in range(len(new_coeffs))
        if new_coeffs[i] != 0 and (np.abs(new_coeffs[i]) > 1.0e-16)
    ]
    coeffs = [
        new_coeffs[i]
        for i in range(len(new_coeffs))
        if new_coeffs[i] != 0 and (np.abs(new_coeffs[i]) > 1.0e-16)
    ]
    print(len(ops))
    return SparsePauliOp.from_list(list(zip(ops, coeffs)))


def removing_redundant_terms(ops: SparsePauliOp):
    paulis = ops.paulis
    coeffs = ops.coeffs
    print("Removing redundant terms... len=", len(paulis))
    new_dict = {}
    for p, c in tqdm(zip(ops.paulis, ops.coeffs)):
        label = p.to_label()
        new_dict[label] = new_dict.get(label, 0) + c

    new_ops = list(new_dict.keys())
    new_coeffs = [new_dict[label] for label in new_ops]

    ops = [
        new_ops[i]
        for i in range(len(new_coeffs))
        if new_coeffs[i] != 0 and (np.abs(new_coeffs[i]) > 1.0e-16)
    ]
    coeffs = [
        new_coeffs[i]
        for i in range(len(new_coeffs))
        if new_coeffs[i] != 0 and (np.abs(new_coeffs[i]) > 1.0e-16)
    ]

    return SparsePauliOp.from_list(list(zip(ops, coeffs)))


def sum_over_J(Hamil):
    new_Hamil = {}
    for key in Hamil.keys():
        if key == "SPE":
            new_Hamil[key] = Hamil[key]
        else:
            new_Hamil[key] = []
            tdict = {}
            for term in Hamil[key]:
                a, b, c, d, totJ, V = term
                tdict[(a, b, c, d)] = tdict.get((a, b, c, d), 0) + V
            for tkey in tdict.keys():
                a, b, c, d = tkey
                V = tdict[tkey]
                new_Hamil[key].append([a, b, c, d, V])
    return new_Hamil

# def hat(a):
#     return np.sqrt(2 * a + 1)


# def permutation_parity(lst):
#     # Returns 0 for even, 1 for odd permutation.
#     par = 1
#     for i in range(len(lst)):
#         for j in range(i + 1, len(lst)):
#             if lst[i] > lst[j]:
#                 par *= -1
#     return 0 if par > 0 else 1


# def sort_3_orbits(a_in, b_in, c_in):
#     # Adjust inputs if they're even
#     a_in = a_in - 1 if a_in % 2 == 0 else a_in
#     b_in = b_in - 1 if b_in % 2 == 0 else b_in
#     c_in = c_in - 1 if c_in % 2 == 0 else c_in
#     # Initialize variables
#     a, b, c = a_in, b_in, c_in
#     # Sort the values to get a >= b >= c through pairwise swaps
#     if a < b:
#         a, b = b, a
#     if b < c:
#         b, c = c, b
#     if a < b:
#         a, b = b, a
#     # Determine index based on comparisons with original inputs
#     if a_in == a:
#         idx = 0 if b_in == b else 3
#     elif a_in == b:
#         idx = 4 if b_in == a else 1
#     else:
#         idx = 2 if b_in == a else 5
#     return a, b, c, idx


# def get_nkey6_shift(a, b, c, d, e, f, int_shift=3):
#     return (
#         ((a + int_shift) << 50)
#         + ((b + int_shift) << 40)
#         + ((c + int_shift) << 30)
#         + ((d + int_shift) << 20)
#         + ((e + int_shift) << 10)
#         + (f + int_shift)
#     )


# def get_nkey6(a, b, c, d, e, f):
#     return (a << 50) + (b << 40) + (c << 30) + (d << 20) + (e << 10) + f


# def unhash_key6j(i):
#     a = i >> 50
#     b = (i >> 40) & 0x3FF
#     c = (i >> 30) & 0x3FF
#     d = (i >> 20) & 0x3FF
#     e = (i >> 10) & 0x3FF
#     f = i & 0x3FF
#     return a, b, c, d, e, f


# class sps_3Blab:
#     def __init__(
#         self,
#         e1max,
#         e1max_file,
#         e2max_file,
#         e3max,
#         e3max_file,
#         norbits_ms,
#         norbits_file,
#         sps,
#         sps_file,
#     ):
#         self.e1max = e1max
#         self.e1max_file = e1max_file
#         self.e2max_file = e2max_file
#         self.e3max = e3max
#         self.e3max_file = e3max_file
#         self.norbits_ms = norbits_ms
#         self.norbits_file = norbits_file
#         self.sps = sps
#         self.sps_file = sps_file


# def valid_check(ea, eb, ec, ed, ee, ef, e1max, e2max, e3max):
#     if ea > e1max or eb > e1max or ec > e1max or ed > e1max or ee > e1max or ef > e1max:
#         return False
#     if ea + eb > e2max or ea + ec > e2max or eb + ec > e2max:
#         return False
#     if ed + ee > e2max or ed + ef > e2max or ee + ef > e2max:
#         return False
#     if ea + eb + ec > e3max or ed + ee + ef > e3max:
#         return False
#     return True


# class ReadThBME_me3jgz:
#     def __init__(
#         self,
#         single_particle_states,
#         JT_orbitals,
#         Z,
#         N,
#         filename,
#         e1max,
#         e1max_file,
#         e2max,
#         e3max,
#         e3max_file,
#         return_pnME_3NF=True,
#     ):
#         self.filename = filename
#         self.single_particle_states = single_particle_states
#         self.JTorbitals = JT_orbitals
#         self.Z = Z
#         self.N = N
#         self.e1max = e1max
#         self.e2max = e2max
#         self.e3max = e3max
#         self.e1max_file = e1max_file
#         self.e2max_file = 2 * self.e1max_file
#         self.e3max_file = e3max_file
#         self.dWs = prep_dicts_for_WignerSymbols(e1max)
#         self.sps_3b = self.get_modelspace(e1max, e1max_file, e2max, e3max, e3max_file)
#         self.dict_idxThBME = self.count_nreads(self.sps_3b)
#         self.nread_v3bme = self.count_nreads(self.sps_3b, "ModelSpace")
#         self.count_ME_file = self.count_me3jgz(self.sps_3b)
#         self.ThBME = self.read_me3jgz(self.filename, self.count_ME_file)
#         self.v3bme, self.dict_3b_idx = self.allocate_3bme(self.sps_3b)
#         if (
#             self.e1max_file == self.e1max
#             and self.e2max_file == 2 * self.e1max
#             and self.e3max_file == self.e3max
#         ):
#             self.v3bme = self.ThBME
#         else:
#             self.v3bme = self.truncate_v3bme(
#                 len(self.v3bme),
#                 self.sps_3b,
#                 self.ThBME,
#                 self.nread_v3bme,
#                 self.dWs,
#                 self.dict_idxThBME,
#             )
#         if return_pnME_3NF:
#             self.pnME_3NF = {}
#             self.monopole_V3(
#                 self.e3max,
#                 self.sps_3b,
#                 self.dict_3b_idx,
#                 self.v3bme,
#                 self.dWs,
#                 self.pnME_3NF,
#             )
#         else:
#             self.V3mono = self.monopole_V3(
#                 self.e3max, self.sps_3b, self.dict_3b_idx, self.v3bme, self.dWs
#             )

#     def get_modelspace(self, e1max, e1max_file, e2max, e3max, e3max_file):
#         sps = {}
#         sps_file = {}
#         norbits_file = norbits_ms = 0
#         for mode in ["File", "ModelSpace"]:
#             norbits = 0
#             target = sps_file if mode == "File" else sps
#             e1adopt = e1max_file if mode == "File" else e1max
#             e3adopt = e3max_file if mode == "File" else e3max
#             for temax in range(0, e1adopt + 1):
#                 lmin = temax % 2
#                 lstep = 2
#                 lmax = temax
#                 for l in range(lmin, lmax + 1, lstep):
#                     n = (temax - l) // 2
#                     for j2 in range(abs(2 * l - 1), 2 * l + 2, 2):
#                         for tz in [-1, 1]:
#                             norbits += 1
#                             target[norbits] = Orbit_nljtz(n, l, j2, tz)
#             if mode == "File":
#                 norbits_file = norbits
#             else:
#                 norbits_ms = norbits
#         print(
#             f"Modelspace {e1max}_{e2max}_{e3max}, norbits (File): {norbits_file}, norbits (ModelSpace): {norbits_ms}"
#         )
#         return sps_3Blab(
#             e1max,
#             e1max_file,
#             e2max,
#             e3max,
#             e3max_file,
#             norbits_ms,
#             norbits_file,
#             sps,
#             sps_file,
#         )

#     def count_nreads(self, sps_3b, mode="File"):
#         # Select parameters based on mode.
#         norbits = sps_3b.norbits_file if mode == "File" else sps_3b.norbits_ms
#         e1max = sps_3b.e1max_file if mode == "File" else sps_3b.e1max
#         e2max = sps_3b.e2max_file if mode == "File" else sps_3b.e1max * 2
#         e3max = sps_3b.e3max_file if mode == "File" else sps_3b.e3max
#         sps = sps_3b.sps_file if mode == "File" else sps_3b.sps

#         dict_idx_ThBME = {}
#         nread = 0
#         # nreads is a list with length equal to norbits//2.
#         nreads = [0] * (norbits // 2)

#         # Begin loop over 'bra' indices (using odd indices starting at 1)
#         for idx_a in range(1, norbits + 1, 2):
#             # Convert to 0-index for Python list assignment.
#             nreads[(idx_a // 2)] = nread
#             oa = sps[idx_a]
#             ea = oa.e
#             if ea > e1max:
#                 continue

#             for idx_b in range(1, idx_a + 1, 2):
#                 ob = sps[idx_b]
#                 eb = ob.e
#                 if ea + eb > e2max:
#                     continue

#                 for idx_c in range(1, idx_b + 1, 2):
#                     oc = sps[idx_c]
#                     ec = oc.e
#                     if ea + eb + ec > e3max:
#                         continue

#                     # Compute angular momentum limits for the bra.
#                     JabMax = (oa.j + ob.j) // 2
#                     JabMin = abs(oa.j - ob.j) // 2
#                     if abs(oa.j - ob.j) > oc.j:
#                         twoJCMindownbra = abs(oa.j - ob.j) - oc.j
#                     elif oc.j < (oa.j + ob.j):
#                         twoJCMindownbra = 1
#                     else:
#                         twoJCMindownbra = oc.j - oa.j - ob.j
#                     twoJCMaxupbra = oa.j + ob.j + oc.j

#                     # Loop for 'ket' part.
#                     for idx_d in range(1, idx_a + 1, 2):
#                         od = sps[idx_d]
#                         ed = od.e
#                         # Determine the upper limit for idx_e based on idx_a and idx_d.
#                         end_idx_e = idx_b if idx_a == idx_d else idx_d
#                         for idx_e in range(1, end_idx_e + 1, 2):
#                             oe = sps[idx_e]
#                             ee = oe.e
#                             # Determine the upper limit for idx_f.
#                             idx_f_max = (
#                                 idx_c if (idx_a == idx_d and idx_b == idx_e) else idx_e
#                             )
#                             for idx_f in range(1, idx_f_max + 1, 2):
#                                 of = sps[idx_f]
#                                 ef = of.e
#                                 if ed + ee + ef > e3max:
#                                     continue
#                                 if (oa.l + ob.l + oc.l + od.l + oe.l + of.l) % 2 != 0:
#                                     continue

#                                 JdeMax = (od.j + oe.j) // 2
#                                 JdeMin = abs(od.j - oe.j) // 2
#                                 if abs(od.j - oe.j) > of.j:
#                                     twoJCMindownket = abs(od.j - oe.j) - of.j
#                                 elif of.j < (od.j + oe.j):
#                                     twoJCMindownket = 1
#                                 else:
#                                     twoJCMindownket = of.j - od.j - oe.j
#                                 twoJCMaxupket = od.j + oe.j + of.j

#                                 twoJCMindown = max(twoJCMindownbra, twoJCMindownket)
#                                 twoJCMaxup = min(twoJCMaxupbra, twoJCMaxupket)
#                                 if twoJCMindown > twoJCMaxup:
#                                     continue

#                                 if mode == "File":
#                                     nkey = (idx_a, idx_b, idx_c, idx_d, idx_e, idx_f)
#                                     dict_idx_ThBME[nkey] = nread

#                                 for Jab in range(JabMin, JabMax + 1):
#                                     for Jde in range(JdeMin, JdeMax + 1):
#                                         twoJCMin = max(
#                                             abs(2 * Jab - oc.j), abs(2 * Jde - of.j)
#                                         )
#                                         twoJCMax = min(2 * Jab + oc.j, 2 * Jde + of.j)
#                                         if twoJCMin > twoJCMax:
#                                             continue
#                                         blocksize = ((twoJCMax - twoJCMin) // 2 + 1) * 5

#                                         nread += blocksize

#         if mode == "File":
#             nkeys = len(dict_idx_ThBME)
#             print("size of dict_idx_ThBME:", nkeys, nkeys * 2 * 8 / 1024**3, "GB")
#             return dict_idx_ThBME
#         else:
#             return nreads

#     def count_me3jgz(self, sps_3b, mode="File"):
#         e1max = sps_3b.e1max
#         e1max_file = sps_3b.e1max_file
#         e2max_file = sps_3b.e2max_file
#         e3max_file = sps_3b.e3max_file
#         e3max = sps_3b.e3max

#         e1max_check = e1max_file if mode == "File" else e1max
#         e3max_check = e3max_file if mode == "File" else e3max
#         sps = sps_3b.sps_file if mode == "File" else sps_3b.sps
#         norbits = sps_3b.norbits_file if mode == "File" else sps_3b.norbits

#         count_ME_file = 0

#         for idx_a in range(1, norbits + 1, 2):
#             oa = sps[idx_a]
#             ea = oa.e
#             if ea > e1max_check:
#                 continue

#             for idx_b in range(1, idx_a + 1, 2):
#                 ob = sps[idx_b]
#                 eb = ob.e
#                 if ea + eb > e2max_file:
#                     continue

#                 for idx_c in range(1, idx_b + 1, 2):
#                     oc = sps[idx_c]
#                     ec = oc.e
#                     if ea + eb + ec > e3max_check:
#                         continue

#                     JabMax = (oa.j + ob.j) // 2
#                     JabMin = abs(oa.j - ob.j) // 2
#                     if abs(oa.j - ob.j) > oc.j:
#                         twoJCMindownbra = abs(oa.j - ob.j) - oc.j
#                     elif oc.j < (oa.j + ob.j):
#                         twoJCMindownbra = 1
#                     else:
#                         twoJCMindownbra = oc.j - oa.j - ob.j
#                     twoJCMaxupbra = oa.j + ob.j + oc.j

#                     for idx_d in range(1, idx_a + 1, 2):
#                         od = sps[idx_d]
#                         ed = od.e
#                         if ed > e1max_check:
#                             continue

#                         if idx_a == idx_d:
#                             upper_idx_e = idx_b
#                         else:
#                             upper_idx_e = idx_d

#                         for idx_e in range(1, upper_idx_e + 1, 2):
#                             oe = sps[idx_e]
#                             ee = oe.e
#                             if ee > e1max_check:
#                                 continue

#                             idx_f_max = (
#                                 idx_c if (idx_a == idx_d and idx_b == idx_e) else idx_e
#                             )

#                             for idx_f in range(1, idx_f_max + 1, 2):
#                                 of = sps[idx_f]
#                                 ef = of.e
#                                 if ef > e1max_check:
#                                     continue
#                                 if ed + ee + ef > e3max_check:
#                                     continue
#                                 if (oa.l + ob.l + oc.l + od.l + oe.l + of.l) % 2 != 0:
#                                     continue

#                                 JdeMax = (od.j + oe.j) // 2
#                                 JdeMin = abs(od.j - oe.j) // 2
#                                 if abs(od.j - oe.j) > of.j:
#                                     twoJCMindownket = abs(od.j - oe.j) - of.j
#                                 elif of.j < (od.j + oe.j):
#                                     twoJCMindownket = 1
#                                 else:
#                                     twoJCMindownket = of.j - od.j - oe.j
#                                 twoJCMaxupket = od.j + oe.j + of.j

#                                 twoJCMindown = max(twoJCMindownbra, twoJCMindownket)
#                                 twoJCMaxup = min(twoJCMaxupbra, twoJCMaxupket)
#                                 if twoJCMindown > twoJCMaxup:
#                                     continue

#                                 for Jab in range(JabMin, JabMax + 1):
#                                     for Jde in range(JdeMin, JdeMax + 1):
#                                         twoJCMin = max(
#                                             abs(2 * Jab - oc.j), abs(2 * Jde - of.j)
#                                         )
#                                         twoJCMax = min(2 * Jab + oc.j, 2 * Jde + of.j)
#                                         if twoJCMin > twoJCMax:
#                                             continue
#                                         blocksize = ((twoJCMax - twoJCMin) // 2 + 1) * 5
#                                         count_ME_file += blocksize
#         print(f"count_ME (File) {count_ME_file}")
#         return count_ME_file

#     def read_me3jgz(self, filename, count_ME_file):
#         if not os.path.isfile(filename):
#             raise FileNotFoundError(f"File not found: {filename}")
#         size_ME = count_ME_file * 8 / 1024**3
#         total_memory = (
#             os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
#         )
#         if size_ME >= 0.9 * (total_memory / 2):
#             raise MemoryError(
#                 f"# of ThBME={size_ME} is beyond available memory: {total_memory} GB"
#             )
#         ThBME = np.zeros(count_ME_file, dtype=np.float64)
#         with gzip.open(filename, "rt") as stream:
#             for idx, line in enumerate(stream):
#                 if idx == 0:
#                     continue
#                 idx_i = (idx - 1) * 10
#                 idx_f = idx_i + 9
#                 subsize = 10
#                 if idx_i > count_ME_file:
#                     break
#                 if idx_f > count_ME_file:
#                     subsize = count_ME_file - idx_i + 1
#                 for i in range(subsize):
#                     tl = line[16 * i : 16 * (i + 1)]
#                     ThBME[idx_i + i] = float(tl)
#         print("Total ThBME entries read:", len(ThBME))
#         return ThBME

#     def allocate_3bme(self, sps_3b, ME_is_double=True):
#         norbits = sps_3b.norbits_ms
#         sps = sps_3b.sps
#         e1max = sps_3b.e1max
#         e3max = sps_3b.e3max

#         dict_3b_idx = {}
#         total_dim = 0
#         # Loop over indices (assumed 1-indexed and odd numbers)
#         for a in range(1, norbits + 1, 2):
#             oa = sps[a]
#             ea = oa.e
#             la = oa.l
#             if ea > e1max or ea > e3max:
#                 continue

#             for b in range(1, a + 1, 2):
#                 ob = sps[b]
#                 eb = ob.e
#                 lb = ob.l
#                 if ea + eb > e3max:
#                     continue

#                 Jab_min = abs(oa.j - ob.j) // 2
#                 Jab_max = (oa.j + ob.j) // 2

#                 for c in range(1, b + 1, 2):
#                     oc = sps[c]
#                     ec = oc.e
#                     if ea + eb + ec > e3max:
#                         continue

#                     for d in range(1, a + 1, 2):
#                         od = sps[d]
#                         ed = od.e

#                         # Upper limit for index e depends on whether d equals a
#                         upper_idx_e = b if d == a else d
#                         for e in range(1, upper_idx_e + 1, 2):
#                             oe = sps[e]
#                             ee = oe.e

#                             # Upper limit for index f depends on a,d and b,e combinations
#                             upper_idx_f = c if (a == d and b == e) else e
#                             for f in range(1, upper_idx_f + 1, 2):
#                                 of = sps[f]
#                                 ef = of.e
#                                 if ed + ee + ef > e3max:
#                                     continue

#                                 # Check parity condition from orbital angular momenta.
#                                 if (la + lb + oc.l + od.l + oe.l + of.l) % 2 != 0:
#                                     continue

#                                 # Record the current dimension offset using a hash key.
#                                 orbit_hash = get_nkey6(a, b, c, d, e, f)
#                                 dict_3b_idx[orbit_hash] = total_dim

#                                 Jde_min = abs(od.j - oe.j) // 2
#                                 Jde_max = (od.j + oe.j) // 2

#                                 for Jab in range(Jab_min, Jab_max + 1):
#                                     for Jde in range(Jde_min, Jde_max + 1):
#                                         J2_min = max(
#                                             abs(2 * Jab - oc.j), abs(2 * Jde - of.j)
#                                         )
#                                         J2_max = min(2 * Jab + oc.j, 2 * Jde + of.j)
#                                         for J2 in range(J2_min, J2_max + 1, 2):
#                                             total_dim += 5

#         # Determine the total memory required (in GB)
#         size_3bme = total_dim * (8 if ME_is_double else 4) / 1024.0**3
#         total_memory = psutil.virtual_memory().total / 1024**3
#         if size_3bme >= 0.9 * total_memory:
#             raise MemoryError(
#                 f"size(3BME) {size_3bme} is beyond your environment memory {total_memory} GB"
#             )
#         print(f"# of 3BME: {total_dim:12d} Mem. {size_3bme:12.5e} GB")

#         v3bme = np.zeros(total_dim, dtype=np.float64)
#         return v3bme, dict_3b_idx

#     def monopole_V3(self, E3max, sps_3b, dict_3b_idx, v3bme, dWS, pnME_3NF=None):
#         n_orbits = sps_3b.norbits_ms
#         sps = sps_3b.sps
#         stored_keys = []
#         # Build list of keys.
#         for i in range(1, n_orbits + 1):
#             oi = sps[i]
#             for j in range(i, n_orbits + 1):
#                 oj = sps[j]
#                 if oi.l != oj.l or oi.j != oj.j or oi.tz != oj.tz:
#                     continue
#                 for a in range(1, n_orbits + 1):
#                     oa = sps[a]
#                     for b in range(1, n_orbits + 1):
#                         ob = sps[b]
#                         if oa.l != ob.l or oa.j != ob.j or oa.tz != ob.tz:
#                             continue
#                         for c in range(1, n_orbits + 1):
#                             oc = sps[c]
#                             if oa.e + oc.e + oi.e > E3max:
#                                 continue
#                             for d in range(1, n_orbits + 1):
#                                 od = sps[d]
#                                 if oc.l != od.l or oc.j != od.j or oc.tz != od.tz:
#                                     continue
#                                 if ob.e + od.e + oj.e > E3max:
#                                     continue
#                                 # Check parity condition from orbital angular momenta.
#                                 if (oi.l + oa.l + ob.l + oc.l + od.l + oj.l) % 2 != 0:
#                                     continue
#                                 key = get_nkey6(a, c, i, b, d, j)
#                                 stored_keys.append(key)

#         nkeys = len(stored_keys)
#         print("Number of keys for V3mono:", nkeys)

#         # Initialize dictionary for monopole matrix elements.
#         Vmon3 = {key: 0.0 for key in stored_keys}

#         # Loop over keys (e.g., parallelize here if needed)
#         for idx, key in enumerate(stored_keys, start=1):
#             a, c, i, b, d, j = unhash_key6j(key)
#             assert get_nkey6(a, c, i, b, d, j) == key

#             ja = sps[a].j
#             jc = sps[c].j
#             ji = sps[i].j
#             jb = sps[b].j
#             jd = sps[d].j
#             jj = sps[j].j

#             j2min = max(abs(ja - jc), abs(jb - jd)) // 2
#             j2max = min(ja + jc, jb + jd) // 2
#             v = 0.0

#             for j2 in range(j2min, j2max + 1):
#                 Jmin = max(abs(2 * j2 - ji), abs(2 * j2 - jj))
#                 Jmax = 2 * j2 + min(ji, jj)
#                 for J2 in range(Jmin, Jmax + 1, 2):
#                     vtmp = self.get_V3_pn(
#                         idx,
#                         E3max,
#                         v3bme,
#                         j2,
#                         j2,
#                         J2,
#                         a,
#                         c,
#                         i,
#                         b,
#                         d,
#                         j,
#                         sps_3b,
#                         dict_3b_idx,
#                         dWS,
#                         pnME_3NF,
#                     ) * (J2 + 1)
#                     v += vtmp
#             Vmon3[key] += v / (ji + 1)
#         norm = np.sqrt(np.sum([float(v) ** 2 for v in list(Vmon3.values())]))
#         print(f"Monopole V3 norm: {norm:.5e}")
#         return Vmon3

#     def get_V3_pn(
#         self,
#         indx,
#         E3max,
#         v3bme,
#         Jab,
#         Jde,
#         J2,
#         a,
#         b,
#         c,
#         d,
#         e,
#         f,
#         sps_3b,
#         dict_3b_idx,
#         dWS,
#         pnME_3NF,
#     ):
#         tza = sps_3b.sps[a].tz
#         tzb = sps_3b.sps[b].tz
#         tzc = sps_3b.sps[c].tz
#         tzd = sps_3b.sps[d].tz
#         tze = sps_3b.sps[e].tz
#         tzf = sps_3b.sps[f].tz
#         dcg_spin = dWS.dcg_spin

#         Vpn = 0.0
#         Tmin = max(abs(tza + tzb + tzc), abs(tzd + tze + tzf))

#         start_Tab = int(abs(tza + tzb) // 2)
#         for Tab in range(start_Tab, 2):
#             key1 = get_nkey6_shift(1, tza, 1, tzb, Tab * 2, tza + tzb)
#             # print(f"in get_V3pn key1 {key1} tz_a {tza} tz_b {tzb} Tab {Tab} ")

#             CG1 = dcg_spin[key1]
#             # Loop over 'Tde': from div(abs(tzd+tze), 2) up to 1 (inclusive)
#             start_Tde = int(abs(tzd + tze) // 2)
#             # print(f"key1 {key1}, CG1 {CG1}")
#             for Tde in range(start_Tde, 2):
#                 key2 = get_nkey6_shift(1, tzd, 1, tze, Tde * 2, tzd + tze)
#                 CG2 = dcg_spin[key2]
#                 if CG1 * CG2 == 0:
#                     continue
#                 Tmax = min(1 + 2 * Tab, 1 + 2 * Tde)
#                 # Loop over T2 from Tmin to Tmax inclusive with step 2
#                 for T2 in range(int(Tmin), int(Tmax) + 1, 2):
#                     key3 = get_nkey6_shift(
#                         Tab * 2, tza + tzb, 1, tzc, T2, tza + tzb + tzc
#                     )
#                     CG3 = dcg_spin[key3]
#                     key4 = get_nkey6_shift(
#                         Tde * 2, tzd + tze, 1, tzf, T2, tzd + tze + tzf
#                     )
#                     CG4 = dcg_spin[key4]
#                     if CG3 * CG4 == 0:
#                         continue
#                     tbme = self.Get3BME_ISO(
#                         indx,
#                         E3max,
#                         v3bme,
#                         dict_3b_idx,
#                         sps_3b,
#                         Jab,
#                         Jde,
#                         J2,
#                         Tab,
#                         Tde,
#                         T2,
#                         a,
#                         b,
#                         c,
#                         d,
#                         e,
#                         f,
#                         dWS,
#                         pnME_3NF,
#                     )
#                     Vpn += float(CG1 * CG2 * CG3 * CG4) * tbme
#         return Vpn

#     def Get3BME_ISO(
#         self,
#         ind,
#         E3max,
#         v3bme,
#         dict_3b_idx,
#         sps_3b,
#         Jab_in,
#         Jde_in,
#         J2,
#         Tab_in,
#         Tde_in,
#         T2,
#         a_in,
#         b_in,
#         c_in,
#         d_in,
#         e_in,
#         f_in,
#         dWS,
#         pnME_3NF,
#         verbose=False,
#     ):
#         sps = sps_3b.sps
#         v = 0.0

#         a, b, c, idx_abc = sort_3_orbits(a_in, b_in, c_in)
#         d, e, f, idx_def = sort_3_orbits(d_in, e_in, f_in)

#         # If the second triple is larger then swap the roles.
#         if d > a or (d == a and e > b) or (d == a and e == b and f > c):
#             a, b, c, d, e, f = d, e, f, a, b, c
#             Jab_in, Jde_in = Jde_in, Jab_in
#             Tab_in, Tde_in = Tde_in, Tab_in
#             idx_abc, idx_def = idx_def, idx_abc

#         tkey = get_nkey6(a, b, c, d, e, f)
#         idx_3borbit = dict_3b_idx.get(tkey, -1)
#         if idx_3borbit == -1:
#             return 0.0

#         oa = sps[a]
#         ob = sps[b]
#         oc = sps[c]
#         od = sps[d]
#         oe = sps[e]
#         of_ = sps[f]  # 'of' is a Python keyword alternative

#         if (oa.e + ob.e + oc.e) > E3max:
#             return 0.0
#         if (od.e + oe.e + of_.e) > E3max:
#             return 0.0
#         aa = a // 2 + 1
#         bb = b // 2 + 1
#         cc = c // 2 + 1
#         dd = d // 2 + 1
#         ee = e // 2 + 1
#         ff = f // 2 + 1

#         ja2, jb2, jc2 = oa.j, ob.j, oc.j
#         jd2, je2, jf2 = od.j, oe.j, of_.j

#         Jab_min = abs(ja2 - jb2) // 2
#         Jab_max = (ja2 + jb2) // 2
#         Jde_min = abs(jd2 - je2) // 2
#         Jde_max = (jd2 + je2) // 2

#         Tab_min = 1 if T2 == 3 else 0
#         Tab_max = 1
#         Tde_min = 1 if T2 == 3 else 0
#         Tde_max = 1

#         J_index = 0
#         # count_inner is used for bookkeeping (unused later)
#         count_inner = 0

#         for Jab in range(Jab_min, Jab_max + 1):
#             Cj_abc = RecouplingCG(idx_abc, ja2, jb2, jc2, Jab_in, Jab, J2, dWS)
#             if (idx_abc // 3) != (idx_def // 3):
#                 Cj_abc *= -1.0
#             for Jde in range(Jde_min, Jde_max + 1):
#                 Cj_def = RecouplingCG(idx_def, jd2, je2, jf2, Jde_in, Jde, J2, dWS)
#                 J2_min = max(abs(2 * Jab - jc2), abs(2 * Jde - jf2))
#                 J2_max = min(2 * Jab + jc2, 2 * Jde + jf2)
#                 if J2_min > J2_max:
#                     continue

#                 J_index += ((J2 - J2_min) // 2) * 5
#                 count_inner += 1

#                 if J2_min <= J2 <= J2_max and abs(Cj_abc * Cj_def) > 1.0e-10:
#                     for Tab in range(Tab_min, Tab_max + 1):
#                         Ct_abc = RecouplingCG(idx_abc, 1, 1, 1, Tab_in, Tab, T2, dWS)
#                         for Tde in range(Tde_min, Tde_max + 1):
#                             Ct_def = RecouplingCG(
#                                 idx_def, 1, 1, 1, Tde_in, Tde, T2, dWS
#                             )

#                             if abs(Ct_abc * Ct_def) < 1.0e-10:
#                                 continue
#                             Tindex = 2 * Tab + Tde + ((T2 - 1) // 2)
#                             idx = idx_3borbit + J_index + Tindex
#                             v += Cj_abc * Cj_def * Ct_abc * Ct_def * v3bme[idx]
#                             # vtmp = ME = v3bme[idx]

#                             # if ME == 0.0:
#                             #     continue
#                             # # If you write out vtmp here, you can get something similar to readable.text fmt.
#                             # # Note that here we are generating bra-ket permutations, which are absent in NuHamil outputs.
#                             # if pnME_3NF is not None:
#                             #     # if not( (aa >= bb >= cc) and (dd >= ee >= ff) ):
#                             #     #     continue
#                             #     pnkey = (aa, bb, cc, Jab, Tab, dd, ee, ff, Jde, Tde, Jde, J2, T2)
#                             #     # if (aa, bb, cc) == (dd, ee, ff) == (1, 1, 1) and J2 == 1:
#                             #     #     print("pnkeyraw: ", pnkey, "ME", ME)
#                             #     orb_a = self.JTorbitals.orbitals[aa]; orb_b = self.JTorbitals.orbitals[bb]; orb_c = self.JTorbitals.orbitals[cc]
#                             #     orb_d = self.JTorbitals.orbitals[dd]; orb_e = self.JTorbitals.orbitals[ee]; orb_f = self.JTorbitals.orbitals[ff]

#                             #     for tz_a, tz_b, tz_c, tz_d, tz_e, tz_f in itertools.product([-1,1], repeat=6):
#                             #         if (tz_a+tz_b+tz_c) != (tz_d+tz_e+tz_f): continue
#                             #         if (tz_a+tz_b) > 2*Tab: continue
#                             #         if (tz_d+tz_e) > 2*Tde: continue
#                             #         if abs(tz_a+tz_b) > Tab or abs(tz_d+tz_e) > Tde: continue
#                             #         pn_a = get_spsidx_from_nljtz(self.single_particle_states, orb_a.n, orb_a.l, orb_a.j, tz_a)
#                             #         pn_b = get_spsidx_from_nljtz(self.single_particle_states, orb_b.n, orb_b.l, orb_b.j, tz_b)
#                             #         pn_c = get_spsidx_from_nljtz(self.single_particle_states, orb_c.n, orb_c.l, orb_c.j, tz_c)
#                             #         pn_d = get_spsidx_from_nljtz(self.single_particle_states, orb_d.n, orb_d.l, orb_d.j, tz_d)
#                             #         pn_e = get_spsidx_from_nljtz(self.single_particle_states, orb_e.n, orb_e.l, orb_e.j, tz_e)
#                             #         pn_f = get_spsidx_from_nljtz(self.single_particle_states, orb_f.n, orb_f.l, orb_f.j, tz_f)
#                             #         Tz_bra = tz_a + tz_b + tz_c
#                             #         if Tz_bra == 3 and self.N < 3: continue # nnn
#                             #         if Tz_bra == 1 and ( self.Z ==0 or self.N <= 1) : continue # pnn
#                             #         if Tz_bra == -1 and ( self.N ==0 or self.Z <= 1): continue # npp
#                             #         if Tz_bra == -3 and self.Z < 3: continue # ppp
#                             #         if abs(Tz_bra) > T2: continue
#                             #         pnkey = (pn_a, pn_b, pn_c, pn_d, pn_e, pn_f, Jab, Jde, J2)
#                             #         key1 = get_nkey6_shift(1, tz_a, 1, tz_b, Tab * 2, (tz_a + tz_b))
#                             #         key2 = get_nkey6_shift(1, tz_d, 1, tz_e, Tde * 2, (tz_d + tz_e))
#                             #         key3 = get_nkey6_shift(Tab * 2, (tz_a + tz_b), 1, tz_c, T2, (tz_a + tz_b + tz_c))
#                             #         key4 = get_nkey6_shift(Tde * 2, (tz_d + tz_e), 1, tz_f, T2, (tz_d + tz_e + tz_f))
#                             #         cg_1 = dWS.dcg_spin[key1]
#                             #         cg_2 = dWS.dcg_spin[key2]
#                             #         cg_3 = dWS.dcg_spin[key3]
#                             #         cg_4 = dWS.dcg_spin[key4]

#                             #         cgfact = cg_1 * cg_2 * cg_3 * cg_4
#                             #         if abs(cgfact) < 1.e-8:
#                             #             continue
#                             #         part = ME * cgfact
#                             #         if pnkey in pnME_3NF.keys():
#                             #             pnME_3NF[pnkey] += part
#                             #         else:
#                             #             pnME_3NF[pnkey] = part

#                 J_index += ((J2_max - J2 + 2) // 2) * 5
#         return v


# def truncate_v3bme(dim_v3bme, sps_3b, ThBME, nreads_v3bme, dWS, dict_idxThBME):
#     v3bme = np.zeros(dim_v3bme, dtype=np.float64)
#     e1max = sps_3b.e1max
#     e2max = e1max * 2
#     e1max_file = sps_3b.e1max_file
#     e2max_file = sps_3b.e2max_file
#     e3max_file = sps_3b.e3max_file
#     e3max = sps_3b.e3max
#     l3max = e1max
#     sps = sps_3b.sps
#     norbits = sps_3b.norbits
#     count_ME_file = 0

#     # Loop over 'bra' indices (odd numbers from 1 to norbits)
#     for idx_a in range(1, norbits + 1, 2):
#         oa = sps[idx_a]
#         ea = oa.e
#         # nreads_v3bme index conversion: Julia's div(idx_a,2)+1 -> Python's idx_a//2
#         nread_v3bme = nreads_v3bme[idx_a // 2]
#         if ea > e1max:
#             continue

#         for idx_b in range(1, idx_a + 1, 2):
#             ob = sps[idx_b]
#             eb = ob.e
#             if ea + eb > e2max:
#                 continue

#             for idx_c in range(1, idx_b + 1, 2):
#                 oc = sps[idx_c]
#                 ec = oc.e
#                 if ea + eb + ec > e3max:
#                     continue

#                 JabMax = (oa.j + ob.j) // 2
#                 JabMin = abs(oa.j - ob.j) // 2
#                 if abs(oa.j - ob.j) > oc.j:
#                     twoJCMindownbra = abs(oa.j - ob.j) - oc.j
#                 elif oc.j < (oa.j + ob.j):
#                     twoJCMindownbra = 1
#                 else:
#                     twoJCMindownbra = oc.j - oa.j - ob.j
#                 twoJCMaxupbra = oa.j + ob.j + oc.j

#                 # Loop for 'ket' part
#                 for idx_d in range(1, idx_a + 1, 2):
#                     od = sps[idx_d]
#                     ed = od.e
#                     if ed > e1max:
#                         continue
#                     upper_idx_e = idx_b if (idx_a == idx_d) else idx_d
#                     for idx_e in range(1, upper_idx_e + 1, 2):
#                         oe = sps[idx_e]
#                         ee = oe.e
#                         if ee > e1max:
#                             continue
#                         idx_f_max = (
#                             idx_c if (idx_a == idx_d and idx_b == idx_e) else idx_e
#                         )
#                         for idx_f in range(1, idx_f_max + 1, 2):
#                             of_ = sps[idx_f]
#                             ef = of_.e
#                             if ef > e1max:
#                                 continue
#                             if ed + ee + ef > e3max:
#                                 continue
#                             if (oa.l + ob.l + oc.l + od.l + oe.l + of_.l) % 2 != 0:
#                                 continue

#                             if not valid_check(
#                                 ea, eb, ec, ed, ee, ef, e1max, e2max, e3max
#                             ):
#                                 continue
#                             if not valid_check(
#                                 ea,
#                                 eb,
#                                 ec,
#                                 ed,
#                                 ee,
#                                 ef,
#                                 e1max_file,
#                                 e2max_file,
#                                 e3max_file,
#                             ):
#                                 continue

#                             JdeMax = (od.j + oe.j) // 2
#                             JdeMin = abs(od.j - oe.j) // 2
#                             if abs(od.j - oe.j) > of_.j:
#                                 twoJCMindownket = abs(od.j - oe.j) - of_.j
#                             elif of_.j < (od.j + oe.j):
#                                 twoJCMindownket = 1
#                             else:
#                                 twoJCMindownket = of_.j - od.j - oe.j
#                             twoJCMaxupket = od.j + oe.j + of_.j

#                             twoJCMindown = max(twoJCMindownbra, twoJCMindownket)
#                             twoJCMaxup = min(twoJCMaxupbra, twoJCMaxupket)
#                             if twoJCMindown > twoJCMaxup:
#                                 continue

#                             key = get_nkey6(idx_a, idx_b, idx_c, idx_d, idx_e, idx_f)
#                             offset_ThBME = dict_idxThBME[key]
#                             idx_ThBME = offset_ThBME

#                             for Jab in range(JabMin, JabMax + 1):
#                                 for Jde in range(JdeMin, JdeMax + 1):
#                                     twoJCMin = max(
#                                         abs(2 * Jab - oc.j), abs(2 * Jde - of_.j)
#                                     )
#                                     twoJCMax = min(2 * Jab + oc.j, 2 * Jde + of_.j)
#                                     if twoJCMin > twoJCMax:
#                                         continue
#                                     blocksize = ((twoJCMax - twoJCMin) // 2 + 1) * 5
#                                     for JTind in range(0, twoJCMax - twoJCMin + 1):
#                                         twoJC = twoJCMin + (JTind // 2) * 2
#                                         twoT = 1 + (JTind % 2) * 2
#                                         for Tab in range(0, 2):
#                                             for Tde in range(0, 2):
#                                                 if twoT > min(2 * Tab + 1, 2 * Tde + 1):
#                                                     continue
#                                                 index_ab = (
#                                                     ((5 * (twoJC - twoJCMin)) // 2)
#                                                     + 2 * Tab
#                                                     + Tde
#                                                     + ((twoT - 1) // 2)
#                                                 )
#                                                 v3idx = nread_v3bme + index_ab + 1
#                                                 idx_ThBME += 1
#                                                 ThBME_idx = idx_ThBME
#                                                 V = 0.0
#                                                 autozero = False
#                                                 if (
#                                                     oa.l > l3max
#                                                     or ob.l > l3max
#                                                     or oc.l > l3max
#                                                     or od.l > l3max
#                                                     or oe.l > l3max
#                                                     or of_.l > l3max
#                                                 ):
#                                                     V = 0.0
#                                                 v3bme[v3idx] = ThBME[ThBME_idx]
#                                                 if (
#                                                     idx_a == idx_b
#                                                     and (Tab + Jab) % 2 == 0
#                                                 ) or (
#                                                     idx_d == idx_e
#                                                     and (Tde + Jde) % 2 == 0
#                                                 ):
#                                                     autozero = True
#                                                 if (
#                                                     idx_a == idx_b
#                                                     and idx_a == idx_c
#                                                     and twoT == 3
#                                                     and oa.j < 3
#                                                 ):
#                                                     autozero = True
#                                                 if (
#                                                     idx_d == idx_e
#                                                     and idx_d == idx_f
#                                                     and twoT == 3
#                                                     and od.j < 3
#                                                 ):
#                                                     autozero = True
#                                     if valid_check(
#                                         ea, eb, ec, ed, ee, ef, e1max, e2max, e3max
#                                     ):
#                                         nread_v3bme += blocksize
#                                     count_ME_file += blocksize


# def RecouplingCG(idx_abc, ja2, jb2, jc2, Jab_in, Jab, J2, dWS) -> float:
#     # Check angular momentum triangle conditions
#     if abs(ja2 - jb2) // 2 > Jab or (ja2 + jb2) // 2 < Jab:
#         return 0.0
#     if abs(jc2 - J2) // 2 > Jab or (jc2 + J2) // 2 < Jab:
#         return 0.0

#     if idx_abc == 0:
#         return 1.0 if Jab == Jab_in else 0.0

#     elif idx_abc == 1:  # bca
#         phase = (-1) ** (((jb2 + jc2) // 2) + Jab_in + 1)
#         t6j = dWS.d6j_lj[get_key6j_sym(ja2, jb2, Jab * 2, jc2, J2, Jab_in * 2)]
#         return phase * hat(Jab_in) * hat(Jab) * t6j

#     elif idx_abc == 2:  # cab
#         phase = (-1) ** (((ja2 + jb2) // 2) - Jab + 1)
#         t6j = dWS.d6j_lj[get_key6j_sym(jb2, ja2, Jab * 2, jc2, J2, Jab_in * 2)]
#         return phase * hat(Jab_in) * hat(Jab) * t6j

#     elif idx_abc == 3:  # acb
#         phase = (-1) ** (((jb2 + jc2) // 2) + Jab_in - Jab)
#         t6j = dWS.d6j_lj[get_key6j_sym(jb2, ja2, Jab * 2, jc2, J2, Jab_in * 2)]
#         return phase * hat(Jab_in) * hat(Jab) * t6j

#     elif idx_abc == 4:  # bac
#         if Jab == Jab_in:
#             phase = (-1) ** (((ja2 + jb2) // 2) - Jab)
#             return 1.0 * phase
#         else:
#             return 0.0

#     elif idx_abc == 5:  # cba
#         t6j = dWS.d6j_lj[get_key6j_sym(ja2, jb2, Jab * 2, jc2, J2, Jab_in * 2)]
#         return -hat(Jab_in) * hat(Jab) * t6j

#     else:
#         raise AssertionError("This should not happen")


# class prep_dicts_for_WignerSymbols:
#     def __init__(self, emax):
#         self.emax = emax
#         self.jmax = 2 * emax + 1
#         self.d6j_int = self.prep_d6j_int(emax, self.jmax)
#         self.dcg_spin = self.prep_dcg_spin()
#         self.d6j_lj = self.prep_d6j_lj(self.jmax)

#     def prep_dcg_spin(self):
#         dcg_spin = {}
#         s_a = 1
#         s_b = 1
#         for sz_a in [-1, 1]:
#             for sz_b in [-1, 1]:
#                 for Sab in [0, 1]:
#                     Sabzmin = 0 if Sab == 0 else -1
#                     for Sab_z in range(Sabzmin, Sab + 1):
#                         nkey = get_nkey6_shift(s_a, sz_a, s_b, sz_b, Sab * 2, Sab_z * 2)
#                         dcg_spin[nkey] = float(
#                             ClebschGordan(
#                                 s_a / 2, sz_a / 2, s_b / 2, sz_b / 2, Sab, Sab_z
#                             ).doit()
#                         )
#         s_c = 1
#         for S_ab in range(0, 2):  # 0, 1
#             for S_ab_z in range(-S_ab, S_ab + 1):
#                 for s_c_z in range(-1, 2, 2):  # -1, 1
#                     S3min = 1
#                     S3max = 1 if S_ab == 0 else 3
#                     for S3 in range(S3min, S3max + 1, 2):
#                         for S3z in range(-S3, S3 + 1, 2):
#                             nkey = get_nkey6_shift(
#                                 S_ab * 2, S_ab_z * 2, s_c, s_c_z, S3, S3z
#                             )
#                             dcg_spin[nkey] = float(
#                                 ClebschGordan(
#                                     S_ab, S_ab_z, s_c / 2, s_c_z / 2, S3 / 2, S3z / 2
#                                 ).doit()
#                             )
#         return dcg_spin

#     def prep_d6j_lj(self, jmax2):
#         d6j_lj = {}
#         for j1 in range(1, jmax2 + 1, 2):
#             for j2 in range(1, jmax2 + 1, 2):
#                 for J12 in range(abs(j1 - j2), j1 + j2 + 1, 2):
#                     for j3 in range(1, jmax2 + 1, 2):
#                         for J23 in range(abs(j2 - j3), j2 + j3 + 1, 2):
#                             start_J = max(abs(j1 - J23), abs(j3 - J12))
#                             end_J = min(j1 + J23, j3 + J12)
#                             for J in range(start_J, end_J + 1, 2):
#                                 nkey = get_key6j_sym(j1, j2, J12, j3, J, J23)
#                                 d6j_lj[nkey] = wigner_6j(
#                                     j1 / 2, j2 / 2, J12 / 2, j3 / 2, J / 2, J23 / 2
#                                 )
#         # Special case for kinetic_tb
#         for j2 in range(1, jmax2 + 1, 2):
#             J12 = 2
#             J23 = 1
#             for j1 in range(abs(j2 - J12), j2 + J12 + 1, 2):
#                 for l1 in range(abs(j1 - 1), j1 + 1 + 1, 2):
#                     for l2 in range(abs(j2 - 1), j2 + 1 + 1, 2):
#                         nkey = get_key6j_sym(j2, j1, J12, l1, l2, J23)
#                         d6j_lj[nkey] = wigner_6j(
#                             j2 / 2, j1 / 2, J12 / 2, l1 / 2, l2 / 2, J23 / 2
#                         )
#         return d6j_lj

#     def prep_d6j_int(self, emax, jmax_in):
#         # 'to' parameter is not used in this function.
#         jmax = jmax_in * 2
#         d6j_int = {}

#         for J12 in range(0, jmax + 1, 2):
#             for j1 in range(0, jmax_in + 1, 2):
#                 for j2 in range(abs(J12 - j1), j1 + J12 + 1, 2):
#                     for j3 in range(0, jmax + 1, 2):
#                         for J23 in range(abs(j2 - j3), j2 + j3 + 1, 2):
#                             for J in range(abs(J23 - j1), J23 + j1 + 1, 2):
#                                 # Check the triangle condition. Note the division by 2.
#                                 if not tri_check(J / 2, J12 / 2, j3 / 2):
#                                     continue
#                                 # Check the inequality condition.
#                                 if not (j1 + j3 <= j2 + J <= J12 + J23):
#                                     continue
#                                 # Compute the key.
#                                 nkey = get_key6j_sym(j1, j2, J12, j3, J, J23)
#                                 # Compute the six-j symbol value.
#                                 value = wigner_6j(
#                                     j1 / 2, j2 / 2, J12 / 2, j3 / 2, J / 2, J23 / 2
#                                 )
#                                 d6j_int[nkey] = value
#         return d6j_int


# def tri_check(a, b, c):
#     if a + b < c or a + c < b or b + c < a:
#         return False
#     if abs(a - b) > c or abs(a - c) > b or abs(b - c) > a:
#         return False
#     return True


# def get_key6j_sym(j1: int, j3: int, j5: int, j2: int, j4: int, j6: int) -> int:
#     # Initialize temporary copies.
#     tj1, tj3, tj5 = j1, j3, j5
#     tj2, tj4, tj6 = j2, j4, j6
#     # Assume get_canonical_order_6j is implemented elsewhere.
#     column_order = get_canonical_order_6j(j1, j2, j3, j4, j5, j6)
#     if column_order == 231:
#         tj1, tj2, tj3, tj4, tj5, tj6 = tj3, tj4, tj5, tj6, tj1, tj2
#     elif column_order == 132:
#         tj3, tj4, tj5, tj6 = tj5, tj6, tj3, tj4
#     elif column_order == 213:
#         tj1, tj2, tj3, tj4 = tj3, tj4, tj1, tj2
#     elif column_order == 312:
#         tj1, tj2, tj3, tj4, tj5, tj6 = tj5, tj6, tj1, tj2, tj3, tj4
#     elif column_order == 321:
#         tj1, tj2, tj5, tj6 = tj5, tj6, tj1, tj2

#     # If any column has equal entries then swap to enforce tjX <= tjY.
#     if tj1 == tj2 or tj3 == tj4 or tj5 == tj6:
#         if tj1 > tj2:
#             tj1, tj2 = tj2, tj1
#         if tj3 > tj4:
#             tj3, tj4 = tj4, tj3
#         if tj5 > tj6:
#             tj5, tj6 = tj6, tj5
#         return get_nkey6(tj1, tj3, tj5, tj2, tj4, tj6)
#     else:
#         tint = (
#             (1 if tj1 < tj2 else 0) + (1 if tj3 < tj4 else 0) + (1 if tj5 < tj6 else 0)
#         )
#         if tint == 0 or tint == 3:
#             return get_nkey6(tj1, tj3, tj5, tj2, tj4, tj6)
#         if (tj5 < tj6 and tint == 1) or (tj5 > tj6 and tint == 2):
#             return get_nkey6(tj2, tj4, tj5, tj1, tj3, tj6)
#         elif (tj3 < tj4 and tint == 1) or (tj3 > tj4 and tint == 2):
#             return get_nkey6(tj2, tj3, tj6, tj1, tj4, tj5)
#         elif (tj1 < tj2 and tint == 1) or (tj1 > tj2 and tint == 2):
#             return get_nkey6(tj1, tj4, tj6, tj2, tj3, tj5)
#     raise RuntimeError("This never happens.")


# def j_col_score(j1: int, j2: int) -> int:
#     return 100 * (j1 + j2) + min(j1, j2)


# def get_canonical_order_6j(j1: int, j2: int, j3: int, j4: int, j5: int, j6: int) -> int:
#     cscore_12 = j_col_score(j1, j2)
#     cscore_34 = j_col_score(j3, j4)
#     cscore_56 = j_col_score(j5, j6)

#     if cscore_12 <= cscore_34:
#         if cscore_34 <= cscore_56:
#             return 123
#         elif cscore_56 < cscore_12:
#             return 312
#         else:
#             return 132
#     else:
#         if cscore_56 <= cscore_34:
#             return 321
#         elif cscore_12 <= cscore_56:
#             return 213
#         else:
#             return 231
