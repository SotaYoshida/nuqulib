"""Nuclear Hamiltonian construction and manipulation module.

This module provides functionality for constructing and manipulating nuclear
Hamiltonians for quantum many-body calculations. It supports both valence-space 
and no-core shell model (NCSM) interactions in the standard snt format used by
nuclear physics codes such as KSHELL, NuHamil, and NuclearToolkit.jl.

The module handles:
- Two-body and three-body nuclear force interactions
- M-scheme and JT-coupled representations
- Mapping to fermionic operators and Pauli strings for quantum computing
- Matrix element calculations and transformations
- Quantum resource estimation for nuclear Hamiltonians

Key Classes:
- Hamiltonian: Main class for nuclear Hamiltonian construction and manipulation
"""

import copy
from collections import Counter, defaultdict
import gzip
import itertools
import multiprocessing
from multiprocessing import get_context
import numpy as np
import os
import psutil
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from sympy.physics.quantum.cg import CG as ClebschGordan
from sympy.physics.wigner import wigner_6j
import time
from tqdm import tqdm
from .encoding import *
from .myutils import Orbit_nlj, Orbit_nljjztz, Orbit_nljtz, get_spsidx_from_nljtz


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


class JTcoupledOrbitals:
    """Class to handle JT-coupled orbitals in the model space.
    
    This class manages orbital states in the JT (angular momentum - isospin) coupled basis.
    It provides methods to add orbitals and convert between different single-particle state
    representations.
    
    Attributes:
        emax (int): Maximum excitation energy for orbitals.
        orbitals (dict): Dictionary mapping orbital indices to Orbit_nlj objects.
        dict_sps2JTorbitals (dict): Mapping from single-particle states to JT orbitals.
    """
    
    def __init__(self, emax):
        """Initialize JT-coupled orbitals manager.
        
        Args:
            emax (int): maximum value for 2n+l quanta in the model space
        """
        self.emax = emax
        self.orbitals = {}
        self.dict_sps2JTorbitals = {}

    def add_orbital(self, idx, n, l, j):
        """Add an orbital to the JT-coupled basis.
        
        Args:
            idx (int): Index identifier for the orbital.
            n (int): Principal quantum number (radial).
            l (int): Orbital angular momentum quantum number.
            j (int): Total angular momentum quantum number (twice the actual value).
        """
        self.orbitals[idx] = Orbit_nlj(n, l, j)

    def get_dict_sps2JTsps(self, sps_defined_in_NN):
        """Convert single particle states to JT-coupled states.
        
        Maps the single particle states defined in the nuclear interaction files
        to the JT-coupled orbital basis for consistent matrix element handling.
        
        Args:
            sps_defined_in_NN (list): Single particle states from NN interaction file.
            
        Returns:
            dict: Mapping from single-particle state indices to JT orbital indices.
            
        Note:
            The single-particle states are defined when the NN interaction file is loaded.
            This method ensures compatibility between different representations.
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


def get_Hamiltonian(fn_NN, Z: int, N: int,
                    fn_3NF=None, emax: int=20, e3max: int=0,
                    ncsm=False,
                    mapping_method="Jordan-Wigner",
                    single_spiecies=0):
    """Get nuclear Hamiltonian from interaction files.

    This is a convenience wrapper function that constructs a Hamiltonian object
    and returns the mapped quantum operators ready for use in quantum algorithms.
    It handles both two-body and three-body nuclear interactions.

    Args:
        fn_NN (str): Path to the snt format interaction file.
        Z (int): Number of protons in the nucleus.
        N (int): Number of neutrons in the nucleus.
        fn_3NF (str, optional): Path to three-body force file. Defaults to "".
        emax (int, optional): Maximum excitation energy for truncation. Defaults to 20.
        e3max (int, optional): Maximum excitation energy for three-body forces. Defaults to 0.
        ncsm (bool, optional): Whether using no-core shell model. Defaults to False.
        mapping_method (str, optional): The mapping method for converting fermionic operators to Pauli operators. Defaults to "Jordan-Wigner".
        single_spiecies (int, optional): Select a single-species calculation mode.
            ``0`` keeps both proton and neutron states, ``1`` keeps protons only,
            and ``2`` keeps neutrons only.

    Returns:
        tuple: A tuple containing:
            - hamil (Hamiltonian): The Hamiltonian object
            - H_mapped (SparsePauliOp): Mapped Hamiltonian as Pauli operators
            - proton_qubits (list): Qubit indices corresponding to proton states
            - neutron_qubits (list): Qubit indices corresponding to neutron states

    Example:
        >>> hamil, H_mapped, p_qubits, n_qubits = get_Hamiltonian("interaction.snt", 8, 8)
        >>> print(f"Total qubits: {len(p_qubits) + len(n_qubits)}")
    """
    assert mapping_method == "Jordan-Wigner", f"Only Jordan-Wigner mapping is implemented, but {mapping_method} was specified."

    if fn_3NF != None:
        hamil = Hamiltonian(fn_NN, Z, N, ncsm=True, emax_truncate=emax, e3max=e3max, fn_3NF=fn_3NF, mapping_method=mapping_method, single_spiecies=single_spiecies)
    else:
        hamil = Hamiltonian(fn_NN, Z, N, ncsm=ncsm, emax_truncate=emax, mapping_method=mapping_method, single_spiecies=single_spiecies)

    n_qubits = hamil.n_qubits
    proton_qubits = list(range(0, hamil.n_qubits_p))
    neutron_qubits = list(range(hamil.n_qubits_p, n_qubits))

    Hdict_M = hamil.get_mscheme_H(opform=True)
    H_1b_p, H_1b_n, H_jz_p, H_jz_n, H_pp, H_nn, H_pn, H_3b = hamil.mapping_opform(mapping_method)

    Hamil_ShellModel = 0 * H_1b_p 
    if hamil.vZ > 0:
        Hamil_ShellModel += H_1b_p 
    if hamil.vN > 0:
        Hamil_ShellModel += H_1b_n
    if hamil.vZ > 1:
        Hamil_ShellModel += H_pp 
    if hamil.vN > 1:
        Hamil_ShellModel += H_nn
    if hamil.vZ > 0 and hamil.vN > 0:
        Hamil_ShellModel += H_pn

    if fn_3NF != None:
        Hamil_ShellModel += H_3b

    Hamil_ShellModel = Hamil_ShellModel.simplify()
    return hamil, Hamil_ShellModel, proton_qubits, neutron_qubits


class Hamiltonian:
    """Nuclear Hamiltonian constructor and manager for quantum many-body calculations.
    
    This class provides comprehensive functionality for constructing nuclear Hamiltonians
    from standard interaction files, supporting both valence-space and no-core shell
    model (NCSM) calculations. It handles two-body and three-body nuclear forces,
    performs basis transformations, and maps to quantum operator representations.
    
    The class reads interaction files in the snt format used by nuclear physics
    codes (KSHELL, NuHamil, NuclearToolkit.jl) and provides methods for:
    - Single-particle basis management
    - Matrix element calculations and transformations
    - M-scheme and JT-coupled representations
    - Mapping to fermionic operators and Pauli strings
    - Three-body force handling and optimization
    
    Attributes:
        fn_NN (str): Path to two-body interaction file.
        Z (int): Number of protons.
        N (int): Number of neutrons.
        Anum (int): Mass number (Z + N).
        nucleus (str): Nuclear symbol (e.g., "16O").
        emax (int): Maximum excitation energy for truncation.
        ncsm (bool): Whether using no-core shell model.
        hw (float): Harmonic oscillator frequency (for NCSM).
        n_qubits (int): Total number of qubits needed.
        n_qubits_p (int): Number of proton qubits.
        n_qubits_n (int): Number of neutron qubits.
        msps (list): M-scheme single-particle states.
        v1b (list): One-body matrix elements.
        v2b (list): Two-body matrix elements.
        v3b_pn (dict): Three-body matrix elements in p-n representation.
        v3b_Mscheme (dict): Three-body matrix elements in M-scheme.
    
    Example:
        >>> # For valence-space calculation
        >>> hamil = Hamiltonian("interaction.snt", Z=8, N=8)
        >>> H_dict = hamil.get_mscheme_H(opform=True)
        >>> 
        >>> # For NCSM with three-body forces
        >>> hamil = Hamiltonian("ncsm_int.snt", Z=4, N=4, 
        ...                     fn_3NF="3nf.me3j.gz", ncsm=True, emax_truncate=2)
    
    References:
        - T. Miyagi et al., Eur. Phys. J. A 59, 150 (2023)
        - Standard snt format used by KSHELL and NuclearToolkit.jl for nuclear interactions
    """

    def __init__(
        self,
        fn_NN: str|os.PathLike,
        Z: int,
        N: int,
        jz: int|None=None,
        fn_3NF: str|os.PathLike|None=None,
        ncsm: bool=False,
        emax_truncate: int=5,
        Qiskit_order: bool=True,
        verbose: bool=False,
        e3max: int|None=None,
        mapping_method: str="Jordan-Wigner",
        single_spiecies: int=0,
    ):
        """Initialize nuclear Hamiltonian from interaction files.
        
        Args:
            fn_NN (str or PathLike): Path to two-body nuclear interaction file in snt format.
            Z (int): Number of protons in the nucleus.
            N (int): Number of neutrons in the nucleus.
            jz (int or None, optional): Total Jz for the M-scheme basis. If None, no Jz constraint is applied. Defaults to None.
            fn_3NF (str, PathLike, or None, optional): Path to three-body force file.
                Supports "readable.txt" and "me3j.gz" formats. Defaults to None.
            ncsm (bool, optional): Whether to use no-core shell model. If True,
                includes kinetic energy and center-of-mass corrections. Defaults to False.
            emax_truncate (int, optional): Maximum excitation energy for model space
                truncation. Defaults to 5.
            Qiskit_order (bool, optional): Whether to use Qiskit qubit ordering
                convention. Defaults to True.
            verbose (bool, optional): Enable verbose output for debugging. Defaults to False.
            e3max (int, optional): Maximum excitation energy for three-body forces.
                If None, uses emax_truncate value. Defaults to None.
            mapping_method (str, optional): Method for mapping fermionic operators to Pauli operators.
            single_spiecies (int, optional): If 0, treat protons and neutrons, if 1/2, treat only protons/neutrons. Defaults to 0.
        
        Raises:
            ValueError: If three-body file format is not supported.
            FileNotFoundError: If interaction files are not found.
            
        Note:
            For NCSM calculations, the harmonic oscillator frequency is extracted
            from the filename. Three-body forces require careful truncation
            parameter matching between 2N and 3N interactions.
        """
        self.channel = {-3: "ppp", -1: "ppn", 1: "pnn", 3: "nnn"}
        self.fn_NN = fn_NN
        self.Z = Z
        self.N = N
        self.Anum = Z + N
        self.nucleus = str(self.Anum) + element[Z]
        self.emax = emax_truncate
        self.verbose = verbose
        self.ncsm = ncsm
        if ncsm:
            self._emax_check(self.emax)
        self.Qiskit_order = Qiskit_order
        self.hw = None
        self.fn_3NF = fn_3NF
        self.Jz = jz
        self.mapping_method = mapping_method
        self.single_spiecies = single_spiecies
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

        self.Hamildict = None
        if ncsm:
            self.e1max_file, self.e2max, self.e3max_file = self.guess_emax_from_fn(
                self.fn_3NF
            )
        if (ncsm or fn_3NF != None) and e3max is None:
            print(
                f"Warning: e3max is not set, then set as {self.e3max_file} ",
                "If you don't like it, you must specify it."
            )
            self.e3max = self.e3max_file

        self.CG_dict = {}
        self.v3b_Mscheme = {}

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
                    self.CG_dict,
                    self.single_spiecies !=0 ,
                    verbose=self.verbose
                )
                self.v3b_pn = obj.pnME_3NF
                # Keep the JT reader data as well as the legacy pn dictionary.
                # The latter is convenient for the old conversion path, while
                # the former is required when evaluating arbitrary m-scheme
                # orderings through the 3NF recoupling formula.
                self._3nf_reader = obj
                self.v3bme = obj.v3bme
                self.sps_3b = obj.sps_3b
                self.dWs_3nf = obj.dWs
                self.dict_3b_idx = obj.dict_3b_idx
            else:
                print("fn_3NF is now: ", self.fn_3NF)
                raise ValueError(
                    "In the current implementation fn_3NF should be in either readable.text or me3j.gz file format"
                )

    def _emax_check(self, emax, emax_acceptable=5):
        if emax >= emax_acceptable:
            print(f"Warning: emax ({emax}) is greater than or equal to the acceptable limit ({emax_acceptable}).")
            print(f"You should know emax=4 (5) corresponds to 140 (232) qubits in total")
            print(f"If you are sure about this value, you can proceed by editing the source code of NuQuLib")

    def guess_emax_from_fn(self, fn_3NF):
        """Extract the model space information from three-body force filename.
        
        Attempts to parse the e1max/e2max/e3max for one-body, two-body,
        and three-body sectors from the three-body force filename pattern.
        
        Args:
            fn_3NF (str): Path to three-body force file.
            
        Returns:
            tuple: A tuple containing (e1max_file, e2max_file, e3max_file).
                If parsing fails, returns default values based on emax_truncate.
                
        Note:
            Expects filename pattern with "_ms{e1max}_{e2max}_{e3max}" substring.
            Falls back to emax_truncate * {1, 2, 3} if parsing fails.
        """
        e1max_file = e2max_file = e3max_file = None
        if fn_3NF is not None and "_ms" in fn_3NF:
            txt = fn_3NF.split("_ms")[-1].split(".")[0].split("_")
            e1max_file, e2max_file, e3max_file = list(map(int, txt))
        if e1max_file is None or e2max_file is None or e3max_file is None:
            if fn_3NF is not None:
                print(
                    "Warning: e1max_file, e2max, e3max_file are not set from the file name. ",
                    "For now, we used emax_truncate value for them multiplied by 1/2/3",
                )
            e1max_file = self.emax
            e2max_file = self.emax * 2
            e3max_file = self.emax * 3
        return e1max_file, e2max_file, e3max_file

    def neutron_number_constraint(self, N0, mapping_method, filepath='./tmp'):
        """Build a quadratic neutron-number penalty operator.

        Args:
            N0 (int | float): Target neutron number.
            mapping_method (str): Fermion-to-qubit mapping method.
            filepath (str | os.PathLike, optional): Base path used by mappings
                that load or save mapper data.

        Returns:
            SparsePauliOp: Qubit operator for ``(N_n - N0)^2``.
        """
        if self.Hamildict is None:
            _=self.get_mscheme_H(opform=True)
        V3p, V3n = self.get_V3_p_n()
        H_dict_n = self.Hamildict['SPE']['n'] | self.Hamildict['Vnn'] | V3n
        H_n = mapping_to_Pauli_string(
            FermionicOp(self.Hamildict["SPE"]['n'], 
                        num_spin_orbitals=self.n_qubits_n),
                        self.n_qubits, self.n_qubits_p,
                        method=mapping_method, 
                        Hamildict_specified = H_dict_n, filepath=filepath+'_n')
        op=SparsePauliOp('I'*self.n_qubits,coeffs=N0**2) + H_n.compose(H_n) - 2*N0*H_n
        return op.simplify()

    def proton_number_constraint(self, N0, mapping_method, filepath='./tmp'):
        """Build a quadratic proton-number penalty operator.

        Args:
            N0 (int | float): Target proton number.
            mapping_method (str): Fermion-to-qubit mapping method.
            filepath (str | os.PathLike, optional): Base path used by mappings
                that load or save mapper data.

        Returns:
            SparsePauliOp: Qubit operator for ``(N_p - N0)^2``.
        """
        # H_p = mapping_to_Pauli_string(
        #     FermionicOp(Hamildict_opform["P"],num_spin_orbitals=self.n_qubits),
        #     self.n_qubits, method=mapping_method, 
        #     Hamildict_opform=Hamildict_opform, 
        #     filepath=filepath)
        # op=N0**2*SparsePauliOp('I'*self.n_qubits)
        # for pauli in H_p:
        #     op += pauli.dot(pauli)
        #     op -= 2*N0*pauli
        if self.Hamildict is None:
            _=self.get_mscheme_H(opform=True)
        V3p, V3n = self.get_V3_p_n()
        H_dict_p = self.Hamildict['SPE']['p'] | self.Hamildict['Vpp'] | V3p
        H_p = mapping_to_Pauli_string(FermionicOp(self.Hamildict['SPE']["p"], num_spin_orbitals=self.n_qubits_p), self.n_qubits, 0, \
                                      method=mapping_method, Hamildict_specified = H_dict_p, filepath=filepath+'_p')
        op=SparsePauliOp('I'*self.n_qubits,coeffs=N0**2) + H_p.compose(H_p) - 2*N0*H_p
        return op.simplify()

    def extract_hw(self):
        """Extract harmonic oscillator frequency from NCSM interaction filename.
        
        Parses the harmonic oscillator frequency (hw) parameter from the filename
        of NCSM interaction files, which typically contain "hw{value}" in the name.
        
        Returns:
            float: Harmonic oscillator frequency in MeV.
            
        Note:
            Required for NCSM calculations to properly scale kinetic energy terms.
        """
        txt = self.fn_NN.split("hw")[-1]
        hw = txt.split("_")[0]
        return float(hw)

    def read_snt_file(self, fn_NN_in):  
        """Read nuclear interaction file in snt format.
        
        Parses nuclear interaction files containing single-particle energies and
        two-body matrix elements in the standard snt format used by nuclear
        physics codes like KSHELL and NuclearToolkit.jl.
        
        Args:
            fn_NN_in (str or PathLike): Path to the snt format interaction file.
            
        Returns:
            tuple: A tuple containing:
                - nsp_p (int): Number of proton single-particle states
                - nsp_n (int): Number of neutron single-particle states  
                - core_p (int): Number of core protons
                - core_n (int): Number of core neutrons
                - single_particle_states (list): List of Orbit_nljtz objects
                - v1b (list): One-body matrix elements as (a, b, value) tuples
                - v2b (list): Two-body matrix elements as (a, b, c, d, J, value) tuples
                
        Note:
            For NCSM calculations, one-body matrix elements represent kinetic terms
            :math:`T_{n,n'}` with :math:`\\hbar\\omega (A-1)/A` scaling factor.
            
            The snt format supports both valence-space and NCSM interactions.
            Mass-dependence factors :math:`(A/A_{ref})^p` are handled for empirical
            shell model interactions.
        
        Raises:
            FileNotFoundError: If the interaction file cannot be found.
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

                    if self.single_spiecies == 1 and tz == 1: # proton only
                        excluded.append(idx)
                    if self.single_spiecies == 2 and tz == -1: # neutron only
                        excluded.append(idx)

                    dict_sps[idx] = count
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
                    if a != b: # for hermitian
                        v1b.append((b, a, Tab))

            if sector == 2:  # reading 2b
                if len(tl) == 7:  # for NuHamil output
                    a, b, c, d, J = list(map(int, tl[:5]))
                    if a in excluded or b in excluded or c in excluded or d in excluded:
                        continue
                    idx_a = dict_sps[a]
                    idx_b = dict_sps[b]
                    idx_c = dict_sps[c]
                    idx_d = dict_sps[d]
                    vint, vkin = list(map(float, tl[5:]))
                    v = vint + vkin * self.hw / self.Anum
                    phase = 1
                    if idx_a > idx_b:
                        phase *= (-1)**(J + (single_particle_states[idx_a].j + single_particle_states[idx_b].j)//2 - 1)
                        idx_b, idx_a = idx_a, idx_b
                    if idx_c > idx_d:
                        phase *= (-1)**(J + (single_particle_states[idx_c].j + single_particle_states[idx_d].j)//2 - 1)
                        idx_d, idx_c = idx_c, idx_d
                    v = v * phase * Vfactor
                    v2b.append((idx_a, idx_b, idx_c, idx_d, J, v))
                if len(tl) == 6:  # valence snt files used in KSHELL code
                    a, b, c, d, J = list(map(int, tl[:5]))
                    if a in excluded or b in excluded or c in excluded or d in excluded:
                        continue
                    idx_a = dict_sps[a]
                    idx_b = dict_sps[b]
                    idx_c = dict_sps[c]
                    idx_d = dict_sps[d]
                    v = vint = float(tl[5]) * Vfactor
                    phase = 1
                    if idx_a > idx_b:
                        phase = (-1)**(J + (single_particle_states[idx_a].j + single_particle_states[idx_b].j)//2 - 1)
                        idx_b, idx_a = idx_a, idx_b
                    if idx_c > idx_d:
                        phase = (-1)**(J + (single_particle_states[idx_c].j + single_particle_states[idx_d].j)//2 - 1)
                        idx_d, idx_c = idx_c, idx_d
                    v *= phase   
                    v2b.append((idx_a, idx_b, idx_c, idx_d, J, v))
        return nsp_p, nsp_n, core_p, core_n, single_particle_states, v1b, v2b

    def get_mscheme_sps(self):
        """Generate M-scheme single-particle states from proton-neutron J-coupled states.
        
        Constructs the M-scheme single-particle states from the JT-coupled orbital basis.
        This includes expanding each j-orbital into its 2j+1 magnetic substates and organizing them by particle type.
        
        Returns:
            tuple: A tuple containing:
                - proton_register (range): Qubit indices for proton states
                - neutron_register (range): Qubit indices for neutron states  
                - mps (list): List of Orbit_nljjztz objects representing M-scheme states
                - dict_sps2msps (dict): Mapping from JT states to M-scheme state lists
                
        Note:
            The proton states are placed first in the qubit register, followed by
            neutron states. This convention is used throughout the quantum mappings.
        """
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
            if self.single_spiecies == 2:
                continue
            dict_sps2msps[idx] = []
            for jz in range(-j, j + 2, 2):
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
            if self.single_spiecies == 1:
                continue
            dict_sps2msps[idx] = []
            for jz in range(-j, j + 2, 2):
                mps.append(Orbit_nljjztz(n, l, j, jz, tz))
                dict_sps2msps[idx].append(count)
                count += 1
        if self.verbose:
            print("mps", mps)
        print("single_particle_states", self.single_spiecies)
        print("dict_sps2msps", dict_sps2msps)
        proton_register = range(num_proton)
        neutron_register = range(num_proton, count)
        return proton_register, neutron_register, mps, dict_sps2msps

    def delta_jj(self, morb_a, morb_b):
        """Check if two orbitals are identical in terms of their n, l, j, and tz quantum numbers.

        Args:
            morb_a (Orbit_nljjztz): First M-scheme orbital.
            morb_b (Orbit_nljjztz): Second M-scheme orbital.
            
        Returns:
            int: 1 if orbitals are identical, 0 otherwise.
        """
        n_a = morb_a.n
        l_a = morb_a.l
        j_a = morb_a.j
        tz_a = morb_a.tz
        n_b = morb_b.n
        l_b = morb_b.l
        j_b = morb_b.j
        tz_b = morb_b.tz
        if n_a == n_b and l_a == l_b and j_a == j_b and tz_a == tz_b:
            return 1
        else:
            return 0

    def op_dict_T1_permutations(self, op_dict, aa, bb, cc, dd, J, v):
        """Add two-body operator terms with proper permutation symmetry.
        
        Adds fermionic operator terms to the operator dictionary, including
        the necessary permutations for antisymmetrized matrix elements.
        
        Args:
            op_dict (dict): Dictionary to store operator strings and coefficients.
            aa (int): First bra-state index.
            bb (int): Second bra-state index.
            cc (int): First ket-state index.
            dd (int): Second ket-state index.
            J (int): Total angular momentum quantum number.
            v (float): Matrix element value.
            
        Note:
            We use the convention of the order of creation and annihilation operators like "+_aa +_bb -_dd -_cc" for the operator string.
            For identical bra and ket configurations, only one term is added.
            For different configurations, both direct and exchanged terms are included
            to ensure proper antisymmetrization of fermionic operators.
        """
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

    def get_mscheme_H(self, opform=False, verbose=0):
        """Construct Hamiltonian matrix elements in M-scheme representation.
        
        Transforms the JT-coupled nuclear interaction matrix elements into
        M-scheme (magnetic quantum number resolved) representation, which is
        required for quantum computing applications. The method handles both
        one-body and two-body terms with proper angular momentum recoupling.
        
        Args:
            opform (bool, optional): If True, returns operators in fermionic
            string form. If False, returns matrix element lists. Defaults to False.
            verbose (int | bool, optional): If truthy, print additional progress
                and basis information while constructing the M-scheme Hamiltonian.
                
        Returns:
            dict: Dictionary containing Hamiltonian components:
                - "SPE": Single-particle energy terms (1-body)
                - "Vpp": Proton-proton interaction terms (2-body)
                - "Vnn": Neutron-neutron interaction terms (2-body)  
                - "Vpn": Proton-neutron interaction terms (2-body)
                
        Note:
            For opform=True, terms are returned as fermionic operator strings
            (e.g., "+_0 -_1" for creation/annihilation operators).
            For opform=False, terms are returned as matrix element lists.
            
            The M-scheme transformation includes proper Clebsch-Gordan coefficients
            and normalization factors for antisymmetrized matrix elements.
        """
        op_dict_1b = {}
        op_dict_1b['n'] = { }
        op_dict_1b['p'] = { }
        op_dict_jz = { }
        op_dict_jz['p'] = { }
        op_dict_jz['n'] = { }
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
                n_a = morb_a.n
                l_a = morb_a.l
                j_a = morb_a.j
                jz_a = morb_a.jz
                tz_a = morb_a.tz
                for bb in self.dict_sps2msps[b]:
                    morb_b = self.msps[bb]
                    n_b = morb_b.n
                    l_b = morb_b.l
                    j_b = morb_b.j
                    jz_b = morb_b.jz
                    tz_b = morb_b.tz
                    if aa == bb or (
                        l_a == l_b and j_a == j_b and jz_a == jz_b and tz_a == tz_b
                    ):
                        if opform:
                            if tz_a == -1:
                                bitstr = "+_"+str(aa)+" -_"+str(bb)
                                op_dict_1b['p'][bitstr] = Tab
                                op_dict_jz['p'][bitstr] = jz_a 
                            elif tz_a==1:
                                bitstr = "+_"+str(aa-self.n_qubits_p)+" -_"+str(bb-self.n_qubits_p)
                                op_dict_1b['n'][bitstr] = Tab
                                op_dict_jz['n'][bitstr] = jz_a 
                            else:
                                raise ValueError("tz_a should be either 1 or -1")
                            num_1b_term += 1
                            if aa != bb: 
                                if tz_a==-1:
                                    bitstr = "+_"+str(aa)+" -_"+str(bb)
                                    op_dict_1b['p'][bitstr] = Tab
                                    op_dict_jz['p'][bitstr] = jz_a 
                                elif tz_a==1:
                                    bitstr = "+_"+str(aa-self.n_qubits_p)+" -_"+str(bb-self.n_qubits_p)
                                    op_dict_1b['n'][bitstr] = Tab
                                    op_dict_jz['n'][bitstr] = jz_a 
                                num_1b_term += 1
                        else:
                            pn_str = "p" if tz_a == -1 else "n"
                            op_dict_1b[pn_str][(aa + 1, bb + 1)] = Tab
                            num_1b_term += 1
                            if aa != bb:
                                op_dict_1b[pn_str][(bb + 1, aa + 1)] = Tab
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

            # Some interaction terms are not "canonical-ordered"
            if a > c or (a == c and b > d):
                a, b, c, d = c, d, a, b

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
                    delta_ab = self.delta_jj(morb_a, morb_b)
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
                            if Tz != 0 and (set([a, b]) == set([c, d])) and aa > cc:  # or (aa == cc and bb > dd):
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
                            delta_cd = self.delta_jj(morb_c, morb_d)
                            if abs(tz_c + tz_d) == 2 and delta_cd == 1:
                                N_cd = 2 / np.sqrt(1 + (-1) ** J)

                            CG1 = self.get_cG(morb_a, morb_b, J)
                            CG2 = self.get_cG(morb_c, morb_d, J)

                            v = V * CG1 * CG2 * N_ab * N_cd
                            if v == 0:
                                continue
                            # if  Tz == 2 and (set([aa, bb]) == set([cc, dd])) and aa == 8:
                            #     print(f"2b in MS: {aa, bb, cc, dd, J, v} Nfac {N_ab*N_cd} CG {CG1} {CG2}")
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
                                if opform: ## 09/18
                                    self.op_dict_T1_permutations(
                                        op_dict_nn, 
                                        aa-self.n_qubits_p, bb-self.n_qubits_p, 
                                        cc-self.n_qubits_p, dd-self.n_qubits_p, 
                                        J, v)
                                else:
                                    op_dict_nn.append(
                                        [aa + 1, bb + 1, cc + 1, dd + 1, J, v]
                                    )
                            else:
                                num_pn += 1
                                # for pn interaction, we need to consider operators separately
                                if opform:
                                    diag_factor = 1.0
                                    if aa == cc and bb == dd:
                                        diag_factor = 0.5
                                    p_str = "+_" + str(aa) + " -_" + str(cc)
                                    n_str = (
                                        "+_"
                                        + str(bb - self.n_qubits_p)
                                        + " -_"
                                        + str(dd - self.n_qubits_p)
                                    )            
                                    if (p_str, n_str) in op_dict_pn:
                                        op_dict_pn[(p_str, n_str)] += v * diag_factor
                                    else:
                                        op_dict_pn[(p_str, n_str)] = v * diag_factor
                                    p_str = "+_" + str(cc) + " -_" + str(aa)
                                    n_str = (
                                        "+_"
                                        + str(dd - self.n_qubits_p)
                                        + " -_"
                                        + str(bb - self.n_qubits_p)
                                    )
                                    if (p_str, n_str) in op_dict_pn:
                                        op_dict_pn[(p_str, n_str)] += v * diag_factor
                                    else:
                                        op_dict_pn[(p_str, n_str)] = v * diag_factor
                                else:
                                    op_dict_pn.append(
                                        [aa + 1, bb + 1, cc + 1, dd + 1, J, v]
                                    )

        if self.fn_3NF is not None:
            self.set_mscheme_3NF()
        
        op_dict_3b = self.get_3NF_Mscheme()
        num_V3 = len(op_dict_3b.keys())  
        print(f"# of H_m terms, 1b: {num_1b_term}, 2b pp: {num_pp}, nn: {num_nn}, pn: {num_pn} v3b {num_V3}")
        Hamildict = {
            "SPE": op_dict_1b,
            "Vpp": op_dict_pp,
            "Vnn": op_dict_nn,
            "Vpn": op_dict_pn,
            "Jz": op_dict_jz,
            "V3": op_dict_3b
        }
        self.Hamildict = Hamildict
        if opform:
            return Hamildict
        else:
            Hamildict = sum_over_J(Hamildict)
            return Hamildict
        
    def mapping_opform(self, 
                       mapping_method: str="Jordan-Wigner", 
                       filepath: str|os.PathLike="./"
                       ):
        """Map nuclear Hamiltonian to qubit operators using specified fermion-to-qubit mapping.
        
        Transforms the fermionic nuclear Hamiltonian into qubit operators using
        standard fermion-to-qubit mappings such as Jordan-Wigner or Bravyi-Kitaev.
        Handles the separate proton and neutron subsystems with appropriate mappings.
        
        Args:
            mapping_method (str): Fermion-to-qubit mapping method (e.g., "Jordan-Wigner").
            filepath (str|os.PathLike): Path to the file where the mapping results will be saved.
            This is currently used when the mapping_method is "HATTMapper".

        Returns:
            tuple: A tuple containing mapped Pauli operators:
                - H_1b_p (SparsePauliOp): One-body terms for protons
                - H_1b_n (SparsePauliOp): One-body terms for neutrons
                - H_jz_p (SparsePauliOp): Jz terms for protons
                - H_jz_n (SparsePauliOp): Jz terms for neutrons
                - H_pp (SparsePauliOp): Proton-proton terms
                - H_nn (SparsePauliOp): Neutron-neutron terms
                - H_pn (SparsePauliOp): Proton-neutron terms
                
        Note:
            The proton-neutron terms require special handling due to the tensor
            product structure of the proton-neutron Hilbert space.
        """
        if self.Hamildict is None:
            _=self.get_mscheme_H(opform=True)
        H_dict_p = self.Hamildict['SPE']['p'] | self.Hamildict['Vpp'] 
        H_dict_n = self.Hamildict['SPE']['n'] | self.Hamildict['Vnn'] 

        H_1b_p = mapping_to_Pauli_string(FermionicOp(self.Hamildict["SPE"]['p'], num_spin_orbitals=self.n_qubits_p), self.n_qubits, 0, method=mapping_method, Hamildict_specified= H_dict_p, filepath=filepath+'_p')
        H_1b_n = mapping_to_Pauli_string(FermionicOp(self.Hamildict["SPE"]['n'], num_spin_orbitals=self.n_qubits_n), self.n_qubits, self.n_qubits_p,method=mapping_method, Hamildict_specified= H_dict_n, filepath=filepath+'_n')
        H_jz_p = mapping_to_Pauli_string(FermionicOp(self.Hamildict["Jz"]['p'], num_spin_orbitals=self.n_qubits_p), self.n_qubits, 0, method=mapping_method, Hamildict_specified= H_dict_p, filepath=filepath+'_p')
        H_jz_n = mapping_to_Pauli_string(FermionicOp(self.Hamildict["Jz"]['n'], num_spin_orbitals=self.n_qubits_n), self.n_qubits, self.n_qubits_p,method=mapping_method, Hamildict_specified= H_dict_n, filepath=filepath+'_n')
        H_pp = mapping_to_Pauli_string(FermionicOp(self.Hamildict["Vpp"], num_spin_orbitals=self.n_qubits_p), self.n_qubits, 0, method=mapping_method, Hamildict_specified= H_dict_p, filepath=filepath+'_p')
        H_nn = mapping_to_Pauli_string(FermionicOp(self.Hamildict["Vnn"], num_spin_orbitals=self.n_qubits_n), self.n_qubits, self.n_qubits_p, method=mapping_method, Hamildict_specified= H_dict_n, filepath=filepath+'_n')
        H_pn = mapping_of_pn_hamiltonians(self.Hamildict["Vpn"], self.n_qubits_p, self.n_qubits_n, method=mapping_method, 
                                          Hamildict_specified_p = H_dict_p, 
                                          Hamildict_specified_n = H_dict_n, 
                                          filepath_p=filepath+'_p',
                                          filepath_n=filepath+'_n')   
        H_pn = removing_redundant_terms(H_pn)     

        if self.fn_3NF is not None:
            H_3b = self.mapping_3NF_Mscheme(method=mapping_method, filepath=filepath)
        else:
            H_3b = SparsePauliOp('I'*self.n_qubits, coeffs=0.0)
        
        return H_1b_p, H_1b_n, H_jz_p, H_jz_n, H_pp, H_nn, H_pn, H_3b

    def get_cG(self, sps_i, sps_j, J):
        """Calculate Clebsch-Gordan coefficient for angular momentum coupling.
        
        Computes the Clebsch-Gordan coefficient for coupling two single-particle
        states to a given total angular momentum J.
        
        Args:
            sps_i (Orbit_nljjztz): First single-particle state in M-scheme.
            sps_j (Orbit_nljjztz): Second single-particle state in M-scheme.
            J (int): Total angular momentum quantum number.
            
        Returns:
            float: Clebsch-Gordan coefficient <j1 m1 j2 m2 | J M>.
            
        """
        j = sps_i.j
        jz = sps_i.jz
        j_ = sps_j.j
        jz_ = sps_j.jz
        M = (jz + jz_) // 2
        return float(ClebschGordan(j / 2, jz / 2, j_ / 2, jz_ / 2, J, M).doit())

    def get_midx_from_nljjztz(self, n, l, j, jz, tz):
        """Find M-scheme single-particle state index from quantum numbers.
        
        Searches for the index of a M-scheme single-particle state with
        specified quantum numbers in the model space.
        
        Args:
            n (int): Principal quantum number (radial).
            l (int): Orbital angular momentum quantum number.
            j (int): Total angular momentum quantum number (twice the actual value).
            jz (int): Magnetic quantum number (twice the actual value).
            tz (int): Isospin projection (-1 for protons, +1 for neutrons).

        Raises:
            ValueError: If the requested M-scheme state is not found.

        """
        if not hasattr(self, "_midx_cache"):
            self._midx_cache = {
                (target.n, target.l, target.j, target.jz, target.tz): idx
                for idx, target in enumerate(self.msps)
            }
        tkey = (n, l, j, jz, tz)
        if tkey in self._midx_cache:
            return self._midx_cache[tkey]
        raise ValueError("Requested M-scheme state not found.")

    def read_3NF_readable(self, verbose=False):
        """Read the NuHamil 3NF file in readable.text fmt and store JT-coupled 3NF.

        In readable.text, V_{3N} is given as a function of a set of quanta,
        {a,b,c, Jab, Tab, d,e,f, Jde, Tde, Jabc, Tabc, Jabc, Tabc}
        where a~f are the single particle states having {n,l,j} quanta.
        Those matrix elements correspond to Eq.(41) in NuHamil paper, `T.Miyagi, EPJA (2023)59:150 <https://doi.org/10.1140/epja/s10050-023-01039-y>`_.
        Then, proton-neutron matrix elements are obtained through the Clebsch-Gordan coefficients as detailed in Eq.(36) of NuHamil paper.
                
        Returns:
            dict: A dictionary containing the 3NF matrix elements in the pn-coupled form.
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
                
                Trans_JT_into_pn(self.Z, self.N, a, b, c, Jab, Tab,
                                 d, e, f, Jde, Tde, Jabc, Tabc, ME, pnME_3NF,
                                 self.JTorbitals, self.single_particle_states,
                                 self.CG_dict)

        if verbose:
            print("pnME_3NF")
            for key, value in pnME_3NF.items():
                pn_a, pn_b, pn_c, pn_d, pn_e, pn_f, Jab, Jde, Jabc = key
                if abs(value) < 1.e-8:
                   continue
                print(f"pn_a {pn_a} pn_b {pn_b} pn_c {pn_c} pn_d {pn_d} pn_e {pn_e} pn_f {pn_f} Jab {Jab} Jde {Jde} Jabc {Jabc} value {value}")
            print(
                "Total number of 3NF matrix elements in pn J-coupld form",
                len(list(pnME_3NF.keys())),
            )
        return pnME_3NF

    def perms(self, i1, i2, i3):
        """Return signed permutations of three single-particle indices.

        Args:
            i1 (int): First index.
            i2 (int): Second index.
            i3 (int): Third index.

        Returns:
            dict[tuple[int, int, int], int]: Permutations mapped to their
            antisymmetry signs.
        """
        return {
                (i1, i2, i3): 1,
                (i2, i3, i1): 1,
                (i3, i1, i2): 1,
                (i2, i1, i3): -1,
                (i1, i3, i2): -1,
                (i3, i2, i1): -1,
        }

    def _set_mscheme_3NF_direct(self, verbose=False):
        """Build canonical m-scheme 3NF edges from the packed JT array.

        The readable-text conversion first creates ordered proton/neutron
        matrix elements and then expands all six-by-six permutations.  That
        is not a valid substitute for recoupling when the particle species
        are mixed.  This path evaluates the JT matrix element for each
        canonical m-scheme triple directly, including the spatial and
        isospin recoupling needed by ppn/pnn terms.
        """
        sps_3b = self.sps_3b
        dWS = self.dWs_3nf
        v3bme = self.v3bme
        dict_3b_idx = self.dict_3b_idx
        orbit_map = {
            (orb.e, orb.n, orb.l, orb.j, orb.tz): idx
            for idx, orb in sps_3b.sps.items()
        }
        mode_to_orbit = [
            orbit_map.get((mode.e, mode.n, mode.l, mode.j, mode.tz), 0)
            for mode in self.msps
        ]

        triples = []
        for triple in itertools.combinations(range(len(self.msps)), 3):
            states = [self.msps[index] for index in triple]
            if abs(sum(state.tz for state in states)) not in (1, 3):
                continue
            if sum(state.e for state in states) > sps_3b.e3max:
                continue
            if any(mode_to_orbit[index] == 0 for index in triple):
                continue
            triples.append(triple)

        vjj_cache = {}

        def direct_mscheme_value(bra, ket):
            bra_states = [self.msps[index] for index in bra]
            ket_states = [self.msps[index] for index in ket]
            if sum(state.tz for state in bra_states) != sum(state.tz for state in ket_states):
                return 0.0
            if sum(state.jz for state in bra_states) != sum(state.jz for state in ket_states):
                return 0.0
            if (sum(state.l for state in bra_states) +
                    sum(state.l for state in ket_states)) % 2 != 0:
                return 0.0
            if sum(state.e for state in ket_states) > sps_3b.e3max:
                return 0.0

            j1, m1 = bra_states[0].j, bra_states[0].jz
            j2, m2 = bra_states[1].j, bra_states[1].jz
            j3, m3 = bra_states[2].j, bra_states[2].jz
            j4, m4 = ket_states[0].j, ket_states[0].jz
            j5, m5 = ket_states[1].j, ket_states[1].jz
            j6, m6 = ket_states[2].j, ket_states[2].jz
            m12_bra = m1 + m2
            m12_ket = m4 + m5
            m_total = m1 + m2 + m3
            value = 0.0

            for Jab in range(abs(j1 - j2) // 2, (j1 + j2) // 2 + 1):
                if abs(m12_bra) > 2 * Jab:
                    continue
                cg12 = get_CGs_from_dict(
                    j1, m1, j2, m2, 2 * Jab, m12_bra, self.CG_dict
                )
                if cg12 == 0.0:
                    continue
                for Jde in range(abs(j4 - j5) // 2, (j4 + j5) // 2 + 1):
                    if abs(m12_ket) > 2 * Jde:
                        continue
                    cg45 = get_CGs_from_dict(
                        j4, m4, j5, m5, 2 * Jde, m12_ket, self.CG_dict
                    )
                    if cg45 == 0.0:
                        continue
                    J2_min = max(abs(2 * Jab - j3), abs(2 * Jde - j6))
                    J2_max = min(2 * Jab + j3, 2 * Jde + j6)
                    for J2 in range(J2_min, J2_max + 1, 2):
                        if abs(m_total) > J2:
                            continue
                        cg123 = get_CGs_from_dict(
                            2 * Jab, m12_bra, j3, m3, J2, m_total,
                            self.CG_dict,
                        )
                        cg456 = get_CGs_from_dict(
                            2 * Jde, m12_ket, j6, m6, J2, m_total,
                            self.CG_dict,
                        )
                        if abs(cg123 * cg456) < 1.0e-12:
                            continue
                        vjj_key = (
                            Jab, Jde, J2,
                            mode_to_orbit[bra[0]], mode_to_orbit[bra[1]],
                            mode_to_orbit[bra[2]], mode_to_orbit[ket[0]],
                            mode_to_orbit[ket[1]], mode_to_orbit[ket[2]],
                        )
                        if vjj_key not in vjj_cache:
                            vjj_cache[vjj_key] = _get_v3_pn(
                                sps_3b.e3max,
                                v3bme,
                                dict_3b_idx,
                                sps_3b,
                                dWS,
                                Jab,
                                Jde,
                                J2,
                                mode_to_orbit[bra[0]],
                                mode_to_orbit[bra[1]],
                                mode_to_orbit[bra[2]],
                                mode_to_orbit[ket[0]],
                                mode_to_orbit[ket[1]],
                                mode_to_orbit[ket[2]],
                            )
                        value += cg12 * cg123 * cg45 * cg456 * vjj_cache[vjj_key]
            return value

        result = {}
        for bra_index, bra in enumerate(triples):
            for ket in triples[bra_index:]:
                value = direct_mscheme_value(bra, ket)
                if abs(value) < 1.0e-12:
                    continue
                result[(*bra, *ket)] = value
                if bra != ket:
                    result[(*ket, *bra)] = value

        if verbose:
            print("After direct m-scheme recoupling.")
        print("Total number of 3NF matrix elements in M-scheme", len(result))
        self.v3b_Mscheme = result
        return result

    def set_mscheme_3NF(self, verbose=False):
        """Calculate the 3NF matrix elements in the mscheme.

        Returns:
            dict: A dictionary containing the 3NF matrix elements in the mscheme.

        Note:
            Q. Why do we need to devide the 3NF matrix elements by 36?
            A. All the permutations of the 3NF matrix elements are explicitly counted below.
            It would be more efficient to avoid redundancy in the first place. It is left for the future.

            Q. Why do we need to multiply the 3NF matrix elements by 9 for pnn and npp?
            A. At first, proton and neutron operators commute.
            We are now generating all the permutations not taking into account of this.
            While transforming the pn JT-coupled form to J-coupled form,
            we generate something like <pnn|V|npn>, <pnn|V|nnp>, etc.
            The factor of 9 is to take account of this.
        """
        if hasattr(self, "_3nf_reader"):
            return self._set_mscheme_3NF_direct(verbose=verbose)

        v3b_Mscheme = {}
        if verbose:
            print("Converting pn J-coupled 3NF to mscheme...")
        print("Setting up 3NF matrix elements in mscheme...")

        if not hasattr(self, "_midx_cache"):
            self._midx_cache = {
                (target.n, target.l, target.j, target.jz, target.tz): idx
                for idx, target in enumerate(self.msps)
            }

        cg_dict = self.CG_dict
        get_midx = self.get_midx_from_nljjztz

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

            # precalc smaller dict for CG coefficients
            cg_ab = {}
            for m_a in ja_range:
                for m_b in jb_range:
                    mab = m_a + m_b
                    if abs(mab) > 2 * Jab:
                        continue
                    cg1 = get_CGs_from_dict(orb_a.j, m_a, orb_b.j, m_b, 2 * Jab, mab, cg_dict)
                    if abs(cg1) < 1.0e-16:
                        continue
                    cg_ab[(m_a, m_b)] = (mab, cg1)
            cg_de = {}
            for m_d in jd_range:
                for m_e in je_range:
                    mde = m_d + m_e
                    if abs(mde) > 2 * Jde:
                        continue
                    cg2 = get_CGs_from_dict(orb_d.j, m_d, orb_e.j, m_e, 2 * Jde, mde, cg_dict)
                    if abs(cg2) < 1.0e-16:
                        continue
                    cg_de[(m_d, m_e)] = (mde, cg2)
            # for m_a, m_b, m_c, m_d, m_e, m_f in itertools.product(
            #     ja_range, jb_range, jc_range, jd_range, je_range, jf_range
            # ):    
            for m_a, m_b, m_c, m_d, m_e in itertools.product(
                ja_range, jb_range, jc_range, jd_range, je_range
            ):  
                m_f = (m_a + m_b + m_c) - (m_d + m_e)
                if m_f not in jf_range:
                    continue

                ab_key = (m_a, m_b)
                de_key = (m_d, m_e)
                if ab_key not in cg_ab or de_key not in cg_de:
                    continue

                mab, cg1 = cg_ab[ab_key]
                mde, cg2 = cg_de[de_key]

                if (m_a + m_b + m_c) != (m_d + m_e + m_f):
                    continue
                if abs(m_a + m_b + m_c) > Jabc:
                    continue
                if abs(m_d + m_e + m_f) > Jabc:
                    continue

                im_a = get_midx(orb_a.n, orb_a.l, orb_a.j, m_a, orb_a.tz)
                im_b = get_midx(orb_b.n, orb_b.l, orb_b.j, m_b, orb_b.tz)
                im_c = get_midx(orb_c.n, orb_c.l, orb_c.j, m_c, orb_c.tz)
                if im_a == im_b or im_b == im_c or im_a == im_c:
                    continue

                im_d = get_midx(orb_d.n, orb_d.l, orb_d.j, m_d, orb_d.tz)
                im_e = get_midx(orb_e.n, orb_e.l, orb_e.j, m_e, orb_e.tz)
                im_f = get_midx(orb_f.n, orb_f.l, orb_f.j, m_f, orb_f.tz)
                if im_d == im_e or im_e == im_f or im_d == im_f:
                    continue

                cg3 = get_CGs_from_dict(2 * Jab, mab, orb_c.j, m_c, Jabc, (m_a + m_b + m_c), cg_dict)
                cg4 = get_CGs_from_dict(2 * Jde, mde, orb_f.j, m_f, Jabc, (m_d + m_e + m_f), cg_dict)

                cgfact = cg1 * cg2 * cg3 * cg4
                if abs(cgfact) < 1.0e-8:
                    continue

                mkey = (im_a, im_b, im_c, im_d, im_e, im_f)
                v3b_Mscheme[mkey] = v3b_Mscheme.get(mkey, 0.0) + ME_3n_pn * cgfact

        tmp = copy.copy(v3b_Mscheme)
        for key, me in tmp.items():
            i1, i2, i3, i4, i5, i6 = key
            perms_bra = self.perms(i1, i2, i3)
            perms_ket = self.perms(i4, i5, i6)
            for i123, sign_123 in perms_bra.items():
                for i456, sign_456 in perms_ket.items():
                    v3b_Mscheme[(*i123, *i456)] = me * sign_123 * sign_456
                    v3b_Mscheme[(*i456, *i123)] = me * sign_123 * sign_456
        if verbose:
            print("After perm.")
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
    
    def write_v3b_Mscheme(self, fn_out: str|os.PathLike="v3b_Mscheme.txt"):
        """Write the 3NF matrix elements in M-scheme to a file.

        Args:
            fn_out (str|os.PathLike): Output filename to write the 3NF matrix elements.

        """
        if self.v3b_Mscheme is None:
            self.set_mscheme_3NF()
        with open(fn_out, "w") as f:
            for key, me in self.v3b_Mscheme.items():
                im_a, im_b, im_c, im_d, im_e, im_f = key
                if abs(me) < 1.e-8:
                    continue
                f.write(
                    f"{im_a} {im_b} {im_c} {im_d} {im_e} {im_f} {me:.8e}\n"
                )
        print(f"3NF matrix elements in M-scheme written to {fn_out}")

    def separate_proton_and_neutron(self, 
                                    im_a: int, im_b: int, im_c: int,
                                    im_d: int, im_e: int, im_f: int):
        """Separate the proton and neutron operators for a given set of indices.

        """
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

    def get_3NF_Mscheme(self, verbose=False):
        """Convert stored M-scheme 3NF matrix elements to operator strings.

        Args:
            verbose (bool, optional): If True, print each generated proton and
                neutron operator string.

        Returns:
            dict: Mapping ``(proton_op, neutron_op)`` pairs to matrix elements.
            Returns an empty dictionary when no M-scheme 3NF data is available.
        """
        op_dict_3b = { }
        if self.v3b_Mscheme is None:
            return op_dict_3b
        if verbose:
            print("Setting up op_dict_3b...")
        for mkey, value in tqdm(self.v3b_Mscheme.items()):
            im_a, im_b, im_c, im_d, im_e, im_f = mkey
            morb_a = self.msps[im_a]; morb_b = self.msps[im_b]; morb_c = self.msps[im_c]
            #morb_d = self.msps[im_d]; morb_e = self.msps[im_e]; morb_f = self.msps[im_f]
            if abs(value) < 1.e-8: 
                continue
            Tz_bra = morb_a.tz + morb_b.tz + morb_c.tz
            if self.verbose:
                print(f"M: <{im_a} {im_b} {im_c} |V| {im_d} {im_e} {im_f}> Chan:{self.channel[Tz_bra]} ME3n_M: {value} ")
            p_op_str, n_op_str = self.separate_proton_and_neutron(im_a, im_b, im_c, im_d, im_e, im_f)
            if self.verbose:
                print(f" => proton_op {p_op_str} x neutron_op {n_op_str}")        
            if (p_op_str, n_op_str) in op_dict_3b:
                op_dict_3b[(p_op_str, n_op_str)] += value
            else:
                op_dict_3b[(p_op_str, n_op_str)] = value
        return op_dict_3b

    def get_V3_p_n(self):
        """Split pure-proton and pure-neutron three-body terms.

        Returns:
            tuple[dict, dict]: Proton-only and neutron-only 3NF operator
            dictionaries extracted from ``self.Hamildict["V3"]``.
        """
        result_n = {}
        result_p = {}
        if self.Hamildict is None:
            _=self.get_mscheme_H(opform=True)
        for (p_str, n_str) in tqdm(self.Hamildict['V3'].keys()):
            if len(p_str)==0:
                result_n[n_str] = self.Hamildict['V3'][(p_str, n_str)] 
            elif len(n_str)==0:
                result_p[p_str] = self.Hamildict['V3'][(p_str, n_str)]
        return result_p, result_n

    def _map_3NF_sector_cached(self, op_str: str, is_proton: bool,
                               method: str, filepath: str|os.PathLike):
        """Map one proton/neutron sector operator and cache the result.

        Returns:
            tuple[list[str], np.ndarray]: Pauli labels and coefficients.
        """
        if not hasattr(self, "_cache_3nf_map_p"):
            self._cache_3nf_map_p = {}
            self._cache_3nf_map_n = {}

        cache = self._cache_3nf_map_p if is_proton else self._cache_3nf_map_n
        key = (method, str(filepath), op_str) if method == "HATTMapper" else (method, op_str)
        if key in cache:
            return cache[key]

        if is_proton:
            num_orb = self.n_qubits_p
            hdict = GLOBAL_H_DICT_P
            fn = filepath + "_p"
        else:
            num_orb = self.n_qubits_n
            hdict = GLOBAL_H_DICT_N
            fn = filepath + "_n"

        if method == "HATTMapper":
            mapped = mapping_to_Pauli_string(
                FermionicOp({op_str: 1.0}, num_spin_orbitals=num_orb),
                num_orb,
                0,
                method=method,
                Hamildict_specified=hdict,
                filepath=fn,
            )
        else:
            mapped = mapping_to_Pauli_string(
                FermionicOp({op_str: 1.0}, num_spin_orbitals=num_orb),
                num_orb,
                0,
                method=method,
            )

        labels = [p.to_label() for p in mapped.paulis]
        coeffs = mapped.coeffs
        cache[key] = (labels, coeffs)
        return labels, coeffs

    def _build_3NF_out(self, p_labels, p_coeffs, n_labels, n_coeffs, coeff_overall):
        """Build cartesian-product Pauli terms for one 3NF p-n term."""
        n_p = len(p_labels)
        n_n = len(n_labels)
        out = [None] * (n_p * n_n)
        idx = 0
        for i, p_label in enumerate(p_labels):
            cp_scaled = coeff_overall * p_coeffs[i]
            for j, n_label in enumerate(n_labels):
                out[idx] = (n_label + p_label, cp_scaled * n_coeffs[j])
                idx += 1
        return out


    def task_3NF_pn(self, p_str, n_str, coeff_overall, method: str, filepath: str|os.PathLike):
        """Map one p-n 3NF term into Pauli-label tuples.

        This path is split into two stages:
        1) map each sector (with cache),
        2) build output tuples from cartesian products.
        """
        p_labels, p_coeffs = self._map_3NF_sector_cached(
            p_str, is_proton=True, method=method, filepath=filepath
        )
        n_labels, n_coeffs = self._map_3NF_sector_cached(
            n_str, is_proton=False, method=method, filepath=filepath
        )
        return self._build_3NF_out(p_labels, p_coeffs, n_labels, n_coeffs, coeff_overall)

    def mapping_3NF_Mscheme(self, method="Jordan-Wigner",
                            filepath: str|os.PathLike="./tmp",
                            verbose=False):
        """Map the 3NF matrix elements to the specified scheme.
    
        """
        if self.Hamildict is None:
            _=self.get_mscheme_H(opform=True)
        if method != "HATTMapper":
            H_dict_p = H_dict_n = None
        else:
            V3p, V3n = self.get_V3_p_n()
            H_dict_p = self.Hamildict['SPE']['p'] | self.Hamildict['Vpp'] | V3p
            H_dict_n = self.Hamildict['SPE']['n'] | self.Hamildict['Vnn'] | V3n
        global GLOBAL_H_DICT_P, GLOBAL_H_DICT_N
        GLOBAL_H_DICT_P = H_dict_p
        GLOBAL_H_DICT_N = H_dict_n
        print(f"Encoding 3NF in {method} mapping... # of 3NF terms: {len(self.Hamildict['V3'].keys())}")
        ti = time.time()
       
        nproc = max(multiprocessing.cpu_count() - 4, 1)
        with get_context("fork").Pool(
            processes=nproc,
            initializer=_init_task_3nf_worker,
            initargs=(self,),
        ) as pool:
            task_iter = (
                (p_str, n_str, coeff_overall, method, filepath)
                for (p_str, n_str), coeff_overall in self.Hamildict["V3"].items()
            )
            mapped_H3b = removing_redundant_ops(
                (
                    item
                    for sublist in pool.imap(_task_3nf_worker, task_iter, chunksize=256)
                    for item in sublist
                )
            )
    
        tf = time.time()
        print(f"3NF mapping done in {tf - ti:.2f} sec.  {len(mapped_H3b)/(tf - ti):.2f} terms/sec")
    
        return mapped_H3b


def export_encoded_Hamil(Hamil: SparsePauliOp,
                         fn: str|os.PathLike):
    """Export the encoded Hamiltonian to a text file.

    Args:
        Hamil (SparsePauliOp): Encoded Hamiltonian to write.
        fn (str | os.PathLike): Output text filename.

    """
    oup  = open(fn, "w")
    for pauli, coeff in zip(Hamil.paulis, Hamil.coeffs):
        oup.write(f"{pauli.to_label()} {np.real(coeff)}\n")
    oup.close()

def process_op(args):
    """Process a single proton-neutron operator term for quantum mapping.
    
    This function processes individual terms from the three-body force dictionary,
    mapping them from fermionic operators to Pauli strings using the specified
    fermion-to-qubit mapping. Designed for parallel processing.
    
    Args:
        args (tuple): A tuple containing:
            - p_str (str): Proton fermionic operator string
            - n_str (str): Neutron fermionic operator string  
            - coeff_overall (complex): Overall coefficient
            - n_qubits_p (int): Number of proton qubits
            - n_qubits_n (int): Number of neutron qubits
            - method (str): Fermion-to-qubit mapping method
            
    Returns:
        list: List of (pauli_label, coefficient) tuples for this operator.
        
    Note:
        This function is designed to be used with multiprocessing for
        efficient parallel processing of large three-body force matrices.
    """
    p_str, n_str, coeff_overall, n_qubits_p, n_qubits_n, method = args
    op_list_local = []
    op_p = mapping_to_Pauli_string(
        FermionicOp({p_str: 1.0}, num_spin_orbitals=n_qubits_p),
        n_qubits_p,
        method,
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


def hat(a):
    """Angular momentum hat notation: sqrt(2*j + 1).
    
    Args:
        a (int or float): Angular momentum quantum number (can be half-integer).
        
    Returns:
        float: sqrt(2*a + 1), commonly used in nuclear physics calculations.
        
    Note:
        This is the standard "hat" notation used in nuclear physics and
        atomic physics for angular momentum algebra.
    """
    return np.sqrt(2 * a + 1)


def permutation_parity(lst):
    """Calculate the parity (even/odd) of a permutation.
    
    Determines whether a permutation is even (0) or odd (1) by counting
    the number of inversions in the list.
    
    Args:
        lst (list): A permutation as a list of elements.
        
    Returns:
        int: 0 for even permutation, 1 for odd permutation.
        
    Note:
        Used in nuclear physics calculations where antisymmetrization
        requires tracking permutation signs.
    """
    # Returns 0 for even, 1 for odd permutation.
    par = 1
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if lst[i] > lst[j]:
                par *= -1
    return 0 if par > 0 else 1


def sort_3_orbits(a_in, b_in, c_in):
    """Sort three orbital labels into descending canonical order.

    Even labels are shifted down by one before sorting, matching the indexing
    convention used by the three-body matrix-element reader.

    Args:
        a_in (int): First orbital label.
        b_in (int): Second orbital label.
        c_in (int): Third orbital label.

    Returns:
        tuple[int, int, int, int]: Sorted labels and the permutation index.
    """
    # Adjust inputs if they're even
    a_in = a_in - 1 if a_in % 2 == 0 else a_in
    b_in = b_in - 1 if b_in % 2 == 0 else b_in
    c_in = c_in - 1 if c_in % 2 == 0 else c_in
    # Initialize variables
    a, b, c = a_in, b_in, c_in
    # Sort the values to get a >= b >= c through pairwise swaps
    if a < b:
        a, b = b, a
    if b < c:
        b, c = c, b
    if a < b:
        a, b = b, a
    # Determine index based on comparisons with original inputs
    if a_in == a:
        idx = 0 if b_in == b else 3
    elif a_in == b:
        idx = 4 if b_in == a else 1
    else:
        idx = 2 if b_in == a else 5
    return a, b, c, idx


def get_CGs_from_dict(j1, m1, j2, m2, J, M, CG_dict: dict):
    """Retrieve or calculate Clebsch-Gordan coefficient.
    
    Gets a Clebsch-Gordan coefficient from cache or calculates and stores it.
    This provides efficient access to frequently used CG coefficients in
    nuclear many-body calculations.
    
    Args:
        j1 (int): First angular momentum (twice the actual value).
        m1 (int): First magnetic quantum number (twice the actual value).
        j2 (int): Second angular momentum (twice the actual value).
        m2 (int): Second magnetic quantum number (twice the actual value).
        J (int): Coupled angular momentum (twice the actual value).
        M (int): Coupled magnetic quantum number (twice the actual value).
        CG_dict (dict): Cache dictionary for storing calculated coefficients.
        
    Returns:
        float: Clebsch-Gordan coefficient <j1 m1 j2 m2 | J M>.
        
    Note:
        The coefficient is calculated using sympy if not found in cache.
        All angular momenta are provided as integers (twice their actual values).
    """
    tkey = (j1, m1, j2, m2, J, M)
    if tkey in CG_dict:
        return CG_dict[tkey]
    else:
        CG = float(
            ClebschGordan(j1 / 2, m1 / 2, j2 / 2, m2 / 2, J / 2, M / 2).doit()
        )
        CG_dict[tkey] = CG
        return CG


def get_nkey6_shift(a, b, c, d, e, f, int_shift=3):
    """Pack six possibly negative integer labels into one shifted integer key.

    Args:
        a (int): First label.
        b (int): Second label.
        c (int): Third label.
        d (int): Fourth label.
        e (int): Fifth label.
        f (int): Sixth label.
        int_shift (int, optional): Shift added before packing.

    Returns:
        int: Packed key.
    """
    return (
        ((a + int_shift) << 50)
        + ((b + int_shift) << 40)
        + ((c + int_shift) << 30)
        + ((d + int_shift) << 20)
        + ((e + int_shift) << 10)
        + (f + int_shift)
    )


def get_nkey6(a, b, c, d, e, f):
    """Pack six non-negative integer labels into one integer key."""
    return (a << 50) + (b << 40) + (c << 30) + (d << 20) + (e << 10) + f


def unhash_key6j(i):
    """Unpack a key produced by :func:`get_nkey6` into six labels."""
    a = i >> 50
    b = (i >> 40) & 0x3FF
    c = (i >> 30) & 0x3FF
    d = (i >> 20) & 0x3FF
    e = (i >> 10) & 0x3FF
    f = i & 0x3FF
    return a, b, c, d, e, f


class sps_3Blab:
    """Single-particle state manager for three-body matrix element calculations.
    
    This class manages single-particle states and model space parameters needed
    for three-body force calculations. It stores both the truncated model space
    parameters and the parameters from the original interaction files.
    
    Attributes:
        e1max (int): Maximum single-particle excitation energy for model space.
        e1max_file (int): Maximum single-particle excitation from original file.
        e2max_file (int): Maximum two-body excitation from original file.  
        e3max (int): Maximum three-body excitation energy for model space.
        e3max_file (int): Maximum three-body excitation from original file.
        norbits_ms (int): Number of orbitals in model space.
        norbits_file (int): Number of orbitals in original file.
        sps (dict): Single-particle states for model space.
        sps_file (dict): Single-particle states from original file.
    """
    
    def __init__(
        self,
        e1max: int,
        e1max_file: int,
        e2max_file: int,
        e3max: int,
        e3max_file: int,
        norbits_ms: int,
        norbits_file: int,
        sps,
        sps_file,
    ):
        """Initialize single-particle state manager for 3NF calculations.
        
        Args:
            e1max (int): Maximum single-particle excitation for model space.
            e1max_file (int): Maximum single-particle excitation from file.
            e2max_file (int): Maximum two-body excitation from file.
            e3max (int): Maximum three-body excitation for model space.
            e3max_file (int): Maximum three-body excitation from file.
            norbits_ms (int): Number of orbitals in model space.
            norbits_file (int): Number of orbitals in original file.
            sps (dict): Model space single-particle states.
            sps_file (dict): File single-particle states.
        """
        self.e1max = e1max
        self.e1max_file = e1max_file
        self.e2max_file = e2max_file
        self.e3max = e3max
        self.e3max_file = e3max_file
        self.norbits_ms = norbits_ms
        self.norbits_file = norbits_file
        self.sps = sps
        self.sps_file = sps_file


# Worker globals for streaming 3NF mapping without starmap bulk materialization.
_TASK_3NF_INSTANCE = None


def _init_task_3nf_worker(instance):
    global _TASK_3NF_INSTANCE
    _TASK_3NF_INSTANCE = instance


def _task_3nf_worker(args):
    p_str, n_str, coeff_overall, method, filepath = args
    return _TASK_3NF_INSTANCE.task_3NF_pn(
        p_str, n_str, coeff_overall, method, filepath
    )


def valid_check(ea, eb, ec, ed, ee, ef, e1max, e2max, e3max):
    """Check if three-body matrix element satisfies model space truncation limits.
    
    Validates that a three-body matrix element with excitation energies ea-ef
    is within the specified model space truncation parameters.
    
    Args:
        ea (int): Excitation energy of the first bra orbital.
        eb (int): Excitation energy of the second bra orbital.
        ec (int): Excitation energy of the third bra orbital.
        ed (int): Excitation energy of the first ket orbital.
        ee (int): Excitation energy of the second ket orbital.
        ef (int): Excitation energy of the third ket orbital.
        e1max (int): Maximum single-particle excitation energy.
        e2max (int): Maximum two-body excitation energy.
        e3max (int): Maximum three-body excitation energy.
        
    Returns:
        bool: True if the matrix element is within model space limits.
        
    Note:
        This enforces the standard nuclear physics truncation scheme where
        single-particle, two-body, and three-body excitations are separately
        limited to maintain computational tractability.
    """
    if ea > e1max or eb > e1max or ec > e1max or ed > e1max or ee > e1max or ef > e1max:
        return False
    if ea + eb > e2max or ea + ec > e2max or eb + ec > e2max:
        return False
    if ed + ee > e2max or ed + ef > e2max or ee + ef > e2max:
        return False
    if ea + eb + ec > e3max or ed + ee + ef > e3max:
        return False
    return True


class ReadThBME_me3jgz:
    """Read three-body matrix elements from a me3j.gz file.

    This class handles the reading and processing of three-body matrix elements
    stored in a compressed me3j.gz file format.


    """
    def __init__(
        self,
        single_particle_states: dict,
        JT_orbitals: dict,
        Z: int,
        N: int,
        filename,
        e1max: int,
        e1max_file: int,
        e2max: int,
        e3max: int,
        e3max_file: int,
        CG_dict_fromHamiltonian: dict,
        single_species: bool = False,
        verbose:bool =False,
    ):
        self.filename = filename
        self.single_particle_states = single_particle_states
        self.JTorbitals = JT_orbitals
        self.Z = Z
        self.N = N
        self.e1max = e1max
        self.e2max = e2max
        self.e3max = e3max
        self.e1max_file = e1max_file
        self.e2max_file = 2 * self.e1max_file
        self.e3max_file = e3max_file
        self.dWs = prep_dicts_for_WignerSymbols(e1max)
        self.sps_3b = self.get_modelspace(e1max, e1max_file, e2max, e3max, e3max_file)
        self.count_ME_file, self.dict_idxheadThBME = self.count_nreads(self.sps_3b)
        self.count_ME_MS, self.dict_idxheadThBME_MS = self.count_nreads(self.sps_3b, "ModelSpace")
        self.ThBME = self.read_me3jgz(self.filename, self.count_ME_file)
        self.v3bme = self.loop_over_JT(self.sps_3b, what_you_need="allocate_v3bme")
        self.verbose = verbose
        self.CG_dict = CG_dict_fromHamiltonian
        if (
            self.e1max_file == self.e1max
            and self.e2max_file == 2 * self.e1max
            and self.e3max_file == self.e3max
        ):
            self.v3bme = self.ThBME
        else:
            self.v3bme = truncate_v3bme(
                self.count_ME_file,
                self.count_ME_MS,
                self.sps_3b,
                self.ThBME,
                self.dWs,
                self.dict_idxheadThBME,
                self.dict_idxheadThBME_MS
            )
        print(f"norm(v3bme):", np.linalg.norm(self.v3bme))
        self.pnME_3NF = self.loop_over_JT(self.sps_3b, what_you_need="me3j->readable.txt", array_in=self.v3bme)


    def get_modelspace(self, e1max, e1max_file, e2max, e3max, e3max_file):
        """Construct model-space labels for file and truncated 3NF spaces.

        Args:
            e1max (int): One-body truncation used in the active model space.
            e1max_file (int): One-body truncation represented in the input file.
            e2max (int): Two-body truncation used in the active model space.
            e3max (int): Three-body truncation used in the active model space.
            e3max_file (int): Three-body truncation represented in the input file.

        Returns:
            sps_3Blab: Model-space descriptor used by the me3j reader.
        """
        sps = {}
        sps_file = {}
        norbits_file = norbits_ms = 0
        for mode in ["File", "ModelSpace"]:
            norbits = 0
            target = sps_file if mode == "File" else sps
            e1adopt = e1max_file if mode == "File" else e1max
            e3adopt = e3max_file if mode == "File" else e3max
            for temax in range(0, e1adopt + 1):
                lmin = temax % 2
                lstep = 2
                lmax = temax
                for l in range(lmin, lmax + 1, lstep):
                    n = (temax - l) // 2
                    for j2 in range(abs(2 * l - 1), 2 * l + 2, 2):
                        for tz in [-1, 1]:
                            norbits += 1
                            target[norbits] = Orbit_nljtz(n, l, j2, tz)
            if mode == "File":
                norbits_file = norbits
            else:
                norbits_ms = norbits
        print(
            f"Modelspace {e1max}_{e2max}_{e3max}, norbits (File): {norbits_file}, norbits (ModelSpace): {norbits_ms}"
        )
        return sps_3Blab(
            e1max,
            e1max_file,
            e2max,
            e3max,
            e3max_file,
            norbits_ms,
            norbits_file,
            sps,
            sps_file,
        )

    def count_nreads(self, sps_3b: sps_3Blab, mode="File"):
        """Count the number of elements of floating-point numbers to be read from the interaction file.

        In some interaction files, odd indices correspond to proton and even indices correspond to neutron.
        One should take this into account when processing the data when working in JT-coupled schemes.

        In the nested loops, the number of reads is counted by considering possible (allowed) <a, b, c [Jab]|V| d, e, f [Jde]>
        in isospin space, i.e. a~e are indices for Orbit_nlj.
        The factors ((twoJCMax - twoJCMin) // 2 + 1) and 5 represents the number of allowed J_3 and T_3.
        Note that T_3 = 1 case has 4 components (Tab, Tde) = (0, 0), (0, 1), (1, 0), (1, 1), while T_3 = 0 case has only 1 component.
        The Tindex will be 2 * Tab + Tde + ((T2 - 1) // 2), i.e. 0: (0, 0, 1), 1: (0, 1, 0), 2: (1, 0, 0), 3: (1, 1, 1), 4: (1, 1, 3).

        :nreads: is a list of the head indices relevant to the first index in bra/ket for 3NF matrix elements.
          This would be helpful if one wants to efficiently access the relevant indices via multi-threading.
        """
        # Select parameters based on mode.
        norbits = sps_3b.norbits_file if mode == "File" else sps_3b.norbits_ms
        e1max = sps_3b.e1max_file if mode == "File" else sps_3b.e1max
        e2max = sps_3b.e2max_file if mode == "File" else sps_3b.e1max * 2
        e3max = sps_3b.e3max_file if mode == "File" else sps_3b.e3max
        sps = sps_3b.sps_file if mode == "File" else sps_3b.sps

        dict_idx_ThBME = {}
        nread = 0
        # nreads is a list with length equal to norbits//2.
        nreads = [0] * (norbits // 2)

        # Begin loop over 'bra' indices (using odd indices starting at 1)
        for idx_a in range(1, norbits + 1, 2):
            nreads[(idx_a // 2)] = nread
            oa = sps[idx_a]
            ea = oa.e
            if ea > e1max:
                continue

            for idx_b in range(1, idx_a + 1, 2):
                ob = sps[idx_b]
                eb = ob.e
                if ea + eb > e2max:
                    continue

                for idx_c in range(1, idx_b + 1, 2):
                    oc = sps[idx_c]
                    ec = oc.e
                    if ea + eb + ec > e3max:
                        continue

                    # Compute angular momentum limits for the bra.
                    JabMax = (oa.j + ob.j) // 2
                    JabMin = abs(oa.j - ob.j) // 2
                    if abs(oa.j - ob.j) > oc.j:
                        twoJCMindownbra = abs(oa.j - ob.j) - oc.j
                    elif oc.j < (oa.j + ob.j):
                        twoJCMindownbra = 1
                    else:
                        twoJCMindownbra = oc.j - oa.j - ob.j
                    twoJCMaxupbra = oa.j + ob.j + oc.j

                    # Loop for 'ket' part.
                    for idx_d in range(1, idx_a + 1, 2):
                        od = sps[idx_d]
                        ed = od.e
                        # Determine the upper limit for idx_e based on idx_a and idx_d.
                        end_idx_e = idx_b if idx_a == idx_d else idx_d
                        for idx_e in range(1, end_idx_e + 1, 2):
                            oe = sps[idx_e]
                            ee = oe.e
                            # Determine the upper limit for idx_f.
                            idx_f_max = (
                                idx_c if (idx_a == idx_d and idx_b == idx_e) else idx_e
                            )
                            for idx_f in range(1, idx_f_max + 1, 2):
                                of = sps[idx_f]
                                ef = of.e
                                if ed + ee + ef > e3max:
                                    continue
                                if (oa.l + ob.l + oc.l + od.l + oe.l + of.l) % 2 != 0:
                                    continue

                                JdeMax = (od.j + oe.j) // 2
                                JdeMin = abs(od.j - oe.j) // 2
                                if abs(od.j - oe.j) > of.j:
                                    twoJCMindownket = abs(od.j - oe.j) - of.j
                                elif of.j < (od.j + oe.j):
                                    twoJCMindownket = 1
                                else:
                                    twoJCMindownket = of.j - od.j - oe.j
                                twoJCMaxupket = od.j + oe.j + of.j

                                twoJCMindown = max(twoJCMindownbra, twoJCMindownket)
                                twoJCMaxup = min(twoJCMaxupbra, twoJCMaxupket)
                                if twoJCMindown > twoJCMaxup:
                                    continue

                                nkey = (idx_a, idx_b, idx_c, idx_d, idx_e, idx_f)
                                dict_idx_ThBME[nkey] = nread

                                for Jab in range(JabMin, JabMax + 1):
                                    for Jde in range(JdeMin, JdeMax + 1):
                                        twoJCMin = max(
                                            abs(2 * Jab - oc.j), abs(2 * Jde - of.j)
                                        )
                                        twoJCMax = min(2 * Jab + oc.j, 2 * Jde + of.j)
                                        if twoJCMin > twoJCMax:
                                            continue
                                        blocksize = ((twoJCMax - twoJCMin) // 2 + 1) * 5

                                        nread += blocksize

        if mode == "File":
            nkeys = len(dict_idx_ThBME)
            print("size of dict_idx_ThBME:", nkeys, nkeys * 2 * 8 / 1024**3, "GB")
        return nread, dict_idx_ThBME

    def read_me3jgz(self, filename, count_ME_file):
        """Read three-body matrix elements from a me3j.gz file.

        The ordering of the matrix elements are uniquely determined
        by the e*max values defining the adopted model space.
        """
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"File not found: {filename}")
        size_ME = count_ME_file * 8 / 1024**3
        total_memory = (
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
        )
        if size_ME >= 0.9 * (total_memory / 2):
            raise MemoryError(
                f"# of ThBME={size_ME} is beyond available memory: {total_memory} GB"
            )
        ThBME = np.zeros(count_ME_file, dtype=np.float64)
        with gzip.open(filename, "rt") as stream:
            for idx, line in enumerate(stream):
                if idx == 0:
                    continue
                idx_i = (idx - 1) * 10
                idx_f = idx_i + 9
                subsize = 10
                if idx_i >= count_ME_file:
                    break
                if idx_f >= count_ME_file:
                    subsize = count_ME_file - idx_i 
                for i in range(subsize):
                    tl = line[16 * i : 16 * (i + 1)]
                    ThBME[idx_i + i] = float(tl)
        print("Total ThBME entries read:", len(ThBME))
        return ThBME

    def loop_over_JT(self, sps_3b: sps_3Blab, what_you_need="", ME_is_double=True, 
                     array_in: np.ndarray=np.zeros(1, dtype=np.float64)
        ): 
        """Method to loop over JT-coupled three-body matrix elements.

        This method iterates over the relevant quantum numbers and constructs
        the necessary data structures for the three-body matrix elements.
        """
        norbits = sps_3b.norbits_ms
        sps = sps_3b.sps
        e1max = sps_3b.e1max
        e3max = sps_3b.e3max
        pnME_3NF = { }
        dict_3b_idx = {}
        total_dim = non_zero_term = 0
        # Loop over indices (assumed 1-indexed and odd numbers)
        for a in range(1, norbits + 1, 2):
            oa = sps[a]
            ea = oa.e
            la = oa.l
            if ea > e1max or ea > e3max:
                continue

            for b in range(1, a + 1, 2):
                ob = sps[b]
                eb = ob.e
                lb = ob.l
                if ea + eb > e3max:
                    continue

                Jab_min = abs(oa.j - ob.j) // 2
                Jab_max = (oa.j + ob.j) // 2

                for c in range(1, b + 1, 2):
                    oc = sps[c]
                    ec = oc.e
                    if ea + eb + ec > e3max:
                        continue

                    for d in range(1, a + 1, 2):
                        od = sps[d]
                        ed = od.e

                        # Upper limit for index e depends on whether d equals a
                        upper_idx_e = b if d == a else d
                        for e in range(1, upper_idx_e + 1, 2):
                            oe = sps[e]
                            ee = oe.e

                            # Upper limit for index f depends on a,d and b,e combinations
                            upper_idx_f = c if (a == d and b == e) else e
                            for f in range(1, upper_idx_f + 1, 2):
                                of = sps[f]
                                ef = of.e
                                if ed + ee + ef > e3max:
                                    continue

                                # Check parity condition from orbital angular momenta.
                                if (la + lb + oc.l + od.l + oe.l + of.l) % 2 != 0:
                                    continue

                                abc_eq_def = ( (a, b, c) == (d, e, f) )

                                # Record the zero-based offset of this canonical
                                # JT orbit block for direct m-scheme recoupling.
                                if what_you_need == "allocate_v3bme":
                                    orbit_hash = get_nkey6(a, b, c, d, e, f)
                                    dict_3b_idx[orbit_hash] = total_dim

                                if what_you_need == "me3j->readable.txt":

                                    Jde_min = abs(od.j - oe.j) // 2
                                    Jde_max = (od.j + oe.j) // 2
                                    for Jab in range(Jab_min, Jab_max + 1):
                                        for Jde in range(Jde_min, Jde_max + 1):
                                            J3_min = max(
                                                abs(2 * Jab - oc.j), abs(2 * Jde - of.j)
                                            )
                                            J3_max = min(2 * Jab + oc.j, 2 * Jde + of.j)
                                            for J3 in range(J3_min, J3_max + 1, 2):
                                                for Tindex in range(5):
                                                    T3 = 1 if Tindex < 4 else 3
                                                    Tab = 0 if Tindex <= 1 else 1
                                                    Tde = Tindex - 2 * Tab - ((T3-1)//2)
                                                    total_dim += 1

                                                    idx_3bme = total_dim - 1
                                                    v3val = array_in[idx_3bme]
                                                    if v3val != 0.0:
                                                        if abc_eq_def:
                                                            JTindex_bra = 2 * Jab + Tab 
                                                            JTindex_ket = 2 * Jde + Tde 
                                                            if JTindex_bra < JTindex_ket:
                                                                continue
                                                        aa = a // 2 + 1
                                                        bb = b // 2 + 1
                                                        cc = c // 2 + 1
                                                        dd = d // 2 + 1
                                                        ee = e // 2 + 1
                                                        ff = f // 2 + 1
                                                        if self.verbose:
                                                            print(f"{aa:<3d} {bb:3d} {cc:3d} [{Jab:3d} {Tab:2d}] {dd:3d} {ee:3d} {ff:3d} [{Jde:3d} {Tde:2d}] : {J3:3d} {T3:2d} v {v3val: 15.10f} Tindex {Tindex} idx_3bme {total_dim}")
                                                        Trans_JT_into_pn(self.Z, self.N, 
                                                                         aa, bb, cc, Jab, Tab, 
                                                                         dd, ee, ff, Jde, Tde, 
                                                                         J3, T3, v3val, pnME_3NF,
                                                                         self.JTorbitals,
                                                                         self.single_particle_states,
                                                                         self.CG_dict)
                                                        non_zero_term += 1

                                else:
                                    Jde_min = abs(od.j - oe.j) // 2
                                    Jde_max = (od.j + oe.j) // 2
                                    for Jab in range(Jab_min, Jab_max + 1):
                                        for Jde in range(Jde_min, Jde_max + 1):
                                            J3_min = max(
                                                abs(2 * Jab - oc.j), abs(2 * Jde - of.j)
                                            )
                                            J3_max = min(2 * Jab + oc.j, 2 * Jde + of.j)
                                            for J3 in range(J3_min, J3_max + 1, 2):
                                                total_dim += 5

        if what_you_need == "allocate_v3bme":
            size_3bme = total_dim * (8 if ME_is_double else 4) / 1024.0**3
            total_memory = psutil.virtual_memory().total / 1024**3
            if size_3bme >= 0.9 * total_memory:
                raise MemoryError(
                    f"size(3BME) {size_3bme} is beyond your environment memory {total_memory} GB"
                )
            print(f"# of 3BME: {total_dim:12d} Mem. {size_3bme:12.5e} GB")
            v3bme = np.zeros(total_dim, dtype=np.float64)
            self.dict_3b_idx = dict_3b_idx
            return v3bme
        elif what_you_need == "me3j->readable.txt":
            if self.verbose:
                print("checking dim for readable.txt...", non_zero_term, "/", total_dim)
            return pnME_3NF
        

def Trans_JT_into_pn(Z: int, N: int,
                     a: int, b: int, c: int, Jab: int, Tab: int,
                     d: int, e: int, f: int, Jde: int, Tde: int,
                     J3: int, T3: int, ME:float, pnME_3NF:dict,
                     JTorbitals: dict, single_particle_states: list,
                     CGdict: dict):
    """Transforms the JT-coupled 3NF into the proton-neutron (single-particle) one.

    Note:
        Developers should be aware that pnME_3NF dict is
        destructively updated in this method.
    """
    orb_a = JTorbitals.orbitals[a]
    orb_b = JTorbitals.orbitals[b]
    orb_c = JTorbitals.orbitals[c]
    orb_d = JTorbitals.orbitals[d]
    orb_e = JTorbitals.orbitals[e]
    orb_f = JTorbitals.orbitals[f]
    for tz_a, tz_b, tz_c, tz_d, tz_e, tz_f in itertools.product(
        [-1, 1], repeat=6
    ):
        if (tz_a + tz_b + tz_c) != (tz_d + tz_e + tz_f):
            continue
        if abs(tz_a + tz_b) > 2 * Tab:
            continue
        if abs(tz_d + tz_e) > 2 * Tde:
            continue

        pn_a = get_spsidx_from_nljtz(
            single_particle_states, orb_a.n, orb_a.l, orb_a.j, tz_a
        )
        pn_b = get_spsidx_from_nljtz(
            single_particle_states, orb_b.n, orb_b.l, orb_b.j, tz_b
        )
        pn_c = get_spsidx_from_nljtz(
            single_particle_states, orb_c.n, orb_c.l, orb_c.j, tz_c
        )
        pn_d = get_spsidx_from_nljtz(
            single_particle_states, orb_d.n, orb_d.l, orb_d.j, tz_d
        )
        pn_e = get_spsidx_from_nljtz(
            single_particle_states, orb_e.n, orb_e.l, orb_e.j, tz_e
        )
        pn_f = get_spsidx_from_nljtz(
            single_particle_states, orb_f.n, orb_f.l, orb_f.j, tz_f
        )
        Tz_bra = tz_a + tz_b + tz_c


        if Tz_bra == 3 and N < 3: # nnn
            continue
        if Tz_bra == 1 and (Z == 0 or N <= 1): # pnn
            continue
        if Tz_bra == -1 and (N == 0 or Z <= 1): # npp
            continue
        if Tz_bra == -3 and Z < 3: # ppp
            continue
        if abs(Tz_bra) > T3:
            continue

        pnkey = (pn_a, pn_b, pn_c, pn_d, pn_e, pn_f, Jab, Jde, J3)
        cg_1 = get_CGs_from_dict(
            1, tz_a, 1, tz_b, Tab * 2, (tz_a + tz_b), CGdict
        )
        cg_2 = get_CGs_from_dict(
            1, tz_d, 1, tz_e, Tde * 2, (tz_d + tz_e), CGdict
        )
        cg_3 = get_CGs_from_dict(
            Tab * 2, (tz_a + tz_b), 1, tz_c, T3, (tz_a + tz_b + tz_c), CGdict
        )
        cg_4 = get_CGs_from_dict(
            Tde * 2, (tz_d + tz_e), 1, tz_f, T3, (tz_d + tz_e + tz_f), CGdict
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
        if pnkey != (pn_d, pn_e, pn_f, pn_a, pn_b, pn_c, Jde, Jab, J3):
            pnkey = (pn_d, pn_e, pn_f, pn_a, pn_b, pn_c, Jde, Jab, J3)
            part = ME * cgfact
            if pnkey in pnME_3NF.keys():
                pnME_3NF[pnkey] += part
            else:
                pnME_3NF[pnkey] = part


def truncate_v3bme(dim_v3bme_file: int, dim_v3bme_ms: int,
                   sps_3b: sps_3Blab,
                   ThBME, dWS, 
                   dict_idxheadThBME_file,
                   dict_idxheadThBME_MS):
    """Truncate JT-coupled 3NF matrix elements with specified model space truncations.

    """
    v3bme = np.zeros(dim_v3bme_ms, dtype=np.float64)
    e1max = sps_3b.e1max
    e2max = e1max * 2
    e3max = sps_3b.e3max
    l3max = e1max
    sps = sps_3b.sps
    norbits = sps_3b.norbits_ms
    count_ME_file = 0

    for a in range(1, norbits + 1, 2):
        oa = sps[a]
        ea = oa.e
        la = oa.l
        if ea > e1max or ea > e3max:
            continue

        for b in range(1, a + 1, 2):
            ob = sps[b]
            eb = ob.e
            lb = ob.l
            if ea + eb > e3max:
                continue

            Jab_min = abs(oa.j - ob.j) // 2
            Jab_max = (oa.j + ob.j) // 2

            for c in range(1, b + 1, 2):
                oc = sps[c]
                ec = oc.e
                if ea + eb + ec > e3max:
                    continue

                for d in range(1, a + 1, 2):
                    od = sps[d]
                    ed = od.e

                    # Upper limit for index e depends on whether d equals a
                    upper_idx_e = b if d == a else d
                    for e in range(1, upper_idx_e + 1, 2):
                        oe = sps[e]
                        ee = oe.e

                        # Upper limit for index f depends on a,d and b,e combinations
                        upper_idx_f = c if (a == d and b == e) else e
                        for f in range(1, upper_idx_f + 1, 2):
                            of = sps[f]
                            ef = of.e
                            if ed + ee + ef > e3max:
                                continue

                            # Check parity condition from orbital angular momenta.
                            if (la + lb + oc.l + od.l + oe.l + of.l) % 2 != 0:
                                continue

                            tkey = (a, b, c, d, e, f)                            
                            ofst_ms = dict_idxheadThBME_MS[tkey]
                            ofst_file = dict_idxheadThBME_file[tkey]

                            idx_intr = 0
                            Jde_min = abs(od.j - oe.j) // 2
                            Jde_max = (od.j + oe.j) // 2
                            for Jab in range(Jab_min, Jab_max + 1):
                                for Jde in range(Jde_min, Jde_max + 1):
                                    J3_min = max(
                                        abs(2 * Jab - oc.j), abs(2 * Jde - of.j)
                                    )
                                    J3_max = min(2 * Jab + oc.j, 2 * Jde + of.j)
                                    for J3 in range(J3_min, J3_max + 1, 2):
                                        for Tindex in range(5):
                                            idx_3bme_file = ofst_file + idx_intr
                                            idx_3bme_ms = ofst_ms + idx_intr
                                            v3bme[idx_3bme_ms] = ThBME[idx_3bme_file]
                                            idx_intr += 1
    return v3bme


def RecouplingCG(idx_abc, ja2, jb2, jc2, Jab_in, Jab, J2, dWS) -> float:
    """Return the angular-momentum recoupling coefficient for a 3NF permutation.

    Args:
        idx_abc (int): Permutation index for the three incoming orbitals.
        ja2 (int): Twice the angular momentum of orbital a.
        jb2 (int): Twice the angular momentum of orbital b.
        jc2 (int): Twice the angular momentum of orbital c.
        Jab_in (int): Input coupled angular momentum of a and b.
        Jab (int): Output coupled angular momentum of the permuted pair.
        J2 (int): Twice the total three-body angular momentum.
        dWS (prep_dicts_for_WignerSymbols): Precomputed Wigner-symbol tables.

    Returns:
        float: Recoupling coefficient.
    """
    # Check angular momentum triangle conditions
    if abs(ja2 - jb2) // 2 > Jab or (ja2 + jb2) // 2 < Jab:
        return 0.0
    if abs(jc2 - J2) // 2 > Jab or (jc2 + J2) // 2 < Jab:
        return 0.0

    if idx_abc == 0:
        return 1.0 if Jab == Jab_in else 0.0

    elif idx_abc == 1:  # bca
        phase = (-1) ** (((jb2 + jc2) // 2) + Jab_in + 1)
        t6j = dWS.d6j_lj[get_key6j_sym(ja2, jb2, Jab * 2, jc2, J2, Jab_in * 2)]
        return phase * hat(Jab_in) * hat(Jab) * t6j

    elif idx_abc == 2:  # cab
        phase = (-1) ** (((ja2 + jb2) // 2) - Jab + 1)
        t6j = dWS.d6j_lj[get_key6j_sym(jb2, ja2, Jab * 2, jc2, J2, Jab_in * 2)]
        return phase * hat(Jab_in) * hat(Jab) * t6j

    elif idx_abc == 3:  # acb
        phase = (-1) ** (((jb2 + jc2) // 2) + Jab_in - Jab)
        t6j = dWS.d6j_lj[get_key6j_sym(jb2, ja2, Jab * 2, jc2, J2, Jab_in * 2)]
        return phase * hat(Jab_in) * hat(Jab) * t6j

    elif idx_abc == 4:  # bac
        if Jab == Jab_in:
            phase = (-1) ** (((ja2 + jb2) // 2) - Jab)
            return 1.0 * phase
        else:
            return 0.0

    elif idx_abc == 5:  # cba
        t6j = dWS.d6j_lj[get_key6j_sym(ja2, jb2, Jab * 2, jc2, J2, Jab_in * 2)]
        return -hat(Jab_in) * hat(Jab) * t6j

    else:
        raise AssertionError("This should not happen")


def _get_3bme_iso(
    e3max,
    v3bme,
    dict_3b_idx,
    sps_3b,
    Jab_in,
    Jde_in,
    J2,
    tab_in,
    tde_in,
    T2,
    a_in,
    b_in,
    c_in,
    d_in,
    e_in,
    f_in,
    dWS,
):
    """Read one recoupled JT matrix element from the packed 3BME array.

    The input orbitals may be in any order.  The file stores only the lower
    triangular, descending orbital blocks, so the required permutation phases
    and the corresponding jj/isospin recouplings are applied here.
    """
    sps = sps_3b.sps
    a, b, c, idx_abc = sort_3_orbits(a_in, b_in, c_in)
    d, e, f, idx_def = sort_3_orbits(d_in, e_in, f_in)

    if d > a or (d == a and e > b) or (d == a and e == b and f > c):
        a, b, c, d, e, f = d, e, f, a, b, c
        Jab_in, Jde_in = Jde_in, Jab_in
        tab_in, tde_in = tde_in, tab_in
        idx_abc, idx_def = idx_def, idx_abc

    orbit_offset = dict_3b_idx.get(get_nkey6(a, b, c, d, e, f), -1)
    if orbit_offset < 0:
        return 0.0

    oa, ob, oc = sps[a], sps[b], sps[c]
    od, oe, of = sps[d], sps[e], sps[f]
    if oa.e + ob.e + oc.e > e3max or od.e + oe.e + of.e > e3max:
        return 0.0

    Jab_min = abs(oa.j - ob.j) // 2
    Jab_max = (oa.j + ob.j) // 2
    Jde_min = abs(od.j - oe.j) // 2
    Jde_max = (od.j + oe.j) // 2
    tab_min = 1 if T2 == 3 else 0
    tde_min = 1 if T2 == 3 else 0
    value = 0.0
    j_index = 0

    for Jab in range(Jab_min, Jab_max + 1):
        c_j_abc = RecouplingCG(
            idx_abc, oa.j, ob.j, oc.j, Jab_in, Jab, J2, dWS
        )
        if idx_abc // 3 != idx_def // 3:
            c_j_abc *= -1.0
        for Jde in range(Jde_min, Jde_max + 1):
            c_j_def = RecouplingCG(
                idx_def, od.j, oe.j, of.j, Jde_in, Jde, J2, dWS
            )
            j2_min = max(abs(2 * Jab - oc.j), abs(2 * Jde - of.j))
            j2_max = min(2 * Jab + oc.j, 2 * Jde + of.j)
            if j2_min > j2_max:
                continue

            j_index += int((J2 - j2_min) / 2) * 5
            if j2_min <= J2 <= j2_max and abs(c_j_abc * c_j_def) > 1.0e-10:
                for tab in range(tab_min, 2):
                    c_t_abc = RecouplingCG(
                        idx_abc, 1, 1, 1, tab_in, tab, T2, dWS
                    )
                    for tde in range(tde_min, 2):
                        c_t_def = RecouplingCG(
                            idx_def, 1, 1, 1, tde_in, tde, T2, dWS
                        )
                        if abs(c_t_abc * c_t_def) < 1.0e-10:
                            continue
                        t_index = 2 * tab + tde + (T2 - 1) // 2
                        idx = orbit_offset + j_index + t_index
                        if not (0 <= idx < len(v3bme)):
                            continue
                        matrix_value = v3bme[idx]
                        if matrix_value != 0.0:
                            value += (
                                c_j_abc
                                * c_j_def
                                * c_t_abc
                                * c_t_def
                                * matrix_value
                            )
            j_index += int((j2_max - J2 + 2) / 2) * 5

    return value


def _get_v3_pn(
    e3max,
    v3bme,
    dict_3b_idx,
    sps_3b,
    dWS,
    Jab,
    Jde,
    J2,
    a,
    b,
    c,
    d,
    e,
    f,
):
    """Evaluate a 3NF matrix element for arbitrary proton/neutron ordering."""
    sps = sps_3b.sps
    tza, tzb, tzc = sps[a].tz, sps[b].tz, sps[c].tz
    tzd, tze, tzf = sps[d].tz, sps[e].tz, sps[f].tz
    dcg_spin = dWS.dcg_spin
    value = 0.0
    t_min = max(abs(tza + tzb + tzc), abs(tzd + tze + tzf))

    for tab in range(abs(tza + tzb) // 2, 2):
        cg1 = dcg_spin.get(
            get_nkey6_shift(1, tza, 1, tzb, 2 * tab, tza + tzb), 0.0
        )
        if cg1 == 0.0:
            continue
        for tde in range(abs(tzd + tze) // 2, 2):
            cg2 = dcg_spin.get(
                get_nkey6_shift(1, tzd, 1, tze, 2 * tde, tzd + tze), 0.0
            )
            if cg1 * cg2 == 0.0:
                continue
            t_max = min(1 + 2 * tab, 1 + 2 * tde)
            for total_t2 in range(t_min, t_max + 1, 2):
                cg3 = dcg_spin.get(
                    get_nkey6_shift(
                        2 * tab, tza + tzb, 1, tzc,
                        total_t2, tza + tzb + tzc,
                    ),
                    0.0,
                )
                cg4 = dcg_spin.get(
                    get_nkey6_shift(
                        2 * tde, tzd + tze, 1, tzf,
                        total_t2, tzd + tze + tzf,
                    ),
                    0.0,
                )
                if cg3 * cg4 == 0.0:
                    continue
                value += (
                    cg1
                    * cg2
                    * cg3
                    * cg4
                    * _get_3bme_iso(
                        e3max,
                        v3bme,
                        dict_3b_idx,
                        sps_3b,
                        Jab,
                        Jde,
                        J2,
                        tab,
                        tde,
                        total_t2,
                        a,
                        b,
                        c,
                        d,
                        e,
                        f,
                        dWS,
                    )
                )
    return value


class prep_dicts_for_WignerSymbols:
    """Precompute Clebsch-Gordan and Wigner 6j lookup tables."""

    def __init__(self, emax):
        self.emax = emax
        self.jmax = 2 * emax + 1
        self.d6j_int = self.prep_d6j_int(emax, self.jmax)
        self.dcg_spin = self.prep_dcg_spin()
        self.d6j_lj = self.prep_d6j_lj(self.jmax)

    def prep_dcg_spin(self):
        """Precompute spin Clebsch-Gordan coefficients used in 3NF recoupling."""
        dcg_spin = {}
        s_a = 1
        s_b = 1
        for sz_a in [-1, 1]:
            for sz_b in [-1, 1]:
                for Sab in [0, 1]:
                    Sabzmin = 0 if Sab == 0 else -1
                    for Sab_z in range(Sabzmin, Sab + 1):
                        nkey = get_nkey6_shift(s_a, sz_a, s_b, sz_b, Sab * 2, Sab_z * 2)
                        dcg_spin[nkey] = float(
                            ClebschGordan(
                                s_a / 2, sz_a / 2, s_b / 2, sz_b / 2, Sab, Sab_z
                            ).doit()
                        )
        s_c = 1
        for S_ab in range(0, 2):  # 0, 1
            for S_ab_z in range(-S_ab, S_ab + 1):
                for s_c_z in range(-1, 2, 2):  # -1, 1
                    S3min = 1
                    S3max = 1 if S_ab == 0 else 3
                    for S3 in range(S3min, S3max + 1, 2):
                        for S3z in range(-S3, S3 + 1, 2):
                            nkey = get_nkey6_shift(
                                S_ab * 2, S_ab_z * 2, s_c, s_c_z, S3, S3z
                            )
                            dcg_spin[nkey] = float(
                                ClebschGordan(
                                    S_ab, S_ab_z, s_c / 2, s_c_z / 2, S3 / 2, S3z / 2
                                ).doit()
                            )
        return dcg_spin

    def prep_d6j_lj(self, jmax2):
        """Precompute 6j symbols for l-j recoupling up to ``jmax2``."""
        d6j_lj = {}
        for j1 in range(1, jmax2 + 1, 2):
            for j2 in range(1, jmax2 + 1, 2):
                for J12 in range(abs(j1 - j2), j1 + j2 + 1, 2):
                    for j3 in range(1, jmax2 + 1, 2):
                        for J23 in range(abs(j2 - j3), j2 + j3 + 1, 2):
                            start_J = max(abs(j1 - J23), abs(j3 - J12))
                            end_J = min(j1 + J23, j3 + J12)
                            for J in range(start_J, end_J + 1, 2):
                                nkey = get_key6j_sym(j1, j2, J12, j3, J, J23)
                                d6j_lj[nkey] = wigner_6j(
                                    j1 / 2, j2 / 2, J12 / 2, j3 / 2, J / 2, J23 / 2
                                )
        # Special case for kinetic_tb
        for j2 in range(1, jmax2 + 1, 2):
            J12 = 2
            J23 = 1
            for j1 in range(abs(j2 - J12), j2 + J12 + 1, 2):
                for l1 in range(abs(j1 - 1), j1 + 1 + 1, 2):
                    for l2 in range(abs(j2 - 1), j2 + 1 + 1, 2):
                        nkey = get_key6j_sym(j2, j1, J12, l1, l2, J23)
                        d6j_lj[nkey] = wigner_6j(
                            j2 / 2, j1 / 2, J12 / 2, l1 / 2, l2 / 2, J23 / 2
                        )
        return d6j_lj

    def prep_d6j_int(self, emax, jmax_in):
        """Precompute integer-grid 6j symbols for 3NF transformations."""
        # 'to' parameter is not used in this function.
        jmax = jmax_in * 2
        d6j_int = {}

        for J12 in range(0, jmax + 1, 2):
            for j1 in range(0, jmax_in + 1, 2):
                for j2 in range(abs(J12 - j1), j1 + J12 + 1, 2):
                    for j3 in range(0, jmax + 1, 2):
                        for J23 in range(abs(j2 - j3), j2 + j3 + 1, 2):
                            for J in range(abs(J23 - j1), J23 + j1 + 1, 2):
                                # Check the triangle condition. Note the division by 2.
                                if not tri_check(J / 2, J12 / 2, j3 / 2):
                                    continue
                                # Check the inequality condition.
                                if not (j1 + j3 <= j2 + J <= J12 + J23):
                                    continue
                                # Compute the key.
                                nkey = get_key6j_sym(j1, j2, J12, j3, J, J23)
                                # Compute the six-j symbol value.
                                value = wigner_6j(
                                    j1 / 2, j2 / 2, J12 / 2, j3 / 2, J / 2, J23 / 2
                                )
                                d6j_int[nkey] = value
        return d6j_int


def tri_check(a, b, c):
    """Return True if three angular momenta satisfy triangle inequalities."""
    if a + b < c or a + c < b or b + c < a:
        return False
    if abs(a - b) > c or abs(a - c) > b or abs(b - c) > a:
        return False
    return True


def get_key6j_sym(j1: int, j3: int, j5: int, j2: int, j4: int, j6: int) -> int:
    """Return a symmetry-canonical packed key for a Wigner 6j symbol."""
    # Initialize temporary copies.
    tj1, tj3, tj5 = j1, j3, j5
    tj2, tj4, tj6 = j2, j4, j6
    # Assume get_canonical_order_6j is implemented elsewhere.
    column_order = get_canonical_order_6j(j1, j2, j3, j4, j5, j6)
    if column_order == 231:
        tj1, tj2, tj3, tj4, tj5, tj6 = tj3, tj4, tj5, tj6, tj1, tj2
    elif column_order == 132:
        tj3, tj4, tj5, tj6 = tj5, tj6, tj3, tj4
    elif column_order == 213:
        tj1, tj2, tj3, tj4 = tj3, tj4, tj1, tj2
    elif column_order == 312:
        tj1, tj2, tj3, tj4, tj5, tj6 = tj5, tj6, tj1, tj2, tj3, tj4
    elif column_order == 321:
        tj1, tj2, tj5, tj6 = tj5, tj6, tj1, tj2

    # If any column has equal entries then swap to enforce tjX <= tjY.
    if tj1 == tj2 or tj3 == tj4 or tj5 == tj6:
        if tj1 > tj2:
            tj1, tj2 = tj2, tj1
        if tj3 > tj4:
            tj3, tj4 = tj4, tj3
        if tj5 > tj6:
            tj5, tj6 = tj6, tj5
        return get_nkey6(tj1, tj3, tj5, tj2, tj4, tj6)
    else:
        tint = (
            (1 if tj1 < tj2 else 0) + (1 if tj3 < tj4 else 0) + (1 if tj5 < tj6 else 0)
        )
        if tint == 0 or tint == 3:
            return get_nkey6(tj1, tj3, tj5, tj2, tj4, tj6)
        if (tj5 < tj6 and tint == 1) or (tj5 > tj6 and tint == 2):
            return get_nkey6(tj2, tj4, tj5, tj1, tj3, tj6)
        elif (tj3 < tj4 and tint == 1) or (tj3 > tj4 and tint == 2):
            return get_nkey6(tj2, tj3, tj6, tj1, tj4, tj5)
        elif (tj1 < tj2 and tint == 1) or (tj1 > tj2 and tint == 2):
            return get_nkey6(tj1, tj4, tj6, tj2, tj3, tj5)
    raise RuntimeError("This never happens.")


def j_col_score(j1: int, j2: int) -> int:
    """Score one 6j-symbol column for canonical ordering."""
    return 100 * (j1 + j2) + min(j1, j2)


def get_canonical_order_6j(j1: int, j2: int, j3: int, j4: int, j5: int, j6: int) -> int:
    """Return the canonical column order code for a 6j-symbol key."""
    cscore_12 = j_col_score(j1, j2)
    cscore_34 = j_col_score(j3, j4)
    cscore_56 = j_col_score(j5, j6)

    if cscore_12 <= cscore_34:
        if cscore_34 <= cscore_56:
            return 123
        elif cscore_56 < cscore_12:
            return 312
        else:
            return 132
    else:
        if cscore_56 <= cscore_34:
            return 321
        elif cscore_12 <= cscore_56:
            return 213
        else:
            return 231


def set_op_list_from_op_dict_3b(
    op_dict_3b, n_qubits_p, n_qubits_n, method="Jordan-Wigner",
):
    """Map three-body force dictionary to Pauli operator list using parallel processing.
    
    Efficiently converts a dictionary of proton-neutron operator pairs to a list
    of Pauli operators using the specified fermion-to-qubit mapping. Employs
    multiprocessing for computational efficiency on large three-body matrices.
    
    Args:
        op_dict_3b (dict): Dictionary of (proton_op, neutron_op) -> coefficient pairs.
        n_qubits_p (int): Number of proton qubits.
        n_qubits_n (int): Number of neutron qubits.
        method (str, optional): Fermion-to-qubit mapping method. Defaults to "Jordan-Wigner".
        
    Returns:
        list: List of (pauli_label, coefficient) tuples representing the mapped operators.
        
    Note:
        Uses multiprocessing with fork context for efficient parallel processing.
        The number of processes is set to cpu_count - 2 to avoid system overload.
    """
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


# def worker_cU(inp):
#     term, dt, n_qubits = inp
#     term = term.to_label()
#     op = SparsePauliOp.from_list(list(zip([term], [1.0])))
#     u = PauliEvolutionGate(op, dt, synthesis=SuzukiTrotter(order=1, reps=1))
#     u = u.control(1)
#     qc = QuantumCircuit(n_qubits + 1)
#     qc.append(u, [0] + list(range(1, n_qubits + 1)))
#     qc = qc.decompose(reps=5)
#     counts = qc.count_ops()
#     return counts


# def worker_U(inp):
#     term, dt, n_qubits = inp
#     term = term.to_label()
#     op = SparsePauliOp.from_list(list(zip([term], [1.0])))
#     u = PauliEvolutionGate(op, dt, synthesis=SuzukiTrotter(order=1, reps=1))
#     qc = QuantumCircuit(n_qubits)
#     qc.append(u, range(n_qubits))
#     qc = qc.decompose(reps=5)
#     counts = qc.count_ops()
#     return counts


# def counting_CNOTs(H_mapped, label_int, label_op, dt, n_qubits):
#     print(f"Counting CNOTs for {label_int} with {label_op} operator...")
#     nproc = multiprocessing.cpu_count() - 2
#     Ntasks = len(H_mapped.paulis)
#     t = tqdm(total=Ntasks)
#     p = get_context("fork").Pool(processes=nproc)
#     if label_op == "c-U":
#         worker_func = worker_cU
#     elif label_op == "U":
#         worker_func = worker_U
#     else:
#         raise ValueError(f"Unknown label_op: {label_op}. Use 'c-U' or 'U'.")

#     results_all = {}
#     with get_context("fork").Pool(processes=nproc) as p:
#         results = []
#         for counts in p.imap_unordered(
#             worker_func,
#             [(term, dt, n_qubits) for term in H_mapped.paulis],
#             chunksize=1000,
#         ):
#             results.append(counts)
#             t.update()
#     total_counts = Counter()
#     for counts in results:
#         total_counts.update(counts)
#     results_all = dict(total_counts)

#     p.close()
#     print(f"{label_int}: # of gates in (multi-processing) {label_op}: {results_all}")


def removing_redundant_ops(op_list):
    """Remove redundant Pauli operators by combining terms with same labels.
    
    Consolidates a list of Pauli operators by combining coefficients for
    operators with identical Pauli labels, removing near-zero terms.
    
    Args:
        op_list (list): List of (pauli_label, coefficient) tuples.
        
    Returns:
        SparsePauliOp: Consolidated Pauli operator with redundant terms removed.
        
    Note:
        Terms with coefficients smaller than 1e-16 are discarded to avoid
        numerical precision issues in quantum simulations.
    """
    print("Removing redundant terms...", end=" ")
    new_dict = {}
    num_input = 0
    for label, c in op_list:
        num_input += 1
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
    print(f"len= {num_input} => {len(ops)}")
    return SparsePauliOp.from_list(list(zip(ops, coeffs)))


def removing_redundant_terms(ops: SparsePauliOp, verbose=False):
    """Remove redundant terms from existing SparsePauliOp.
    
    Similar to removing_redundant_ops but works on existing SparsePauliOp objects.
    Combines terms with identical Pauli labels and removes negligible coefficients.
    
    Args:
        ops (SparsePauliOp): Input Pauli operator.
        verbose (bool, optional): If True, print the retained Pauli strings and
            coefficients after consolidation.
        
    Returns:
        SparsePauliOp: Cleaned Pauli operator with redundant terms combined.
        
    Note:
        This is used for final cleanup of mapped Hamiltonians before
        quantum algorithm applications. Without this step, imaginary
        terms may persist and affect the results.
    """
    if len(ops) <= 1:
        return ops
    paulis = ops.paulis
    coeffs = ops.coeffs
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
    tmp = SparsePauliOp.from_list(list(zip(ops, coeffs)))
    tmp = sum(tmp).simplify()
    print("Removing redundant terms... len=", len(paulis) , " => ", len(tmp.paulis))
    return tmp


def sum_over_J(Hamil):
    """Sum two-body matrix elements over total angular momentum J.
    
    Combines two-body matrix elements with the same orbital indices but
    different total angular momentum quantum numbers, which is needed
    for certain nuclear structure calculations.
    
    Args:
        Hamil (dict): Dictionary of Hamiltonian components with matrix element lists.
        
    Returns:
        dict: Modified Hamiltonian dictionary with J-summed matrix elements.
        
    Note:
        Single-particle energies (SPE) are passed through unchanged.
        Two-body terms (Vpp, Vnn, Vpn) are combined over J quantum numbers.
    """
    new_Hamil = {}
    for key in Hamil.keys():
        if key == "SPE" or key == "P" or key == "N" or key == "Jz":
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
