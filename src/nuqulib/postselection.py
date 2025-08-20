"""Post-selection techniques for quantum nuclear simulations.

This module provides post-selection and measurement analysis tools for quantum
nuclear physics simulations, including particle number conservation constraints
and expectation value calculations for different types of Hamiltonians.
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from .ansatz import pair_ansatz_qiskit


def reweight_results(
    sampler_results,
    Norb,
    Nocc,
    normalize=True,
    verbose=False,
    make_number_conservation=True,
):
    """Reweight quantum measurement results to enforce particle number conservation.
    
    This function filters and reweights measurement results to include only
    configurations with the correct number of particles, which is essential
    for nuclear quantum simulations.
    
    Args:
        sampler_results (dict): Raw measurement results from quantum simulation.
        Norb (int): Total number of orbitals (qubits).
        Nocc (int): Number of occupied orbitals (target particle number).
        normalize (bool, optional): Whether to normalize results to unity. 
                                   Defaults to True.
        verbose (bool, optional): Whether to print valid counts ratio. 
                                 Defaults to False.
        make_number_conservation (bool, optional): Whether to enforce particle 
                                                  number conservation. Defaults to True.
    
    Returns:
        dict: Reweighted measurement results with proper particle number.
        
    Note:
        Automatically handles both bitstring and hexadecimal input formats.
    """

    # Checking results are in bitstring or hexadecimal format. if hexadecimal, convert to bitstring
    tkey = list(sampler_results.keys())[0]
    if "x" in tkey:
        sampler_results = {
            format(int(key, 0), "0" + str(Norb) + "b"): val
            for key, val in sampler_results.items()
        }

    if verbose:
        print(
            "Valid counts ratio: ",
            sum([val for key, val in sampler_results.items() if key.count("1") == Nocc])
            / sum(sampler_results.values()),
        )

    reweighted_results = {}
    for key, val in sampler_results.items():
        num_one = key.count("1")
        if num_one == Nocc and make_number_conservation:
            reweighted_results[key] = val

    if normalize:
        norm = sum(reweighted_results.values())
        for key in reweighted_results.keys():
            reweighted_results[key] /= norm

        if not (make_number_conservation):
            norm = sum(sampler_results.values())
            reweighted_results = {
                key: val / norm for key, val in sampler_results.items()
            }

    return reweighted_results


def eval_expec_diag_postselection(
    hamiltonian_op_diag: SparsePauliOp, counts: dict, Nq, Nocc
):
    """Evaluate expectation value of diagonal Hamiltonian terms with post-selection.
    
    Computes the expectation value of Hamiltonian terms that are diagonal in the
    computational basis (I and Z terms) using post-selected measurement results.
    
    Args:
        hamiltonian_op_diag (SparsePauliOp): Hamiltonian with only diagonal terms.
        counts (dict): Dictionary of measurement results.
        Nq (int): Number of qubits.
        Nocc (int): Number of occupied orbitals for post-selection.
    
    Returns:
        float: Expectation value of the diagonal Hamiltonian terms.
    """
    counts = reweight_results(counts, Nq, Nocc, normalize=True)
    expec_val = 0.0
    for op, coeff in zip(hamiltonian_op_diag.paulis, hamiltonian_op_diag.coeffs):
        op = str(op)
        idx_Z = [idx for idx, p in enumerate(op) if p == "Z"]
        for config, prob in counts.items():
            fac = 1.0
            for idx in idx_Z:
                fac *= (-1) ** int(config[idx])
            expec_val += float(np.real(coeff)) * prob * fac
    return expec_val


def eval_expec_XXYY_postselection(
    hamiltonian_op_XXYY,
    counts,
    Nq,
    Nocc,
    p,
    q,
    further_postselection_on_particle_number=False,
):
    """Evaluate expectation value of XX+YY Hamiltonian terms with post-selection.
    
    Computes the expectation value of XX and YY Pauli terms using measurement
    results from Givens rotation circuits. This is used for calculating
    off-diagonal matrix elements in nuclear Hamiltonians.
    
    Args:
        hamiltonian_op_XXYY (SparsePauliOp): Hamiltonian with XX and YY terms.
        counts (dict): Dictionary of measurement results.
        Nq (int): Number of qubits.
        Nocc (int): Number of occupied orbitals.
        p (int): Index of first qubit in XX/YY term.
        q (int): Index of second qubit in XX/YY term.
        further_postselection_on_particle_number (bool, optional): Whether to 
                                                                  enforce strict particle number conservation. 
                                                                  Defaults to False.
    
    Returns:
        float: Expectation value of the XX+YY terms.
        
    Note:
        - XX and YY terms should have identical coefficients in physical Hamiltonians
        - Index ordering: p < q for Pauli strings, but expectation value uses Zq-Zp
        - This function processes only the XX terms and assumes matching YY coefficients
    """
    counts = reweight_results(
        counts,
        Nq,
        Nocc,
        normalize=True,
        make_number_conservation=further_postselection_on_particle_number,
    )
    expec_val = 0.0
    for op, coeff in zip(hamiltonian_op_XXYY.paulis, hamiltonian_op_XXYY.coeffs):
        op = str(op)
        idx_XX = [idx for idx, pauli in enumerate(op) if pauli == "X"]
        if [p, q] != idx_XX:
            continue
        for config, prob in counts.items():
            expZp = (-1) ** int(config[p])
            expZq = (-1) ** int(config[q])
            expec_val += (
                float(np.real(coeff)) * prob * (-(expZp - expZq))
            )  # for minus sign, see the note above
    return expec_val


def even_swap(labeling):
    """Swap elements at even indices with their next neighbors.
    
    Args:
        labeling (list): Input list to perform swaps on.
        
    Returns:
        list: New list with even-indexed elements swapped.
        
    Example:
        >>> even_swap([0, 1, 2, 3, 4, 5])
        [1, 0, 3, 2, 5, 4]
    """
    new = labeling.copy()
    for i in range(0, len(labeling) - 1, 2):
        new[i] = labeling[i + 1]
        new[i + 1] = labeling[i]
    return new


def odd_swap(labeling):
    """Swap elements at odd indices with their next neighbors.
    
    Args:
        labeling (list): Input list to perform swaps on.
        
    Returns:
        list: New list with odd-indexed elements swapped.
        
    Example:
        >>> odd_swap([0, 1, 2, 3, 4, 5])
        [0, 2, 1, 4, 3, 5]
    """
    new = labeling.copy()
    for i in range(1, len(labeling) - 1, 2):
        new[i] = labeling[i + 1]
        new[i + 1] = labeling[i]
    return new


def eval_Ediag(
    adopted,
    Nq,
    Nocc,
    hamiltonian_op_diag: SparsePauliOp,
    qc_ansatz: QuantumCircuit,
    backend,
    sampler,
    nshot,
    postselection_diag=True,
):
    """Evaluate diagonal energy terms using quantum circuit measurements.
    
    Runs a quantum circuit to measure diagonal Hamiltonian terms and computes
    their expectation values with optional post-selection.
    
    Args:
        adopted (str): Simulation mode ("simNISQ", "Real", etc.).
        Nq (int): Number of qubits.
        Nocc (int): Number of occupied orbitals.
        hamiltonian_op_diag (SparsePauliOp): Diagonal Hamiltonian terms.
        qc_ansatz (QuantumCircuit): Ansatz circuit to evaluate.
        backend: Quantum backend for transpilation/execution.
        sampler: Quantum sampler for measurements.
        nshot (int): Number of measurement shots.
        postselection_diag (bool, optional): Whether to apply particle number 
                                           post-selection. Defaults to True.
    
    Returns:
        float: Expectation value of diagonal Hamiltonian terms.
    """
    qc_ansatz = qc_ansatz.decompose(reps=5)
    qc_ansatz.measure_all()
    if adopted == "simNISQ":
        qc_ansatz = transpile(qc_ansatz, backend)
    elif adopted == "Real":
        pm = generate_preset_pass_manager(backend=backend)
        qc_ansatz = pm.run(qc_ansatz)

    if adopted == "simNISQ":
        job = sampler.run([qc_ansatz], shots=nshot)
        results = job.result().results
        counts = results[0].data.counts
    else:
        job = sampler.run([qc_ansatz], shots=nshot)
        results = job.result()
        counts = results[0].data.meas.get_counts()
    counts_ansatz = reweight_results(
        counts, Nq, Nocc, make_number_conservation=postselection_diag
    )
    Ediag = eval_expec_diag_postselection(hamiltonian_op_diag, counts_ansatz, Nq, Nocc)
    return Ediag


def eval_Energy_using_GoogleCircuit(
    Nq: int,
    Nocc: int,
    hamiltonian_op_XXYY: SparsePauliOp,
    qc_list_XXYY: list,
    sampler,
    nshot: int,
    num_experiment: int,
    using_noisy_simulation: bool,
    postselection_XXYY: bool = True,
    adopted: str = "simFTQC",
    debug_mode: bool = False,
    verbose: bool = False,
):
    """Evaluate XX+YY energy terms using Google's circuit approach.
    
    This function evaluates the expectation values of XX and YY Hamiltonian terms
    using the circuit design from Google's quantum simulation approach. It handles
    the specific qubit ordering and measurement strategies for optimal performance.
    
    Args:
        Nq (int): Number of qubits.
        Nocc (int): Number of occupied orbitals.
        hamiltonian_op_XXYY (SparsePauliOp): Hamiltonian with XX+YY terms.
        qc_list_XXYY (list): List of quantum circuits for measuring XX+YY terms.
        sampler: Quantum sampler for measurements.
        nshot (int): Number of measurement shots per circuit.
        num_experiment (int): Number of experimental repetitions.
        using_noisy_simulation (bool): Whether using noisy quantum simulation.
        postselection_XXYY (bool, optional): Whether to apply post-selection. 
                                            Defaults to True.
        adopted (str, optional): Simulation mode. Defaults to "simFTQC".
        debug_mode (bool, optional): Enable debug output. Defaults to False.
        verbose (bool, optional): Enable verbose output. Defaults to False.
    
    Returns:
        list: Energy values from multiple experimental runs.
        
    Note:
        - Even circuits measure XX+YY for i=0,2,4,... 
        - Odd circuits measure XX+YY for i=1,3,5,...
        - For odd Nq systems, special handling of X_0X_1 term is required
        - Qubit indices are flipped due to Qiskit ordering conventions
    """
    E_Google = []
    for exper in range(num_experiment):
        job = sampler.run(qc_list_XXYY, shots=nshot)
        results = job.result()
        if using_noisy_simulation:
            results = results.results
        labeling = list(range(Nq))
        E_XXYY = 0.0
        idx_pool = []
        for idx_cirq in range(len(qc_list_XXYY)):
            eo_pair = idx_cirq % 2
            idxs_operator = [Nq - 1 - tmp for tmp in labeling]
            if using_noisy_simulation:
                counts = results[idx_cirq].data.counts
            else:
                counts = results[idx_cirq].data.meas.get_counts()
            counts = reweight_results(
                counts,
                Nq,
                Nocc,
                verbose=(verbose and exper == 0),
                make_number_conservation=postselection_XXYY,
            )
            if debug_mode:
                print("\ncirc_idx: ", idx_cirq, "labeling", labeling)
                print("idxs_operator: ", idxs_operator)
                # print("counts for idx="+str(idx), counts)

            idx_pair_start = Nq - 2 - eo_pair if Nq % 2 == 0 else Nq - 3 + eo_pair
            for idx_pair in range(idx_pair_start, -1, -2):
                i = idx_pair
                j = idx_pair + 1
                p = idxs_operator[i]
                q = idxs_operator[j]
                if set([p, q]) in idx_pool:
                    continue
                idx_pool.append(set([p, q]))
                expec_val = coeff_XXYY = 0.0
                for op, coeff in zip(
                    hamiltonian_op_XXYY.paulis, hamiltonian_op_XXYY.coeffs
                ):
                    op = str(op)
                    idx_XX = [idx__ for idx__, pauli in enumerate(op) if pauli == "X"]
                    if set([p, q]) == set(idx_XX):
                        coeff_XXYY = coeff
                        break
                if coeff_XXYY is None:
                    print("idx_pair: ", p, q)
                    Warning(
                        "XXYY term on the qubits above was not found! You should check the Hamiltonian operator."
                    )
                for config, prob in counts.items():
                    expZi = (-1) ** int(config[Nq - 1 - i])
                    expZj = (-1) ** int(config[Nq - 1 - j])
                    expec_val += float(np.real(coeff_XXYY)) * prob * (expZi - expZj)
                E_XXYY += expec_val
                if debug_mode:  # and expec_val != 0.0:
                    print(
                        "Debug mode(G): <XXYY>_{%d, %d} = %f" % (p, q, expec_val),
                        "(i,j)= ",
                        i,
                        j,
                        "coeff_op: ",
                        coeff_XXYY,
                    )

            if idx_cirq > 0 and idx_cirq % 2 == 1:
                labeling = even_swap(labeling)
                labeling = odd_swap(labeling)
        E_Google.append(E_XXYY)
    return np.array(E_Google)


def single_eval_XXYY_Google(
    Nq,
    Nocc,
    qc_list_XXYY,
    list_counts_G,
    hamiltonian_op_XXYY: SparsePauliOp,
    postselection_XXYY: bool,
):
    E_XXYY = 0.0
    labeling = list(range(Nq))
    idx_pool = []
    for idx_cirq in range(len(qc_list_XXYY)):
        eo_pair = idx_cirq % 2
        idx_pair_start = Nq - 2 - eo_pair if Nq % 2 == 0 else Nq - 3 + eo_pair
        idxs_operator = [Nq - 1 - tmp for tmp in labeling]
        counts = list_counts_G[idx_cirq]
        reweight_results(counts, Nq, Nocc, make_number_conservation=postselection_XXYY)
        for idx_pair in range(idx_pair_start, -1, -2):
            i = idx_pair
            j = idx_pair + 1
            p = idxs_operator[i]
            q = idxs_operator[j]
            if set([p, q]) in idx_pool:
                continue
            idx_pool.append(set([p, q]))
            expec_val = coeff_XXYY = 0.0
            for op, coeff in zip(
                hamiltonian_op_XXYY.paulis, hamiltonian_op_XXYY.coeffs
            ):
                op = str(op)
                idx_XX = [idx__ for idx__, pauli in enumerate(op) if pauli == "X"]
                if set([p, q]) == set(idx_XX):
                    coeff_XXYY = coeff
                    break
            if coeff_XXYY is None:
                print("idx_pair: ", p, q)
                Warning(
                    "XXYY term on the qubits above was not found! You should check the Hamiltonian operator."
                )
            for config, prob in counts.items():
                expZi = (-1) ** int(config[Nq - 1 - i])
                expZj = (-1) ** int(config[Nq - 1 - j])
                expec_val += float(np.real(coeff_XXYY)) * prob * (expZi - expZj)
            E_XXYY += expec_val

        if idx_cirq > 0 and idx_cirq % 2 == 1:
            labeling = even_swap(labeling)
            labeling = odd_swap(labeling)
    return E_XXYY


def eval_XXYY_w_basis_rotation(
    adopted,
    params,
    Nq,
    Nocc,
    hamiltonian_op_XXYY,
    sampler,
    nshot,
    using_noisy_simulation,
    postselection_XXYY=True,
    backend=None,
    debug_mode: bool = False,
    verbose: bool = False,
    type_of_ansatz="pUCCD",
    idxs_hole_in=[],
):
    """
    Evaluate the expectation value of the Hamiltonian with XX+YY terms using
    basis rotations to diagonalize XX+YY term.

    It should be noted that Pauli strings for Hamiltonians are used "as it is" in the calculation,
    i.e. X_pX_q means "II...IX...XI...II" where the qubit p and q are involved in the X operator.
    However, the qubits in the ansatz are indexed in descending order, i.e. qubit 0 is the most significant bit.
    Hence, the indices for Pauli strings, p, q, should be converted to the indices in the ansatz, Nq-q-1, Nq-p-1,
    where the ordering comes from the fact we are assuming p < q in the Pauli strings.
    """
    qc_list = []
    relevant_idxs = []
    for i in range(Nq):
        for j in range(i + 1, Nq):
            relevant_idxs.append([i, j])
            # We should be careful about difference between the indices of the qubits in Pauli strings and those in the ansatz
            rotation_XXYY = [Nq - j - 1, Nq - i - 1]

            qc = pair_ansatz_qiskit(
                params,
                Nq,
                Nocc,
                method=type_of_ansatz,
                decent_order=True,
                rotation_XXYY=rotation_XXYY,
                idxs_hole_in=idxs_hole_in,
            )
            qc.measure_all()
            qc = qc.decompose(reps=5)
            if using_noisy_simulation:
                qc = transpile(qc, backend)
            if adopted == "Real":
                pm = generate_preset_pass_manager(backend=backend)
                qc = pm.run(qc)
            qc_list.append(qc)
    job = sampler.run(qc_list, shots=nshot)
    results = job.result()
    if using_noisy_simulation:
        results = results.results
    E_XXYY = 0.0
    for idx in range(len(results)):
        if using_noisy_simulation:
            counts = results[idx].data.counts
        else:
            counts = results[idx].data.meas.get_counts()
        counts = reweight_results(
            counts, Nq, Nocc, verbose=verbose, make_number_conservation=True
        )
        i, j = relevant_idxs[idx]
        tmp = eval_expec_XXYY_postselection(
            hamiltonian_op_XXYY,
            counts,
            Nq,
            Nocc,
            i,
            j,
            further_postselection_on_particle_number=postselection_XXYY,
        )
        E_XXYY += tmp
        if debug_mode and tmp != 0.0:
            print("Debug mode(R): E_XXYY_{%d, %d} = %f" % (i, j, tmp))
            print("Debug mode(R): counts: ", counts)
    return E_XXYY


def eval_XXYY_w_Hgates(
    adopted,
    Nq,
    Nocc,
    hamiltonian_op_XXYY,
    qc_ansatz,
    sampler,
    nshot,
    using_noisy_simulation=False,
    backend=None,
    return_counts_as_well=False,
):
    qc = qc_ansatz.copy()
    qc.h(range(Nq))
    qc.measure_all()
    qc = qc.decompose(reps=5)
    if using_noisy_simulation:
        qc = transpile(qc, backend)
    if adopted == "Real":
        pm = generate_preset_pass_manager(backend=backend)
        qc = pm.run(qc)
    job = sampler.run([qc], shots=nshot)
    results = job.result()
    if using_noisy_simulation:
        results = results.results
    if using_noisy_simulation:
        counts = results[0].data.counts
    else:
        counts = results[0].data.meas.get_counts()
    counts = reweight_results(counts, Nq, Nocc, make_number_conservation=False)
    return eval_XXYY_from_Hcounts(
        counts, Nq, hamiltonian_op_XXYY, return_counts_as_well=return_counts_as_well
    )


def eval_XXYY_from_Hcounts(counts, Nq, hamiltonian_op_XXYY, return_counts_as_well):
    exp_XXYY = np.zeros(Nq * (Nq - 1) // 2, dtype=np.float64)
    hit = 0
    for op, coeff in zip(hamiltonian_op_XXYY.paulis, hamiltonian_op_XXYY.coeffs):
        op = str(op)
        if not ("X" in op):
            continue
        i, j = [idx for idx, pauli in enumerate(op) if pauli == "X"]
        for config, prob in counts.items():
            Zp = (-1) ** int(config[i])
            Zq = (-1) ** int(config[j])
            tmp = prob * Zp * Zq * np.real(coeff) * 2
            exp_XXYY[hit] += tmp
        hit += 1
    E_XXYY = np.sum(exp_XXYY)
    if return_counts_as_well:
        return E_XXYY, counts
    return E_XXYY
