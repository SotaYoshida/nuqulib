"""Nakanishi-Fujii-Todo (NFT) optimization method for VQE.

This module implements the Nakanishi-Fujii-Todo optimization algorithm
for Variational Quantum Eigensolver (VQE) problems. The NFT method is
a derivative-free and sequential optimization algorithm that uses
discrete Fourier transform to optimize the gate angles by exploiting
the periodic structure on the parameter space.

Ref: `KM.Nakanishi, K.Fujii, T.Todo, Phys. Rev. Research 2, 043158 (2020), <https://doi.org/10.1103/PhysRevResearch.2.043158>`_
"""

from collections.abc import Iterable
import matplotlib.pyplot as plt
import numpy as np
from qiskit.primitives import StatevectorEstimator
from qiskit_aer.primitives import Estimator as AerEstimator
from scipy.optimize import minimize_scalar
import seaborn as sns
cols = sns.color_palette("deep")
from .ansatz import pair_ansatz_qiskit, nucl_ansatz


def draw_DFT_curve(xplot, DFTcurve, spot, Es, x_min, y_min):
    """Draw DFT curve with measured points and minimum candidate.
    
    Args:
        xplot (array): X-axis values for plotting.
        DFTcurve (array): DFT curve values.
        spot (array): Measurement points.
        Es (array): Energy values at measurement points.
        x_min (float): X-coordinate of minimum candidate.
        y_min (float): Y-coordinate of minimum candidate.
    """
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(xplot, DFTcurve, label="DFT curve", color=cols[0])
    ax.scatter(spot, Es, marker="o", label="Measured", color=cols[1])
    ax.scatter(x_min, y_min, marker="*", color=cols[3], label="Candidate")
    ax.legend()
    plt.show()
    plt.close()


def eval_DFT_coeff(Es, spot, theta_scale_for_DFT: float=1.0):
    """Evaluate discrete Fourier transform coefficients.
    
    Computes DFT coefficients from energy measurements at specific points
    to reconstruct the periodic energy landscape.
    
    Args:
        Es (array): Energy measurements.
        spot (array): Parameter values where energies were measured.
        theta_scale_for_DFT (float): Optional scaling factor for theta in DFT.
    
    Returns:
        array: DFT coefficients.
    """
    N = len(Es)
    DFT_coef = np.zeros(N, dtype=float)
    for idx_M in range(N):
        t_data = Es[idx_M]
        for idx_coeff in range(N):
            if idx_coeff == N - 1:
                DFT_coef[idx_coeff] += t_data
                continue
            theta_order = idx_coeff // 2 + 1
            if idx_coeff % 2 == 0:
                DFT_coef[idx_coeff] += (
                    2 * t_data * np.cos(theta_order * spot[idx_M] * theta_scale_for_DFT)
                )
            else:
                DFT_coef[idx_coeff] += (
                    2 * t_data * np.sin(theta_order * spot[idx_M] * theta_scale_for_DFT)
                )
    DFT_coef /= N
    if abs(np.mean(Es) - DFT_coef[-1]) > 1e-10:
        print("Something wrong in DFT")
        exit()
    return DFT_coef


def func_DFT_coeff(DFT_coef: Iterable[float], theta_scale_for_DFT=1.0):
    """Create function from DFT coefficients for energy landscape reconstruction.
    
    Args:
        DFT_coef : DFT coefficients.
        theta_scale_for_DFT : Scaling factor for theta. Defaults to 1.0.
    
    Returns:
        function: Function that evaluates DFT curve at given parameter values.
    """
    def DFTcurve(x):
        y = 0
        for n in range(len(DFT_coef)):
            if n == len(DFT_coef) - 1:
                y += DFT_coef[n]
                break
            theta_order = n // 2 + 1
            if n % 2 == 0:
                y += DFT_coef[n] * np.cos(theta_order * x * theta_scale_for_DFT)
            else:
                y += DFT_coef[n] * np.sin(theta_order * x * theta_scale_for_DFT)
        return y

    return DFTcurve


def DiscreteFT(it: int, Es: Iterable[float], 
               spot: Iterable[float], 
               theta_scale_for_DFT: float, verbose: bool, method: str="scipy", eps: float=1.0e-8):
    """Perform discrete Fourier transform optimization step.
    
    Uses DFT to fit energy measurements and find optimal parameter value
    by minimizing the reconstructed energy landscape.
    
    Args:
        it (int): Current iteration number.
        Es (array): Energy measurements.
        spot (array): Parameter values where energies were measured.
        theta_scale_for_DFT (float): Scaling factor for DFT.
        verbose (bool): Whether to print debug information.
        method (str, optional): Optimization method. Options: "scipy", "naive". 
                              Defaults to "scipy".
        eps (float, optional): Numerical precision parameter. Defaults to 1.0e-8.
    
    Returns:
        tuple: Tuple containing:
            - new_theta (float): Optimal parameter value.
            - y_min (float): Minimum energy value.
    """
    N_duration = (len(Es) - 1) // 2
    DFT_coef = eval_DFT_coeff(Es, spot, theta_scale_for_DFT)
    theta_current = spot[0]
    xplot = np.arange(theta_current, theta_current + N_duration * 2 * np.pi, 1.0e-2)
    DFTcurve = func_DFT_coeff(DFT_coef, theta_scale_for_DFT)(xplot)
    if method == "scipy":
        obj = minimize_scalar(
            func_DFT_coeff(DFT_coef, theta_scale_for_DFT),
            bounds=(0, N_duration * 2 * np.pi),
            method="bounded",
        )
        x_min = obj.x
        y_min = obj.fun
    elif method == "naive":
        idx_min = np.argmin(DFTcurve)
        x_min = xplot[idx_min]
        y_min = DFTcurve[idx_min]

    if verbose and it % 10 == 0:
        draw_DFT_curve(xplot, DFTcurve, spot, Es, x_min, y_min)
    new_theta = x_min % (2 * np.pi * N_duration)
    return new_theta, y_min


def cost_func(
    hamiltonian_op,
    params: Iterable[float],
    Nq: int,
    Nocc: int,
    method_ansatz: str,
    method_measure: str,
    pairinghamiltonian: bool=True,
    proton_number: int=0,
    neutron_number: int=0,
    proton_qubits=[],
    neutron_qubits=[],
):
    """Evaluate cost function for VQE optimization with NFT method.
    
    Computes the expectation value of the Hamiltonian for given parameters
    using the specified ansatz and measurement method.
    
    Args:
        hamiltonian_op: Hamiltonian operator to evaluate.
        params (array): Variational parameters.
        Nq (int): Number of qubits.
        Nocc (int): Number of occupied orbitals.
        method_ansatz (str): Ansatz method to use.
        method_measure (str): Measurement method ("statevector" or "Aer").
        pairinghamiltonian (bool, optional): Whether using pairing Hamiltonian. 
                                           Defaults to True.
        proton_number (int, optional): Number of protons. Defaults to 0.
        neutron_number (int, optional): Number of neutrons. Defaults to 0.
        proton_qubits (list, optional): Proton qubit indices. Defaults to [].
        neutron_qubits (list, optional): Neutron qubit indices. Defaults to [].
    
    Returns:
        float: Energy expectation value.
        
    Raises:
        ValueError: If invalid measurement method is provided.
    """
    if pairinghamiltonian:
        qc = pair_ansatz_qiskit(params, Nq, Nocc, method_ansatz)
    else:
        qc = nucl_ansatz(
            Nq,
            proton_qubits,
            neutron_qubits,
            proton_number,
            neutron_number,
            params,
            method_ansatz,
        )
    if method_measure == "statevector":
        estimator = StatevectorEstimator()
        job = estimator.run(
            [
                (
                    qc,
                    hamiltonian_op,
                )
            ]
        )
        results = job.result()
        E_meas = results[0].data.evs
        return E_meas
    elif method_measure == "Aer":
        estimator = AerEstimator()
        job = estimator.run(
            [
                (
                    qc,
                    hamiltonian_op,
                )
            ]
        )
        results = job.result()
        E_meas = results.values[0]
        return E_meas
    else:
        raise ValueError("method_masure in cost_func should be statevector/Aer for now")


def NFTmethod(
    it,
    Ecurrent,
    hamiltonian_op,
    params,
    Nq,
    Nocc,
    method_ansatz,
    method_measure,
    where_is_G_or_cG1,
    which_Gate,
    verbose,
    pairinghamiltonian=True,
    proton_number=0,
    neutron_number=0,
    proton_qubits=[],
    neutron_qubits=[],
):
    """Perform one step of NFT method optimization
    
    Optimizes a single parameter using the NFT method by measuring energies
    at periodic intervals and using DFT to find the optimal parameter value.
    
    Args:
        it (int): Current iteration number.
        Ecurrent (float): Current energy value.
        hamiltonian_op: Hamiltonian operator.
        params (array): Current parameter values.
        Nq (int): Number of qubits.
        Nocc (int): Number of occupied orbitals.
        method_ansatz (str): Ansatz method.
        method_measure (str): Measurement method.
        where_is_G_or_cG1 (list): List indicating gate types ("G" or "cG1").
        which_Gate (int): Index of gate to optimize.
        verbose (bool): Whether to print debug information.
        pairinghamiltonian (bool, optional): Whether using pairing Hamiltonian.
                                           Defaults to True.
        proton_number (int, optional): Number of protons. Defaults to 0.
        neutron_number (int, optional): Number of neutrons. Defaults to 0.
        proton_qubits (list, optional): Proton qubit indices. Defaults to [].
        neutron_qubits (list, optional): Neutron qubit indices. Defaults to [].
    
    Returns:
        array: Updated parameter values with optimized parameter.
    """
    params_ = params.copy()
    # Making point to measure for Discrete Fourier Transformation
    if where_is_G_or_cG1[which_Gate] == "G":
        spot = [params_[which_Gate] + n * 2 * (2 * np.pi) / 5 for n in range(5)]
        spot = np.array(spot)
        theta_scale_for_DFT = 1 / 2
    elif where_is_G_or_cG1[which_Gate] == "cG1":
        spot = [params_[which_Gate] + n * 4 * (2 * np.pi) / 9 for n in range(9)]
        spot = np.array(spot)
        theta_scale_for_DFT = 1 / 4
    else:
        print(
            "Encountered an unknown gate: ",
            where_is_G_or_cG1[which_Gate],
            "something wrong!",
        )
    if verbose:
        print(
            "which_Gate",
            which_Gate,
            "type",
            where_is_G_or_cG1[which_Gate],
            "Current",
            spot[0],
        )

    # Measure the energy at the spot
    tmp = params_.copy()
    Es = np.zeros(len(spot))
    Es[0] = Ecurrent
    for idx in range(1, len(spot)):  # 0 is already measured
        tmp[which_Gate] = spot[idx]
        E = cost_func(
            hamiltonian_op,
            tmp,
            Nq,
            Nocc,
            method_ansatz,
            method_measure,
            pairinghamiltonian=pairinghamiltonian,
            proton_number=proton_number,
            neutron_number=neutron_number,
            proton_qubits=proton_qubits,
            neutron_qubits=neutron_qubits,
        )
        Es[idx] = E
    param_best, ymin_estimated = DiscreteFT(it, Es, spot, theta_scale_for_DFT, verbose)

    if verbose:
        print("Emeasured", Es)
        print("param_best", param_best)

    # Find a minimum point
    params_[which_Gate] = param_best

    return params_


def optimize_params_with_NFT(
    it_max: int,
    hamiltonian_op,
    params: Iterable[float],
    Nq: int,
    Nocc: int,
    ngate: int,
    where_is_G_or_cG1: dict,
    method_ansatz: str,
    method_measure: str,
    verbose: bool=False,
    pairinghamiltonian: bool=True,
    proton_number: int=0,
    neutron_number: int=0,
    proton_qubits=[],
    neutron_qubits=[],
):
    if (proton_number > 0 and proton_qubits == []) or (
        neutron_number > 0 and neutron_qubits == []
    ):
        raise ValueError(
            "proton/neutron_number > 0 requires proton/neutron_qubits to be set"
        )
    stag = 0
    params_opt = None
    E_best = E_current = cost_func(
        hamiltonian_op,
        params,
        Nq,
        Nocc,
        method_ansatz,
        method_measure,
        pairinghamiltonian=pairinghamiltonian,
        proton_number=proton_number,
        neutron_number=neutron_number,
        proton_qubits=proton_qubits,
        neutron_qubits=neutron_qubits,
    )
    print("Optimizing parameters...")
    for it in range(it_max):
        print("iteration %5d " % it, end="")
        if it > 0:
            verbose = False
        for which_Gate in np.random.choice(ngate, ngate, replace=False):
            E_prev = E_current
            params_ = NFTmethod(
                it,
                E_current,
                hamiltonian_op,
                params,
                Nq,
                Nocc,
                method_ansatz,
                method_measure,
                where_is_G_or_cG1,
                which_Gate,
                verbose,
                pairinghamiltonian=pairinghamiltonian,
                proton_number=proton_number,
                neutron_number=neutron_number,
                proton_qubits=proton_qubits,
                neutron_qubits=neutron_qubits,
            )
            E_current = cost_func(
                hamiltonian_op,
                params_,
                Nq,
                Nocc,
                method_ansatz,
                method_measure,
                pairinghamiltonian=pairinghamiltonian,
                proton_number=proton_number,
                neutron_number=neutron_number,
                proton_qubits=proton_qubits,
                neutron_qubits=neutron_qubits,
            )
            if verbose:
                print(
                    which_Gate,
                    params[which_Gate],
                    "=>",
                    params_[which_Gate],
                    "E:",
                    E_prev,
                    "=>",
                    E_current,
                )
            params = params_.copy()
            if E_current < E_best:
                E_best = E_current
                params_opt = params_.copy()
                stag = 0
            else:
                stag += 1
        print("E_current %15.10f" % E_current)
        if stag > 10 * ngate:
            break
    return params_opt, E_best
