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
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(xplot, DFTcurve, label="DFT curve", color=cols[0])
    ax.scatter(spot, Es, marker="o", label="Measured", color=cols[1])
    ax.scatter(x_min, y_min, marker="*", color=cols[3], label="Candidate")
    ax.legend()
    plt.show()
    plt.close()


def eval_DFT_coeff(Es, spot, theta_scale_for_DFT):
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


def func_DFT_coeff(DFT_coef, theta_scale_for_DFT=1.0):
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


# def DiscreteFT(it, Es, spot,theta_scale_for_DFT, verbose, method="naive", eps=1.e-8):
def DiscreteFT(it, Es, spot, theta_scale_for_DFT, verbose, method="scipy", eps=1.0e-8):
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
    params,
    Nq,
    Nocc,
    method_ansatz,
    method_measure,
    pairinghamiltonian=True,
    proton_number=0,
    neutron_number=0,
    proton_qubits=[],
    neutron_qubits=[],
):
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
