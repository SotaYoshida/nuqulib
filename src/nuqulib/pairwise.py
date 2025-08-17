import numpy as np
import itertools
from qiskit.quantum_info import SparsePauliOp
import openfermion as of


def read_msnt(
    fn,
    A,
    Acore,
    pow_A=-0.3,
    massdep=0,
    degenerate_SPE=False,
    verbose=False,
    neutron=True,
    pn_system=False,
    only_monopole=False,
):
    if pn_system:
        neutron = False
    Vfactor = 1.0
    if massdep == 1 and A > 16:
        Vfactor *= (A / (Acore + 2)) ** (pow_A)
    inp = open(fn, "r")
    lines = inp.readlines()
    inp.close()
    Hamil = {}
    label = ""
    p_sps = {}
    n_sps = {}
    for i, line in enumerate(lines):
        if "num   n   l   j  tz   mz       SPE(MeV)" in line:
            label = "SPE"
            Hamil[label] = {}
            continue
        if "Vpp:" in line:
            label = "Vpp"
            Hamil[label] = []
            continue
        if "Vnn:" in line:
            label = "Vnn"
            Hamil[label] = []
            continue
        if "Vpn:" in line:
            label = "Vpn"
            Hamil[label] = []
            continue
        if label == "":
            continue
        elif label == "SPE":
            tl = line.rstrip().split()
            num, n, l, j, tz, mz = list(map(int, tl[:-1]))
            SPE = float(tl[-1])
            if degenerate_SPE:
                SPE = -3.0
            if tz == -1:
                p_sps[num] = [n, l, j, mz, SPE]
            else:
                n_sps[num] = [n, l, j, mz, SPE]
            Hamil[label][num] = SPE
        else:
            tl = line.rstrip().split()
            a, b, c, d, totJ = list(map(int, tl[:-1]))
            if only_monopole:
                if a != c or b != d:
                    continue
            V = float(tl[-1]) * Vfactor
            Hamil[label] += [[a, b, c, d, totJ, V]]

    # make a dictionary to convert the sps to qubits and vice versa
    Dict_qubits_to_sps = {}
    Dict_sps_to_qubits = {}
    count_qubits = count_sps = 0
    if neutron:
        target_sps1, target_sps2 = n_sps, n_sps
    else:
        target_sps1, target_sps2 = p_sps, p_sps
    if pn_system:
        target_sps1, target_sps2 = p_sps, n_sps

    for num in target_sps1.keys():
        n, l, j, mz, SPE = target_sps1[num]
        if mz > 0 and neutron:
            continue
        count_sps += 1
        Dict_sps_to_qubits[count_sps] = count_qubits
        Dict_qubits_to_sps[count_qubits] = [num]
        for num in target_sps2.keys():
            n_, l_, j_, mz_, SPE_ = target_sps2[num]
            if mz > 0 and neutron:
                continue
            if n == n_ and l == l_ and j == j_ and mz == -mz_:
                Dict_qubits_to_sps[count_qubits] += [num]
        count_qubits += 1

    if verbose:
        print("p_sps", p_sps)
        print("n_sps", n_sps)
        print("Vpp", Hamil["Vpp"])
        print("Vnn", Hamil["Vnn"])
        print("Vpn", Hamil["Vpn"])
        print("Dict_sps_to_qubits", Dict_sps_to_qubits)
        print("Dict_qubits_to_sps", Dict_qubits_to_sps)

    return (Hamil, p_sps, n_sps, Dict_qubits_to_sps, Dict_sps_to_qubits)


def add_Vmonopole(
    Hamil, config, n_qubits, Nocc, Dict_qubits_to_sps, neutron=True, verbose=False
):
    # Monopole term: V_{ijij} N_i N_j
    target_H2b = Hamil["Vnn"] if neutron else Hamil["Vpp"]
    if Nocc <= 1:
        return 0.0
    Vmono = 0.0
    for idx_i_qubit in range(n_qubits):
        bit_i = config[idx_i_qubit]
        if bit_i == "0":
            continue
        idxs_sps_i = Dict_qubits_to_sps[idx_i_qubit]
        for idx_j_qubit in range(idx_i_qubit + 1, n_qubits):
            bit_j = config[idx_j_qubit]
            idxs_sps_j = Dict_qubits_to_sps[idx_j_qubit]
            if bit_j == "0":
                continue
            if verbose:
                print(
                    "qubit_i",
                    idx_i_qubit,
                    "i_sps",
                    idxs_sps_i,
                    "qubit_j",
                    idx_j_qubit,
                    "j_sps",
                    idxs_sps_j,
                )

            for tmp in target_H2b:
                a, b, c, d, totJ, V = tmp
                if a == c and b == d:
                    if (a in idxs_sps_i and b in idxs_sps_j) or (
                        b in idxs_sps_i and a in idxs_sps_j
                    ):
                        if verbose:
                            print("monohit: ", a, b, c, d, "J=", totJ, "V", V)
                        Vmono += V
    return Vmono


def get_pairwise_Hamil(Hamil, configs, neutron=True):
    target_H1b = Hamil["SPE"]
    target_H2b = Hamil["Vnn"] if neutron else Hamil["Vpp"]
    nconfigs = len(configs)
    h1b = np.zeros(nconfigs, dtype=float)
    h2b = np.zeros((nconfigs, nconfigs), dtype=float)
    for i, config_l in enumerate(configs):
        h1b[i] = target_H1b[config_l[0]] + target_H1b[config_l[1]]
        for j in range(i, nconfigs):
            config_r = configs[j]
            for tmp in target_H2b:
                a, b, c, d, totJ, V = tmp
                if (
                    (a in config_l)
                    and (b in config_l)
                    and (c in config_r)
                    and (d in config_r)
                ):
                    h2b[i, j] += V
                elif (
                    (a in config_r)
                    and (b in config_r)
                    and (c in config_l)
                    and (d in config_l)
                ):
                    print("This must not occur!\n\n\n\n")
                    h2b[i, j] += V
            if i != j:
                h2b[j, i] = h2b[i, j]
    return h1b, h2b


def eval_Hflat_eigen(
    configs,
    n_qubits,
    Nocc,
    Hamil,
    h1b,
    h2b,
    Dict_qubits_to_sps,
    verbose=False,
    Vmonofac=0.25,
):
    dim = len(configs)
    Hflat = np.zeros((dim, dim), dtype=float)
    H2b_flat = np.zeros((dim, dim), dtype=float)
    h2b_mono = np.zeros((dim, dim), dtype=float)
    for idx_i in range(dim):
        config_i = configs[idx_i]
        for i in range(n_qubits):
            if config_i[i] == "1":
                Vmono = (
                    add_Vmonopole(Hamil, config_i, n_qubits, Nocc, Dict_qubits_to_sps)
                    / Nocc
                )
                if verbose:
                    print(idx_i, config_i, "Vmono ", Vmono)
                Hflat[idx_i, idx_i] += h1b[i] + Vmono
                h2b_mono[idx_i, idx_i] += Vmono

        for idx_j in range(idx_i, dim):
            config_j = configs[idx_j]
            # 配位の2進表現で、"差"が1以下(A^+Aで遷移可能)の場合のみ、H2を足す
            diff = n_qubits - sum([config_i[i] == config_j[i] for i in range(n_qubits)])
            if diff > 2:
                continue
            idxs_hit_i = [i for i in range(n_qubits) if config_i[i] == "1"]
            idxs_hit_j = [i for i in range(n_qubits) if config_j[i] == "1"]
            if diff == 0:
                for i in idxs_hit_i:
                    Hflat[idx_i, idx_j] += h2b[i, i]
            if diff == 2:
                for i in idxs_hit_i:
                    if i in idxs_hit_j:
                        continue
                    for j in idxs_hit_j:
                        if j in idxs_hit_i:
                            continue
                        Hflat[idx_i, idx_j] += h2b[i, j]
                        H2b_flat[idx_i, idx_j] += h2b[i, j]
            Hflat[idx_j, idx_i] = Hflat[idx_i, idx_j]
            H2b_flat[idx_j, idx_i] = H2b_flat[idx_i, idx_j]
    evals, evecs = np.linalg.eig(Hflat)
    if verbose:
        print("Hflat???")
        for i in range(dim):
            print(Hflat[i, :])
        print("evals_ Hflat!!", evals)
    return evals, evecs


def get_possible_configs(sps, verbose=False):
    configs = []
    for tmp in sps.keys():
        n, l, j, mz, SPE = sps[tmp]
        for tmp_2 in sps.keys():
            n_2, l_2, j_2, mz_2, SPE_2 = sps[tmp_2]
            if n != n_2 or l != l_2 or j != j_2 or mz != -mz_2:
                continue
            if tmp > tmp_2:
                continue
            if verbose:
                print("pair orbits: ", tmp, tmp_2)
            configs += [[tmp, tmp_2]]
    return configs


def generate_config_bitstr_list(n_qubits, Nocc, rev=False, verbose=False):
    config_list_int = []
    for config in itertools.combinations(range(n_qubits), Nocc):
        config_list_int += [sum([2**i for i in config])]
    config_list = [f"{config:0{n_qubits}b}" for config in config_list_int]
    if rev:
        config_list = [config[::-1] for config in config_list]
    config_list.sort()
    return config_list


def make_pw_hamil_qiskit(Hamil, h1b, Nq, Nocc, Dict_qubits_to_sps):
    target = Hamil["Vnn"]
    Vspe = {i: 0.0 for i in range(Nq)}
    Vmono = {}
    Vpair = {}
    for i in range(Nq):
        a = Dict_qubits_to_sps[i][0]
        abar = Dict_qubits_to_sps[i][1]
        # Vspe
        vtmp_spe = h1b[i]
        for tmp in target:
            i_, j_, k_, l_, J, V = tmp
            if i_ == k_ == a and j_ == l_ == abar:
                vtmp_spe += V
        Vspe[i] = vtmp_spe / 2
        for j in range(i + 1, Nq):
            b = Dict_qubits_to_sps[j][0]
            bbar = Dict_qubits_to_sps[j][1]
            vtmp_M = vtmp_P = 0.0
            for tmp in target:
                i_, j_, k_, l_, J, V = tmp
                # Vpair
                if i_ == a and j_ == abar and k_ == b and l_ == bbar:
                    vtmp_P += V
                # Vmono
                if (
                    (i_ == a and j_ == b and k_ == a and l_ == b)
                    or (i_ == a and j_ == bbar and k_ == a and l_ == bbar)
                    or (i_ == b and j_ == abar and k_ == b and l_ == abar)
                    or (i_ == abar and j_ == b and k_ == abar and l_ == b)
                    or (i_ == abar and j_ == bbar and k_ == abar and l_ == bbar)
                    or (i_ == bbar and j_ == abar and k_ == bbar and l_ == abar)
                ):
                    vtmp_M += V
            Vpair[(i, j)] = 0.5 * vtmp_P
            Vmono[(i, j)] = vtmp_M

    # Qiskit Operator
    ops = []
    coeffs = []

    # single particle-like terms (e_i + Vii) * (I-Z)/2
    op = "I" * Nq
    coeff = sum(Vspe.values())
    ops.append(op)
    coeffs.append(coeff)

    for tkey in Vspe.keys():
        op = ["I" for _ in range(Nq)]
        coeff = Vspe[tkey]
        op[tkey] = "Z"
        op = "".join(op)
        op = op[::-1]
        ops.append(op)
        coeffs.append(-coeff)

    ## pair terms (XX+YY)
    for tkey in Vpair.keys():
        i, j = tkey
        op = ["I" for _ in range(Nq)]
        coeff = Vpair[tkey]
        opXX = op.copy()
        opXX[i] = "X"
        opXX[j] = "X"
        opXX = "".join(opXX)
        opXX = opXX[::-1]
        opYY = op.copy()
        opYY[i] = "Y"
        opYY[j] = "Y"
        opYY = "".join(opYY)
        opYY = opYY[::-1]
        ops.append(opXX)
        coeffs.append(coeff)
        ops.append(opYY)
        coeffs.append(coeff)

    ## monopole terms (I-Zi)(I-Zj)/4
    if Nocc > 1:
        for tkey in Vmono.keys():
            coeff = Vmono[tkey] / 4
            # I term
            op_I = "I" * Nq
            idx = ops.index(op_I)
            coeffs[idx] += coeff

            i, j = tkey
            op = ["I" for _ in range(Nq)]
            op_Zi = op.copy()
            op_Zi[i] = "Z"
            op_Zi = "".join(op_Zi)
            op_Zi = op_Zi[::-1]
            if op_Zi in ops:
                idx = ops.index(op_Zi)
                coeffs[idx] -= coeff
            else:
                ops.append(op_Zi)
                coeffs.append(-coeff)

            op_Zj = op.copy()
            op_Zj[j] = "Z"
            op_Zj = "".join(op_Zj)
            op_Zj = op_Zj[::-1]
            if op_Zj in ops:
                idx = ops.index(op_Zj)
                coeffs[idx] -= coeff
            else:
                ops.append(op_Zj)
                coeffs.append(-coeff)

            op_ZiZj = op.copy()
            op_ZiZj[i] = "Z"
            op_ZiZj[j] = "Z"
            op_ZiZj = "".join(op_ZiZj)
            op_ZiZj = op_ZiZj[::-1]
            ops.append(op_ZiZj)
            coeffs.append(coeff)

    return ops, coeffs, SparsePauliOp.from_list(list(zip(ops, coeffs)))
