import numpy as np

def hat(a):
    return np.sqrt(2 * a + 1)

def permutation_parity(lst):
    # Returns 0 for even, 1 for odd permutation.
    par = 1
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if lst[i] > lst[j]:
                par *= -1
    return 0 if par > 0 else 1

def sort_3_orbits(a_in, b_in, c_in):
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


def get_nkey6_shift(a, b, c, d, e, f, int_shift=3):
    return (
        ((a + int_shift) << 50)
        + ((b + int_shift) << 40)
        + ((c + int_shift) << 30)
        + ((d + int_shift) << 20)
        + ((e + int_shift) << 10)
        + (f + int_shift)
    )


def get_nkey6(a, b, c, d, e, f):
    return (a << 50) + (b << 40) + (c << 30) + (d << 20) + (e << 10) + f


def unhash_key6j(i):
    a = i >> 50
    b = (i >> 40) & 0x3FF
    c = (i >> 30) & 0x3FF
    d = (i >> 20) & 0x3FF
    e = (i >> 10) & 0x3FF
    f = i & 0x3FF
    return a, b, c, d, e, f


class sps_3Blab:
    def __init__(
        self,
        e1max,
        e1max_file,
        e2max_file,
        e3max,
        e3max_file,
        norbits_ms,
        norbits_file,
        sps,
        sps_file,
    ):
        self.e1max = e1max
        self.e1max_file = e1max_file
        self.e2max_file = e2max_file
        self.e3max = e3max
        self.e3max_file = e3max_file
        self.norbits_ms = norbits_ms
        self.norbits_file = norbits_file
        self.sps = sps
        self.sps_file = sps_file


def valid_check(ea, eb, ec, ed, ee, ef, e1max, e2max, e3max):
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
    def __init__(
        self,
        single_particle_states,
        JT_orbitals,
        filename,
        e1max,
        e1max_file,
        e2max,
        e3max,
        e3max_file,
        return_pnME_3NF=True,
    ):
        self.filename = filename
        self.single_particle_states = single_particle_states
        self.JT_orbitals = JT_orbitals
        self.e1max = e1max
        self.e2max = e2max
        self.e3max = e3max
        self.e1max_file = e1max_file
        self.e2max_file = 2 * self.e1max_file
        self.e3max_file = e3max_file
        self.dWs = prep_dicts_for_WignerSymbols(e1max)
        self.sps_3b = self.get_modelspace(e1max, e1max_file, e2max, e3max, e3max_file)
        self.dict_idxThBME = self.count_nreads(self.sps_3b)
        self.nread_v3bme = self.count_nreads(self.sps_3b, "ModelSpace")
        self.count_ME_file = self.count_me3jgz(self.sps_3b)
        self.ThBME = self.read_me3jgz(self.filename, self.count_ME_file)
        self.v3bme, self.dict_3b_idx = self.allocate_3bme(self.sps_3b)
        if (
            self.e1max_file == self.e1max
            and self.e2max_file == 2 * self.e1max
            and self.e3max_file == self.e3max
        ):
            self.v3bme = self.ThBME
        else:
            self.v3bme = self.truncate_v3bme(
                len(self.v3bme),
                self.sps_3b,
                self.ThBME,
                self.nread_v3bme,
                self.dWs,
                self.dict_idxThBME,
            )
        if return_pnME_3NF:
            self.pnME_3NF = {}
            self.monopole_V3(
                self.e3max,
                self.sps_3b,
                self.dict_3b_idx,
                self.v3bme,
                self.dWs,
                self.pnME_3NF,
            )
        else:
            self.V3mono = self.monopole_V3(
                self.e3max, self.sps_3b, self.dict_3b_idx, self.v3bme, self.dWs
            )

    def get_modelspace(self, e1max, e1max_file, e2max, e3max, e3max_file):
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

    def count_nreads(self, sps_3b, mode="File"):
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
            # Convert to 0-index for Python list assignment.
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
                    JabMax = (oa.j2 + ob.j2) // 2
                    JabMin = abs(oa.j2 - ob.j2) // 2
                    if abs(oa.j2 - ob.j2) > oc.j2:
                        twoJCMindownbra = abs(oa.j2 - ob.j2) - oc.j2
                    elif oc.j2 < (oa.j2 + ob.j2):
                        twoJCMindownbra = 1
                    else:
                        twoJCMindownbra = oc.j2 - oa.j2 - ob.j2
                    twoJCMaxupbra = oa.j2 + ob.j2 + oc.j2

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

                                JdeMax = (od.j2 + oe.j2) // 2
                                JdeMin = abs(od.j2 - oe.j2) // 2
                                if abs(od.j2 - oe.j2) > of.j2:
                                    twoJCMindownket = abs(od.j2 - oe.j2) - of.j2
                                elif of.j2 < (od.j2 + oe.j2):
                                    twoJCMindownket = 1
                                else:
                                    twoJCMindownket = of.j2 - od.j2 - oe.j2
                                twoJCMaxupket = od.j2 + oe.j2 + of.j2

                                twoJCMindown = max(twoJCMindownbra, twoJCMindownket)
                                twoJCMaxup = min(twoJCMaxupbra, twoJCMaxupket)
                                if twoJCMindown > twoJCMaxup:
                                    continue

                                if mode == "File":
                                    nkey = (idx_a, idx_b, idx_c, idx_d, idx_e, idx_f)
                                    dict_idx_ThBME[nkey] = nread

                                for Jab in range(JabMin, JabMax + 1):
                                    for Jde in range(JdeMin, JdeMax + 1):
                                        twoJCMin = max(
                                            abs(2 * Jab - oc.j2), abs(2 * Jde - of.j2)
                                        )
                                        twoJCMax = min(2 * Jab + oc.j2, 2 * Jde + of.j2)
                                        if twoJCMin > twoJCMax:
                                            continue
                                        blocksize = ((twoJCMax - twoJCMin) // 2 + 1) * 5

                                        nread += blocksize

        if mode == "File":
            nkeys = len(dict_idx_ThBME)
            print("size of dict_idx_ThBME:", nkeys, nkeys * 2 * 8 / 1024**3, "GB")
            return dict_idx_ThBME
        else:
            return nreads

    def count_me3jgz(self, sps_3b, mode="File"):
        e1max = sps_3b.e1max
        e1max_file = sps_3b.e1max_file
        e2max_file = sps_3b.e2max_file
        e3max_file = sps_3b.e3max_file
        e3max = sps_3b.e3max

        e1max_check = e1max_file if mode == "File" else e1max
        e3max_check = e3max_file if mode == "File" else e3max
        sps = sps_3b.sps_file if mode == "File" else sps_3b.sps
        norbits = sps_3b.norbits_file if mode == "File" else sps_3b.norbits

        count_ME_file = 0

        for idx_a in range(1, norbits + 1, 2):
            oa = sps[idx_a]
            ea = oa.e
            if ea > e1max_check:
                continue

            for idx_b in range(1, idx_a + 1, 2):
                ob = sps[idx_b]
                eb = ob.e
                if ea + eb > e2max_file:
                    continue

                for idx_c in range(1, idx_b + 1, 2):
                    oc = sps[idx_c]
                    ec = oc.e
                    if ea + eb + ec > e3max_check:
                        continue

                    JabMax = (oa.j2 + ob.j2) // 2
                    JabMin = abs(oa.j2 - ob.j2) // 2
                    if abs(oa.j2 - ob.j2) > oc.j2:
                        twoJCMindownbra = abs(oa.j2 - ob.j2) - oc.j2
                    elif oc.j2 < (oa.j2 + ob.j2):
                        twoJCMindownbra = 1
                    else:
                        twoJCMindownbra = oc.j2 - oa.j2 - ob.j2
                    twoJCMaxupbra = oa.j2 + ob.j2 + oc.j2

                    for idx_d in range(1, idx_a + 1, 2):
                        od = sps[idx_d]
                        ed = od.e
                        if ed > e1max_check:
                            continue

                        if idx_a == idx_d:
                            upper_idx_e = idx_b
                        else:
                            upper_idx_e = idx_d

                        for idx_e in range(1, upper_idx_e + 1, 2):
                            oe = sps[idx_e]
                            ee = oe.e
                            if ee > e1max_check:
                                continue

                            idx_f_max = (
                                idx_c if (idx_a == idx_d and idx_b == idx_e) else idx_e
                            )

                            for idx_f in range(1, idx_f_max + 1, 2):
                                of = sps[idx_f]
                                ef = of.e
                                if ef > e1max_check:
                                    continue
                                if ed + ee + ef > e3max_check:
                                    continue
                                if (oa.l + ob.l + oc.l + od.l + oe.l + of.l) % 2 != 0:
                                    continue

                                JdeMax = (od.j2 + oe.j2) // 2
                                JdeMin = abs(od.j2 - oe.j2) // 2
                                if abs(od.j2 - oe.j2) > of.j2:
                                    twoJCMindownket = abs(od.j2 - oe.j2) - of.j2
                                elif of.j2 < (od.j2 + oe.j2):
                                    twoJCMindownket = 1
                                else:
                                    twoJCMindownket = of.j2 - od.j2 - oe.j2
                                twoJCMaxupket = od.j2 + oe.j2 + of.j2

                                twoJCMindown = max(twoJCMindownbra, twoJCMindownket)
                                twoJCMaxup = min(twoJCMaxupbra, twoJCMaxupket)
                                if twoJCMindown > twoJCMaxup:
                                    continue

                                for Jab in range(JabMin, JabMax + 1):
                                    for Jde in range(JdeMin, JdeMax + 1):
                                        twoJCMin = max(
                                            abs(2 * Jab - oc.j2), abs(2 * Jde - of.j2)
                                        )
                                        twoJCMax = min(2 * Jab + oc.j2, 2 * Jde + of.j2)
                                        if twoJCMin > twoJCMax:
                                            continue
                                        blocksize = ((twoJCMax - twoJCMin) // 2 + 1) * 5
                                        count_ME_file += blocksize
        print(f"count_ME (File) {count_ME_file}")
        return count_ME_file

    def read_me3jgz(self, filename, count_ME_file):
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
                if idx_i > count_ME_file:
                    break
                if idx_f > count_ME_file:
                    subsize = count_ME_file - idx_i + 1
                for i in range(subsize):
                    tl = line[16 * i : 16 * (i + 1)]
                    ThBME[idx_i + i] = float(tl)
        print("Total ThBME entries read:", len(ThBME))
        return ThBME

    def allocate_3bme(self, sps_3b, ME_is_double=True):
        norbits = sps_3b.norbits_ms
        sps = sps_3b.sps
        e1max = sps_3b.e1max
        e3max = sps_3b.e3max

        dict_3b_idx = {}
        total_dim = 0
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

                Jab_min = abs(oa.j2 - ob.j2) // 2
                Jab_max = (oa.j2 + ob.j2) // 2

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

                                # Record the current dimension offset using a hash key.
                                orbit_hash = get_nkey6(a, b, c, d, e, f)
                                dict_3b_idx[orbit_hash] = total_dim

                                Jde_min = abs(od.j2 - oe.j2) // 2
                                Jde_max = (od.j2 + oe.j2) // 2

                                for Jab in range(Jab_min, Jab_max + 1):
                                    for Jde in range(Jde_min, Jde_max + 1):
                                        J2_min = max(
                                            abs(2 * Jab - oc.j2), abs(2 * Jde - of.j2)
                                        )
                                        J2_max = min(2 * Jab + oc.j2, 2 * Jde + of.j2)
                                        for J2 in range(J2_min, J2_max + 1, 2):
                                            total_dim += 5

        # Determine the total memory required (in GB)
        size_3bme = total_dim * (8 if ME_is_double else 4) / 1024.0**3
        total_memory = psutil.virtual_memory().total / 1024**3
        if size_3bme >= 0.9 * total_memory:
            raise MemoryError(
                f"size(3BME) {size_3bme} is beyond your environment memory {total_memory} GB"
            )
        print(f"# of 3BME: {total_dim:12d} Mem. {size_3bme:12.5e} GB")

        v3bme = np.zeros(total_dim, dtype=np.float64)
        return v3bme, dict_3b_idx

    def monopole_V3(self, E3max, sps_3b, dict_3b_idx, v3bme, dWS, pnME_3NF=None):
        n_orbits = sps_3b.norbits_ms
        sps = sps_3b.sps
        stored_keys = []
        # Build list of keys.
        for i in range(1, n_orbits + 1):
            oi = sps[i]
            for j in range(i, n_orbits + 1):
                oj = sps[j]
                if oi.l != oj.l or oi.j2 != oj.j2 or oi.tz != oj.tz:
                    continue
                for a in range(1, n_orbits + 1):
                    oa = sps[a]
                    for b in range(1, n_orbits + 1):
                        ob = sps[b]
                        if oa.l != ob.l or oa.j2 != ob.j2 or oa.tz != ob.tz:
                            continue
                        for c in range(1, n_orbits + 1):
                            oc = sps[c]
                            if oa.e + oc.e + oi.e > E3max:
                                continue
                            for d in range(1, n_orbits + 1):
                                od = sps[d]
                                if oc.l != od.l or oc.j2 != od.j2 or oc.tz != od.tz:
                                    continue
                                if ob.e + od.e + oj.e > E3max:
                                    continue
                                # Check parity condition from orbital angular momenta.
                                if (oi.l + oa.l + ob.l + oc.l + od.l + oj.l) % 2 != 0:
                                    continue
                                key = get_nkey6(a, c, i, b, d, j)
                                stored_keys.append(key)

        nkeys = len(stored_keys)
        # print("Number of keys for V3mono:", nkeys)

        # Initialize dictionary for monopole matrix elements.
        Vmon3 = {key: 0.0 for key in stored_keys}

        # Loop over keys (e.g., parallelize here if needed)
        for idx, key in enumerate(stored_keys, start=1):
            a, c, i, b, d, j = unhash_key6j(key)
            assert get_nkey6(a, c, i, b, d, j) == key

            ja = sps[a].j2
            jc = sps[c].j2
            ji = sps[i].j2
            jb = sps[b].j2
            jd = sps[d].j2
            jj = sps[j].j2

            j2min = max(abs(ja - jc), abs(jb - jd)) // 2
            j2max = min(ja + jc, jb + jd) // 2
            v = 0.0

            for j2 in range(j2min, j2max + 1):
                Jmin = max(abs(2 * j2 - ji), abs(2 * j2 - jj))
                Jmax = 2 * j2 + min(ji, jj)
                for J2 in range(Jmin, Jmax + 1, 2):
                    vtmp = self.get_V3_pn(
                        idx,
                        E3max,
                        v3bme,
                        j2,
                        j2,
                        J2,
                        a,
                        c,
                        i,
                        b,
                        d,
                        j,
                        sps_3b,
                        dict_3b_idx,
                        dWS,
                        pnME_3NF,
                    ) * (J2 + 1)
                    v += vtmp
            Vmon3[key] += v / (ji + 1)
        norm = np.sqrt(np.sum([float(v) ** 2 for v in list(Vmon3.values())]))
        print(f"Monopole V3 norm: {norm:.5e}")
        return Vmon3

    def get_V3_pn(
        self,
        indx,
        E3max,
        v3bme,
        Jab,
        Jde,
        J2,
        a,
        b,
        c,
        d,
        e,
        f,
        sps_3b,
        dict_3b_idx,
        dWS,
        pnME_3NF,
    ):
        tza = sps_3b.sps[a].tz
        tzb = sps_3b.sps[b].tz
        tzc = sps_3b.sps[c].tz
        tzd = sps_3b.sps[d].tz
        tze = sps_3b.sps[e].tz
        tzf = sps_3b.sps[f].tz
        dcg_spin = dWS.dcg_spin

        Vpn = 0.0
        Tmin = max(abs(tza + tzb + tzc), abs(tzd + tze + tzf))

        # Loop over 'Tab': from div(abs(tza+tzb), 2) up to 1 (inclusive)
        start_Tab = int(abs(tza + tzb) // 2)
        # Python's range(stop) is exclusive, so use 2 as stop to include 1.
        for Tab in range(start_Tab, 2):
            key1 = get_nkey6_shift(1, tza, 1, tzb, Tab * 2, tza + tzb)
            print(
                f"key1 {key1} tz_a {tza} tz_b {tzb} Tab {Tab} tz_c {tzc} tz_d {tzd} Tde {Tde} T2 {T2}"
            )

            CG1 = dcg_spin[key1]
            # Loop over 'Tde': from div(abs(tzd+tze), 2) up to 1 (inclusive)
            start_Tde = int(abs(tzd + tze) // 2)
            # print(f"key1 {key1}, CG1 {CG1}")
            for Tde in range(start_Tde, 2):
                key2 = get_nkey6_shift(1, tzd, 1, tze, Tde * 2, tzd + tze)
                CG2 = dcg_spin[key2]
                if CG1 * CG2 == 0:
                    continue
                Tmax = min(1 + 2 * Tab, 1 + 2 * Tde)
                # Loop over T2 from Tmin to Tmax inclusive with step 2
                for T2 in range(int(Tmin), int(Tmax) + 1, 2):
                    key3 = get_nkey6_shift(
                        Tab * 2, tza + tzb, 1, tzc, T2, tza + tzb + tzc
                    )
                    CG3 = dcg_spin[key3]
                    key4 = get_nkey6_shift(
                        Tde * 2, tzd + tze, 1, tzf, T2, tzd + tze + tzf
                    )
                    CG4 = dcg_spin[key4]
                    if CG3 * CG4 == 0:
                        continue
                    tbme = self.Get3BME_ISO(
                        indx,
                        E3max,
                        v3bme,
                        dict_3b_idx,
                        sps_3b,
                        Jab,
                        Jde,
                        J2,
                        Tab,
                        Tde,
                        T2,
                        a,
                        b,
                        c,
                        d,
                        e,
                        f,
                        dWS,
                        pnME_3NF,
                    )
                    Vpn += float(CG1 * CG2 * CG3 * CG4) * tbme
        return Vpn

    def Get3BME_ISO(
        self,
        ind,
        E3max,
        v3bme,
        dict_3b_idx,
        sps_3b,
        Jab_in,
        Jde_in,
        J2,
        Tab_in,
        Tde_in,
        T2,
        a_in,
        b_in,
        c_in,
        d_in,
        e_in,
        f_in,
        dWS,
        pnME_3NF,
        verbose=False,
    ):
        sps = sps_3b.sps
        v = 0.0

        a, b, c, idx_abc = sort_3_orbits(a_in, b_in, c_in)
        d, e, f, idx_def = sort_3_orbits(d_in, e_in, f_in)

        # If the second triple is larger then swap the roles.
        if d > a or (d == a and e > b) or (d == a and e == b and f > c):
            a, b, c, d, e, f = d, e, f, a, b, c
            Jab_in, Jde_in = Jde_in, Jab_in
            Tab_in, Tde_in = Tde_in, Tab_in
            idx_abc, idx_def = idx_def, idx_abc

        tkey = get_nkey6(a, b, c, d, e, f)
        idx_3borbit = dict_3b_idx.get(tkey, -1)
        if idx_3borbit == -1:
            return 0.0

        oa = sps[a]
        ob = sps[b]
        oc = sps[c]
        od = sps[d]
        oe = sps[e]
        of_ = sps[f]  # 'of' is a Python keyword alternative

        if (oa.e + ob.e + oc.e) > E3max:
            return 0.0
        if (od.e + oe.e + of_.e) > E3max:
            return 0.0

        ja2, jb2, jc2 = oa.j2, ob.j2, oc.j2
        jd2, je2, jf2 = od.j2, oe.j2, of_.j2

        Jab_min = abs(ja2 - jb2) // 2
        Jab_max = (ja2 + jb2) // 2
        Jde_min = abs(jd2 - je2) // 2
        Jde_max = (jd2 + je2) // 2

        Tab_min = 1 if T2 == 3 else 0
        Tab_max = 1
        Tde_min = 1 if T2 == 3 else 0
        Tde_max = 1

        J_index = 0
        # count_inner is used for bookkeeping (unused later)
        count_inner = 0

        for Jab in range(Jab_min, Jab_max + 1):
            Cj_abc = RecouplingCG(idx_abc, ja2, jb2, jc2, Jab_in, Jab, J2, dWS)
            if (idx_abc // 3) != (idx_def // 3):
                Cj_abc *= -1.0
            for Jde in range(Jde_min, Jde_max + 1):
                Cj_def = RecouplingCG(idx_def, jd2, je2, jf2, Jde_in, Jde, J2, dWS)
                J2_min = max(abs(2 * Jab - jc2), abs(2 * Jde - jf2))
                J2_max = min(2 * Jab + jc2, 2 * Jde + jf2)
                if J2_min > J2_max:
                    continue

                J_index += ((J2 - J2_min) // 2) * 5
                count_inner += 1

                if J2_min <= J2 <= J2_max and abs(Cj_abc * Cj_def) > 1.0e-10:
                    for Tab in range(Tab_min, Tab_max + 1):
                        Ct_abc = RecouplingCG(idx_abc, 1, 1, 1, Tab_in, Tab, T2, dWS)
                        for Tde in range(Tde_min, Tde_max + 1):
                            Ct_def = RecouplingCG(
                                idx_def, 1, 1, 1, Tde_in, Tde, T2, dWS
                            )
                            if abs(Ct_abc * Ct_def) < 1.0e-10:
                                continue
                            Tindex = 2 * Tab + Tde + ((T2 - 1) // 2)
                            idx = idx_3borbit + J_index + Tindex
                            v += Cj_abc * Cj_def * Ct_abc * Ct_def * v3bme[idx]
                            vtmp = v3bme[idx]

                            # If you write out vtmp here, you can get something similar to readable.text fmt.
                            # Note that here we are generating bra-ket permutations, which are absent in NuHamil outputs.
                            if pnME_3NF is not None:
                                # pnkey = (a//2+1, b//2+1, c//2+1, d//2+1, e//2+1, f//2+1, Jab, Jde, J2)

                                orb_a = self.JTorbitals.orbitals[a]
                                orb_b = self.JTorbitals.orbitals[b]
                                orb_c = self.JTorbitals.orbitals[c]
                                orb_d = self.JTorbitals.orbitals[d]
                                orb_e = self.JTorbitals.orbitals[e]
                                orb_f = self.JTorbitals.orbitals[f]

                                for (
                                    tz_a,
                                    tz_b,
                                    tz_c,
                                    tz_d,
                                    tz_e,
                                    tz_f,
                                ) in itertools.product([-1, 1], repeat=6):
                                    if (tz_a + tz_b + tz_c) != (tz_d + tz_e + tz_f):
                                        continue
                                    if (tz_a + tz_b) > 2 * Tab:
                                        continue
                                    if (tz_d + tz_e) > 2 * Tde:
                                        continue
                                    pn_a = get_spsidx_from_nljtz(
                                        self.single_particle_states,
                                        orb_a.n,
                                        orb_a.l,
                                        orb_a.j,
                                        tz_a,
                                    )
                                    pn_b = get_spsidx_from_nljtz(
                                        self.single_particle_states,
                                        orb_b.n,
                                        orb_b.l,
                                        orb_b.j,
                                        tz_b,
                                    )
                                    pn_c = get_spsidx_from_nljtz(
                                        self.single_particle_states,
                                        orb_c.n,
                                        orb_c.l,
                                        orb_c.j,
                                        tz_c,
                                    )
                                    pn_d = get_spsidx_from_nljtz(
                                        self.single_particle_states,
                                        orb_d.n,
                                        orb_d.l,
                                        orb_d.j,
                                        tz_d,
                                    )
                                    pn_e = get_spsidx_from_nljtz(
                                        self.single_particle_states,
                                        orb_e.n,
                                        orb_e.l,
                                        orb_e.j,
                                        tz_e,
                                    )
                                    pn_f = get_spsidx_from_nljtz(
                                        self.single_particle_states,
                                        orb_f.n,
                                        orb_f.l,
                                        orb_f.j,
                                        tz_f,
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
                                    pnkey = (
                                        pn_a,
                                        pn_b,
                                        pn_c,
                                        pn_d,
                                        pn_e,
                                        pn_f,
                                        Jab,
                                        Jde,
                                        Jabc,
                                    )
                                    cg_1 = self.get_CGs_from_dict(
                                        1, tz_a, 1, tz_b, Tab * 2, (tz_a + tz_b)
                                    )
                                    cg_2 = self.get_CGs_from_dict(
                                        1, tz_d, 1, tz_e, Tde * 2, (tz_d + tz_e)
                                    )
                                    cg_3 = self.get_CGs_from_dict(
                                        Tab * 2,
                                        (tz_a + tz_b),
                                        1,
                                        tz_c,
                                        Tabc,
                                        (tz_a + tz_b + tz_c),
                                    )
                                    cg_4 = self.get_CGs_from_dict(
                                        Tde * 2,
                                        (tz_d + tz_e),
                                        1,
                                        tz_f,
                                        Tabc,
                                        (tz_d + tz_e + tz_f),
                                    )
                                    cgfact = cg_1 * cg_2 * cg_3 * cg_4
                                    if abs(cgfact) < 1.0e-8:
                                        continue
                                    part = ME * cgfact
                                    if pnkey in pnME_3NF.keys():
                                        pnME_3NF[pnkey] += part
                                    else:
                                        pnME_3NF[pnkey] = part

                            #     pn_a = get_spsidx_from_nljtz(self.single_particle_states, orb_a.n, orb_a.l, orb_a.j, orb_a.tz)
                            #     pn_b = get_spsidx_from_nljtz(self.single_particle_states, orb_b.n, orb_b.l, orb_b.j, orb_b.tz)
                            #     pn_c = get_spsidx_from_nljtz(self.single_particle_states, orb_c.n, orb_c.l, orb_c.j, orb_c.tz)
                            #     pn_d = get_spsidx_from_nljtz(self.single_particle_states, orb_d.n, orb_d.l, orb_d.j, orb_d.tz)
                            #     pn_e = get_spsidx_from_nljtz(self.single_particle_states, orb_e.n, orb_e.l, orb_e.j, orb_e.tz)
                            #     pn_f = get_spsidx_from_nljtz(self.single_particle_states, orb_f.n, orb_f.l, orb_f.j, orb_f.tz)

                            #     pnkey = (pn_a, pn_b, pn_c, pn_d, pn_e, pn_f, Jab, Jde, J2)

                            #     vtmp = Cj_abc * Cj_def * Ct_abc * Ct_def * v3bme[idx]
                            #     if pnkey not in pnME_3NF:
                            #         pnME_3NF[pnkey] = vtmp
                            #     else:
                            #         pnME_3NF[pnkey] += vtmp

                            # if abs(vtmp) > 1.e-8 and verbose:
                            #     print(f"{a//2+1:3d} {b//2+1:3d} {c//2+1:3d} {Jab:3d} {Tab:3d} ",
                            #             f"{d//2+1:3d} {e//2+1:3d} {f//2+1:3d} {Jde:3d} {Tde:3d} ",
                            #             f"{J2:3d} {T2:3d} {vtmp:16.9f}")
                J_index += ((J2_max - J2 + 2) // 2) * 5
        return v


def truncate_v3bme(dim_v3bme, sps_3b, ThBME, nreads_v3bme, dWS, dict_idxThBME):
    v3bme = np.zeros(dim_v3bme, dtype=np.float64)
    e1max = sps_3b.e1max
    e2max = e1max * 2
    e1max_file = sps_3b.e1max_file
    e2max_file = sps_3b.e2max_file
    e3max_file = sps_3b.e3max_file
    e3max = sps_3b.e3max
    l3max = e1max
    sps = sps_3b.sps
    norbits = sps_3b.norbits
    count_ME_file = 0

    # Loop over 'bra' indices (odd numbers from 1 to norbits)
    for idx_a in range(1, norbits + 1, 2):
        oa = sps[idx_a]
        ea = oa.e
        # nreads_v3bme index conversion: Julia's div(idx_a,2)+1 -> Python's idx_a//2
        nread_v3bme = nreads_v3bme[idx_a // 2]
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

                JabMax = (oa.j2 + ob.j2) // 2
                JabMin = abs(oa.j2 - ob.j2) // 2
                if abs(oa.j2 - ob.j2) > oc.j2:
                    twoJCMindownbra = abs(oa.j2 - ob.j2) - oc.j2
                elif oc.j2 < (oa.j2 + ob.j2):
                    twoJCMindownbra = 1
                else:
                    twoJCMindownbra = oc.j2 - oa.j2 - ob.j2
                twoJCMaxupbra = oa.j2 + ob.j2 + oc.j2

                # Loop for 'ket' part
                for idx_d in range(1, idx_a + 1, 2):
                    od = sps[idx_d]
                    ed = od.e
                    if ed > e1max:
                        continue
                    upper_idx_e = idx_b if (idx_a == idx_d) else idx_d
                    for idx_e in range(1, upper_idx_e + 1, 2):
                        oe = sps[idx_e]
                        ee = oe.e
                        if ee > e1max:
                            continue
                        idx_f_max = (
                            idx_c if (idx_a == idx_d and idx_b == idx_e) else idx_e
                        )
                        for idx_f in range(1, idx_f_max + 1, 2):
                            of_ = sps[idx_f]
                            ef = of_.e
                            if ef > e1max:
                                continue
                            if ed + ee + ef > e3max:
                                continue
                            if (oa.l + ob.l + oc.l + od.l + oe.l + of_.l) % 2 != 0:
                                continue

                            if not valid_check(
                                ea, eb, ec, ed, ee, ef, e1max, e2max, e3max
                            ):
                                continue
                            if not valid_check(
                                ea,
                                eb,
                                ec,
                                ed,
                                ee,
                                ef,
                                e1max_file,
                                e2max_file,
                                e3max_file,
                            ):
                                continue

                            JdeMax = (od.j2 + oe.j2) // 2
                            JdeMin = abs(od.j2 - oe.j2) // 2
                            if abs(od.j2 - oe.j2) > of_.j2:
                                twoJCMindownket = abs(od.j2 - oe.j2) - of_.j2
                            elif of_.j2 < (od.j2 + oe.j2):
                                twoJCMindownket = 1
                            else:
                                twoJCMindownket = of_.j2 - od.j2 - oe.j2
                            twoJCMaxupket = od.j2 + oe.j2 + of_.j2

                            twoJCMindown = max(twoJCMindownbra, twoJCMindownket)
                            twoJCMaxup = min(twoJCMaxupbra, twoJCMaxupket)
                            if twoJCMindown > twoJCMaxup:
                                continue

                            key = get_nkey6(idx_a, idx_b, idx_c, idx_d, idx_e, idx_f)
                            offset_ThBME = dict_idxThBME[key]
                            idx_ThBME = offset_ThBME

                            for Jab in range(JabMin, JabMax + 1):
                                for Jde in range(JdeMin, JdeMax + 1):
                                    twoJCMin = max(
                                        abs(2 * Jab - oc.j2), abs(2 * Jde - of_.j2)
                                    )
                                    twoJCMax = min(2 * Jab + oc.j2, 2 * Jde + of_.j2)
                                    if twoJCMin > twoJCMax:
                                        continue
                                    blocksize = ((twoJCMax - twoJCMin) // 2 + 1) * 5
                                    for JTind in range(0, twoJCMax - twoJCMin + 1):
                                        twoJC = twoJCMin + (JTind // 2) * 2
                                        twoT = 1 + (JTind % 2) * 2
                                        for Tab in range(0, 2):
                                            for Tde in range(0, 2):
                                                if twoT > min(2 * Tab + 1, 2 * Tde + 1):
                                                    continue
                                                index_ab = (
                                                    ((5 * (twoJC - twoJCMin)) // 2)
                                                    + 2 * Tab
                                                    + Tde
                                                    + ((twoT - 1) // 2)
                                                )
                                                v3idx = nread_v3bme + index_ab + 1
                                                idx_ThBME += 1
                                                ThBME_idx = idx_ThBME
                                                V = 0.0
                                                autozero = False
                                                if (
                                                    oa.l > l3max
                                                    or ob.l > l3max
                                                    or oc.l > l3max
                                                    or od.l > l3max
                                                    or oe.l > l3max
                                                    or of_.l > l3max
                                                ):
                                                    V = 0.0
                                                v3bme[v3idx] = ThBME[ThBME_idx]
                                                if (
                                                    idx_a == idx_b
                                                    and (Tab + Jab) % 2 == 0
                                                ) or (
                                                    idx_d == idx_e
                                                    and (Tde + Jde) % 2 == 0
                                                ):
                                                    autozero = True
                                                if (
                                                    idx_a == idx_b
                                                    and idx_a == idx_c
                                                    and twoT == 3
                                                    and oa.j2 < 3
                                                ):
                                                    autozero = True
                                                if (
                                                    idx_d == idx_e
                                                    and idx_d == idx_f
                                                    and twoT == 3
                                                    and od.j2 < 3
                                                ):
                                                    autozero = True
                                    if valid_check(
                                        ea, eb, ec, ed, ee, ef, e1max, e2max, e3max
                                    ):
                                        nread_v3bme += blocksize
                                    count_ME_file += blocksize


def RecouplingCG(idx_abc, ja2, jb2, jc2, Jab_in, Jab, J2, dWS) -> float:
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


class prep_dicts_for_WignerSymbols:
    def __init__(self, emax):
        self.emax = emax
        self.jmax = 2 * emax + 1
        self.d6j_int = self.prep_d6j_int(emax, self.jmax)
        self.dcg_spin = self.prep_dcg_spin()
        self.d6j_lj = self.prep_d6j_lj(self.jmax)

    def prep_dcg_spin(self):
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
    if a + b < c or a + c < b or b + c < a:
        return False
    if abs(a - b) > c or abs(a - c) > b or abs(b - c) > a:
        return False
    return True


def get_key6j_sym(j1: int, j3: int, j5: int, j2: int, j4: int, j6: int) -> int:
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
    return 100 * (j1 + j2) + min(j1, j2)


def get_canonical_order_6j(j1: int, j2: int, j3: int, j4: int, j5: int, j6: int) -> int:
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
