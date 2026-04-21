import numpy as np
from numba import njit
from qiskit.quantum_info import PauliList


# ============================================================
# Packed-bit helpers
# ============================================================

def _pack_rows_bool_to_u64(a_bool: np.ndarray) -> np.ndarray:
    """
    Pack boolean matrix (n, q) into uint64 words (n, w), little-endian bit order.
    """
    n, q = a_bool.shape
    w = (q + 63) // 64
    out = np.zeros((n, w), dtype=np.uint64)

    for bit in range(q):
        word = bit >> 6          # bit // 64
        off = bit & 63           # bit % 64
        # set bit 'off' in out[:, word] where a_bool[:, bit] is True
        out[:, word] |= (a_bool[:, bit].astype(np.uint64) << np.uint64(off))
    return out


@njit(cache=True, fastmath=True, inline="always")
def _popcount_u64(x):
    """
    Branchless popcount for uint64 (Hacker's Delight style).
    Numba-friendly fallback without Python int.bit_count().
    """
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


@njit(cache=True, fastmath=True, inline="always")
def _commutes_pair_packed(xp, zp, i, j):
    """
    Full commutativity check using packed symplectic product parity:
      parity( popcount( x_i&z_j ) + popcount( z_i&x_j ) ) == 0
    """
    words = xp.shape[1]
    parity = 0
    for k in range(words):
        a = xp[i, k] & zp[j, k]
        b = zp[i, k] & xp[j, k]
        # parity only: popcount % 2
        parity ^= int(_popcount_u64(a) & np.uint64(1))
        parity ^= int(_popcount_u64(b) & np.uint64(1))
    return parity == 0

# ============================================================
# Chunk grouping (local) with packed bits
# ============================================================

@njit(cache=True, fastmath=True)
def _build_chunk_groups_packed(xc, zc):
    """
    Greedy grouping inside chunk.
    Returns CSR-like local groups:
      flat_local, off_local
    """
    m = xc.shape[0]
    used = np.zeros(m, dtype=np.uint8)

    flat = np.empty(m, dtype=np.int64)       # worst case m total elements
    off = np.empty(m + 1, dtype=np.int64)    # worst case m groups + 1

    f = 0
    g = 0
    off[0] = 0

    for seed in range(m):
        if used[seed]:
            continue

        flat[f] = seed
        f += 1
        used[seed] = 1

        for c in range(seed + 1, m):
            if used[c]:
                continue

            ok = True
            for p in range(off[g], f):
                gi = flat[p]
                if not _commutes_pair_packed(xc, zc, gi, c):
                    ok = False
                    break
            if ok:
                flat[f] = c
                f += 1
                used[c] = 1

        g += 1
        off[g] = f

    return flat[:f], off[:g + 1]


# ============================================================
# Global merge with packed bits
# ============================================================

@njit(cache=True, fastmath=True)
def _find_first_compatible_group(xp, zp, idx, gflat, goff, ng):
    for g in range(ng):
        s = goff[g]
        e = goff[g + 1]
        ok = True
        for p in range(s, e):
            j = gflat[p]
            if not _commutes_pair_packed(xp, zp, idx, j):
                ok = False
                break
        if ok:
            return g
    return -1


def _groups_can_merge_all_commuting(paulis: PauliList, g1, g2) -> bool:
    """True iff every element of g1 commutes with every element of g2."""
    for i in g1:
        pi = paulis[i]
        for j in g2:
            if not pi.commutes(paulis[j]):
                return False
    return True


def _merge_fully_compatible_groups(paulis: PauliList, groups_idx):
    """
    Repeatedly merge any pair of groups that are fully cross-commuting.
    Result: no two remaining groups are fully mergeable.
    """
    groups = [list(g) for g in groups_idx if len(g) > 0]
    changed = True
    while changed:
        changed = False
        m = len(groups)
        i = 0
        while i < m:
            j = i + 1
            while j < m:
                if _groups_can_merge_all_commuting(paulis, groups[i], groups[j]):
                    groups[i].extend(groups[j])
                    del groups[j]
                    m -= 1
                    changed = True
                else:
                    j += 1
            i += 1
    return groups

# ---------- user-facing function  ----------
# returns commuting groups by first breaking the pauli list in chunks
def commuting_groups_chunked(
    paulis: PauliList,
    chunk_size: int = 1000,
    return_type: str = "indices",     # "indices" | "pauli_lists" | "both"
    enforce_nonmergeable_groups: bool = False,
):
    """
    Packed-bit + Numba grouping with optional post-merge step so that no two
    output groups are fully cross-commuting (i.e., cannot be merged).
    """
    n = len(paulis)
    if n == 0:
        if return_type == "both":
            return [], []
        return []

    if return_type not in ("indices", "pauli_lists", "both"):
        raise ValueError("return_type must be one of: 'indices', 'pauli_lists', 'both'")

    x_bool = paulis.x.astype(bool, copy=False)
    z_bool = paulis.z.astype(bool, copy=False)

    xp = _pack_rows_bool_to_u64(x_bool)
    zp = _pack_rows_bool_to_u64(z_bool)

    gflat = np.empty(max(16, n), dtype=np.int64)
    goff = np.empty(max(16, n + 1), dtype=np.int64)

    ng = 0
    flen = 0
    goff[0] = 0

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        xc = xp[start:end]
        zc = zp[start:end]

        lflat, loff = _build_chunk_groups_packed(xc, zc)

        for lg in range(loff.shape[0] - 1):
            ls, le = loff[lg], loff[lg + 1]
            for p in range(ls, le):
                idx = start + int(lflat[p])
                g = _find_first_compatible_group(xp, zp, idx, gflat, goff, ng)

                if g >= 0:
                    ins = goff[g + 1]

                    if flen + 1 > gflat.shape[0]:
                        new = np.empty(gflat.shape[0] * 2, dtype=np.int64)
                        new[:flen] = gflat[:flen]
                        gflat = new

                    if ins < flen:
                        gflat[ins + 1:flen + 1] = gflat[ins:flen]
                    gflat[ins] = idx
                    flen += 1
                    goff[g + 1:ng + 1] += 1
                else:
                    if ng + 2 > goff.shape[0]:
                        newo = np.empty(goff.shape[0] * 2, dtype=np.int64)
                        newo[:ng + 1] = goff[:ng + 1]
                        goff = newo

                    if flen + 1 > gflat.shape[0]:
                        new = np.empty(gflat.shape[0] * 2, dtype=np.int64)
                        new[:flen] = gflat[:flen]
                        gflat = new

                    gflat[flen] = idx
                    flen += 1
                    ng += 1
                    goff[ng] = flen

    groups_idx = [gflat[goff[g]:goff[g + 1]].tolist() for g in range(ng)]

    if enforce_nonmergeable_groups:
        groups_idx = _merge_fully_compatible_groups(paulis, groups_idx)

    if return_type == "indices":
        return groups_idx

    groups_pl = [paulis[np.asarray(grp, dtype=np.int64)] for grp in groups_idx]

    if return_type == "pauli_lists":
        return groups_pl
    return groups_idx, groups_pl

# ---------- user-facing function  ----------
# convert grous of indices into groups of paulis
def groups_indices_to_pauli_lists(paulis: PauliList, groups) -> list[PauliList]:
    """
    Convert groups of indices into list[PauliList].

    Parameters
    ----------
    paulis : PauliList
        Source PauliList.
    groups : iterable[iterable[int]]
        Each inner iterable contains indices into `paulis`.

    Returns
    -------
    list[PauliList]
        One PauliList per group.
    """
    return [paulis[np.asarray(list(g), dtype=np.int64)] for g in groups]


# ---------- numba kernel ----------

@njit(cache=True, fastmath=True)
def _noncomm_count_and_coeffsum_numba(xp, zp, flat, off, coeffs):
    """
    Returns:
      count_noncomm: int64
      coeff_prod_sum: float64  (sum of coeffs[i]*coeffs[j] over non-commuting cross-group pairs)
    """
    ng = off.shape[0] - 1
    count_noncomm = 0
    coeff_prod_sum = 0.0

    for g1 in range(ng):
        s1, e1 = off[g1], off[g1 + 1]
        for g2 in range(g1 + 1, ng):
            s2, e2 = off[g2], off[g2 + 1]
            for p1 in range(s1, e1):
                i = flat[p1]
                ci = coeffs[i]
                for p2 in range(s2, e2):
                    j = flat[p2]
                    if not _commutes_pair_packed(xp, zp, i, j):
                        count_noncomm += 1
                        coeff_prod_sum += ci * coeffs[j]

    return count_noncomm, coeff_prod_sum


# ---------- user-facing function ----------

def noncomm_cross_group_count_and_coeff_product_sum(
    paulis: PauliList,
    groups,
    coeffs: list[float],
    xp: np.ndarray = None,
    zp: np.ndarray = None,
):
    """
    Compute over all cross-group pairs (i, j), i and j in different groups:
      1) number of non-commuting pairs
      2) sum of coeffs[i] * coeffs[j] over those non-commuting pairs

    Parameters
    ----------
    paulis : PauliList
    groups : iterable[iterable[int]]
        Groups of indices into `paulis`.
    coeffs : list[float] | np.ndarray
        Length must be len(paulis).
    xp, zp : optional packed arrays (n, words), dtype=uint64
        Provide for repeated calls to avoid repacking.

    Returns
    -------
    (int, float)
        (non_commuting_pair_count, sum_of_coeff_products)
    """
    n = len(paulis)
    coeffs_arr = np.asarray(coeffs, dtype=np.float64)
    if coeffs_arr.shape[0] != n:
        raise ValueError(f"coeffs length ({coeffs_arr.shape[0]}) must match len(paulis) ({n}).")

    # pack once unless provided
    if xp is None or zp is None:
        x_bool = paulis.x.astype(bool, copy=False)
        z_bool = paulis.z.astype(bool, copy=False)
        xp = _pack_rows_bool_to_u64(x_bool)
        zp = _pack_rows_bool_to_u64(z_bool)

    # groups -> CSR
    groups = list(groups)
    ng = len(groups)
    if ng <= 1:
        return 0, 0.0

    total = sum(len(g) for g in groups)
    flat = np.empty(total, dtype=np.int64)
    off = np.empty(ng + 1, dtype=np.int64)

    k = 0
    off[0] = 0
    for gi, g in enumerate(groups):
        arr = np.asarray(list(g), dtype=np.int64)
        m = arr.shape[0]
        if m:
            if np.any(arr < 0) or np.any(arr >= n):
                raise ValueError(f"group {gi} contains index out of range [0, {n-1}]")
            flat[k:k + m] = arr
            k += m
        off[gi + 1] = k

    count, coeff_sum = _noncomm_count_and_coeffsum_numba(xp, zp, flat, off, coeffs_arr)
    return int(count), float(coeff_sum)
    

# ---------- core numba kernel ----------

@njit(cache=True, fastmath=True)
def _sum_noncommuting_cross_groups_numba(xp, zp, flat, off):
    """
    flat/off is CSR encoding of groups over Pauli indices.
    Returns total number of non-commuting pairs across different groups.
    """
    ng = off.shape[0] - 1
    total = 0

    for g1 in range(ng):
        s1, e1 = off[g1], off[g1 + 1]
        for g2 in range(g1 + 1, ng):
            s2, e2 = off[g2], off[g2 + 1]
            for p1 in range(s1, e1):
                i = flat[p1]
                for p2 in range(s2, e2):
                    j = flat[p2]
                    if not _commutes_pair_packed(xp, zp, i, j):
                        total += 1
    return total


# ---------- user-facing function ----------
# for computing all non-commuting pairs
def sum_noncommuting_cross_group_pairs(
    paulis: PauliList,
    groups,
    xp: np.ndarray = None,
    zp: np.ndarray = None,
) -> int:
    """
    Sum all non-commuting cross-group pairs over all group pairs.

    Parameters
    ----------
    paulis : PauliList
    groups : list[list[int]] (or any iterable of iterables of indices)
    xp, zp : optional prepacked uint64 arrays, shape (n, words)
             pass these for repeated calls to avoid repacking.

    Returns
    -------
    int
        Total number of non-commuting pairs (i, j) with i and j in different groups.
    """
    # Pack once unless provided
    if xp is None or zp is None:
        x_bool = paulis.x.astype(bool, copy=False)
        z_bool = paulis.z.astype(bool, copy=False)
        xp = _pack_rows_bool_to_u64(x_bool)
        zp = _pack_rows_bool_to_u64(z_bool)

    # groups -> CSR (flat, off)
    ng = len(groups)
    if ng <= 1:
        return 0

    total_len = sum(len(g) for g in groups)
    flat = np.empty(total_len, dtype=np.int64)
    off = np.empty(ng + 1, dtype=np.int64)

    k = 0
    off[0] = 0
    for gi, g in enumerate(groups):
        arr = np.asarray(list(g), dtype=np.int64)
        m = arr.shape[0]
        if m > 0:
            flat[k:k + m] = arr
            k += m
        off[gi + 1] = k

    return int(_sum_noncommuting_cross_groups_numba(xp, zp, flat, off))
    
# ---------- user-facing function  ----------
#----------------- compare with qiskit ---------------------- 

def test_compare_with_qiskit_group_commuting(
    paulis: PauliList,
    chunk_size: int = 1000,
    enforce_nonmergeable_groups: bool = False,
):
    """
    Compare:
      - qiskit: paulis.group_commuting(qubit_wise=False)
      - custom: commuting_groups_chunked_numba_packed(...)

    This test checks VALIDITY (not exact partition equality), because multiple
    different valid groupings can exist.

    Checked:
      1) Both outputs are valid partitions of all input elements.
      2) Each group in both outputs is internally pairwise commuting.
      3) If enforce_nonmergeable_groups=True on custom output, verify no two
         custom groups are fully cross-commuting (cannot be merged).

    Returns
    -------
    dict summary.
    Raises AssertionError on failure.
    """
    import numpy as np

    n = len(paulis)

    # ---------- run both ----------
    custom_idx_groups = commuting_groups_chunked_numba_packed(
        paulis=paulis,
        chunk_size=chunk_size,
        return_type="indices",
        enforce_nonmergeable_groups=enforce_nonmergeable_groups,
    )
    qiskit_pl_groups = paulis.group_commuting(qubit_wise=False)  # list[PauliList]

    # ---------- map qiskit groups back to source indices (handles duplicates) ----------
    src_labels = paulis.to_labels()
    label_to_idxs = {}
    for i, lab in enumerate(src_labels):
        label_to_idxs.setdefault(lab, []).append(i)

    ptr = {lab: 0 for lab in label_to_idxs}
    qiskit_idx_groups = []
    for g in qiskit_pl_groups:
        grp = []
        for lab in g.to_labels():
            if lab not in label_to_idxs:
                raise AssertionError(f"Qiskit produced unknown label '{lab}'.")
            k = ptr[lab]
            if k >= len(label_to_idxs[lab]):
                raise AssertionError(f"Too many occurrences of label '{lab}' while mapping back.")
            grp.append(label_to_idxs[lab][k])
            ptr[lab] = k + 1
        qiskit_idx_groups.append(grp)

    # ---------- helpers ----------
    def assert_partition(groups, who):
        flat = [i for grp in groups for i in grp]
        assert len(flat) == n, f"{who}: assigned {len(flat)} elements, expected {n}"
        assert len(set(flat)) == n, f"{who}: duplicates or missing indices"
        assert set(flat) == set(range(n)), f"{who}: partition does not cover all indices"

    def assert_internal_commuting(groups, who):
        for grp in groups:
            for a in range(len(grp)):
                ia = grp[a]
                for b in range(a + 1, len(grp)):
                    ib = grp[b]
                    if not paulis[ia].commutes(paulis[ib]):
                        raise AssertionError(f"{who}: non-commuting pair in same group ({ia}, {ib})")

    def groups_fully_cross_commute(g1, g2):
        for i in g1:
            pi = paulis[i]
            for j in g2:
                if not pi.commutes(paulis[j]):
                    return False
        return True

    # ---------- required checks ----------
    assert_partition(custom_idx_groups, "Custom")
    assert_partition(qiskit_idx_groups, "Qiskit")

    assert_internal_commuting(custom_idx_groups, "Custom")
    assert_internal_commuting(qiskit_idx_groups, "Qiskit")

    # ---------- optional property check for custom ----------
    if enforce_nonmergeable_groups:
        for i in range(len(custom_idx_groups)):
            for j in range(i + 1, len(custom_idx_groups)):
                if groups_fully_cross_commute(custom_idx_groups[i], custom_idx_groups[j]):
                    raise AssertionError(
                        f"Custom: groups {i} and {j} are fully cross-commuting, "
                        "so they are still mergeable."
                    )

    return {
        "n_paulis": n,
        "custom_num_groups": len(custom_idx_groups),
        "qiskit_num_groups": len(qiskit_idx_groups),
        "custom_group_sizes": sorted(len(g) for g in custom_idx_groups),
        "qiskit_group_sizes": sorted(len(g) for g in qiskit_idx_groups),
        "enforce_nonmergeable_groups": enforce_nonmergeable_groups,
        "ok": True,
    }