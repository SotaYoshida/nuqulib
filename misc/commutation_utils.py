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

@njit(cache=True)
def _pack_rows_bool_to_u64(b):
    """
    Pack boolean matrix b (shape: n_rows, n_bits) into uint64 words per row.
    Bit convention: bit j goes to word (j // 64), bit position (j % 64), LSB-first.
    Returns packed array shape (n_rows, ceil(n_bits/64)).
    """
    n_rows, n_bits = b.shape
    words = (n_bits + 63) // 64
    out = np.zeros((n_rows, words), dtype=np.uint64)

    for i in range(n_rows):
        for j in range(n_bits):
            if b[i, j]:
                w = j // 64
                s = j - 64 * w  # j % 64
                out[i, w] |= (np.uint64(1) << np.uint64(s))
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

@njit(cache=True)
def _dedup_sorted_inplace(arr):
    n = arr.shape[0]
    if n == 0:
        return 0
    w = 1
    prev = arr[0]
    for i in range(1, n):
        v = arr[i]
        if v != prev:
            arr[w] = v
            w += 1
            prev = v
    return w


@njit(cache=True)
def _normalize_groups_to_csr_numba(flat_in, off_in, n):
    ng = off_in.shape[0] - 1
    flat_out = np.empty(flat_in.shape[0], dtype=np.int64)
    off_out = np.empty(ng + 1, dtype=np.int64)

    k = 0
    off_out[0] = 0
    for g in range(ng):
        s = off_in[g]
        e = off_in[g + 1]
        m = e - s
        if m == 0:
            off_out[g + 1] = k
            continue

        tmp = np.empty(m, dtype=np.int64)
        for t in range(m):
            v = flat_in[s + t]
            if v < 0 or v >= n:
                raise ValueError("group index out of range")
            tmp[t] = v

        tmp.sort()
        u = _dedup_sorted_inplace(tmp)

        flat_out[k:k + u] = tmp[:u]
        k += u
        off_out[g + 1] = k

    return flat_out[:k], off_out


@njit(cache=True)
def _count_raw_pairs_numba(flat, off):
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
                    if i != j:
                        total += 1
    return total


@njit(cache=True)
def _build_all_pairs_raw_numba(flat, off, pair_i, pair_j):
    ng = off.shape[0] - 1
    k = 0
    for g1 in range(ng):
        s1, e1 = off[g1], off[g1 + 1]
        for g2 in range(g1 + 1, ng):
            s2, e2 = off[g2], off[g2 + 1]
            for p1 in range(s1, e1):
                i = flat[p1]
                for p2 in range(s2, e2):
                    j = flat[p2]
                    if i == j:
                        continue
                    if i < j:
                        pair_i[k] = i
                        pair_j[k] = j
                    else:
                        pair_i[k] = j
                        pair_j[k] = i
                    k += 1
    return k


# @njit(cache=True)
# def _argsort_pairs_lex_numba(a, b):
#     return np.lexsort((b, a))


# @njit(cache=True)
# def _dedup_pairs_sorted_numba(a_sorted, b_sorted):
#     n = a_sorted.shape[0]
#     if n == 0:
#         return 0
#     w = 1
#     pa = a_sorted[0]
#     pb = b_sorted[0]
#     for i in range(1, n):
#         ca = a_sorted[i]
#         cb = b_sorted[i]
#         if ca != pa or cb != pb:
#             a_sorted[w] = ca
#             b_sorted[w] = cb
#             w += 1
#             pa = ca
#             pb = cb
#     return w


# @njit(cache=True)
# def _build_unique_cross_group_pairs_numba(flat_norm, off_norm):
#     m_raw = _count_raw_pairs_numba(flat_norm, off_norm)
#     if m_raw == 0:
#         return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

#     pair_i = np.empty(m_raw, dtype=np.int64)
#     pair_j = np.empty(m_raw, dtype=np.int64)
#     m_fill = _build_all_pairs_raw_numba(flat_norm, off_norm, pair_i, pair_j)

#     ai = pair_i[:m_fill].copy()
#     bj = pair_j[:m_fill].copy()
#     idx = _argsort_pairs_lex_numba(ai, bj)
#     ai = ai[idx]
#     bj = bj[idx]

#     u = _dedup_pairs_sorted_numba(ai, bj)
#     return ai[:u], bj[:u]

@njit(cache=True)
def _pair_to_pos(i, j, n):
    """
    Map unordered pair (i<j) to unique integer in [0, n*(n-1)/2).
    """
    return (i * (2 * n - i - 1)) // 2 + (j - i - 1)

@njit(cache=True)
def _count_unique_cross_group_pairs_numba(flat_norm, off_norm, n):
    """
    Count unique unordered cross-group pairs using a marker array over upper triangle.
    """
    ng = off_norm.shape[0] - 1
    n_pairs = (n * (n - 1)) // 2
    seen = np.zeros(n_pairs, dtype=np.uint8)

    cnt = 0
    for g1 in range(ng):
        s1, e1 = off_norm[g1], off_norm[g1 + 1]
        for g2 in range(g1 + 1, ng):
            s2, e2 = off_norm[g2], off_norm[g2 + 1]
            for p1 in range(s1, e1):
                a = flat_norm[p1]
                for p2 in range(s2, e2):
                    b = flat_norm[p2]
                    if a == b:
                        continue
                    i = a if a < b else b
                    j = b if a < b else a
                    pos = _pair_to_pos(i, j, n)
                    if seen[pos] == 0:
                        seen[pos] = 1
                        cnt += 1
    return cnt

@njit(cache=True)
def _build_unique_cross_group_pairs_numba(flat_norm, off_norm, n):
    """
    Build unique unordered cross-group pairs (i<j), no lexsort/no Python set.
    """
    ng = off_norm.shape[0] - 1
    n_pairs = (n * (n - 1)) // 2
    seen = np.zeros(n_pairs, dtype=np.uint8)

    m = _count_unique_cross_group_pairs_numba(flat_norm, off_norm, n)
    if m == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    pair_i = np.empty(m, dtype=np.int64)
    pair_j = np.empty(m, dtype=np.int64)

    k = 0
    for g1 in range(ng):
        s1, e1 = off_norm[g1], off_norm[g1 + 1]
        for g2 in range(g1 + 1, ng):
            s2, e2 = off_norm[g2], off_norm[g2 + 1]
            for p1 in range(s1, e1):
                a = flat_norm[p1]
                for p2 in range(s2, e2):
                    b = flat_norm[p2]
                    if a == b:
                        continue
                    i = a if a < b else b
                    j = b if a < b else a
                    pos = _pair_to_pos(i, j, n)
                    if seen[pos] == 0:
                        seen[pos] = 1
                        pair_i[k] = i
                        pair_j[k] = j
                        k += 1

    return pair_i, pair_j



@njit(cache=True, fastmath=True)
def _noncomm_count_and_coeffsum_unique_pairs_numba(xp, zp, pair_i, pair_j, coeffs):
    """
    Evaluate commutation on a precomputed list of unique index pairs and accumulate
    non-commuting statistics.

    -------------------------------------------------------------------------
    Purpose
    -------------------------------------------------------------------------
    This is the low-level numeric kernel used after pair generation/deduplication.
    It assumes `(pair_i[k], pair_j[k])` already represents the exact set of pairs
    you want to evaluate (typically unique unordered pairs with pair_i[k] < pair_j[k]).

    For each pair (i, j):
      - compute whether Pauli i and Pauli j commute (via packed symplectic parity),
      - if they do NOT commute:
          count_noncomm += 1
          coeff_prod_sum += coeffs[i] * coeffs[j]

    -------------------------------------------------------------------------
    Inputs
    -------------------------------------------------------------------------
    xp : np.ndarray, dtype=uint64, shape (n, words)
        Packed X bits for each Pauli term.
        - n: number of Pauli terms
        - words: ceil(num_qubits / 64)

    zp : np.ndarray, dtype=uint64, shape (n, words)
        Packed Z bits for each Pauli term.
        Must match `xp` shape exactly.

    pair_i : np.ndarray, dtype=int64, shape (m,)
    pair_j : np.ndarray, dtype=int64, shape (m,)
        Pair index arrays. pair_i[k] and pair_j[k] define the k-th pair.
        Expected (by convention from the caller):
          - i != j
          - indices are in [0, n-1]
          - no duplicate pairs if “count once globally” semantics are desired
          - often pair_i[k] < pair_j[k], though not required for correctness

    coeffs : np.ndarray, dtype=float64, shape (n,)
        Coefficients aligned with Pauli indices.
        Contribution for a non-commuting pair is coeffs[i] * coeffs[j].

    -------------------------------------------------------------------------
    Outputs
    -------------------------------------------------------------------------
    (count_noncomm, coeff_prod_sum) : (int64, float64)
        count_noncomm:
            Number of pairs among the provided list that are non-commuting.
        coeff_prod_sum:
            Sum of coeff products over those non-commuting pairs only.

    -------------------------------------------------------------------------
    Algorithm details
    -------------------------------------------------------------------------
    Commutation test uses packed symplectic parity:
        parity = popcount(x_i & z_j) + popcount(z_i & x_j)  (mod 2)
    commute iff parity == 0.

    Because xp/zp are packed in uint64 words, each pair check is O(words).

    -------------------------------------------------------------------------
    Complexity
    -------------------------------------------------------------------------
    Let m = len(pair_i) = len(pair_j), and words = xp.shape[1].
    Time:  O(m * words)
    Space: O(1) extra (excluding inputs)

    -------------------------------------------------------------------------
    Assumptions / safety
    -------------------------------------------------------------------------
    This function performs no bounds/shape validation (for speed).
    Caller is responsible for ensuring:
      - pair_i and pair_j have equal length,
      - xp and zp compatible shapes,
      - all indices are valid.

    Violating these assumptions may produce undefined behavior or runtime failure
    in nopython mode.
    """
    m = pair_i.shape[0]
    count_noncomm = 0
    coeff_prod_sum = 0.0
    for k in range(m):
        i = pair_i[k]
        j = pair_j[k]
        if not _commutes_pair_packed(xp, zp, i, j):
            count_noncomm += 1
            coeff_prod_sum += coeffs[i] * coeffs[j]
    return count_noncomm, coeff_prod_sum

@njit(cache=True, fastmath=True)
def _count_noncomm_unique_pairs_numba(xp, zp, pair_i, pair_j):
    """
    Count how many provided index pairs are non-commuting.

    -------------------------------------------------------------------------
    Purpose
    -------------------------------------------------------------------------
    Low-level numeric kernel that evaluates commutation for a precomputed list
    of index pairs and returns only the non-commuting pair count.

    This function does NOT build or deduplicate pairs. It assumes `pair_i/pair_j`
    already encode the exact pairs to evaluate (typically unique unordered pairs).

    -------------------------------------------------------------------------
    Inputs
    -------------------------------------------------------------------------
    xp : np.ndarray, dtype=uint64, shape (n, words)
        Packed X-bit table for n Pauli terms.
        - words = ceil(num_qubits / 64)
        - xp[t, w] stores 64 X bits of Pauli term t.

    zp : np.ndarray, dtype=uint64, shape (n, words)
        Packed Z-bit table for n Pauli terms.
        Must have same shape/layout as `xp`.

    pair_i : np.ndarray, dtype=int64, shape (m,)
    pair_j : np.ndarray, dtype=int64, shape (m,)
        Pair-index arrays. The k-th pair is (pair_i[k], pair_j[k]).
        Expected caller-side invariants:
          - pair_i.shape[0] == pair_j.shape[0]
          - all indices are in [0, n-1]
          - usually pair_i[k] < pair_j[k] and no duplicates
            (required only if you want global unique-pair semantics)

    -------------------------------------------------------------------------
    Output
    -------------------------------------------------------------------------
    count_noncomm : int64
        Number of pairs among the m input pairs that do NOT commute.

    -------------------------------------------------------------------------
    Commutation rule used
    -------------------------------------------------------------------------
    For pair (i, j), compute symplectic parity:
        parity = (popcount(x_i & z_j) + popcount(z_i & x_j)) mod 2
    - parity == 0  -> commute
    - parity == 1  -> non-commute (counted)

    -------------------------------------------------------------------------
    Complexity
    -------------------------------------------------------------------------
    Let m = number of pairs, words = packed uint64 columns.
    Time:  O(m * words)
    Space: O(1) extra (excluding input arrays)

    -------------------------------------------------------------------------
    Safety / validation
    -------------------------------------------------------------------------
    No shape/bounds checks are done here for speed.
    Caller must ensure valid, consistent inputs.
    """
    m = pair_i.shape[0]
    count_noncomm = 0
    for k in range(m):
        i = pair_i[k]
        j = pair_j[k]
        if not _commutes_pair_packed(xp, zp, i, j):
            count_noncomm += 1
    return count_noncomm


# ---------- user-facing function ----------

def noncomm_cross_group_count_and_coeff_product_sum(
    paulis: PauliList,
    groups,
    coeffs,
    xp: np.ndarray = None,
    zp: np.ndarray = None,
):
    """
    Compute non-commuting cross-group interactions with *global unordered-pair uniqueness*.

    -------------------------------------------------------------------------
    What this function computes
    -------------------------------------------------------------------------
    Let each Pauli term be identified by its index i in `paulis` (0 <= i < n).

    Given a collection of groups (each group is an iterable of Pauli indices), this
    function considers only pairs of indices that appear across different groups.

    IMPORTANT UNIQUENESS RULE:
      Each unordered index pair {i, j} with i != j is counted at most once globally,
      even if:
        - i or j appears multiple times in the same group,
        - groups overlap,
        - the same pair appears through multiple group-pair combinations.

    For every unique eligible pair {i, j}, it checks whether Pauli(i) and Pauli(j)
    commute (symplectic parity test). If they do NOT commute:
      - increment count by 1
      - add coeffs[i] * coeffs[j] to the sum

    Returns:
      (non_commuting_unique_pair_count, sum_of_coeff_products_over_those_pairs)

    -------------------------------------------------------------------------
    Parameters
    -------------------------------------------------------------------------
    paulis : qiskit.quantum_info.PauliList
        PauliList of length n containing all Pauli terms referenced by `groups`.

    groups : iterable[iterable[int]]
        Grouped Pauli indices. Each element is one group, containing indices into
        `paulis`.
        Example:
            groups = [
                [0, 1, 2],
                [2, 4],
                [1, 5, 6]
            ]
        Notes:
          - Indices may overlap across groups.
          - Duplicate indices within a group are allowed (they are internally deduped).
          - If fewer than 2 groups are provided, result is (0, 0.0).

    coeffs : array-like of float, shape (n,)
        Coefficient for each Pauli term by index. Must align with `paulis` indexing.
        `coeffs[i]` is used in pair product `coeffs[i] * coeffs[j]`.

    xp, zp : np.ndarray, optional
        Pre-packed X/Z symplectic bit tables, dtype=uint64, shape (n, words).
        If provided, they are used directly (skips repacking from `paulis`).
        If omitted, they are built internally from `paulis.x` and `paulis.z`.

        Packing convention:
          - words = ceil(num_qubits / 64)
          - qubit bit k stored in word (k // 64), offset (k % 64), LSB-first.

    -------------------------------------------------------------------------
    Returns
    -------------------------------------------------------------------------
    tuple[int, float]
        count_noncomm : int
            Number of unique unordered cross-group pairs {i, j} (i != j) that
            do not commute.
        coeff_prod_sum : float
            Sum of coeffs[i] * coeffs[j] over exactly those non-commuting pairs.

    -------------------------------------------------------------------------
    Validation / error behavior
    -------------------------------------------------------------------------
    - Raises ValueError if len(coeffs) != len(paulis).
    - Raises ValueError if any group index is outside [0, n-1].

    -------------------------------------------------------------------------
    Complexity notes
    -------------------------------------------------------------------------
    Let U be the number of unique unordered cross-group pairs after deduplication.
    The commutation stage is O(U * words), where words = ceil(num_qubits / 64).

    Pair construction is fully numba-compiled and does:
      1) cross-group pair generation (with possible duplicates),
      2) lexicographic sort,
      3) unique scan.
    This avoids Python set overhead and is better suited for very large workloads.

    -------------------------------------------------------------------------
    Semantics example
    -------------------------------------------------------------------------
    If groups = [[0, 1, 1], [1, 2], [0, 2]],
    cross-group raw appearances include repeated (0,2), (1,2), etc.
    This function evaluates each unordered pair only once:
      {0,1}, {0,2}, {1,2}
    (subject to actually appearing across at least one pair of distinct groups).

    -------------------------------------------------------------------------
    Practical tips
    -------------------------------------------------------------------------
    - For repeated calls with the same `paulis`, precompute `xp, zp` once and pass
      them in to avoid repeated packing.
    - Ensure `coeffs` uses the same indexing/order as `paulis`.
    - If you need signed/weighted variants beyond coeff product sum, modify only
      the final accumulation kernel once unique pairs are built.
    """
    
    n = len(paulis)
    if coeffs is not None:
        coeffs_arr = np.asarray(coeffs, dtype=np.float64)
        if coeffs_arr.shape[0] != n:
            raise ValueError(f"coeffs length ({coeffs_arr.shape[0]}) must match len(paulis) ({n}).")

    if xp is None or zp is None:
        x_bool = paulis.x.astype(np.bool_)
        z_bool = paulis.z.astype(np.bool_)
        xp = _pack_rows_bool_to_u64(x_bool)
        zp = _pack_rows_bool_to_u64(z_bool)

    groups = list(groups)
    ng = len(groups)
    if ng <= 1:
        return 0, 0.0

    total = sum(len(g) for g in groups)
    flat_in = np.empty(total, dtype=np.int64)
    off_in = np.empty(ng + 1, dtype=np.int64)

    k = 0
    off_in[0] = 0
    for gi, g in enumerate(groups):
        arr = np.asarray(list(g), dtype=np.int64)
        m = arr.shape[0]
        if m:
            flat_in[k:k + m] = arr
            k += m
        off_in[gi + 1] = k
    flat_in = flat_in[:k]

    flat_norm, off_norm = _normalize_groups_to_csr_numba(flat_in, off_in, n)
    # pair_i, pair_j = _build_unique_cross_group_pairs_numba(flat_norm, off_norm)
    pair_i, pair_j = _build_unique_cross_group_pairs_numba(flat_norm, off_norm, n)

    if pair_i.shape[0] == 0:
        if coeffs is None:
            return 0
        else:
            return 0, 0.0
    if coeffs is None:
        # count non-commuting among those unique pairs
        count = _count_noncomm_unique_pairs_numba(xp, zp, pair_i, pair_j)
        

        return int(count)
    else:   
        count, coeff_sum = _noncomm_count_and_coeffsum_unique_pairs_numba(
            xp, zp, pair_i, pair_j, coeffs_arr
            )
        return int(count), float(coeff_sum)
    

# ---------- core numba kernel ----------

# @njit(cache=True, fastmath=True)
# def _count_noncomm_unique_pairs_numba(xp, zp, pair_i, pair_j):
#     m = pair_i.shape[0]
#     count_noncomm = 0
#     for k in range(m):
#         i = pair_i[k]
#         j = pair_j[k]
#         if not _commutes_pair_packed(xp, zp, i, j):
#             count_noncomm += 1
#     return count_noncomm


# ---------- user-facing function ----------
# # for computing all non-commuting pairs
# def noncomm_cross_group_unique_pair_count(
#     paulis: PauliList,
#     groups,
#     xp: np.ndarray = None,
#     zp: np.ndarray = None,
# ):
#     """
#     Return number of UNIQUE unordered non-commuting pairs {i,j}, i!=j, that appear
#     across different groups (counted once globally).
#     """
#     n = len(paulis)

#     if xp is None or zp is None:
#         x_bool = paulis.x.astype(np.bool_)
#         z_bool = paulis.z.astype(np.bool_)
#         xp = _pack_rows_bool_to_u64(x_bool)
#         zp = _pack_rows_bool_to_u64(z_bool)

#     groups = list(groups)
#     ng = len(groups)
#     if ng <= 1:
#         return 0

#     # Python-side ragged -> CSR
#     total = sum(len(g) for g in groups)
#     flat_in = np.empty(total, dtype=np.int64)
#     off_in = np.empty(ng + 1, dtype=np.int64)

#     k = 0
#     off_in[0] = 0
#     for gi, g in enumerate(groups):
#         arr = np.asarray(list(g), dtype=np.int64)
#         m = arr.shape[0]
#         if m:
#             flat_in[k:k + m] = arr
#             k += m
#         off_in[gi + 1] = k
#     flat_in = flat_in[:k]

#     # normalize groups (in-range check + sort + dedup within group)
#     flat_norm, off_norm = _normalize_groups_to_csr_numba(flat_in, off_in, n)

#     # build global unique unordered cross-group pairs
#     pair_i, pair_j = _build_unique_cross_group_pairs_numba(flat_norm, off_norm)
#     if pair_i.shape[0] == 0:
#         return 0

#     # count non-commuting among those unique pairs
#     count = _count_noncomm_unique_pairs_numba(xp, zp, pair_i, pair_j)
#     return int(count)    
# ---------- user-facing function  ----------
#----------------- compare with qiskit ---------------------- 

# def test_compare_with_qiskit_group_commuting(
#     paulis: PauliList,
#     chunk_size: int = 1000,
#     enforce_nonmergeable_groups: bool = False,
# ):
#     """
#     Compare:
#       - qiskit: paulis.group_commuting(qubit_wise=False)
#       - custom: commuting_groups_chunked_numba_packed(...)

#     This test checks VALIDITY (not exact partition equality), because multiple
#     different valid groupings can exist.

#     Checked:
#       1) Both outputs are valid partitions of all input elements.
#       2) Each group in both outputs is internally pairwise commuting.
#       3) If enforce_nonmergeable_groups=True on custom output, verify no two
#          custom groups are fully cross-commuting (cannot be merged).

#     Returns
#     -------
#     dict summary.
#     Raises AssertionError on failure.
#     """
#     import numpy as np

#     n = len(paulis)

#     # ---------- run both ----------
#     custom_idx_groups = commuting_groups_chunked_numba_packed(
#         paulis=paulis,
#         chunk_size=chunk_size,
#         return_type="indices",
#         enforce_nonmergeable_groups=enforce_nonmergeable_groups,
#     )
#     qiskit_pl_groups = paulis.group_commuting(qubit_wise=False)  # list[PauliList]

#     # ---------- map qiskit groups back to source indices (handles duplicates) ----------
#     src_labels = paulis.to_labels()
#     label_to_idxs = {}
#     for i, lab in enumerate(src_labels):
#         label_to_idxs.setdefault(lab, []).append(i)

#     ptr = {lab: 0 for lab in label_to_idxs}
#     qiskit_idx_groups = []
#     for g in qiskit_pl_groups:
#         grp = []
#         for lab in g.to_labels():
#             if lab not in label_to_idxs:
#                 raise AssertionError(f"Qiskit produced unknown label '{lab}'.")
#             k = ptr[lab]
#             if k >= len(label_to_idxs[lab]):
#                 raise AssertionError(f"Too many occurrences of label '{lab}' while mapping back.")
#             grp.append(label_to_idxs[lab][k])
#             ptr[lab] = k + 1
#         qiskit_idx_groups.append(grp)

#     # ---------- helpers ----------
#     def assert_partition(groups, who):
#         flat = [i for grp in groups for i in grp]
#         assert len(flat) == n, f"{who}: assigned {len(flat)} elements, expected {n}"
#         assert len(set(flat)) == n, f"{who}: duplicates or missing indices"
#         assert set(flat) == set(range(n)), f"{who}: partition does not cover all indices"

#     def assert_internal_commuting(groups, who):
#         for grp in groups:
#             for a in range(len(grp)):
#                 ia = grp[a]
#                 for b in range(a + 1, len(grp)):
#                     ib = grp[b]
#                     if not paulis[ia].commutes(paulis[ib]):
#                         raise AssertionError(f"{who}: non-commuting pair in same group ({ia}, {ib})")

#     def groups_fully_cross_commute(g1, g2):
#         for i in g1:
#             pi = paulis[i]
#             for j in g2:
#                 if not pi.commutes(paulis[j]):
#                     return False
#         return True

#     # ---------- required checks ----------
#     assert_partition(custom_idx_groups, "Custom")
#     assert_partition(qiskit_idx_groups, "Qiskit")

#     assert_internal_commuting(custom_idx_groups, "Custom")
#     assert_internal_commuting(qiskit_idx_groups, "Qiskit")

#     # ---------- optional property check for custom ----------
#     if enforce_nonmergeable_groups:
#         for i in range(len(custom_idx_groups)):
#             for j in range(i + 1, len(custom_idx_groups)):
#                 if groups_fully_cross_commute(custom_idx_groups[i], custom_idx_groups[j]):
#                     raise AssertionError(
#                         f"Custom: groups {i} and {j} are fully cross-commuting, "
#                         "so they are still mergeable."
#                     )

#     return {
#         "n_paulis": n,
#         "custom_num_groups": len(custom_idx_groups),
#         "qiskit_num_groups": len(qiskit_idx_groups),
#         "custom_group_sizes": sorted(len(g) for g in custom_idx_groups),
#         "qiskit_group_sizes": sorted(len(g) for g in qiskit_idx_groups),
#         "enforce_nonmergeable_groups": enforce_nonmergeable_groups,
#         "ok": True,
#     }