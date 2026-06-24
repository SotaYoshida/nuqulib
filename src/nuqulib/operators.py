"""Functions to build operators in the M-scheme


"""
import numpy as np


def _state_Mtot(state: int, msps: list) -> int:
    Mtot = 0
    for idx, msp in enumerate(msps):
        if (state >> idx) & 1:
            Mtot += msp.jz
    return Mtot


def _ladder_coefficient(j2: int, m2: int, delta_m2: int) -> float:
    if delta_m2 == 2:
        return 0.5 * np.sqrt((j2 - m2) * (j2 + m2 + 2))
    if delta_m2 == -2:
        return 0.5 * np.sqrt((j2 + m2) * (j2 - m2 + 2))
    raise ValueError(f"delta_m2 must be +2 or -2, got {delta_m2}")


def _angular_momentum_ladder_terms(msps: list, delta_m2: int) -> list[tuple[int, int, float]]:
    sps_to_idx = {
        (msp.n, msp.l, msp.j, msp.jz, msp.tz): idx
        for idx, msp in enumerate(msps)
    }
    terms = []
    for from_idx, msp in enumerate(msps):
        to_jz = msp.jz + delta_m2
        if to_jz < -msp.j or to_jz > msp.j:
            continue
        to_idx = sps_to_idx.get((msp.n, msp.l, msp.j, to_jz, msp.tz))
        if to_idx is None:
            continue
        coeff = _ladder_coefficient(msp.j, msp.jz, delta_m2)
        if coeff != 0.0:
            terms.append((to_idx, from_idx, coeff))
    return terms


def _apply_fermion_one_body_term(
    state: int, create_idx: int, annihilate_idx: int
) -> tuple[int, int] | None:
    if ((state >> annihilate_idx) & 1) == 0:
        return None

    phase = -1 if (state & ((1 << annihilate_idx) - 1)).bit_count() % 2 else 1
    new_state = state ^ (1 << annihilate_idx)

    if (new_state >> create_idx) & 1:
        return None

    if (new_state & ((1 << create_idx) - 1)).bit_count() % 2:
        phase *= -1
    new_state |= 1 << create_idx
    return new_state, phase


def _apply_one_body_operator(
    state: int, terms: list[tuple[int, int, float]]
) -> dict[int, complex]:
    out = {}
    for create_idx, annihilate_idx, coeff in terms:
        applied = _apply_fermion_one_body_term(state, create_idx, annihilate_idx)
        if applied is None:
            continue
        new_state, phase = applied
        out[new_state] = out.get(new_state, 0.0) + coeff * phase
    return out
