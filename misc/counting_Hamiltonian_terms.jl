using Base.Threads
using Distributed
using Printf
using Random
using Dates
import Base.Iterators: product
using TimerOutputs
using StaticArrays
using StatProfilerHTML


mutable struct Orbit_nljjztz
    n::Int
    l::Int
    j::Int  # will store 2*j as Int
    jz::Int
    tz::Int
    e::Int
    function Orbit_nljjztz(n::Int, l::Int, j::Int, jz::Int, tz::Int)
        e = 2*n + l
        new(n, l, j, jz, tz, e)
    end
end

struct PauliMask{N}
    x::NTuple{N,UInt64}
    z::NTuple{N,UInt64}
end

mutable struct MeasBasisGroup{N}
    gx::NTuple{N,UInt64}   # group X mask
    gz::NTuple{N,UInt64}   # group Z mask
    terms::Vector{PauliMask{N}}
end

const IZ_patterns_diag = Dict(0=> (),
    1 => (('Z'), ),
    2 => (('Z', 'Z')),
    3 => (('Z', 'Z', 'Z')),
)

const ZZtuple = [('Z','Z')]

const IZ_patterns = Dict(0=> (),
    1 => (('I'), ('Z') ),
    2 => (('I', 'I'), ('Z', 'Z'), ('I', 'Z'), ('Z', 'I') ),
    3 => (('I', 'I', 'I'), ('Z', 'Z', 'Z'),
          ('I', 'I', 'Z'), ('I', 'Z', 'I'), ('Z', 'I', 'I'),
          ('I', 'Z', 'Z'), ('Z', 'I', 'Z'), ('Z', 'Z', 'I'))
)

const XY_patterns_T1_2 = [('X','X'), ('Y','Y')]

const XY_patterns_T1_4 = [('X', 'X', 'X', 'X'), ('Y', 'Y', 'Y', 'Y'),
         ('X', 'X', 'Y', 'Y'), ('X', 'Y', 'X', 'Y'), ('X', 'Y', 'Y', 'X'),
         ('Y', 'Y', 'X', 'X'), ('Y', 'X', 'Y', 'X'), ('Y', 'X', 'X', 'Y')]

const XY_patterns_ppn_2 = [('X','X'), ('X','Y'), ('Y','X'), ('Y','Y')]

const XY_patterns_ppn_4 = [
    ('X', 'X', 'X', 'X'), ('Y', 'Y', 'Y', 'Y'),
    ('X', 'X', 'Y', 'Y'), ('X', 'Y', 'X', 'Y'), ('X', 'Y', 'Y', 'X'),
    ('Y', 'Y', 'X', 'X'), ('Y', 'X', 'Y', 'X'), ('Y', 'X', 'X', 'Y'),
    ('X', 'X', 'X', 'Y'), ('X', 'X', 'Y', 'X'), ('X', 'Y', 'X', 'X'), ('Y', 'X', 'X', 'X'),
    ('Y', 'Y', 'Y', 'X'), ('Y', 'Y', 'X', 'Y'), ('Y', 'X', 'Y', 'Y'), ('X', 'Y', 'Y', 'Y')
]

const XY_patterns_T3 = Dict(2=> [('X', 'X'), ('Y', 'Y')],
    4=> [('X', 'X', 'X', 'X'), ('Y', 'Y', 'Y', 'Y'),
         ('X', 'X', 'Y', 'Y'), ('X', 'Y', 'X', 'Y'), ('X', 'Y', 'Y', 'X'),
         ('Y', 'Y', 'X', 'X'), ('Y', 'X', 'Y', 'X'), ('Y', 'X', 'X', 'Y')],
    6 => [('X', 'X', 'X', 'X', 'X', 'X'), ('Y', 'Y', 'X', 'X', 'X', 'X'), 
          ('Y', 'X', 'Y', 'X', 'X', 'X'), ('X', 'Y', 'Y', 'X', 'X', 'X'), 
          ('Y', 'X', 'X', 'Y', 'X', 'X'), ('X', 'Y', 'X', 'Y', 'X', 'X'), 
          ('X', 'X', 'Y', 'Y', 'X', 'X'), ('Y', 'Y', 'Y', 'Y', 'X', 'X'), 
          ('Y', 'X', 'X', 'X', 'Y', 'X'), ('X', 'Y', 'X', 'X', 'Y', 'X'), 
          ('X', 'X', 'Y', 'X', 'Y', 'X'), ('Y', 'Y', 'Y', 'X', 'Y', 'X'), 
          ('X', 'X', 'X', 'Y', 'Y', 'X'), ('Y', 'Y', 'X', 'Y', 'Y', 'X'), 
          ('Y', 'X', 'Y', 'Y', 'Y', 'X'), ('X', 'Y', 'Y', 'Y', 'Y', 'X'), 
          ('Y', 'X', 'X', 'X', 'X', 'Y'), ('X', 'Y', 'X', 'X', 'X', 'Y'), 
          ('X', 'X', 'Y', 'X', 'X', 'Y'), ('Y', 'Y', 'Y', 'X', 'X', 'Y'), 
          ('X', 'X', 'X', 'Y', 'X', 'Y'), ('Y', 'Y', 'X', 'Y', 'X', 'Y'), 
          ('Y', 'X', 'Y', 'Y', 'X', 'Y'), ('X', 'Y', 'Y', 'Y', 'X', 'Y'), 
          ('X', 'X', 'X', 'X', 'Y', 'Y'), ('Y', 'Y', 'X', 'X', 'Y', 'Y'), 
          ('Y', 'X', 'Y', 'X', 'Y', 'Y'), ('X', 'Y', 'Y', 'X', 'Y', 'Y'), 
          ('Y', 'X', 'X', 'Y', 'Y', 'Y'), ('X', 'Y', 'X', 'Y', 'Y', 'Y'), 
          ('X', 'X', 'Y', 'Y', 'Y', 'Y'), ('Y', 'Y', 'Y', 'Y', 'Y', 'Y')]
)


const local_set_per_thread = OncePerThread{Set{String}}() do
    s = Set{String}()
    return s
end

const local_vect_per_thread = OncePerThread{Vector{BigInt}}() do
    v = [BigInt(0) for _ in 1:6]
    return v
end


function _combinations(iterable::AbstractVector, k::Int)
    n = length(iterable)
    if k == 0
        return [Tuple{}]
    end
    res = []
    function rec(start, curr)
        if length(curr) == k
            push!(res, tuple(curr...))
            return
        end
        for i in start:(n - (k - length(curr)) + 1)
            push!(curr, iterable[i])
            rec(i+1, curr)
            pop!(curr)
        end
    end
    rec(1, Int[])
    return res
end


function count_msps(emax::Int, vemin::Int=0, vemax::Int=100)
    msps_p = Orbit_nljjztz[]
    msps_n = Orbit_nljjztz[]
    for te in 0:emax
        if te < vemin || te > vemax
            continue
        end
        for l in 0:te
            n = (te - l) ÷ 2
            if n < 0 || 2*n + l != te
                continue
            end
            j_vals = l > 0 ? [l - 0.5, l + 0.5] : [0.5]
            for j in j_vals
                j2 = Int(2*j)
                for jz in -j2:2:j2
                    push!(msps_p, Orbit_nljjztz(n, l, j2, jz, -1))
                    push!(msps_n, Orbit_nljjztz(n, l, j2, jz, +1))
                end
            end
        end
    end
    return msps_p, msps_n
end

function write_pauli_terms(io, pauli_terms; label="")
    if io == stdout
        return nothing
    end
    for term in pauli_terms
        println(io, label *term)
    end
    return nothing
end

function Jbracket_check(bra1::Orbit_nljjztz, bra2::Orbit_nljjztz,
                        ket1::Orbit_nljjztz, ket2::Orbit_nljjztz;
                        same_species::Bool=true, verbose::Bool=false)
    Jmin_b = abs(bra1.j - bra2.j)
    Jmax_b = bra1.j + bra2.j
    Jmin_k = abs(ket1.j - ket2.j)
    Jmax_k = ket1.j + ket2.j

    # iterate over bra J-values (step 2) and check membership & conditions on-the-fly
    @inbounds for tJ in Jmin_b:2:Jmax_b
        if tJ < Jmin_k || tJ > Jmax_k
            continue
        end
        if same_species
            if (bra1.n == bra2.n && bra1.l == bra2.l && bra1.j == bra2.j && bra1.tz == bra2.tz) &&
               ((tJ ÷ 2) % 2 != 0)
                continue
            end
            if (ket1.n == ket2.n && ket1.l == ket2.l && ket1.j == ket2.j && ket1.tz == ket2.tz) &&
               ((tJ ÷ 2) % 2 != 0)
                continue
            end
        end
        if abs(bra1.jz + bra2.jz) <= tJ && abs(ket1.jz + ket2.jz) <= tJ
            return true
        end
    end
    return false
end

function Jbracket_check_3body_T3(
    bra1::Orbit_nljjztz, bra2::Orbit_nljjztz, bra3::Orbit_nljjztz,
    ket1::Orbit_nljjztz, ket2::Orbit_nljjztz, ket3::Orbit_nljjztz;
    verbose::Bool=false
)

    if !(bra1.tz == bra2.tz == bra3.tz == ket1.tz == ket2.tz == ket3.tz)
        return false
    end

    J12min_b = abs(bra1.j - bra2.j)
    J12max_b = bra1.j + bra2.j
    J12min_k = abs(ket1.j - ket2.j)
    J12max_k = ket1.j + ket2.j

    @inbounds for tJ12 in J12min_b:2:J12max_b
        if bra1.n == bra2.n && bra1.l == bra2.l &&
           bra1.j == bra2.j && bra1.tz == bra2.tz
            if ((tJ12 ÷ 2) % 2) != 1
                continue
            end
        end
        # This removes <1, 2, 3|V|1, 2, 6> type terms
        # It may be insufficient in some cases, but enough for counting purposes.
        if bra1.n == bra2.n == ket1.n == ket2.n &&
           bra1.l == bra2.l == ket1.l == ket2.l &&
           bra1.j == bra2.j == ket1.j == ket2.j
            if ((tJ12 ÷ 2) % 2) == 1
                continue
            end
        end
        if tJ12 < J12min_k || tJ12 > J12max_k
            continue
        end

        if ket1.n == ket2.n && ket1.l == ket2.l &&
           ket1.j == ket2.j && ket1.tz == ket2.tz
            if ((tJ12 ÷ 2) % 2) != 1
                continue
            end
        end

        Jmin_b = abs(tJ12 - bra3.j)
        Jmax_b = tJ12 + bra3.j
        Jmin_k = abs(tJ12 - ket3.j)
        Jmax_k = tJ12 + ket3.j
        for tJ in max(Jmin_b, Jmin_k):2:min(Jmax_b, Jmax_k)
            if abs(bra1.jz + bra2.jz + bra3.jz) > tJ
                continue
            end
            if abs(ket1.jz + ket2.jz + ket3.jz) > tJ
                continue
            end
            return true
        end
    end
    return false
end

# function Jbracket_check(bra1::Orbit_nljjztz, bra2::Orbit_nljjztz,
#                        ket1::Orbit_nljjztz, ket2::Orbit_nljjztz;
#                        same_species::Bool=true, verbose::Bool=false)
#     Jmin_b = abs(bra1.j - bra2.j)
#     Jmax_b = bra1.j + bra2.j
#     Jmin_k = abs(ket1.j - ket2.j)
#     Jmax_k = ket1.j + ket2.j
#     Jbra = Jmin_b:2:Jmax_b
#     Jket = Jmin_k:2:Jmax_k
#     Jbra_list = collect(Jbra)
#     Jket_list = collect(Jket)
#     if same_species
#         Jbra_list = [tJ for tJ in Jbra_list if (tJ ÷ 2 % 2 == 0) || !(bra1.n == bra2.n && bra1.l == bra2.l && bra1.j == bra2.j && bra1.tz == bra2.tz)]
#         Jket_list = [tJ for tJ in Jket_list if (tJ ÷ 2 % 2 == 0) || !(ket1.n == ket2.n && ket1.l == ket2.l && ket1.j == ket2.j && ket1.tz == ket2.tz)]
#     end
#     Jpossible = intersect(Set(Jbra_list), Set(Jket_list))
#     Jpossible = [tJ for tJ in Jpossible if abs(bra1.jz + bra2.jz) <= tJ && abs(ket1.jz + ket2.jz) <= tJ]
#     return !isempty(Jpossible)
# end


function generate_JW_endoded_NN_T1(
                      Nchunk::Int,
                      diag_idxs::Vector{UInt16},
                      nond_idxs::BitVector,
                      zmask::BitVector,
                      n_qubits::Int,
                      XY_pattern::Vector{Tuple{Char, Char}},
                      io,
                      to::TimerOutput) #where {Nchunk}
    xp = MVector{Nchunk,UInt64}(zeros(UInt64, Nchunk))
    zp = MVector{Nchunk,UInt64}(zeros(UInt64, Nchunk))
    xn = MVector{Nchunk,UInt64}(zeros(UInt64, Nchunk))
    zn = MVector{Nchunk,UInt64}(zeros(UInt64, Nchunk))
    num_arr = 0
    local_arr = Vector{PauliMask{Nchunk}}()
    for pauli_xy in XY_pattern
        fill!(xp, UInt64(0))
        fill!(zp, UInt64(0))
        nond_count = 0
        for (q, tf) in enumerate(nond_idxs)
            if !tf
                continue
            end
            nond_count += 1
            p = pauli_xy[nond_count]
            set_pauli!(xp, zp, q, UInt8(p))
        end
        for q in 1:n_qubits
            if zmask[q] 
                set_pauli!(xp, zp, q, UInt8('Z'))
            end
        end

        if isempty(diag_idxs)
            fill!(xn, UInt64(0))
            fill!(zn, UInt64(0))
            x = Tuple(xp) .| Tuple(xn)
            z = Tuple(zp) .| Tuple(zn)
            push!(local_arr, PauliMask{Nchunk}(x, z))
            num_arr += 1
        else
            for idx_d in 1:length(diag_idxs)
                fill!(xn, UInt64(0))
                fill!(zn, UInt64(0))
                pos = Int(diag_idxs[idx_d])
                set_pauli!(xn, zn, pos, UInt8('Z'))
                x = ntuple(i -> xp[i] ⊻ xn[i], Nchunk)
                z = ntuple(i -> zp[i] ⊻ zn[i], Nchunk)
                push!(local_arr, PauliMask{Nchunk}(x, z))
                num_arr += 1
            end
        end
    end
    @assert length(Set(local_arr)) == num_arr "num_arr $num_arr != length(Set(local_arr)) $(length(Set(local_arr)))"

    # code to write out will be made
    for term in local_arr
        pauli_str = decode_pauli_string(term.x, term.z, n_qubits)
        write(io, "pp: "* pauli_str * "\n")
    end

    return nothing
end


function _process_triplet_terms!(local_set::Set{String}, 
                                 pauli_patterns::Vector{NTuple{N, Char}},
                                 Nchunk::Int,
                                 nond_idxs::Vector{Int},
                                 zmask::BitVector,
                                 vector_diag_idxs::Vector{Vector{Int}},
                                 n_qubits_p::Int, n_qubits::Int,
                                 write_pauli::Bool) where {N}
    num_terms = 0
    num_nond = length(nond_idxs)

    xp = zp = xn = zn = nothing
    if write_pauli
        xp = MVector{Nchunk,UInt64}(zeros(UInt64, Nchunk))
        zp = MVector{Nchunk,UInt64}(zeros(UInt64, Nchunk))
        xn = MVector{Nchunk,UInt64}(zeros(UInt64, Nchunk))
        zn = MVector{Nchunk,UInt64}(zeros(UInt64, Nchunk))
    end
    for pauli_xy in pauli_patterns
        if write_pauli 
            fill!(xp, UInt64(0))
            fill!(zp, UInt64(0))
            for idx in 1:num_nond
                pos = nond_idxs[idx]
                p = pauli_xy[idx]
                set_pauli!(xp, zp, pos, UInt8(p))
            end

            for q in 1:n_qubits
                if zmask[q] 
                    set_pauli!(xp, zp, q, UInt8('Z'))
                end
            end
        end

        for diag_idxs in vector_diag_idxs
            num_d = length(diag_idxs)
            iz_pattern = IZ_patterns[num_d] 

            if isempty(diag_idxs) # This does not have overlap with NN-terms
                num_terms += 1
                if write_pauli
                    bitstr = decode_pauli_string(xp, zp, n_qubits)
                    union!(local_set, [bitstr])
                end
                break
            else                
                if write_pauli 
                    fill!(xn, UInt64(0))
                    fill!(zn, UInt64(0))
                    for idx in 1:num_d
                        pos = diag_idxs[idx]
                        set_pauli!(xn, zn, pos, UInt8('Z'))
                    end
                    bitstr = decode_pauli_string(xp .⊻ xn, zp .⊻ zn, n_qubits)
                    union!(local_set, [bitstr])
                end
                num_terms += 1
            end
        end
    end
    return num_terms
end


@inline function count_XY(p1::Char, p2::Char)
    cx = (p1 == 'X') + (p2 == 'X')
    cy = (p1 == 'Y') + (p2 == 'Y')
    return cx, cy
end

@inline function set_pauli!(
    x::MVector{N,UInt64},
    z::MVector{N,UInt64},
    q::Int,
    p::UInt8,
    nshift::Int = 0
) where {N}
    idx = q - 1 + nshift
    chunk = (idx >>> 6) + 1
    bit   = UInt64(1) << (idx & 63)

    @inbounds begin
        if p == UInt8('X')
            x[chunk] |= bit
        elseif p == UInt8('Y')
            x[chunk] |= bit
            z[chunk] |= bit
        elseif p == UInt8('Z')
            z[chunk] |= bit
        elseif p == UInt8('I')
            # Those should be zero
            x[chunk] &= ~bit
            z[chunk] &= ~bit
        end
    end
    return nothing
end

@inline function apply_JW_Z!(
    z::MVector{N,UInt64},
    qrange::UnitRange{Int},
    nshift::Int
) where {N}
    for q in qrange
        idx = q - 1 + nshift
        chunk = (idx >>> 6) + 1
        bit   = UInt64(1) << (idx & 63)
        @inbounds z[chunk] |= bit
    end
    return nothing
end

"""
Since proton (neutron) part in pn-terms have only two non-diagonal operators,
Z-strings can be generated only between the two non-diagonal operators.
"""
function pn_loop_dev!(local_set::Set{PauliMask{Nchunk}},
                      iter_p::Vector{Tuple{Char, Char}},
                      iter_n::Vector{Tuple{Char, Char}},
                      cre_p::Int, ani_p::Int, cre_n::Int, ani_n::Int,
                      n_qubits_p::Int, n_qubits_n::Int,
                      write_pauli::Bool,
                      io
                      ) where {Nchunk}
    local_arr = nothing
    xp = zp = xn = zn = nothing
    if write_pauli
        local_arr = PauliMask{Nchunk}[]
        xp = MVector{Nchunk,UInt64}(zeros(UInt64, Nchunk))
        zp = MVector{Nchunk,UInt64}(zeros(UInt64, Nchunk))
        xn = MVector{Nchunk,UInt64}(zeros(UInt64, Nchunk))
        zn = MVector{Nchunk,UInt64}(zeros(UInt64, Nchunk))
    end

    num_term_in_ch = 0
    czp = czn = 0
    for (pp1, pp2) in iter_p
        if write_pauli
            fill!(xp, UInt64(0))
            fill!(zp, UInt64(0))
        end
        # Count X/Y on proton side (cheap)
        cxp = (pp1 == 'X') + (pp2 == 'X')
        cyp = (pp1 == 'Y') + (pp2 == 'Y')
        czp = (pp1 == 'Z') + (pp2 == 'Z')
        isodd(cxp + cyp) && continue


        if cre_p != ani_p
            if write_pauli
                set_pauli!(xp, zp, cre_p, UInt8(pp1))
                set_pauli!(xp, zp, ani_p, UInt8(pp2))
            end
        else # For the case of cre_p == ani_p
            if write_pauli
                set_pauli!(xp, zp, cre_p, UInt8('Z'))
            end
            czp += 1
        end
        # Apply JW Z strings between cre_p and ani_p       
        min_q = min(cre_p, ani_p)+1
        max_q = max(cre_p, ani_p)-1
        if write_pauli
            apply_JW_Z!(zp, min_q:max_q, 0)
        end
        czp += max(0, max_q - min_q + 1)

        # Skip trivial identity
        if cxp + cyp + czp == 0
            continue
        end

        for (np1, np2) in iter_n
            if write_pauli
                fill!(xn, UInt64(0))
                fill!(zn, UInt64(0))
            end
            cxn = (np1 == 'X') + (np2 == 'X')
            cyn = (np1 == 'Y') + (np2 == 'Y')
            isodd(cxn + cyn) && continue

            # Cross parity rules
            ((cxp + cxn) & 1 != 0) && continue
            ((cyp + cyn) & 1 != 0) && continue
            
            if write_pauli
                if cre_n != ani_n
                    set_pauli!(xn, zn, cre_n, UInt8(np1), n_qubits_p)
                    set_pauli!(xn, zn, ani_n, UInt8(np2), n_qubits_p)
                else
                    set_pauli!(xn, zn, cre_n, UInt8('Z'), n_qubits_p)
                end
                apply_JW_Z!(zn, min(cre_n, ani_n)+1:max(cre_n, ani_n)-1, n_qubits_p)
            end

            # if all(iszero, xn) && all(iszero, zn)
            #     continue
            # end

            # Merge neutron + proton 
            if write_pauli 
                x = Tuple(xp) .| Tuple(xn)
                z = Tuple(zp) .| Tuple(zn)
                push!(local_arr, PauliMask{Nchunk}(x, z))
            end
            num_term_in_ch += 1
        end
    end    
    if write_pauli
        @assert length(Set(local_arr)) == length(local_arr) "Duplicate terms found in local_arr!"
        for term in local_arr
            pauli_str = decode_pauli_string(term.x, term.z, n_qubits_p + n_qubits_n)
            write(io, "pn: "* pauli_str * "\n")
        end
    end
    return num_term_in_ch 
end

function generate_ppp_args(msps_p, emin, emax, e3min, e3max, n_qubits; chunk_stride=1)
    n_qubits_p = length(msps_p)
    nond_idxs = Int[ ]
    diag_idxs = Int16[ ]
    args = Dict{Vector{Int}, Vector{Vector{Int}}}()  # key: nond_idxs, value: diag_idxs
    for ibra1 in (length(msps_p)-2):-chunk_stride:1
        bra1 = msps_p[ibra1]
        for ibra2 in (ibra1+1):(n_qubits_p-1)
            bra2 = msps_p[ibra2]
            if !(emin <= bra2.e <= emax) continue end
            for ibra3 in (ibra2+1):n_qubits_p
                bra3 = msps_p[ibra3]
                if !(emin <= bra3.e <= emax) continue end
                ebra = bra1.e + bra2.e + bra3.e
                if !(e3min <= ebra <= e3max) continue end
                Mbra = bra1.jz + bra2.jz + bra3.jz
                Pbra = (bra1.l + bra2.l + bra3.l) % 2
                for iket1 = ibra1:n_qubits_p
                    ket1 = msps_p[iket1]
                    if !(emin <= ket1.e <= emax) continue end
                    for iket2 in (iket1+1):n_qubits_p
                        ket2 = msps_p[iket2]
                        if !(emin <= ket2.e <= emax) continue end
                        for iket3 in (iket2+1):n_qubits_p
                            ket3 = msps_p[iket3]
                            if !(emin <= ket3.e <= emax) continue end
                            eket = ket1.e + ket2.e + ket3.e
                            if !(e3min <= eket <= e3max) continue end
                            Mket = ket1.jz + ket2.jz + ket3.jz
                            Pket = (ket1.l + ket2.l + ket3.l) % 2
                            if Pbra != Pket || Mbra != Mket
                                continue
                            end
                            Jbracket_check_3body_T3(bra1, bra2, bra3, ket1, ket2, ket3)|| continue
                            empty!(nond_idxs)
                            empty!(diag_idxs)
                            for idx_target in (ibra1, ibra2, ibra3, iket1, iket2, iket3)
                                count_same = 0
                                for idx_other in (ibra1, ibra2, ibra3, iket1, iket2, iket3)
                                    if idx_target == idx_other
                                        count_same += 1
                                    end
                                end
                                if count_same == 1
                                    push!(nond_idxs, idx_target)    
                                else
                                    if !(idx_target in diag_idxs)
                                        push!(diag_idxs, idx_target)
                                    end
                                end
                            end                            
                            @assert length(diag_idxs) * 2 + length(nond_idxs) == 6 "Total indices not equal to 6! ($ibra1, $ibra2, $ibra3, $iket1, $iket2, $iket3) nond_idxs $nond_idxs diag_idxs $diag_idxs"

                            if length(nond_idxs) == 0
                                # This has been already counted in diag part.
                                continue
                            end
                            @assert Set(nond_idxs) ∩ Set(diag_idxs) == Set{Int}() "Some indices are both in nond_idxs and diag_idxs! nond $nond_idxs diag $diag_idxs"
                            sort!(nond_idxs)
                            sort!(diag_idxs)
                            tkey = copy(nond_idxs)

                            if !haskey(args, tkey)
                                args[tkey] = Vector{Vector{Int}}()
                            end
                            if !(diag_idxs in args[tkey])
                                push!(args[tkey], copy(diag_idxs))
                            end
                        end
                    end
                end
            end
        end
    end
    return args
end


function Tcounts_add_sofar!(Hterms_set::Set{String}, 
                            vect_Tops::Vector{BigInt};                     
                            label::String="",                             
                            symmfactor::Int=1,
                            cost_CH_u::Int=4, cost_CH_cx::Int=2, cost_CH_T::Int=0,
                            cost_CS_u::Int=4, cost_CS_cx::Int=2, cost_CS_T::Int=0,
                            cost_Toffoli_u::Int=9, cost_Toffoli_cx::Int=6, cost_Toffoli_T::Int=4,
                            cost_CRz_u::Int=3, cost_CRz_cx::Int=2, cost_CRz_T::Int=2,
                            )
    Tfac_on_Rz = 100
    num_u = num_cx = num_Testimate = 0
    num_u_ = num_cx_ = num_Testimate_ = 0
    
    for bitstr in Hterms_set        
        count_X = count(x->x=='X', bitstr)
        count_Y = count(x->x=='Y', bitstr)
        count_Z = count(x->x=='Z', bitstr)
        non_I = count_X + count_Y + count_Z
        if non_I == 0
            continue
        end

        # New (correct?) way: 
        num_cx += 2*(non_I-1) # ladder
        num_cx_ += 2* non_I #ladder + c-Rz (2)

        num_Testimate  += Tfac_on_Rz 
        num_Testimate_ += 2 * Tfac_on_Rz 

    end
    vect_Tops[1] += num_u * symmfactor
    vect_Tops[2] += num_cx * symmfactor
    vect_Tops[3] += num_Testimate * symmfactor
    vect_Tops[4] += num_u_ * symmfactor
    vect_Tops[5] += num_cx_ * symmfactor
    vect_Tops[6] += num_Testimate_ * symmfactor
    return nothing
end

function update_zmask!(
    n::Int,
    nond_idxs,
    zmask::BitVector
)
    parity = false
    @inbounds for q in n:-1:1
        zmask[q] = parity & !( q in nond_idxs)
        parity ⊻= q in nond_idxs
    end
    return nothing
end


function update_zmask!(
    n::Int,
    nond_idxs::BitVector,
    zmask::BitVector
)
    parity = false
    @inbounds for q in n:-1:1
        zmask[q] = parity & !nond_idxs[q]
        parity ⊻= nond_idxs[q]
    end
    return nothing
end


function possible_2b_T1_terms!(Measurement_Basis,
                               vect_Tops, num_of_Hterms,
                               msps_p, msps_n, to, io,
                               write_pauli,
                               Nchunk::Int,
                               is_3NF::Bool,                               
                               count_Ppattern::Bool,
                               count_gates_during_process::Bool,
                               debug_mode::Int;
                               emin::Int=0, emax::Int=100,                               
                               symmfactor::Int=1,
                               pp_bucket_mod::Int=1,
                               ) 

    println("Processing pp-channel ...")
    n_qubits_p = length(msps_p)
    n_qubits_n = length(msps_n)
    n_qubits = n_qubits_p + n_qubits_n

    if pp_bucket_mod < 1
        error("pp_bucket_mod must be >= 1")
    end

    num_of_T1terms = 0 
    zmask = BitVector(falses(n_qubits))
    @timeit to "T1-loop" for bucket_id in 0:(pp_bucket_mod - 1)
        #println("  pp-bucket $(bucket_id + 1)/$pp_bucket_mod")
        bucket = Dict{UInt64, Vector{UInt16}}()
        @timeit to "gen.args" @inbounds for ibra1 in 1:(n_qubits_p - 1)
            bra1 = msps_p[ibra1]
            if !(emin <= bra1.e <= emax) continue end
            for ibra2 in (ibra1+1):n_qubits_p
                bra2 = msps_p[ibra2]
                if !(emin <= bra2.e <= emax) continue end
                Mbra = bra1.jz + bra2.jz
                Pbra = (bra1.l + bra2.l) % 2
                for iket1 = ibra1:n_qubits_p
                    ket1 = msps_p[iket1]
                    if !(emin <= ket1.e <= emax) continue end
                    iket2_start = iket1+1
                    if ibra1 == iket1
                        iket2_start = ibra2+1
                    end
                    for iket2 in iket2_start:n_qubits_p
                        ket2 = msps_p[iket2]
                        if ibra1 == iket1 && ibra2 == iket2
                            continue
                        end
                        if !(emin <= ket2.e <= emax) continue end
                        Mket = ket1.jz + ket2.jz 
                        Pket = (ket1.l + ket2.l) % 2
                        if Pbra != Pket || Mbra != Mket
                            continue
                        end
                        if !Jbracket_check(bra1, bra2, ket1, ket2)
                            continue
                        end

                        bra1_idx = ibra1
                        bra2_idx = ibra2
                        ket1_idx = iket1
                        ket2_idx = iket2

                        if bra2_idx > ket1_idx
                            bra2_idx, ket1_idx = ket1_idx, bra2_idx
                        end
                        if ket1_idx > ket2_idx
                            ket1_idx, ket2_idx = ket2_idx, ket1_idx
                        end

                        tkey = encode_nondiag_UInt64_pp(bra1_idx, bra2_idx, ket1_idx, ket2_idx)
                        if Int(mod(tkey, UInt64(pp_bucket_mod))) != bucket_id
                            continue
                        end

                        diag_idx = 0
                        if bra1_idx == bra2_idx
                            diag_idx = bra1_idx
                        elseif bra2_idx == ket1_idx
                            diag_idx = bra2_idx
                        elseif ket1_idx == ket2_idx
                            diag_idx = ket1_idx
                        end

                        if !haskey(bucket, tkey)
                            bucket[tkey] = Vector{UInt16}()
                        end
                        if diag_idx != 0
                            v = bucket[tkey]
                            if !(UInt16(diag_idx) in v)
                                push!(v, UInt16(diag_idx))
                            end
                        end
                    end
                end
            end
        end
        local_count = 0
        nond_idxs = BitVector(falses(n_qubits))
        for (tkey, diag_idxs) in bucket
            len_nond = (tkey >> 32) == 0 ? 2 : 4
            XY_pattern = len_nond == 2 ? XY_patterns_T1_2 : XY_patterns_T1_4            
            num_arr = length(XY_pattern)
            if len_nond == 4
                local_count += num_arr
            else
                dcount = length(diag_idxs)
                dcount = dcount == 0 ? 1 : dcount
                local_count += num_arr * dcount
            end
            if write_pauli
                get_nond_idxs!(tkey, nond_idxs)
                update_zmask!(n_qubits, nond_idxs, zmask)           
                generate_JW_endoded_NN_T1(Nchunk, diag_idxs, nond_idxs, zmask,
                                          n_qubits, XY_pattern, io, to)

            end
        end
        num_of_T1terms += local_count
        empty!(bucket)
    end

    num_of_Hterms += num_of_T1terms * symmfactor
    println("# of pp terms: $(num_of_T1terms) num_of_Hterms so far: ", num_of_Hterms)
    return num_of_Hterms
end


function generate_diag_terms(msps_p, msps_n, emax, e3max, Z, N, is_3NF)
    diag_terms = Dict{Int, Set{String}}()
    n_qubits_p = length(msps_p)
    n_qubits_n = length(msps_n)
    n_qubits = n_qubits_p + n_qubits_n
    rank = is_3NF ? 3 : 2
    for num_of_Z in 0:rank
        diag_terms[num_of_Z] = Set{String}()
        for qubit_idxs in _combinations(collect(1:(n_qubits)), num_of_Z)
            bitlist = fill('I', n_qubits)
            if num_of_Z == 0
                bitstr = String(bitlist[end:-1:1])
                push!(diag_terms[num_of_Z], bitstr)
                continue
            end
            te3max = 0
            for idx in qubit_idxs
                bitlist[idx] = 'Z'
                if idx <= n_qubits_p
                    te3max += msps_p[idx].e
                else
                    te3max += msps_n[idx - n_qubits_p].e
                end
            end
            Z_p = count(x->x=='Z', bitlist[1:n_qubits_p])
            Z_n = num_of_Z - Z_p
            if Z_p > Z || Z_n > N
                continue
            end
            if te3max > e3max && is_3NF
                continue
            end
            bitstr = String(bitlist[end:-1:1])
            push!(diag_terms[num_of_Z], bitstr)
        end
        println(" #diag terms with $num_of_Z Z: $(length(diag_terms[num_of_Z]))")
        if e3max == emax * 3 && Z >= 3 && N >= 3
            # One can easily count the expected number of diag terms for full e3max = emax * 3
            if num_of_Z == 2
                expected = n_qubits * (n_qubits - 1) ÷ 2
                tf = length(diag_terms[num_of_Z]) == expected
                if !tf
                    println("Z(2) from combinatorics: $expected")
                    println("Z(2) from generated terms: $(length(diag_terms[num_of_Z]))")
                end
                @assert tf "Mismatch in diag term count for #Z=2"
            end
            if num_of_Z == 3
                expected = 2 * n_qubits_p * (n_qubits_p - 1) * (n_qubits_p - 2) ÷ 6
                expected += 2 * n_qubits_p * (n_qubits_p - 1) ÷ 2 * n_qubits_n
                tf = length(diag_terms[num_of_Z]) == expected
                if !tf
                    println("Z(3) from combinatorics: $expected")
                    println("Z(3) from generated terms: $(length(diag_terms[num_of_Z]))")
                end
                @assert tf "Mismatch in diag term count for #Z=3"
            end
        end
    end
    println("Generated $(length(diag_terms)) diagonal terms (rank $rank)")
    return diag_terms
end


function loop_over_pp_in_ppn!(ibra_n::Int, iket_n::Int, 
                              msps_p::Vector{Orbit_nljjztz}, msps_n::Vector{Orbit_nljjztz}, 
                              vemin::Int, vemax::Int,
                              e3min::Int, e3max::Int,
                              res:: Dict{UInt64, Vector{NTuple{2, UInt64}}})
    n_qubits_p = length(msps_p)
    bra_n = msps_n[ibra_n]
    ket_n = msps_n[iket_n]
    n_pattern = encode_2(ibra_n, iket_n)
    for ibra_1 in 1:n_qubits_p
        bra1 = msps_p[ibra_1]
        if !(vemin <= bra1.e <= vemax) continue end
        for ibra_2 in (ibra_1+1):n_qubits_p
            bra2 = msps_p[ibra_2]
            if !(vemin <= bra2.e <= vemax) continue end
            Mbra = bra1.jz + bra2.jz + bra_n.jz
            Pbra = (bra1.l + bra2.l + bra_n.l) % 2
            if !(e3min <= bra1.e + bra2.e + bra_n.e <= e3max) continue end
            iket_1_start = ifelse(ibra_n==iket_n, ibra_1, 1)
            for iket_1 in iket_1_start:n_qubits_p
                ket1 = msps_p[iket_1]
                if !(vemin <= ket1.e <= vemax) continue end
                for iket_2 in (iket_1+1):n_qubits_p
                    ket2 = msps_p[iket_2]
                    if !(vemin <= ket2.e <= vemax) continue end
                    if !(e3min <= ket1.e + ket2.e + ket_n.e <= e3max) continue end
                    if ibra_n == iket_n && ibra_1 == iket_1 && ibra_2 == iket_2
                        continue
                    end
                    Mket = ket1.jz + ket2.jz + ket_n.jz
                    Pket = (ket1.l + ket2.l + ket_n.l) % 2
                    if Pbra != Pket || Mbra != Mket
                        continue
                    end

                    i, j, k, l = sort((ibra_1, ibra_2, iket_1, iket_2))
                    tkey = pp_pattern = encode_4(i, j, k, l)
                                  
                    if !(haskey(res, tkey))
                        res[tkey] = Vector{NTuple{2, UInt64}}()
                    end
                    if !((pp_pattern, n_pattern) in res[tkey])  
                        push!(res[tkey], (pp_pattern, n_pattern))
                    end
                end
            end
        end
    end
    return nothing
end

function gen_ppn_tasks(msps_p, msps_n, vemin, vemax, e3min, e3max)
    # generate all combinations with diagonal neutron part
    # For ppn terms with diagonal neutron part,
    # we only need to put one Z on a neutron qubit.
    # Otherwise, those terms will be counted in pp part.
    # The position of the Z on neutron side are merely come from emax(e3max) truncations.
    n_qubits_n = length(msps_n)
    res_diag_n = Dict{UInt64, Vector{NTuple{2, UInt64}}}()
    res_nond_n = Dict{UInt64, Vector{NTuple{2, UInt64}}}()
    for ibra_n in 1:n_qubits_n
        iket_n = ibra_n
        loop_over_pp_in_ppn!(ibra_n, iket_n, msps_p, msps_n, vemin, vemax, e3min, e3max, res_diag_n)
    end
    # Then, generate nondiag neutron parts
    ibra_iket_pattern = [ (i, j) for i in 1:n_qubits_n-1 for j in (i+1):n_qubits_n ]
    plock = Threads.SpinLock()
    @threads for pair in ibra_iket_pattern
        ibra_n, iket_n = pair 
        dict_thread = Dict{UInt64, Vector{NTuple{2, UInt64}}}()
        loop_over_pp_in_ppn!(ibra_n, iket_n, msps_p, msps_n, vemin, vemax, e3min, e3max, dict_thread)
        lock(plock) do
            for (k, v) in dict_thread
                if !(haskey(res_nond_n, k))
                    res_nond_n[k] = Vector{NTuple{2, UInt64}}()
                end
                for item in v
                    if !(item in res_nond_n[k])
                        push!(res_nond_n[k], item)
                    end
                end
            end
        end
    end
    return res_diag_n, res_nond_n
end


function ncsm_model_spaces(emax, is_3NF) # function do nothing
    vemin = e3min = 0
    vemax = emax
    e3max = emax * 3
    return (emax, vemin, vemax, e3min, e3max, is_3NF)
end

function valence_model_spaces(label::String)
    emax = vemax = vemin = 0
    is_3NF = false
    if label == "p-shell"  || label == "p"
        emax = vemax = vemin = 1
    elseif label == "sd-shell" || label == "sd"
        emax = vemax = vemin = 2
    elseif label == "pf-shell" || label == "pf"
        emax = vemax = vemin = 3
    elseif label == "sdg-shell" || label == "sdg"
        emax = vemax = vemin = 4
    elseif label == "sd-pf-shell" || label == "sdpf"
        emax = vemax = 3; vemin = 2
    elseif label == "p-sd-shell" || label == "psd"
        emax = vemax = 2; vemin = 1
    elseif label == "sdpfsdg-shell" || label == "sdpfsdg"
        emax = vemax = 4; vemin = 2
    elseif label == "pf-sdg-shell" || label == "pfsdg"
        emax = vemax = 4; vemin = 3
    else
        error("Unknown valence model space label: $label")
    end
    e3min = e3max = emax * 3
    return (emax, vemin, vemax, e3min, e3max, is_3NF)
end

function encode_4(i::Int, j::Int, k::Int, l::Int)::UInt64
    return (UInt64(i) << 48) | (UInt64(j) << 32) | (UInt64(k) << 16) | UInt64(l)
end

function decode_4(hash_::UInt64)::NTuple{4,Int}
    i = Int((hash_ >> 48) & 0xFFFF)
    j = Int((hash_ >> 32) & 0xFFFF)
    k = Int((hash_ >> 16) & 0xFFFF)
    l = Int(hash_ & 0xFFFF)
    return (i, j, k, l)
end

function decode_2(hash_::UInt64)::NTuple{2,Int}
    i = Int((hash_ >> 16) & 0xFFFF)
    j = Int(hash_ & 0xFFFF)
    return (i, j)
end

function encode_2(i::Int, j::Int)::UInt64
    return (UInt64(i) << 16) | UInt64(j)
end

function encode_nondiag_UInt64_pp(i::Int, j::Int, k::Int, l::Int)
    # This is for pp patterns; if any of two are identical,
    # we still encode both indices, but the latter two will be used to avoid double counting.
    key = UInt64(0)
    if i == j
        key |= UInt64(k) << 16
        key |= UInt64(l) 
        return key
    end
    if j == k
        key |= UInt64(i) << 16
        key |= UInt64(l)
        return key
    end
    if k == l
        key |= UInt64(i) << 16
        key |= UInt64(j)
        return key
    end
    key |= UInt64(i) << 48
    key |= UInt64(j) << 32
    key |= UInt64(k) << 16
    key |= UInt64(l)
    return key
end

function encode_nondiag_UInt64(ibra_p_::Int, iket_p_::Int, ibra_n_::Int, iket_n_::Int)
    # Encode nondiagonal pattern into UInt64
    # from left to right (most significant to least significant):
    # ibra_p (in 16 bits), iket_p (in 16 bits), ibra_n (in 16 bits), iket_n (in 16 bits)
    # indices are already sorted in each species
    key = UInt64(0)
    if ibra_p_ != iket_p_
        key |= UInt64(ibra_p_) << 48
        key |= UInt64(iket_p_) << 32
    end
    if ibra_n_ != iket_n_
        key |= UInt64(ibra_n_) << 16
        key |= UInt64(iket_n_)
    end
    return key
end


function hash_int4(ibra_p::Int, iket_p::Int, ibra_n::Int, iket_n::Int)::UInt64
    return (UInt64(ibra_p) << 48) | (UInt64(iket_p) << 32) | (UInt64(ibra_n) << 16) | UInt64(iket_n)
end

function unhash_U_int4(hash_::UInt64)::NTuple{4,Int}
    ibra_p = Int((hash_ >> 48) & 0xFFFF)
    iket_p = Int((hash_ >> 32) & 0xFFFF)
    ibra_n = Int((hash_ >> 16) & 0xFFFF)
    iket_n = Int(hash_ & 0xFFFF)
    return (ibra_p, iket_p, ibra_n, iket_n)
end

function get_nond_idxs!(hash_::UInt64, nond_idxs::BitVector)
    nond_idxs .= false 
    for q in unhash_U_int4(hash_)
        if q != 0
            nond_idxs[q] = true
        end
    end
    return nothing
end

## NOTE: Legacy pn channel generation removed; use possible_2b_pn_terms_streaming! instead.


function possible_2b_pn_terms_streaming!(num_of_Hterms,
                                         msps_p, msps_n, to, io,
                                         Nchunk::Int,
                                         write_pauli::Bool;
                                         emin::Int=0, emax::Int=100,
                                         symmfactor::Int=1,
                                         pn_bucket_mod::Int=1)
    n_qubits_p = length(msps_p)
    n_qubits_n = length(msps_n)

    if pn_bucket_mod < 1
        error("pn_bucket_mod must be >= 1")
    end

    local_set = Set{PauliMask{Nchunk}}()
    num_of_pn_terms = 0

    for bucket_id in 0:(pn_bucket_mod - 1)
        seen = Set{UInt64}()
        for ibra_p in 1:n_qubits_p
            bra_p = msps_p[ibra_p]
            if !(emin <= bra_p.e <= emax) continue end
            for ibra_n in 1:n_qubits_n
                bra_n = msps_n[ibra_n]
                if !(emin <= bra_n.e <= emax) continue end
                Mbra = bra_p.jz + bra_n.jz
                Pbra = (bra_p.l + bra_n.l) % 2
                for iket_p in ibra_p:n_qubits_p
                    ket_p = msps_p[iket_p]
                    if !(emin <= ket_p.e <= emax) continue end
                    for iket_n in 1:n_qubits_n
                        ket_n = msps_n[iket_n]
                        if !(emin <= ket_n.e <= emax) continue end

                        if ibra_p == iket_p && ibra_n == iket_n
                            continue
                        end

                        Mket = ket_p.jz + ket_n.jz
                        Pket = (ket_p.l + ket_n.l) % 2
                        if Pbra != Pket || Mbra != Mket
                            continue
                        end
                        if !Jbracket_check(bra_p, bra_n, ket_p, ket_n, same_species=false)
                            continue
                        end

                        ibra_p_ = ibra_p
                        iket_p_ = iket_p
                        ibra_n_ = ibra_n
                        iket_n_ = iket_n

                        if ibra_p_ > iket_p_
                            ibra_p_, iket_p_ = iket_p_, ibra_p_
                        end
                        if ibra_n_ > iket_n_
                            ibra_n_, iket_n_ = iket_n_, ibra_n_
                        end

                        hash_ = hash_int4(ibra_p_, iket_p_, ibra_n_, iket_n_)
                        if Int(mod(hash_, UInt64(pn_bucket_mod))) != bucket_id
                            continue
                        end
                        if hash_ in seen
                            continue
                        end
                        push!(seen, hash_)

                        diagonal_p = ibra_p_ == iket_p_
                        diagonal_n = ibra_n_ == iket_n_
                        iter_p = diagonal_p ? ZZtuple : XY_patterns_ppn_2
                        iter_n = diagonal_n ? ZZtuple : XY_patterns_ppn_2

                        num_of_pn_terms += pn_loop_dev!(local_set, iter_p, iter_n,
                                                       ibra_p_, iket_p_, ibra_n_, iket_n_,
                                                       n_qubits_p, n_qubits_n, write_pauli, io)
                    end
                end
            end
        end
        empty!(seen)
    end

    num_of_Hterms += num_of_pn_terms * symmfactor
    return num_of_Hterms, num_of_pn_terms
end


function possible_3b_terms_parallel(Measurement_Basis,
                                    Nchunk::Int,
                                    vect_Tops::Vector{BigInt}, num_of_Hterms::BigInt,
                                    Z::Int, N::Int, msps_p, msps_n, to, io,
                                    write_pauli::Bool,
                                    debug_mode::Int;
                                    emin::Int=0, emax::Int=100,
                                    e3min::Int=0, e3max::Int=20,
                                    symfactor::Int=1)
    n_qubits_p = length(msps_p)
    n_qubits_n = length(msps_n)
    n_qubits = n_qubits_p + n_qubits_n

    if Z >= 3
        println("Processing ppp channel ... ")
        args = generate_ppp_args(msps_p, emin, emax, e3min, e3max, n_qubits)
        println(" Total #ppp tasks: $(length(args))")
        plock = Threads.SpinLock()
        argkeys = collect(keys(args))
        num_of_T3terms = BigInt(0)
        @timeit to "work" @threads for i in 1:length(argkeys)
            nond_idxs = argkeys[i]
            local_set = local_set_per_thread()
            empty!(local_set)
            vector_diag_idxs = args[nond_idxs]
            
            zmask = BitVector(falses(n_qubits))
            update_zmask!(n_qubits, nond_idxs, zmask)    
            
            pauli_patterns = XY_patterns_T3[length(nond_idxs)]
            num_terms = _process_triplet_terms!(local_set, pauli_patterns, Nchunk,
                                                nond_idxs, zmask, vector_diag_idxs,
                                                n_qubits_p, n_qubits, write_pauli)
            lock(plock) do 
                # update_MeasBasisGroup!(Measurement_Basis, local_set, n_qubits)
                # Tcounts_add_sofar!(local_set, vect_Tops; symmfactor=2)
                num_of_Hterms += num_terms * symfactor
                num_of_T3terms += num_terms
                if io != stdout
                    write_pauli_terms(io, local_set; label="ppp: ")
                end
            end
        end
        println("ppp done. #unique pauli terms = $(num_of_T3terms)")
    end
    return num_of_Hterms
end

function XY_loop_in_ppn!(local_arr::Vector{String}, 
                         bitlist_p::Vector{Char}, ibra_1::Int, iket_1::Int, ibra_2::Int, iket_2::Int,
                         n_qubits_p::Int, nondiag_n::Bool, 
                         n_str_pool::Vector{String}, 
                         XY_patterns_::NTuple{N, NTuple{4,Char}}) where {N}
    for (xy1, xy2, xy3, xy4) in XY_patterns_
        bitlist_p .= 'I'
        bitlist_p[ibra_1] = xy1
        bitlist_p[iket_1] = xy2
        bitlist_p[ibra_2] = xy3
        bitlist_p[iket_2] = xy4
        Xp = count(x->x=='X', bitlist_p)
        Yp = count(x->x=='Y', bitlist_p)
        if !nondiag_n && (Xp % 2 == 1 || Yp % 2 == 1)
            exit()
            continue
        end
        padd_Z_from_JW!(bitlist_p, ibra_1, iket_1, ibra_2, iket_2, n_qubits_p)
        p_str = String( @view bitlist_p[end:-1:1])
        for n_str in n_str_pool
            Xn = count(x->x=='X', n_str)
            if (Xp + Xn) % 2 == 1
                continue
            end
            Yn = count(x->x=='Y', n_str)
            if (Yp + Yn) % 2 == 1
                continue
            end
            push!(local_arr, n_str * p_str)
        end
    end
end

function nloop_in_ppn(n_idxs::NTuple{2, Int}, n_XY_patterns::Vector{Tuple{Char, Char}}, 
                      count_Xp::Int, count_Yp::Int,
                      n_qubits_p::Int, n_qubits_n::Int,
                      xp::MVector{Nchunk, UInt64}, zp::MVector{Nchunk, UInt64},
                      xn::MVector{Nchunk, UInt64}, zn::MVector{Nchunk, UInt64},
                      zmask::BitVector, write_pauli::Bool,
                      to::TimerOutput
    ) where {Nchunk}
    num_local_terms = 0
    nondiag_n = n_idxs[1] != n_idxs[2]

    if !nondiag_n # neutron is diagonal
        if write_pauli
            set_pauli!(xn, zn, n_idxs[1], UInt8('Z'), n_qubits_p)
        end
        count_Xn = count_Yn = 0
        if (count_Xp + count_Xn) % 2 != 0 && (count_Yp + count_Yn) % 2 != 0
            return 0 
        end
        num_local_terms += 1
    else
        zmask .= false
        update_zmask!(n_qubits_n, n_idxs, zmask)
        for pauli_tuple_n in n_XY_patterns
            count_Xn, count_Yn = count_XY(pauli_tuple_n[1], pauli_tuple_n[2])

            if (count_Xp + count_Xn) % 2 != 0 && (count_Yp + count_Yn) % 2 != 0
                continue
            end
            num_local_terms += 1

            if write_pauli
                fill!(xn, UInt64(0))
                fill!(zn, UInt64(0))
                set_pauli!(xn, zn, n_idxs[1], UInt8(pauli_tuple_n[1]), n_qubits_p)
                set_pauli!(xn, zn, n_idxs[2], UInt8(pauli_tuple_n[2]), n_qubits_p)
                for q in 1:n_qubits_n
                    if zmask_n[q]
                        set_pauli!(xn, zn, q, UInt8('Z'), n_qubits_p)
                    end
                end
            end
        end
    end
    return num_local_terms
end

"""
v is diag.: diag part should be Z for neutron, otherwise, this should have been counted in pp part.
v is nond.: X/Y for n determines the unique channels.

In both cases, we need to consider patterns for proton part.
If the index corresponding to diagonal (ani. and cre.) term, those qubits should have Z in the final pattern.
This is because we are considering unique patterns not counted in proton-neutron part.

"""
function _worker_job_ppn!(local_arr::Vector{String}, XY_patterns_4::Vector{NTuple{4, Char}},
                          pp_idxs::NTuple{4,Int}, n_idxs::NTuple{2,Int},
                          xp::MVector{Nchunk, UInt64}, zp::MVector{Nchunk, UInt64}, 
                          xn::MVector{Nchunk, UInt64}, zn::MVector{Nchunk, UInt64},
                          zmask::BitVector,
                          n_qubits_p::Int, n_qubits_n::Int, to::TimerOutput,
                          write_pauli::Bool=false)::Int where {Nchunk}
    num_local_terms = 0
  
    n_qubits = n_qubits_p + n_qubits_n
    nondiag_n = n_idxs[1] != n_idxs[2]
    n_XY_patterns = XY_patterns_ppn_2
    zmask_n = zmask

    if pp_idxs[1] != pp_idxs[2] && pp_idxs[2] != pp_idxs[3] && pp_idxs[3] != pp_idxs[4]
        # protons are all non-diagonal -> neutron can be either diag. or non-diag.
        XY_patterns_ = nondiag_n ? XY_patterns_ppn_4 : XY_patterns_T1_4
        zmask .= false
        update_zmask!(n_qubits_p, pp_idxs, zmask)

        for pattern_p in XY_patterns_
            if write_pauli
                fill!(xp, UInt64(0))
                fill!(zp, UInt64(0))
                for q in eachindex(pp_idxs)
                    set_pauli!(xp, zp, pp_idxs[q], UInt8(pattern_p[q]))
                end
                for q in 1:n_qubits_p
                    if zmask[q]
                        set_pauli!(xp, zp, q, UInt8('Z'))
                    end
                end
            end

            count_1 = count_XY(pattern_p[1], pattern_p[2])
            count_2 = count_XY(pattern_p[3], pattern_p[4])
            count_Xp = count_1[1] + count_2[1]
            count_Yp = count_1[2] + count_2[2]

            num_local_terms += nloop_in_ppn(n_idxs, n_XY_patterns, count_Xp, count_Yp, 
                                            n_qubits_p, n_qubits_n, xp, zp, xn, zn, zmask, write_pauli, to)

        end
    elseif pp_idxs[1] == pp_idxs[2] && pp_idxs[3] == pp_idxs[4] # protons are diagonal (neutron should be non-diag.)
        @assert nondiag_n "Expected non-diagonal neutron part for diagonal proton part."
        if write_pauli
            set_pauli!(xp, zp, pp_idxs[1], UInt8('Z'))
            set_pauli!(xp, zp, pp_idxs[2], UInt8('Z'))
        end
        num_local_terms += nloop_in_ppn(n_idxs, n_XY_patterns, 0, 0,
                                        n_qubits_p, n_qubits_n, xp, zp, xn, zn, zmask, write_pauli, to)
    else
        diag_idx = nond_idx1 = nond_idx2 = 0
        if pp_idxs[1] == pp_idxs[2]
            diag_idx = pp_idxs[1]
            nond_idx1 = pp_idxs[3]
            nond_idx2 = pp_idxs[4]
        elseif pp_idxs[2] == pp_idxs[3]
            diag_idx = pp_idxs[2]
            nond_idx1 = pp_idxs[1]
            nond_idx2 = pp_idxs[4]
        elseif pp_idxs[3] == pp_idxs[4]
            diag_idx = pp_idxs[3]
            nond_idx1 = pp_idxs[1]
            nond_idx2 = pp_idxs[2]
        else
            @error "Unexpected case in mixed diagonal proton part."
        end

        if write_pauli
            zmask .= false
            zmask[nond_idx1+1:nond_idx2-1] .= true
        end

        for pattern_2 in XY_patterns_ppn_2
            if write_pauli
                fill!(xp, UInt64(0))
                fill!(zp, UInt64(0))
                set_pauli!(xp, zp, nond_idx1, UInt8(pattern_2[1]))
                set_pauli!(xp, zp, nond_idx2, UInt8(pattern_2[2]))     
                for q in nond_idx1+1:nond_idx2-1
                    if zmask[q]
                        set_pauli!(xp, zp, q, UInt8('Z'))
                    end
                end      
                if zmask[diag_idx]
                    set_pauli!(xp, zp, diag_idx, UInt8('I'))
                else
                    set_pauli!(xp, zp, diag_idx, UInt8('Z'))
                end
            end
            count_Xp, count_Yp = count_XY(pattern_2[1], pattern_2[2])

            num_local_terms += nloop_in_ppn(n_idxs, n_XY_patterns, count_Xp, count_Yp, 
                                            n_qubits_p, n_qubits_n, xp, zp, xn, zn, zmask, write_pauli, to)
        end
    end
    return num_local_terms
end

function padd_Z_from_JW!(bitlist_p::Vector{Char},
                         ibra_1::Int, iket_1::Int,
                         ibra_2::Int, iket_2::Int,
                         n_qubits_p::Int)
    @inbounds for q in 1:(n_qubits_p-1)
        if q == ibra_1 || q == iket_1 || q == ibra_2 || q == iket_2
            continue
        end
        cnt = (ibra_1 > q) + (iket_1 > q) + (ibra_2 > q) + (iket_2 > q)
        if isodd(cnt)
            bitlist_p[q] = 'Z'
        end
    end
    return nothing
end

function count_XYZ(bitstr; split=true)
    n_qubits = length(bitstr)
    if split
        n_qubits_n = div(length(bitstr), 2)
        n_str = bitstr[1:n_qubits_n]
        p_str = bitstr[n_qubits_n+1:end]
        count_Xp = count(ch->ch=='X', p_str)
        count_Yp = count(ch->ch=='Y', p_str)
        count_Zp = count(ch->ch=='Z', p_str)
        count_Xn = count(ch->ch=='X', n_str)
        count_Yn = count(ch->ch=='Y', n_str)
        count_Zn = count(ch->ch=='Z', n_str)
        return count_Xp, count_Yp, count_Zp, count_Xn, count_Yn, count_Zn
    else
        n_qubits = length(bitstr)
        count_X = count(ch->ch=='X', bitstr)
        count_Y = count(ch->ch=='Y', bitstr)
        count_Z = count(ch->ch=='Z', bitstr)
        return count_X, count_Y, count_Z
    end
end

function select_io(write_pauli::Bool, no_core::Bool,
                   emax::Int, e3max::Int,
                   valence_space::String)
    io = if write_pauli
        if no_core
            open("encoded_Hamil_nW_e$(emax)_$(emax*2)_$(e3max).txt", "w")
        else
            if isempty(valence_space)
                error("For valence systems, please specify valence_space label.")
            end
            open("encoded_Hamil_nW_$(valence_space).txt", "w")
        end
    else
        stdout
    end
    return io
end


function encode_pauli(::Val{N}, term::String) where {N}
    x = ntuple(_->UInt64(0), N)
    z = ntuple(_->UInt64(0), N)

    @inbounds for (i,c) in enumerate(codeunits(term))
        chunk = (i-1) >>> 6 + 1
        bit   = UInt64(1) << ((i-1) & 63)
        if c == UInt8('X')
            x = Base.setindex(x, x[chunk] | bit, chunk)
        elseif c == UInt8('Y')
            x = Base.setindex(x, x[chunk] | bit, chunk)
            z = Base.setindex(z, z[chunk] | bit, chunk)
        elseif c == UInt8('Z')
            z = Base.setindex(z, z[chunk] | bit, chunk)
        end
    end
    return PauliMask{N}(x,z)
end

function decode_pauli_string(
    x::MVector{N,UInt64},
    z::MVector{N,UInt64},
    nqubits::Int,
    reversed::Bool=true
) where {N}
    @assert nqubits ≤ 64N

    chars = Vector{UInt8}(undef, nqubits)

    @inbounds for i in 1:nqubits
        chunk = (i - 1) >>> 6 + 1
        bit   = UInt64(1) << ((i - 1) & 63)

        x_bit = (x[chunk] & bit) != 0
        z_bit = (z[chunk] & bit) != 0

        chars[i] = x_bit ? (z_bit ? UInt8('Y') : UInt8('X')) :
                           (z_bit ? UInt8('Z') : UInt8('I'))
    end
    if reversed
        return String( @view chars[end:-1:1] )
    end
    return String(chars)

end

@inline function decode_pauli_string(
    gx::NTuple{N,UInt64},
    gz::NTuple{N,UInt64},
    nqubits::Int,
    reversed::Bool=true
) where {N}
    @assert nqubits ≤ 64N

    chars = Vector{UInt8}(undef, nqubits)

    @inbounds for i in 1:nqubits
        chunk = (i - 1) >>> 6 + 1
        bit   = UInt64(1) << ((i - 1) & 63)

        x = (gx[chunk] & bit) != 0
        z = (gz[chunk] & bit) != 0

        chars[i] = x ? (z ? UInt8('Y') : UInt8('X')) :
                       (z ? UInt8('Z') : UInt8('I'))
    end
    if reversed
        return String( @view chars[end:-1:1] )
    end
    return String(chars)
end


"""
If two Pauli terms are identical or either has I, they are qubit-wise compatible.
I: 0-0 X: 1-0 Y: 1-1 Z: 0-1
The only incompatible combinations are:
X/Y/Z vs I, X vs X, Y vs Y, Z vs Z
"""
@inline function qwc_compatible(x_p::NTuple{N,UInt64},
                        z_p::NTuple{N,UInt64},
                        x_g::NTuple{N,UInt64},
                        z_g::NTuple{N,UInt64}) where {N}
    @inbounds @simd for i in 1:N
        xp = x_p[i]; zp = z_p[i]
        xg = x_g[i]; zg = z_g[i]
        both_nonI = (xp | zp) & (xg | zg)
        if ((xor(xp, xg) | xor(zp, zg)) & both_nonI) != 0
            return false
        end
    end
    return true
end

"""
The function to check whether the measurement basis group gA can be measured
simultaneously with gB.
"""
function is_measurable_A_by_B(gA::MeasBasisGroup{N},
                              gB::MeasBasisGroup{N}) where {N}
    @inbounds @simd for i in 1:N
        if (gA.gx[i] & gB.gz[i]) != 0 || (gA.gz[i] & gB.gx[i]) != 0
            return false
        end
    end
    return true
end

@inline function add_to_group!(g::MeasBasisGroup{N},
                               p::PauliMask{N}) where {N}
    g.gx = ntuple(i -> g.gx[i] | p.x[i], N)
    g.gz = ntuple(i -> g.gz[i] | p.z[i], N)
    push!(g.terms, p)
    return nothing
end

function compute_tkey_and_nonIcount(term::String)
    count_Xp, count_Yp, count_Zp, count_Xn, count_Yn, count_Zn = count_XYZ(term)
    return (count_Xp, count_Yp, count_Xn, count_Yn), count_Xp + count_Yp + count_Zp + count_Xn + count_Yn + count_Zn
end

@inline function update_MeasBasisGroup!(
    Measurement_Basis::Dict{K,Vector{MeasBasisGroup{N}}},
    terms::Set{String},
    n_qubits::Int
) where {K,N}
    for term in terms
        p = encode_pauli(Val(N), term)
        tkey, nonIcount = compute_tkey_and_nonIcount(term)
        heavy_Pauli = (2 *nonIcount > n_qubits)
        groups = get!(Measurement_Basis, tkey) do
            Vector{MeasBasisGroup{N}}()
        end

        placed = false
        if !heavy_Pauli          
            @inbounds for g in groups
                xp = p.x
                zp = p.z
                xg = g.gx
                zg = g.gz
                if qwc_compatible(xp, zp, xg, zg)
                    add_to_group!(g, p)
                    placed = true
                    break
                end
            end
        end
        if !placed
            push!(groups, MeasBasisGroup{N}(p.x, p.z, [p]))
        end
    end
    return nothing
end


"""
In the current implementation, Measurement_Basis dict may contain
groups that can be measured simultaneously.
This originates from the sequence of terms processed in update_MeasBasisGroup! function.
For example, let us suppose that the first term appeared in `update_MeasBasisGroup!` was 
"ZIXX", and the second term was "ZZXX".
When processing the second term, since it is not measurable by the measurement basis for
the first term, a new group will be created.
However, if the order of terms were reversed, the measurement basis for "ZZXX" could
have accommodated "ZIXX" as well.
To resolve this, we need to reorder the groups in Measurement_Basis such that
groups that can be measured simultaneously are merged together as much as possible
with a compromise of computational cost.

The strategy is...

1. For each key in Measurement_Basis, take some groups with heavier non-I Paulis as reference groups.
2. For each of the remaining groups, check if it can be merged into any of the reference groups.
3. If it can be merged, do so. If not, keep it as a separate group.
4. Repeat until all groups are processed once.
"""
function ordering_MeasBasisGroup!(
    Measurement_Basis::Dict{K,Vector{MeasBasisGroup{N}}},
    n_qubits::Int
) where {K,N}
    for tkey in keys(Measurement_Basis)
        groups = Measurement_Basis[tkey]
        # Sort groups by the number of non-I Paulis in descending order
        # g has gx, gz, terms fields, so we can compute nonI as gx | gz
        sorted_groups = sort(
            groups,
            by = g -> -count_ones(reduce(|, g.gx) | reduce(|, g.gz))
        )
        num_g = length(sorted_groups)
        num_ref = min(ceil(Int, 2 * num_g/3), num_g)
        reference_group = @view sorted_groups[1:num_ref]
        new_groups = Vector{MeasBasisGroup{N}}()
        println("group merging for tkey = $tkey")
        for ref_g in reference_group
            println("  ref_g ", decode_pauli_string(ref_g.gx, ref_g.gz, n_qubits))
        end
        println("sample term: ", decode_pauli_string(sorted_groups[end].gx, sorted_groups[end].gz, n_qubits))
        for i in num_ref+1:num_g
            g = sorted_groups[i]
            merged = false
            for ref_g in reference_group
                if is_measurable_A_by_B(g, ref_g)
                    for p in g.terms
                        add_to_group!(ref_g, p)
                    end
                    merged = true
                    println("any merged?")
                    break
                end
            end
            if !merged
                push!(new_groups, g)
            end
        end
        # Update Measurement_Basis with the merged groups
        Measurement_Basis[tkey] = vcat(reference_group, new_groups) 
        println("For tkey = $tkey, reduced # of groups from $num_g to $(length(Measurement_Basis[tkey]))")
    end
    return nothing
end


"""
Function to mimic NN terms by PP terms in Measurement_Basis.
It can also be used to mimic NNN by PPP terms.

This function is only for debug since nn (nnn) terms can be 
simultaneously measured by corresponding pp (ppp) terms.
"""
function mimic_nn_by_pp!(Measurement_Basis, n_qubits, debug_mode, to)
    if debug_mode >= 2
        println("Since debug_mode set >=2, we are preparing MeasBasis for nn(nnn) terms",
                " which is to be redundant")
    else
        return nothing
    end
    term_missed = Dict{NTuple{4, Int}, Vector{String}}()
    for XYkey in keys(Measurement_Basis)
        groups = Measurement_Basis[XYkey]
        for g in groups
            for term in g.terms
                bitstr = decode_pauli_string(term.x, term.z, n_qubits)
                Xp, Yp, Zp, Xn, Yn, Zn = count_XYZ(bitstr)
                if Xn + Yn + Zn == 0 && Xp + Yp > 0
                    nn_key = (Xn, Yn, Xp, Yp)
                    if !haskey(term_missed, nn_key)
                        term_missed[nn_key] = String[ ]
                    end
                    n_str = bitstr[1:div(end,2)]
                    p_str = bitstr[div(end,2)+1:end]
                    bitstr = p_str * n_str
                    push!(term_missed[nn_key], bitstr) 
                end
            end
        end
    end
    println("term_missed ")
    num = idx = 0
    for tkey in keys(term_missed)        
        idx += 1
        terms = term_missed[tkey]
        num += length(terms) 
        update_MeasBasisGroup!(Measurement_Basis, term_missed[tkey], n_qubits)
    end
    println("# of term missed (nn+nnn) = $num")
    return nothing
end


"""
Julia code to estimate T-gate counts for nuclear shell model Hamiltonians.

This code generates all possible Pauli term patterns arising from
Jordan-Wigner transformed nuclear shell model Hamiltonians,
including NN (either valence or no-core) and 3NF interactions.

One can specify emax (maximum single-particle energy level), vemax (maximum valence energy level),
and whether to include 3NF terms (is_3NF).

Key assumptions behind our T-count estimates:
- Each unique Pauli term is counted once toward the total T-count,
  even if it appears multiple times in the Hamiltonian with different coefficients.
- Exponential of each Pauli term is implemented by the ordinary method:
  basis change (single-qubit unitaries) + Rz + basis change back.
  If you have terms like X_0 Y_2 X_5, you will get 
  basis change (on qubits 0,2,5) + Rz + basis change back + CX chains (6 in total).
- Single Trotter step is assumed for the entire Hamiltonian.
- Any compilation optimizations (e.g., term cancellations across Trotter steps)
  are not considered in this estimation.
- "Symmetric modelspaces" are assumed, i.e., proton and neutron modelspaces
  have the same emax/vemax settings.
- Regarding T-gate controlled Unitary operations:
    - Single Toffoli gate decomposed into 4 T-gates.
    - Ry gate decomposed into 100 T-gates. This could be further reduced, but we assume
      angles are arbitrary here.
- Note that a small portion of terms may be double-counted in this estimation,
  such as ones cancelling out exactly due to symmetry.

## Augments
- `debug_mode`: 0=false, 1=> storing all terms to be measured by the MeasBasisGroup
"""
function main(debug_mode::Int = 0)

    to = TimerOutput()
    
    ## Parameterscd
    emax_specified = 10

    is_3NF = !true
    count_Ppattern = !true
    
    ## If you want to write out Pauli strings set write_pauli = true
    write_pauli = !true
    if write_pauli && ((is_3NF && emax_specified > 2) || emax_specified >= 3)
        error("Currently, write_pauli option with emax > 2 for 3NF and emax > 3 for 2NF are not recommended")
        write_pauli = false
    end

    # Since the number of T-gates could be counted by the number of encoded Pauli terms,
    # one usually does not need to count the gate-counts during the process.
    # However, if you want to debug the counting process, one can choose to count gates
    # during the process by setting count_gates_during_process = true.
    count_gates_during_process = false
    Tgate_per_Rz = 100 # T-gate count per Rz gate ~ 1.e-10 tolerance
            
    ## Specify nucleus. This is only used to limit interaction channels.
    ## You should be aware of nn, nnp, and nnn part are not explicitly considered in this code,
    ## If you want to nn, nnp, nnn channels, please evaluate them by counting pp/ppn/ppp and then read them as nn, nnp, nnn.
    ## Do not specify like Z, N = (1, 2) to count "nn + pn + pnn". Try (2,1) instead.
    Z, N = (3, 3)    
    #Z, N = (1, 1)
    #Z, N = (2, 1)
    #Z, N = (2, 1) 
    valence_space = ""  # e.g., "p", "sd", "pf", "sdpf", etc.; empty string for no-core
    
    ## Define model space parameters
    no_core = ifelse(valence_space=="", true, false)
    emax, vemin, vemax, e3min, e3max = ncsm_model_spaces(emax_specified, is_3NF)
    if !no_core
        emax, vemin, vemax, e3min, e3max, is_3NF = valence_model_spaces(valence_space)
    end
    io = select_io(write_pauli, no_core, emax, e3max, valence_space)

    #e3max = emax_specified * 2

    ## Defining model space
    msps_p, msps_n = count_msps(emax, vemin, vemax)
    n_qubits_p = length(msps_p)
    n_qubits_n = length(msps_n)
    n_qubits = n_qubits_p + n_qubits_n
    Nchunk = cld(n_qubits, 64)


    println("=== emax=$(emax), e3max=$(e3max) Nq=$(n_qubits) ($(n_qubits_p), $(n_qubits_n))===")
    vect_Tops = zeros(BigInt, 6)
    num_of_Hterms = BigInt(0)

    # Memory-efficient representation for measurement-basis patterns:
    Measurement_Basis = Dict{NTuple{4, Int}, Vector{MeasBasisGroup{Nchunk}}}()

    ## ==== diagonal (I, Z,...) contributions from any channals =====
    diag_terms = generate_diag_terms(msps_p, msps_n, emax, e3max, Z, N, is_3NF)
    all_diag = Set{String}()
    for k in keys(diag_terms)
        union!(all_diag, diag_terms[k])
    end
    if count_Ppattern
        update_MeasBasisGroup!(Measurement_Basis, all_diag, n_qubits)
        println("group after diag terms: ", length(Measurement_Basis), " measurement basis groups")
    end    
    write_pauli_terms(io, all_diag; label="diag: ")
    if count_gates_during_process
        Tcounts_add_sofar!(all_diag, vect_Tops)
    end
    num_of_Hterms += sum(length(diag_terms[k]) for k in keys(diag_terms))
    println("[diag]: # of terms (total): $num_of_Hterms")

    ## ==== 2NF terms pp / nn =====
    if Z >= 2 || N >= 2
        @timeit to "2NF pp/nn terms" begin
            symfactor_2NF = ifelse(Z >= 2 && N >=2, 2, 1)
            num_of_Hterms = possible_2b_T1_terms!(Measurement_Basis, vect_Tops, num_of_Hterms, 
                                                  msps_p, msps_n, to, io, write_pauli,
                                                  Nchunk, is_3NF, count_Ppattern,
                                                  count_gates_during_process, debug_mode; 
                                                  emin=vemin, emax=vemax, symmfactor=symfactor_2NF)
        end
    end

    ## ==== 2NF terms pn =====
    if Z >=1 && N >=1
        @timeit to "2NF pn terms" begin
            println("Processing pn (streaming) ...")
            num_of_Hterms, num_of_pn_terms = possible_2b_pn_terms_streaming!(
                num_of_Hterms,
                msps_p, msps_n, to, io, Nchunk, write_pauli;
                emin=vemin, emax=vemax
            )
            println("[diag+NN]: pn terms $(num_of_pn_terms), # of terms (total): $num_of_Hterms")
        end
    end

    ## ==== 3NF terms ppn / pnn =====
    if is_3NF && Z >= 2 && N >= 1
        @timeit to "3NF (ppn)" begin
            symfactor_ppn = ifelse(Z >= 2 && N >=2, 2, 1)
            @timeit to "gen.args" res_diag_n, res_nond_n = gen_ppn_tasks(msps_p, msps_n, vemin, vemax, e3min, e3max)
            plock = ReentrantLock()
            println("---\n\nProcessing ppn channel... tasks: $(length(res_diag_n)+length(res_nond_n)) ($(length(res_diag_n)) diag + $(length(res_nond_n)) nondiag)")
            num_of_Hterms_ppn = 0
            #@timeit to "diag_n" @threads for task in collect(keys(res_diag_n))

            xp = MVector{Nchunk, UInt64}(zeros(UInt64, Nchunk))
            zp = MVector{Nchunk, UInt64}(zeros(UInt64, Nchunk))
            xn = MVector{Nchunk, UInt64}(zeros(UInt64, Nchunk))
            zn = MVector{Nchunk, UInt64}(zeros(UInt64, Nchunk))

            @timeit to "diag_n" for task in collect(keys(res_diag_n))
                local_arr = String[ ]
                zmask = BitVector(falses(n_qubits_n))
                num_local_terms = 0
                for (pp_pattern, n_pattern) in res_diag_n[task]    
                    pp_idxs = decode_4(pp_pattern)
                    n_idxs = decode_2(n_pattern)
                    XY_pattern_4 = ifelse(n_idxs[1] == n_idxs[2], XY_patterns_ppn_4, XY_patterns_T1_4)
                    num_local_terms += _worker_job_ppn!(local_arr, XY_pattern_4, pp_idxs, n_idxs, xp, zp, xn, zn, zmask,
                                                        n_qubits_p, n_qubits_n, to, write_pauli)
                end
                if num_local_terms == 0
                    continue
                end
                num_of_Hterms_ppn += num_local_terms
                # lock(plock) do
                #     if count_Ppattern
                #         update_MeasBasisGroup!(Measurement_Basis, local_set, n_qubits)
                #     end
                #     if count_gates_during_process
                #         vect_Tops .+= local_vect
                #     end
                #     if io != stdout
                #         write_pauli_terms(io, local_set; label="ppn: ")
                #     end
                # end
            end

            #@timeit to "nondiag_n" @threads for task in collect(keys(res_nond_n))
            @timeit to "nondiag_n" for task in collect(keys(res_nond_n))
                local_arr = String[ ]
                zmask = BitVector(falses(n_qubits_n))
                num_local_terms = 0
                for (pp_pattern, n_pattern) in res_nond_n[task]
                    pp_idxs = decode_4(pp_pattern)
                    n_idxs = decode_2(n_pattern)
                    XY_pattern_4 = ifelse(n_idxs[1] == n_idxs[2], XY_patterns_ppn_4, XY_patterns_T1_4)
                    num_local_terms += _worker_job_ppn!(local_arr, XY_pattern_4, pp_idxs, n_idxs, xp, zp, xn, zn, 
                                                        zmask, n_qubits_p, n_qubits_n, to, write_pauli)
                end
                num_of_Hterms_ppn += num_local_terms
                # lock(plock) do
                #     vect_Tops .+= local_vect
                #     if count_Ppattern
                #         update_MeasBasisGroup!(Measurement_Basis, local_set, n_qubits)
                #     end
                #     if io != stdout
                #         write_pauli_terms(io, local_set; label="ppn: ")
                #     end
                # end
            end
            num_of_Hterms += num_of_Hterms_ppn * symfactor_ppn
            println("[diag+NN+ppn]: # of ppn terms $(num_of_Hterms_ppn), # of terms (total): $num_of_Hterms")
            println("----\n")
        end
    end

    ## ==== 3NF terms ppp / nnn =====
    if is_3NF && (Z >= 3 || N >= 3)
        symfactor = ifelse(Z>=3 && N>=3, 2, 1)
        @timeit to "3NF (ppp)" begin
            num_of_Hterms_total = possible_3b_terms_parallel(Measurement_Basis, Nchunk, vect_Tops, num_of_Hterms,
                                                            Z, N, msps_p, msps_n, to, io, write_pauli, debug_mode;
                                                            emin=vemin, emax=vemax,
                                                            e3min=e3min, e3max=e3max, symfactor=symfactor)
            println("[NN+3NF]:", vect_Tops, " # of terms (ppp): $(div(num_of_Hterms_total-num_of_Hterms, symfactor)), total: $num_of_Hterms_total")
            num_of_Hterms = num_of_Hterms_total
        end
    end


    dict_Tops = Dict{String,BigInt}("u"=>vect_Tops[1], "cx"=>vect_Tops[2], "T-estimate"=>vect_Tops[3])
    if !count_gates_during_process
        dict_Tops["T-estimate"] = (num_of_Hterms-1)*2 * Tgate_per_Rz
    end

    println("=== T-gate estimates for various algorithms ===")
    single_T = get(dict_Tops, "T-estimate", 0)
    Nq = n_qubits
    Na = 10
    fac = 2^Na - 1
    println(@sprintf("T-estimate (QPE; Nq=%d, Na=%d):  %8.1e", Nq, Na, fac * single_T))

    Niter = 50
    Ncirq = num_of_Hterms
    fac2 = Ncirq * Niter * (Niter^2 + Niter)
    println(@sprintf("T-estimate (QKrylov; Niter=%d, Ncirq=%d):  %8.1e", Niter, Ncirq, big(fac2) * single_T))

    Niter = 50
    fac3 = Niter * (Niter + 1)
    println(@sprintf("T-estimate (ODMD; Niter=%d):  %8.1e", Niter, fac3 * single_T))
    print("Nh: ", num_of_Hterms, "   ")

     if count_Ppattern
        if N >= 2 
            mimic_nn_by_pp!(Measurement_Basis, n_qubits, debug_mode, to)
        end
        if Z >= 2 && N >= 1 && is_3NF
            @timeit to "mimic_pnn" mimic_pnn_by_ppn!(Measurement_Basis, n_qubits_p, n_qubits_n, debug_mode, to)
        end
        numsum = num_d = num_T1 = num_pn = num_g = 0
        for XYcounts in keys(Measurement_Basis)
            n_term = sum( [ length(g.terms) for g in Measurement_Basis[XYcounts]])
            println("XYcounts: ", XYcounts, 
                    " -> # of group=$(@sprintf("%9i", length(Measurement_Basis[XYcounts])))  ",
                    " # of terms=", @sprintf("%9i", n_term))                    
            num_g += length(Measurement_Basis[XYcounts])
            for group in Measurement_Basis[XYcounts]
                #println("   MeasBasis: ", decode_pauli_string(group.gx, group.gz, n_qubits),
                #        "  #terms=", @sprintf("%9i", length(group.terms)) )
                n_terms = length(group.terms)
                numsum += n_terms
                tmp = String[ ]
                for t in group.terms
                    bitstr = decode_pauli_string(t.x, t.z, n_qubits)
                    push!(tmp, bitstr)
                    #measureable(group, t) 
                end               
                for t in tmp
                    count_Xp, count_Yp, count_Zp, count_Xn, count_Yn, count_Zn = count_XYZ(t; split=true)
                    if count_Xn + count_Yn + count_Xp + count_Yp == 0
                        num_d += 1
                    elseif (count_Xp + count_Yp + count_Zp) > 0 && (count_Xn + count_Yn + count_Zn) == 0
                        num_T1 += 1
                    elseif (count_Xn + count_Yn + count_Zn) > 0 && (count_Xp + count_Yp + count_Zp) == 0
                        num_T1 += 1
                    else
                        num_pn += 1
                    end
                end
            end
        end
        println("Measurement basis groups $num_g groups. Total terms: $numsum")
        println("  of which diag: $num_d, T1-like: $num_T1, pn-like: $num_pn")
    end

    println("\n\n")
    show(to);print("\n")

end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
    #@profilehtml main()
end

