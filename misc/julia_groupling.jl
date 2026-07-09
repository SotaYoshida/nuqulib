using Random
Random.seed!(1234)

function qubitwise_commute(a::AbstractString, b::AbstractString)
    for (ca, cb) in zip(codeunits(a), codeunits(b))
        if ca != UInt8('I') && cb != UInt8('I') && ca != cb
            return false
        end
    end
    return true
end

function decode_term(term::Tuple{BitVector, BitVector})
    x, z = term
    n_qubits = length(x)
    text = ""
    for i in 1:n_qubits
        if !x[i] && !z[i]
            text *= "I"
        elseif x[i] && !z[i]
            text *= "X"
        elseif x[i] && z[i]
            text *= "Y"
        elseif !x[i] && z[i]
            text *= "Z"
        end
    end
    return text
end

function main(sntf; verbose=false)
    
    filename = joinpath(@__DIR__, "encoded_Hamil", "encoded_$(sntf).txt")
    lines = readlines(filename)
    terms = [split(l)[1] for l in lines if !isempty(strip(l)) && !startswith(strip(l), "#")]
    unq_terms = Set{String}()
    for term in terms
        push!(unq_terms, term)
    end
    println("Loaded terms: ", length(terms))
    # random shuffle to avoid any ordering bias
    #shuffle!(terms)
    n_qubits = length(terms[1])
    coeffs = fill(1e-2, length(terms))

    groups = Vector{Tuple{BitVector, BitVector}}[]
    for (i, term) in enumerate(terms)
        x = BitVector(undef, n_qubits)
        z = BitVector(undef, n_qubits)
        for (j, c) in enumerate(codeunits(term))
            if c == UInt8('I')
                x[j] = false
                z[j] = false
            elseif c == UInt8('X')
                x[j] = true
                z[j] = false
            elseif c == UInt8('Y')
                x[j] = true
                z[j] = true
            elseif c == UInt8('Z')
                x[j] = false
                z[j] = true
            else
                error("Invalid character in term: $term")
            end
        end
        placed = false
        # Check if it commutes with all terms in existing groups
        for (idx_g, g) in enumerate(groups)
            commutes_with_all = true
            for (xg, zg) in g
                if !all(((x .& zg) .⊻ (z .& xg)) .% 2 .== 0)
                    commutes_with_all = false
                    break
                end
            end                
            if commutes_with_all
                placed = true
                push!(groups[idx_g], (x, z))
                break
            end
        end
        if !placed
            push!(groups, [(x, z)])                
        end
    end

    # println("# of terms: ", length(terms))
    if true #$verbose
        for (i, g) in enumerate(groups)
            if length(g) > 10
                println("Group $i (size=$(length(g))): ", [decode_term(term) for term in g])
            end
        end
    end
    println("Total number of groups: ", length(groups))
end

main("ckpot")
#main("usdb")