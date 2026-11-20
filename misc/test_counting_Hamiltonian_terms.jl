using Test
include(joinpath(@__DIR__, "counting_Hamiltonian_terms.jl"))

# Small-space regression tests. Run with Julia; no interaction files are needed.
function count_test_space(emax; valence="", threebody=false, e3max=3emax,
                          pp_bucket_mod=1, pn_bucket_mod=1, use_fast_count=true, reverse_order=false)
    emin = isempty(valence) ? 0 : emax
    p, n = count_msps(emax, emin, emax)
    if reverse_order
        reverse!(p); reverse!(n)
    end
    np, nn = length(p), length(n)
    nchunk = cld(np+nn, 64)
    timer = TimerOutput()
    groups = Dict{NTuple{4,Int},Vector{MeasBasisGroup{nchunk}}}()
    gates = zeros(BigInt, 6)
    total = if use_fast_count
        sum(count_diag_terms(p,n,e3max,3,3,threebody))
    else
        diag = generate_diag_terms(p,n,emax,e3max,3,3,threebody)
        big(sum(length,values(diag)))
    end
    total = possible_1b_terms(total, p, n, 3, 3, devnull, false; emin, emax)
    total = possible_2b_T1_terms!(groups, gates, total, p, n, timer, devnull,
                                 false, nchunk, threebody, false, false, 0;
                                 emin, emax, symmfactor=2, pp_bucket_mod, use_fast_count)
    total, _ = possible_2b_pn_terms_streaming!(total, p, n, timer, devnull, nchunk, false;
                                             emin, emax, pn_bucket_mod, use_fast_count)
    if threebody
        ppn = if use_fast_count
            count_3b_ppn_terms(p,n; emin,emax,e3min=0,e3max)
        else
            ppn = 0
            xp, zp, xn, zn = ntuple(_ -> zeros(MVector{nchunk,UInt64}), 4)
            for tasks in gen_ppn_tasks(p, n, emin, emax, 0, e3max), patterns in values(tasks)
                for (pp, neutron) in patterns
                    pp_idxs, n_idxs = decode_4(pp), decode_2(neutron)
                    xy = n_idxs[1] == n_idxs[2] ? XY_patterns_ppn_4 : XY_patterns_T1_4
                    ppn += _worker_job_ppn!(String[], xy, pp_idxs, n_idxs, xp, zp, xn, zn,
                                           falses(nn), np, nn, timer, false)
                end
            end
            ppn
        end
        total += 2ppn
        total = possible_3b_terms_parallel(groups, nchunk, gates, total, 3, 3, p, n,
                                            timer, devnull, false, 0;
                                            emin, emax, e3min=0, e3max, symfactor=2, use_fast_count)
    end
    total
end

@testset "Batch counting and TSV export" begin
    configs = [CountConfig(valence_space="p"), CountConfig(emax=2),
               CountConfig(emax=1, is_3NF=true),
               CountConfig(emax=1, is_3NF=true, e3max=1)]
    mktempdir() do dir
        path = joinpath(dir, "counts.tsv")
        results = export_counts(configs; output_path=path, progress=false)
        @test [r.nterms for r in results[1:3]] == [975, 57445, 51377]
        @test [r.izterms for r in results[1:3]] == [79, 821, 697]
        @test [r.nqubits for r in results] == [12, 40, 16, 16]
        @test results[4].modelspace == "1_2_1"
        redirect_stdout(devnull) do
            @test results[4].nterms == count_test_space(1; threebody=true, e3max=1, use_fast_count=false)
            p, n = count_msps(1)
            @test results[4].izterms == sum(length, values(generate_diag_terms(p, n, 1, 1, 3, 3, true)))
        end
        @test results[4].izterms < results[3].izterms
        lines = readlines(path)
        header = split(first(lines), '\t')
        @test header[1:6] == ["Interaction-type", "modelspace", "Nq", "T-estimates-controlled", "Num of Hterms", "IZterms"]
        @test length(lines) == 5
        for (line, r) in zip(lines[2:end], results)
            fields = split(line, '\t'; keepempty=true)
            @test length(fields) == length(header)
            row = Dict(zip(header, fields))
            @test parse(BigInt, row["Num of Hterms"]) == r.nterms
            @test parse(BigInt, row["IZterms"]) == r.izterms
            @test parse(BigInt, row["T-estimates-controlled"]) == 200*(r.nterms-1)
            @test row["e3max"] == (isnothing(r.e3max) ? "" : string(r.e3max))
            @test row["source_sha256"] == bytes2hex(open(sha256, joinpath(@__DIR__, "counting_Hamiltonian_terms.jl")))
            @test row["count_kind"] == "symmetry_structural"
            @test row["identity_included"] == "true"
        end
        # Failed batches/validation must leave the previous complete TSV intact.
        previous = read(path)
        @test_throws ArgumentError export_counts([CountConfig(emax=-1)]; output_path=path, progress=false)
        @test_throws ArgumentError write_count_tsv([results[1], results[1]], path)
        @test_throws ArgumentError write_count_tsv([merge(results[1], (; nterms=big(0)))], path)
        @test_throws ArgumentError export_counts(CountConfig[]; output_path=path, progress=false)
        @test read(path) == previous
        @test readdir(dir) == ["counts.tsv"]
        # Replacing an existing complete file succeeds as well.
        write_count_tsv(results[1:1], path)
        @test length(readlines(path)) == 2
    end
    @test_throws ArgumentError count_hamiltonian_terms(CountConfig(emax=1, e3max=1))
    @test_throws ArgumentError count_hamiltonian_terms(CountConfig(emax=1, is_3NF=true, e3max=4))
    @test_throws ArgumentError count_hamiltonian_terms(CountConfig(valence_space="p", is_3NF=true))
    @test_throws ErrorException count_hamiltonian_terms(CountConfig(valence_space="unknown"))
end

@testset "Scalar one-body terms" begin
    for (emax, expected) in ((0,0), (1,0), (2,8), (3,32), (4,88))
        p, n = count_msps(emax)
        redirect_stdout(devnull) do
            @test possible_1b_terms(0, p, n, 1, 1, devnull, false) == expected
            @test possible_1b_terms(0, p, n, 1, 0, devnull, false) == expected÷2
            @test possible_1b_terms(0, p, n, 0, 1, devnull, false) == expected÷2
            @test possible_1b_terms(0, p, n, 0, 0, devnull, false) == 0
        end
    end
    p, n = count_msps(2)
    io = IOBuffer()
    nterms = redirect_stdout(devnull) do
        possible_1b_terms(0, p, n, 1, 1, io, true)
    end
    labels = [split(line, ": ")[2] for line in split(chomp(String(take!(io))), '\n')]
    @test length(labels) == length(Set(labels)) == nterms == 8
    @test all(s -> count(c -> c in ('X','Y'), s) == 2, labels)
    @test all(s -> !occursin('X',s) || !occursin('Y',s), labels)
    io = IOBuffer()
    redirect_stdout(devnull) do
        possible_1b_terms(0, reverse(p), reverse(n), 1, 1, io, true)
    end
    reversed_labels = [split(line, ": ")[2] for line in split(chomp(String(take!(io))), '\n')]
    @test Set(s[21:40]*s[1:20] for s in reverse.(reversed_labels)) == Set(labels)
end

@testset "Exact angular-momentum selection" begin
    @test iszero(cg_racah_sum(3,-1,3,-1,4))
    @test iszero(cg_racah_sum(7,-3,3,1,6))
    @test !iszero(cg_racah_sum(3,-1,3,-1,6))
    @test !iszero(cg_racah_sum(7,-3,3,1,4))
    a = Orbit_nljjztz(0,1,3,-1,-1)
    b = Orbit_nljjztz(0,1,3,-1,1)
    c = Orbit_nljjztz(0,3,7,-3,-1)
    d = Orbit_nljjztz(0,1,3,1,1)
    @test !Jbracket_check(a,b,c,d; same_species=false)
    @test Jbracket_check(a,b,a,b; same_species=false)

    sm = Orbit_nljjztz(0,0,1,-1,-1)
    sp = Orbit_nljjztz(0,0,1,1,-1)
    sp1 = Orbit_nljjztz(1,0,1,1,-1)
    p1 = Orbit_nljjztz(0,1,1,1,1)
    p3 = Orbit_nljjztz(0,1,3,1,1)
    @test allowed_pair_Js(sm,sp) == [0]
    @test !Jbracket_check_3body(sm,sp,p1,sm,sp,p3)
    # Distinct intermediate J12 values can share the same total J.
    @test allowed_pair_Js(sp,sp1) == [2]
    @test Jbracket_check_3body(sm,sp,p3,sp,sp1,Orbit_nljjztz(0,1,1,-1,1))

    ps = [Orbit_nljjztz(0,1,3,m,-1) for m in -3:2:3]
    for (i,j,k) in _combinations(collect(eachindex(ps)),3)
        @test allowed_three_body_Js(ps[i],ps[j],ps[k]) == [3]
    end
    @test isempty(allowed_three_body_Js(ps[1],ps[1],ps[2]))
    p, _ = count_msps(1)
    for (i,j,k) in _combinations(collect(eachindex(p)),3)
        expected = allowed_three_body_Js(p[i],p[j],p[k])
        for (a,b,c) in ((i,k,j),(j,i,k),(j,k,i),(k,i,j),(k,j,i))
            @test allowed_three_body_Js(p[a],p[b],p[c]) == expected
        end
    end
end

@testset "Small-space structural counts" begin
    redirect_stdout(devnull) do
        @test count_test_space(0) == 19
        @test count_test_space(1) == 1793
        @test count_test_space(2) == 57445  # Includes 160 coefficient-cancelled candidates.
        @test count_test_space(1; valence="p") == 975
        @test count_test_space(2; valence="sd") == 12869
        @test count_test_space(3; valence="pf") == 82861
        @test count_test_space(0; threebody=true) == 23
        @test count_test_space(1; threebody=true) == 51377
        @test count_test_space(2; pp_bucket_mod=3, pn_bucket_mod=3) == 57445
    end
end

@testset "Optimized versus reference enumeration" begin
    redirect_stdout(devnull) do
        for emax in 0:3, reverse_order in (false,true)
            @test count_test_space(emax; reverse_order) ==
                  count_test_space(emax; reverse_order,use_fast_count=false)
        end
        for (emax,e3max) in ((0,0),(1,0),(1,1),(1,2),(1,3),(2,2),(2,4),(2,6))
            @test count_test_space(emax; threebody=true,e3max) ==
                  count_test_space(emax; threebody=true,e3max,use_fast_count=false)
        end
        @test count_test_space(2; threebody=true,reverse_order=true) == 15346373
        @test count_test_space(2; threebody=true,reverse_order=true,use_fast_count=false) == 15346373
        for emax in 1:3
            @test count_test_space(emax; valence="shell") ==
                  count_test_space(emax; valence="shell",use_fast_count=false)
        end
        p,n = count_msps(1)
        for threebody in (false,true), cutoff in 0:3, Z in 0:3, N in 0:3
            expected = generate_diag_terms(p,n,1,cutoff,Z,N,threebody)
            @test count_diag_terms(p,n,cutoff,Z,N,threebody) == [length(expected[r]) for r in 0:(threebody ? 3 : 2)]
        end
        # Unequal species spaces and shuffled quantum numbers must not change the kernels.
        p = p[[8,1,6,3,5]]; n = n[[4,7,2,5]]
        timer=TimerOutput()
        expected,_ = possible_2b_pn_terms_streaming!(big(0),p,n,timer,devnull,1,false; use_fast_count=false)
        @test count_2b_terms(p,n;same_species=false) == expected
        for cutoff in 0:3
            expected = 0
            xp,zp,xn,zn = ntuple(_ -> zeros(MVector{1,UInt64}),4)
            for tasks in gen_ppn_tasks(p,n,0,1,0,cutoff), patterns in values(tasks)
                for (pp,neutron) in patterns
                    pp_idxs,n_idxs = decode_4(pp),decode_2(neutron)
                    xy = n_idxs[1]==n_idxs[2] ? XY_patterns_ppn_4 : XY_patterns_T1_4
                    expected += _worker_job_ppn!(String[],xy,pp_idxs,n_idxs,xp,zp,xn,zn,falses(length(p)+length(n)),length(p),length(n),timer,false)
                end
            end
            @test count_3b_ppn_terms(p,n;emax=1,e3max=cutoff) == expected
        end
    end
    @test angular_mask_type(63) == UInt64
    @test angular_mask_type(64) == UInt128
    @test angular_mask_type(127) == UInt128
    @test angular_mask_type(128) == BigInt
    @test angular_mask([0,64,128],BigInt) == big(1)+(big(1)<<64)+(big(1)<<128)
    @test angular_mask([63],UInt64) == UInt64(1)<<63
end
