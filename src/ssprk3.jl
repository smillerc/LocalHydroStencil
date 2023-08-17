module SSPRKType

using Polyester, StaticArrays, .Threads, Adapt
using LIKWID

using ..StencilType
using ..EOSType
using ..ReconstructionType
using ..RiemannSolverType

export SSPRK3IntegratorCPU, integrate!

abstract type SSPRK3Integrator end

struct SSPRK3IntegratorCPU{T<:Number,AT<:AbstractArray{T}} <: SSPRK3Integrator
    U⃗1::AT
    U⃗2::AT
    U⃗3::AT
end

struct SSPRK3IntegratorCPUSplit{T<:Number,AT<:AbstractArray{T}} <: SSPRK3Integrator
    U⃗1::AT
    U⃗2::AT
    U⃗3::AT
end

struct SSPRK3IntegratorGPU{T<:Number,AT<:AbstractArray{T}} <: SSPRK3Integrator
    U⃗1::AT
    U⃗2::AT
    U⃗3::AT
end

function SSPRK3IntegratorCPU(U)
    U1 = similar(U)
    U2 = similar(U)
    U3 = similar(U)

    fill!(U1, 0)
    fill!(U2, 0)
    fill!(U3, 0)

    return SSPRK3IntegratorCPU(U1, U2, U3)
end

function SSPRK3IntegratorCPUSplit(U)
    U1 = similar(U)
    U2 = similar(U)
    U3 = similar(U)

    fill!(U1, 0)
    fill!(U2, 0)
    fill!(U3, 0)

    return SSPRK3IntegratorCPUSplit(U1, U2, U3)
end

function SSPRK3IntegratorGPU(U)
    U1 = similar(U)
    U2 = similar(U)
    U3 = similar(U)

    fill!(U1, 0)
    fill!(U2, 0)
    fill!(U3, 0)

    return SSPRK3IntegratorGPU(U1, U2, U3)
end

function Adapt.adapt_structure(to, SS::SSPRK3IntegratorGPU)
    U1 = Adapt.adapt_structure(to, SS.U⃗1)
    U2 = Adapt.adapt_structure(to, SS.U⃗2)
    U3 = Adapt.adapt_structure(to, SS.U⃗3)

    return SSPRK3IntegratorGPU(U1, U2, U3)
end

function sync_halo!(U, nhalo)
    ilohi = axes(U, 2)
    jlohi = axes(U, 3)
    ilo = first(ilohi) + nhalo
    jlo = first(jlohi) + nhalo
    ihi = last(ilohi) - nhalo
    jhi = last(jlohi) - nhalo

    for j in 1:(jlo - 1)
        for i in ilo:ihi
            for q in axes(U, 1)
                U[q, i, j] = U[q, i, jlo]
            end
        end
    end

    for j in (jhi - nhalo):last(jlohi)
        for i in ilo:ihi
            for q in axes(U, 1)
                U[q, i, j] = U[q, i, jhi]
            end
        end
    end

    for j in jlo:jhi
        for i in first(ilohi):(ilo - 1)
            for q in axes(U, 1)
                U[q, i, j] = U[q, ilo, j]
            end
        end
    end

    for j in jlo:jhi
        for i in (ihi - nhalo):last(ilohi)
            for q in axes(U, 1)
                U[q, i, j] = U[q, ihi, j]
            end
        end
    end

    return nothing
end


@inbounds function integrate!(
    SS::SSPRK3IntegratorCPU,
    U⃗n::AbstractArray{T},
    mesh,
    EOS,
    dt::Number,
    # BCs,
    riemann_solver,
    recon::F2,
    limiter::F3, skip_uniform=true
) where {T,F2,F3}
    nhalo = mesh.nhalo
    ilohi = axes(U⃗n, 2)
    jlohi = axes(U⃗n, 3)
    ilo = first(ilohi) + nhalo
    jlo = first(jlohi) + nhalo
    ihi = last(ilohi) - nhalo
    jhi = last(jlohi) - nhalo

    looplimits = (ilo, ihi, jlo, jhi)
    U⃗1 = SS.U⃗1
    U⃗2 = SS.U⃗2
    U⃗3 = SS.U⃗3

    ΔS_face = mesh.facelen
    vol = mesh.volume
    norms = mesh.facenorms
    centroid_pos = mesh.centroid

    # @timeit "applyBC!" applyBC!(BCs, mesh, EOS, U⃗n)
    # applyBC!(BCs, mesh, EOS, U⃗n)
    # sync_halo!(Un, nhalo)

    # Stage 1
    # @timeit "stage 1" begin
    @batch per=core for j in jlo:jhi
        for i in ilo:ihi
            # @marker "dUdt" begin
                U⁽ⁿ⁾ = SVector{4,T}(view(U⃗n, :, i, j))
                U_local = get_block(U⃗n, i, j)
                S⃗ = @SVector zeros(4)
                n̂ = SMatrix{2,4}(view(norms,:,:,i,j))
                ΔS = SVector{4,T}(view(ΔS_face,:,i,j))
                Ω = vol[i, j]
                x⃗_c = SVector{2,T}(view(centroid_pos,:,i,j))
                stencil = Stencil9Point(U_local, S⃗, n̂, ΔS, Ω, EOS, x⃗_c)
                ∂U⁽ⁿ⁾∂t = riemann_solver.∂U∂t(stencil, recon, limiter, skip_uniform)
                U⃗1[:, i, j] = U⁽ⁿ⁾ + ∂U⁽ⁿ⁾∂t * dt
            # end
        end
    end


    # @timeit "applyBC!" applyBC!(BCs, mesh, EOS, U⃗1)
    # applyBC!(BCs, mesh, EOS, U⃗1)
    # sync_halo!(U⃗1, nhalo)

    # Stage 2
    @batch per=core for j in jlo:jhi
        for i in ilo:ihi
            # @marker "dUdt" begin
                U⁽¹⁾ = SVector{4,T}(view(U⃗1, :, i, j))
                U⁽ⁿ⁾ = SVector{4,T}(view(U⃗n, :, i, j))
                U_local = get_block(U⃗1, i, j)
                S⃗ = @SVector zeros(4)
                n̂ = SMatrix{2,4}(view(norms,:,:,i,j))
                ΔS = SVector{4,T}(view(ΔS_face,:,i,j))
                Ω = vol[i, j]
                x⃗_c = SVector{2,T}(view(centroid_pos,:,i,j))
                stencil = Stencil9Point(U_local, S⃗, n̂, ΔS, Ω, EOS, x⃗_c)
                ∂U⁽¹⁾∂t = riemann_solver.∂U∂t(stencil, recon, limiter, skip_uniform)
                U⁽²⁾ = 0.75U⁽ⁿ⁾ + 0.25U⁽¹⁾ + ∂U⁽¹⁾∂t * 0.25dt
                U⃗2[:, i, j] = U⁽²⁾
            # end
        end
    end


    # @timeit "applyBC!" applyBC!(BCs, mesh, EOS, U⃗2)
    # applyBC!(BCs, mesh, EOS, U⃗2)
    # sync_halo!(U⃗2, nhalo)

    # Stage 3
    @batch per=core for j in jlo:jhi
        for i in ilo:ihi
            # @marker "dUdt" begin
                U⁽²⁾ = SVector{4,T}(view(U⃗2, :, i, j))
                U⁽ⁿ⁾ = SVector{4,T}(view(U⃗n, :, i, j))
                U_local = get_block(U⃗2, i, j)
                S⃗ = @SVector zeros(4)
                n̂ = SMatrix{2,4}(view(norms,:,:,i,j))
                ΔS = SVector{4,T}(view(ΔS_face,:,i,j))
                Ω = vol[i, j]
                x⃗_c = SVector{2,T}(view(centroid_pos,:,i,j))
                stencil = Stencil9Point(U_local, S⃗, n̂, ΔS, Ω, EOS, x⃗_c)
                ∂U⁽²⁾∂t = riemann_solver.∂U∂t(stencil, recon, limiter, skip_uniform)
                U⁽ⁿ⁺¹⁾ = (1 / 3) * U⁽ⁿ⁾ + (2 / 3) * U⁽²⁾ + ∂U⁽²⁾∂t * (2 / 3) * dt
                U⃗3[:, i, j] = U⁽ⁿ⁺¹⁾
            # end
        end
    end

    # sync_halo!(U⃗3, nhalo)
    # resids = check_residuals(U⃗2,U⃗1,looplimits)
    resids = @SVector zeros(4)
    success = true
    return success, resids
end

@inbounds function integrate!(
    SS::SSPRK3IntegratorCPUSplit,
    U⃗n::AbstractArray{T},
    mesh,
    EOS,
    dt::Number,
    # BCs,
    riemann_solver,
    recon::F2,
    limiter::F3, skip_uniform=true
) where {T,F2,F3}
    nhalo = mesh.nhalo
    ilohi = axes(U⃗n, 2)
    jlohi = axes(U⃗n, 3)
    ilo = first(ilohi) + nhalo
    jlo = first(jlohi) + nhalo
    ihi = last(ilohi) - nhalo
    jhi = last(jlohi) - nhalo

    looplimits = (ilo, ihi, jlo, jhi)
    U⃗1 = SS.U⃗1
    U⃗2 = SS.U⃗2
    U⃗3 = SS.U⃗3

    ΔS_face = mesh.facelen
    vol = mesh.volume
    norms = mesh.facenorms
    centroid_pos = mesh.centroid

    # @timeit "applyBC!" applyBC!(BCs, mesh, EOS, U⃗n)
    # applyBC!(BCs, mesh, EOS, U⃗n)
    # sync_halo!(Un, nhalo)

    # Stage 1
    # @timeit "stage 1" begin
    @batch per=core for j in jlo:jhi
        for i in ilo:ihi
            @marker "dUdt" begin
            U⁽ⁿ⁾ = SVector{4,T}(view(U⃗n, :, i, j))
            # U_local = get_block(U⃗n, i, j)

            U⃗ᵢ = get_iblock(U⃗n)
            U⃗ⱼ = get_jblock(U⃗n)
            U⃗ᵢ₊₁ⱼ₊₁ = SVector{4,T}(view(U⃗n, :, i+1, j+1))
            U⃗ᵢ₊₁ⱼ₋₁ = SVector{4,T}(view(U⃗n, :, i+1, j-1))
            U⃗ᵢ₋₁ⱼ₊₁ = SVector{4,T}(view(U⃗n, :, i-1, j+1))
            U⃗ᵢ₋₁ⱼ₋₁ = SVector{4,T}(view(U⃗n, :, i-1, j-1))

            S⃗ = @SVector zeros(4)
            n̂ = SMatrix{2,4}(view(norms,:,:,i,j))
            ΔS = SVector{4,T}(view(ΔS_face,:,i,j))
            Ω = vol[i, j]
            x⃗_c = SVector{2,T}(view(centroid_pos,:,i,j))
            stencil = Stencil9PointSplit(U⃗ᵢ, U⃗ⱼ, U⃗ᵢ₊₁ⱼ₊₁, U⃗ᵢ₊₁ⱼ₋₁, U⃗ᵢ₋₁ⱼ₊₁, U⃗ᵢ₋₁ⱼ₋₁, S⃗, n̂, ΔS, Ω, EOS, x⃗_c)
            ∂U⁽ⁿ⁾∂t = riemann_solver.∂U∂t(stencil, recon, limiter, skip_uniform)
            U⃗1[:, i, j] = U⁽ⁿ⁾ + ∂U⁽ⁿ⁾∂t * dt
            end
        end
    end


    # # @timeit "applyBC!" applyBC!(BCs, mesh, EOS, U⃗1)
    # # applyBC!(BCs, mesh, EOS, U⃗1)
    # # sync_halo!(U⃗1, nhalo)

    # # Stage 2
    # @batch per=thread for j in jlo:jhi
    #     for i in ilo:ihi
    #         @marker "dUdt" begin
    #         U⁽¹⁾ = SVector{4,T}(view(U⃗1, :, i, j))
    #         U⁽ⁿ⁾ = SVector{4,T}(view(U⃗n, :, i, j))
    #         U_local = get_block(U⃗1, i, j)
    #         S⃗ = @SVector zeros(4)
    #         n̂ = SMatrix{2,4}(view(norms,:,:,i,j))
    #         ΔS = SVector{4,T}(view(ΔS_face,:,i,j))
    #         Ω = vol[i, j]
    #         x⃗_c = SVector{2,T}(view(centroid_pos,:,i,j))
    #         stencil = Stencil9Point(U_local, S⃗, n̂, ΔS, Ω, EOS, x⃗_c)
    #         ∂U⁽¹⁾∂t = riemann_solver.∂U∂t(stencil, recon, limiter, skip_uniform)
    #         U⁽²⁾ = 0.75U⁽ⁿ⁾ + 0.25U⁽¹⁾ + ∂U⁽¹⁾∂t * 0.25dt
    #         U⃗2[:, i, j] = U⁽²⁾
    #         end
    #     end
    # end


    # # @timeit "applyBC!" applyBC!(BCs, mesh, EOS, U⃗2)
    # # applyBC!(BCs, mesh, EOS, U⃗2)
    # # sync_halo!(U⃗2, nhalo)

    # # Stage 3
    # @batch per=thread for j in jlo:jhi
    #     for i in ilo:ihi
    #         @marker "dUdt" begin
    #         U⁽²⁾ = SVector{4,T}(view(U⃗2, :, i, j))
    #         U⁽ⁿ⁾ = SVector{4,T}(view(U⃗n, :, i, j))
    #         U_local = get_block(U⃗2, i, j)
    #         S⃗ = @SVector zeros(4)
    #         n̂ = SMatrix{2,4}(view(norms,:,:,i,j))
    #         ΔS = SVector{4,T}(view(ΔS_face,:,i,j))
    #         Ω = vol[i, j]
    #         x⃗_c = SVector{2,T}(view(centroid_pos,:,i,j))
    #         stencil = Stencil9Point(U_local, S⃗, n̂, ΔS, Ω, EOS, x⃗_c)
    #         ∂U⁽²⁾∂t = riemann_solver.∂U∂t(stencil, recon, limiter, skip_uniform)
    #         U⁽ⁿ⁺¹⁾ = (1 / 3) * U⁽ⁿ⁾ + (2 / 3) * U⁽²⁾ + ∂U⁽²⁾∂t * (2 / 3) * dt
    #         U⃗3[:, i, j] = U⁽ⁿ⁺¹⁾
    #         end
    #     end
    # end

    # sync_halo!(U⃗3, nhalo)
    # resids = check_residuals(U⃗2,U⃗1,looplimits)
    resids = @SVector zeros(4)
    success = true
    return success, resids
end

function check_residuals(U1, Un, looplimits)
    # TODO: make this work for diff sizes

    ilo, ihi, jlo, jhi = looplimits
    ϕ1_denoms = @MVector zeros(4)
    resids = @MVector zeros(4)
    numerators = @MVector zeros(4)
    fill!(resids, -Inf)

    @batch for j in jlo:jhi
        for i in ilo:ihi
            for q in eachindex(ϕ1_denoms)
                ϕ1_denoms[q] += U1[q,i,j]^2
            end
        end
    end
    ϕ1_denoms = sqrt.(ϕ1_denoms)


    # if isinf.(ϕ1_denoms) || iszero(ϕ1_denoms)
    #     resids = -Inf
    # else

        @batch for j in jlo:jhi
            for i in ilo:ihi
                for q in eachindex(ϕ1_denoms)
                    numerators[q] += (Un[q,i,j] - U1[q,i,j])^2
                end
            end
        end

        resids = sqrt.(numerators) ./ ϕ1_denoms
    # end

    return SVector(resids)
end

end
