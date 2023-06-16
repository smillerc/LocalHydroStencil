module SSPRKType

using Polyester, StaticArrays, .Threads, Adapt

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
    # blocks::B
end

# struct SSPRK3IntegratorCPUSplit{T<:Number,AT<:AbstractArray{T},B} <: SSPRK3Integrator
#     U⃗1::AT
#     U⃗2::AT
#     U⃗3::AT
#     iblocks::B
#     jblocks::B
# end

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

    # blocks = @SVector [MArray{Tuple{4,5,5},Float64}(undef) for _ in 1:nthreads()]

    return SSPRK3IntegratorCPU(U1, U2, U3)
end

# function SSPRK3IntegratorCPUSplit(U)
#     U1 = similar(U)
#     U2 = similar(U)
#     U3 = similar(U)
#     T = eltype(U)

#     fill!(U1, 0)
#     fill!(U2, 0)
#     fill!(U3, 0)

#     iblocks = @SVector [MMatrix{4,5,T,20}(undef) for _ in 1:nthreads()]
#     jblocks = @SVector [MMatrix{4,5,T,20}(undef) for _ in 1:nthreads()]

#     return SSPRK3IntegratorCPUSplit(U1, U2, U3, iblocks, jblocks)
# end

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
    SS::SSPRK3Integrator,
    U⃗n::AbstractArray{T},
    riemann_solver,
    mesh,
    EOS,
    dt::Number,
    recon::F1,
    limiter::F2, skip_uniform=true
) where {T,F1,F2}
    nhalo = mesh.nhalo
    ilohi = axes(U⃗n, 2)
    jlohi = axes(U⃗n, 3)
    ilo = first(ilohi) + nhalo
    jlo = first(jlohi) + nhalo
    ihi = last(ilohi) - nhalo
    jhi = last(jlohi) - nhalo

    U⃗1 = SS.U⃗1
    U⃗2 = SS.U⃗2
    U⃗3 = SS.U⃗3

    ΔS_face = mesh.facelen
    vol = mesh.volume
    norms = mesh.facenorms

    # @batch needs to work with isbits types -- somehow see if I can make mesh contain only isbits stuff
    # Stage 1
    @batch per = thread for j in jlo:jhi
        for i in ilo:ihi
            U⁽ⁿ⁾ = SVector{4,T}(view(U⃗n, :, i, j))
            U_local = get_block(U⃗n, i, j)
            S⃗ = @SVector zeros(4)
            n̂ = SMatrix{2,4}(view(norms,:,:,i,j))
            ΔS = SVector{4,T}(view(ΔS_face,:,i,j))
            # n̂ = norms[i, j]
            # ΔS = ΔS_face[i, j]
            Ω = vol[i, j]
            stencil = Stencil9Point(U_local, S⃗, n̂, ΔS, Ω, EOS)
            ∂U⁽ⁿ⁾∂t = ∂U∂t(riemann_solver, stencil, recon, limiter)
            U⁽¹⁾ = U⁽ⁿ⁾ + ∂U⁽ⁿ⁾∂t * dt
            U⃗1[:, i, j] = U⁽¹⁾
        end
    end

    sync_halo!(U⃗1, nhalo)

    # Stage 2
    @batch per = thread for j in jlo:jhi
        for i in ilo:ihi
            U⁽¹⁾ = SVector{4,T}(view(U⃗1, :, i, j))
            U⁽ⁿ⁾ = SVector{4,T}(view(U⃗n, :, i, j))
            U_local = get_block(U⃗1, i, j)
            S⃗ = @SVector zeros(4)
            n̂ = SMatrix{2,4}(view(norms,:,:,i,j))
            ΔS = SVector{4,T}(view(ΔS_face,:,i,j))
            # n̂ = norms[i, j]
            # ΔS = ΔS_face[i, j]
            Ω = vol[i, j]
            stencil = Stencil9Point(U_local, S⃗, n̂, ΔS, Ω, EOS)
            ∂U⁽¹⁾∂t = ∂U∂t(riemann_solver, stencil, recon, limiter)
            U⁽²⁾ = 0.75U⁽ⁿ⁾ + 0.25U⁽¹⁾ + ∂U⁽¹⁾∂t * 0.25dt
            U⃗2[:, i, j] = U⁽²⁾
        end
    end

    sync_halo!(U⃗2, nhalo)

    # Stage 3
    @batch per = thread for j in jlo:jhi
        for i in ilo:ihi
            U⁽²⁾ = SVector{4,T}(view(U⃗2, :, i, j))
            U⁽ⁿ⁾ = SVector{4,T}(view(U⃗n, :, i, j))
            U_local = get_block(U⃗2, i, j)
            S⃗ = @SVector zeros(4)
            n̂ = SMatrix{2,4}(view(norms,:,:,i,j))
            ΔS = SVector{4,T}(view(ΔS_face,:,i,j))
            # n̂ = norms[i, j]
            # ΔS = ΔS_face[i, j]
            Ω = vol[i, j]
            stencil = Stencil9Point(U_local, S⃗, n̂, ΔS, Ω, EOS)
            ∂U⁽²⁾∂t = ∂U∂t(riemann_solver, stencil, recon, limiter)
            U⁽ⁿ⁺¹⁾ = (1 / 3) * U⁽ⁿ⁾ + (2 / 3) * U⁽²⁾ + ∂U⁽²⁾∂t * (2 / 3) * dt
            U⃗3[:, i, j] = U⁽ⁿ⁺¹⁾
        end
    end

    sync_halo!(U⃗3, nhalo)

    return nothing
end

end
