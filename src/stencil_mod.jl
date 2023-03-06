# module StencilType

using StaticArrays, Base.Threads, CUDA, Polyester, LinearAlgebra, Unitful
using Adapt, MuladdMacro
using StrideArrays
using VectorizedStatistics

# export mesh2stencil, SSPRK3, SSPRK3_turbo, M_AUSMPWPlus2D
# export SSPRK3_gpu!, ∂U∂t, riemann_solver, minmod, muscl

abstract type AbstractRiemannSolver end
struct M_AUSMPWPlus2D <: AbstractRiemannSolver end



const ϵ = 2eps(Float64)

struct Stencil9Point{T,AA,E}
    U⃗::AA
    S⃗::SVector{4,T} # source terms
    n̂::SMatrix{2,4,T,8}
    ΔS::SVector{4,T}
    Ω::T
    EOS::E
end

struct Stencil9PointSplit{T,AA,AB,E}
    U⃗ᵢ::AA
    U⃗ⱼ::AA
    U⃗ᵢ₊₁ⱼ₊₁::AB
    U⃗ᵢ₊₁ⱼ₋₁::AB
    U⃗ᵢ₋₁ⱼ₊₁::AB
    U⃗ᵢ₋₁ⱼ₋₁::AB
    S⃗::SVector{4,T} # source terms
    n̂::SMatrix{2,4,T,8}
    ΔS::SVector{4,T}
    Ω::T
    EOS::E
end

struct CartesianMesh{T,
	AA2<:AbstractArray{T,2}, 
	AA3<:AbstractArray{T,3}, 
	AA4<:AbstractArray{T,4}
    }
    xy::AA3
    centroid::AA3
    volume::AA2
    facenorms::AA4
    facelen::AA3
    nhalo::Int
end

function CartesianMesh(x::AbstractVector{T}, y::AbstractVector{T}, nhalo) where T
    M = length(x) - 1 #+ 2nhalo
    N = length(y) - 1 #+ 2nhalo

    xy = zeros(2, M + 1, N + 1)
    for j in 1:M+1
        for i in 1:N+1
            xy[1,i,j] = x[i]
            xy[2,i,j] = y[j]
        end
    end

    centroid = quad_centroids(xy)
    volume = quad_volumes(xy)

    facelen, facenorms = quad_face_areas_and_vecs(xy)

    return CartesianMesh(xy, centroid, volume, facenorms, facelen, nhalo)
end


abstract type SSPRK3Integrator end

struct SSPRK3IntegratorCPU{T<:Number, AT<:AbstractArray{T}, B} <: SSPRK3Integrator
    U⃗1::AT
    U⃗2::AT
    U⃗3::AT
    blocks::B
end

struct SSPRK3IntegratorCPUSplit{T<:Number, AT<:AbstractArray{T}, B} <: SSPRK3Integrator
    U⃗1::AT
    U⃗2::AT
    U⃗3::AT
    iblocks::B
    jblocks::B
end

struct SSPRK3IntegratorGPU{T<:Number, AT<:AbstractArray{T}} <: SSPRK3Integrator
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

    blocks = @SVector [MArray{Tuple{4,5,5},Float64}(undef) for _ in 1:nthreads()]

    SSPRK3IntegratorCPU(U1, U2, U3, blocks)
end

function SSPRK3IntegratorCPU(U)
    U1 = similar(U)
    U2 = similar(U)
    U3 = similar(U)
    
    fill!(U1, 0)
    fill!(U2, 0)
    fill!(U3, 0)

    blocks = @SVector [MArray{Tuple{4,5,5},Float64}(undef) for _ in 1:nthreads()]

    SSPRK3IntegratorCPU(U1, U2, U3, blocks)
end

function SSPRK3IntegratorCPUSplit(U)
    U1 = similar(U)
    U2 = similar(U)
    U3 = similar(U)
    T = eltype(U)
    
    fill!(U1, 0)
    fill!(U2, 0)
    fill!(U3, 0)

    iblocks = @SVector [MMatrix{4,5,T,20}(undef) for _ in 1:nthreads()]
    jblocks = @SVector [MMatrix{4,5,T,20}(undef) for _ in 1:nthreads()]

    SSPRK3IntegratorCPUSplit(U1, U2, U3, iblocks, jblocks)
end

function SSPRK3IntegratorGPU(U)
    U1 = similar(U)
    U2 = similar(U)
    U3 = similar(U)
    
    fill!(U1, 0)
    fill!(U2, 0)
    fill!(U3, 0)

    SSPRK3IntegratorGPU(U1, U2, U3)
end

function Adapt.adapt_structure(to, SS::SSPRK3IntegratorGPU)
    U1 = Adapt.adapt_structure(to, SS.U⃗1)
    U2 = Adapt.adapt_structure(to, SS.U⃗2)
    U3 = Adapt.adapt_structure(to, SS.U⃗3)

    SSPRK3IntegratorGPU(U1, U2, U3)
end

function Adapt.adapt_structure(to, mesh::CartesianMesh)

    _xy = Adapt.adapt_structure(to,  mesh.xy)
    _centroid = Adapt.adapt_structure(to,  mesh.centroid)
    _volume = Adapt.adapt_structure(to,  mesh.volume)
    _facenorms = Adapt.adapt_structure(to,  mesh.facenorms)
    _facelen = Adapt.adapt_structure(to,  mesh.facelen)
    _nhalo = Adapt.adapt_structure(to,  mesh.nhalo)

    CartesianMesh(_xy,
        _centroid,
        _volume,
        _facenorms,
        _facelen,
        _nhalo,
    )

end

include("cpu.jl")
include("gpu.jl")

function cons2prim(EOS, ρ, ρu, ρv, ρE)
    invρ = 1/ρ
    u = ρu * invρ
    v = ρv * invρ
    E = ρE * invρ
    p = pressure(EOS, ρ, u, v, E)
    # p = ρ * EOS._γ_m_1 * (E - 0.5(u^2 + v^2))

    return @SVector [ρ, u, v, p]
end

function cons2prim_vec(EOS, U)
    ρ = U[1] 
    invρ = 1/ρ
    u = U[2] * invρ
    v = U[3] * invρ
    E = U[4] * invρ
    # p = pressure(EOS, ρ, u, v, E)
    p = ρ * EOS._γ_m_1 * (E - 0.5(u^2 + v^2))

    return @SVector [ρ, u, v, p]
end

# function cons2prim(gamma, ρ, ρu, ρv, ρE)
#     invρ = 1/ρ
#     u = ρu * invρ
#     v = ρv * invρ
#     E = ρE * invρ
#     p = ρ * (gamma - 1) * (E - 0.5(u^2 + v^2))

#     return ρ, u, v, p
# end


function cons2prim(gamma, U)
    W = similar(U)
    for j in 1:5
        for i in 1:5
            ρ = U[1,i,j]
            invρ = 1/ρ
            u = U[2,i,j] * invρ
            v = U[3,i,j] * invρ
            E = U[4,i,j] * invρ

            W[1,i,j] = ρ
            W[2,i,j] = u * invρ
            W[3,i,j] = v * invρ
            W[4,i,j] = ρ * (gamma - 1) * (E - 0.5(u^2 + v^2))
        end
    end

    return SArray{Tuple{4,5,5}}(W)
end

function cons2prim2(gamma, U)
    W = similar(U)
    for j in 1:5
        for i in 1:5
            ρ = U[1,i,j]
            invρ = 1/ρ
            u = U[2,i,j] * invρ
            v = U[3,i,j] * invρ
            E = U[4,i,j] * invρ

            W[1,i,j] = ρ
            W[2,i,j] = u * invρ
            W[3,i,j] = v * invρ
            W[4,i,j] = ρ * (gamma - 1) * (E - 0.5(u^2 + v^2))
        end
    end

    return SArray{Tuple{4,5,5}}(W)
end

function getblock_stride(U,i,j)
        blk = (
            StrideArray{Float64}(undef,(StaticInt(4),StaticInt(5),StaticInt(5)))
        )
        # uview = @view U[:,i-2:i+2,j-2:j+2]
        uview = view(U,:,
            i-StaticInt(2):i+StaticInt(2),
            j-StaticInt(2):j+StaticInt(2)
        )
        # copy!(blk, uview)
        for idx in eachindex(uview)
            @inbounds blk[idx] = uview[idx]
        end
    blk
    # blk
end

function getblock_static(U,i,j)
    @inbounds begin
        uview = @view U[:,i-2:i+2,j-2:j+2]
        blk = SArray{Tuple{4,5,5},Float64}(uview)
    end
    blk
end


function getview(U,i,j)
    view(U,:,i-2:i+2,j-2:j+2)
end

function get_block(U,i,j)
    uview = getview(U,i,j)
    SArray{Tuple{4,5,5},Float64,3,100}(uview)
end


function getblock_marray(U,i,j)
    # @inbounds begin
        blk = MArray{Tuple{4,5,5},Float64}(undef)
        fill!(blk, 0)
        i0 = 1
        j0 = 1
        for jdx in j-2:j+2
            for idx in i-2:i+2
                for q in 1:4
                    blk[q, i0, j0] = U[q,idx,jdx]
                    # @show U[q,idx,jdx], blk[q, i0, j0]
                end
            end
            i0+=1
            j0+=1
        end

        # uview = @view U[:,i-2:i+2,j-2:j+2]
        # for idx in eachindex(blk)
        #     blk[idx] = uview[idx]
        # end
    # end
    # SArray{Tuple{4,5,5},Float64,3,100}(blk)
    # @SArray blk
    SArray(blk)
end

# function getblock_marray(U,i,j)
#     @inbounds begin
#         blk = MArray{Tuple{4,5,5},Float64}(undef)
#         uview = @view U[:,i-2:i+2,j-2:j+2]
#         for idx in eachindex(blk)
#             blk[idx] = uview[idx]
#         end
#     end
#     SArray(blk)
# end

# function getblock_marray(U,i,j)
#     # @inbounds begin
#         blk = MArray{Tuple{4,5,5},Float64}(undef)
#         uview = @view U[:,i-2:i+2,j-2:j+2]
#         copy!(blk, uview)
#     # end
#     blk
# end

# @benchmark begin 
#     StrideArrays.@gc_preserve getblock_stride($U,50,50)
# end

@inbounds function getblock(A::AbstractArray{T,3}, i, j, nh) where {T}
    S = Tuple{4,5,5}
    block = MArray{S,T}(undef)
    aview = view(A, :, i-nh:i+nh, j-nh:j+nh)
    for i in eachindex(block)
        @inbounds block[i] = aview[i]
    end
    return SArray(block)
end

function getvec(A::AbstractArray{T,3}, i, j) where {T}
    SVector{4,T}(view(A, 1:4, i, j))
end

function get_nhat(mesh::CartesianMesh{T}, i, j) where T
    SMatrix{2,4,T}(view(mesh.facenorms,1:2,1:4,i,j))
end

function get_ΔS(mesh::CartesianMesh{T}, i, j) where T
    SVector{4,T}(view(mesh.facelen,1:4,i,j))
end

function _2halo2dstencil(U⃗::AbstractArray{T,3}, Idx, mesh, EOS::E, nh) where {T,E}
    i, j = Idx
    U_local = getblock(U⃗, i, j, nh)
    S⃗ = @SVector zeros(4)
    n̂ = get_nhat(mesh, i, j)
    ΔS = get_ΔS(mesh, i, j)
    Ω = mesh.volume[i,j]
    BT = SArray{Tuple{4,5,5},T,3,100}

    return Stencil9Point{T,BT,E}(U_local, S⃗, n̂, ΔS, Ω, EOS)
end

function mesh2stencil(U⃗::AbstractArray{T,3}, Idx, mesh, EOS, nh::Int=2) where {T}
    I = Tuple(Idx)[end-1:end]
    if nh == 2
        return _2halo2dstencil(U⃗, I, mesh, EOS)
    end
end

function sync_halo!(U)
    ilohi = axes(U, 2)
    jlohi = axes(U, 3)
    ilo = first(ilohi) + 2
    jlo = first(jlohi) + 2
    ihi = last(ilohi) - 2
    jhi = last(jlohi) - 2

    for j in 1:jlo-1
        for i in ilo:ihi
            for q in axes(U, 1)
                U[q, i, j] = U[q, i, jlo]
            end
        end
    end

    for j in jhi-2:last(jlohi)
        for i in ilo:ihi
            for q in axes(U, 1)
                U[q, i, j] = U[q, i, jhi]
            end
        end
    end

    for j in jlo:jhi
        for i in first(ilohi):ilo-1
            for q in axes(U, 1)
                U[q, i, j] = U[q, ilo, j]
            end
        end
    end

    for j in jlo:jhi
        for i in ihi-2:last(ilohi)
            for q in axes(U, 1)
                U[q, i, j] = U[q, ihi, j]
            end
        end
    end

    nothing
end

"""Ideal gas equation of state as a function of γ"""
struct IdealEOS{T<:AbstractFloat}
    γ::T  # polytropic index
    cᵥ::T # molar specific heat capacity of a gas at constant volume
    cₚ::T # molar specific heat capacity of a gas at constant pressure
    R::T  # specific gas constant per unit mass in cgs units
    _γ_m_1::T # γ-1
    _inv_γ_m_1::T # 1 / (γ-1)
    _γ_over_γ_m_1::T # γ / (γ-1)
end

function IdealEOS(γ::T, R=287.05u"J * kg^-1 * K^-1") where {T<:Real}
    R = ustrip(u"erg * g^-1 * K^-1", R)

    _inv_γ_m_1 = round(1 / (γ - 1), sigdigits=15)
    _γ_m_1 = 1 / _inv_γ_m_1
    _γ_over_γ_m_1 = γ / (γ - 1)

    cᵥ = R * _inv_γ_m_1
    cₚ = γ * cᵥ

    return IdealEOS(γ, cᵥ, cₚ, R, _γ_m_1, _inv_γ_m_1, _γ_over_γ_m_1)
end

@inline total_enthalpy(eos, ρ, u, v, p) = (p / ρ) * eos._γ_over_γ_m_1 + 0.5(u^2 + v^2)
@inline sound_speed(eos, ρ, p) = sqrt(eos.γ * abs(p / ρ))
@inline pressure(eos, ρ, u, v, E) = ρ * eos._γ_m_1 * (E - 0.5(u^2 + v^2))
@inline specific_total_energy(eos, ρ, u, v, p) = (p / ρ) * eos._inv_γ_m_1 + 0.5(u^2 + v^2)

@inline minmod(r) = max(0, min(r, 1))
@inline superbee(r) = max(0, min(2r, 1), min(r, 2))

function muscl(ϕᵢ₋₂, ϕᵢ₋₁, ϕᵢ, ϕᵢ₊₁, ϕᵢ₊₂, limiter)
    Δ_plus_half = ϕᵢ₊₁ - ϕᵢ
    Δ_plus_three_half = ϕᵢ₊₂ - ϕᵢ₊₁
    Δ_minus_half = ϕᵢ - ϕᵢ₋₁
    Δ_minus_three_half = ϕᵢ₋₂ - ϕᵢ₋₁

    # Δ_plus_half = Δ_plus_half * (abs(Δ_plus_half) >= ϵ)
    # Δ_plus_three_half = Δ_plus_three_half * (abs(Δ_plus_three_half) >= ϵ)
    # Δ_minus_half = Δ_minus_half * (abs(Δ_minus_half) >= ϵ)
    # Δ_minus_three_half = Δ_minus_three_half * (abs(Δ_minus_three_half) >= ϵ)

    eps = 1e-20
    rL⁺ = Δ_plus_half / (Δ_minus_half + eps)
    rR⁺ = Δ_plus_half / (Δ_plus_three_half + eps)
    rL⁻ = Δ_minus_half / (Δ_minus_three_half + eps)
    rR⁻ = Δ_minus_half / (Δ_plus_half + eps)

    lim_L⁺ = limiter(rL⁺)
    lim_R⁺ = limiter(rR⁺)
    lim_L⁻ = limiter(rL⁻)
    lim_R⁻ = limiter(rR⁻)

    ϕ_L⁺ = ϕᵢ + 0.5lim_L⁺ * Δ_plus_half          # i+1/2
    ϕ_R⁺ = ϕᵢ₊₁ - 0.5lim_R⁺ * Δ_plus_three_half  # i+1/2

    ϕ_R⁻ = ϕᵢ - 0.5lim_R⁻ * Δ_plus_half        # i-1/2
    ϕ_L⁻ = ϕᵢ₋₁ + 0.5lim_L⁻ * Δ_minus_half       # i-1/2

    return @SVector [(ϕ_L⁻, ϕ_R⁻), (ϕ_L⁺, ϕ_R⁺)]
end

@inline function pressure_based_weight_function(p_L::T, p_R::T, p_s::T, min_neighbor_press::T) where T
    if abs(p_s) < typemin(T) # p_s == 0
        f_L = zero(T)
        f_R = zero(T)
    else
        min_term = min(1.0, min_neighbor_press / min(p_L, p_R))^2

        f_L_term = (p_L / p_s) - 1.0
        f_R_term = (p_R / p_s) - 1.0

        f_L_term = f_L_term * (abs(f_L_term) >= ϵ)
        f_R_term = f_R_term * (abs(f_R_term) >= ϵ)

        f_L = f_L_term * min_term
        f_R = f_R_term * min_term
    end

    return f_L, f_R
end

@inline function discontinuity_sensor(p_L::T, p_R::T) where T
    min_term = min((p_L / p_R), (p_R / p_L))
    w = 1 - (min_term * min_term * min_term)
    w = w * (abs(w) >= ϵ)
    return w
end

@inline function pressure_split_plus(M::T; α=zero(T)) where T
    if abs(M) > 1
        P⁺ = 0.5(1 + sign(M))
    else #  |M| <= 1
        P⁺ = 0.25(M + 1)^2 * (2 - M) + α * M * (M^2 - 1)^2
    end
    return P⁺
end

@inline function pressure_split_minus(M::T; α=zero(T)) where T
    if abs(M) > 1
        P⁻ = 0.5(1 - sign(M))
    else #  |M| <= 1
        P⁻ = 0.25(M - 1)^2 * (2 + M) - α * M * (M^2 - 1)^2
    end
    return P⁻
end

@inline function mach_split_plus(M)
    if abs(M) > 1
        M⁺ = 0.5(M + abs(M))
    else #  |M| <= 1
        M⁺ = 0.25(M + 1)^2
    end
    return M⁺
end

@inline function mach_split_minus(M)
    if abs(M) > 1
        M⁻ = 0.5(M - abs(M))
    else #  |M| <= 1
        M⁻ = -0.25(M - 1)^2
    end
    return M⁻
end

@inline function transverse_component(v⃗, n̂)
    # velocity component transverse to the face edge
    v_perp = (v⃗ ⋅ n̂) .* n̂ # normal
    v_parallel = v⃗ .- v_perp # transverse
    v_parallel = v_parallel .* (abs.(v_parallel) .>= 1e-15)
    return norm(v_parallel)
end

@inline function reconstruct_sb(ϕᵢ, ϕᵢ₊₁, ϕᵢ₊₂)

    @inline superbee(r) = max(0, min(2r, 1), min(r, 2))

    Δ⁻½ = ϕᵢ - ϕᵢ₊₁
    Δ⁺½ = ϕᵢ₊₁ - ϕᵢ
    Δ⁺_three_half = ϕᵢ₊₂ - ϕᵢ₊₁

    rL = Δ⁺½ / (Δ⁻½ + ϵ)
    rR = Δ⁺½ / (Δ⁺_three_half + ϵ)

    ϕ_left = ϕᵢ + 0.5superbee(rL) * Δ⁻½
    ϕ_right = ϕᵢ₊₁ - 0.5superbee(rR) * Δ⁺_three_half

    return ϕ_left, ϕ_right
end

@inline function modified_pressure_split(Mʟ, Mʀ, Mstarᵢ, Mstarᵢ₊₁, ρᵢ, uᵢ, pᵢ, ρᵢ₊₁, uᵢ₊₁, pᵢ₊₁)

    if Mstarᵢ > 1 && Mstarᵢ₊₁ < 1 && 0 < Mstarᵢ * Mstarᵢ₊₁ < 1
        P⁻ᵢ₊₁ = max(0, min(0.5, 1 - ((ρᵢ * uᵢ * (uᵢ - uᵢ₊₁) + pᵢ) / pᵢ₊₁)))
    else
        P⁻ᵢ₊₁ = pressure_split_minus(Mʀ)
    end

    if Mstarᵢ > -1 && Mstarᵢ₊₁ < -1 && 0 < Mstarᵢ * Mstarᵢ₊₁ < 1
        P⁺ᵢ = max(0, min(0.5, 1 - (ρᵢ₊₁ * uᵢ₊₁ * (uᵢ₊₁ - uᵢ) + pᵢ₊₁) / pᵢ))
    else
        P⁺ᵢ = pressure_split_plus(Mʟ)
    end

    return P⁺ᵢ, P⁻ᵢ₊₁
end

@inline function modified_discontinuity_sensor_ξ(p̄ᵢⱼ::T, p̄ᵢ₊₁ⱼ::T, p̄ᵢ₊₁ⱼ₊₁::T, p̄ᵢ₊₁ⱼ₋₁::T, p̄ᵢⱼ₊₁::T, p̄ᵢⱼ₋₁::T) where T
    Δp = abs(p̄ᵢ₊₁ⱼ - p̄ᵢⱼ)
    Δpᵢ₊₁ = abs(p̄ᵢ₊₁ⱼ₊₁ - p̄ᵢ₊₁ⱼ₋₁)
    Δpᵢ = abs(p̄ᵢⱼ₊₁ - p̄ᵢⱼ₋₁)

    w₂ = (1 - min(1, Δp / (0.25(Δpᵢ₊₁ + Δpᵢ))))^2 * (1 - min((p̄ᵢⱼ / p̄ᵢ₊₁ⱼ), (p̄ᵢ₊₁ⱼ / p̄ᵢⱼ)))^2
    if isnan(w₂) || abs(w₂) < ϵ
        w₂ = zero(T)
    end

    return w₂
end

@inline function w2_ξ(p̄ᵢⱼ::T, p̄ᵢ₊₁ⱼ::T, p̄ᵢ₊₁ⱼ₊₁::T, p̄ᵢ₊₁ⱼ₋₁::T, p̄ᵢⱼ₊₁::T, p̄ᵢⱼ₋₁::T) where T
    denom = 0.25(p̄ᵢ₊₁ⱼ₊₁ + p̄ᵢⱼ₊₁ - p̄ᵢ₊₁ⱼ₋₁ - p̄ᵢⱼ₋₁)
    first_term = (1 - min(1, (p̄ᵢ₊₁ⱼ - p̄ᵢⱼ) / denom))^2
    second_term = (1 - min((p̄ᵢⱼ / p̄ᵢ₊₁ⱼ), (p̄ᵢ₊₁ⱼ / p̄ᵢⱼ)))^2
    w₂ = first_term * second_term
    return w₂
end

@inline function modified_discontinuity_sensor_η(p̄ᵢⱼ::T, p̄ᵢ₊₁ⱼ::T, p̄ᵢ₊₁ⱼ₊₁::T, p̄ᵢ₋₁ⱼ₊₁::T, p̄ᵢ₋₁ⱼ::T, p̄ᵢⱼ₊₁::T) where T
    Δp = abs(p̄ᵢⱼ₊₁ - p̄ᵢⱼ)
    Δpⱼ₊₁ = abs(p̄ᵢ₊₁ⱼ₊₁ - p̄ᵢ₋₁ⱼ₊₁)
    Δpⱼ = abs(p̄ᵢ₊₁ⱼ - p̄ᵢ₋₁ⱼ)

    w₂ = (1 - min(1, Δp / (0.25(Δpⱼ₊₁ + Δpⱼ))))^2 * (1 - min((p̄ᵢⱼ / p̄ᵢⱼ₊₁), (p̄ᵢⱼ₊₁ / p̄ᵢⱼ)))^2
    if isnan(w₂) || abs(w₂) < ϵ
        w₂ = zero(T)
    end

    return w₂
end

@inline function w2_η(p̄ᵢⱼ::T, p̄ᵢ₊₁ⱼ::T, p̄ᵢ₊₁ⱼ₊₁::T, p̄ᵢ₋₁ⱼ₊₁::T, p̄ᵢ₋₁ⱼ::T, p̄ᵢⱼ₊₁::T) where T
    denom = 0.25(p̄ᵢ₊₁ⱼ₊₁ + p̄ᵢ₊₁ⱼ - p̄ᵢ₋₁ⱼ₊₁ - p̄ᵢ₋₁ⱼ)
    first_term = (1 - min(1, (p̄ᵢⱼ₊₁ - p̄ᵢⱼ) / denom))^2
    second_term = (1 - min((p̄ᵢⱼ / p̄ᵢⱼ₊₁), (p̄ᵢⱼ₊₁ / p̄ᵢⱼ)))^2
    w₂ = first_term * second_term
    return w₂
end

@inline function modified_f(pʟʀ::T, pₛ::T, w₂::T) where T
    if abs(pₛ) > zero(T)
        f = ((pʟʀ / pₛ) - 1) * (1 - w₂)
    else
        f = zero(T)
    end
    return f
end

@inline function ϕ_L_half(ϕ_L, ϕ_R, ϕ_L_sb, a)

    if abs(ϕ_R - ϕ_L) < 1.0 || abs(ϕ_L_sb - ϕ_L) < ϵ
        ϕ_L_half = ϕ_L
    else
        ϕ_L_half = ϕ_L + max(0, (ϕ_R - ϕ_L) * (ϕ_L_sb - ϕ_L)) /
                         ((ϕ_R - ϕ_L) * abs(ϕ_L_sb - ϕ_L)) * min(a, 0.5 * abs(ϕ_R - ϕ_L), abs(ϕ_L_sb - ϕ_L))
    end

end

@inline function ϕ_R_half(ϕ_L, ϕ_R, ϕ_R_sb, a)
    if abs(ϕ_L - ϕ_R) < ϵ || abs(ϕ_R_sb - ϕ_R) < 1.0
        ϕ_R_half = ϕ_R
    else
        ϕ_R_half = ϕ_R + max(0, (ϕ_L - ϕ_R) * (ϕ_R_sb - ϕ_R)) /
                         ((ϕ_L - ϕ_R) * abs(ϕ_R_sb - ϕ_R)) * min(a, 0.5 * abs(ϕ_L - ϕ_R), abs(ϕ_R_sb - ϕ_R))
    end
end

function MAUSMPW⁺(n̂, 
    ρ_LR::NTuple{2,T}, u_LR::NTuple{2,T}, v_LR::NTuple{2,T}, p_LR::NTuple{2,T}, 
    ρ_LR_SB::NTuple{2,T}, u_LR_SB::NTuple{2,T}, v_LR_SB::NTuple{2,T}, p_LR_SB::NTuple{2,T},
    W⃗ᵢ, W⃗ᵢ₊₁, w₂, EOS) where {T}

    ρʟ, ρʀ = ρ_LR
    uʟ, uʀ = u_LR
    vʟ, vʀ = v_LR
    pʟ, pʀ = p_LR

    ρʟ_sb, ρʀ_sb = ρ_LR_SB
    uʟ_sb, uʀ_sb = u_LR_SB
    vʟ_sb, vʀ_sb = v_LR_SB
    pʟ_sb, pʀ_sb = p_LR_SB

    nx, ny = n̂
    v⃗ʟ = SVector{2,Float64}(uʟ, vʟ)
    v⃗ʀ = SVector{2,Float64}(uʀ, vʀ)

    ρᵢ, uᵢ, vᵢ, pᵢ = W⃗ᵢ
    ρᵢ₊₁, uᵢ₊₁, vᵢ₊₁, pᵢ₊₁ = W⃗ᵢ₊₁

    Hʟ = total_enthalpy(EOS, ρʟ, uʟ, vʟ, pʟ)
    Hʀ = total_enthalpy(EOS, ρʀ, uʀ, vʀ, pʀ)

    # velocity component normal to the face edge
    Uʟ = uʟ * nx + vʟ * ny
    Uʀ = uʀ * nx + vʀ * ny
    Uᵢ = uᵢ * nx + vᵢ * ny
    Uᵢ₊₁ = uᵢ₊₁ * nx + vᵢ₊₁ * ny

    Vʟ = transverse_component(v⃗ʟ, n̂)
    Vʀ = transverse_component(v⃗ʀ, n̂)

    # Total enthalpy normal to the edge
    H_normal = min(Hʟ - 0.5Vʟ^2, Hʀ - 0.5Vʀ^2)

    # Speed of sound normal to the edge, also like the critical sound speed across a normal shock
    cₛ = sqrt(abs(2((EOS.γ - 1) / (EOS.γ + 1)) * H_normal))

    # Interface sound speed
    if 0.5(Uʟ + Uʀ) > 0
        c½ = cₛ^2 / max(abs(Uʟ), cₛ)
    else
        c½ = cₛ^2 / max(abs(Uʀ), cₛ)
    end

    # Left/Right Mach number
    Mʟ = Uʟ / c½
    Mʀ = Uʀ / c½

    # Mach splitting functions
    Mʟ⁺ = mach_split_plus(Mʟ)
    Mʀ⁻ = mach_split_minus(Mʀ)

    # Modified functions for M-AUSMPW+
    Mstarᵢ = Uᵢ / cₛ
    Mstarᵢ₊₁ = Uᵢ₊₁ / cₛ
    Pʟ⁺, Pʀ⁻ = modified_pressure_split(Mʟ, Mʀ, Mstarᵢ, Mstarᵢ₊₁, ρᵢ, Uᵢ, pᵢ, ρᵢ₊₁, Uᵢ₊₁, pᵢ₊₁)
    pₛ = pʟ * Pʟ⁺ + pʀ * Pʀ⁻
    w₁ = discontinuity_sensor(pʟ, pʀ)
    w = max(w₁, w₂)
    fʟ = modified_f(pʟ, pₛ, w₂)
    fʀ = modified_f(pʀ, pₛ, w₂)

    # More like AUSMPW+
    # Pʀ⁻ = pressure_split_minus(Mʀ)
    # Pʟ⁺ = pressure_split_plus(Mʟ)
    # pₛ = pʟ * Pʟ⁺ + pʀ * Pʀ⁻
    # w = discontinuity_sensor(pʟ, pʀ)
    # fʟ, fʀ = pressure_based_weight_function(pʟ, pʀ, pₛ, min_neighbor_press)

    # From Eq. 24 (ii) in Ref [1]
    if Mʟ⁺ + Mʀ⁻ < 0
        M̄ʟ⁺ = Mʟ⁺ * w * (1 + fʟ)
        M̄ʀ⁻ = Mʀ⁻ + Mʟ⁺ * ((1 - w) * (1 + fʟ) - fʀ)
    else # From Eq. 24 (i) in Ref [1]
        M̄ʟ⁺ = Mʟ⁺ + Mʀ⁻ * ((1 - w) * (1 + fʀ) - fʟ)
        M̄ʀ⁻ = Mʀ⁻ * w * (1 + fʀ)
    end
    M̄ʟ⁺ = M̄ʟ⁺ * (abs(M̄ʟ⁺) >= 1e-15)
    M̄ʀ⁻ = M̄ʀ⁻ * (abs(M̄ʀ⁻) >= 1e-15)

    a = 1 - min(1, max(abs(Mʟ), abs(Mʀ)))^2
    ρʟ½ = ϕ_L_half(ρʟ, ρʀ, ρʟ_sb, a)
    uʟ½ = ϕ_L_half(uʟ, uʀ, uʟ_sb, a)
    vʟ½ = ϕ_L_half(vʟ, vʀ, vʟ_sb, a)
    pʟ½ = ϕ_L_half(pʟ, pʀ, pʟ_sb, a)

    ρʀ½ = ϕ_R_half(ρʟ, ρʀ, ρʀ_sb, a)
    uʀ½ = ϕ_R_half(uʟ, uʀ, uʀ_sb, a)
    vʀ½ = ϕ_R_half(vʟ, vʀ, vʀ_sb, a)
    pʀ½ = ϕ_R_half(pʟ, pʀ, pʀ_sb, a)

    # mass fluxes
    ṁʟ = M̄ʟ⁺ * c½ * ρʟ½
    ṁʀ = M̄ʀ⁻ * c½ * ρʀ½

    Hʟ½ = total_enthalpy(EOS, ρʟ½, uʟ½, vʟ½, pʟ½)
    Hʀ½ = total_enthalpy(EOS, ρʀ½, uʀ½, vʀ½, pʀ½)

    ρflux = ṁʟ + ṁʀ
    ρuflux = (ṁʟ * uʟ½) + (ṁʀ * uʀ½) + ((Pʟ⁺ * nx * pʟ½) + (Pʀ⁻ * nx * pʀ½))
    ρvflux = (ṁʟ * vʟ½) + (ṁʀ * vʀ½) + ((Pʟ⁺ * ny * pʟ½) + (Pʀ⁻ * ny * pʀ½))
    Eflux = (ṁʟ * Hʟ½) + (ṁʀ * Hʀ½)

    ρflux = ρflux * (abs(ρflux) >= ϵ)
    ρuflux = ρuflux * (abs(ρuflux) >= ϵ)
    ρvflux = ρvflux * (abs(ρvflux) >= ϵ)
    Eflux = Eflux * (abs(Eflux) >= ϵ)

    return SVector{4,Float64}(ρflux, ρuflux, ρvflux, Eflux)
end

function get_me_block(U,i,j)
    ublk = @SArray [SVector{4,Float64}(view(U,:,i+io,j+jo)) for jo in -2:2, io in -2:2]
end

function ∂U∂tvec(::M_AUSMPWPlus2D, stencil::Stencil9Point, recon::F1, limiter::F2) where {F1,F2}
    U⃗ = stencil.U⃗
    EOS = stencil.EOS

    # Conserved to primitive variables
    # W⃗ = cons2prim.(Ref(EOS), ρ, ρu, ρv, ρE)
    W⃗ = cons2prim_vec.(Ref(EOS),U⃗)

    W⃗ᵢ₋₂ = W⃗[1, 3]
    W⃗ⱼ₋₂ = W⃗[3, 1]
    W⃗ᵢ₋₁ = W⃗[2, 3]
    W⃗ⱼ₋₁ = W⃗[3, 2]
    W⃗ᵢ = W⃗[3, 3]
    W⃗ⱼ = W⃗[3, 3]
    W⃗ᵢ₊₁ = W⃗[4, 3]
    W⃗ⱼ₊₁ = W⃗[3, 4]
    W⃗ᵢ₊₂ = W⃗[5, 3]
    W⃗ⱼ₊₂ = W⃗[3, 5]

    # Reconstruct the left/right states
    ρᴸᴿᵢ, uᴸᴿᵢ, vᴸᴿᵢ, pᴸᴿᵢ = recon.(W⃗ᵢ₋₂, W⃗ᵢ₋₁, W⃗ᵢ, W⃗ᵢ₊₁, W⃗ᵢ₊₂, limiter) # i-1/2, i+1/2
    ρᴸᴿⱼ, uᴸᴿⱼ, vᴸᴿⱼ, pᴸᴿⱼ = recon.(W⃗ⱼ₋₂, W⃗ⱼ₋₁, W⃗ⱼ, W⃗ⱼ₊₁, W⃗ⱼ₊₂, limiter) # j-1/2, j+1/2

    ρᴸᴿᵢSB, uᴸᴿᵢSB, vᴸᴿᵢSB, pᴸᴿᵢSB = recon.(W⃗ᵢ₋₂, W⃗ᵢ₋₁, W⃗ᵢ, W⃗ᵢ₊₁, W⃗ᵢ₊₂, superbee) # i-1/2, i+1/2
    ρᴸᴿⱼSB, uᴸᴿⱼSB, vᴸᴿⱼSB, pᴸᴿⱼSB = recon.(W⃗ⱼ₋₂, W⃗ⱼ₋₁, W⃗ⱼ, W⃗ⱼ₊₁, W⃗ⱼ₊₂, superbee) # j-1/2, j+1/2

    ρᴸᴿᵢ⁻, ρᴸᴿᵢ⁺ = ρᴸᴿᵢ
    uᴸᴿᵢ⁻, uᴸᴿᵢ⁺ = uᴸᴿᵢ
    vᴸᴿᵢ⁻, vᴸᴿᵢ⁺ = vᴸᴿᵢ
    pᴸᴿᵢ⁻, pᴸᴿᵢ⁺ = pᴸᴿᵢ

    ρᴸᴿⱼ⁻, ρᴸᴿⱼ⁺ = ρᴸᴿⱼ
    uᴸᴿⱼ⁻, uᴸᴿⱼ⁺ = uᴸᴿⱼ
    vᴸᴿⱼ⁻, vᴸᴿⱼ⁺ = vᴸᴿⱼ
    pᴸᴿⱼ⁻, pᴸᴿⱼ⁺ = pᴸᴿⱼ

    ρᴸᴿᵢ⁻SB, ρᴸᴿᵢ⁺SB = ρᴸᴿᵢSB
    uᴸᴿᵢ⁻SB, uᴸᴿᵢ⁺SB = uᴸᴿᵢSB
    vᴸᴿᵢ⁻SB, vᴸᴿᵢ⁺SB = vᴸᴿᵢSB
    pᴸᴿᵢ⁻SB, pᴸᴿᵢ⁺SB = pᴸᴿᵢSB

    ρᴸᴿⱼ⁻SB, ρᴸᴿⱼ⁺SB = ρᴸᴿⱼSB
    uᴸᴿⱼ⁻SB, uᴸᴿⱼ⁺SB = uᴸᴿⱼSB
    vᴸᴿⱼ⁻SB, vᴸᴿⱼ⁺SB = vᴸᴿⱼSB
    pᴸᴿⱼ⁻SB, pᴸᴿⱼ⁺SB = pᴸᴿⱼSB

    W⃗ᵢc = (W⃗ᵢ₋₁, W⃗ᵢ) # i average state
    W⃗ᵢc1 = (W⃗ᵢ, W⃗ᵢ₊₁) # i+1 average state

    W⃗ⱼc = (W⃗ⱼ₋₁, W⃗ᵢ) # j average state
    W⃗ⱼc1 = (W⃗ᵢ, W⃗ⱼ₊₁) # j+1 average state

    n̂1 = -stencil.n̂[:, 1]
    n̂2 = stencil.n̂[:, 2]
    n̂3 = stencil.n̂[:, 3]
    n̂4 = -stencil.n̂[:, 4]

    # i⁻ = 2
    # i⁺ = 3
    # j = 3
    # p0ᵢ = (W⃗[i⁻, j][4],     W⃗[i⁺, j][4])
    # p1ᵢ = (W⃗[i⁻+1, j][4],   W⃗[i⁺+1, j][4])
    # p2ᵢ = (W⃗[i⁻+1, j+1][4], W⃗[i⁺+1, j+1][4])
    # p3ᵢ = (W⃗[i⁻+1, j-1][4], W⃗[i⁺+1, j-1][4])
    # p4ᵢ = (W⃗[i⁻, j+1][4],   W⃗[i⁺, j+1][4])
    # p5ᵢ = (W⃗[i⁻, j-1][4],   W⃗[i⁺, j-1][4])
    ωᵢ = (1.0, 1.0) #modified_discontinuity_sensor_ξ.(p0ᵢ, p1ᵢ, p2ᵢ, p3ᵢ, p4ᵢ, p5ᵢ)

    # i = 3
    # j⁻ = 2
    # j⁺ = 3
    # p0ⱼ = (W⃗[i, j⁻][4],     W⃗[i,   j⁺][4])
    # p1ⱼ = (W⃗[i+1, j⁻][4],   W⃗[i+1, j⁺][4])
    # p2ⱼ = (W⃗[i+1, j⁻+1][4], W⃗[i+1, j⁺+1][4])
    # p3ⱼ = (W⃗[i-1, j⁻+1][4], W⃗[i-1, j⁺+1][4])
    # p4ⱼ = (W⃗[i-1, j⁻][4],   W⃗[i-1, j⁺][4])
    # p5ⱼ = (W⃗[i, j⁻+1][4],   W⃗[i,   j⁺+1][4])
    ωⱼ = (1.0, 1.0) #modified_discontinuity_sensor_η.(p0ⱼ, p1ⱼ, p2ⱼ, p3ⱼ, p4ⱼ, p5ⱼ)

    F⃗ᵢ_m_half = MAUSMPW⁺(
        n̂4, 
        ρᴸᴿᵢ⁻, uᴸᴿᵢ⁻, vᴸᴿᵢ⁻, pᴸᴿᵢ⁻, 
        ρᴸᴿᵢ⁻SB, uᴸᴿᵢ⁻SB, vᴸᴿᵢ⁻SB, pᴸᴿᵢ⁻SB, 
        W⃗ᵢc[1], W⃗ᵢc1[1], ωᵢ[1], EOS
    )

    F⃗ⱼ_m_half = MAUSMPW⁺(
        n̂1, 
        ρᴸᴿⱼ⁻, uᴸᴿⱼ⁻, vᴸᴿⱼ⁻, pᴸᴿⱼ⁻, 
        ρᴸᴿⱼ⁻SB, uᴸᴿⱼ⁻SB, vᴸᴿⱼ⁻SB, pᴸᴿⱼ⁻SB, 
        W⃗ⱼc[1], W⃗ⱼc1[1], ωⱼ[1], EOS
    )

    F⃗ᵢ_p_half = MAUSMPW⁺(
        n̂2, 
        ρᴸᴿᵢ⁺, uᴸᴿᵢ⁺, vᴸᴿᵢ⁺, pᴸᴿᵢ⁺, 
        ρᴸᴿᵢ⁺SB, uᴸᴿᵢ⁺SB, vᴸᴿᵢ⁺SB, pᴸᴿᵢ⁺SB, 
        W⃗ᵢc[2], W⃗ᵢc1[2], ωᵢ[2], EOS
    )

    F⃗ⱼ_p_half = MAUSMPW⁺(
        n̂3, 
        ρᴸᴿⱼ⁺, uᴸᴿⱼ⁺, vᴸᴿⱼ⁺, pᴸᴿⱼ⁺, 
        ρᴸᴿⱼ⁺SB, uᴸᴿⱼ⁺SB, vᴸᴿⱼ⁺SB, pᴸᴿⱼ⁺SB, 
        W⃗ⱼc[2], W⃗ⱼc1[2], ωⱼ[2], EOS
    )

    dUdt = (
        -(F⃗ᵢ_p_half * stencil.ΔS[2] - F⃗ᵢ_m_half * stencil.ΔS[4] +
          F⃗ⱼ_p_half * stencil.ΔS[3] - F⃗ⱼ_m_half * stencil.ΔS[1]) / stencil.Ω
    ) 
    + stencil.S⃗

    return dUdt
end

@inbounds function ∂U∂t(::M_AUSMPWPlus2D, stencil::Stencil9Point, recon::F1, limiter::F2) where {F1,F2}
    U⃗ = stencil.U⃗
    EOS = stencil.EOS

    # Conserved to primitive variables
    # W⃗ = cons2prim.(Ref(EOS), ρ, ρu, ρv, ρE)
    W⃗ = cons2prim.(Ref(EOS),U⃗[1,:,:],U⃗[2,:,:],U⃗[3,:,:],U⃗[4,:,:])

    W⃗ᵢ₋₂ = W⃗[1, 3]
    W⃗ⱼ₋₂ = W⃗[3, 1]
    W⃗ᵢ₋₁ = W⃗[2, 3]
    W⃗ⱼ₋₁ = W⃗[3, 2]
    W⃗ᵢ = W⃗[3, 3]
    W⃗ⱼ = W⃗[3, 3]
    W⃗ᵢ₊₁ = W⃗[4, 3]
    W⃗ⱼ₊₁ = W⃗[3, 4]
    W⃗ᵢ₊₂ = W⃗[5, 3]
    W⃗ⱼ₊₂ = W⃗[3, 5]

    # Reconstruct the left/right states
    ρᴸᴿᵢ, uᴸᴿᵢ, vᴸᴿᵢ, pᴸᴿᵢ = recon.(W⃗ᵢ₋₂, W⃗ᵢ₋₁, W⃗ᵢ, W⃗ᵢ₊₁, W⃗ᵢ₊₂, limiter) # i-1/2, i+1/2
    ρᴸᴿⱼ, uᴸᴿⱼ, vᴸᴿⱼ, pᴸᴿⱼ = recon.(W⃗ⱼ₋₂, W⃗ⱼ₋₁, W⃗ⱼ, W⃗ⱼ₊₁, W⃗ⱼ₊₂, limiter) # j-1/2, j+1/2

    ρᴸᴿᵢSB, uᴸᴿᵢSB, vᴸᴿᵢSB, pᴸᴿᵢSB = recon.(W⃗ᵢ₋₂, W⃗ᵢ₋₁, W⃗ᵢ, W⃗ᵢ₊₁, W⃗ᵢ₊₂, superbee) # i-1/2, i+1/2
    ρᴸᴿⱼSB, uᴸᴿⱼSB, vᴸᴿⱼSB, pᴸᴿⱼSB = recon.(W⃗ⱼ₋₂, W⃗ⱼ₋₁, W⃗ⱼ, W⃗ⱼ₊₁, W⃗ⱼ₊₂, superbee) # j-1/2, j+1/2

    ρᴸᴿᵢ⁻, ρᴸᴿᵢ⁺ = ρᴸᴿᵢ
    uᴸᴿᵢ⁻, uᴸᴿᵢ⁺ = uᴸᴿᵢ
    vᴸᴿᵢ⁻, vᴸᴿᵢ⁺ = vᴸᴿᵢ
    pᴸᴿᵢ⁻, pᴸᴿᵢ⁺ = pᴸᴿᵢ

    ρᴸᴿⱼ⁻, ρᴸᴿⱼ⁺ = ρᴸᴿⱼ
    uᴸᴿⱼ⁻, uᴸᴿⱼ⁺ = uᴸᴿⱼ
    vᴸᴿⱼ⁻, vᴸᴿⱼ⁺ = vᴸᴿⱼ
    pᴸᴿⱼ⁻, pᴸᴿⱼ⁺ = pᴸᴿⱼ

    ρᴸᴿᵢ⁻SB, ρᴸᴿᵢ⁺SB = ρᴸᴿᵢSB
    uᴸᴿᵢ⁻SB, uᴸᴿᵢ⁺SB = uᴸᴿᵢSB
    vᴸᴿᵢ⁻SB, vᴸᴿᵢ⁺SB = vᴸᴿᵢSB
    pᴸᴿᵢ⁻SB, pᴸᴿᵢ⁺SB = pᴸᴿᵢSB

    ρᴸᴿⱼ⁻SB, ρᴸᴿⱼ⁺SB = ρᴸᴿⱼSB
    uᴸᴿⱼ⁻SB, uᴸᴿⱼ⁺SB = uᴸᴿⱼSB
    vᴸᴿⱼ⁻SB, vᴸᴿⱼ⁺SB = vᴸᴿⱼSB
    pᴸᴿⱼ⁻SB, pᴸᴿⱼ⁺SB = pᴸᴿⱼSB

    W⃗ᵢc = (W⃗ᵢ₋₁, W⃗ᵢ) # i average state
    W⃗ᵢc1 = (W⃗ᵢ, W⃗ᵢ₊₁) # i+1 average state

    W⃗ⱼc = (W⃗ⱼ₋₁, W⃗ᵢ) # j average state
    W⃗ⱼc1 = (W⃗ᵢ, W⃗ⱼ₊₁) # j+1 average state

    n̂1 = -stencil.n̂[:, 1]
    n̂2 = stencil.n̂[:, 2]
    n̂3 = stencil.n̂[:, 3]
    n̂4 = -stencil.n̂[:, 4]

    # i⁻ = 2
    # i⁺ = 3
    # j = 3
    # p0ᵢ = (W⃗[i⁻, j][4],     W⃗[i⁺, j][4])
    # p1ᵢ = (W⃗[i⁻+1, j][4],   W⃗[i⁺+1, j][4])
    # p2ᵢ = (W⃗[i⁻+1, j+1][4], W⃗[i⁺+1, j+1][4])
    # p3ᵢ = (W⃗[i⁻+1, j-1][4], W⃗[i⁺+1, j-1][4])
    # p4ᵢ = (W⃗[i⁻, j+1][4],   W⃗[i⁺, j+1][4])
    # p5ᵢ = (W⃗[i⁻, j-1][4],   W⃗[i⁺, j-1][4])
    ωᵢ = (1.0, 1.0) #modified_discontinuity_sensor_ξ.(p0ᵢ, p1ᵢ, p2ᵢ, p3ᵢ, p4ᵢ, p5ᵢ)

    # i = 3
    # j⁻ = 2
    # j⁺ = 3
    # p0ⱼ = (W⃗[i, j⁻][4],     W⃗[i,   j⁺][4])
    # p1ⱼ = (W⃗[i+1, j⁻][4],   W⃗[i+1, j⁺][4])
    # p2ⱼ = (W⃗[i+1, j⁻+1][4], W⃗[i+1, j⁺+1][4])
    # p3ⱼ = (W⃗[i-1, j⁻+1][4], W⃗[i-1, j⁺+1][4])
    # p4ⱼ = (W⃗[i-1, j⁻][4],   W⃗[i-1, j⁺][4])
    # p5ⱼ = (W⃗[i, j⁻+1][4],   W⃗[i,   j⁺+1][4])
    ωⱼ = (1.0, 1.0) #modified_discontinuity_sensor_η.(p0ⱼ, p1ⱼ, p2ⱼ, p3ⱼ, p4ⱼ, p5ⱼ)

    F⃗ᵢ_m_half = MAUSMPW⁺(
        n̂4, 
        ρᴸᴿᵢ⁻, uᴸᴿᵢ⁻, vᴸᴿᵢ⁻, pᴸᴿᵢ⁻, 
        ρᴸᴿᵢ⁻SB, uᴸᴿᵢ⁻SB, vᴸᴿᵢ⁻SB, pᴸᴿᵢ⁻SB, 
        W⃗ᵢc[1], W⃗ᵢc1[1], ωᵢ[1], EOS
    )

    F⃗ⱼ_m_half = MAUSMPW⁺(
        n̂1, 
        ρᴸᴿⱼ⁻, uᴸᴿⱼ⁻, vᴸᴿⱼ⁻, pᴸᴿⱼ⁻, 
        ρᴸᴿⱼ⁻SB, uᴸᴿⱼ⁻SB, vᴸᴿⱼ⁻SB, pᴸᴿⱼ⁻SB, 
        W⃗ⱼc[1], W⃗ⱼc1[1], ωⱼ[1], EOS
    )

    F⃗ᵢ_p_half = MAUSMPW⁺(
        n̂2, 
        ρᴸᴿᵢ⁺, uᴸᴿᵢ⁺, vᴸᴿᵢ⁺, pᴸᴿᵢ⁺, 
        ρᴸᴿᵢ⁺SB, uᴸᴿᵢ⁺SB, vᴸᴿᵢ⁺SB, pᴸᴿᵢ⁺SB, 
        W⃗ᵢc[2], W⃗ᵢc1[2], ωᵢ[2], EOS
    )

    F⃗ⱼ_p_half = MAUSMPW⁺(
        n̂3, 
        ρᴸᴿⱼ⁺, uᴸᴿⱼ⁺, vᴸᴿⱼ⁺, pᴸᴿⱼ⁺, 
        ρᴸᴿⱼ⁺SB, uᴸᴿⱼ⁺SB, vᴸᴿⱼ⁺SB, pᴸᴿⱼ⁺SB, 
        W⃗ⱼc[2], W⃗ⱼc1[2], ωⱼ[2], EOS
    )

    dUdt = (
        -(F⃗ᵢ_p_half * stencil.ΔS[2] - F⃗ᵢ_m_half * stencil.ΔS[4] +
          F⃗ⱼ_p_half * stencil.ΔS[3] - F⃗ⱼ_m_half * stencil.ΔS[1]) / stencil.Ω
    ) 
    + stencil.S⃗

    return dUdt
end

@inbounds function ∂U∂t(::M_AUSMPWPlus2D, stencil::Stencil9PointSplit, recon::F1, limiter::F2) where {F1,F2}
    U⃗ᵢ = stencil.U⃗ᵢ
    U⃗ⱼ = stencil.U⃗ⱼ
    U⃗ᵢ₊₁ⱼ₊₁ = stencil.U⃗ᵢ₊₁ⱼ₊₁
    U⃗ᵢ₊₁ⱼ₋₁ = stencil.U⃗ᵢ₊₁ⱼ₋₁
    U⃗ᵢ₋₁ⱼ₊₁ = stencil.U⃗ᵢ₋₁ⱼ₊₁
    U⃗ᵢ₋₁ⱼ₋₁ = stencil.U⃗ᵢ₋₁ⱼ₋₁

    EOS = stencil.EOS

    # Conserved to primitive variables
    W⃗ᵢ = cons2prim.(Ref(EOS), U⃗ᵢ[1,:], U⃗ᵢ[2,:], U⃗ᵢ[3,:], U⃗ᵢ[4,:])
    W⃗ⱼ = cons2prim.(Ref(EOS), U⃗ⱼ[1,:], U⃗ⱼ[2,:], U⃗ⱼ[3,:], U⃗ⱼ[4,:])
    W⃗ᵢ₊₁ⱼ₊₁ = cons2prim.(Ref(EOS), U⃗ᵢ₊₁ⱼ₊₁[1], U⃗ᵢ₊₁ⱼ₊₁[2], U⃗ᵢ₊₁ⱼ₊₁[3], U⃗ᵢ₊₁ⱼ₊₁[4])
    W⃗ᵢ₊₁ⱼ₋₁ = cons2prim.(Ref(EOS), U⃗ᵢ₊₁ⱼ₋₁[1], U⃗ᵢ₊₁ⱼ₋₁[2], U⃗ᵢ₊₁ⱼ₋₁[3], U⃗ᵢ₊₁ⱼ₋₁[4])
    W⃗ᵢ₋₁ⱼ₊₁ = cons2prim.(Ref(EOS), U⃗ᵢ₋₁ⱼ₊₁[1], U⃗ᵢ₋₁ⱼ₊₁[2], U⃗ᵢ₋₁ⱼ₊₁[3], U⃗ᵢ₋₁ⱼ₊₁[4])
    W⃗ᵢ₋₁ⱼ₋₁ = cons2prim.(Ref(EOS), U⃗ᵢ₋₁ⱼ₋₁[1], U⃗ᵢ₋₁ⱼ₋₁[2], U⃗ᵢ₋₁ⱼ₋₁[3], U⃗ᵢ₋₁ⱼ₋₁[4])

    # Reconstruct the left/right states
    ρᴸᴿᵢ, uᴸᴿᵢ, vᴸᴿᵢ, pᴸᴿᵢ = recon.(W⃗ᵢ[1], W⃗ᵢ[2], W⃗ᵢ[3], W⃗ᵢ[4], W⃗ᵢ[5], limiter) # i-1/2, i+1/2
    ρᴸᴿⱼ, uᴸᴿⱼ, vᴸᴿⱼ, pᴸᴿⱼ = recon.(W⃗ⱼ[1], W⃗ⱼ[2], W⃗ⱼ[3], W⃗ⱼ[4], W⃗ⱼ[5], limiter) # j-1/2, j+1/2

    ρᴸᴿᵢSB, uᴸᴿᵢSB, vᴸᴿᵢSB, pᴸᴿᵢSB = recon.(W⃗ᵢ[1], W⃗ᵢ[2], W⃗ᵢ[3], W⃗ᵢ[4], W⃗ᵢ[5], superbee) # i-1/2, i+1/2
    ρᴸᴿⱼSB, uᴸᴿⱼSB, vᴸᴿⱼSB, pᴸᴿⱼSB = recon.(W⃗ⱼ[1], W⃗ⱼ[2], W⃗ⱼ[3], W⃗ⱼ[4], W⃗ⱼ[5], superbee) # j-1/2, j+1/2

    ρᴸᴿᵢ⁻, ρᴸᴿᵢ⁺ = ρᴸᴿᵢ
    uᴸᴿᵢ⁻, uᴸᴿᵢ⁺ = uᴸᴿᵢ
    vᴸᴿᵢ⁻, vᴸᴿᵢ⁺ = vᴸᴿᵢ
    pᴸᴿᵢ⁻, pᴸᴿᵢ⁺ = pᴸᴿᵢ

    ρᴸᴿⱼ⁻, ρᴸᴿⱼ⁺ = ρᴸᴿⱼ
    uᴸᴿⱼ⁻, uᴸᴿⱼ⁺ = uᴸᴿⱼ
    vᴸᴿⱼ⁻, vᴸᴿⱼ⁺ = vᴸᴿⱼ
    pᴸᴿⱼ⁻, pᴸᴿⱼ⁺ = pᴸᴿⱼ

    ρᴸᴿᵢ⁻SB, ρᴸᴿᵢ⁺SB = ρᴸᴿᵢSB
    uᴸᴿᵢ⁻SB, uᴸᴿᵢ⁺SB = uᴸᴿᵢSB
    vᴸᴿᵢ⁻SB, vᴸᴿᵢ⁺SB = vᴸᴿᵢSB
    pᴸᴿᵢ⁻SB, pᴸᴿᵢ⁺SB = pᴸᴿᵢSB

    ρᴸᴿⱼ⁻SB, ρᴸᴿⱼ⁺SB = ρᴸᴿⱼSB
    uᴸᴿⱼ⁻SB, uᴸᴿⱼ⁺SB = uᴸᴿⱼSB
    vᴸᴿⱼ⁻SB, vᴸᴿⱼ⁺SB = vᴸᴿⱼSB
    pᴸᴿⱼ⁻SB, pᴸᴿⱼ⁺SB = pᴸᴿⱼSB

    W⃗ᵢc =  (W⃗ᵢ[2], W⃗ᵢ[3]) # i average state
    W⃗ᵢc1 = (W⃗ᵢ[3], W⃗ᵢ[4]) # i+1 average state

    W⃗ⱼc =  (W⃗ⱼ[2], W⃗ⱼ[3]) # j average state
    W⃗ⱼc1 = (W⃗ⱼ[3], W⃗ⱼ[4]) # j+1 average state

    n̂1 = -stencil.n̂[:, 1]
    n̂2 = stencil.n̂[:, 2]
    n̂3 = stencil.n̂[:, 3]
    n̂4 = -stencil.n̂[:, 4]

    # i⁻ = 2
    # i⁺ = 3
    # j = 3
    # p0ᵢ = (W⃗[i⁻, j][4],     W⃗[i⁺, j][4])
    # p1ᵢ = (W⃗[i⁻+1, j][4],   W⃗[i⁺+1, j][4])
    # p2ᵢ = (W⃗[i⁻+1, j+1][4], W⃗[i⁺+1, j+1][4])
    # p3ᵢ = (W⃗[i⁻+1, j-1][4], W⃗[i⁺+1, j-1][4])
    # p4ᵢ = (W⃗[i⁻, j+1][4],   W⃗[i⁺, j+1][4])
    # p5ᵢ = (W⃗[i⁻, j-1][4],   W⃗[i⁺, j-1][4])

    ωᵢ = (1.0, 1.0) #modified_discontinuity_sensor_ξ.(p0ᵢ, p1ᵢ, p2ᵢ, p3ᵢ, p4ᵢ, p5ᵢ)

    # i = 3
    # j⁻ = 2
    # j⁺ = 3
    # p0ⱼ = (W⃗[i, j⁻][4],     W⃗[i,   j⁺][4])
    # p1ⱼ = (W⃗[i+1, j⁻][4],   W⃗[i+1, j⁺][4])
    # p2ⱼ = (W⃗[i+1, j⁻+1][4], W⃗[i+1, j⁺+1][4])
    # p3ⱼ = (W⃗[i-1, j⁻+1][4], W⃗[i-1, j⁺+1][4])
    # p4ⱼ = (W⃗[i-1, j⁻][4],   W⃗[i-1, j⁺][4])
    # p5ⱼ = (W⃗[i, j⁻+1][4],   W⃗[i,   j⁺+1][4])
    ωⱼ = (1.0, 1.0)# modified_discontinuity_sensor_η.(p0ⱼ, p1ⱼ, p2ⱼ, p3ⱼ, p4ⱼ, p5ⱼ)

    F⃗ᵢ_m_half = MAUSMPW⁺(
        n̂4, 
        ρᴸᴿᵢ⁻, uᴸᴿᵢ⁻, vᴸᴿᵢ⁻, pᴸᴿᵢ⁻, 
        ρᴸᴿᵢ⁻SB, uᴸᴿᵢ⁻SB, vᴸᴿᵢ⁻SB, pᴸᴿᵢ⁻SB, 
        W⃗ᵢc[1], W⃗ᵢc1[1], ωᵢ[1], EOS
    )

    F⃗ⱼ_m_half = MAUSMPW⁺(
        n̂1, 
        ρᴸᴿⱼ⁻, uᴸᴿⱼ⁻, vᴸᴿⱼ⁻, pᴸᴿⱼ⁻, 
        ρᴸᴿⱼ⁻SB, uᴸᴿⱼ⁻SB, vᴸᴿⱼ⁻SB, pᴸᴿⱼ⁻SB, 
        W⃗ⱼc[1], W⃗ⱼc1[1], ωⱼ[1], EOS
    )

    F⃗ᵢ_p_half = MAUSMPW⁺(
        n̂2, 
        ρᴸᴿᵢ⁺, uᴸᴿᵢ⁺, vᴸᴿᵢ⁺, pᴸᴿᵢ⁺, 
        ρᴸᴿᵢ⁺SB, uᴸᴿᵢ⁺SB, vᴸᴿᵢ⁺SB, pᴸᴿᵢ⁺SB, 
        W⃗ᵢc[2], W⃗ᵢc1[2], ωᵢ[2], EOS
    )

    F⃗ⱼ_p_half = MAUSMPW⁺(
        n̂3, 
        ρᴸᴿⱼ⁺, uᴸᴿⱼ⁺, vᴸᴿⱼ⁺, pᴸᴿⱼ⁺, 
        ρᴸᴿⱼ⁺SB, uᴸᴿⱼ⁺SB, vᴸᴿⱼ⁺SB, pᴸᴿⱼ⁺SB, 
        W⃗ⱼc[2], W⃗ⱼc1[2], ωⱼ[2], EOS
    )

    dUdt = (
        -(F⃗ᵢ_p_half * stencil.ΔS[2] - F⃗ᵢ_m_half * stencil.ΔS[4] +
          F⃗ⱼ_p_half * stencil.ΔS[3] - F⃗ⱼ_m_half * stencil.ΔS[1]) / stencil.Ω
    ) 
    + stencil.S⃗

    return dUdt
end

@inbounds function SSPRK3_init_wrong(SS::SSPRK3Integrator, U⃗n::AbstractArray{T}, 
    riemann_solver, mesh, EOS, dt) where T

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
    @batch for j in jlo:jhi
        for i in ilo:ihi
            U⁽ⁿ⁾ = SVector{4,Float64}(view(U⃗n, :, i, j))

            U_local = SArray{Tuple{4,5,5},Float64}(view(U⃗n,:,i-2:i+2,j-2:j+2))

            S⃗ = @SVector zeros(4)
            n̂ = SMatrix{2,4}(view(norms,:,:,i,j))
            ΔS = SVector{4,T}(view(ΔS_face,:,i,j))
            Ω = vol[i,j]
            stencil = Stencil9Point(U_local, S⃗, n̂, ΔS, Ω, EOS)
            ∂U⁽ⁿ⁾∂t = ∂U∂t(riemann_solver, stencil, muscl, minmod)
            U⁽¹⁾ = U⁽ⁿ⁾ + ∂U⁽ⁿ⁾∂t * dt
            U⃗1[:,i,j] = U⁽¹⁾
        end
    end

    sync_halo!(U⃗1)

    # Stage 2
    @batch for j in jlo:jhi
        for i in ilo:ihi
            U⁽¹⁾ = SVector{4,Float64}(view(U⃗1, :, i, j))
            U⁽ⁿ⁾ = SVector{4,Float64}(view(U⃗n, :, i, j))

            U_local = SArray{Tuple{4,5,5},Float64}(view(U⃗1,:,i-2:i+2,j-2:j+2))

            S⃗ = @SVector zeros(4)
            n̂ = SMatrix{2,4}(view(norms,:,:,i,j))
            ΔS = SVector{4,T}(view(ΔS_face,:,i,j))
            Ω = vol[i,j]
            
            stencil = Stencil9Point(U_local, S⃗, n̂, ΔS, Ω, EOS)

            ∂U⁽¹⁾∂t = ∂U∂t(riemann_solver, stencil, muscl, minmod)
            U⁽²⁾ = 0.75U⁽ⁿ⁾ + 0.25U⁽¹⁾ + ∂U⁽¹⁾∂t * 0.25dt
            U⃗2[:,i,j] = U⁽²⁾
        end
    end

    sync_halo!(U⃗2)

    # Stage 3
    @batch for j in jlo:jhi
        for i in ilo:ihi
            U⁽²⁾ = SVector{4,Float64}(view(U⃗2, :, i, j))
            U⁽ⁿ⁾ = SVector{4,Float64}(view(U⃗n, :, i, j))

            U_local = SArray{Tuple{4,5,5},Float64}(view(U⃗2,:,i-2:i+2,j-2:j+2))

            S⃗ = @SVector zeros(4)
            n̂ = SMatrix{2,4}(view(norms,:,:,i,j))
            ΔS = SVector{4,T}(view(ΔS_face,:,i,j))
            Ω = vol[i,j]
            
            stencil = Stencil9Point(U_local, S⃗, n̂, ΔS, Ω, EOS)

            ∂U⁽²⁾∂t = ∂U∂t(riemann_solver, stencil, muscl, minmod)
            U⁽ⁿ⁺¹⁾ = (1 / 3) * U⁽ⁿ⁾ + (2 / 3) * U⁽²⁾ + ∂U⁽²⁾∂t * (2 / 3) * dt
            U⃗3[:, i, j] = U⁽ⁿ⁺¹⁾
        end
    end

    sync_halo!(U⃗3)

    return nothing
end

@inbounds function SSPRK3_vec(SS::SSPRK3Integrator, U⃗n::AbstractArray{T}, 
    riemann_solver, mesh, EOS, dt) where T

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

    blocks = SS.blocks

    # @batch needs to work with isbits types -- somehow see if I can make mesh contain only isbits stuff
    # Stage 1
    @batch for j in jlo:jhi
        for i in ilo:ihi
            U⁽ⁿ⁾ = SVector{4,Float64}(view(U⃗n, :, i, j))

            U_local = get_me_block(U⃗n,i,j)
            S⃗ = @SVector zeros(4)
            n̂ = SMatrix{2,4}(view(norms,:,:,i,j))
            ΔS = SVector{4,T}(view(ΔS_face,:,i,j))
            Ω = vol[i,j]
            stencil = Stencil9Point(U_local, S⃗, n̂, ΔS, Ω, EOS)
            ∂U⁽ⁿ⁾∂t = ∂U∂tvec(riemann_solver, stencil, muscl, minmod)
            U⁽¹⁾ = U⁽ⁿ⁾ + ∂U⁽ⁿ⁾∂t * dt
            U⃗1[:,i,j] = U⁽¹⁾
        end
    end

    sync_halo!(U⃗1)

    # Stage 2
    @batch for j in jlo:jhi
        for i in ilo:ihi
            U⁽¹⁾ = SVector{4,Float64}(view(U⃗1, :, i, j))
            U⁽ⁿ⁾ = SVector{4,Float64}(view(U⃗n, :, i, j))

            U_local =  get_me_block(U⃗1,i,j)

            S⃗ = @SVector zeros(4)
            n̂ = SMatrix{2,4}(view(norms,:,:,i,j))
            ΔS = SVector{4,T}(view(ΔS_face,:,i,j))
            Ω = vol[i,j]
            
            stencil = Stencil9Point(U_local, S⃗, n̂, ΔS, Ω, EOS)

            ∂U⁽¹⁾∂t = ∂U∂tvec(riemann_solver, stencil, muscl, minmod)
            U⁽²⁾ = 0.75U⁽ⁿ⁾ + 0.25U⁽¹⁾ + ∂U⁽¹⁾∂t * 0.25dt
            U⃗2[:,i,j] = U⁽²⁾
        end
    end

    sync_halo!(U⃗2)

    # Stage 3
    @batch for j in jlo:jhi
        for i in ilo:ihi
            U⁽²⁾ = SVector{4,Float64}(view(U⃗2, :, i, j))
            U⁽ⁿ⁾ = SVector{4,Float64}(view(U⃗n, :, i, j))

            U_local =  get_me_block(U⃗2,i,j)

            S⃗ = @SVector zeros(4)
            n̂ = SMatrix{2,4}(view(norms,:,:,i,j))
            ΔS = SVector{4,T}(view(ΔS_face,:,i,j))
            Ω = vol[i,j]
            
            stencil = Stencil9Point(U_local, S⃗, n̂, ΔS, Ω, EOS)

            ∂U⁽²⁾∂t = ∂U∂tvec(riemann_solver, stencil, muscl, minmod)
            U⁽ⁿ⁺¹⁾ = (1 / 3) * U⁽ⁿ⁾ + (2 / 3) * U⁽²⁾ + ∂U⁽²⁾∂t * (2 / 3) * dt
            U⃗3[:, i, j] = U⁽ⁿ⁺¹⁾
        end
    end

    sync_halo!(U⃗3)

    return nothing
end

@inbounds function SSPRK3_gc_preserve(SS::SSPRK3Integrator, U⃗n::AbstractArray{T}, 
    riemann_solver, mesh, EOS, dt) where T

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

    blocks = SS.blocks

    # @batch needs to work with isbits types -- somehow see if I can make mesh contain only isbits stuff
    # Stage 1
    @batch for j in jlo:jhi
        for i in ilo:ihi
            U⁽ⁿ⁾ = SVector{4,Float64}(view(U⃗n, :, i, j))

            # U_local = get_block(U⃗n,i,j)
            U_local = StrideArrays.@gc_preserve getblock_marray(U⃗n,i,j)
            # U_local = StrideArrays.@gc_preserve getblock_stride(U⃗n,i,j)
            # U_local = StrideArrays.@gc_preserve getblock_static(U⃗n,i,j)
            S⃗ = @SVector zeros(4)
            n̂ = SMatrix{2,4}(view(norms,:,:,i,j))
            ΔS = SVector{4,T}(view(ΔS_face,:,i,j))
            Ω = vol[i,j]
            stencil = Stencil9Point(U_local, S⃗, n̂, ΔS, Ω, EOS)
            ∂U⁽ⁿ⁾∂t = ∂U∂t(riemann_solver, stencil, muscl, minmod)
            U⁽¹⁾ = U⁽ⁿ⁾ + ∂U⁽ⁿ⁾∂t * dt
            U⃗1[:,i,j] = U⁽¹⁾
        end
    end

    sync_halo!(U⃗1)

    # Stage 2
    @batch for j in jlo:jhi
        for i in ilo:ihi
            U⁽¹⁾ = SVector{4,Float64}(view(U⃗1, :, i, j))
            U⁽ⁿ⁾ = SVector{4,Float64}(view(U⃗n, :, i, j))

            # U_local =  get_block(U⃗1,i,j)
            U_local = StrideArrays.@gc_preserve getblock_marray(U⃗1,i,j)
            # U_local = StrideArrays.@gc_preserve getblock_stride(U⃗1,i,j)
            # U_local = StrideArrays.@gc_preserve getblock_static(U⃗1,i,j)
            S⃗ = @SVector zeros(4)
            n̂ = SMatrix{2,4}(view(norms,:,:,i,j))
            ΔS = SVector{4,T}(view(ΔS_face,:,i,j))
            Ω = vol[i,j]
            
            stencil = Stencil9Point(U_local, S⃗, n̂, ΔS, Ω, EOS)

            ∂U⁽¹⁾∂t = ∂U∂t(riemann_solver, stencil, muscl, minmod)
            U⁽²⁾ = 0.75U⁽ⁿ⁾ + 0.25U⁽¹⁾ + ∂U⁽¹⁾∂t * 0.25dt
            U⃗2[:,i,j] = U⁽²⁾
        end
    end

    sync_halo!(U⃗2)

    # Stage 3
    @batch for j in jlo:jhi
        for i in ilo:ihi
            U⁽²⁾ = SVector{4,Float64}(view(U⃗2, :, i, j))
            U⁽ⁿ⁾ = SVector{4,Float64}(view(U⃗n, :, i, j))

            # U_local =  get_block(U⃗2,i,j)
            U_local = StrideArrays.@gc_preserve getblock_marray(U⃗2,i,j)
            # U_local = StrideArrays.@gc_preserve getblock_stride(U⃗2,i,j)
            # U_local = StrideArrays.@gc_preserve getblock_static(U⃗2,i,j)
            S⃗ = @SVector zeros(4)
            n̂ = SMatrix{2,4}(view(norms,:,:,i,j))
            ΔS = SVector{4,T}(view(ΔS_face,:,i,j))
            Ω = vol[i,j]
            
            stencil = Stencil9Point(U_local, S⃗, n̂, ΔS, Ω, EOS)

            ∂U⁽²⁾∂t = ∂U∂t(riemann_solver, stencil, muscl, minmod)
            U⁽ⁿ⁺¹⁾ = (1 / 3) * U⁽ⁿ⁾ + (2 / 3) * U⁽²⁾ + ∂U⁽²⁾∂t * (2 / 3) * dt
            U⃗3[:, i, j] = U⁽ⁿ⁺¹⁾
        end
    end

    sync_halo!(U⃗3)

    return nothing
end

@inbounds function SSPRK3(SS::SSPRK3Integrator, U⃗n::AbstractArray{T}, 
    riemann_solver, mesh, EOS, dt) where T

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

    blocks = SS.blocks

    # @batch needs to work with isbits types -- somehow see if I can make mesh contain only isbits stuff
    # Stage 1
    @batch for j in jlo:jhi
        for i in ilo:ihi
            U⁽ⁿ⁾ = SVector{4,Float64}(view(U⃗n, :, i, j))

            Ublock = blocks[threadid()]
            uview = view(U⃗n,:,i-nhalo:i+nhalo, j-nhalo:j+nhalo)
            copy!(Ublock, uview)
            U_local = SArray{Tuple{4,5,5},T,3,100}(Ublock)

            S⃗ = @SVector zeros(4)
            n̂ = SMatrix{2,4}(view(norms,:,:,i,j))
            ΔS = SVector{4,T}(view(ΔS_face,:,i,j))
            Ω = vol[i,j]
            stencil = Stencil9Point(U_local, S⃗, n̂, ΔS, Ω, EOS)
            ∂U⁽ⁿ⁾∂t = ∂U∂t(riemann_solver, stencil, muscl, minmod)
            U⁽¹⁾ = U⁽ⁿ⁾ + ∂U⁽ⁿ⁾∂t * dt
            U⃗1[:,i,j] = U⁽¹⁾
        end
    end

    sync_halo!(U⃗1)

    # Stage 2
    @batch for j in jlo:jhi
        for i in ilo:ihi
            U⁽¹⁾ = SVector{4,Float64}(view(U⃗1, :, i, j))
            U⁽ⁿ⁾ = SVector{4,Float64}(view(U⃗n, :, i, j))

            Ublock = blocks[threadid()]
            uview = view(U⃗1,:,i-nhalo:i+nhalo, j-nhalo:j+nhalo)
            copy!(Ublock, uview)
            U_local = SArray{Tuple{4,5,5},T,3,100}(Ublock)

            S⃗ = @SVector zeros(4)
            n̂ = SMatrix{2,4}(view(norms,:,:,i,j))
            ΔS = SVector{4,T}(view(ΔS_face,:,i,j))
            Ω = vol[i,j]
            
            stencil = Stencil9Point(U_local, S⃗, n̂, ΔS, Ω, EOS)

            ∂U⁽¹⁾∂t = ∂U∂t(riemann_solver, stencil, muscl, minmod)
            U⁽²⁾ = 0.75U⁽ⁿ⁾ + 0.25U⁽¹⁾ + ∂U⁽¹⁾∂t * 0.25dt
            U⃗2[:,i,j] = U⁽²⁾
        end
    end

    sync_halo!(U⃗2)

    # Stage 3
    @batch for j in jlo:jhi
        for i in ilo:ihi
            U⁽²⁾ = SVector{4,Float64}(view(U⃗2, :, i, j))
            U⁽ⁿ⁾ = SVector{4,Float64}(view(U⃗n, :, i, j))

            Ublock = blocks[threadid()]
            uview = view(U⃗2,:,i-nhalo:i+nhalo, j-nhalo:j+nhalo)
            copy!(Ublock, uview)
            U_local = SArray{Tuple{4,5,5},T,3,100}(Ublock)

            S⃗ = @SVector zeros(4)
            n̂ = SMatrix{2,4}(view(norms,:,:,i,j))
            ΔS = SVector{4,T}(view(ΔS_face,:,i,j))
            Ω = vol[i,j]
            
            stencil = Stencil9Point(U_local, S⃗, n̂, ΔS, Ω, EOS)

            ∂U⁽²⁾∂t = ∂U∂t(riemann_solver, stencil, muscl, minmod)
            U⁽ⁿ⁺¹⁾ = (1 / 3) * U⁽ⁿ⁾ + (2 / 3) * U⁽²⁾ + ∂U⁽²⁾∂t * (2 / 3) * dt
            U⃗3[:, i, j] = U⁽ⁿ⁺¹⁾
        end
    end

    sync_halo!(U⃗3)

    return nothing
end

@inbounds function SSPRK3Split(SS::SSPRK3Integrator, U⃗n::AbstractArray{T}, 
    riemann_solver, mesh, EOS, dt) where T

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
    
    iblocks = @SVector [MMatrix{4,5,T,20}(undef) for _ in 1:nthreads()]
    jblocks = @SVector [MMatrix{4,5,T,20}(undef) for _ in 1:nthreads()]

    # Stage 1
    @batch for j in jlo:jhi
        for i in ilo:ihi
            U⁽ⁿ⁾ = SVector{4,T}(view(U⃗n, :, i, j))
            
            iview = view(U⃗n, :, i-nhalo:i+nhalo, j)
            jview = view(U⃗n, :, i, j-nhalo:j+nhalo)
            iblk = iblocks[threadid()]
            jblk = jblocks[threadid()]
            copy!(iblk, iview)
            copy!(jblk, jview)

            U⃗ᵢ = SMatrix{4,5,T,20}(iblk)
            U⃗ⱼ = SMatrix{4,5,T,20}(jblk)
            U⃗ᵢ₊₁ⱼ₊₁ = SVector{4,T}(view(U⃗n, :, i+1, j+1))
            U⃗ᵢ₊₁ⱼ₋₁ = SVector{4,T}(view(U⃗n, :, i+1, j-1))
            U⃗ᵢ₋₁ⱼ₊₁ = SVector{4,T}(view(U⃗n, :, i-1, j+1))
            U⃗ᵢ₋₁ⱼ₋₁ = SVector{4,T}(view(U⃗n, :, i-1, j-1))

            S⃗ = @SVector zeros(4)
            n̂ = SMatrix{2,4,T,8}(view(norms,:,:,i,j))
            ΔS = SVector{4,T}(view(ΔS_face,:,i,j))
            Ω = vol[i,j]
            stencil = Stencil9PointSplit(
                U⃗ᵢ,
                U⃗ⱼ,
                U⃗ᵢ₊₁ⱼ₊₁,
                U⃗ᵢ₊₁ⱼ₋₁,
                U⃗ᵢ₋₁ⱼ₊₁,
                U⃗ᵢ₋₁ⱼ₋₁,
                S⃗, n̂, ΔS, Ω, EOS
            )

            ∂U⁽ⁿ⁾∂t = ∂U∂t(riemann_solver, stencil, muscl, minmod)
            U⁽¹⁾ = U⁽ⁿ⁾ + ∂U⁽ⁿ⁾∂t * dt
            U⃗1[:,i,j] = U⁽¹⁾
        end
    end

    sync_halo!(U⃗1)

    # Stage 2
    @batch for j in jlo:jhi
        for i in ilo:ihi
            U⁽¹⁾ = SVector{4,T}(view(U⃗1, :, i, j))
            U⁽ⁿ⁾ = SVector{4,T}(view(U⃗n, :, i, j))

            iview = view(U⃗1, :, i-nhalo:i+nhalo, j)
            jview = view(U⃗1, :, i, j-nhalo:j+nhalo)
            iblk = iblocks[threadid()]
            jblk = jblocks[threadid()]
            copy!(iblk, iview)
            copy!(jblk, jview)

            U⃗ᵢ = SMatrix{4,5,T,20}(iblk)
            U⃗ⱼ = SMatrix{4,5,T,20}(jblk)

            U⃗ᵢ₊₁ⱼ₊₁ = SVector{4, T}(view(U⃗1, :, i+1, j+1))
            U⃗ᵢ₊₁ⱼ₋₁ = SVector{4, T}(view(U⃗1, :, i+1, j-1))
            U⃗ᵢ₋₁ⱼ₊₁ = SVector{4, T}(view(U⃗1, :, i-1, j+1))
            U⃗ᵢ₋₁ⱼ₋₁ = SVector{4, T}(view(U⃗1, :, i-1, j-1))

            S⃗ = @SVector zeros(4)
            n̂ = SMatrix{2,4,T,8}(view(norms,:,:,i,j))
            ΔS = SVector{4,T}(view(ΔS_face,:,i,j))
            Ω = vol[i,j]
            stencil = Stencil9PointSplit(
                U⃗ᵢ,
                U⃗ⱼ,
                U⃗ᵢ₊₁ⱼ₊₁,
                U⃗ᵢ₊₁ⱼ₋₁,
                U⃗ᵢ₋₁ⱼ₊₁,
                U⃗ᵢ₋₁ⱼ₋₁,
                S⃗, n̂, ΔS, Ω, EOS
            )

            ∂U⁽¹⁾∂t = ∂U∂t(riemann_solver, stencil, muscl, minmod)
            U⁽²⁾ = 0.75U⁽ⁿ⁾ + 0.25U⁽¹⁾ + ∂U⁽¹⁾∂t * 0.25dt
            U⃗2[:,i,j] = U⁽²⁾
        end
    end

    sync_halo!(U⃗2)

    # Stage 3
    @batch for j in jlo:jhi
        for i in ilo:ihi
            U⁽²⁾ = SVector{4,T}(view(U⃗2, :, i, j))
            U⁽ⁿ⁾ = SVector{4,T}(view(U⃗n, :, i, j))

            iview = view(U⃗2, :, i-nhalo:i+nhalo, j)
            jview = view(U⃗2, :, i, j-nhalo:j+nhalo)
            iblk = iblocks[threadid()]
            jblk = jblocks[threadid()]
            copy!(iblk, iview)
            copy!(jblk, jview)

            U⃗ᵢ = SMatrix{4,5,T,20}(iblk)
            U⃗ⱼ = SMatrix{4,5,T,20}(jblk)
            U⃗ᵢ₊₁ⱼ₊₁ = SVector{4, T}(view(U⃗2, :, i+1, j+1))
            U⃗ᵢ₊₁ⱼ₋₁ = SVector{4, T}(view(U⃗2, :, i+1, j-1))
            U⃗ᵢ₋₁ⱼ₊₁ = SVector{4, T}(view(U⃗2, :, i-1, j+1))
            U⃗ᵢ₋₁ⱼ₋₁ = SVector{4, T}(view(U⃗2, :, i-1, j-1))

            S⃗ = @SVector zeros(4)
            n̂ = SMatrix{2,4,T,8}(view(norms,:,:,i,j))
            ΔS = SVector{4,T}(view(ΔS_face,:,i,j))
            Ω = vol[i,j]
            stencil = Stencil9PointSplit(
                U⃗ᵢ,
                U⃗ⱼ,
                U⃗ᵢ₊₁ⱼ₊₁,
                U⃗ᵢ₊₁ⱼ₋₁,
                U⃗ᵢ₋₁ⱼ₊₁,
                U⃗ᵢ₋₁ⱼ₋₁,
                S⃗, n̂, ΔS, Ω, EOS
            )

            ∂U⁽²⁾∂t = ∂U∂t(riemann_solver, stencil, muscl, minmod)
            U⁽ⁿ⁺¹⁾ = (1 / 3) * U⁽ⁿ⁾ + (2 / 3) * U⁽²⁾ + ∂U⁽²⁾∂t * (2 / 3) * dt
            U⃗3[:, i, j] = U⁽ⁿ⁺¹⁾
        end
    end

    sync_halo!(U⃗3)

    return nothing
end

@inbounds function SSPRK3Split2(SS::SSPRK3Integrator, U⃗n::AbstractArray{T}, 
    riemann_solver, mesh, EOS, dt) where T

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
    
    iblocks = SS.iblocks
    jblocks = SS.jblocks

    # Stage 1
    @batch for j in jlo:jhi
        for i in ilo:ihi
            U⁽ⁿ⁾ = SVector{4,T}(view(U⃗n, :, i, j))
            
            iview = view(U⃗n, :, i-nhalo:i+nhalo, j)
            jview = view(U⃗n, :, i, j-nhalo:j+nhalo)
            # iblk = iblocks[threadid()]
            # jblk = jblocks[threadid()]
            # copy!(iblk, iview)
            # copy!(jblk, jview)

            U⃗ᵢ = SMatrix{4,5,T,20}(iview)
            U⃗ⱼ = SMatrix{4,5,T,20}(jview)
            U⃗ᵢ₊₁ⱼ₊₁ = SVector{4,T}(view(U⃗n, :, i+1, j+1))
            U⃗ᵢ₊₁ⱼ₋₁ = SVector{4,T}(view(U⃗n, :, i+1, j-1))
            U⃗ᵢ₋₁ⱼ₊₁ = SVector{4,T}(view(U⃗n, :, i-1, j+1))
            U⃗ᵢ₋₁ⱼ₋₁ = SVector{4,T}(view(U⃗n, :, i-1, j-1))

            S⃗ = @SVector zeros(4)
            n̂ = SMatrix{2,4,T,8}(view(norms,:,:,i,j))
            ΔS = SVector{4,T}(view(ΔS_face,:,i,j))
            Ω = vol[i,j]
            stencil = Stencil9PointSplit(
                U⃗ᵢ,
                U⃗ⱼ,
                U⃗ᵢ₊₁ⱼ₊₁,
                U⃗ᵢ₊₁ⱼ₋₁,
                U⃗ᵢ₋₁ⱼ₊₁,
                U⃗ᵢ₋₁ⱼ₋₁,
                S⃗, n̂, ΔS, Ω, EOS
            )

            ∂U⁽ⁿ⁾∂t = ∂U∂t(riemann_solver, stencil, muscl, minmod)
            U⁽¹⁾ = U⁽ⁿ⁾ + ∂U⁽ⁿ⁾∂t * dt
            U⃗1[:,i,j] = U⁽¹⁾
        end
    end

    sync_halo!(U⃗1)

    # Stage 2
    @batch for j in jlo:jhi
        for i in ilo:ihi
            U⁽¹⁾ = SVector{4,T}(view(U⃗1, :, i, j))
            U⁽ⁿ⁾ = SVector{4,T}(view(U⃗n, :, i, j))

            iview = view(U⃗1, :, i-nhalo:i+nhalo, j)
            jview = view(U⃗1, :, i, j-nhalo:j+nhalo)
            # iblk = iblocks[threadid()]
            # jblk = jblocks[threadid()]
            # copy!(iblk, iview)
            # copy!(jblk, jview)

            U⃗ᵢ = SMatrix{4,5,T,20}(iview)
            U⃗ⱼ = SMatrix{4,5,T,20}(jview)

            U⃗ᵢ₊₁ⱼ₊₁ = SVector{4, T}(view(U⃗1, :, i+1, j+1))
            U⃗ᵢ₊₁ⱼ₋₁ = SVector{4, T}(view(U⃗1, :, i+1, j-1))
            U⃗ᵢ₋₁ⱼ₊₁ = SVector{4, T}(view(U⃗1, :, i-1, j+1))
            U⃗ᵢ₋₁ⱼ₋₁ = SVector{4, T}(view(U⃗1, :, i-1, j-1))

            S⃗ = @SVector zeros(4)
            n̂ = SMatrix{2,4,T,8}(view(norms,:,:,i,j))
            ΔS = SVector{4,T}(view(ΔS_face,:,i,j))
            Ω = vol[i,j]
            stencil = Stencil9PointSplit(
                U⃗ᵢ,
                U⃗ⱼ,
                U⃗ᵢ₊₁ⱼ₊₁,
                U⃗ᵢ₊₁ⱼ₋₁,
                U⃗ᵢ₋₁ⱼ₊₁,
                U⃗ᵢ₋₁ⱼ₋₁,
                S⃗, n̂, ΔS, Ω, EOS
            )

            ∂U⁽¹⁾∂t = ∂U∂t(riemann_solver, stencil, muscl, minmod)
            U⁽²⁾ = 0.75U⁽ⁿ⁾ + 0.25U⁽¹⁾ + ∂U⁽¹⁾∂t * 0.25dt
            U⃗2[:,i,j] = U⁽²⁾
        end
    end

    sync_halo!(U⃗2)

    # Stage 3
    @batch for j in jlo:jhi
        for i in ilo:ihi
            U⁽²⁾ = SVector{4,T}(view(U⃗2, :, i, j))
            U⁽ⁿ⁾ = SVector{4,T}(view(U⃗n, :, i, j))

            iview = view(U⃗2, :, i-nhalo:i+nhalo, j)
            jview = view(U⃗2, :, i, j-nhalo:j+nhalo)
            # iblk = iblocks[threadid()]
            # jblk = jblocks[threadid()]
            # copy!(iblk, iview)
            # copy!(jblk, jview)

            U⃗ᵢ = SMatrix{4,5,T,20}(iview)
            U⃗ⱼ = SMatrix{4,5,T,20}(jview)
            U⃗ᵢ₊₁ⱼ₊₁ = SVector{4, T}(view(U⃗2, :, i+1, j+1))
            U⃗ᵢ₊₁ⱼ₋₁ = SVector{4, T}(view(U⃗2, :, i+1, j-1))
            U⃗ᵢ₋₁ⱼ₊₁ = SVector{4, T}(view(U⃗2, :, i-1, j+1))
            U⃗ᵢ₋₁ⱼ₋₁ = SVector{4, T}(view(U⃗2, :, i-1, j-1))

            S⃗ = @SVector zeros(4)
            n̂ = SMatrix{2,4,T,8}(view(norms,:,:,i,j))
            ΔS = SVector{4,T}(view(ΔS_face,:,i,j))
            Ω = vol[i,j]
            stencil = Stencil9PointSplit(
                U⃗ᵢ,
                U⃗ⱼ,
                U⃗ᵢ₊₁ⱼ₊₁,
                U⃗ᵢ₊₁ⱼ₋₁,
                U⃗ᵢ₋₁ⱼ₊₁,
                U⃗ᵢ₋₁ⱼ₋₁,
                S⃗, n̂, ΔS, Ω, EOS
            )

            ∂U⁽²⁾∂t = ∂U∂t(riemann_solver, stencil, muscl, minmod)
            U⁽ⁿ⁺¹⁾ = (1 / 3) * U⁽ⁿ⁾ + (2 / 3) * U⁽²⁾ + ∂U⁽²⁾∂t * (2 / 3) * dt
            U⃗3[:, i, j] = U⁽ⁿ⁺¹⁾
        end
    end

    sync_halo!(U⃗3)

    return nothing
end

function get_blk2(U, i, j, nhalo)
    Ublock = MArray{Tuple{4,5,5},Float64,3,100}(undef)
    Ublockview = @view U[:,i-nhalo:i+nhalo, j-nhalo:j+nhalo]
    copy!(Ublock,Ublockview)
    SArray(Ublock)
end

function get_blk(U, i, j, nhalo)
    # Ublock = MArray{Tuple{4,5,5},T,3,100}(undef)
    Ublock = StrideArray{Float64}(undef, (StaticInt(4),StaticInt(5),StaticInt(5)))
    Ublockview = @view U[:,i-nhalo:i+nhalo, j-nhalo:j+nhalo]
    copy!(Ublock,Ublockview)
    Ublock
    # SArray{Tuple{4,5,5},Float64,3,100}(Ublock)
end

@inbounds function cs_gpu!(U::AbstractArray{T,3}, cs::AbstractArray{T,2}, eos) where T
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
	strx = blockDim().x * gridDim().x
    stry = blockDim().y * gridDim().y
	
    _, Nx, Ny = size(U⃗n)
	ihi = Nx - 3
	jhi = Ny - 3
	ilo = 3 
	jlo = 3

    if (ilo <= idx <= ihi) && (jlo <= idy <= jhi)
        for j = idy:stry:jhi
            for i = idx:strx:ihi
               ρ = U[1,i,j]
               u = U[2,i,j] / ρ
               v = U[3,i,j] / ρ
               E = U[4,i,j] / ρ
               p = pressure(eos, ρ, u, v, E)

               cs[i,j] = sound_speed(eos, ρ, p)
            end
        end
    end
end

# end # module
