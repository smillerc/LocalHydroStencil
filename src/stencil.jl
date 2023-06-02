# module StencilType

using StaticArrays, .Threads, Polyester, LinearAlgebra, Unitful
using Adapt

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

include("cpu.jl")
include("gpu.jl")


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

function get_block(U::AbstractArray{T,3}, i, j) where {T}
    ublk = @SArray [
        SVector{4,T}(view(U, :, i + io, j + jo)) for jo in -2:2, io in -2:2
    ]
    return ublk
end

function ∂U∂t(::M_AUSMPWPlus2D, stencil::Stencil9Point, recon::F1, limiter::F2) where {F1,F2}
    U⃗ = stencil.U⃗
    EOS = stencil.EOS

    # If the entire block is uniform, skip the riemann solve and just return 
    # if all_same(U⃗)
    #     return @SVector zeros(size(stencil.S⃗,1))
    # end
    # Conserved to primitive variables
    # W⃗ = cons2prim.(Ref(EOS), ρ, ρu, ρv, ρE)
    W⃗ = cons2prim.(Ref(EOS), U⃗)

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
    +stencil.S⃗

    return dUdt
end

"""Check if all values in the block are the same. This lets us skip the Riemann solve"""
function all_same(blk)
    blk0 = @view first(blk)[:]

    for cell in blk
        for q in eachindex(cell)
            if !isapprox(blk0[q], cell[q])
                return false
            end
        end
    end
    return true
end

# end # module
