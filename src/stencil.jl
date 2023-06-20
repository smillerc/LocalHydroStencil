module StencilType

using StaticArrays, .Threads, Polyester, LinearAlgebra, Unitful
using Adapt

export Stencil9Point, get_block

const ϵ = 2eps(Float64)

struct Stencil9Point{T,AA,E}
    U⃗::AA
    S⃗::SVector{4,T} # source terms
    n̂::SMatrix{2,4,T,8}
    ΔS::SVector{4,T}
    Ω::T
    EOS::E
    centroid::SVector{2,T}
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

@inline function get_block(U::AbstractArray{T,3}, i, j) where {T}
    ublk = @SArray [SVector{4,T}(view(U, :, i + io, j + jo)) for jo in -2:2, io in -2:2]
    return ublk
end

end # module
