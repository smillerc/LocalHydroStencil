module StencilType

using StaticArrays, .Threads, Polyester, LinearAlgebra, Unitful
using Adapt

export Stencil9Point, Stencil9PointSplit, get_block, get_iblock, get_jblock

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
  U⃗ᵢ₊₁ⱼ₊₁::SVector{4,T}
  U⃗ᵢ₊₁ⱼ₋₁::SVector{4,T}
  U⃗ᵢ₋₁ⱼ₊₁::SVector{4,T}
  U⃗ᵢ₋₁ⱼ₋₁::SVector{4,T}
  S⃗::SVector{4,T} # source terms
  n̂::SMatrix{2,4,T,8}
  ΔS::SVector{4,T}
  Ω::T
  EOS::E
  centroid::SVector{2,T}
end

@inline function get_block(U::AbstractArray{T,3}, i, j) where {T}
  ublk = @SArray [SVector{4,T}(view(U, :, i + io, j + jo)) for io in -2:2, jo in -2:2]
  return ublk
end

@inline function get_jblock(U::AbstractArray{T,3}, i, j) where {T}
  ublk = @SVector [SVector{4,T}(view(U, :, i, j + jo)) for jo in -2:2]
  return ublk
end

@inline function get_iblock(U::AbstractArray{T,3}, i, j) where {T}
  ublk = @SVector [SVector{4,T}(view(U, :, i + io, j)) for io in -2:2]
  return ublk
end

end # module
