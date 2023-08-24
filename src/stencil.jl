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
  ublk = @SArray [SVector{4,T}(view(U, 1:4, i + io, j + jo)) for io in -2:2, jo in -2:2]
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


# function getvec(A::AbstractArray{T,3}, i, j) where {T}
#   return SVector{4,T}(view(A, 1:4, i, j))
# end

# function get_nhat(mesh::CartesianMesh{T}, i, j) where {T}
#   return SMatrix{2,4,T}(view(mesh.facenorms, 1:2, 1:4, i, j))
# end

# function get_ΔS(mesh::CartesianMesh{T}, i, j) where {T}
#   return SVector{4,T}(view(mesh.facelen, 1:4, i, j))
# end

# @inbounds function get_gpublock(A::AbstractArray{T,3}, i, j, nh) where {T}
#   S = Tuple{4,5,5}
#   block = MArray{S,T}(undef)
#   aview = view(A, :, (i - nh):(i + nh), (j - nh):(j + nh))
#   for i in eachindex(block)
#     @inbounds block[i] = aview[i]
#   end
#   return SArray(block)
# end

# function _2halo2dstencil(U⃗::AbstractArray{T,3}, Idx, mesh, EOS::E, nh) where {T,E}
#   i, j = Idx
#   U_local = getblock(U⃗, i, j, nh)
#   S⃗ = @SVector zeros(4)
#   n̂ = get_nhat(mesh, i, j)
#   ΔS = get_ΔS(mesh, i, j)
#   Ω = mesh.volume[i, j]
#   BT = SArray{Tuple{4,5,5},T,3,100}

#   return Stencil9Point{T,BT,E}(U_local, S⃗, n̂, ΔS, Ω, EOS)
# end

# function gpu_2halo2dstencil(U⃗::AbstractArray{T,N}, Idx) where {T,N}
#   i, j = Idx

#   U_local = getgpublock(U⃗, i, j, 2)

#   S⃗ = @SVector zeros(4)
#   n̂ = @SMatrix [0.0; -1.0;; 1.0; 0.0;; 0.0; 1.0;; -1.0; 0.0]
#   ΔS = @SVector ones(4)
#   Ω = 1.0
#   γ = 5 / 3
#   return Stencil9Point(U_local, S⃗, n̂, ΔS, Ω, γ)
# end



end # module
