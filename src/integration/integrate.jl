module Integration

using .Threads
using Polyester
using StaticArrays
using Adapt

using ..StencilType
using ..EOSType
using ..ReconstructionType
using ..RiemannSolverType

export SSPRK3IntegratorCPU, integrate!

abstract type AbstractIntegrator end

struct SSPRK3{T<:Number,AT<:AbstractArray{T}} <: AbstractIntegrator
  U⃗1::AT
  U⃗2::AT
  U⃗3::AT
end

# CPU constructor
function SSPRK3(U::Array{T,N}) where {T}
  U1 = similar(U)
  U2 = similar(U)
  U3 = similar(U)

  fill!(U1, 0)
  fill!(U2, 0)
  fill!(U3, 0)

  return SSPRK3(U1, U2, U3)
end

function Adapt.adapt_structure(to, SS::SSPRK3)
  U1 = Adapt.adapt_structure(to, SS.U⃗1)
  U2 = Adapt.adapt_structure(to, SS.U⃗2)
  U3 = Adapt.adapt_structure(to, SS.U⃗3)

  return SSPRK3(U1, U2, U3)
end

include("integrate_cpu.jl")
include("integrate_gpu.jl")

end
