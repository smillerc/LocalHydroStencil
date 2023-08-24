using Revise
using KernelAbstractions
using LocalHydroStencil 
#include("D:\Scratch\LocalHydroStencil\src\LocalHydroStencil.jl")
#import LocalHydroStencil
#include("../../src/LocalHydroStencil.jl")
#using .LocalHydroStencil
#using ../../src/LocalHydroStencil
using Adapt
const BACKEND = :CUDA

if BACKEND == :CUDA
  using CUDA
  using CUDA.CUDAKernels
  const ArrayT = CuArray
  #const Device = CUDADevice()
  const backend = CUDABackend()
  CUDA.allowscalar(false)
elseif BACKEND == :METAL
  using Metal
  using Metal.MetalKernels
  const ArrayT = MtlArray
  const Device = MTLDevice
  const backend = MetalBackend()
else
  BACKEND == :CPU
  const ArrayT = Array
  const Device = CPU
  const backend = CPU()
end

function initialize(mesh, eos)
  ρL, ρR = 1.0, 0.125
  pL, pR = 1.0, 0.1

  M, N = size(mesh.volume)
  ρ0 = zeros(size(mesh.volume))
  u0 = zeros(size(mesh.volume))
  v0 = zeros(size(mesh.volume))
  p0 = zeros(size(mesh.volume))

  ρ0[begin:(N ÷ 2), :] .= ρL
  ρ0[(N ÷ 2):end, :] .= ρR

  p0[begin:(N ÷ 2), :] .= pL
  p0[(N ÷ 2):end, :] .= pR

  E0 = specific_total_energy.(Ref(eos), ρ0, u0, v0, p0)

  U⃗ = zeros(4, size(mesh.volume)...)

  for j in axes(mesh.volume, 2)
    for i in axes(mesh.volume, 1)
      U⃗[1, i, j] = ρ0[i, j]
      U⃗[2, i, j] = ρ0[i, j] * u0[i, j]
      U⃗[3, i, j] = ρ0[i, j] * v0[i, j]
      U⃗[4, i, j] = ρ0[i, j] * E0[i, j]
    end
  end

  return U⃗
end

function main()
  eos = IdealEOS(1.4)
  dx = 0.0005
  x = collect(-0.2:dx:0.2)
  y = collect(-0.2:dx:0.2)

  nhalo = 2
  mesh = CartesianMesh(x, y, nhalo)

  U = initialize(mesh, eos)

  if BACKEND == :METAL
    U_device = adapt(ArrayT, U .|> Float32)
  else
    U_device = adapt(ArrayT, U)
    mesh = adapt(ArrayT, mesh)
  end

  copy!(U_device, U)
  U = adapt(ArrayT, U)
  riemann_solver = M_AUSMPWPlus2D()
  time_int = SSPRK3(U_device)
  println(typeof(U_device))
  println(typeof(time_int))
  println(typeof(U))
  #println(eltype(U_device))

  dt = 1e-5

  skip_uniform = false
  println("Running integrate on ", backend)
  integrate!(time_int, U, mesh, eos, dt, riemann_solver, muscl, minmod, backend)

  println("Updating solution")
  copy!(U, time_int.U⃗3)
  return nothing
end

main()
