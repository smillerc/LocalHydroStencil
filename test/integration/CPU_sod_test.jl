
using Revise
using LocalHydroStencil
using BenchmarkTools

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

# function main()
eos = IdealEOS(1.4)
dx = dx = 1e-4
x = collect(-0.2:dx:0.2)
y = collect(-0.2:dx:0.2)
M, N = size(mesh.volume)
@show (M, N)

nhalo = 2
mesh = CartesianMesh(x, y, nhalo)

U = initialize(mesh, eos)

riemann_solver = M_AUSMPWPlus2D()
time_int = SSPRK3(U)

dt = 1e-5

skip_uniform = false
integrate!(time_int, U, mesh, eos, dt, riemann_solver, muscl, minmod, skip_uniform)
println("Updating solution")
@benchmark integrate!(
  $time_int, $U, $mesh, $eos, $dt, $riemann_solver, $muscl, $minmod, $skip_uniform
) # 16 threads -> 1.823s
