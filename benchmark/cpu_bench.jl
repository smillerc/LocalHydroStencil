
using .Threads, BenchmarkTools
using LocalHydroStencil

eos = IdealEOS(1.4)
dx = 0.001
# dx = 4e-4
x = collect(-0.2:dx:0.2)
y = collect(-0.2:dx:0.2)
# x = range(-.2,.2,2980)
# y = range(-.2,.2,220)

nhalo = 2
mesh = CartesianMesh(x, y, nhalo)
M, N = size(mesh.volume)

# @show (M, N)
ρL, ρR = 1.0, 0.125
pL, pR = 1.0, 0.1

ρ0 = zeros(size(mesh.volume))
u0 = zeros(size(mesh.volume))
v0 = zeros(size(mesh.volume))
p0 = zeros(size(mesh.volume))

ρ0[begin:(N ÷ 2), :] .= ρL
ρ0[(N ÷ 2):end, :] .= ρR

p0[begin:(N ÷ 2), :] .= pL
p0[(N ÷ 2):end, :] .= pR

E0 = specific_total_energy.(Ref(eos), ρ0, u0, v0, p0);

dt = 1e-5

U⃗ = zeros(4, M, N);

for j in axes(mesh.volume, 2)
    for i in axes(mesh.volume, 1)
        U⃗[1, i, j] = ρ0[i, j]
        U⃗[2, i, j] = ρ0[i, j] * u0[i, j]
        U⃗[3, i, j] = ρ0[i, j] * v0[i, j]
        U⃗[4, i, j] = ρ0[i, j] * E0[i, j]
    end
end

ρ = @view U⃗[1, :, :]
ρu = @view U⃗[2, :, :]
ρv = @view U⃗[3, :, :]
ρE = @view U⃗[4, :, :]

RS = M_AUSMPWPlus2D()
time_int = SSPRK3IntegratorCPU(U⃗)

println("nthreads: ", nthreads())
integrate!(time_int, U⃗, RS, mesh, eos, dt, muscl, minmod)
@benchmark begin
    integrate!($time_int, $U⃗, $RS, $mesh, $eos, $dt, $muscl, $minmod)
end
