
using Revise, .Threads, BenchmarkTools
using KernelAbstractions
using CUDA, CUDAKernels
# using Cygnus
using CairoMakie
using ThreadPinning

pinthreads(:compact)

include("stencil_mod.jl")

eos = IdealEOS(1.4)
# x = 0:.5e-3:1 |> collect
# y = 0:.5e-3:1 |> collect
x = 0:2e-2:1 |> collect
y = 0:2e-2:1 |> collect
nhalo = 2
cmesh = CartesianMesh(x, y, nhalo)
M = length(x) - 1 # 270
N = length(y) - 1 # 2320

@show (M, N)
ρL, ρR = 1.0, 0.125
pL, pR = 1.0, 0.1

ρ0 = zeros(M, N)
u0 = zeros(M, N)
v0 = zeros(M, N)
p0 = zeros(M, N)

ρ0[begin:N÷2, :] .= ρL
ρ0[N÷2:end, :] .= ρR

p0[begin:N÷2, :] .= pL
p0[N÷2:end, :] .= pR

E0 = specific_total_energy.(Ref(eos), ρ0, u0, v0, p0);

dt = 1e-5

U⃗0 = zeros(4, M, N);

for j in 1:M
    for i in 1:N
        U⃗0[1, i, j] = ρ0[i, j]
        U⃗0[2, i, j] = ρ0[i, j] * u0[i, j]
        U⃗0[3, i, j] = ρ0[i, j] * v0[i, j]
        U⃗0[4, i, j] = ρ0[i, j] * E0[i, j]
    end
end

RS = M_AUSMPWPlus2D()

# -------------------------------------------------------------------------
# -                                  GPU                                  -
# -------------------------------------------------------------------------
const Device = CUDADevice


U⃗ = adapt(CuArray, U⃗0)
time_int_gpu = SSPRK3IntegratorGPU(U⃗);
mesh_gpu = cu(cmesh) 
eos_gpu = cu(eos) 

# SSPRK_kernel = SSPRK3_gpu!(Device(), (32,32))
SSPRK_kernel = SSPRK3_gpu_lmem!(Device(), (32,32))

kernel_event = SSPRK_kernel(time_int_gpu, U⃗, RS, mesh_gpu, eos_gpu, dt, Val(cmesh.nhalo), ndrange=(M,N))
wait(kernel_event)

# @benchmark begin
#     wait(
#         SSPRK_kernel($time_int_gpu, $U⃗, $RS, $mesh_gpu, $eos_gpu, $dt, ndrange=($M,$N))
#     )
# end

# U⃗np1 = Array(time_int_gpu.U⃗3)

# begin
#     fig, ax, hm = heatmap(ρ)
#     Colorbar(fig[:, end+1], hm)
#     fig
# end

# xc = cumsum(diff(x))
# fig = Figure()
# ax = Axis(fig[1,1])
# # scatterlines!(ax, xc, U⃗[1,:,N÷2])
# scatterlines!(ax, xc, Ugpu_np1[1,:,N÷2])
# fig
# save("out_gpu.png", fig)
