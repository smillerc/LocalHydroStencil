
using .Threads, BenchmarkTools
using ThreadPinning
using CairoMakie
using KernelAbstractions

pinthreads(:compact)

include("stencil_mod.jl")

eos = IdealEOS(1.4)
dx = 1e-3
x = collect(-0.2:dx:0.2)
y = collect(-0.2:dx:0.2)

nhalo = 2
mesh = CartesianMesh(x, y, nhalo)
M = length(x) - 1
N = length(y) - 1

@show (M, N)
ρL, ρR = 1.0, 0.125
pL, pR = 1.0, 0.1

ρ0 = zeros(M, N)
u0 = zeros(M, N)
v0 = zeros(M, N)
p0 = zeros(M, N)

ρ0[begin:(N ÷ 2), :] .= ρL
ρ0[(N ÷ 2):end, :] .= ρR

p0[begin:(N ÷ 2), :] .= pL
p0[(N ÷ 2):end, :] .= pR

E0 = specific_total_energy.(Ref(eos), ρ0, u0, v0, p0)

dt = 1e-5

U⃗ = zeros(4, M, N)

for j in 1:M
    for i in 1:N
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
# time_int2 = SSPRK3IntegratorCPUSplit(U⃗)

CFL = 0.8
dt = CFL * next_Δt(U⃗, mesh, eos)

t = 0.0

for iter in 1:500
    println("i=$iter, t=$t, dt=$dt")
    if t > 0.1
        break
    end
    global dt = CFL * next_Δt(U⃗, mesh, eos)
    # SSPRK3(time_int,U⃗,RS,mesh,eos,dt)
    # SSPRK3_gc_preserve(time_int,U⃗,RS,mesh,eos,dt)
    SSPRK3_vec(time_int, U⃗, RS, mesh, eos, dt)
    # SSPRK3(time_int,U⃗,RS,mesh,eos,dt)
    copy!(U⃗, time_int.U⃗3)
    global t += dt
end

xc = cumsum(diff(x));
fig = Figure();
ax = Axis(fig[1, 1]);
lines!(ax, xc, U⃗[1, :, N ÷ 2]);
fig
save("sod.png", fig)

# println("SSPRK3")
# @benchmark SSPRK3($time_int,$U⃗,$RS,$mesh,$eos,$dt)

# println("SSPRK3_gc_preserve")
# @benchmark SSPRK3_gc_preserve($time_int,$U⃗,$RS,$mesh,$eos,$dt)

# println("SSPRK3_init_wrong")
# @benchmark SSPRK3_init_wrong($time_int,$U⃗,$RS,$mesh,$eos,$dt)

# println("SSPRK3Split")
# @benchmark SSPRK3Split($time_int,$U⃗,$RS,$mesh,$eos,$dt)

# println("SSPRK3_vec, nthreads=$(nthreads())")
# @btime SSPRK3_vec($time_int,$U⃗,$RS,$mesh,$eos,$dt)

# @perfmon "FLOPS_DP" SSPRK3_vec(time_int,U⃗,RS,mesh,eos,dt)
# @profview SSPRK3_vec(time_int,U⃗,RS,mesh,eos,dt)
# SSPRK3_vec(time_int,U⃗,RS,mesh,eos,dt)

# @benchmark begin
#     getblock_static($U⃗,50,50)
# end

# println("getblock_static")
# @benchmark begin
#     StrideArrays.@gc_preserve getblock_static($U⃗,50,50)
# end

# println("getblock_stride")
# @benchmark begin
#     getblock_stride($U⃗,50,50)
# end

# getblock_stride(U⃗,50,50)
# println("getblock_marray")
# @benchmark begin
#     StrideArrays.@gc_preserve getblock_marray($U⃗,50,50)
# end

# getblock_marray(U⃗,50,50)

# StrideArrays.@gc_preserve getblock_marray(U⃗,50,50)

# @code_warntype getview(U⃗,50,50)
# @code_warntype get_block(U⃗,50,50)

# @benchmark begin
#     StrideArrays.@gc_preserve get_me_block($U⃗,50,50)
# end
# @benchmark begin
#     get_me_block($U⃗,50,50)
# end

# begin

# timeo = TimerOutput()
# enable_timer!(timeo)
# ni = 2000
# nj = 2000
# Ur = rand(4,ni,nj);
# norm_ij = zeros(ni,nj)
# norm_ij_thread = zeros(ni,nj)
# norm_ij_batch =  zeros(ni,nj)
# ilohi = axes(Ur, 2)
# jlohi = axes(Ur, 3)
# ilo = first(ilohi) + 2
# jlo = first(jlohi) + 2
# ihi = last(ilohi) - 2
# jhi = last(jlohi) - 2

# @timeit timeo "serial" begin
# for j in jlo:jhi
#     for i in ilo:ihi
#         # uview = view(Ur,:,i-2:i+2,j-2:j+2)
#         # blk = SArray{Tuple{4,5,5}, Float64, 3}(uview)
#         blk = get_me_block(Ur,i,j)
#         norm_ij[i,j] = norm(blk)
#     end
# end
# end

# @timeit timeo "@threads" begin
# @threads for j in jlo:jhi
#     for i in ilo:ihi
#         uview = view(Ur,:,i-2:i+2,j-2:j+2)
#         blk = SArray{Tuple{4,5,5}, Float64, 3}(uview)
#         # blk = get_me_block(Ur,i,j)
#         norm_ij_thread[i,j] = norm(blk)
#     end
# end
# end

# # @timeit timeo "@batch" begin
# # @batch for j in jlo:jhi
# #     for i in ilo:ihi
# #         uview = view(Ur,:,i-2:i+2,j-2:j+2)
# #         blk = SArray{Tuple{4,5,5}, Float64, 3}(uview)
# #         # blk = get_me_block(Ur,i,j)
# #         norm_ij_batch[i,j] = norm(blk)
# #     end
# # end
# # end

# @show all(norm_ij_thread .== norm_ij)
# # @show all(norm_ij_thread .== norm_ij_batch)

# show(timeo)
# end
