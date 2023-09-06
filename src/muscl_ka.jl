using Revise
using KernelAbstractions
# using LocalHydroStencil
using BenchmarkTools

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

@kernel function muscl_gmem!(U::AbstractArray{T,N}, i_face, j_face, limits) where {T,N}
  i, j = @index(Global, NTuple)
  ilo, ihi, jlo, jhi = limits

  SMALL = T(1e-30)
  minmod(r) = max(0, min(r, 1))

  # i face
  if (jlo <= j <= jhi) && (ilo - 1 <= i <= ihi)
    @inbounds for q in 1:4
      U[q, i, j] = U[q, i, j]
      U[q, i + 1, j] = U[q, i + 1, j]

      Δi_minus_half = U[q, i, j] - U[q, i - 1, j]
      Δi_plus_half = U[q, i + 1, j] - U[q, i, j]
      Δi_plus_three_half = U[q, i + 2, j] - U[q, i + 1, j]

      rLi = Δi_plus_half / (Δi_minus_half + SMALL)
      rRi = Δi_plus_half / (Δi_plus_three_half + SMALL)

      lim_Li = minmod(rLi)
      lim_Ri = minmod(rRi)

      i_face[1, q, i, j] = U[q, i, j] + lim_Li / 2 * Δi_minus_half
      i_face[2, q, i, j] = U[q, i + 1, j] - lim_Ri / 2 * Δi_plus_three_half
    end
  end

  # j face
  if (jlo - 1 <= j <= jhi) && (ilo <= i <= ihi)
    @inbounds for q in 1:4
      Δj_minus_half = U[q, i, j] - U[q, i, j - 1]
      Δj_plus_half = U[q, i, j + 1] - U[q, i, j]
      Δj_plus_three_half = U[q, i, j + 2] - U[q, i, j + 1]

      rLj = Δj_plus_half / (Δj_minus_half + SMALL)
      rRj = Δj_plus_half / (Δj_plus_three_half + SMALL)

      lim_Lj = minmod(rLj)
      lim_Rj = minmod(rRj)

      j_face[1, q, i, j] = U[q, i, j] + lim_Lj / 2 * Δj_minus_half
      j_face[2, q, i, j] = U[q, i, j + 1] - lim_Rj / 2 * Δj_plus_three_half
    end
  end
end

@kernel function muscl_lmem!(U::AbstractArray{T,NT}, i_face, j_face, limits) where {T,NT}
  ilo, ihi, jlo, jhi = limits

  i, j = @index(Global, NTuple)
  li, lj = @index(Local, NTuple)

  nhalo = 2
  @uniform Q = 4
  N = @uniform @groupsize()[1]
  M = @uniform @groupsize()[2]

  local_U = @localmem T (4, 8, 8)

  # Load data from global to local buffer
  @inbounds for q in 1:4
    local_U[q, li, lj] = U[q, i, j]
    # if i == 1
    #   local_U[q, li - 1, lj] = U[q, i - 1, j]
    # end
    # if i == @groupsize()[1]
    #   local_U[q, li + 1, lj] = U[q, i + 1, j]
    # end
    # if j == 1
    #   local_U[q, li, lj - 1] = U[q, i, j - 1]
    # end
    # if j == @groupsize()[2]
    #   local_U[q, li, lj + 1] = U[q, i, j + 1]
    # end
  end
  @synchronize()

  SMALL = T(1e-30)
  minmod(r) = max(0, min(r, 1))

  # i face
  if (jlo <= j <= jhi) && (ilo - 1 <= i <= ihi)
    @inbounds for q in 1:4
      local_U[q, li, lj] = local_U[q, li, lj]
      local_U[q, li + 1, lj] = local_U[q, li + 1, lj]

      Δi_minus_half = local_U[q, li, lj] - local_U[q, li - 1, lj]
      Δi_plus_half = local_U[q, li + 1, lj] - local_U[q, li, lj]
      Δi_plus_three_half = local_U[q, li + 2, lj] - local_U[q, li + 1, lj]

      rLi = Δi_plus_half / (Δi_minus_half + SMALL)
      rRi = Δi_plus_half / (Δi_plus_three_half + SMALL)

      lim_Li = minmod(rLi)
      lim_Ri = minmod(rRi)

      i_face[1, q, i, j] = local_U[q, li, lj] + lim_Li / 2 * Δi_minus_half
      i_face[2, q, i, j] = local_U[q, li + 1, lj] - lim_Ri / 2 * Δi_plus_three_half
    end
  end

  # j face
  if (jlo - 1 <= j <= jhi) && (ilo <= i <= ihi)
    @inbounds for q in 1:4
      Δj_minus_half = local_U[q, li, lj] - local_U[q, li, lj - 1]
      Δj_plus_half = local_U[q, li, lj + 1] - local_U[q, li, lj]
      Δj_plus_three_half = local_U[q, li, lj + 2] - local_U[q, li, lj + 1]

      rLj = Δj_plus_half / (Δj_minus_half + SMALL)
      rRj = Δj_plus_half / (Δj_plus_three_half + SMALL)

      lim_Lj = minmod(rLj)
      lim_Rj = minmod(rRj)

      j_face[1, q, i, j] = local_U[q, li, lj] + lim_Lj / 2 * Δj_minus_half
      j_face[2, q, i, j] = local_U[q, li, lj + 1] - lim_Rj / 2 * Δj_plus_three_half
    end
  end
end

# @kernel function muscl_lmem!(out, @Const(data), a, dt, dx, dy)
#   i, j = @index(Global, NTuple)
#   li, lj = @index(Local, NTuple)
#   lmem = @localmem eltype(data) (@groupsize()[1] + 2, @groupsize()[2] + 2)
#   @uniform ldata = OffsetArray(lmem, 0:(@groupsize()[1] + 1), 0:(@groupsize()[2] + 1))

#   # Load data from global to local buffer
#   @inbounds begin
#     ldata[li, lj] = data[i, j]
#     if i == 1
#       ldata[li - 1, lj] = data[i - 1, j]
#     end
#     if i == @groupsize()[1]
#       ldata[li + 1, lj] = data[i + 1, j]
#     end
#     if j == 1
#       ldata[li, lj - 1] = data[i, j - 1]
#     end
#     if j == @groupsize()[2]
#       ldata[li, lj + 1] = data[i, j + 1]
#     end
#   end
#   @synchronize()

#   @inbounds begin
#     dij = ldata[li, lj]
#     dim1j = ldata[li - 1, lj]
#     dijm1 = ldata[li, lj - 1]
#     dip1j = ldata[li + 1, lj]
#     dijp1 = ldata[li, lj + 1]

#     dij += a * dt * ((dim1j - 2 * dij + dip1j) / dx^2 + (dijm1 - 2 * dij + dijp1) / dy^2)

#     out[i, j] = dij
#   end
# end

@inline minmod(r) = max(0, min(r, 1))

ker = muscl_gmem!(backend)
# ker_shmem = muscl_lmem!(backend)

M, N = (5000, 5000)
@show M * N
nh = 2
T = Float64
limits = (nh + 1, M - nh, nh + 1, N - nh)
W = adapt(ArrayT, rand(T, 4, M, N));
W_iface = adapt(ArrayT, zeros(T, 4, 2, M, N));
W_jface = adapt(ArrayT, zeros(T, 4, 2, M, N));

const blkdim = 8
ker_shmem(W, W_iface, W_jface, limits; ndrange=(M, N), workgroupsize=(blkdim, blkdim))
KernelAbstractions.synchronize(backend)

# @tim begin
#   ker($W, $W_iface, $W_jface, $limits; ndrange=($M, $N))
#   KernelAbstractions.synchronize($backend)
# end

# @benchmark begin
#   ker_shmem(
#     $W, $W_iface, $W_jface, $limits; ndrange=($M, $N), workgroupsize=(blkdim, blkdim)
#   )
#   KernelAbstractions.synchronize($backend)
# end
# nothing
# CUDA: 5k x 5k -> ~40ms with Float64
# CUDA: 5k x 5k -> ~15ms with Float32

# CPU, 1 threads: 5k x 5k -> ~2s with Float64
# CPU, 6 threads: 5k x 5k -> ~360ms with Float64
# CPU, 32 threads: 5k x 5k -> ~148ms with Float64

# CPU, 32 threads: 5k x 5k -> ~121ms with Float32
