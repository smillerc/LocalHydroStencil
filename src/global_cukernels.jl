using Revise
using KernelAbstractions
using BenchmarkTools
using LocalHydroStencil
using StaticArrays
using Adapt
using Polyester
using .Threads
const BACKEND = :CUDA

# using KernelAbstractions: synchronize

if BACKEND == :CUDA
  using CUDA
  using CUDA.CUDAKernels
  const ArrayT = CuArray{Float32}
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

function cons2prim!(U, W, EOS)
  i = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  @inbounds begin
    ρ = U[1, i, j]
    invρ = 1 / ρ
    u = U[2, i, j] * invρ
    v = U[3, i, j] * invρ
    E = U[4, i, j] * invρ
    p = ρ * EOS._γ_m_1 * (E - 0.5(u^2 + v^2))

    W[1, i, j] = ρ
    W[2, i, j] = u
    W[3, i, j] = v
    W[4, i, j] = p
  end
  return
end

function cons2prim_cpu!(U, W, EOS, limits)
  #i = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  #j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  ilo, ihi, jlo, jhi = limits
  Threads.@threads for j in jlo:jhi
    for i in ilo:ihi
      @inbounds begin
        ρ = U[1, i, j]
        invρ = 1 / ρ
        u = U[2, i, j] * invρ
        v = U[3, i, j] * invρ
        E = U[4, i, j] * invρ
        p = ρ * EOS._γ_m_1 * (E - 0.5(u^2 + v^2))

        W[1, i, j] = ρ
        W[2, i, j] = u
        W[3, i, j] = v
        W[4, i, j] = p
      end
    end
  end
  return W
end

# function cons2prim_stride!(U, W, EOS)
#   row = (blockIdx().y - 1) * blockDim().y + threadIdx().y
#   col = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#   stride_x = blockDim().x * gridDim().x
#   stride_y = blockDim().y * gridDim().y

#   #i = row
#   #j = col
#   @inbounds begin
#     for i=row:stride_y:2048
#       for j=col:stride_x:2048
#         ρ = U[1, i, j]
#         invρ = 1 / ρ
#         u = U[2, i, j] * invρ
#         v = U[3, i, j] * invρ
#         E = U[4, i, j] * invρ
#         p = ρ * EOS._γ_m_1 * (E - 0.5(u^2 + v^2))

#         W[1, i, j] = ρ
#         W[2, i, j] = u
#         W[3, i, j] = v
#         W[4, i, j] = p
#       end
#     end
#   end
#   return
# end

function cons2prim_stride!(U, W, EOS)
  i = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  #stride_x = blockDim().x * gridDim().x
  #stride_y = blockDim().y * gridDim().y
  i1 = i + blockDim().y * gridDim().y
  j1 = j + blockDim().x * gridDim().x

  @inbounds begin
  #for i in row:stride_y:2048, j in col:stride_x:2048
    ρ = U[1, i, j]
    invρ = 1 / ρ
    u = U[2, i, j] * invρ
    v = U[3, i, j] * invρ
    E = U[4, i, j] * invρ
    p = ρ * EOS._γ_m_1 * (E - 0.5(u^2 + v^2))

    W[1, i, j] = ρ
    W[2, i, j] = u
    W[3, i, j] = v
    W[4, i, j] = p

    ρ = U[1, i, j1]
    invρ = 1 / ρ
    u = U[2, i, j1] * invρ
    v = U[3, i, j1] * invρ
    E = U[4, i, j1] * invρ
    p = ρ * EOS._γ_m_1 * (E - 0.5(u^2 + v^2))

    W[1, i, j1] = ρ
    W[2, i, j1] = u
    W[3, i, j1] = v
    W[4, i, j1] = p

    ρ = U[1, i1, j]
    invρ = 1 / ρ
    u = U[2, i1, j] * invρ
    v = U[3, i1, j] * invρ
    E = U[4, i1, j] * invρ
    p = ρ * EOS._γ_m_1 * (E - 0.5(u^2 + v^2))

    W[1, i1, j] = ρ
    W[2, i1, j] = u
    W[3, i1, j] = v
    W[4, i1, j] = p

    ρ = U[1, i1, j1]
    invρ = 1 / ρ
    u = U[2, i1, j1] * invρ
    v = U[3, i1, j1] * invρ
    E = U[4, i1, j1] * invρ
    p = ρ * EOS._γ_m_1 * (E - 0.5(u^2 + v^2))

    W[1, i1, j1] = ρ
    W[2, i1, j1] = u
    W[3, i1, j1] = v
    W[4, i1, j1] = p
   #end
  end
  return
end

function cons2prim_stride_splitUW!(U1, U2, U3, U4, W1, W2, W3, W4, EOS)
  i = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  #stride_x = blockDim().x * gridDim().x
  #stride_y = blockDim().y * gridDim().y
  i1 = i + blockDim().y * gridDim().y
  j1 = j + blockDim().x * gridDim().x

  @inbounds begin
  #for i in row:stride_y:2048, j in col:stride_x:2048
    ρ = U1[i, j]
    invρ = 1 / ρ
    u = U2[i, j] * invρ
    v = U3[i, j] * invρ
    E = U4[i, j] * invρ
    p = ρ * EOS._γ_m_1 * (E - 0.5(u^2 + v^2))

    W1[i, j] = ρ
    W2[i, j] = u
    W3[i, j] = v
    W4[i, j] = p

    ρ = U1[i, j1]
    invρ = 1 / ρ
    u = U2[i, j1] * invρ
    v = U3[i, j1] * invρ
    E = U4[i, j1] * invρ
    p = ρ * EOS._γ_m_1 * (E - 0.5(u^2 + v^2))

    W1[i, j1] = ρ
    W2[i, j1] = u
    W3[i, j1] = v
    W4[i, j1] = p

    ρ = U1[i1, j]
    invρ = 1 / ρ
    u = U2[i1, j] * invρ
    v = U3[i1, j] * invρ
    E = U4[i1, j] * invρ
    p = ρ * EOS._γ_m_1 * (E - 0.5(u^2 + v^2))

    W1[i1, j] = ρ
    W2[i1, j] = u
    W3[i1, j] = v
    W4[i1, j] = p

    ρ = U1[i1, j1]
    invρ = 1 / ρ
    u = U2[i1, j1] * invρ
    v = U3[i1, j1] * invρ
    E = U4[i1, j1] * invρ
    p = ρ * EOS._γ_m_1 * (E - 0.5(u^2 + v^2))

    W1[i1, j1] = ρ
    W2[i1, j1] = u
    W3[i1, j1] = v
    W4[i1, j1] = p
   #end
  end
  return
end

function cons2prim_splitUW!(U1, U2, U3, U4, W1, W2, W3, W4, EOS)
  i = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  @inbounds begin
    ρ = U1[i, j]
    invρ = 1 / ρ
    u = U2[i, j] * invρ
    v = U3[i, j] * invρ
    E = U4[i, j] * invρ
    p = ρ * EOS._γ_m_1 * (E - 0.5(u^2 + v^2))

    W1[i, j] = ρ
    W2[i, j] = u
    W3[i, j] = v
    W4[i, j] = p
  end
  return
end

## Launch this kernel with blkdim_x = 64, blkdim_y = 4
function cons2prim_splitUW_tp!(U1, U2, U3, U4, W1, W2, W3, W4, EOS)
  i = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  @inbounds begin
    ρ = U1[j, i]
    invρ = 1 / ρ
    u = U2[j, i] * invρ
    v = U3[j, i] * invρ
    E = U4[j, i] * invρ
    p = ρ * EOS._γ_m_1 * (E - 0.5(u^2 + v^2))

    W1[j, i] = ρ
    W2[j, i] = u
    W3[j, i] = v
    W4[j, i] = p
  end
  return
end

function muscl_gmem_cpu!(W::AbstractArray{T,N}, i_face, j_face, limits) where {T,N}
  #i, j = @index(Global, NTuple)
  # i = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  # j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  ilo, ihi, jlo, jhi = limits

  SMALL = T(1e-30)
  minmod(r) = max(0, min(r, 1))

  # i face
  Threads.@threads for j in jlo:jhi
    for i in ilo:ihi
      for q in 1:4
        #W[q, i, j] = W[q, i, j]
        #W[q, i + 1, j] = W[q, i + 1, j]

        Δi_minus_half = W[q, i, j] - W[q, i - 1, j]
        Δi_plus_half = W[q, i + 1, j] - W[q, i, j]
        Δi_plus_three_half = W[q, i + 2, j] - W[q, i + 1, j]

        rLi = Δi_plus_half / (Δi_minus_half + SMALL)
        rRi = Δi_plus_half / (Δi_plus_three_half + SMALL)

        lim_Li = minmod(rLi)
        lim_Ri = minmod(rRi)

        i_face[1, q, i, j] = W[q, i, j] + lim_Li / 2 * Δi_minus_half
        i_face[2, q, i, j] = W[q, i + 1, j] - lim_Ri / 2 * Δi_plus_three_half
      end

      # j face   
      for q in 1:4
        Δj_minus_half = W[q, i, j] - W[q, i, j - 1]
        Δj_plus_half = W[q, i, j + 1] - W[q, i, j]
        Δj_plus_three_half = W[q, i, j + 2] - W[q, i, j + 1]

        rLj = Δj_plus_half / (Δj_minus_half + SMALL)
        rRj = Δj_plus_half / (Δj_plus_three_half + SMALL)

        lim_Lj = minmod(rLj)
        lim_Rj = minmod(rRj)

        j_face[1, 1, i, j] = W[q, i, j] + lim_Lj / 2 * Δj_minus_half
        j_face[2, 1, i, j] = W[q, i, j + 1] - lim_Rj / 2 * Δj_plus_three_half
      end
    end
  end
  return nothing
end

### Insted of the if conditions, launch only 2048x2048 threads and do a i+2, j+2
### Refer muscl_gmem_i_Wsplit
function muscl_gmem!(W::AbstractArray{T,N}, i_face, j_face, limits) where {T,N}
  #i, j = @index(Global, NTuple)
  i = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  ilo, ihi, jlo, jhi = limits

  SMALL = T(1e-30)
  minmod(r) = max(0, min(r, 1))

  # i face
  if (jlo <= j <= jhi) && (ilo - 1 <= i <= ihi)
    @inbounds for q in 1:4
      W[q, i, j] = W[q, i, j]
      W[q, i + 1, j] = W[q, i + 1, j]

      Δi_minus_half = W[q, i, j] - W[q, i - 1, j]
      Δi_plus_half = W[q, i + 1, j] - W[q, i, j]
      Δi_plus_three_half = W[q, i + 2, j] - W[q, i + 1, j]

      rLi = Δi_plus_half / (Δi_minus_half + SMALL)
      rRi = Δi_plus_half / (Δi_plus_three_half + SMALL)

      lim_Li = minmod(rLi)
      lim_Ri = minmod(rRi)

      i_face[1, q, i, j] = W[q, i, j] + lim_Li / 2 * Δi_minus_half
      i_face[2, q, i, j] = W[q, i + 1, j] - lim_Ri / 2 * Δi_plus_three_half
    end
  end

  # j face
  if (jlo - 1 <= j <= jhi) && (ilo <= i <= ihi)
    @inbounds for q in 1:4
      Δj_minus_half = W[q, i, j] - W[q, i, j - 1]
      Δj_plus_half = W[q, i, j + 1] - W[q, i, j]
      Δj_plus_three_half = W[q, i, j + 2] - W[q, i, j + 1]

      rLj = Δj_plus_half / (Δj_minus_half + SMALL)
      rRj = Δj_plus_half / (Δj_plus_three_half + SMALL)

      lim_Lj = minmod(rLj)
      lim_Rj = minmod(rRj)

      j_face[1, q, i, j] = W[q, i, j] + lim_Lj / 2 * Δj_minus_half
      j_face[2, q, i, j] = W[q, i, j + 1] - lim_Rj / 2 * Δj_plus_three_half
    end
  end
  return
end

function muscl_gmem_i!(W::AbstractArray{T,N}, i_face, limits) where {T,N}
  #i, j = @index(Global, NTuple)
  i = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  ilo, ihi, jlo, jhi = limits

  SMALL = T(1e-30)
  minmod(r) = max(0, min(r, 1))

  # i face
  if (jlo <= j <= jhi) && (ilo - 1 <= i <= ihi)
    @inbounds for q in 1:4
      W[q, i, j] = W[q, i, j]
      W[q, i + 1, j] = W[q, i + 1, j]

      Δi_minus_half = W[q, i, j] - W[q, i - 1, j]
      Δi_plus_half = W[q, i + 1, j] - W[q, i, j]
      Δi_plus_three_half = W[q, i + 2, j] - W[q, i + 1, j]

      rLi = Δi_plus_half / (Δi_minus_half + SMALL)
      rRi = Δi_plus_half / (Δi_plus_three_half + SMALL)

      lim_Li = minmod(rLi)
      lim_Ri = minmod(rRi)

      i_face[1, q, i, j] = W[q, i, j] + lim_Li / 2 * Δi_minus_half
      i_face[2, q, i, j] = W[q, i + 1, j] - lim_Ri / 2 * Δi_plus_three_half
    end
  end
  return
end

## Launch this kernel with threads packed in y-direction.
function muscl_gmem_i_Wsplit!(W::AbstractArray{T,N}, i_face, limits) where {T,N}
  #i, j = @index(Global, NTuple)
  i = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  i = i + 2
  j = j + 2
  ilo, ihi, jlo, jhi = limits

  SMALL = T(1e-30)
  minmod(r) = max(0, min(r, 1))

  # i face
  #if (jlo <= j <= jhi) && (ilo - 1 <= i <= ihi)
  @inbounds begin
    elem1 = W[i, j]
    elem2 = W[i + 1, j]
    elem3 = W[i - 1, j]
    elem4 = W[i + 2, j]

    Δi_minus_half = elem1 - elem3
    Δi_plus_half = elem2 - elem1
    Δi_plus_three_half = elem4 - elem2

    rLi = Δi_plus_half / (Δi_minus_half + SMALL)
    rRi = Δi_plus_half / (Δi_plus_three_half + SMALL)

    lim_Li = minmod(rLi)
    lim_Ri = minmod(rRi)

    i_face[1, i, j] = elem1 + lim_Li / 2 * Δi_minus_half
    i_face[2, i, j] = elem2 - lim_Ri / 2 * Δi_plus_three_half
  end
  #end
  return
end


## Launch this kernel with threads packed heavily in x-direction
## The thread ids are transposed to cater to column major data layout
function muscl_gmem_i_Wsplit_tp!(W::AbstractArray{T,N}, i_face, limits) where {T,N}
  #i, j = @index(Global, NTuple)
  i = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  i = i + 2
  j = j + 2
  ilo, ihi, jlo, jhi = limits

  SMALL = T(1e-30)
  minmod(r) = max(0, min(r, 1))

  # i face
  #if (jlo <= j <= jhi) && (ilo - 1 <= i <= ihi)
  @inbounds begin
    elem1 = W[j, i]
    elem2 = W[j + 1, i]
    elem3 = W[j - 1, i]
    elem4 = W[j + 2, i]

    Δi_minus_half = elem1 - elem3
    Δi_plus_half = elem2 - elem1
    Δi_plus_three_half = elem4 - elem2

    rLi = Δi_plus_half / (Δi_minus_half + SMALL)
    rRi = Δi_plus_half / (Δi_plus_three_half + SMALL)

    lim_Li = minmod(rLi)
    lim_Ri = minmod(rRi)

    i_face[1, j, i] = elem1 + lim_Li / 2 * Δi_minus_half
    i_face[2, j, i] = elem2 - lim_Ri / 2 * Δi_plus_three_half
  end
  #end
  return
end

function muscl_gmem_i_Wsplit_tpshufl!(W::AbstractArray{T,N}, iface, limits) where {T,N}
  #i, j = @index(Global, NTuple)
  i = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  # i = i + 2
  # j = j + 2
  ilo, ihi, jlo, jhi = limits

  SMALL = T(1e-30)
  minmod(r) = max(0, min(r, 1))

  @inbounds elem1 = W[j, i]
  elem2 = CUDA.shfl_down_sync(0xffffffff, elem1, 1)
  elem3 = CUDA.shfl_up_sync(0xffffffff, elem1, 1)
  elem4 = CUDA.shfl_down_sync(0xffffffff, elem1, 2)
  # i face
  #if (jlo <= j <= jhi) && (ilo - 1 <= i <= ihi)
  # @inbounds begin
  #   elem1 = W[j, i]
  #   elem2 = W[j + 1, i]
  #   elem3 = W[j - 1, i]
  #   elem4 = W[j + 2, i]

  Δi_minus_half = elem1 - elem3
  Δi_plus_half = elem2 - elem1
  Δi_plus_three_half = elem4 - elem2

  rLi = Δi_plus_half / (Δi_minus_half + SMALL)
  rRi = Δi_plus_half / (Δi_plus_three_half + SMALL)

  lim_Li = minmod(rLi)
  lim_Ri = minmod(rRi)

  @inbounds begin
    # i1_face[j, i] = elem1 + lim_Li / 2 * Δi_minus_half
    # i2_face[j, i] = elem2 - lim_Ri / 2 * Δi_plus_three_half
    iface[1, j, i] = elem1 + lim_Li / 2 * Δi_minus_half
    iface[2, j, i] = elem2 - lim_Ri / 2 * Δi_plus_three_half
  end
  #end
  #end
  return
end

function muscl_gmem_i_Wsplit_tpshufl!(W::AbstractArray{T,N}, iface, limits) where {T,N}
  #i, j = @index(Global, NTuple)
  i = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  # i = i + 2
  # j = j + 2
  ilo, ihi, jlo, jhi = limits

  SMALL = T(1e-30)
  minmod(r) = max(0, min(r, 1))

  @inbounds elem1 = W[j, i]
  elem2 = CUDA.shfl_down_sync(0xffffffff, elem1, 1)
  elem3 = CUDA.shfl_up_sync(0xffffffff, elem1, 1)
  elem4 = CUDA.shfl_down_sync(0xffffffff, elem1, 2)
  # i face
  #if (jlo <= j <= jhi) && (ilo - 1 <= i <= ihi)
  # @inbounds begin
  #   elem1 = W[j, i]
  #   elem2 = W[j + 1, i]
  #   elem3 = W[j - 1, i]
  #   elem4 = W[j + 2, i]

  Δi_minus_half = elem1 - elem3
  Δi_plus_half = elem2 - elem1
  Δi_plus_three_half = elem4 - elem2

  rLi = Δi_plus_half / (Δi_minus_half + SMALL)
  rRi = Δi_plus_half / (Δi_plus_three_half + SMALL)

  lim_Li = minmod(rLi)
  lim_Ri = minmod(rRi)

  @inbounds begin
    # i1_face[j, i] = elem1 + lim_Li / 2 * Δi_minus_half
    # i2_face[j, i] = elem2 - lim_Ri / 2 * Δi_plus_three_half
    iface[1, j, i] = elem1 + lim_Li / 2 * Δi_minus_half
    iface[2, j, i] = elem2 - lim_Ri / 2 * Δi_plus_three_half
  end
  #end
  #end
  return
end

function muscl_shmem_i_Wsplit!(W::AbstractArray{T,N}, i1_face, i2_face, limits) where {T,N}
  #i, j = @index(Global, NTuple)
  row = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  col = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  ilo, ihi, jlo, jhi = limits
  ker_width = 2

  blkdim_x = blockDim().x
  blkdim_y = blockDim().y
  new_row = row + ker_width
  new_col = col + ker_width
  tid_x = threadIdx().x
  tid_y = threadIdx().y

  W_type = eltype(W)
  #iface_type = eltype(i_face)
  ## Halo region only in y direction
  #w_sh = CUDA.CuDynamicSharedArray(W_type, (blkdim_y + 2*ker_width, blkdim_x))
  w_sh = CUDA.CuDynamicSharedArray(W_type, (blkdim_x, blkdim_y + 2*ker_width))
  #w_sh = CUDA.CuStaticSharedArray(W_type, (4, 64 + 2*ker_width))
  #iface_sh = CUDA.CuStaticArray(iface_type, (2, blkdim_y + 2*ker_width, blkdim_x))

  #sh_arr = CUDA.CuStaticArray(arr_type, (blkdim_y + 4, blkdim_x + 4))
  
  ### Populating Shared Memory
  ### Populating the non-halo region of Shared Memory

  # w_sh[tid_y + ker_width, tid_x] = W[new_row, new_col]
  # w_sh[tid_y + ker_width, tid_x] = W[row, col]
  @inbounds begin
    w_sh[tid_x, tid_y + ker_width] = W[new_row, new_col]
    # w_sh[tid_x, tid_y + ker_width] = W[new_col, new_row]
    ### Populating the halo regions of shared memory ( Note the halo is only in y direction)
    if tid_y in 1:ker_width
      #w_sh[tid_y, tid_x] = W[new_row - ker_width, new_col]
      w_sh[tid_x, tid_y] = W[new_row - ker_width, new_col]
    end
    #if(tid_y == (blk_dim-1) || tid_y == blk_dim)
    if tid_y in (blkdim_y - ker_width + 1):blkdim_y
      #w_sh[tid_y + 2*ker_width, tid_x] = W[new_row + ker_width, new_col]
      w_sh[tid_x, tid_y + 2*ker_width] = W[new_row + ker_width, new_col]
    end
  end
  # if tid_x in 1:ker_width
  #   w_sh[tid_y + ker_width, tid_x] = W[new_row, new_col - ker_width]
  # end
  # if tid_x in (blkdim_x - ker_width + 1):blkdim_x
  #   w_sh[tid_y + ker_width, tid_x + 2*ker_width] = W[new_row, new_col + ker_width]
  # end
  sync_threads()

  SMALL = T(1e-30)
  minmod(r) = max(0, min(r, 1))

  # i face
  #if (jlo <= j <= jhi) && (ilo - 1 <= i <= ihi)
  # elem1 = w_sh[tid_y + ker_width, tid_x]
  # elem2 = w_sh[tid_y + ker_width - 1, tid_x]
  # elem3 = w_sh[tid_y + ker_width + 1, tid_x]
  # elem4 = w_sh[tid_y + ker_width + 2, tid_x]
  @inbounds begin
    elem1 = w_sh[tid_x, tid_y + ker_width]
    elem2 = w_sh[tid_x, tid_y + ker_width - 1]
    elem3 = w_sh[tid_x, tid_y + ker_width + 1]
    elem4 = w_sh[tid_x, tid_y + ker_width + 2]
  end
  Δi_minus_half = elem1 - elem2
  Δi_plus_half = elem3 - elem1
  Δi_plus_three_half = elem4 - elem3

  rLi = Δi_plus_half / (Δi_minus_half + SMALL)
  rRi = Δi_plus_half / (Δi_plus_three_half + SMALL)

  lim_Li = minmod(rLi)
  lim_Ri = minmod(rRi)

  # iface_sh[1, tid_y, tid_x] = elem1 + lim_Li / 2 * Δi_minus_half
  # iface_sh[2, tid_y, tid_x] = elem3 - lim_Ri / 2 * Δi_plus_three_half
  @inbounds begin
    i1_face[row, col] = elem1 + lim_Li / 2 * Δi_minus_half
    i2_face[row, col] = elem3 - lim_Ri / 2 * Δi_plus_three_half
  end
  #i_face[1, row, col] = elem1 + lim_Li / 2 * Δi_minus_half
  #i_face[2, row, col] = elem3 - lim_Ri / 2 * Δi_plus_three_half

  return
end

function muscl_shmem_i_Wsplit_tp!(W::AbstractArray{T,N}, i1_face, i2_face, limits) where {T,N}
  #i, j = @index(Global, NTuple)
  row = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  col = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  ilo, ihi, jlo, jhi = limits
  ker_width = 2

  blkdim_x = blockDim().x
  blkdim_y = blockDim().y
  new_row = row + ker_width
  new_col = col + ker_width
  tid_x = threadIdx().x
  tid_y = threadIdx().y

  W_type = eltype(W)
  #iface_type = eltype(i_face)
  ## Halo region only in y direction
  #w_sh = CUDA.CuDynamicSharedArray(W_type, (blkdim_y + 2*ker_width, blkdim_x))
  w_sh = CUDA.CuDynamicSharedArray(W_type, (blkdim_y, blkdim_x + 2*ker_width))
  #w_sh = CUDA.CuStaticSharedArray(W_type, (4, 64 + 2*ker_width))
  #iface_sh = CUDA.CuStaticArray(iface_type, (2, blkdim_y + 2*ker_width, blkdim_x))

  #sh_arr = CUDA.CuStaticArray(arr_type, (blkdim_y + 4, blkdim_x + 4))
  
  ### Populating Shared Memory
  ### Populating the non-halo region of Shared Memory

  # w_sh[tid_y + ker_width, tid_x] = W[new_row, new_col]
  # w_sh[tid_y + ker_width, tid_x] = W[row, col]
  @inbounds begin
    w_sh[tid_y, tid_x + ker_width] = W[new_col, new_row]
    ### Populating the halo regions of shared memory ( Note the halo is only in y direction)
    if tid_x in 1:ker_width
      #w_sh[tid_y, tid_x] = W[new_row - ker_width, new_col]
      w_sh[tid_y, tid_x] = W[col, new_row]
    end
    #if(tid_y == (blk_dim-1) || tid_y == blk_dim)
    if tid_x in (blkdim_x - ker_width + 1):blkdim_x
      #w_sh[tid_y + 2*ker_width, tid_x] = W[new_row + ker_width, new_col]
      w_sh[tid_y, tid_x + 2*ker_width] = W[new_col + ker_width, new_row]
    end
  end
  # if tid_x in 1:ker_width
  #   w_sh[tid_y + ker_width, tid_x] = W[new_row, new_col - ker_width]
  # end
  # if tid_x in (blkdim_x - ker_width + 1):blkdim_x
  #   w_sh[tid_y + ker_width, tid_x + 2*ker_width] = W[new_row, new_col + ker_width]
  # end
  sync_threads()

  SMALL = T(1e-30)
  minmod(r) = max(0, min(r, 1))

  # i face
  #if (jlo <= j <= jhi) && (ilo - 1 <= i <= ihi)
  # elem1 = w_sh[tid_y + ker_width, tid_x]
  # elem2 = w_sh[tid_y + ker_width - 1, tid_x]
  # elem3 = w_sh[tid_y + ker_width + 1, tid_x]
  # elem4 = w_sh[tid_y + ker_width + 2, tid_x]
  @inbounds begin
    elem1 = w_sh[tid_y, tid_x + ker_width]
    elem2 = w_sh[tid_y, tid_x + ker_width - 1]
    elem3 = w_sh[tid_y, tid_x + ker_width + 1]
    elem4 = w_sh[tid_y, tid_x + ker_width + 2]
  end
  Δi_minus_half = elem1 - elem2
  Δi_plus_half = elem3 - elem1
  Δi_plus_three_half = elem4 - elem3

  rLi = Δi_plus_half / (Δi_minus_half + SMALL)
  rRi = Δi_plus_half / (Δi_plus_three_half + SMALL)

  lim_Li = minmod(rLi)
  lim_Ri = minmod(rRi)

  # iface_sh[1, tid_y, tid_x] = elem1 + lim_Li / 2 * Δi_minus_half
  # iface_sh[2, tid_y, tid_x] = elem3 - lim_Ri / 2 * Δi_plus_three_half
  @inbounds begin
    i1_face[col, row] = elem1 + lim_Li / 2 * Δi_minus_half
    i2_face[col, row] = elem3 - lim_Ri / 2 * Δi_plus_three_half
  end
  #i_face[1, row, col] = elem1 + lim_Li / 2 * Δi_minus_half
  #i_face[2, row, col] = elem3 - lim_Ri / 2 * Δi_plus_three_half

  return
end

function muscl_shmem_i_Wsplit_tp_var!(W::AbstractArray{T,N}, i1_face, i2_face, limits) where {T,N}
  #i, j = @index(Global, NTuple)
  row = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  col = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  ilo, ihi, jlo, jhi = limits
  ker_width = 2

  blkdim_x = blockDim().x
  blkdim_y = blockDim().y
  new_row = row + ker_width
  new_col = col + ker_width
  tid_x = threadIdx().x
  tid_y = threadIdx().y

  W_type = eltype(W)
  #iface_type = eltype(i_face)
  ## Halo region only in y direction
  w_sh = CUDA.CuDynamicSharedArray(W_type, (blkdim_x + 2*ker_width, blkdim_y))
  #w_sh = CUDA.CuDynamicSharedArray(W_type, (blkdim_y, blkdim_x + 2*ker_width))
  
  ### Populating Shared Memory
  ### Populating the non-halo region of Shared Memory

  # w_sh[tid_y + ker_width, tid_x] = W[new_row, new_col]
  # w_sh[tid_y + ker_width, tid_x] = W[row, col]
  @inbounds begin
    w_sh[tid_x + ker_width, tid_y] = W[new_col, new_row]
    ### Populating the halo regions of shared memory ( Note the halo is only in y direction)
    if tid_x in 1:ker_width
      #w_sh[tid_y, tid_x] = W[new_row - ker_width, new_col]
      w_sh[tid_x, tid_y] = W[col, new_row]
    end
    #if(tid_y == (blk_dim-1) || tid_y == blk_dim)
    if tid_x in (blkdim_x - ker_width + 1):blkdim_x
      #w_sh[tid_y + 2*ker_width, tid_x] = W[new_row + ker_width, new_col]
      w_sh[tid_x + 2*ker_width, tid_y] = W[new_col + ker_width, new_row]
    end
  end
  # if tid_x in 1:ker_width
  #   w_sh[tid_y + ker_width, tid_x] = W[new_row, new_col - ker_width]
  # end
  # if tid_x in (blkdim_x - ker_width + 1):blkdim_x
  #   w_sh[tid_y + ker_width, tid_x + 2*ker_width] = W[new_row, new_col + ker_width]
  # end
  sync_threads()

  SMALL = T(1e-30)
  minmod(r) = max(0, min(r, 1))

  # i face
  #if (jlo <= j <= jhi) && (ilo - 1 <= i <= ihi)
  # elem1 = w_sh[tid_y + ker_width, tid_x]
  # elem2 = w_sh[tid_y + ker_width - 1, tid_x]
  # elem3 = w_sh[tid_y + ker_width + 1, tid_x]
  # elem4 = w_sh[tid_y + ker_width + 2, tid_x]
  @inbounds begin
    elem1 = w_sh[tid_x + ker_width, tid_y]
    elem2 = w_sh[tid_x + ker_width - 1, tid_y]
    elem3 = w_sh[tid_x + ker_width + 1, tid_y]
    elem4 = w_sh[tid_x + ker_width + 2, tid_y]
  end
  Δi_minus_half = elem1 - elem2
  Δi_plus_half = elem3 - elem1
  Δi_plus_three_half = elem4 - elem3

  rLi = Δi_plus_half / (Δi_minus_half + SMALL)
  rRi = Δi_plus_half / (Δi_plus_three_half + SMALL)

  lim_Li = minmod(rLi)
  lim_Ri = minmod(rRi)

  # iface_sh[1, tid_y, tid_x] = elem1 + lim_Li / 2 * Δi_minus_half
  # iface_sh[2, tid_y, tid_x] = elem3 - lim_Ri / 2 * Δi_plus_three_half
  @inbounds begin
    i1_face[col, row] = elem1 + lim_Li / 2 * Δi_minus_half
    i2_face[col, row] = elem3 - lim_Ri / 2 * Δi_plus_three_half
  end
  #i_face[1, row, col] = elem1 + lim_Li / 2 * Δi_minus_half
  #i_face[2, row, col] = elem3 - lim_Ri / 2 * Δi_plus_three_half

  return
end

## Launch this kernel with threads packed in y-direction.
function muscl_gmem_j!(W::AbstractArray{T,N}, j_face, limits) where {T,N}
  #i, j = @index(Global, NTuple)
  i = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  ilo, ihi, jlo, jhi = limits

  SMALL = T(1e-30)
  minmod(r) = max(0, min(r, 1))

  if (jlo - 1 <= j <= jhi) && (ilo <= i <= ihi)
    @inbounds for q in 1:4
      Δj_minus_half = W[q, i, j] - W[q, i, j - 1]
      Δj_plus_half = W[q, i, j + 1] - W[q, i, j]
      Δj_plus_three_half = W[q, i, j + 2] - W[q, i, j + 1]

      rLj = Δj_plus_half / (Δj_minus_half + SMALL)
      rRj = Δj_plus_half / (Δj_plus_three_half + SMALL)

      lim_Lj = minmod(rLj)
      lim_Rj = minmod(rRj)

      j_face[1, q, i, j] = W[q, i, j] + lim_Lj / 2 * Δj_minus_half
      j_face[2, q, i, j] = W[q, i, j + 1] - lim_Rj / 2 * Δj_plus_three_half
    end
  end
  return
end

function muscl_gmem_j_Wsplit!(W::AbstractArray{T,N}, j_face, limits) where {T,N}
  #i, j = @index(Global, NTuple)
  i = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  i = i + 2
  j = j + 2
  ilo, ihi, jlo, jhi = limits

  SMALL = T(1e-30)
  minmod(r) = max(0, min(r, 1))

  #if (jlo - 1 <= j <= jhi) && (ilo <= i <= ihi)
  @inbounds begin
    Δj_minus_half = W[i, j] - W[i, j - 1]
    Δj_plus_half = W[i, j + 1] - W[i, j]
    Δj_plus_three_half = W[i, j + 2] - W[i, j + 1]

    rLj = Δj_plus_half / (Δj_minus_half + SMALL)
    rRj = Δj_plus_half / (Δj_plus_three_half + SMALL)

    lim_Lj = minmod(rLj)
    lim_Rj = minmod(rRj)

    j_face[1, i, j] = W[i, j] + lim_Lj / 2 * Δj_minus_half
    j_face[2, i, j] = W[i, j + 1] - lim_Rj / 2 * Δj_plus_three_half
  end
  #end
  return
end

## Launch this kernel with threads packed heavily in x-direction
## The thread ids are transposed to cater to column major data layout
function muscl_gmem_j_Wsplit_tp!(W::AbstractArray{T,N}, j_face, limits) where {T,N}
  #i, j = @index(Global, NTuple)
  i = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  i = i + 2
  j = j + 2
  ilo, ihi, jlo, jhi = limits

  SMALL = T(1e-30)
  minmod(r) = max(0, min(r, 1))

  #if (jlo - 1 <= j <= jhi) && (ilo <= i <= ihi)
  @inbounds begin
    Δj_minus_half = W[j, i] - W[j, i - 1]
    Δj_plus_half = W[j, i + 1] - W[j, i]
    Δj_plus_three_half = W[j, i + 2] - W[j, i + 1]

    rLj = Δj_plus_half / (Δj_minus_half + SMALL)
    rRj = Δj_plus_half / (Δj_plus_three_half + SMALL)

    lim_Lj = minmod(rLj)
    lim_Rj = minmod(rRj)

    j_face[1, j, i] = W[j, i] + lim_Lj / 2 * Δj_minus_half
    j_face[2, j, i] = W[j, i + 1] - lim_Rj / 2 * Δj_plus_three_half
  end
  #end
  return
end

function sum_fluxes(
  dUdt::AbstractArray{T,N}, iflux, jflux, facelen, volume, limits
) where {T,N}
  #i, j = @index(Global, NTuple)
  i = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  ilo, ihi, jlo, jhi = limits

  ΔS1, ΔS2, ΔS3, ΔS4 = @view facelen[1:4, i, j]
  @inbounds begin
    if (jlo <= j <= jhi) && (ilo <= i <= ihi)
      for q in 1:4
        dUdt[q, i, j] =
          (
            jflux[q, i, j - 1] * ΔS1 +
            iflux[q, i, j] * ΔS2 +
            jflux[q, i, j] * ΔS3 +
            iflux[q, i - 1, j] * ΔS4
          ) / volume[i, j]
      end
    end
  end
  return
end

function _next_Δt(U1::AbstractArray{T}, U2::AbstractArray{T}, U3::AbstractArray{T}, U4::AbstractArray{T}, glb_dt::AbstractArray{T}, facelen, volume, norms, EOS) where {T}

  i = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  
  ## Shared mem of size blockdim.x * blockdim.y OR the Block Size
  ## linear tid in a block: blockdim.x * (tid_y - 1) + tid_x
  blk_size = blockDim().x * blockDim().y
  th_id = blockDim().x * (threadIdx().y - 1) + threadIdx().x
  blk_id = gridDim().x * (blockIdx().y - 1) + blockIdx().x

  arr_type = eltype(U1)
  sh_dt = CUDA.CuDynamicSharedArray(arr_type, blk_size)

  ##Δt_min = Inf

  ΔS_face = facelen
  vol = volume
  norms = norms

  # for j in jlo:jhi
  #   for i in ilo:ihi
      n̂ = view(norms, :, :, i, j)
      n1x = n̂[1, 1]
      n1y = n̂[2, 1]
      n2x = n̂[1, 2]
      n2y = n̂[2, 2]
      n3x = n̂[1, 3]
      n3y = n̂[2, 3]
      n4x = n̂[1, 4]
      n4y = n̂[2, 4]

      s1 = ΔS_face[1, i, j]
      s2 = ΔS_face[2, i, j]
      s3 = ΔS_face[3, i, j]
      s4 = ΔS_face[4, i, j]

      ρ = U1[i, j]
      u = U2[i, j] / ρ
      v = U3[i, j] / ρ
      E = U4[i, j] / ρ

      p = pressure(EOS, ρ, u, v, E)
      cs = sound_speed(EOS, ρ, p)

      ave_n_i = @SVector [0.5(n2x - n4x), 0.5(n2y - n4y)]
      ave_ds_i = 0.5(s2 + s4)

      v_i = abs(ave_n_i[1] * u + ave_n_i[2] * v)
      spec_radius_i = (v_i + cs) * ave_ds_i

      ave_n_j = @SVector [0.5(n3x - n1x), 0.5(n3y - n1y)]
      ave_ds_j = 0.5(s1 + s3)

      v_j = abs(ave_n_j[1] * u + ave_n_j[2] * v)
      spec_radius_j = (v_j + cs) * ave_ds_j

      Δt_tid = vol[i, j] / (spec_radius_i + spec_radius_j)
      sh_dt[th_id] = Δt_tid

      sync_threads()
      ##Δt_min = min(Δt_tid, Δt_min)
      s = fld(blk_size, 2)
      while s > 0
        if (th_id - 1) < s
          sh_dt[th_id] = min(sh_dt[th_id], sh_dt[th_id + s])
        end
        s = fld(s,2)
        sync_threads()
      end

      if th_id == 1
        glb_dt[blk_id] = sh_dt[th_id]
      end

  return
end

## Function to calculate dt for each cell and store it in a global array
function calc_Δt(U1::AbstractArray{T}, U2::AbstractArray{T}, U3::AbstractArray{T}, U4::AbstractArray{T}, glb_dt::AbstractArray{T}, facelen, volume, norms, EOS) where {T}
  
  i = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  ## Shared mem of size blockdim.x * blockdim.y OR the Block Size
  ## linear tid in a block: blockdim.x * (tid_y - 1) + tid_x
  # blk_size = blockDim().x * blockDim().y
  # th_id = blockDim().x * (threadIdx().y - 1) + threadIdx().x
  # blk_id = gridDim().x * (blockIdx().y - 1) + blockIdx().x

  # arr_type = eltype(U1)
  # sh_dt = CUDA.CuDynamicSharedArray(arr_type, blk_size)

  ##Δt_min = Inf

  ΔS_face = facelen
  vol = volume
  norms = norms

  # for j in jlo:jhi
  #   for i in ilo:ihi
      n̂ = view(norms, :, :, i, j)
      n1x = n̂[1, 1]
      n1y = n̂[2, 1]
      n2x = n̂[1, 2]
      n2y = n̂[2, 2]
      n3x = n̂[1, 3]
      n3y = n̂[2, 3]
      n4x = n̂[1, 4]
      n4y = n̂[2, 4]

      s1 = ΔS_face[1, i, j]
      s2 = ΔS_face[2, i, j]
      s3 = ΔS_face[3, i, j]
      s4 = ΔS_face[4, i, j]

      ρ = U1[i, j]
      u = U2[i, j] / ρ
      v = U3[i, j] / ρ
      E = U4[i, j] / ρ

      p = pressure(EOS, ρ, u, v, E)
      cs = sound_speed(EOS, ρ, p)

      ave_n_i = @SVector [0.5(n2x - n4x), 0.5(n2y - n4y)]
      ave_ds_i = 0.5(s2 + s4)

      v_i = abs(ave_n_i[1] * u + ave_n_i[2] * v)
      spec_radius_i = (v_i + cs) * ave_ds_i

      ave_n_j = @SVector [0.5(n3x - n1x), 0.5(n3y - n1y)]
      ave_ds_j = 0.5(s1 + s3)

      v_j = abs(ave_n_j[1] * u + ave_n_j[2] * v)
      spec_radius_j = (v_j + cs) * ave_ds_j

      Δt_tid = vol[i, j] / (spec_radius_i + spec_radius_j)
      glb_dt[i,j] = Δt_tid
      # sh_dt[th_id] = Δt_tid

      # sync_threads()
      # ##Δt_min = min(Δt_tid, Δt_min)
      # s = fld(blk_size, 2)
      # while s > 0
      #   if (th_id - 1) < s
      #     sh_dt[th_id] = min(sh_dt[th_id], sh_dt[th_id + s])
      #   end
      #   s = fld(s,2)
      #   sync_threads()
      # end

      # if th_id == 1
      #   glb_dt[blk_id] = sh_dt[th_id]
      # end

  return
end

function reduce_dt(glb_min_dt::AbstractArray{T}, glb_dt::AbstractArray{T}) where {T}
  i = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  ## Shared mem of size blockdim.x * blockdim.y OR the Block Size
  ## linear tid in a block: blockdim.x * (tid_y - 1) + tid_x
  blk_size = blockDim().x * blockDim().y
  th_id = blockDim().x * (threadIdx().y - 1) + threadIdx().x
  blk_id = gridDim().x * (blockIdx().y - 1) + blockIdx().x

  arr_type = eltype(glb_dt)
  sh_dt = CUDA.CuDynamicSharedArray(arr_type, blk_size)

  sh_dt[th_id] = glb_dt[i, j]
  sync_threads()

  s = fld(blk_size, 2)
  while s > 0
    if (th_id - 1) < s
      sh_dt[th_id] = min(sh_dt[th_id], sh_dt[th_id + s])
    end
    s = fld(s,2)
    sync_threads()
  end

  if th_id == 1
    glb_min_dt[blk_id] = sh_dt[th_id]
  end

  return 
end

function initialize(mesh, eos)
  ρL, ρR = 1.0, 0.125
  pL, pR = 1.0, 0.1

  M, N = size(mesh.volume)
  ρ0 = zeros(Float32, size(mesh.volume))
  u0 = zeros(Float32, size(mesh.volume))
  v0 = zeros(Float32, size(mesh.volume))
  p0 = zeros(Float32, size(mesh.volume))

  # ρ0 = zeros(M+4, N+4)
  # u0 = zeros(M+4, N+4)
  # v0 = zeros(M+4, N+4)
  # p0 = zeros(M+4, N+4)

  ρ0[begin:(N ÷ 2), :] .= ρL
  ρ0[(N ÷ 2):end, :] .= ρR

  p0[begin:(N ÷ 2), :] .= pL
  p0[(N ÷ 2):end, :] .= pR

  E0 = specific_total_energy.(Ref(eos), ρ0, u0, v0, p0)

  U⃗ = zeros(Float32, 4, size(mesh.volume)...)

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
  dx = 0.00025

  T = Float32
  ######## USe this for muscl.
  # x = collect(T, -0.256 - (2*dx):dx:0.256 + (2*dx))
  # y = collect(T, -0.256 - (2*dx):dx:0.256 + (2*dx))

  ##### USe this for cons2prim, when data needs to be aligned
  x = collect(T, -0.256:dx:0.256)
  y = collect(T, -0.256:dx:0.256)
  # x = collect(T, -0.2 - (2*dx):dx:0.2 + (2*dx))
  # y = collect(T, -0.2 - (2*dx):dx:0.2 + (2*dx))
  @show typeof(x)
  @show typeof(y)
  #@show typeof(ArrayT)
  nhalo = 2
  mesh = CartesianMesh(x, y, nhalo);
  gpumesh = adapt(CuArray{T}, CartesianMesh(x, y, nhalo));

  eos = IdealEOS(1.4)

  M, N = size(mesh.volume)
  @show (M, N)

  blkdim_x = 4
  blkdim_y = 64
  stride = 2
  blk_size = blkdim_x * blkdim_y
  # grddim_x = cld(N-4, blkdim_x)
  # grddim_y = cld(M-4, blkdim_y)
  # grdsize = grddim_x * grddim_y

  # glb_dt = adapt(CuArray{T}, zeros(M-4, N-4));
  # glb_min_dt = adapt(CuArray{T}, zeros(grdsize));

  nh = 2
  
  limits = (nh + 1, M - nh, nh + 1, N - nh)
  U = adapt(CuArray{T}, initialize(mesh, eos));
  W = similar(U); # primitive variables [ρ, u, v, p]

  U_cpu = initialize(mesh, eos)
  W_cpu = similar(U_cpu)
  W_iface_cpu = zeros(2, 4, M, N) # reconstructed primitive variable at the i+1/2 face
  W_jface_cpu = zeros(2, 4, M, N)

  W_stride = copy(W)
  # W1 = W[1, : , :]
  # W2 = W[2, : , :]
  # W3 = W[3, : , :]
  # W4 = W[4, : , :]
  @show size(U)
  @show size(W)
  dUdt = similar(U)
  W_iface = adapt(CuArray{T}, zeros(2, 4, M-4, N-4)); # reconstructed primitive variable at the i+1/2 face
  W_jface = adapt(CuArray{T}, zeros(2, 4, M-4, N-4)); # reconstructed primitive variable at the j+1/2 face
  W_i1_face1 = copy(W_iface[1,1,:,:])
  W_i2_face1 = copy(W_iface[2,1,:,:])
  W_i1_face2 = copy(W_iface[1,2,:,:])
  W_i2_face2 = copy(W_iface[2,2,:,:])
  W_i1_face3 = copy(W_iface[1,3,:,:])
  W_i2_face3 = copy(W_iface[2,3,:,:])
  W_i1_face4 = copy(W_iface[1,4,:,:])
  W_i2_face4 = copy(W_iface[2,4,:,:])

  W_jface1 = copy(W_jface[:,1,:,:])
  W_jface2 = copy(W_jface[:,2,:,:])
  W_jface3 = copy(W_jface[:,3,:,:])
  W_jface4 = copy(W_jface[:,4,:,:])
  W_iface1 = copy(W_iface[:,1,:,:])
  W_iface2 = copy(W_iface[:,2,:,:])
  W_iface3 = copy(W_iface[:,3,:,:])
  W_iface4 = copy(W_iface[:,4,:,:])
  @show typeof(W_i1_face1)
  @show size(W_i1_face1)
  #@show typeof(W_iface2)
  #@show size(W_iface2)
  flux_iface = adapt(ArrayT, zeros(4, M, N)); # flux at the i+1/2 face
  flux_jface = adapt(ArrayT, zeros(4, M, N)); # flux at the j+1/2 face

  # U1 = copy(U[1, 3:M-2, 3:N-2])
  # @show typeof(U1)
  # @show size(U1)
  # U2 = copy(U[2, 3:M-2, 3:N-2])
  # U3 = copy(U[3, 3:M-2, 3:N-2])
  # U4 = copy(U[4, 3:M-2, 3:N-2])

  U1 = copy(U[1, :, :])
  @show typeof(U1)
  @show size(U1)
  U2 = copy(U[2, :, :])
  U3 = copy(U[3, :, :])
  U4 = copy(U[4, :, :])

  W1 = copy(W[1, : , :])
  W2 = copy(W[2, : , :])
  W3 = copy(W[3, : , :])
  W4 = copy(W[4, : , :])

  ##### a single stage consists of the following kernels
  # cons2prim_kernel = cons2prim!(backend)
  # recon_kernel = muscl_gmem!(backend)
  # riemann_kernel = LocalHydroStencil.RiemannSolverType.riemann_solver!(backend)
  # # riemann_kernel_iface = LocalHydroStencil.RiemannSolverType.riemann_solver_iface!(backend)
  # flux_kernel = sum_fluxes(backend)

  # begin
  #   # Conservative to Primitive
  #   cons2prim_kernel(U, W, eos; ndrange=(M, N))
  #   KernelAbstractions.synchronize(backend)

  #   # Reconstruction
  #   recon_kernel(W, W_iface, W_jface, limits; ndrange=(M, N))
  #   KernelAbstractions.synchronize(backend)

  #   # Riemann solver
  #   riemann_kernel(
  #     W, W_iface, W_jface, flux_iface, flux_jface, gpumesh, eos, limits; ndrange=(M, N)
  #   )
  #   KernelAbstractions.synchronize(backend)
  # end

  # @benchmark begin
  #   # Conservative to Primitive
  #   cons2prim_kernel($U, $W, $eos; ndrange=($M, $N))
  #   KernelAbstractions.synchronize($backend)
  # end # -> 6ms

  # @benchmark begin
  #   # Reconstruction
  #   recon_kernel($W, $W_iface, $W_jface, $limits; ndrange=($M, $N))
  #   KernelAbstractions.synchronize($backend)
  # end # -> 47ms

  # @benchmark begin
  #   riemann_kernel(
  #     $W,
  #     $W_iface,
  #     $W_jface,
  #     $flux_iface,
  #     $flux_jface,
  #     $gpumesh,
  #     $eos,
  #     $limits;
  #     ndrange=($M, $N),
  #   )
  #   KernelAbstractions.synchronize($backend)
  # end  # -> 111ms

  # # @benchmark begin
  # #   riemann_kernel_iface($W, $W_iface, $flux_iface, $gpumesh, $eos, $limits; ndrange=($M, $N))
  # #   KernelAbstractions.synchronize($backend)
  # # end # -> 54ms
  println("Running integrate for CUDA") 
  # CUDA.@profile begin
  #   ### Conservative to Primitive
  #   CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N, blkdim_x), cld(M, blkdim_y)) cons2prim!(U, W, eos)
  #   #Array(W)

  #   CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N, (stride*blkdim_x)), cld(M, (stride*blkdim_y))) cons2prim_stride!(U, W_stride, eos)
  #   # #KernelAbstractions.synchronize($backend)
  #   CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N, blkdim_x), cld(M, blkdim_y)) cons2prim_splitUW!(U1, U2, U3, U4, W1, W2, W3, W4, eos)
     CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N, (stride*blkdim_x)), cld(M, (stride*blkdim_y))) cons2prim_stride_splitUW!(U1, U2, U3, U4, W1, W2, W3, W4, eos)

    # ### Reconstruction
    # ### Version 1
    # ### Orignal Reconstruction Kernel.
    # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N, blkdim_x), cld(M, blkdim_y)) muscl_gmem!(W, W_iface, W_jface, limits)
    
    # ### Version 2
    # ### Reconstruction kernel split into I (y-direction) & J(x-direction) faces
    # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) muscl_gmem_i!(W, W_iface, limits)
    # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) muscl_gmem_j!(W, W_jface, limits)
    
    ## Version 3: Global Memory + Non-transpose
    ## Reconstruction kernel split into I (y-direction) & J(x-direction) faces
    ## W is split into individual components corresponding to the physical quantiites.
    ## Launch with blkdim_x = 4, blkdim_y = 64. Threads are column heavy due to col major layout of the data.
    CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) muscl_gmem_i_Wsplit!(W1, W_iface1, limits)
    CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) muscl_gmem_i_Wsplit!(W2, W_iface2, limits)
    CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) muscl_gmem_i_Wsplit!(W3, W_iface3, limits)
    CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) muscl_gmem_i_Wsplit!(W4, W_iface4, limits)

    CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) muscl_gmem_j_Wsplit!(W1, W_jface1, limits)
    CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) muscl_gmem_j_Wsplit!(W2, W_jface2, limits)
    CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) muscl_gmem_j_Wsplit!(W3, W_jface3, limits)
    CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) muscl_gmem_j_Wsplit!(W4, W_jface4, limits)

  #   # ## Shared Memory + Non-transpose versions
  #   # el_type = eltype(W1)
  #   # shmem_size = sizeof(el_type) * (blkdim_y + (2*nh)) * (blkdim_x)
  #   # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) shmem=shmem_size muscl_shmem_i_Wsplit!(W1, W_i1_face1, W_i2_face1, limits)
  #   # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) shmem=shmem_size muscl_shmem_i_Wsplit!(W2, W_i1_face2, W_i2_face2, limits)
  #   # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) shmem=shmem_size muscl_shmem_i_Wsplit!(W3, W_i1_face3, W_i2_face3, limits)
  #   # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) shmem=shmem_size muscl_shmem_i_Wsplit!(W4, W_i1_face4, W_i2_face4, limits)

  #   ## Global Memory + Transpose versions
  #   ## The block dimensions in x & y are flipped. The kernel transposes the data loaded.
  #   #el_type = eltype(W1)
  #   #blkdim_x = 64
  #   #blkdim_y = 4

  #   # # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N, blkdim_x), cld(M, blkdim_y)) cons2prim_splitUW_tp!(U1, U2, U3, U4, W1, W2, W3, W4, eos)

  #   # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) muscl_gmem_i_Wsplit_tp!(W1, W_iface1, limits)
  #   # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) muscl_gmem_i_Wsplit_tp!(W2, W_iface2, limits)
  #   # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) muscl_gmem_i_Wsplit_tp!(W3, W_iface3, limits)
  #   # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) muscl_gmem_i_Wsplit_tp!(W4, W_iface4, limits)

  #   # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) muscl_gmem_j_Wsplit_tp!(W1, W_jface1, limits)
  #   # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) muscl_gmem_j_Wsplit_tp!(W2, W_jface2, limits)
  #   # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) muscl_gmem_j_Wsplit_tp!(W3, W_jface3, limits)
  #   # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) muscl_gmem_j_Wsplit_tp!(W4, W_jface4, limits)

  #   # # # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) muscl_gmem_i_Wsplit_tpshufl!(W1[3:2050], W_i1_face1, W_i2_face1, limits)
  #   # # # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) muscl_gmem_i_Wsplit_tpshufl!(W2[3:2050], W_i1_face2, W_i2_face2, limits)
  #   # # # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) muscl_gmem_i_Wsplit_tpshufl!(W3[3:2050], W_i1_face3, W_i2_face3, limits)
  #   # # # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) muscl_gmem_i_Wsplit_tpshufl!(W4[3:2050], W_i1_face4, W_i2_face4, limits)

  #   # # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) muscl_gmem_i_Wsplit_tpshufl!(W1[3:2050], W_iface1, limits)
  #   # # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) muscl_gmem_i_Wsplit_tpshufl!(W2[3:2050], W_iface2, limits)
  #   # # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) muscl_gmem_i_Wsplit_tpshufl!(W3[3:2050], W_iface3, limits)
  #   # # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) muscl_gmem_i_Wsplit_tpshufl!(W4[3:2050], W_iface4, limits)

  #   # ## Shared Memory + Transpose versions
  #   # ## The block dimensions in x & y are flipped. The kernel transposes the data loaded.
  #   # shmem_size = sizeof(el_type) * (blkdim_x + (2*nh)) * (blkdim_y)
  #   # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) shmem=shmem_size muscl_shmem_i_Wsplit_tp!(W1, W_i1_face1, W_i2_face1, limits)
  #   # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) shmem=shmem_size muscl_shmem_i_Wsplit_tp!(W2, W_i1_face2, W_i2_face2, limits)
  #   # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) shmem=shmem_size muscl_shmem_i_Wsplit_tp!(W3, W_i1_face3, W_i2_face3, limits)
  #   # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) shmem=shmem_size muscl_shmem_i_Wsplit_tp!(W4, W_i1_face4, W_i2_face4, limits)

  #   # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) shmem=shmem_size muscl_shmem_i_Wsplit_tp_var!(W1, W_i1_face1, W_i2_face1, limits)
  #   # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) shmem=shmem_size muscl_shmem_i_Wsplit_tp_var!(W2, W_i1_face2, W_i2_face2, limits)
  #   # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) shmem=shmem_size muscl_shmem_i_Wsplit_tp_var!(W3, W_i1_face3, W_i2_face3, limits)
  #   # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) shmem=shmem_size muscl_shmem_i_Wsplit_tp_var!(W4, W_i1_face4, W_i2_face4, limits)
  
    

  #   # Riemann solver
  #   # @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N, blkdim_x), cld(M, blkdim_y)) LocalHydroStencil.RiemannSolverType.riemann_solver!(
  #   #   W,
  #   #   W_iface,
  #   #   W_jface,
  #   #   flux_iface,
  #   #   flux_jface,
  #   #   gpumesh,
  #   #   eos,
  #   #   limits)

  #   # # Sum fluxes
  #   # @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N, blkdim_x), cld(M, blkdim_y)) sum_fluxes(
  #   #   dUdt,
  #   #   flux_iface,
  #   #   flux_jface,
  #   #   gpumesh.facelen,
  #   #   gpumesh.volume,
  #   #   limits)
  #   #end # -> 180ms

  #   ### Next Time Step
  #   # el_type = eltype(U1)
  #   # shmem_size = sizeof(el_type) * blk_size
  #   # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) shmem=shmem_size _next_Δt(U1, U2, U3, U4, glb_min_dt, gpumesh.facelen, gpumesh.volume, gpumesh.facenorms, eos)
  #   # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) calc_Δt(U1, U2, U3, U4, glb_dt, gpumesh.facelen, gpumesh.volume, gpumesh.facenorms, eos)
  #   # blkdim_x = 16
  #   # blkdim_y = 64
  #   # blk_size = blkdim_x * blkdim_y
  #   # el_type = eltype(glb_min_dt)
  #   # shmem_size = sizeof(el_type) * blk_size
  #   # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) shmem=shmem_size reduce_dt(glb_min_dt, glb_dt)

  # end
  # Array(W)
  # #@show W_stride
  # println("cons2prim strided matches cons2prim non-strided :", Array(W_stride) == Array(W))

  # blkdim_x = 4
  # blkdim_y = 64
  # @benchmark CUDA.@sync @cuda threads=($blkdim_x, $blkdim_y) blocks=(cld($N, $blkdim_x), cld($M, $blkdim_y)) cons2prim!($U, $W, $eos)
  #   #KernelAbstractions.synchronize($backend)
  # @benchmark CUDA.@sync @cuda threads=($blkdim_x, $blkdim_y) blocks=(cld($N, $blkdim_x), cld($M, $blkdim_y)) cons2prim_splitUW!($U1, $U2, $U3, $U4, $W1, $W2, $W3, $W4, $eos)
  # blkdim_x = 64
  # blkdim_y = 4
  # @benchmark CUDA.@sync @cuda threads=($blkdim_x, $blkdim_y) blocks=(cld($N, $blkdim_x), cld($M, $blkdim_y)) cons2prim_splitUW_tp!($U1, $U2, $U3, $U4, $W1, $W2, $W3, $W4, $eos)

  ## CPU based muscl & cons2prim
  cons2prim_cpu!(W_cpu, U_cpu, eos, limits)
  muscl_gmem_cpu!(W_cpu, W_iface_cpu, W_jface_cpu, limits)
  @benchmark muscl_gmem_cpu!($W_cpu, $W_iface_cpu, $W_jface_cpu, $limits)
  @benchmark cons2prim_cpu!($W_cpu, $U_cpu, $eos, $limits)

end

isinteractive() || main()