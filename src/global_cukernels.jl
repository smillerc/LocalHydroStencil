using Revise
using KernelAbstractions
using BenchmarkTools
using LocalHydroStencil
using StaticArrays
using Adapt
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

function muscl_shmem_i_Wsplit!(W::AbstractArray{T,N}, i1_face, i2_face, limits) where {T,N}
  #i, j = @index(Global, NTuple)
  row = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  col = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  ilo, ihi, jlo, jhi = limits
  ker_width = 4

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
    w_sh[tid_x, tid_y + ker_width] = W[row, col]
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
  ilo, ihi, jlo, jhi = limits

  SMALL = T(1e-30)
  minmod(r) = max(0, min(r, 1))

  if (jlo - 1 <= j <= jhi) && (ilo <= i <= ihi)
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
  end
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

function initialize(mesh, eos)
  ρL, ρR = 1.0, 0.125
  pL, pR = 1.0, 0.1

  M, N = size(mesh.volume)
  ρ0 = zeros(size(mesh.volume))
  u0 = zeros(size(mesh.volume))
  v0 = zeros(size(mesh.volume))
  p0 = zeros(size(mesh.volume))

  # ρ0 = zeros(M+4, N+4)
  # u0 = zeros(M+4, N+4)
  # v0 = zeros(M+4, N+4)
  # p0 = zeros(M+4, N+4)

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
  dx = 0.00025

  T = Float32
  x = collect(T, -0.2 - (2*dx):dx:0.2 + (2*dx))
  y = collect(T, -0.2 - (2*dx):dx:0.2 + (2*dx))
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

  nh = 2
  
  limits = (nh + 1, M - nh, nh + 1, N - nh)
  U = adapt(CuArray{T}, initialize(mesh, eos));
  W = similar(U); # primitive variables [ρ, u, v, p]
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
  @show typeof(W_i1_face1)
  @show size(W_i1_face1)
  #@show typeof(W_iface2)
  #@show size(W_iface2)
  flux_iface = adapt(ArrayT, zeros(4, M, N)); # flux at the i+1/2 face
  flux_jface = adapt(ArrayT, zeros(4, M, N)); # flux at the j+1/2 face

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
  CUDA.@profile begin
    ### Conservative to Primitive
    @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N, blkdim_x), cld(M, blkdim_y)) cons2prim!(U, W, eos)
    #KernelAbstractions.synchronize($backend)
    W1 = copy(W[1, : , :])
    W2 = copy(W[2, : , :])
    W3 = copy(W[3, : , :])
    W4 = copy(W[4, : , :])

    ### Reconstruction
    ### Version 1
    ### Orignal Reconstruction Kernel.
    CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N, blkdim_x), cld(M, blkdim_y)) muscl_gmem!(W, W_iface, W_jface, limits)
    
    ### Version 2
    ### Reconstruction kernel split into I (y-direction) & J(x-direction) faces
    CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N, blkdim_x), cld(M, blkdim_y)) muscl_gmem_i!(W, W_iface, limits)
    CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N, blkdim_x), cld(M, blkdim_y)) muscl_gmem_j!(W, W_jface, limits)
    
    ### Version 3
    ### W is split into individual components.
    ### I face being further Split. Shared memory used.
    ### J face is not split. Global memory used.
    el_type = eltype(W1)
    shmem_size = sizeof(el_type) * (blkdim_y + (2*nh)) * (blkdim_x)
    CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) shmem=shmem_size muscl_shmem_i_Wsplit!(W1, W_i1_face1, W_i2_face1, limits)
    CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) shmem=shmem_size muscl_shmem_i_Wsplit!(W2, W_i1_face2, W_i2_face2, limits)
    CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) shmem=shmem_size muscl_shmem_i_Wsplit!(W3, W_i1_face3, W_i2_face3, limits)
    CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) shmem=shmem_size muscl_shmem_i_Wsplit!(W4, W_i1_face4, W_i2_face4, limits)

    CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) muscl_gmem_j_Wsplit!(W1, W_jface1, limits)
    CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) muscl_gmem_j_Wsplit!(W2, W_jface2, limits)
    CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) muscl_gmem_j_Wsplit!(W3, W_jface3, limits)
    CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N-4, blkdim_x), cld(M-4, blkdim_y)) muscl_gmem_j_Wsplit!(W4, W_jface4, limits)

    # Riemann solver
    # @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N, blkdim_x), cld(M, blkdim_y)) LocalHydroStencil.RiemannSolverType.riemann_solver!(
    #   W,
    #   W_iface,
    #   W_jface,
    #   flux_iface,
    #   flux_jface,
    #   gpumesh,
    #   eos,
    #   limits)

    # # Sum fluxes
    # @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(N, blkdim_x), cld(M, blkdim_y)) sum_fluxes(
    #   dUdt,
    #   flux_iface,
    #   flux_jface,
    #   gpumesh.facelen,
    #   gpumesh.volume,
    #   limits)
    #end # -> 180ms
  end
end

isinteractive() || main()