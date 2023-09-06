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

@kernel function cons2prim!(U, W, EOS)
  i, j = @index(Global, NTuple)
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

@kernel function muscl_gmem!(W::AbstractArray{T,N}, i_face, j_face, limits) where {T,N}
  i, j = @index(Global, NTuple)
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

# function main()
eos = IdealEOS(1.4)
dx = 1e-4
x = collect(-0.2:dx:0.2)
y = collect(-0.2:dx:0.2)

nhalo = 2
mesh = CartesianMesh(x, y, nhalo);
gpumesh = adapt(ArrayT, CartesianMesh(x, y, nhalo));

eos = IdealEOS(1.4)

M, N = size(mesh.volume)
@show (M, N)
nh = 2
T = Float64
limits = (nh + 1, M - nh, nh + 1, N - nh)
U = adapt(ArrayT, initialize(mesh, eos));
W = similar(U); # primitive variables [ρ, u, v, p]
W_iface = adapt(ArrayT, zeros(2, 4, M, N)); # reconstructed primitive variable at the i+1/2 face
W_jface = adapt(ArrayT, zeros(2, 4, M, N)); # reconstructed primitive variable at the j+1/2 face

flux_iface = adapt(ArrayT, zeros(4, M, N)); # flux at the i+1/2 face
flux_jface = adapt(ArrayT, zeros(4, M, N)); # flux at the j+1/2 face

cons2prim_kernel = cons2prim!(backend)
recon_kernel = muscl_gmem!(backend)
riemann_kernel = LocalHydroStencil.RiemannSolverType.riemann_solver!(backend)

begin
  # Conservative to Primitive
  cons2prim_kernel(U, W, eos; ndrange=(M, N))
  KernelAbstractions.synchronize(backend)

  # Reconstruction
  recon_kernel(W, W_iface, W_jface, limits; ndrange=(M, N))
  KernelAbstractions.synchronize(backend)

  # Riemann solver
  riemann_kernel(
    W, W_iface, W_jface, flux_iface, flux_jface, gpumesh, eos, limits; ndrange=(M, N)
  )
  KernelAbstractions.synchronize(backend)
end

@benchmark begin
  # Conservative to Primitive
  cons2prim_kernel($U, $W, $eos; ndrange=($M, $N))
  KernelAbstractions.synchronize($backend)

  # Reconstruction
  recon_kernel($W, $W_iface, $W_jface, $limits; ndrange=($M, $N))
  KernelAbstractions.synchronize($backend)

  # Riemann solver
  riemann_kernel(
    $W,
    $W_iface,
    $W_jface,
    $flux_iface,
    $flux_jface,
    $gpumesh,
    $eos,
    $limits;
    ndrange=($M, $N),
  )
  KernelAbstractions.synchronize($backend)
end