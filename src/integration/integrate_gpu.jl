
function integrate!(
  SS::SSPRK3,
  U⃗n::AbstractArray{T,N},
  mesh,
  EOS,
  dt::Number,
  # BCs,
  riemann_solver,
  recon::F2,
  limiter::F3,
  skip_uniform=true,
) where {T,N,F2,F3}

  # call the kernel here
  println("calling the KA version!")

end

@kernel function _integrate_ka!(
  @Const(U⃗n::AbstractArray{T,N}),
  U⃗1::AbstractArray{T,N},
  U⃗2::AbstractArray{T,N},
  U⃗3::AbstractArray{T,N},
  mesh,
  EOS,
  dt::Number,
  # BCs,
  riemann_solver,
  recon::F2,
  limiter::F3,
  skip_uniform=true,
) where {T,N,F2,F3}
  i, j = @index(Global, NTuple)

  nh = mesh.nhalo
  ilohi = axes(U⃗n, 2)
  jlohi = axes(U⃗n, 3)

  ilo = first(ilohi) + nh
  jlo = first(jlohi) + nh
  ihi = last(ilohi) - nh
  jhi = last(jlohi) - nh

  # stage 1
  if (ilo <= i <= ihi) && (jlo <= j <= jhi)
    U⁽ⁿ⁾ = SVector{4,T}(view(U⃗n, :, i, j))
    U_local = get_block(U⃗n, i, j)
    S⃗ = @SVector zeros(4)
    n̂ = SMatrix{2,4}(view(norms, :, :, i, j))
    ΔS = SVector{4,T}(view(ΔS_face, :, i, j))
    Ω = vol[i, j]
    x⃗_c = SVector{2,T}(view(centroid_pos, :, i, j))
    stencil = Stencil9Point(U_local, S⃗, n̂, ΔS, Ω, EOS, x⃗_c)
    ∂U⁽ⁿ⁾∂t = riemann_solver.∂U∂t(stencil, recon, limiter, skip_uniform)
    U⃗1[:, i, j] = U⁽ⁿ⁾ + ∂U⁽ⁿ⁾∂t * dt
  end

  @synchronize()

  # @inbounds begin
  #   if (ilo <= i <= ihi) && (jlo <= j <= jhi)
  #     U⁽¹⁾ = getvec(U⃗1, i, j)
  #     U⁽ⁿ⁾ = getvec(U⃗n, i, j)
  #     stencil = _2halo2dstencil(U⃗1, (i, j), mesh, EOS, nh)
  #     ∂U⁽¹⁾∂t = ∂U∂t(riemann_solver, stencil, muscl, minmod)
  #     U⁽²⁾ = 0.75U⁽ⁿ⁾ + 0.25U⁽¹⁾ + ∂U⁽¹⁾∂t * 0.25dt
  #     for q in 1:Nq
  #       U⃗2[q, i, j] = U⁽²⁾[q]
  #     end
  #   end
  # end

  # @synchronize()

  # @inbounds begin
  #   if (ilo <= i <= ihi) && (jlo <= j <= jhi)
  #     U⁽²⁾ = getvec(U⃗2, i, j)
  #     U⁽ⁿ⁾ = getvec(U⃗n, i, j)
  #     stencil = _2halo2dstencil(U⃗2, (i, j), mesh, EOS, nh)
  #     ∂U⁽²⁾∂t = ∂U∂t(riemann_solver, stencil, muscl, minmod)
  #     U⁽ⁿ⁺¹⁾ = (1 / 3) * U⁽ⁿ⁾ + (2 / 3) * U⁽²⁾ + ∂U⁽²⁾∂t * (2 / 3) * dt
  #     for q in 1:Nq
  #       U⃗3[q, i, j] = U⁽ⁿ⁺¹⁾[q]
  #     end
  #   end
  # end
end

# @kernel function SSPRK3_gpu_lmem!(
#   SS::SSPRK3Integrator,
#   @Const(U⃗n::AbstractArray{T,N}),
#   @Const(riemann_solver),
#   @Const(mesh),
#   @Const(EOS::E),
#   @Const(dt),
#   ::Val{2},
# ) where {T,N,E}
#   nh = mesh.nhalo
#   Nq = size(U⃗n, 1)
#   ilohi = axes(U⃗n, 2)
#   jlohi = axes(U⃗n, 3)

#   ilo = first(ilohi) + nh
#   jlo = first(jlohi) + nh
#   ihi = last(ilohi) - nh
#   jhi = last(jlohi) - nh

#   i, j = @index(Global, NTuple)
#   li, lj = @index(Local, NTuple)

#   # These are hardcoded to 4 components of U, and a halo region of 2 cells... this needs to be more flexible!
#   lmem = @localmem eltype(U⃗n) (4, @groupsize()[1] + 4, @groupsize()[2] + 4)
#   @uniform ldata = OffsetArray(lmem, 1:4, 0:(@groupsize()[1] + 3), 0:(@groupsize()[2] + 3))

#   BT = SArray{Tuple{4,5,5},T,3,100}

#   # Load U⃗ into GPU local memory
#   @inbounds begin
#     if (ilo <= i <= ihi) && (jlo <= j <= jhi)
#       for q in 1:Nq
#         ldata[q, li, lj] = U⃗n[q, i, j]
#       end

#       if i == ilo
#         for offset in 1:2
#           for q in 1:Nq
#             ldata[q, li - offset, lj] = U⃗n[q, i - offset, j]
#           end
#         end
#       end

#       if i == @groupsize()[1]
#         for offset in 1:2
#           for q in 1:Nq
#             ldata[q, li + offset, lj] = U⃗n[q, i + offset, j]
#           end
#         end
#       end

#       if j == jlo
#         for offset in 1:2
#           for q in 1:Nq
#             ldata[q, li, lj - offset] = U⃗n[q, i, j - offset]
#           end
#         end
#       end

#       if j == @groupsize()[2]
#         for offset in 1:2
#           for q in 1:Nq
#             ldata[q, li, lj + offset] = U⃗n[q, i, j + offset]
#           end
#         end
#       end
#     end
#   end
#   @synchronize()

#   @inbounds begin
#     if (ilo <= i <= ihi) && (jlo <= j <= jhi)
#       U⁽ⁿ⁾ = getvec(ldata, li, lj)
#       n̂ = get_nhat(mesh, i, j)
#       ΔS = get_ΔS(mesh, i, j)
#       Ω = mesh.volume[i, j]
#       S⃗ = @SVector zeros(4)
#       U_local = SArray{Tuple{4,5,5},T,3,100}(
#         view(ldata, :, (li - 2):(li + 2), (lj - 2):(lj + 2))
#       )

#       stencil = Stencil9Point{T,BT,E}(U_local, S⃗, n̂, ΔS, Ω, EOS)

#       ∂U⁽ⁿ⁾∂t = ∂U∂t(riemann_solver, stencil, muscl, minmod)
#       # U⁽¹⁾ = U⁽ⁿ⁾ + ∂U⁽ⁿ⁾∂t * dt
#       # for q in 1:Nq
#       #     U⃗1[q,i,j] = U⁽¹⁾[q]
#       # end
#     end
#   end

#   # @synchronize()

#   # @inbounds begin
#   #     if (ilo <= i <= ihi) && (jlo <= j <= jhi)
#   #         U⁽¹⁾ = getvec(U⃗1,i,j)
#   #         U⁽ⁿ⁾ = getvec(U⃗n,i,j)
#   #         stencil = _2halo2dstencil(U⃗1, (i,j), mesh, EOS, nh)
#   #         ∂U⁽¹⁾∂t = ∂U∂t(riemann_solver, stencil, muscl, minmod)
#   #         U⁽²⁾ = .75U⁽ⁿ⁾ + .25U⁽¹⁾ + ∂U⁽¹⁾∂t * .25dt
#   #         for q in 1:Nq
#   #             U⃗2[q,i,j] = U⁽²⁾[q]
#   #         end
#   #     end
#   # end

#   # @synchronize()

#   # @inbounds begin
#   #     if (ilo <= i <= ihi) && (jlo <= j <= jhi)
#   #         U⁽²⁾ = getvec(U⃗2, i, j)
#   #         U⁽ⁿ⁾ = getvec(U⃗n, i, j)
#   #         stencil = _2halo2dstencil(U⃗2, (i,j), mesh, EOS, nh)
#   #         ∂U⁽²⁾∂t = ∂U∂t(riemann_solver, stencil, muscl, minmod)
#   #         U⁽ⁿ⁺¹⁾ = (1/3) * U⁽ⁿ⁾ + (2/3) * U⁽²⁾+ ∂U⁽²⁾∂t * (2/3) * dt
#   #         for q in 1:Nq
#   #             U⃗3[q,i,j] = U⁽ⁿ⁺¹⁾[q]
#   #         end
#   #     end
#   # end

# end
