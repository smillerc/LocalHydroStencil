function sync_halo!(U, nhalo, i, j)
  ilohi = axes(U, 2)
  jlohi = axes(U, 3)
  ilo = first(ilohi) + nhalo
  jlo = first(jlohi) + nhalo
  ihi = last(ilohi) - nhalo
  jhi = last(jlohi) - nhalo

  if (j in 1:(jlo - 1)) && (i in ilo:ihi)
    for q in axes(U, 1)
      U[q, i, j] = U[q, i, jlo]
    end
  end

  if (j in (jhi - nhalo):last(jlohi)) && (i in ilo:ihi)
    for q in axes(U, 1)
      U[q, i, j] = U[q, i, jhi]
    end
  end

  if (j in jlo:jhi) && (i in first(ilohi):(ilo - 1))
    for q in axes(U, 1)
      U[q, i, j] = U[q, ilo, j]
    end
  end

  if (j in jlo:jhi) && (i in (ihi - nhalo):last(ilohi))
    for q in axes(U, 1)
      U[q, i, j] = U[q, ihi, j]
    end
  end

  return nothing
end

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
  kernel,
) where {T,N,F2,F3}
  skip_uniform = true
  # println("calling the KA version on ", backend)
  rows = size(U⃗n, 2)
  cols = size(U⃗n, 3)
  U⃗1 = SS.U⃗1
  U⃗2 = SS.U⃗2
  U⃗3 = SS.U⃗3
  blkdim_x = 16
  blkdim_y = 16

  nh = mesh.nhalo
  ilohi = axes(U⃗n, 2)
  jlohi = axes(U⃗n, 3)

  ilo = first(ilohi) + nh
  jlo = first(jlohi) + nh
  ihi = last(ilohi) - nh
  jhi = last(jlohi) - nh
  lims = (ilo, ihi, jlo, jhi)

  kernel(
    U⃗n,
    U⃗1,
    U⃗2,
    U⃗3,
    mesh,
    EOS,
    dt,
    riemann_solver,
    recon,
    limiter,
    lims,
    skip_uniform;
    ndrange=(rows, cols),
    workgroupsize=(blkdim_y, blkdim_x),
  )
  return nothing
  # return synchronize(backend)
end

function getvec(A::AbstractArray{T,3}, i, j) where {T}
  return SVector{4,T}(view(A, :, i, j))
end

@kernel function _integrate_ka!(
  @Const(U⃗n::AbstractArray{T,N}),
  U⃗1::AbstractArray{T,N},
  U⃗2::AbstractArray{T,N},
  U⃗3::AbstractArray{T,N},
  @Const(mesh),
  @Const(EOS),
  @Const(dt::Number),
  # BCs),
  @Const(riemann_solver),
  @Const(recon::F2),
  @Const(limiter::F3),
  @Const(lims),
  @Const(skip_uniform),
) where {T,N,F2,F3}
  i, j = @index(Global, NTuple)

  ilo, ihi, jlo, jhi = lims
  S⃗ = @SVector zeros(4)
  n̂ = SMatrix{2,4}(view(mesh.facenorms, 1:2, 1:4, i, j))
  ΔS = SVector{4,T}(view(mesh.facelen, 1:4, i, j))
  Ω = mesh.volume[i, j]
  x⃗_c = SVector{2,T}(view(mesh.centroid, 1:2, i, j))
  U⁽ⁿ⁾ = SVector{4,T}(view(U⃗n, 1:4, i, j))

  # stage 1
  @inbounds if (ilo <= i <= ihi) && (jlo <= j <= jhi)
    U_local = get_block(U⃗n, i, j)
    stencil = Stencil9Point(U_local, S⃗, n̂, ΔS, Ω, EOS, x⃗_c)
    ∂U⁽ⁿ⁾∂t = riemann_solver.∂U∂t(stencil, recon, limiter, skip_uniform)

    @inbounds for ind in 1:4
      U⃗1[ind, i, j] = U⁽ⁿ⁾[ind] + (∂U⁽ⁿ⁾∂t[ind] * dt)
    end
  end

  @synchronize()
  sync_halo!(U⃗1, mesh.nhalo, i, j)
  @synchronize()

  # Stage 2
  if (ilo <= i <= ihi) && (jlo <= j <= jhi)
    U⁽¹⁾ = SVector{4,T}(view(U⃗1, 1:4, i, j))
    U_local = get_block(U⃗1, i, j)
    stencil = Stencil9Point(U_local, S⃗, n̂, ΔS, Ω, EOS, x⃗_c)
    ∂U⁽¹⁾∂t = riemann_solver.∂U∂t(stencil, recon, limiter, skip_uniform)
    U⁽²⁾ = 0.75U⁽ⁿ⁾ + 0.25U⁽¹⁾ + ∂U⁽¹⁾∂t * 0.25dt

    @inbounds for ind in 1:4
      U⃗2[ind, i, j] = U⁽²⁾[ind]
    end
  end

  @synchronize()
  sync_halo!(U⃗2, mesh.nhalo, i, j)
  @synchronize()

  # Stage 3
  if (ilo <= i <= ihi) && (jlo <= j <= jhi)
    U⁽²⁾ = SVector{4,T}(view(U⃗2, 1:4, i, j))
    U_local = get_block(U⃗2, i, j)
    stencil = Stencil9Point(U_local, S⃗, n̂, ΔS, Ω, EOS, x⃗_c)
    ∂U⁽²⁾∂t = riemann_solver.∂U∂t(stencil, recon, limiter, skip_uniform)
    U⁽ⁿ⁺¹⁾ = (1 / 3) * U⁽ⁿ⁾ + (2 / 3) * U⁽²⁾ + ∂U⁽²⁾∂t * (2 / 3) * dt

    @inbounds for ind in 1:4
      U⃗3[ind, i, j] = U⁽ⁿ⁺¹⁾[ind]
    end
  end
end

@kernel function SSPRK3_gpu_lmem!(
  SS::SSPRK3Integrator,
  @Const(U⃗n::AbstractArray{T,N}),
  @Const(riemann_solver),
  @Const(mesh),
  @Const(EOS::E),
  @Const(dt),
  ::Val{2},
) where {T,N,E}
  nh = mesh.nhalo
  Nq = size(U⃗n, 1)
  ilohi = axes(U⃗n, 2)
  jlohi = axes(U⃗n, 3)

  ilo = first(ilohi) + nh
  jlo = first(jlohi) + nh
  ihi = last(ilohi) - nh
  jhi = last(jlohi) - nh

  i, j = @index(Global, NTuple)
  li, lj = @index(Local, NTuple)

  # These are hardcoded to 4 components of U, and a halo region of 2 cells... this needs to be more flexible!
  lmem = @localmem eltype(U⃗n) (4, @groupsize()[1] + 4, @groupsize()[2] + 4)
  @uniform ldata = OffsetArray(lmem, 1:4, 0:(@groupsize()[1] + 3), 0:(@groupsize()[2] + 3))

  BT = SArray{Tuple{4,5,5},T,3,100}

  # Load U⃗ into GPU local memory
  @inbounds begin
    if (ilo <= i <= ihi) && (jlo <= j <= jhi)
      for q in 1:Nq
        ldata[q, li, lj] = U⃗n[q, i, j]
      end

      if i == ilo
        for offset in 1:2
          for q in 1:Nq
            ldata[q, li - offset, lj] = U⃗n[q, i - offset, j]
          end
        end
      end

      if i == @groupsize()[1]
        for offset in 1:2
          for q in 1:Nq
            ldata[q, li + offset, lj] = U⃗n[q, i + offset, j]
          end
        end
      end

      if j == jlo
        for offset in 1:2
          for q in 1:Nq
            ldata[q, li, lj - offset] = U⃗n[q, i, j - offset]
          end
        end
      end

      if j == @groupsize()[2]
        for offset in 1:2
          for q in 1:Nq
            ldata[q, li, lj + offset] = U⃗n[q, i, j + offset]
          end
        end
      end
    end
  end
  @synchronize()

  @inbounds begin
    if (ilo <= i <= ihi) && (jlo <= j <= jhi)
      U⁽ⁿ⁾ = getvec(ldata, li, lj)
      n̂ = get_nhat(mesh, i, j)
      ΔS = get_ΔS(mesh, i, j)
      Ω = mesh.volume[i, j]
      S⃗ = @SVector zeros(4)
      U_local = SArray{Tuple{4,5,5},T,3,100}(
        view(ldata, :, (li - 2):(li + 2), (lj - 2):(lj + 2))
      )

      stencil = Stencil9Point{T,BT,E}(U_local, S⃗, n̂, ΔS, Ω, EOS)

      ∂U⁽ⁿ⁾∂t = ∂U∂t(riemann_solver, stencil, muscl, minmod)
      # U⁽¹⁾ = U⁽ⁿ⁾ + ∂U⁽ⁿ⁾∂t * dt
      # for q in 1:Nq
      #     U⃗1[q,i,j] = U⁽¹⁾[q]
      # end
    end
  end

  # @synchronize()

  # @inbounds begin
  #     if (ilo <= i <= ihi) && (jlo <= j <= jhi)
  #         U⁽¹⁾ = getvec(U⃗1,i,j)
  #         U⁽ⁿ⁾ = getvec(U⃗n,i,j)
  #         stencil = _2halo2dstencil(U⃗1, (i,j), mesh, EOS, nh)
  #         ∂U⁽¹⁾∂t = ∂U∂t(riemann_solver, stencil, muscl, minmod)
  #         U⁽²⁾ = .75U⁽ⁿ⁾ + .25U⁽¹⁾ + ∂U⁽¹⁾∂t * .25dt
  #         for q in 1:Nq
  #             U⃗2[q,i,j] = U⁽²⁾[q]
  #         end
  #     end
  # end

  # @synchronize()

  # @inbounds begin
  #     if (ilo <= i <= ihi) && (jlo <= j <= jhi)
  #         U⁽²⁾ = getvec(U⃗2, i, j)
  #         U⁽ⁿ⁾ = getvec(U⃗n, i, j)
  #         stencil = _2halo2dstencil(U⃗2, (i,j), mesh, EOS, nh)
  #         ∂U⁽²⁾∂t = ∂U∂t(riemann_solver, stencil, muscl, minmod)
  #         U⁽ⁿ⁺¹⁾ = (1/3) * U⁽ⁿ⁾ + (2/3) * U⁽²⁾+ ∂U⁽²⁾∂t * (2/3) * dt
  #         for q in 1:Nq
  #             U⃗3[q,i,j] = U⁽ⁿ⁺¹⁾[q]
  #         end
  #     end
  # end

end

@kernel function diffusion_lmem!(out, @Const(data), a, dt, dx, dy)
  i, j = @index(Global, NTuple)
  li, lj = @index(Local, NTuple)
  lmem = @localmem eltype(data) (@groupsize()[1] + 2, @groupsize()[2] + 2)
  @uniform ldata = OffsetArray(lmem, 0:(@groupsize()[1] + 1), 0:(@groupsize()[2] + 1))

  # Load data from global to local buffer
  @inbounds begin
    ldata[li, lj] = data[i, j]
    if i == 1
      ldata[li - 1, lj] = data[i - 1, j]
    end
    if i == @groupsize()[1]
      ldata[li + 1, lj] = data[i + 1, j]
    end
    if j == 1
      ldata[li, lj - 1] = data[i, j - 1]
    end
    if j == @groupsize()[2]
      ldata[li, lj + 1] = data[i, j + 1]
    end
  end
  @synchronize()

  @inbounds begin
    dij = ldata[li, lj]
    dim1j = ldata[li - 1, lj]
    dijm1 = ldata[li, lj - 1]
    dip1j = ldata[li + 1, lj]
    dijp1 = ldata[li, lj + 1]

    dij += a * dt * ((dim1j - 2 * dij + dip1j) / dx^2 + (dijm1 - 2 * dij + dijp1) / dy^2)

    out[i, j] = dij
  end
end
