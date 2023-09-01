function sync_halo!(U, nhalo)
  ilohi = axes(U, 2)
  jlohi = axes(U, 3)
  ilo = first(ilohi) + nhalo
  jlo = first(jlohi) + nhalo
  ihi = last(ilohi) - nhalo
  jhi = last(jlohi) - nhalo

  for j in 1:(jlo - 1)
    for i in ilo:ihi
      for q in axes(U, 1)
        U[q, i, j] = U[q, i, jlo]
      end
    end
  end

  for j in (jhi - nhalo):last(jlohi)
    for i in ilo:ihi
      for q in axes(U, 1)
        U[q, i, j] = U[q, i, jhi]
      end
    end
  end

  for j in jlo:jhi
    for i in first(ilohi):(ilo - 1)
      for q in axes(U, 1)
        U[q, i, j] = U[q, ilo, j]
      end
    end
  end

  for j in jlo:jhi
    for i in (ihi - nhalo):last(ilohi)
      for q in axes(U, 1)
        U[q, i, j] = U[q, ihi, j]
      end
    end
  end

  return nothing
end

#@inbounds function integrate_cpu!(
@inbounds function integrate!(
  SS::SSPRK3,
  U⃗n::Array{T,N},
  mesh,
  EOS,
  dt::Number,
  # BCs,
  riemann_solver,
  recon::F2,
  limiter::F3,
  skip_uniform=true,
) where {T,N,F2,F3}
  nhalo = mesh.nhalo
  ilohi = axes(U⃗n, 2)
  jlohi = axes(U⃗n, 3)
  ilo = first(ilohi) + nhalo
  jlo = first(jlohi) + nhalo
  ihi = last(ilohi) - nhalo
  jhi = last(jlohi) - nhalo

  looplimits = (ilo, ihi, jlo, jhi)
  U⃗1 = SS.U⃗1
  U⃗2 = SS.U⃗2
  U⃗3 = SS.U⃗3

  ΔS_face = mesh.facelen
  vol = mesh.volume
  norms = mesh.facenorms
  centroid_pos = mesh.centroid

  # sync_halo!(Un, nhalo)

  # Stage 1
  @batch per = core for j in jlo:jhi
    for i in ilo:ihi
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
  end

  sync_halo!(U⃗1, nhalo)

  # Stage 2
  @batch per = core for j in jlo:jhi
    for i in ilo:ihi
      U⁽¹⁾ = SVector{4,T}(view(U⃗1, :, i, j))
      U⁽ⁿ⁾ = SVector{4,T}(view(U⃗n, :, i, j))
      U_local = get_block(U⃗1, i, j)
      S⃗ = @SVector zeros(4)
      n̂ = SMatrix{2,4}(view(norms, :, :, i, j))
      ΔS = SVector{4,T}(view(ΔS_face, :, i, j))
      Ω = vol[i, j]
      x⃗_c = SVector{2,T}(view(centroid_pos, :, i, j))
      stencil = Stencil9Point(U_local, S⃗, n̂, ΔS, Ω, EOS, x⃗_c)
      ∂U⁽¹⁾∂t = riemann_solver.∂U∂t(stencil, recon, limiter, skip_uniform)
      U⁽²⁾ = 0.75U⁽ⁿ⁾ + 0.25U⁽¹⁾ + ∂U⁽¹⁾∂t * 0.25dt
      U⃗2[:, i, j] = U⁽²⁾
    end
  end

  sync_halo!(U⃗2, nhalo)

  # Stage 3
  @batch per = core for j in jlo:jhi
    for i in ilo:ihi
      U⁽²⁾ = SVector{4,T}(view(U⃗2, :, i, j))
      U⁽ⁿ⁾ = SVector{4,T}(view(U⃗n, :, i, j))
      U_local = get_block(U⃗2, i, j)
      S⃗ = @SVector zeros(4)
      n̂ = SMatrix{2,4}(view(norms, :, :, i, j))
      ΔS = SVector{4,T}(view(ΔS_face, :, i, j))
      Ω = vol[i, j]
      x⃗_c = SVector{2,T}(view(centroid_pos, :, i, j))
      stencil = Stencil9Point(U_local, S⃗, n̂, ΔS, Ω, EOS, x⃗_c)
      ∂U⁽²⁾∂t = riemann_solver.∂U∂t(stencil, recon, limiter, skip_uniform)
      U⁽ⁿ⁺¹⁾ = (1 / 3) * U⁽ⁿ⁾ + (2 / 3) * U⁽²⁾ + ∂U⁽²⁾∂t * (2 / 3) * dt
      U⃗3[:, i, j] = U⁽ⁿ⁺¹⁾
    end
  end

  # sync_halo!(U⃗3, nhalo)
  # resids = check_residuals(U⃗2,U⃗1,looplimits)
  #resids = @SVector zeros(4)
  #success = true
  #return success, resids
  return nothing
end

function check_residuals(U1, Un, looplimits)
  # TODO: make this work for diff sizes

  ilo, ihi, jlo, jhi = looplimits
  ϕ1_denoms = @MVector zeros(4)
  resids = @MVector zeros(4)
  numerators = @MVector zeros(4)
  fill!(resids, -Inf)

  @batch for j in jlo:jhi
    for i in ilo:ihi
      for q in eachindex(ϕ1_denoms)
        ϕ1_denoms[q] += U1[q, i, j]^2
      end
    end
  end
  ϕ1_denoms = sqrt.(ϕ1_denoms)

  # if isinf.(ϕ1_denoms) || iszero(ϕ1_denoms)
  #     resids = -Inf
  # else

  @batch for j in jlo:jhi
    for i in ilo:ihi
      for q in eachindex(ϕ1_denoms)
        numerators[q] += (Un[q, i, j] - U1[q, i, j])^2
      end
    end
  end

  resids = sqrt.(numerators) ./ ϕ1_denoms
  # end

  return SVector(resids)
end
