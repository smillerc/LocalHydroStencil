
"""
    getflux(F_l, ds_l, F_r, ds_r, ϵ_abs, ϵ_rel)

Compute the flux across a cell based on the left/right cell edges (for quad cells). An attempt to
impose error constraints is based on relative and absolute values.
"""
@inline function getflux(F_l, ds_l, F_r, ds_r, ϵ_abs, ϵ_rel)
  l = F_l * ds_l
  r = F_r * ds_r
  flux = r - l
  max_val = max(abs(F_l), abs(F_r))

  return flux * (abs(flux) >= ϵ_abs) * (abs(F_l - F_r) > ϵ_rel * max_val)
end

@kernel function _calc_dUdt(
  dUdt::AbstractArray{T,N}, F, G, mesh, ϵ_abs=eps(Float64); ϵ_rel=1e-6
) where {T,N}
  i, j = @index(Global, NTuple)
  ilo, ihi, jlo, jhi = mesh.limits

  @inbounds if (jlo <= j <= jhi) && (ilo <= i <= ihi)
    ds1, ds2, ds3, ds4 = facearea(mesh, i, j)
    # xy = centroid(mesh, i, j)

    F_i_minus_half, F_i_plus_half = @views F[1:4, (i - 1):i, j]
    F_j_minus_half, F_j_plus_half = @views G[1:4, i, (j - 1):j]

    iflux = getflux(F_i_minus_half, ds4, F_i_plus_half, ds2, ϵ_abs, ϵ_rel)
    jflux = getflux(F_j_minus_half, ds1, F_j_plus_half, ds3, ϵ_abs, ϵ_rel)
    flux = -(iflux + jflux)
    flux = flux * (abs(flux) >= 1e-14)

    dUdt[i, j] = flux / mesh.volume[i, j]
  end

  return nothing
end

@kernel function _riemann_gmem!(W, i_face, j_face, limits) where {T,N}
  i, j = @index(Global, NTuple)
  ilo, ihi, jlo, jhi = limits

  # Do the i face
  if (jlo <= j <= jhi) && (ilo - 1 <= i <= ihi)
    _, n2, _, _ = facenorm(mesh, i, j)
    n̂2 = SMatrix{2,4}(view(norms, 1:2, 2, i, j))

    ρʟ, ρʀ = @views i_face[1:2, 1, i, j]
    uʟ, uʀ = @views i_face[1:2, 2, i, j]
    vʟ, vʀ = @views i_face[1:2, 3, i, j]
    pʟ, pʀ = @views i_face[1:2, 4, i, j]

    p_2L, p_2R = @views i_face.p[1:2, 4, i, j + 1]
    p_1L, p_1R = @views i_face.p[1:2, 4, i, j - 1]
    neighbor_press = SVector{4,T}(p_2L, p_2R, p_1L, p_1R)
    min_neighbor_press = min_greater_than_zero(neighbor_press)

    ρᵢ, ρᵢ₊₁, ρᵢ₊₂ = @views W.ρ[i:(i + 2), j]
    uᵢ, uᵢ₊₁, uᵢ₊₂ = @views W.u[i:(i + 2), j]
    vᵢ, vᵢ₊₁, vᵢ₊₂ = @views W.v[i:(i + 2), j]
    pᵢ, pᵢ₊₁, pᵢ₊₂ = @views W.p[i:(i + 2), j]

    # normalize by density
    pᵢⱼ₊₁ = W.p[i, j + 1] / ρᵢ
    pᵢ₊₁ⱼ = W.p[i + 1, j] / ρᵢ
    pᵢⱼ₋₁ = W.p[i, j - 1] / ρᵢ
    pᵢ₊₁ⱼ₊₁ = W.p[i + 1, j + 1] / ρᵢ
    pᵢ₊₁ⱼ₋₁ = W.p[i + 1, j - 1] / ρᵢ
    # w₂ = modified_discontinuity_sensor_ξ(pᵢ, pᵢ₊₁ⱼ, pᵢ₊₁ⱼ₊₁, pᵢ₊₁ⱼ₋₁, pᵢⱼ₊₁, pᵢⱼ₋₁)
    w₂1 = modified_discontinuity_sensor_ξ(1.0, pᵢ₊₁ⱼ, pᵢ₊₁ⱼ₊₁, pᵢ₊₁ⱼ₋₁, pᵢⱼ₊₁, pᵢⱼ₋₁)
    # w₂2 = w2_ξ(1.0, pᵢ₊₁ⱼ, pᵢ₊₁ⱼ₊₁, pᵢ₊₁ⱼ₋₁, pᵢⱼ₊₁, pᵢⱼ₋₁)
    w₂ = w₂1

    Uᵢ = SVector{4,T}(ρᵢ, uᵢ, vᵢ, pᵢ)
    Uᵢ₊₁ = SVector{4,T}(ρᵢ₊₁, uᵢ₊₁, vᵢ₊₁, pᵢ₊₁)

    Uʟ = SVector{4,T}(ρʟ, uʟ, vʟ, pʟ)
    Uʀ = SVector{4,T}(ρʀ, uʀ, vʀ, pʀ)

    ρʟ_sb, ρʀ_sb = reconstruct_sb(ρᵢ, ρᵢ₊₁, ρᵢ₊₂)
    uʟ_sb, uʀ_sb = reconstruct_sb(uᵢ, uᵢ₊₁, uᵢ₊₂)
    vʟ_sb, vʀ_sb = reconstruct_sb(vᵢ, vᵢ₊₁, vᵢ₊₂)
    pʟ_sb, pʀ_sb = reconstruct_sb(pᵢ, pᵢ₊₁, pᵢ₊₂)

    Uʟ_sb = SVector{4,T}(ρʟ_sb, uʟ_sb, vʟ_sb, pʟ_sb)
    Uʀ_sb = SVector{4,T}(ρʀ_sb, uʀ_sb, vʀ_sb, pʀ_sb)

    # edge flux
    flux = M_AUSMPWPlus_2Dflux(
      solver, n̂2, Uᵢ, Uᵢ₊₁, Uʟ, Uʀ, Uʟ_sb, Uʀ_sb, min_neighbor_press, w₂, EOS
    )

    Fᵢ[i, j] = flux[1]
    Fᵢ[i, j] = flux[2]
    Fᵢ[i, j] = flux[3]
    Fᵢ[i, j] = flux[4]
  end

  # Do the j face
  if (jlo - 1 <= j <= jhi) && (ilo <= i <= ihi)
    _, _, n3, _ = facenorm(mesh, i, j)
    n̂3 = SVector{2}(n3)

    ρʟ, ρʀ = @views j_face.ρ[:, i, j]
    uʟ, uʀ = @views j_face.u[:, i, j]
    vʟ, vʀ = @views j_face.v[:, i, j]
    pʟ, pʀ = @views j_face.p[:, i, j]

    p_2L, p_2R = @views j_face.p[:, i - 1, j]
    p_1L, p_1R = @views j_face.p[:, i + 1, j]
    neighbor_press = SVector{4,T}(p_2L, p_2R, p_1L, p_1R)
    min_neighbor_press = min_greater_than_zero(neighbor_press)

    ρᵢ, ρⱼ₊₁, ρⱼ₊₂ = @views W.ρ[i, j:(j + 2)]
    uᵢ, uⱼ₊₁, uⱼ₊₂ = @views W.u[i, j:(j + 2)]
    vᵢ, vⱼ₊₁, vⱼ₊₂ = @views W.v[i, j:(j + 2)]
    pᵢ, pⱼ₊₁, pⱼ₊₂ = @views W.p[i, j:(j + 2)]

    pᵢ₊₁ⱼ = W.p[i + 1, j] / ρᵢ
    pᵢ₋₁ⱼ = W.p[i - 1, j] / ρᵢ
    pᵢⱼ₊₁ = W.p[i, j + 1] / ρᵢ
    pᵢ₊₁ⱼ₊₁ = W.p[i + 1, j + 1] / ρᵢ
    pᵢ₋₁ⱼ₊₁ = W.p[i - 1, j + 1] / ρᵢ
    # w₂ = modified_discontinuity_sensor_η(pᵢ, pᵢ₊₁ⱼ, pᵢ₊₁ⱼ₊₁, pᵢ₋₁ⱼ₊₁, pᵢ₋₁ⱼ, pᵢⱼ₊₁)
    w₂ = modified_discontinuity_sensor_η(1.0, pᵢ₊₁ⱼ, pᵢ₊₁ⱼ₊₁, pᵢ₋₁ⱼ₊₁, pᵢ₋₁ⱼ, pᵢⱼ₊₁)
    # w₂ = w2_η(1.0, pᵢ₊₁ⱼ, pᵢ₊₁ⱼ₊₁, pᵢ₋₁ⱼ₊₁, pᵢ₋₁ⱼ, pᵢⱼ₊₁)

    Uᵢ = SVector{4,T}(ρᵢ, uᵢ, vᵢ, pᵢ)
    Uⱼ₊₁ = SVector{4,T}(ρⱼ₊₁, uⱼ₊₁, vⱼ₊₁, pⱼ₊₁)

    Uʟ = SVector{4,T}(ρʟ, uʟ, vʟ, pʟ)
    Uʀ = SVector{4,T}(ρʀ, uʀ, vʀ, pʀ)

    ρʟ_sb, ρʀ_sb = reconstruct_sb(ρᵢ, ρⱼ₊₁, ρⱼ₊₂)
    uʟ_sb, uʀ_sb = reconstruct_sb(uᵢ, uⱼ₊₁, uⱼ₊₂)
    vʟ_sb, vʀ_sb = reconstruct_sb(vᵢ, vⱼ₊₁, vⱼ₊₂)
    pʟ_sb, pʀ_sb = reconstruct_sb(pᵢ, pⱼ₊₁, pⱼ₊₂)

    Uʟ_sb = SVector{4,T}(ρʟ_sb, uʟ_sb, vʟ_sb, pʟ_sb)
    Uʀ_sb = SVector{4,T}(ρʀ_sb, uʀ_sb, vʀ_sb, pʀ_sb)

    # edge flux
    flux = M_AUSMPWPlus_2Dflux(
      solver, n̂3, Uᵢ, Uⱼ₊₁, Uʟ, Uʀ, Uʟ_sb, Uʀ_sb, min_neighbor_press, w₂, EOS
    )

    G[i, j] = flux[1]
    G[i, j] = flux[2]
    G[i, j] = flux[3]
    G[i, j] = flux[4]
  end
end
