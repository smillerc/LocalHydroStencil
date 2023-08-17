module EOSType

using Unitful, StaticArrays, LoopVectorization

export IdealEOS, total_enthalpy, sound_speed, pressure, specific_total_energy, cons2prim

"""Ideal gas equation of state as a function of γ"""
struct IdealEOS{T<:AbstractFloat}
  γ::T  # polytropic index
  cᵥ::T # molar specific heat capacity of a gas at constant volume
  cₚ::T # molar specific heat capacity of a gas at constant pressure
  R::T  # specific gas constant per unit mass in cgs units
  _γ_m_1::T # γ-1
  _inv_γ_m_1::T # 1 / (γ-1)
  _γ_over_γ_m_1::T # γ / (γ-1)
end

function IdealEOS(γ::T, R=287.05u"J * kg^-1 * K^-1") where {T<:Real}
  R = ustrip(u"erg * g^-1 * K^-1", R)

  _inv_γ_m_1 = round(1 / (γ - 1); sigdigits=15)
  _γ_m_1 = 1 / _inv_γ_m_1
  _γ_over_γ_m_1 = γ / (γ - 1)

  cᵥ = R * _inv_γ_m_1
  cₚ = γ * cᵥ

  return IdealEOS(γ, cᵥ, cₚ, R, _γ_m_1, _inv_γ_m_1, _γ_over_γ_m_1)
end

@inline total_enthalpy(eos, ρ, u, v, p) = (p / ρ) * eos._γ_over_γ_m_1 + 0.5(u^2 + v^2)
@inline sound_speed(eos, ρ, p) = sqrt(eos.γ * abs(p / ρ))
@inline pressure(eos, ρ, u, v, E) = ρ * eos._γ_m_1 * (E - 0.5(u^2 + v^2))
@inline specific_total_energy(eos, ρ, u, v, p) = (p / ρ) * eos._inv_γ_m_1 + 0.5(u^2 + v^2)

function cons2prim(EOS::IdealEOS, U)
  ρ = U[1]
  invρ = 1 / ρ
  u = U[2] * invρ
  v = U[3] * invρ
  E = U[4] * invρ
  # p = pressure(EOS, ρ, u, v, E)
  p = ρ * EOS._γ_m_1 * (E - 0.5(u^2 + v^2))

  return @SVector [ρ, u, v, p]
end

function cons2prim_sarr(EOS::IdealEOS, U)
  W = MArray{Tuple{4,5,5},Float64,3,100}(undef)
  # for i in axes(U, )

  for j in axes(W, 3)
    for i in axes(W, 2)
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

  return SArray(W)
end

function cons2prim_sarr_turbo(EOS::IdealEOS, U)
  W = MArray{Tuple{4,5,5},Float64,3,100}(undef)
  # for i in axes(U, )

  @turbo for j in axes(W, 3)
    for i in axes(W, 2)
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

  return SArray(W)
end

function cons2prim_sarr_turbo_split(EOS::IdealEOS, ρ, ρu, ρv, E)
  u = similar(ρ)
  v = similar(ρ)
  p = similar(ρ)

  @turbo for i in eachindex(ρ)
    u[i] = ρu[i] / ρ[i]
    v[i] = ρv[i] / ρ[i]
    p[i] = ρ[i] * EOS._γ_m_1 * (E[i] - 0.5(u[i]^2 + v[i]^2))
  end
  # @turbo for i in eachindex(ρ)
  #     u[i] = ρu[i] / ρ[i]
  # end
  # @turbo for i in eachindex(ρ)
  #     v[i] = ρv[i] / ρ[i]
  # end
  # @turbo for i in eachindex(ρ)
  #     p[i] = ρ[i] * EOS._γ_m_1 * (E[i] - 0.5(u[i]^2 + v[i]^2))
  # end

  return SArray(u), SArray(v), SArray(p)
end

function cons2prim_sarr_turbo_blk(EOS::IdealEOS, U)

  # W = MArray{Tuple{4, 5, 5}, Float64, 3, 100}(undef)
  # W = similar(U)
  # W = SMatrix{5, 5, MVector{4, Float64}, 25}(undef)
  W = @SArray [MVector{4,Float64}(undef) for j in 1:5, i in 1:5]
  for i in eachindex(W)
    ρ = U[i][1]
    invρ = 1 / ρ

    u = U[i][2] * invρ
    v = U[i][3] * invρ
    E = U[i][4] * invρ

    p = ρ * EOS._γ_m_1 * (E - 0.5(u^2 + v^2))
    W[i][1] = ρ
    W[i][2] = u
    W[i][3] = v
    W[i][4] = p
  end

  return SArray.(W)
end

end
