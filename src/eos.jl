module EOSType

using Unitful, StaticArrays

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

end
