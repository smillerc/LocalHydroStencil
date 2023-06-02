
abstract type AbstractRiemannSolver end
struct M_AUSMPWPlus2D <: AbstractRiemannSolver end

function MAUSMPW⁺(n̂, 
    ρ_LR::NTuple{2,T}, u_LR::NTuple{2,T}, v_LR::NTuple{2,T}, p_LR::NTuple{2,T}, 
    ρ_LR_SB::NTuple{2,T}, u_LR_SB::NTuple{2,T}, v_LR_SB::NTuple{2,T}, p_LR_SB::NTuple{2,T},
    W⃗ᵢ, W⃗ᵢ₊₁, w₂, EOS) where {T}

    ρʟ, ρʀ = ρ_LR
    uʟ, uʀ = u_LR
    vʟ, vʀ = v_LR
    pʟ, pʀ = p_LR

    ρʟ_sb, ρʀ_sb = ρ_LR_SB
    uʟ_sb, uʀ_sb = u_LR_SB
    vʟ_sb, vʀ_sb = v_LR_SB
    pʟ_sb, pʀ_sb = p_LR_SB

    nx, ny = n̂
    v⃗ʟ = SVector{2,Float64}(uʟ, vʟ)
    v⃗ʀ = SVector{2,Float64}(uʀ, vʀ)

    ρᵢ, uᵢ, vᵢ, pᵢ = W⃗ᵢ
    ρᵢ₊₁, uᵢ₊₁, vᵢ₊₁, pᵢ₊₁ = W⃗ᵢ₊₁

    Hʟ = total_enthalpy(EOS, ρʟ, uʟ, vʟ, pʟ)
    Hʀ = total_enthalpy(EOS, ρʀ, uʀ, vʀ, pʀ)

    # velocity component normal to the face edge
    Uʟ = uʟ * nx + vʟ * ny
    Uʀ = uʀ * nx + vʀ * ny
    Uᵢ = uᵢ * nx + vᵢ * ny
    Uᵢ₊₁ = uᵢ₊₁ * nx + vᵢ₊₁ * ny

    Vʟ = transverse_component(v⃗ʟ, n̂)
    Vʀ = transverse_component(v⃗ʀ, n̂)

    # Total enthalpy normal to the edge
    H_normal = min(Hʟ - 0.5Vʟ^2, Hʀ - 0.5Vʀ^2)

    # Speed of sound normal to the edge, also like the critical sound speed across a normal shock
    cₛ = sqrt(abs(2((EOS.γ - 1) / (EOS.γ + 1)) * H_normal))

    # Interface sound speed
    if 0.5(Uʟ + Uʀ) > 0
        c½ = cₛ^2 / max(abs(Uʟ), cₛ)
    else
        c½ = cₛ^2 / max(abs(Uʀ), cₛ)
    end

    # Left/Right Mach number
    Mʟ = Uʟ / c½
    Mʀ = Uʀ / c½

    # Mach splitting functions
    Mʟ⁺ = mach_split_plus(Mʟ)
    Mʀ⁻ = mach_split_minus(Mʀ)

    # Modified functions for M-AUSMPW+
    Mstarᵢ = Uᵢ / cₛ
    Mstarᵢ₊₁ = Uᵢ₊₁ / cₛ
    Pʟ⁺, Pʀ⁻ = modified_pressure_split(Mʟ, Mʀ, Mstarᵢ, Mstarᵢ₊₁, ρᵢ, Uᵢ, pᵢ, ρᵢ₊₁, Uᵢ₊₁, pᵢ₊₁)
    pₛ = pʟ * Pʟ⁺ + pʀ * Pʀ⁻
    w₁ = discontinuity_sensor(pʟ, pʀ)
    w = max(w₁, w₂)
    fʟ = modified_f(pʟ, pₛ, w₂)
    fʀ = modified_f(pʀ, pₛ, w₂)

    # More like AUSMPW+
    # Pʀ⁻ = pressure_split_minus(Mʀ)
    # Pʟ⁺ = pressure_split_plus(Mʟ)
    # pₛ = pʟ * Pʟ⁺ + pʀ * Pʀ⁻
    # w = discontinuity_sensor(pʟ, pʀ)
    # fʟ, fʀ = pressure_based_weight_function(pʟ, pʀ, pₛ, min_neighbor_press)

    # From Eq. 24 (ii) in Ref [1]
    if Mʟ⁺ + Mʀ⁻ < 0
        M̄ʟ⁺ = Mʟ⁺ * w * (1 + fʟ)
        M̄ʀ⁻ = Mʀ⁻ + Mʟ⁺ * ((1 - w) * (1 + fʟ) - fʀ)
    else # From Eq. 24 (i) in Ref [1]
        M̄ʟ⁺ = Mʟ⁺ + Mʀ⁻ * ((1 - w) * (1 + fʀ) - fʟ)
        M̄ʀ⁻ = Mʀ⁻ * w * (1 + fʀ)
    end
    M̄ʟ⁺ = M̄ʟ⁺ * (abs(M̄ʟ⁺) >= 1e-15)
    M̄ʀ⁻ = M̄ʀ⁻ * (abs(M̄ʀ⁻) >= 1e-15)

    a = 1 - min(1, max(abs(Mʟ), abs(Mʀ)))^2
    ρʟ½ = ϕ_L_half(ρʟ, ρʀ, ρʟ_sb, a)
    uʟ½ = ϕ_L_half(uʟ, uʀ, uʟ_sb, a)
    vʟ½ = ϕ_L_half(vʟ, vʀ, vʟ_sb, a)
    pʟ½ = ϕ_L_half(pʟ, pʀ, pʟ_sb, a)

    ρʀ½ = ϕ_R_half(ρʟ, ρʀ, ρʀ_sb, a)
    uʀ½ = ϕ_R_half(uʟ, uʀ, uʀ_sb, a)
    vʀ½ = ϕ_R_half(vʟ, vʀ, vʀ_sb, a)
    pʀ½ = ϕ_R_half(pʟ, pʀ, pʀ_sb, a)

    # mass fluxes
    ṁʟ = M̄ʟ⁺ * c½ * ρʟ½
    ṁʀ = M̄ʀ⁻ * c½ * ρʀ½

    Hʟ½ = total_enthalpy(EOS, ρʟ½, uʟ½, vʟ½, pʟ½)
    Hʀ½ = total_enthalpy(EOS, ρʀ½, uʀ½, vʀ½, pʀ½)

    ρflux = ṁʟ + ṁʀ
    ρuflux = (ṁʟ * uʟ½) + (ṁʀ * uʀ½) + ((Pʟ⁺ * nx * pʟ½) + (Pʀ⁻ * nx * pʀ½))
    ρvflux = (ṁʟ * vʟ½) + (ṁʀ * vʀ½) + ((Pʟ⁺ * ny * pʟ½) + (Pʀ⁻ * ny * pʀ½))
    Eflux = (ṁʟ * Hʟ½) + (ṁʀ * Hʀ½)

    ρflux = ρflux * (abs(ρflux) >= ϵ)
    ρuflux = ρuflux * (abs(ρuflux) >= ϵ)
    ρvflux = ρvflux * (abs(ρvflux) >= ϵ)
    Eflux = Eflux * (abs(Eflux) >= ϵ)

    return SVector{4,Float64}(ρflux, ρuflux, ρvflux, Eflux)
end

@inline function pressure_based_weight_function(p_L::T, p_R::T, p_s::T, min_neighbor_press::T) where T
    if abs(p_s) < typemin(T) # p_s == 0
        f_L = zero(T)
        f_R = zero(T)
    else
        min_term = min(1.0, min_neighbor_press / min(p_L, p_R))^2

        f_L_term = (p_L / p_s) - 1.0
        f_R_term = (p_R / p_s) - 1.0

        f_L_term = f_L_term * (abs(f_L_term) >= ϵ)
        f_R_term = f_R_term * (abs(f_R_term) >= ϵ)

        f_L = f_L_term * min_term
        f_R = f_R_term * min_term
    end

    return f_L, f_R
end

@inline function discontinuity_sensor(p_L::T, p_R::T) where T
    min_term = min((p_L / p_R), (p_R / p_L))
    w = 1 - (min_term * min_term * min_term)
    w = w * (abs(w) >= ϵ)
    return w
end

@inline function pressure_split_plus(M::T; α=zero(T)) where T
    if abs(M) > 1
        P⁺ = 0.5(1 + sign(M))
    else #  |M| <= 1
        P⁺ = 0.25(M + 1)^2 * (2 - M) + α * M * (M^2 - 1)^2
    end
    return P⁺
end

@inline function pressure_split_minus(M::T; α=zero(T)) where T
    if abs(M) > 1
        P⁻ = 0.5(1 - sign(M))
    else #  |M| <= 1
        P⁻ = 0.25(M - 1)^2 * (2 + M) - α * M * (M^2 - 1)^2
    end
    return P⁻
end

@inline function mach_split_plus(M)
    if abs(M) > 1
        M⁺ = 0.5(M + abs(M))
    else #  |M| <= 1
        M⁺ = 0.25(M + 1)^2
    end
    return M⁺
end

@inline function mach_split_minus(M)
    if abs(M) > 1
        M⁻ = 0.5(M - abs(M))
    else #  |M| <= 1
        M⁻ = -0.25(M - 1)^2
    end
    return M⁻
end

@inline function transverse_component(v⃗, n̂)
    # velocity component transverse to the face edge
    v_perp = (v⃗ ⋅ n̂) .* n̂ # normal
    v_parallel = v⃗ .- v_perp # transverse
    v_parallel = v_parallel .* (abs.(v_parallel) .>= 1e-15)
    return norm(v_parallel)
end

@inline function reconstruct_sb(ϕᵢ, ϕᵢ₊₁, ϕᵢ₊₂)

    @inline superbee(r) = max(0, min(2r, 1), min(r, 2))

    Δ⁻½ = ϕᵢ - ϕᵢ₊₁
    Δ⁺½ = ϕᵢ₊₁ - ϕᵢ
    Δ⁺_three_half = ϕᵢ₊₂ - ϕᵢ₊₁

    rL = Δ⁺½ / (Δ⁻½ + ϵ)
    rR = Δ⁺½ / (Δ⁺_three_half + ϵ)

    ϕ_left = ϕᵢ + 0.5superbee(rL) * Δ⁻½
    ϕ_right = ϕᵢ₊₁ - 0.5superbee(rR) * Δ⁺_three_half

    return ϕ_left, ϕ_right
end

@inline function modified_pressure_split(Mʟ, Mʀ, Mstarᵢ, Mstarᵢ₊₁, ρᵢ, uᵢ, pᵢ, ρᵢ₊₁, uᵢ₊₁, pᵢ₊₁)

    if Mstarᵢ > 1 && Mstarᵢ₊₁ < 1 && 0 < Mstarᵢ * Mstarᵢ₊₁ < 1
        P⁻ᵢ₊₁ = max(0, min(0.5, 1 - ((ρᵢ * uᵢ * (uᵢ - uᵢ₊₁) + pᵢ) / pᵢ₊₁)))
    else
        P⁻ᵢ₊₁ = pressure_split_minus(Mʀ)
    end

    if Mstarᵢ > -1 && Mstarᵢ₊₁ < -1 && 0 < Mstarᵢ * Mstarᵢ₊₁ < 1
        P⁺ᵢ = max(0, min(0.5, 1 - (ρᵢ₊₁ * uᵢ₊₁ * (uᵢ₊₁ - uᵢ) + pᵢ₊₁) / pᵢ))
    else
        P⁺ᵢ = pressure_split_plus(Mʟ)
    end

    return P⁺ᵢ, P⁻ᵢ₊₁
end

@inline function modified_discontinuity_sensor_ξ(p̄ᵢⱼ::T, p̄ᵢ₊₁ⱼ::T, p̄ᵢ₊₁ⱼ₊₁::T, p̄ᵢ₊₁ⱼ₋₁::T, p̄ᵢⱼ₊₁::T, p̄ᵢⱼ₋₁::T) where T
    Δp = abs(p̄ᵢ₊₁ⱼ - p̄ᵢⱼ)
    Δpᵢ₊₁ = abs(p̄ᵢ₊₁ⱼ₊₁ - p̄ᵢ₊₁ⱼ₋₁)
    Δpᵢ = abs(p̄ᵢⱼ₊₁ - p̄ᵢⱼ₋₁)

    w₂ = (1 - min(1, Δp / (0.25(Δpᵢ₊₁ + Δpᵢ))))^2 * (1 - min((p̄ᵢⱼ / p̄ᵢ₊₁ⱼ), (p̄ᵢ₊₁ⱼ / p̄ᵢⱼ)))^2
    if isnan(w₂) || abs(w₂) < ϵ
        w₂ = zero(T)
    end

    return w₂
end

@inline function w2_ξ(p̄ᵢⱼ::T, p̄ᵢ₊₁ⱼ::T, p̄ᵢ₊₁ⱼ₊₁::T, p̄ᵢ₊₁ⱼ₋₁::T, p̄ᵢⱼ₊₁::T, p̄ᵢⱼ₋₁::T) where T
    denom = 0.25(p̄ᵢ₊₁ⱼ₊₁ + p̄ᵢⱼ₊₁ - p̄ᵢ₊₁ⱼ₋₁ - p̄ᵢⱼ₋₁)
    first_term = (1 - min(1, (p̄ᵢ₊₁ⱼ - p̄ᵢⱼ) / denom))^2
    second_term = (1 - min((p̄ᵢⱼ / p̄ᵢ₊₁ⱼ), (p̄ᵢ₊₁ⱼ / p̄ᵢⱼ)))^2
    w₂ = first_term * second_term
    return w₂
end

@inline function modified_discontinuity_sensor_η(p̄ᵢⱼ::T, p̄ᵢ₊₁ⱼ::T, p̄ᵢ₊₁ⱼ₊₁::T, p̄ᵢ₋₁ⱼ₊₁::T, p̄ᵢ₋₁ⱼ::T, p̄ᵢⱼ₊₁::T) where T
    Δp = abs(p̄ᵢⱼ₊₁ - p̄ᵢⱼ)
    Δpⱼ₊₁ = abs(p̄ᵢ₊₁ⱼ₊₁ - p̄ᵢ₋₁ⱼ₊₁)
    Δpⱼ = abs(p̄ᵢ₊₁ⱼ - p̄ᵢ₋₁ⱼ)

    w₂ = (1 - min(1, Δp / (0.25(Δpⱼ₊₁ + Δpⱼ))))^2 * (1 - min((p̄ᵢⱼ / p̄ᵢⱼ₊₁), (p̄ᵢⱼ₊₁ / p̄ᵢⱼ)))^2
    if isnan(w₂) || abs(w₂) < ϵ
        w₂ = zero(T)
    end

    return w₂
end

@inline function w2_η(p̄ᵢⱼ::T, p̄ᵢ₊₁ⱼ::T, p̄ᵢ₊₁ⱼ₊₁::T, p̄ᵢ₋₁ⱼ₊₁::T, p̄ᵢ₋₁ⱼ::T, p̄ᵢⱼ₊₁::T) where T
    denom = 0.25(p̄ᵢ₊₁ⱼ₊₁ + p̄ᵢ₊₁ⱼ - p̄ᵢ₋₁ⱼ₊₁ - p̄ᵢ₋₁ⱼ)
    first_term = (1 - min(1, (p̄ᵢⱼ₊₁ - p̄ᵢⱼ) / denom))^2
    second_term = (1 - min((p̄ᵢⱼ / p̄ᵢⱼ₊₁), (p̄ᵢⱼ₊₁ / p̄ᵢⱼ)))^2
    w₂ = first_term * second_term
    return w₂
end

@inline function modified_f(pʟʀ::T, pₛ::T, w₂::T) where T
    if abs(pₛ) > zero(T)
        f = ((pʟʀ / pₛ) - 1) * (1 - w₂)
    else
        f = zero(T)
    end
    return f
end

@inline function ϕ_L_half(ϕ_L, ϕ_R, ϕ_L_sb, a)

    if abs(ϕ_R - ϕ_L) < 1.0 || abs(ϕ_L_sb - ϕ_L) < ϵ
        ϕ_L_half = ϕ_L
    else
        ϕ_L_half = ϕ_L + max(0, (ϕ_R - ϕ_L) * (ϕ_L_sb - ϕ_L)) /
                         ((ϕ_R - ϕ_L) * abs(ϕ_L_sb - ϕ_L)) * min(a, 0.5 * abs(ϕ_R - ϕ_L), abs(ϕ_L_sb - ϕ_L))
    end

end

@inline function ϕ_R_half(ϕ_L, ϕ_R, ϕ_R_sb, a)
    if abs(ϕ_L - ϕ_R) < ϵ || abs(ϕ_R_sb - ϕ_R) < 1.0
        ϕ_R_half = ϕ_R
    else
        ϕ_R_half = ϕ_R + max(0, (ϕ_L - ϕ_R) * (ϕ_R_sb - ϕ_R)) /
                         ((ϕ_L - ϕ_R) * abs(ϕ_R_sb - ϕ_R)) * min(a, 0.5 * abs(ϕ_L - ϕ_R), abs(ϕ_R_sb - ϕ_R))
    end
end
