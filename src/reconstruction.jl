
@inline minmod(r) = max(0, min(r, 1))
@inline superbee(r) = max(0, min(2r, 1), min(r, 2))

function muscl(ϕᵢ₋₂, ϕᵢ₋₁, ϕᵢ, ϕᵢ₊₁, ϕᵢ₊₂, limiter)
    Δ_plus_half = ϕᵢ₊₁ - ϕᵢ
    Δ_plus_three_half = ϕᵢ₊₂ - ϕᵢ₊₁
    Δ_minus_half = ϕᵢ - ϕᵢ₋₁
    Δ_minus_three_half = ϕᵢ₋₂ - ϕᵢ₋₁

    # Δ_plus_half = Δ_plus_half * (abs(Δ_plus_half) >= ϵ)
    # Δ_plus_three_half = Δ_plus_three_half * (abs(Δ_plus_three_half) >= ϵ)
    # Δ_minus_half = Δ_minus_half * (abs(Δ_minus_half) >= ϵ)
    # Δ_minus_three_half = Δ_minus_three_half * (abs(Δ_minus_three_half) >= ϵ)

    eps = 1e-20
    rL⁺ = Δ_plus_half / (Δ_minus_half + eps)
    rR⁺ = Δ_plus_half / (Δ_plus_three_half + eps)
    rL⁻ = Δ_minus_half / (Δ_minus_three_half + eps)
    rR⁻ = Δ_minus_half / (Δ_plus_half + eps)

    lim_L⁺ = limiter(rL⁺)
    lim_R⁺ = limiter(rR⁺)
    lim_L⁻ = limiter(rL⁻)
    lim_R⁻ = limiter(rR⁻)

    ϕ_L⁺ = ϕᵢ + 0.5lim_L⁺ * Δ_plus_half          # i+1/2
    ϕ_R⁺ = ϕᵢ₊₁ - 0.5lim_R⁺ * Δ_plus_three_half  # i+1/2

    ϕ_R⁻ = ϕᵢ - 0.5lim_R⁻ * Δ_plus_half        # i-1/2
    ϕ_L⁻ = ϕᵢ₋₁ + 0.5lim_L⁻ * Δ_minus_half       # i-1/2

    return @SVector [(ϕ_L⁻, ϕ_R⁻), (ϕ_L⁺, ϕ_R⁺)]
end