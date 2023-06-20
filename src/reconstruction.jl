module ReconstructionType

using LoopVectorization
using StaticArrays

export muscl, minmod, superbee
export muscl_sarr_turbo_split2

@inline minmod(r) = max(0, min(r, 1))
@inline superbee(r) = max(0, min(2r, 1), min(r, 2))

function muscl(ϕᵢ₋₂, ϕᵢ₋₁, ϕᵢ, ϕᵢ₊₁, ϕᵢ₊₂, limiter)
    Δ_plus_half = ϕᵢ₊₁ - ϕᵢ
    Δ_minus_half = ϕᵢ - ϕᵢ₋₁
    Δ_plus_three_half = ϕᵢ₊₂ - ϕᵢ₊₁
    Δ_minus_three_half = ϕᵢ₋₁ - ϕᵢ₋₂

    # Δ_plus_half = Δ_plus_half * (abs(Δ_plus_half) >= ϵ)
    # Δ_plus_three_half = Δ_plus_three_half * (abs(Δ_plus_three_half) >= ϵ)
    # Δ_minus_half = Δ_minus_half * (abs(Δ_minus_half) >= ϵ)
    # Δ_minus_three_half = Δ_minus_three_half * (abs(Δ_minus_three_half) >= ϵ)

    eps = 1e-20
    rL⁺ = Δ_plus_half / (Δ_minus_half + eps)
    rL⁻ = Δ_minus_half / (Δ_minus_three_half + eps)
    
    rR⁺ = Δ_plus_half / (Δ_plus_three_half + eps)
    rR⁻ = Δ_minus_half / (Δ_plus_half + eps)

    lim_L⁺ = limiter(rL⁺)
    lim_L⁻ = limiter(rL⁻)
    lim_R⁺ = limiter(rR⁺)
    lim_R⁻ = limiter(rR⁻)

   ϕ_L⁺ = ϕᵢ   + 0.5lim_L⁺ * Δ_minus_half        # i+1/2
   ϕ_L⁻ = ϕᵢ₋₁ + 0.5lim_L⁻ * Δ_minus_three_half  # i-1/2

   ϕ_R⁺ = ϕᵢ₊₁ - 0.5lim_R⁺ * Δ_plus_three_half  # i+1/2
   ϕ_R⁻ = ϕᵢ   - 0.5lim_R⁻ * Δ_plus_half        # i-1/2

    return @SVector [(ϕ_L⁻, ϕ_R⁻), (ϕ_L⁺, ϕ_R⁺)]
end

function muscl_sarr(ϕᵢ₋₂::SVector{4,T}, ϕᵢ₋₁::SVector{4,T}, ϕᵢ::SVector{4,T}, ϕᵢ₊₁::SVector{4,T}, ϕᵢ₊₂::SVector{4,T}, limiter::F) where {T, F}
    LR = @MArray zeros(4,4)

    L⁻ = 1
    R⁻ = 2
    L⁺ = 3
    R⁺ = 4
    eps = 1e-20

    @inbounds for q in eachindex(ϕᵢ)
        Δ_plus_half = ϕᵢ₊₁[q] - ϕᵢ[q]
        Δ_minus_half = ϕᵢ[q] - ϕᵢ₋₁[q]
        Δ_plus_three_half = ϕᵢ₊₂[q] - ϕᵢ₊₁[q]
        Δ_minus_three_half = ϕᵢ₋₁[q] - ϕᵢ₋₂[q]
                
        rL⁺ = Δ_plus_half / (Δ_minus_half + eps)
        rL⁻ = Δ_minus_half / (Δ_minus_three_half + eps)
        
        rR⁺ = Δ_plus_half / (Δ_plus_three_half + eps)
        rR⁻ = Δ_minus_half / (Δ_plus_half + eps)
    
        lim_L⁺ = limiter(rL⁺)
        lim_L⁻ = limiter(rL⁻)
        lim_R⁺ = limiter(rR⁺)
        lim_R⁻ = limiter(rR⁻)
        
        LR[L⁻,q] = ϕᵢ₋₁[q] + 0.5lim_L⁻ * Δ_minus_three_half  # i-1/2
        LR[R⁻,q] = ϕᵢ[q]   - 0.5lim_R⁻ * Δ_plus_half         # i-1/2
        
        LR[L⁺,q] = ϕᵢ[q]   + 0.5lim_L⁺ * Δ_minus_half        # i+1/2
        LR[R⁺,q] = ϕᵢ₊₁[q] - 0.5lim_R⁺ * Δ_plus_three_half   # i+1/2

        # ϕ_L⁻ = ϕᵢ₋₁ + 0.5lim_L⁻ * Δ_minus_three_half  # i-1/2
        # ϕ_R⁻ = ϕᵢ   - 0.5lim_R⁻ * Δ_plus_half        # i-1/2
        # ϕ_L⁺ = ϕᵢ   + 0.5lim_L⁺ * Δ_minus_half        # i+1/2
        # ϕ_R⁺ = ϕᵢ₊₁ - 0.5lim_R⁺ * Δ_plus_three_half  # i+1/2
    end

    return SArray(LR)
    # return @SVector [(ϕ_L⁻, ϕ_R⁻), (ϕ_L⁺, ϕ_R⁺)]
end

function muscl_sarr_turbo(ϕᵢ₋₂::SVector{4,T}, ϕᵢ₋₁::SVector{4,T}, ϕᵢ::SVector{4,T}, ϕᵢ₊₁::SVector{4,T}, ϕᵢ₊₂::SVector{4,T}, limiter::F) where {T, F}
    LR = @MArray zeros(4,4)

    eps = 1e-20

    @turbo for q in eachindex(ϕᵢ)
        Δ_plus_half = ϕᵢ₊₁[q] - ϕᵢ[q]
        Δ_minus_half = ϕᵢ[q] - ϕᵢ₋₁[q]
        Δ_plus_three_half = ϕᵢ₊₂[q] - ϕᵢ₊₁[q]
        Δ_minus_three_half = ϕᵢ₋₁[q] - ϕᵢ₋₂[q]
                
        rL⁺ = Δ_plus_half / (Δ_minus_half + eps)
        rL⁻ = Δ_minus_half / (Δ_minus_three_half + eps)
        
        rR⁺ = Δ_plus_half / (Δ_plus_three_half + eps)
        rR⁻ = Δ_minus_half / (Δ_plus_half + eps)
    
        lim_L⁺ = limiter(rL⁺)
        lim_L⁻ = limiter(rL⁻)
        lim_R⁺ = limiter(rR⁺)
        lim_R⁻ = limiter(rR⁻)
        
        LR[1,q] = ϕᵢ₋₁[q] + 0.5lim_L⁻ * Δ_minus_three_half  # i-1/2
        LR[2,q] = ϕᵢ[q]   - 0.5lim_R⁻ * Δ_plus_half         # i-1/2
        LR[3,q] = ϕᵢ[q]   + 0.5lim_L⁺ * Δ_minus_half        # i+1/2
        LR[4,q] = ϕᵢ₊₁[q] - 0.5lim_R⁺ * Δ_plus_three_half   # i+1/2
    end

    return SArray(LR)
    # return @SVector [(ϕ_L⁻, ϕ_R⁻), (ϕ_L⁺, ϕ_R⁺)]
end

function muscl_sarr_turbo_split(ϕᵢ₋₂::SVector{4,T}, ϕᵢ₋₁::SVector{4,T}, ϕᵢ::SVector{4,T}, ϕᵢ₊₁::SVector{4,T}, ϕᵢ₊₂::SVector{4,T}, limiter::F) where {T, F}
    LR = @MArray zeros(4,4)

    eps = 1e-20

    @turbo for q in eachindex(ϕᵢ)
        Δ_minus_half = ϕᵢ[q] - ϕᵢ₋₁[q]
        Δ_minus_three_half = ϕᵢ₋₁[q] - ϕᵢ₋₂[q]
        rL⁻ = Δ_minus_half / (Δ_minus_three_half + eps)
        lim_L⁻ = limiter(rL⁻)
        LR[1,q] = ϕᵢ₋₁[q] + 0.5lim_L⁻ * Δ_minus_three_half  # i-1/2
    end

    @turbo for q in eachindex(ϕᵢ)
        Δ_plus_half = ϕᵢ₊₁[q] - ϕᵢ[q]
        Δ_minus_half = ϕᵢ[q] - ϕᵢ₋₁[q]               
        rR⁻ = Δ_minus_half / (Δ_plus_half + eps)
        lim_R⁻ = limiter(rR⁻)
        LR[2,q] = ϕᵢ[q]   - 0.5lim_R⁻ * Δ_plus_half         # i-1/2
    end

    @turbo for q in eachindex(ϕᵢ)
        Δ_plus_half = ϕᵢ₊₁[q] - ϕᵢ[q]
        Δ_minus_half = ϕᵢ[q] - ϕᵢ₋₁[q]
        rL⁺ = Δ_plus_half / (Δ_minus_half + eps)
        lim_L⁺ = limiter(rL⁺)
        LR[3,q] = ϕᵢ[q]   + 0.5lim_L⁺ * Δ_minus_half        # i+1/2
    end

    @turbo for q in eachindex(ϕᵢ)
        Δ_plus_half = ϕᵢ₊₁[q] - ϕᵢ[q]
        Δ_plus_three_half = ϕᵢ₊₂[q] - ϕᵢ₊₁[q]
        rR⁺ = Δ_plus_half / (Δ_plus_three_half + eps)
        lim_R⁺ = limiter(rR⁺)
        LR[4,q] = ϕᵢ₊₁[q] - 0.5lim_R⁺ * Δ_plus_three_half   # i+1/2
    end
    return SArray(LR)
    # return @SVector [(ϕ_L⁻, ϕ_R⁻), (ϕ_L⁺, ϕ_R⁺)]
end

function muscl_sarr_turbo_split2(ϕᵢ₋₂::SVector{4,T}, ϕᵢ₋₁::SVector{4,T}, ϕᵢ::SVector{4,T}, ϕᵢ₊₁::SVector{4,T}, ϕᵢ₊₂::SVector{4,T}, limiter::F) where {T, F}
    ϕ_L⁻ = @MVector zeros(4)
    ϕ_R⁻ = @MVector zeros(4)
    ϕ_L⁺ = @MVector zeros(4)
    ϕ_R⁺ = @MVector zeros(4)

    eps = 1e-20

    @turbo for q in eachindex(ϕᵢ)
        Δ_minus_half = ϕᵢ[q] - ϕᵢ₋₁[q]
        Δ_minus_three_half = ϕᵢ₋₁[q] - ϕᵢ₋₂[q]
        rL⁻ = Δ_minus_half / (Δ_minus_three_half + eps)
        lim_L⁻ = limiter(rL⁻)
        ϕ_L⁻[q] = ϕᵢ₋₁[q] + 0.5lim_L⁻ * Δ_minus_three_half  # i-1/2
    end

    @turbo for q in eachindex(ϕᵢ)
        Δ_plus_half = ϕᵢ₊₁[q] - ϕᵢ[q]
        Δ_minus_half = ϕᵢ[q] - ϕᵢ₋₁[q]               
        rR⁻ = Δ_minus_half / (Δ_plus_half + eps)
        lim_R⁻ = limiter(rR⁻)
        ϕ_R⁻[q] = ϕᵢ[q]   - 0.5lim_R⁻ * Δ_plus_half         # i-1/2
    end

    @turbo for q in eachindex(ϕᵢ)
        Δ_plus_half = ϕᵢ₊₁[q] - ϕᵢ[q]
        Δ_minus_half = ϕᵢ[q] - ϕᵢ₋₁[q]
        rL⁺ = Δ_plus_half / (Δ_minus_half + eps)
        lim_L⁺ = limiter(rL⁺)
        ϕ_L⁺[q] = ϕᵢ[q]   + 0.5lim_L⁺ * Δ_minus_half        # i+1/2
    end

    @turbo for q in eachindex(ϕᵢ)
        Δ_plus_half = ϕᵢ₊₁[q] - ϕᵢ[q]
        Δ_plus_three_half = ϕᵢ₊₂[q] - ϕᵢ₊₁[q]
        rR⁺ = Δ_plus_half / (Δ_plus_three_half + eps)
        lim_R⁺ = limiter(rR⁺)
        ϕ_R⁺[q] = ϕᵢ₊₁[q] - 0.5lim_R⁺ * Δ_plus_three_half   # i+1/2
    end
    
    return SVector(ϕ_L⁻), SVector(ϕ_R⁻), SVector(ϕ_L⁺), SVector(ϕ_R⁺)
end


end
