using StaticArrays, BenchmarkTools, Polyester, .Threads

# @inline minmod(r) = max(0, min(r, 1))
# @inline superbee(r) = max(0, min(2r, 1), min(r, 2))

# function muscl(ϕᵢ₋₁, ϕᵢ, ϕᵢ₊₁, ϕᵢ₊₂, limiter)
#     Δ_plus_half = ϕᵢ₊₁ - ϕᵢ
#     Δ_plus_three_half = ϕᵢ₊₂ - ϕᵢ₊₁
#     Δ_minus_half = ϕᵢ - ϕᵢ₋₁
#     # Δ_minus_three_half = ϕᵢ₋₂ - ϕᵢ₋₁

#     # Δ_plus_half = Δ_plus_half * (abs(Δ_plus_half) >= ϵ)
#     # Δ_plus_three_half = Δ_plus_three_half * (abs(Δ_plus_three_half) >= ϵ)
#     # Δ_minus_half = Δ_minus_half * (abs(Δ_minus_half) >= ϵ)
#     # Δ_minus_three_half = Δ_minus_three_half * (abs(Δ_minus_three_half) >= ϵ)

#     eps = 1e-20
#     rL⁺ = Δ_plus_half / (Δ_minus_half + eps)
#     rR⁺ = Δ_plus_half / (Δ_plus_three_half + eps)
#     # rL⁻ = Δ_minus_half / (Δ_minus_three_half + eps)
#     # rR⁻ = Δ_minus_half / (Δ_plus_half + eps)

#     lim_L⁺ = limiter(rL⁺)
#     lim_R⁺ = limiter(rR⁺)
#     # lim_L⁻ = limiter(rL⁻)
#     # lim_R⁻ = limiter(rR⁻)

#     ϕ_L⁺ = ϕᵢ + 0.5lim_L⁺ * Δ_plus_half          # i+1/2
#     ϕ_R⁺ = ϕᵢ₊₁ - 0.5lim_R⁺ * Δ_plus_three_half  # i+1/2

#     # ϕ_R⁻ = ϕᵢ - 0.5lim_R⁻ * Δ_plus_half        # i-1/2
#     # ϕ_L⁻ = ϕᵢ₋₁ + 0.5lim_L⁻ * Δ_minus_half       # i-1/2

#     return ϕ_L⁺, ϕ_R⁺
# end

# function muscl_both(ϕᵢ₋₂, ϕᵢ₋₁, ϕᵢ, ϕᵢ₊₁, ϕᵢ₊₂, limiter)
#     Δ_plus_half = ϕᵢ₊₁ - ϕᵢ
#     Δ_plus_three_half = ϕᵢ₊₂ - ϕᵢ₊₁
#     Δ_minus_half = ϕᵢ - ϕᵢ₋₁
#     Δ_minus_three_half = ϕᵢ₋₂ - ϕᵢ₋₁

#     # Δ_plus_half = Δ_plus_half * (abs(Δ_plus_half) >= ϵ)
#     # Δ_plus_three_half = Δ_plus_three_half * (abs(Δ_plus_three_half) >= ϵ)
#     # Δ_minus_half = Δ_minus_half * (abs(Δ_minus_half) >= ϵ)
#     # Δ_minus_three_half = Δ_minus_three_half * (abs(Δ_minus_three_half) >= ϵ)

#     eps = 1e-20
#     rL⁺ = Δ_plus_half / (Δ_minus_half + eps)
#     rR⁺ = Δ_plus_half / (Δ_plus_three_half + eps)
#     rL⁻ = Δ_minus_half / (Δ_minus_three_half + eps)
#     rR⁻ = Δ_minus_half / (Δ_plus_half + eps)

#     lim_L⁺ = limiter(rL⁺)
#     lim_R⁺ = limiter(rR⁺)
#     lim_L⁻ = limiter(rL⁻)
#     lim_R⁻ = limiter(rR⁻)

#     ϕ_L⁺ = ϕᵢ + 0.5lim_L⁺ * Δ_plus_half          # i+1/2
#     ϕ_R⁺ = ϕᵢ₊₁ - 0.5lim_R⁺ * Δ_plus_three_half  # i+1/2

#     ϕ_R⁻ = ϕᵢ - 0.5lim_R⁻ * Δ_plus_half        # i-1/2
#     ϕ_L⁻ = ϕᵢ₋₁ + 0.5lim_L⁻ * Δ_minus_half       # i-1/2

#     return @SVector [(ϕ_L⁻, ϕ_R⁻), (ϕ_L⁺, ϕ_R⁺)]
# end

# ni = 10
# nj = 10

# looplims = (3,ni-2,3,nj-2)

# W = (ρ=rand(ni, nj), u=rand(ni, nj), v=rand(ni, nj), p=rand(ni, nj));

# Wpacked = rand(4, ni, nj);
# W_edge_i_packed = rand(2, 4, ni, nj);
# W_edge_j_packed = rand(2, 4, ni, nj);

# Whybrid = [
#     @SVector rand(4) for j in 1:nj, i in 1:ni
# ];

# Wedge_i = [
#     @SVector [@SVector [0.0,0.0] for _ in 1:4] for j in 1:nj, i in 1:ni
# ]

# Wedge_j = [
#     @SVector [@SVector [0.0,0.0] for _ in 1:4] for j in 1:nj, i in 1:ni
# ]

# function recon_hybrid(W, W_edge_i, W_edge_j, limiter::F, looplimits) where {F}
#     ilo, ihi, jlo, jhi = looplimits

#     @batch for j in jlo:jhi
#         for i in ilo:ihi
#             ϕᵢ₋₁ = W[i-1, j]
#             ϕᵢ = W[i, j]
#             ϕᵢ₊₁ = W[i+1, j]
#             ϕᵢ₊₂ = W[i+2, j]

#             @inbounds LR = muscl.(ϕᵢ₋₁, ϕᵢ, ϕᵢ₊₁, ϕᵢ₊₂, limiter)
#             W_edge_i[i, j] = LR
#             # ρᴸᴿᵢ, uᴸᴿᵢ, vᴸᴿᵢ, pᴸᴿᵢ = muscl.(ϕᵢ₋₁, ϕᵢ, ϕᵢ₊₁, ϕᵢ₊₂, limiter)
#             # W_edge_i[i, j] = @SVector [ρᴸᴿᵢ, uᴸᴿᵢ, vᴸᴿᵢ, pᴸᴿᵢ]
#         end
#     end
# end

# function recon_packed(W, W_edge_i, W_edge_j, limiter::F, looplimits) where {F}
#     ilo, ihi, jlo, jhi = looplimits

#     @batch for j in jlo:jhi
#         for i in ilo:ihi
#             @simd for q in 1:4
#                 ϕᵢ₋₁ = W[q, i-1, j]
#                 ϕᵢ =   W[q, i, j]
#                 ϕᵢ₊₁ = W[q, i+1, j]
#                 ϕᵢ₊₂ = W[q, i+2, j]
#                 L, R = muscl(ϕᵢ₋₁, ϕᵢ, ϕᵢ₊₁, ϕᵢ₊₂, limiter)
#                 W_edge_i[1, q, i, j] = L
#                 W_edge_i[2, q, i, j] = R
#             end
#         end
#     end
# end

# function recon_packed_nosave(W, W_edge_i, W_edge_j, limiter::F, looplimits) where {F}
#     ilo, ihi, jlo, jhi = looplimits

#     @batch for j in jlo:jhi
#         for i in ilo:ihi
#             @simd for q in 1:4
#                 ϕᵢ₋₁ = W[q, i-1, j]
#                 ϕᵢ =   W[q, i, j]
#                 ϕᵢ₊₁ = W[q, i+1, j]
#                 ϕᵢ₊₂ = W[q, i+2, j]
#                 L, R = muscl(ϕᵢ₋₁, ϕᵢ, ϕᵢ₊₁, ϕᵢ₊₂, limiter)
#                 # W_edge_i[1, q, i, j] = L
#                 # W_edge_i[2, q, i, j] = R
#             end
#         end
#     end
# end

# function recon_packed_skip(W, W_edge_i, W_edge_j, limiter::F, looplimits) where {F}
#     ilo, ihi, jlo, jhi = looplimits

#     @batch for j in jlo:jhi
#         for i in ilo:2:ihi
#             @simd for q in 1:4
#                 ϕᵢ₋₂ = W[q, i-2, j]
#                 ϕᵢ₋₁ = W[q, i-1, j]
#                 ϕᵢ =   W[q, i, j]
#                 ϕᵢ₊₁ = W[q, i+1, j]
#                 ϕᵢ₊₂ = W[q, i+2, j]
#                 L, R = muscl(ϕᵢ₋₁, ϕᵢ, ϕᵢ₊₁, ϕᵢ₊₂, limiter)
#                 L2, R2 = muscl(ϕᵢ₋₁, ϕᵢ, ϕᵢ₊₁, ϕᵢ₊₂, limiter)
#                 W_edge_i[1, q, i, j] = L
#                 W_edge_i[2, q, i, j] = R
#                 W_edge_i[1, q, i+1, j] = L2
#                 W_edge_i[2, q, i+1, j] = R2
#             end
#         end
#     end
# end

# recon_hybrid(Whybrid, Wedge_i, nothing, minmod, looplims)
# recon_packed(Wpacked, W_edge_i_packed, W_edge_j_packed, minmod, looplims)
# recon_packed_skip(Wpacked, W_edge_i_packed, W_edge_j_packed, minmod, looplims)
# # @code_warntype recon_hybrid(Whybrid, Wedge_i, nothing, minmod, looplims)

# @benchmark recon_hybrid($Whybrid, $Wedge_i, $Wedge_j, $minmod, $looplims)
# @benchmark recon_packed($Wpacked, $W_edge_i_packed, $W_edge_j_packed, $minmod, $looplims)
# @benchmark recon_packed_nosave($Wpacked, $W_edge_i_packed, $W_edge_j_packed, $minmod, $looplims)
# @benchmark recon_packed_skip($Wpacked, $W_edge_i_packed, $W_edge_j_packed, $minmod, $looplims)

@inline function pressure_split_minus_nobranch(M, α=0.0)
  P⁻_a = 0.5(1 - sign(M))
  P⁻_b = 0.25(M - 1)^2 * (2 + M) - α * M * (M^2 - 1)^2

  supersonic = abs(M) > 1
  P⁻ = supersonic * P⁻_a + !supersonic * P⁻_b
  return P⁻
end

@inline function pressure_split_minus(M, α=0.0)
  if abs(M) > 1
    P⁻ = 0.5(1 - sign(M))
  else #  |M| <= 1
    P⁻ = 0.25(M - 1)^2 * (2 + M) - α * M * (M^2 - 1)^2
  end

  return P⁻
end

M = 0.2
Mall = @SVector rand(4)
@benchmark pressure_split_minus_nobranch($M)
@benchmark pressure_split_minus($M)

pressure_split_minus_nobranch(M)
pressure_split_minus(M)

pressure_split_minus_nobranch.(Mall)
pressure_split_minus.(Mall)

@benchmark pressure_split_minus_nobranch.($Mall)
@benchmark pressure_split_minus.($Mall)

args10 = abs.(rand(10))

@inline function modified_pressure_split_nobranch(
  Mʟ, Mʀ, Mstarᵢ, Mstarᵢ₊₁, ρᵢ, uᵢ, pᵢ, ρᵢ₊₁, uᵢ₊₁, pᵢ₊₁
)
  Mach_criteria = Mstarᵢ > 1 && Mstarᵢ₊₁ < 1 && 0 < Mstarᵢ * Mstarᵢ₊₁ < 1
  P⁻ᵢ₊₁_a = max(0, min(0.5, 1 - ((ρᵢ * uᵢ * (uᵢ - uᵢ₊₁) + pᵢ) / pᵢ₊₁)))
  P⁻ᵢ₊₁_b = pressure_split_minus(Mʀ)

  P⁻ᵢ₊₁ = Mach_criteria * P⁻ᵢ₊₁_a + !Mach_criteria * P⁻ᵢ₊₁_b
  return P⁻ᵢ₊₁
end

@inline function modified_pressure_split(
  Mʟ, Mʀ, Mstarᵢ, Mstarᵢ₊₁, ρᵢ, uᵢ, pᵢ, ρᵢ₊₁, uᵢ₊₁, pᵢ₊₁
)
  if Mstarᵢ > 1 && Mstarᵢ₊₁ < 1 && 0 < Mstarᵢ * Mstarᵢ₊₁ < 1
    P⁻ᵢ₊₁ = max(0, min(0.5, 1 - ((ρᵢ * uᵢ * (uᵢ - uᵢ₊₁) + pᵢ) / pᵢ₊₁)))
  else
    P⁻ᵢ₊₁ = pressure_split_minus(Mʀ)
  end

  return P⁻ᵢ₊₁
end

@inline function modified_pressure_split2(
  Mʟ, Mʀ, Mstarᵢ, Mstarᵢ₊₁, ρᵢ, uᵢ, pᵢ, ρᵢ₊₁, uᵢ₊₁, pᵢ₊₁
)
  if Mstarᵢ > 1 && Mstarᵢ₊₁ < 1 && 0 < Mstarᵢ * Mstarᵢ₊₁ < 1
    P⁻ᵢ₊₁ = max(0, min(0.5, 1 - ((ρᵢ * uᵢ * (uᵢ - uᵢ₊₁) + pᵢ) / pᵢ₊₁)))
  else
    P⁻ᵢ₊₁ = pressure_split_minus_nobranch(Mʀ)
  end

  return P⁻ᵢ₊₁
end

begin
  Mʟ = abs(rand())
  Mʀ = abs(rand())
  Mstarᵢ = abs(rand())
  Mstarᵢ₊₁ = abs(rand())
  ρᵢ = abs(rand())
  uᵢ = abs(rand())
  pᵢ = abs(rand())
  ρᵢ₊₁ = abs(rand())
  uᵢ₊₁ = abs(rand())
  pᵢ₊₁ = abs(rand())

  Mʟ_vec = @SVector (rand(4))
  Mʀ_vec = @SVector (rand(4))
  Mstarᵢ_vec = @SVector (rand(4))
  Mstarᵢ₊₁_vec = @SVector (rand(4))
  ρᵢ_vec = @SVector (rand(4))
  uᵢ_vec = @SVector (rand(4))
  pᵢ_vec = @SVector (rand(4))
  ρᵢ₊₁_vec = @SVector (rand(4))
  uᵢ₊₁_vec = @SVector (rand(4))
  pᵢ₊₁_vec = @SVector (rand(4))
end

modified_pressure_split(Mʟ, Mʀ, Mstarᵢ, Mstarᵢ₊₁, ρᵢ, uᵢ, pᵢ, ρᵢ₊₁, uᵢ₊₁, pᵢ₊₁)
modified_pressure_split_nobranch(Mʟ, Mʀ, Mstarᵢ, Mstarᵢ₊₁, ρᵢ, uᵢ, pᵢ, ρᵢ₊₁, uᵢ₊₁, pᵢ₊₁)

@benchmark modified_pressure_split(
  $Mʟ, $Mʀ, $Mstarᵢ, $Mstarᵢ₊₁, $ρᵢ, $uᵢ, $pᵢ, $ρᵢ₊₁, $uᵢ₊₁, $pᵢ₊₁
)
@benchmark modified_pressure_split_nobranch(
  $Mʟ, $Mʀ, $Mstarᵢ, $Mstarᵢ₊₁, $ρᵢ, $uᵢ, $pᵢ, $ρᵢ₊₁, $uᵢ₊₁, $pᵢ₊₁
)

modified_pressure_split.(
  Mʟ_vec,
  Mʀ_vec,
  Mstarᵢ_vec,
  Mstarᵢ₊₁_vec,
  ρᵢ_vec,
  uᵢ_vec,
  pᵢ_vec,
  ρᵢ₊₁_vec,
  uᵢ₊₁_vec,
  pᵢ₊₁_vec,
)
modified_pressure_split_nobranch.(
  Mʟ_vec,
  Mʀ_vec,
  Mstarᵢ_vec,
  Mstarᵢ₊₁_vec,
  ρᵢ_vec,
  uᵢ_vec,
  pᵢ_vec,
  ρᵢ₊₁_vec,
  uᵢ₊₁_vec,
  pᵢ₊₁_vec,
)

@benchmark modified_pressure_split.(
  $Mʟ_vec,
  $Mʀ_vec,
  $Mstarᵢ_vec,
  $Mstarᵢ₊₁_vec,
  $ρᵢ_vec,
  $uᵢ_vec,
  $pᵢ_vec,
  $ρᵢ₊₁_vec,
  $uᵢ₊₁_vec,
  $pᵢ₊₁_vec,
)
@benchmark modified_pressure_split2.(
  $Mʟ_vec,
  $Mʀ_vec,
  $Mstarᵢ_vec,
  $Mstarᵢ₊₁_vec,
  $ρᵢ_vec,
  $uᵢ_vec,
  $pᵢ_vec,
  $ρᵢ₊₁_vec,
  $uᵢ₊₁_vec,
  $pᵢ₊₁_vec,
)
@benchmark modified_pressure_split_nobranch.(
  $Mʟ_vec,
  $Mʀ_vec,
  $Mstarᵢ_vec,
  $Mstarᵢ₊₁_vec,
  $ρᵢ_vec,
  $uᵢ_vec,
  $pᵢ_vec,
  $ρᵢ₊₁_vec,
  $uᵢ₊₁_vec,
  $pᵢ₊₁_vec,
)
