@testitem "Test Stencil Fluxing" begin
    include("common.jl")

    eos = IdealEOS(1.4)
    ρ0 = 1.0
    u0 = 0.0
    v0 = 0.0
    p0 = 1.0
    p1 = 2.0

    E0 = specific_total_energy.(Ref(eos), ρ0, u0, v0, p0)
    E1 = specific_total_energy.(Ref(eos), ρ0, u0, v0, p1)

    U = zeros(4, 5, 5)

    ρ = @view U[1, :, :]
    ρu = @view U[2, :, :]
    ρv = @view U[3, :, :]
    ρE = @view U[4, :, :]

    fill!(ρ, 1.0)
    ρE[:, 1:3] .= E0
    ρE[:, 4:5] .= E1

    S⃗ = @SVector zeros(4)

    n̂ = @SMatrix [
         0.0 1.0 0.0 -1.0
        -1.0 0.0 1.0  0.0
    ]

    ΔS = @SVector ones(4)
    Ω = 1.0
    x⃗_c = @SVector ones(2)

    Ublock = get_block(U, 3, 3)

    display(Ublock)
    stencil = Stencil9Point(Ublock, S⃗, n̂, ΔS, Ω, eos, x⃗_c)

    RS = M_AUSMPWPlus2D()
    ∂U∂t = RS.∂U∂t(stencil, muscl_sarr_turbo_split2, minmod, false)
    
    @show ∂U∂t   
    @test iszero(∂U∂t[1])
    @test iszero(∂U∂t[2])
    @test ∂U∂t[3] > 0
    @test ∂U∂t[4] > 0

end