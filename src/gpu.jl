
using OffsetArrays

function gpu_2halo2dstencil(U⃗::AbstractArray{T,N}, Idx) where {T,N}
	i,j = Idx

    U_local = getblock(U⃗,i,j)
    
	S⃗ = @SVector zeros(4)
	n̂ = @SMatrix [0.0;-1.0;;1.0;0.0;;0.0;1.0;;-1.0;0.0]
	ΔS = @SVector ones(4)
	Ω = 1.0
	γ = 5/3
	return Stencil9Point(U_local,S⃗,n̂,ΔS,Ω,γ)
end

@kernel function SSPRK3_gpu!(SS::SSPRK3Integrator,
    @Const(U⃗n::AbstractArray{T,N}),@Const(riemann_solver),@Const(mesh),@Const(EOS),@Const(dt)) where {T,N}
    
    i, j = @index(Global, NTuple)
	
    Nq = size(U⃗n, 1)
    nh = mesh.nhalo
    ilohi = axes(U⃗n, 2)
    jlohi = axes(U⃗n, 3)

    ilo = first(ilohi) + nh
    jlo = first(jlohi) + nh
    ihi = last(ilohi) - nh
    jhi = last(jlohi) - nh

    @inbounds begin
        if (ilo <= i <= ihi) && (jlo <= j <= jhi)
            U⁽ⁿ⁾ = getvec(U⃗n,i,j)
            stencil = _2halo2dstencil(U⃗n, (i,j), mesh, EOS, nh)
            ∂U⁽ⁿ⁾∂t = ∂U∂t(riemann_solver, stencil, muscl, minmod)
            U⁽¹⁾ = U⁽ⁿ⁾ + ∂U⁽ⁿ⁾∂t * dt
            for q in 1:Nq
                SS.U⃗1[q,i,j] = U⁽¹⁾[q]
            end
        end
    end

    @synchronize()

    @inbounds begin
        if (ilo <= i <= ihi) && (jlo <= j <= jhi)
            U⁽¹⁾ = getvec(SS.U⃗1,i,j)
            U⁽ⁿ⁾ = getvec(U⃗n,i,j)
            stencil = _2halo2dstencil(SS.U⃗1, (i,j), mesh, EOS, nh)
            ∂U⁽¹⁾∂t = ∂U∂t(riemann_solver, stencil, muscl, minmod)
            U⁽²⁾ = .75U⁽ⁿ⁾ + .25U⁽¹⁾ + ∂U⁽¹⁾∂t * .25dt
            for q in 1:Nq
                SS.U⃗2[q,i,j] = U⁽²⁾[q]
            end
        end
    end

    @synchronize()

    @inbounds begin
        if (ilo <= i <= ihi) && (jlo <= j <= jhi)
            U⁽²⁾ = getvec(SS.U⃗2, i, j)
            U⁽ⁿ⁾ = getvec(U⃗n, i, j)
            stencil = _2halo2dstencil(SS.U⃗2, (i,j), mesh, EOS, nh)
            ∂U⁽²⁾∂t = ∂U∂t(riemann_solver, stencil, muscl, minmod)
            U⁽ⁿ⁺¹⁾ = (1/3) * U⁽ⁿ⁾ + (2/3) * U⁽²⁾+ ∂U⁽²⁾∂t * (2/3) * dt
            for q in 1:Nq
                SS.U⃗3[q,i,j] = U⁽ⁿ⁺¹⁾[q]
            end
        end
    end

end


@kernel function SSPRK3_gpu_lmem!(SS::SSPRK3Integrator,
    @Const(U⃗n::AbstractArray{T,N}),@Const(riemann_solver),@Const(mesh),@Const(EOS::E),@Const(dt),::Val{2}) where {T,N,E}
    
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
    @uniform ldata = OffsetArray(lmem, 1:4, 0:(@groupsize()[1]+3), 0:(@groupsize()[2]+3))
	
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
                        ldata[q, li-offset, lj] = U⃗n[q, i-offset, j]
                    end
                end
            end

            if i == @groupsize()[1]
                for offset in 1:2
                    for q in 1:Nq
                        ldata[q, li+offset, lj] = U⃗n[q, i+offset, j]
                    end
                end
            end
                        
            if j == jlo
                for offset in 1:2
                    for q in 1:Nq
                        ldata[q, li, lj-offset] = U⃗n[q, i, j-offset]
                    end
                end
            end

            if j == @groupsize()[2]
                for offset in 1:2
                    for q in 1:Nq
                        ldata[q, li, lj+offset] = U⃗n[q, i, j+offset]
                    end
                end
            end
        end
    end
    @synchronize()

    @inbounds begin
        if (ilo <= i <= ihi) && (jlo <= j <= jhi)
            U⁽ⁿ⁾ = getvec(ldata,li,lj)
            n̂ = get_nhat(mesh, i, j)
            ΔS = get_ΔS(mesh, i, j)
            Ω = mesh.volume[i,j]
            S⃗ = @SVector zeros(4)
            U_local = SArray{Tuple{4,5,5},T,3,100}(view(ldata, :, li-2:li+2, lj-2:lj+2))

            stencil = Stencil9Point{T,BT,E}(U_local, S⃗, n̂, ΔS, Ω, EOS)

            ∂U⁽ⁿ⁾∂t = ∂U∂t(riemann_solver, stencil, muscl, minmod)
            # U⁽¹⁾ = U⁽ⁿ⁾ + ∂U⁽ⁿ⁾∂t * dt
            # for q in 1:Nq
            #     SS.U⃗1[q,i,j] = U⁽¹⁾[q]
            # end
        end
    end

    # @synchronize()

    # @inbounds begin
    #     if (ilo <= i <= ihi) && (jlo <= j <= jhi)
    #         U⁽¹⁾ = getvec(SS.U⃗1,i,j)
    #         U⁽ⁿ⁾ = getvec(U⃗n,i,j)
    #         stencil = _2halo2dstencil(SS.U⃗1, (i,j), mesh, EOS, nh)
    #         ∂U⁽¹⁾∂t = ∂U∂t(riemann_solver, stencil, muscl, minmod)
    #         U⁽²⁾ = .75U⁽ⁿ⁾ + .25U⁽¹⁾ + ∂U⁽¹⁾∂t * .25dt
    #         for q in 1:Nq
    #             SS.U⃗2[q,i,j] = U⁽²⁾[q]
    #         end
    #     end
    # end

    # @synchronize()

    # @inbounds begin
    #     if (ilo <= i <= ihi) && (jlo <= j <= jhi)
    #         U⁽²⁾ = getvec(SS.U⃗2, i, j)
    #         U⁽ⁿ⁾ = getvec(U⃗n, i, j)
    #         stencil = _2halo2dstencil(SS.U⃗2, (i,j), mesh, EOS, nh)
    #         ∂U⁽²⁾∂t = ∂U∂t(riemann_solver, stencil, muscl, minmod)
    #         U⁽ⁿ⁺¹⁾ = (1/3) * U⁽ⁿ⁾ + (2/3) * U⁽²⁾+ ∂U⁽²⁾∂t * (2/3) * dt
    #         for q in 1:Nq
    #             SS.U⃗3[q,i,j] = U⁽ⁿ⁺¹⁾[q]
    #         end
    #     end
    # end

end