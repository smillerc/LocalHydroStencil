function next_Δt(U::AbstractArray{T}, mesh, EOS) where {T}
    return _next_Δt(U, mesh, EOS, Val(nthreads()))
    # _next_Δt(U, mesh, EOS)
end

function _next_Δt(U::AbstractArray{T}, mesh, EOS) where {T}
    nhalo = mesh.nhalo
    ilohi = axes(U, 2)
    jlohi = axes(U, 3)
    ilo = first(ilohi) + nhalo
    jlo = first(jlohi) + nhalo
    ihi = last(ilohi) - nhalo
    jhi = last(jlohi) - nhalo

    Δt_min = Inf

    ΔS_face = mesh.facelen
    vol = mesh.volume
    norms = mesh.facenorms

    for j in jlo:jhi
        for i in ilo:ihi
            n̂ = view(norms, :, :, i, j)
            n1x = n̂[1, 1]
            n1y = n̂[2, 1]
            n2x = n̂[1, 2]
            n2y = n̂[2, 2]
            n3x = n̂[1, 3]
            n3y = n̂[2, 3]
            n4x = n̂[1, 4]
            n4y = n̂[2, 4]

            s1 = ΔS_face[1, i, j]
            s2 = ΔS_face[2, i, j]
            s3 = ΔS_face[3, i, j]
            s4 = ΔS_face[4, i, j]

            ρ = U[1, i, j]
            u = U[2, i, j] / ρ
            v = U[3, i, j] / ρ
            E = U[4, i, j] / ρ

            p = pressure(EOS, ρ, u, v, E)
            cs = sound_speed(EOS, ρ, p)

            ave_n_i = @SVector [0.5(n2x - n4x), 0.5(n2y - n4y)]
            ave_ds_i = 0.5(s2 + s4)

            v_i = abs(ave_n_i[1] * u + ave_n_i[2] * v)
            spec_radius_i = (v_i + cs) * ave_ds_i

            ave_n_j = @SVector [0.5(n3x - n1x), 0.5(n3y - n1y)]
            ave_ds_j = 0.5(s1 + s3)

            v_j = abs(ave_n_j[1] * u + ave_n_j[2] * v)
            spec_radius_j = (v_j + cs) * ave_ds_j

            Δt_tid = vol[i, j] / (spec_radius_i + spec_radius_j)
            Δt_min = min(Δt_tid, Δt_min)
        end
    end

    return Δt_min
end

function _next_Δt(U::AbstractArray{T}, mesh, EOS, ::Val{NT}) where {T,NT}
    nhalo = mesh.nhalo
    ilohi = axes(U, 2)
    jlohi = axes(U, 3)
    ilo = first(ilohi) + nhalo
    jlo = first(jlohi) + nhalo
    ihi = last(ilohi) - nhalo
    jhi = last(jlohi) - nhalo

    Δt_min = @MVector ones(NT)
    Δt_min .*= Inf

    ΔS_face = mesh.facelen
    vol = mesh.volume
    norms = mesh.facenorms

    @batch per = thread for j in jlo:jhi
        for i in ilo:ihi
            tid = threadid()

            n̂ = view(norms, :, :, i, j)
            n1x = n̂[1, 1]
            n1y = n̂[2, 1]
            n2x = n̂[1, 2]
            n2y = n̂[2, 2]
            n3x = n̂[1, 3]
            n3y = n̂[2, 3]
            n4x = n̂[1, 4]
            n4y = n̂[2, 4]

            s1 = ΔS_face[1, i, j]
            s2 = ΔS_face[2, i, j]
            s3 = ΔS_face[3, i, j]
            s4 = ΔS_face[4, i, j]

            ρ = U[1, i, j]
            u = U[2, i, j] / ρ
            v = U[3, i, j] / ρ
            E = U[4, i, j] / ρ

            p = pressure(EOS, ρ, u, v, E)
            cs = sound_speed(EOS, ρ, p)

            ave_n_i = @SVector [0.5(n2x - n4x), 0.5(n2y - n4y)]
            ave_ds_i = 0.5(s2 + s4)

            v_i = abs(ave_n_i[1] * u + ave_n_i[2] * v)
            spec_radius_i = (v_i + cs) * ave_ds_i

            ave_n_j = @SVector [0.5(n3x - n1x), 0.5(n3y - n1y)]
            ave_ds_j = 0.5(s1 + s3)

            v_j = abs(ave_n_j[1] * u + ave_n_j[2] * v)
            spec_radius_j = (v_j + cs) * ave_ds_j

            Δt_tid = vol[i, j] / (spec_radius_i + spec_radius_j)
            Δt_tid = isfinite(Δt_tid) * Δt_tid
            # if isnan(Δt_tid)
            #     @show ρ, u, v, p, cs
            #     @show spec_radius_i, spec_radius_j
            #     @show vol[i,j]
            #     @show (n1x, n1y) (n2x, n2y) (n3x, n3y) (n4x, n4y)
            #     @show s1, s2, s3, s4
            #     error("ouch!")
            # end
            Δt_min[tid] = min(Δt_tid, Δt_min[tid])
        end
    end

    return minimum(Δt_min)
end
