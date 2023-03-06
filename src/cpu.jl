function next_Δt(U::AbstractArray{T}, mesh, EOS) where {T}
    _next_Δt(U, mesh, EOS, Val(nthreads()))
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
            
            n̂ = view(norms,:,:,i,j)
            n1x = n̂[1,1] 
            n1y = n̂[2,1] 
            n2x = n̂[1,2] 
            n2y = n̂[2,2] 
            n3x = n̂[1,3] 
            n3y = n̂[2,3] 
            n4x = n̂[1,4] 
            n4y = n̂[2,4]

            s1 = ΔS_face[1, i, j]
            s2 = ΔS_face[2, i, j]
            s3 = ΔS_face[3, i, j]
            s4 = ΔS_face[4, i, j]

            ρ = U[1,i,j]
            u = U[2,i,j] / ρ
            v = U[3,i,j] / ρ
            E = U[4,i,j] / ρ

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

    @batch per=thread for j in jlo:jhi
        for i in ilo:ihi
            
            tid = threadid()

            n̂ = view(norms,:,:,i,j)
            n1x = n̂[1,1] 
            n1y = n̂[2,1] 
            n2x = n̂[1,2] 
            n2y = n̂[2,2] 
            n3x = n̂[1,3] 
            n3y = n̂[2,3] 
            n4x = n̂[1,4] 
            n4y = n̂[2,4]

            s1 = ΔS_face[1, i, j]
            s2 = ΔS_face[2, i, j]
            s3 = ΔS_face[3, i, j]
            s4 = ΔS_face[4, i, j]

            ρ = U[1,i,j]
            u = U[2,i,j] / ρ
            v = U[3,i,j] / ρ
            E = U[4,i,j] / ρ

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


function quad_volumes(xy::AbstractArray{T,3}) where {T}

    ni = size(xy, 2) - 1
    nj = size(xy, 3) - 1
    volumes = zeros(T, (ni, nj))

    ϵ = eps(T)

    for j in 1:nj
        for i in 1:ni
            x1, y1 = @views xy[:, i, j]
            x2, y2 = @views xy[:, i+1, j]
            x3, y3 = @views xy[:, i+1, j+1]
            x4, y4 = @views xy[:, i, j+1]

            dx13 = x1 - x3
            dy24 = y2 - y4
            dx42 = x4 - x2
            dy13 = y1 - y3

            dx13 = (dx13) * (abs(dx13) >= ϵ)
            dy24 = (dy24) * (abs(dy24) >= ϵ)
            dx42 = (dx42) * (abs(dx42) >= ϵ)
            dy13 = (dy13) * (abs(dy13) >= ϵ)

            volumes[i, j] = 0.5 * (dx13 * dy24 + dx42 * dy13)
        end
    end

    return volumes
end

function quad_centroids(xy::AbstractArray{T,3}) where {T}
    ni = size(xy, 2) - 1
    nj = size(xy, 3) - 1
    centroids = similar(xy, (2, ni, nj))

    for j in 1:nj
        for i in 1:ni
            centroids[1, i, j] = 0.25(xy[1, i, j] + xy[1, i+1, j] + xy[1, i+1, j+1] + xy[1, i, j+1])
            centroids[2, i, j] = 0.25(xy[2, i, j] + xy[2, i+1, j] + xy[2, i+1, j+1] + xy[2, i, j+1])
        end
    end
    centroids
end

function quad_face_areas_and_vecs(xy::AbstractArray{T,3}) where {T}
    ni = size(xy, 2) - 1
    nj = size(xy, 3) - 1
    facelens = zeros(T, (4, ni, nj))
    normvecs = zeros(T, (2, 4, ni, nj))

    ϵ = eps(T)

    for j in 1:nj
        for i in 1:ni
            x1, y1 = @views xy[:, i, j]
            x2, y2 = @views xy[:, i+1, j]
            x3, y3 = @views xy[:, i+1, j+1]
            x4, y4 = @views xy[:, i, j+1]

            Δx21 = x2 - x1
            Δx32 = x3 - x2
            Δx43 = x4 - x3
            Δx14 = x1 - x4
            Δy32 = y3 - y2
            Δy43 = y4 - y3
            Δy14 = y1 - y4
            Δy21 = y2 - y1

            Δy21 = Δy21 * (abs(Δy21) >= ϵ)
            Δy32 = Δy32 * (abs(Δy32) >= ϵ)
            Δy43 = Δy43 * (abs(Δy43) >= ϵ)
            Δy14 = Δy14 * (abs(Δy14) >= ϵ)
            Δx21 = Δx21 * (abs(Δx21) >= ϵ)
            Δx32 = Δx32 * (abs(Δx32) >= ϵ)
            Δx43 = Δx43 * (abs(Δx43) >= ϵ)
            Δx14 = Δx14 * (abs(Δx14) >= ϵ)

            Δx12 = -Δx21
            Δx23 = -Δx32
            Δx34 = -Δx43
            Δx41 = -Δx14

            Δs1 = sqrt((Δx21)^2 + (Δy21)^2)
            Δs2 = sqrt((Δx32)^2 + (Δy32)^2)
            Δs3 = sqrt((Δx43)^2 + (Δy43)^2)
            Δs4 = sqrt((Δx14)^2 + (Δy14)^2)

            facelens[1, i, j] = Δs1
            facelens[2, i, j] = Δs2
            facelens[3, i, j] = Δs3
            facelens[4, i, j] = Δs4

            normvecs[1, 1, i, j] = Δy21 / Δs1
            normvecs[2, 1, i, j] = Δx12 / Δs1

            normvecs[1, 2, i, j] = Δy32 / Δs2
            normvecs[2, 2, i, j] = Δx23 / Δs2

            normvecs[1, 3, i, j] = Δy43 / Δs3
            normvecs[2, 3, i, j] = Δx34 / Δs3

            normvecs[1, 4, i, j] = Δy14 / Δs4
            normvecs[2, 4, i, j] = Δx41 / Δs4
        end
    end

    return facelens, normvecs
end