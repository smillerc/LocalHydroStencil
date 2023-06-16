module MeshType

using Adapt, StaticArrays

export CartesianMesh

struct CartesianMesh{AA1,AA2,AA3,AA4,AA5}
    xy::AA1
    centroid::AA2
    volume::AA3
    facenorms::AA4
    facelen::AA5
    nhalo::Int
end

function CartesianMesh(x::AbstractVector{T}, y::AbstractVector{T}, nhalo) where {T}
    M = length(x) #+ 2nhalo
    N = length(y) #+ 2nhalo

    xy = zeros(2, M, N)
    for j in 1:N
        for i in 1:M
            xy[1, i, j] = x[i]
            xy[2, i, j] = y[j]
        end
    end

    centroid = quad_centroids(xy)
    volume = quad_volumes(xy)

    facelen, facenorms = quad_face_areas_and_vecs(xy)

    nhat = [
        SMatrix{2,4}(view(facenorms, :, :, i, j)) for j in axes(facenorms, 4),
        i in axes(facenorms, 3)
    ]

    dS = [
        SVector{4}(view(facelen, :, i, j)) for j in axes(facelen, 3), i in axes(facelen, 2)
    ]

    return CartesianMesh(xy, centroid, volume, facenorms, facelen, nhalo)
    # return CartesianMesh(xy, centroid, volume, nhat, dS, nhalo)
end

function Adapt.adapt_structure(to, mesh::CartesianMesh)
    _xy = Adapt.adapt_structure(to, mesh.xy)
    _centroid = Adapt.adapt_structure(to, mesh.centroid)
    _volume = Adapt.adapt_structure(to, mesh.volume)
    _facenorms = Adapt.adapt_structure(to, mesh.facenorms)
    _facelen = Adapt.adapt_structure(to, mesh.facelen)
    _nhalo = Adapt.adapt_structure(to, mesh.nhalo)

    return CartesianMesh(_xy, _centroid, _volume, _facenorms, _facelen, _nhalo)
end

function quad_volumes(xy::AbstractArray{T,3}) where {T}
    ni = size(xy, 2) - 1
    nj = size(xy, 3) - 1
    volumes = zeros(T, (ni, nj))

    ϵ = eps(T)

    for j in 1:nj
        for i in 1:ni
            x1, y1 = @views xy[:, i, j]
            x2, y2 = @views xy[:, i + 1, j]
            x3, y3 = @views xy[:, i + 1, j + 1]
            x4, y4 = @views xy[:, i, j + 1]

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
            centroids[1, i, j] =
                0.25(xy[1, i, j] + xy[1, i + 1, j] + xy[1, i + 1, j + 1] + xy[1, i, j + 1])
            centroids[2, i, j] =
                0.25(xy[2, i, j] + xy[2, i + 1, j] + xy[2, i + 1, j + 1] + xy[2, i, j + 1])
        end
    end
    return centroids
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
            x2, y2 = @views xy[:, i + 1, j]
            x3, y3 = @views xy[:, i + 1, j + 1]
            x4, y4 = @views xy[:, i, j + 1]

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

end
