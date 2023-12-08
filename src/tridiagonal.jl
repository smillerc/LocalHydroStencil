using LinearAlgebra
using CUDA
using Cthulhu
using Test
using .Threads
using BenchmarkTools
using Revise
using Polyester
using CUDA.CUSPARSE

function tridiag_cpu(M::Tridiagonal{T,<:Array}, rhs::Vector{T})::Vector{T} where T
    N = length(rhs)
    phi = similar(rhs)
    gamma = similar(rhs)

    beta = M.d[1]
    phi[1] = rhs[1] / beta

    for j=2:N
        gamma[j] = M.du[j-1] / beta
        beta = M.d[j]-M.dl[j-1]*gamma[j]
        if abs(beta) < 1.e-12
            break
        end
        phi[j] = (rhs[j]-M.dl[j-1]*phi[j-1])/beta
    end

    for j=1:N-1
        k = N-j
        phi[k] = phi[k]-gamma[k+1]*phi[k+1]
    end

    return phi
end

function cyclic_owens(d_a, d_b, d_c, d_d, d_x)
    thid = threadIdx().x
    blid = blockIdx().x

    numThreads = blockDim().x
    batchStride = numThreads * 2

    # iterations = floor(Int, CUDAnative.log2(Float32(N ÷ 2)))
    iterations = floor(Int, CUDA.log2(numThreads))
    T = eltype(d_a)
    N = batchStride


    @inbounds begin
        # load data into shared memory

        a = CUDA.CuDynamicSharedArray(T, (N,))
        b = CUDA.CuDynamicSharedArray(T, (N,), N*sizeof(T))
        c = CUDA.CuDynamicSharedArray(T, (N,), N*sizeof(T)*2)
        d = CUDA.CuDynamicSharedArray(T, (N,), N*sizeof(T)*3)
        x = CUDA.CuDynamicSharedArray(T, (N,), N*sizeof(T)*4)

        a[thid] = d_a[thid + (blid-1) * batchStride]
        a[thid + blockDim().x] = d_a[thid + blockDim().x + (blid-1) * batchStride]

        b[thid] = d_b[thid + (blid-1) * batchStride]
        b[thid + blockDim().x] = d_b[thid + blockDim().x + (blid-1) * batchStride]

        c[thid] = d_c[thid + (blid-1) * batchStride]
        c[thid + blockDim().x] = d_c[thid + blockDim().x + (blid-1) * batchStride]

        d[thid] = d_d[thid + (blid-1) * batchStride]
        d[thid + blockDim().x] = d_d[thid + blockDim().x + (blid-1) * batchStride]

        sync_threads()

        # forward elimination
        stride = 1
        for j = 1:iterations
            sync_threads()
            stride *= 2
            delta = stride ÷ 2

            if threadIdx().x <= numThreads
                i = stride * (threadIdx().x - 1) + stride
                iLeft = i - delta
                iRight = i + delta
                if iRight > N
                    iRight = N
                end
                tmp1 = a[i] / b[iLeft]
                tmp2 = c[i] / b[iRight]
                b[i] = b[i] - c[iLeft] * tmp1 - a[iRight] * tmp2
                d[i] = d[i] - d[iLeft] * tmp1 - d[iRight] * tmp2
                a[i] = -a[iLeft] * tmp1
                c[i] = -c[iRight] * tmp2
            end

            numThreads ÷= 2
        end

        if thid <= 2
            addr1 = stride;
            addr2 = 2 * stride;
            tmp3 = b[addr2]*b[addr1] - c[addr1]*a[addr2]
            x[addr1] = (b[addr2]*d[addr1]-c[addr1]*d[addr2])/tmp3
            x[addr2] = (d[addr2]*b[addr1]-d[addr1]*a[addr2])/tmp3
        end

        # backward substitution
        numThreads = 2
        for j = 1:iterations
            delta = stride ÷ 2
            sync_threads()
            if thid <= numThreads
                i = stride * (thid - 1) + stride ÷ 2
                if i == delta
                    x[i] = (d[i] - c[i]*x[i+delta]) / b[i]
                else
                    x[i] = (d[i] - a[i]*x[i-delta] - c[i]*x[i+delta]) / b[i]
                end
            end
            stride ÷= 2
            numThreads *= 2
        end

        sync_threads()

        # write back to global memory
        d_x[thid + (blid-1) * batchStride] = x[thid]
        d_x[thid + blockDim().x + (blid-1) * batchStride] = x[thid + blockDim().x]
    end

    return
end

function cyclic_red(d_a, d_b, d_c, d_d, d_x)
    th_id = threadIdx().x
    blk_size = blockDim().x
    blk_id = blockIdx().x
    batchstride = blk_size * 2
    arr_type = eltype(d_a)

     
    sh_a = CUDA.CuDynamicSharedArray(arr_type, (2*blk_size))
    sh_b = CUDA.CuDynamicSharedArray(arr_type, (2*blk_size))
    sh_c = CUDA.CuDynamicSharedArray(arr_type, (2*blk_size))
    sh_d = CUDA.CuDynamicSharedArray(arr_type, (2*blk_size))
    sh_x = CUDA.CuDynamicSharedArray(arr_type, (2*blk_size))

    # sh_a = CUDA.CuStaticSharedArray(arr_type, (2*blk_size))
    # sh_b = CUDA.CuStaticSharedArray(arr_type, (2*blk_size))
    # sh_c = CUDA.CuStaticSharedArray(arr_type, (2*blk_size))
    # sh_d = CUDA.CuStaticSharedArray(arr_type, (2*blk_size))
    # sh_x = CUDA.CuStaticSharedArray(arr_type, (2*blk_size))
    ## Loading the 3 diagonals into shared memory
    sh_a[th_id] = d_a[th_id + (blk_id-1) * batchstride]
    sh_a[th_id + blk_size] = d_a[th_id + (blk_id-1) * batchstride + blk_size]
    sh_b[th_id] = d_b[th_id + (blk_id-1) * batchstride]
    sh_b[th_id + blk_size] = d_b[th_id + (blk_id-1) * batchstride + blk_size]
    sh_c[th_id] = d_c[th_id + (blk_id-1) * batchstride]
    sh_c[th_id + blk_size] = d_c[th_id + (blk_id-1) * batchstride + blk_size]
    sh_d[th_id] = d_d[th_id + (blk_id-1) * batchstride]
    sh_d[th_id + blk_size] = d_d[th_id + (blk_id-1) * batchstride + blk_size]
    sync_threads()

    ## The forward reduction stage.
    stride = 1
    while stride <= ((2*blk_size) >> 2)
        index = 2*stride*th_id
        if(index <= (2*blk_size))
            iright = index + stride
            ileft = index - stride
            if iright > (2*blk_size)
                iright = (2*blk_size)
            end
            var1 = sh_a[index] / sh_b[ileft]
            var2 = sh_c[index] / sh_b[iright]
            sh_b[index] = sh_b[index] - sh_c[ileft] * var1 - sh_a[iright] * var2
            sh_d[index] = sh_d[index] - sh_d[ileft] * var1 - sh_d[iright] * var2
            sh_a[index] = -sh_a[ileft] * var1
            sh_c[index] = -sh_c[iright] * var2
        end
        stride = stride*2
    end
    sync_threads()

    ## Solving the 2 equations 2 unknowns.
    stride = fld(stride,2)
    if th_id == 1
        index = 2*stride*th_id
        var3 = sh_b[index*2]*sh_b[index] - sh_c[index]*sh_a[index*2]
        sh_x[index] = (sh_b[index*2]*sh_d[index]-sh_c[index]*sh_d[index*2])/var3
        sh_x[index*2] = (sh_d[index*2]*sh_b[index]-sh_d[index]*sh_a[index*2])/var3
    end

    # Backsubstitution
    while stride > 0
        index = 2 * stride * th_id
        if (index <= (2*blk_size))
            i = index - stride
            if (i == stride)
                sh_x[i] = (sh_d[i] - sh_c[i]*sh_x[i+stride])/ sh_b[i]
            else
                sh_x[i] = (sh_d[i] - sh_a[i]*sh_x[i-stride] - sh_c[i]*sh_x[i+stride]) / sh_b[i]
            end
        end
        stride = fld(stride,2)
    end
    sync_threads()

    d_x[th_id + (blk_id-1) * batchstride] = sh_x[th_id]
    d_x[th_id + (blk_id-1) * batchstride + blk_size] = sh_x[th_id + blk_size]

    return nothing
end

#function main()
    system = 512
    eq = 512
    T = Float64

    ## Lower Diagonals
    a = CUDA.rand(T, (eq, system))
    a[1, : ] .= 0
    ## Upper Diagonal
    c = CUDA.rand(T, (eq, system))
    c[eq, :] .= 0
    ## Main Diag
    b = CUDA.rand(T, (eq, system))
    ## RHS
    d = CUDA.rand(T, (eq, system))
    x = CUDA.zeros(T, (eq, system))

    flat_a = reshape(a, system*eq)
    flat_b = reshape(b, system*eq)
    all(!=(0), flat_b)
    flat_c = reshape(c, system*eq)
    flat_d = reshape(d, system*eq)
    flat_x = reshape(x, system*eq)

    sh_mem = 5*sizeof(eltype(flat_a))*eq
    # numThreads=fld(eq, 2)
    # iterations = floor(Int, CUDA.log2(numThreads))
    # @device_code_warntype interactive=true @cuda threads=fld(eq, 2) blocks=system shmem=sh_mem cyclic_red(flat_a, flat_b, flat_c, flat_d, flat_x)
    #@cuda threads=fld(eq, 2) blocks=system shmem=sh_mem cyclic_red(flat_a, flat_b, flat_c, flat_d, flat_x)
    CUDA.@sync @cuda threads=fld(eq, 2) blocks=system shmem=sh_mem cyclic_owens(flat_a, flat_b, flat_c, flat_d, flat_x)
    #CUDA.@sync CUDA.CUSPARSE.gtsvStridedBatch(flat_a, flat_b, flat_c, flat_d, system, eq)
    Array(flat_x)
    #Array(flat_x_cusp)

    x_result = reshape(Array(flat_x), eq, system)
    
    for i in 1:system
        x_cyred = Array(x_result[:, i])
        in_a = Array(a[:, i])
        in_b = Array(b[:, i])
        in_c = Array(c[:, i])
        in_d = Array(d[:, i])

        tri_mat = Tridiagonal(in_a[2:end], in_b, in_c[1:end-1])
        @test tri_mat * x_cyred ≈ in_d
        break
    end

    function cpu_solve(a, b , c, d)
        Threads.@threads for i in 1:512
            in_a = Array(a[:, i])
            in_b = Array(b[:, i])
            in_c = Array(c[:, i])
            in_d = Array(d[:, i])
            tri_mat = Tridiagonal(in_a[2:end], in_b, in_c[1:end-1])
            x_cpured = tridiag_cpu(tri_mat, in_d)
        end
        return
    end

    cpu_solve
    @benchmark cpu_solve($a, $b, $c, $d)

    #function solve_cpusystem(flat_a, flat_b)
    ## Benchmark numbers

    @benchmark CUDA.@sync @cuda threads=fld($eq, 2) blocks=$system shmem=$sh_mem cyclic_owens($flat_a, $flat_b, $flat_c, $flat_d, $flat_x)
    CUDA.@profile CUDA.@sync @cuda threads=fld(eq, 2) blocks=system shmem=sh_mem cyclic_owens(flat_a, flat_b, flat_c, flat_d, flat_x)
        
#end