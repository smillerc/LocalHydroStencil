using KernelAbstractions, Adapt, OffsetArrays
using OffsetArrays

const BACKEND = :CUDA

if BACKEND == :CUDA
    using CUDA, CUDAKernels
    const ArrayT = CuArray
    const Device = CUDADevice
elseif BACKEND == :AMD
    using AMDGPU, ROCMKernels
    const ArrayT = CuArray
    const Device = CUDADevice
else
    BACKEND == :CPU
    const ArrayT = Array
    const Device = CPU
end

@kernel function saxpy!(z, α, x, y)
    I = @index(Global)
    @inbounds z[I] = α * x[I] + y[I]
end

ndrange = (128,)
workgroupsize = (16,)
blocks, workgroupsize, dynamic = KernelAbstractions.NDIteration.partition(
    ndrange, workgroupsize
)

kernel = saxpy!(Device())

x = adapt(ArrayT, rand(64, 32))
y = adapt(ArrayT, rand(64, 32))
z = similar(x)

kernel(z, 0.01, x, y; ndrange=size(z))

iterspace, dynamic = KernelAbstractions.partition(kernel, size(x), nothing)

KernelAbstractions.NDIteration.blocks(iterspace)
KernelAbstractions.NDIteration.workitems(iterspace)

# kernel = saxpy!(Device(), (16,), (1024, 32))

begin
    # 1. Allocate data
    x = adapt(ArrayT, rand(64, 32))
    y = adapt(ArrayT, rand(64, 32))
    z = similar(x)

    # Note: CUDA.jl uses asynchronous allocations, and we are moving data from host
    #       to the device.

    allocation_event = Event(Device())

    # 2. Kernel event, kernel needs to synchronize against allocation and data
    #    movement from above.

    kernel_event = kernel(z, 0.01, x, y; ndrange=size(z), dependencies=allocation_event)

    # 3.
    # Scenario A: reading `z` from the host
    wait(kernel_event)
    adapt(Array, z)

    # Scenario B: Using `z` in the next kernel
    kernel_event = kernel(x, 0.01, z, y; ndrange=size(z), dependencies=kernel_event)

    # Note: We need to wait on `x` now

    # Scenario C: Using `z` as part of GPUArrays
    wait(Device(), kernel_event)
    zz = z .^ 2 # Broadcast expression is dependent on `z`
    nothing
end

@kernel function diffusion!(data, a, dt, dx, dy)
    i, j = @index(Global, NTuple)

    @inbounds begin
        dij = data[i, j]
        dim1j = data[i - 1, j]
        dijm1 = data[i, j - 1]
        dip1j = data[i + 1, j]
        dijp1 = data[i, j + 1]

        dij +=
            a * dt * ((dim1j - 2 * dij + dip1j) / dx^2 + (dijm1 - 2 * dij + dijp1) / dy^2)

        data[i, j] = dij
    end
end

N = 20
dx = 0.01 # x-grid spacing
dy = 0.01 # y-grid spacing
a = 0.001
dt = dx^2 * dy^2 / (2.0 * a * (dx^2 + dy^2)) # Largest stable time step

domain = OffsetArray(zeros(N + 2, N + 2), 0:(N + 1), 0:(N + 1))
domain[5:10, 5:10] .= 5
domain = adapt(ArrayT, domain)

diffusion_kernel = diffusion!(Device())

wait(diffusion_kernel(domain, a, dt, dx, dy; ndrange=(N, N)))

@kernel function diffusion_lmem!(out, @Const(data), a, dt, dx, dy)
    i, j = @index(Global, NTuple)
    li, lj = @index(Local, NTuple)
    lmem = @localmem eltype(data) (@groupsize()[1] + 2, @groupsize()[2] + 2)
    @uniform ldata = OffsetArray(
        lmem, 
        0:(@groupsize()[1] + 1), 
        0:(@groupsize()[2] + 1)
    )

    # Load data from global to local buffer
    @inbounds begin
        ldata[li, lj] = data[i, j]
        if i == 1
            ldata[li - 1, lj] = data[i - 1, j]
        end
        if i == @groupsize()[1]
            ldata[li + 1, lj] = data[i + 1, j]
        end
        if j == 1
            ldata[li, lj - 1] = data[i, j - 1]
        end
        if j == @groupsize()[2]
            ldata[li, lj + 1] = data[i, j + 1]
        end
    end
    @synchronize()  

    @inbounds begin
        dij = ldata[li, lj]
        dim1j = ldata[li - 1, lj]
        dijm1 = ldata[li, lj - 1]
        dip1j = ldata[li + 1, lj]
        dijp1 = ldata[li, lj + 1]

        dij +=
            a * dt * ((dim1j - 2 * dij + dip1j) / dx^2 + (dijm1 - 2 * dij + dijp1) / dy^2)

        out[i, j] = dij
    end
end

out = similar(domain)

diffusion_lmem_kernel = diffusion_lmem!(Device(), (32, 32))

wait(diffusion_lmem_kernel(out, domain, a, dt, dx, dy; ndrange=(N, N)))
out
