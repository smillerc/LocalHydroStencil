using CUDA
using BenchmarkTools
using Revise
using .Threads
CUDA.allowscalar(false)

next_dt_1d(v⃗, cs, dx) = 1 / (abs(v⃗) + cs * dx)

function next_dt_2d(v⃗x, v⃗y, cs, ξx, ξy)
  ξnorm = sqrt(ξx^2 + ξy^2)
  U = v⃗x * ξx + v⃗y * ξy
  return 1 / (abs(U) + cs * ξnorm)
end

function next_dt_3d(v⃗x, v⃗y, v⃗z, cs, ξx, ξy, ξz)
  ξnorm = sqrt(ξx^2 + ξy^2 + ξz^2)
  U = v⃗x * ξx + v⃗y * ξy + v⃗z * ξz
  return 1 / (abs(U) + cs * ξnorm)
end

function next_dt((v⃗x, v⃗y)::NTuple{2,AA}, cs, (ξx, ξy, ηx, ηy), CFL) where {AA}
  Δt_ξ = mapreduce(next_dt_2d, min, v⃗x, v⃗y, cs, ξx, ξy)
  Δt_η = mapreduce(next_dt_2d, min, v⃗x, v⃗y, cs, ηx, ηy)

  Δt = CFL * min(Δt_ξ, Δt_η)
  return Δt
end

# function next_dt_multith((v⃗x, v⃗y)::NTuple{2,AA}, cs, (ξx, ξy, ηx, ηy), CFL) where {AA}
#   Threads.@threads Δt_ξ = mapreduce(next_dt_2d, min, v⃗x, v⃗y, cs, ξx, ξy)
#   Threads.@threads Δt_η = mapreduce(next_dt_2d, min, v⃗x, v⃗y, cs, ηx, ηy)

#   Δt = CFL * min(Δt_ξ, Δt_η)
#   return Δt
# end

function next_dt((v⃗x, v⃗y, v⃗z), cs, (ξx, ξy, ξz, ηx, ηy, ηz, ζx, ζy, ζz), CFL)
  Δt_ξ = mapreduce(next_dt_3d, min, (v⃗x, v⃗y, v⃗z), cs, (ξx, ξy, ξz))
  Δt_η = mapreduce(next_dt_3d, min, (v⃗x, v⃗y, v⃗z), cs, (ηx, ηy, ηz))
  Δt_ζ = mapreduce(next_dt_3d, min, (v⃗x, v⃗y, v⃗z), cs, (ζx, ζy, ζz))

  Δt = CFL * min(Δt_ξ, Δt_η, Δt_ζ)
  return Δt
end

function reduce_dt_interleaved((v⃗x, v⃗y)::NTuple{2,AA}, cs, (ξx, ξy, ηx, ηy), CFL, glb_min_dt1, glb_min_dt2) where {AA}
  i = (blockIdx().y - 1) * (blockDim().y) + threadIdx().y
  j = (blockIdx().x - 1) * (blockDim().x) + threadIdx().x
  # i1 = i + blockDim().y
  # j1 = j + blockDim().x

  blk_size = blockDim().x * blockDim().y
  ## Thread's ID inside its block
  th_id = blockDim().x * (threadIdx().y - 1) + threadIdx().x
  ## Block's ID inside its grid
  blk_id = gridDim().x * (blockIdx().y - 1) + blockIdx().x

  arr_type = eltype(glb_min_dt1)
  #m, n = sizeof(glb_min_dt1)
  sh_dt1 = CUDA.CuDynamicSharedArray(arr_type, (blk_size))
  sh_dt2 = CUDA.CuDynamicSharedArray(arr_type, (blk_size))

  @inbounds begin
    sh_dt1[th_id] = next_dt_2d(v⃗x[i,j], v⃗y[i,j], cs[i,j], ξx[i,j], ξy[i,j])
    #sh_dt1[th_id+blk_size] = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ξx[i1,j1], ξy[i1,j1])
    sh_dt2[th_id] = next_dt_2d(v⃗x[i,j], v⃗y[i,j], cs[i,j], ηx[i,j], ηy[i,j])
    #sh_dt2[th_id+blk_size] = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ηx[i1,j1], ηy[i1,j1])
  end
  sync_threads()

  s = 1
  while s < (blk_size/2)
    if ((th_id - 1) % (2*s)) == 0
      @inbounds begin
        sh_dt1[th_id] = min(sh_dt1[th_id], sh_dt1[th_id + s])
        sh_dt2[th_id] = min(sh_dt2[th_id], sh_dt2[th_id + s])
      end
    end
    s = s*2
    sync_threads()
  end

  if th_id == 1
    @inbounds begin
      glb_min_dt1[blk_id] = sh_dt1[th_id]
      glb_min_dt2[blk_id] = sh_dt2[th_id]
    end
  end

  return 
end

function reduce_dt_nondiv((v⃗x, v⃗y)::NTuple{2,AA}, cs, (ξx, ξy, ηx, ηy), CFL, glb_min_dt1, glb_min_dt2) where {AA}
  i = (blockIdx().y - 1) * (blockDim().y) + threadIdx().y
  j = (blockIdx().x - 1) * (blockDim().x) + threadIdx().x
  # i1 = i + blockDim().y
  # j1 = j + blockDim().x

  blk_size = blockDim().x * blockDim().y
  ## Thread's ID inside its block
  th_id = blockDim().x * (threadIdx().y - 1) + threadIdx().x
  ## Block's ID inside its grid
  blk_id = gridDim().x * (blockIdx().y - 1) + blockIdx().x

  arr_type = eltype(glb_min_dt1)
  #m, n = sizeof(glb_min_dt1)
  sh_dt1 = CUDA.CuDynamicSharedArray(arr_type, (blk_size))
  sh_dt2 = CUDA.CuDynamicSharedArray(arr_type, (blk_size))

  @inbounds begin
    sh_dt1[th_id] = next_dt_2d(v⃗x[i,j], v⃗y[i,j], cs[i,j], ξx[i,j], ξy[i,j])
    #sh_dt1[th_id+blk_size] = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ξx[i1,j1], ξy[i1,j1])
    sh_dt2[th_id] = next_dt_2d(v⃗x[i,j], v⃗y[i,j], cs[i,j], ηx[i,j], ηy[i,j])
    #sh_dt2[th_id+blk_size] = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ηx[i1,j1], ηy[i1,j1])
  end
  sync_threads()

  s = 1
  while s < fld(blk_size,2)
    index = 2*s*(th_id - 1)
    if index < blk_size
      @inbounds begin
        sh_dt1[index+1] = min(sh_dt1[index+1], sh_dt1[(index+1) + s])
        sh_dt2[index+1] = min(sh_dt2[index+1], sh_dt2[(index+1) + s])
      end
    end
      s = s*2
      sync_threads()
  end

  if th_id == 1
    @inbounds begin
      glb_min_dt1[blk_id] = sh_dt1[th_id]
      glb_min_dt2[blk_id] = sh_dt2[th_id]
    end
  end

  return 
end

function reduce_dt_nondiv_stride((v⃗x, v⃗y)::NTuple{2,AA}, cs, (ξx, ξy, ηx, ηy), CFL, glb_min_dt1, glb_min_dt2) where {AA}
  i = (blockIdx().y - 1) * (blockDim().y) + threadIdx().y
  j = (blockIdx().x - 1) * (blockDim().x) + threadIdx().x
  i1 = i + (blockDim().y * gridDim().y)
  j1 = j + (blockDim().x * gridDim().x)

  blk_size = blockDim().x * blockDim().y
  ## Thread's ID inside its block
  th_id = blockDim().x * (threadIdx().y - 1) + threadIdx().x
  ## Block's ID inside its grid
  blk_id = gridDim().x * (blockIdx().y - 1) + blockIdx().x

  arr_type = eltype(glb_min_dt1)
  #m, n = sizeof(glb_min_dt1)
  sh_dt1 = CUDA.CuDynamicSharedArray(arr_type, (blk_size))
  sh_dt2 = CUDA.CuDynamicSharedArray(arr_type, (blk_size))

  @inbounds begin
    @inbounds begin
      dt1_1 = next_dt_2d(v⃗x[i,j], v⃗y[i,j], cs[i,j], ξx[i,j], ξy[i,j])
      dt1_2 = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ξx[i1,j1], ξy[i1,j1])
      dt1_3 = next_dt_2d(v⃗x[i1,j], v⃗y[i1,j], cs[i1,j], ξx[i1,j], ξy[i1,j])
      dt1_4 = next_dt_2d(v⃗x[i,j1], v⃗y[i,j1], cs[i,j1], ξx[i,j1], ξy[i,j1])
      dt2_1 = next_dt_2d(v⃗x[i,j], v⃗y[i,j], cs[i,j], ηx[i,j], ηy[i,j])
      dt2_2 = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ηx[i1,j1], ηy[i1,j1])
      dt2_3 = next_dt_2d(v⃗x[i1,j], v⃗y[i1,j], cs[i1,j], ηx[i1,j], ηy[i1,j])
      dt2_4 = next_dt_2d(v⃗x[i,j1], v⃗y[i,j1], cs[i,j1], ηx[i,j1], ηy[i,j1])
      sh_dt1[th_id] = min(dt1_1, dt1_2, dt1_3, dt1_4)
      #sh_dt1[th_id+blk_size] = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ξx[i1,j1], ξy[i1,j1])
      sh_dt2[th_id] = min(dt2_1, dt2_2, dt2_3, dt2_4)
      #sh_dt2[th_id+blk_size] = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ηx[i1,j1], ηy[i1,j1])
    end
  end
  sync_threads()

  s = 1
  while s < fld(blk_size,2)
    index = 2*s*(th_id - 1)
    if index < blk_size
      @inbounds begin
        sh_dt1[index+1] = min(sh_dt1[index+1], sh_dt1[(index+1) + s])
        sh_dt2[index+1] = min(sh_dt2[index+1], sh_dt2[(index+1) + s])
      end
    end
      s = s*2
      sync_threads()
  end

  if th_id == 1
    @inbounds begin
      glb_min_dt1[blk_id] = sh_dt1[th_id]
      glb_min_dt2[blk_id] = sh_dt2[th_id]
    end
  end

  return 
end

function reduce_dt_sequential((v⃗x, v⃗y)::NTuple{2,AA}, cs, (ξx, ξy, ηx, ηy), CFL, glb_min_dt1, glb_min_dt2) where {AA}
  i = (blockIdx().y - 1) * (blockDim().y) + threadIdx().y
  j = (blockIdx().x - 1) * (blockDim().x) + threadIdx().x
  # i1 = i + blockDim().y
  # j1 = j + blockDim().x

  blk_size = blockDim().x * blockDim().y
  ## Thread's ID inside its block
  th_id = blockDim().x * (threadIdx().y - 1) + threadIdx().x
  ## Block's ID inside its grid
  blk_id = gridDim().x * (blockIdx().y - 1) + blockIdx().x

  arr_type = eltype(glb_min_dt1)
  #m, n = sizeof(glb_min_dt1)
  sh_dt1 = CUDA.CuDynamicSharedArray(arr_type, (blk_size))
  sh_dt2 = CUDA.CuDynamicSharedArray(arr_type, (blk_size))

  @inbounds begin
    sh_dt1[th_id] = next_dt_2d(v⃗x[i,j], v⃗y[i,j], cs[i,j], ξx[i,j], ξy[i,j])
    #sh_dt1[th_id+blk_size] = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ξx[i1,j1], ξy[i1,j1])
    sh_dt2[th_id] = next_dt_2d(v⃗x[i,j], v⃗y[i,j], cs[i,j], ηx[i,j], ηy[i,j])
    #sh_dt2[th_id+blk_size] = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ηx[i1,j1], ηy[i1,j1])
  end
  sync_threads()

  s = fld(blk_size, 2)
  while s > 0
    if th_id <= s
      @inbounds begin
        sh_dt1[th_id] = min(sh_dt1[th_id], sh_dt1[th_id + s])
        sh_dt2[th_id] = min(sh_dt2[th_id], sh_dt2[th_id + s])
      end
    end
    s = fld(s,2)
    sync_threads()
  end

  if th_id == 1
    @inbounds begin
      glb_min_dt1[blk_id] = sh_dt1[th_id]
      glb_min_dt2[blk_id] = sh_dt2[th_id]
    end
  end

  return 
end

function reduce_dt_unroll((v⃗x, v⃗y)::NTuple{2,AA}, cs, (ξx, ξy, ηx, ηy), CFL, glb_min_dt1, glb_min_dt2) where {AA}
  i = (blockIdx().y - 1) * (blockDim().y) + threadIdx().y
  j = (blockIdx().x - 1) * (blockDim().x) + threadIdx().x
  # i1 = i + blockDim().y
  # j1 = j + blockDim().x

  blk_size = blockDim().x * blockDim().y
  ## Thread's ID inside its block
  th_id = blockDim().x * (threadIdx().y - 1) + threadIdx().x
  ## Block's ID inside its grid
  blk_id = gridDim().x * (blockIdx().y - 1) + blockIdx().x

  arr_type = eltype(glb_min_dt1)
  #m, n = sizeof(glb_min_dt1)
  sh_dt1 = CUDA.CuDynamicSharedArray(arr_type, (blk_size))
  sh_dt2 = CUDA.CuDynamicSharedArray(arr_type, (blk_size))

  @inbounds begin
    sh_dt1[th_id] = next_dt_2d(v⃗x[i,j], v⃗y[i,j], cs[i,j], ξx[i,j], ξy[i,j])
    #sh_dt1[th_id+blk_size] = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ξx[i1,j1], ξy[i1,j1])
    sh_dt2[th_id] = next_dt_2d(v⃗x[i,j], v⃗y[i,j], cs[i,j], ηx[i,j], ηy[i,j])
    #sh_dt2[th_id+blk_size] = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ηx[i1,j1], ηy[i1,j1])
  end
  sync_threads()

  s = fld(blk_size, 2)
  #s = blk_size
  while s > 32
    if (th_id - 1) < s
      @inbounds begin
        sh_dt1[th_id] = min(sh_dt1[th_id], sh_dt1[th_id + s])
        sh_dt2[th_id] = min(sh_dt2[th_id], sh_dt2[th_id + s])
      end
    end
    s = fld(s,2)
    sync_threads()
  end

  if(th_id <= 32)
    @inbounds begin
      sh_dt1[th_id] = min(sh_dt1[th_id], sh_dt1[th_id + 32])
      sh_dt2[th_id] = min(sh_dt2[th_id], sh_dt2[th_id + 32])
      sh_dt1[th_id] = min(sh_dt1[th_id], sh_dt1[th_id + 16])
      sh_dt2[th_id] = min(sh_dt2[th_id], sh_dt2[th_id + 16])
      sh_dt1[th_id] = min(sh_dt1[th_id], sh_dt1[th_id + 8])
      sh_dt2[th_id] = min(sh_dt2[th_id], sh_dt2[th_id + 8])
      sh_dt1[th_id] = min(sh_dt1[th_id], sh_dt1[th_id + 4])
      sh_dt2[th_id] = min(sh_dt2[th_id], sh_dt2[th_id + 4])
      sh_dt1[th_id] = min(sh_dt1[th_id], sh_dt1[th_id + 2])
      sh_dt2[th_id] = min(sh_dt2[th_id], sh_dt2[th_id + 2])
      sh_dt1[th_id] = min(sh_dt1[th_id], sh_dt1[th_id + 1])
      sh_dt2[th_id] = min(sh_dt2[th_id], sh_dt2[th_id + 1])
    end
  end

  if th_id == 1
    @inbounds begin
      glb_min_dt1[blk_id] = sh_dt1[th_id]
      glb_min_dt2[blk_id] = sh_dt2[th_id]
    end
  end

  return 
end


function reduce_dt_shufl((v⃗x, v⃗y)::NTuple{2,AA}, cs, (ξx, ξy, ηx, ηy), CFL, glb_min_dt1, glb_min_dt2) where {AA}
  i = (blockIdx().y - 1) * (blockDim().y) + threadIdx().y
  j = (blockIdx().x - 1) * (blockDim().x) + threadIdx().x
  # i1 = i + (blockDim().y * gridDim().y)
  # j1 = j + (blockDim().x * gridDim().x)
  # i1 = i + blockDim().y
  # j1 = j + blockDim().x
  blk_size = blockDim().x * blockDim().y
  #warpsize = CUDA.warpsize()
  warpsize = 32
  ## Thread's ID inside its block
  th_id = blockDim().x * (threadIdx().y - 1) + threadIdx().x
  lane_id = (th_id - 1) % warpsize
  lane_id = lane_id + 1
  #lane_id = CUDA.laneid()
  warp_id = fld((th_id - 1), warpsize)
  warp_id = warp_id + 1
  ## Block's ID inside its grid
  blk_id = gridDim().x * (blockIdx().y - 1) + blockIdx().x
  arr_type = eltype(glb_min_dt1)
  #m, n = sizeof(glb_min_dt1)
  m = 2048
  n = 2048
  size_sh = fld(blk_size, warpsize)
  stride_x = blockDim().x * gridDim().x
  stride_y = blockDim().y * gridDim().y
  
  #@inbounds begin 
  sh_dt1 = CUDA.CuDynamicSharedArray(arr_type, (size_sh))
  sh_dt2 = CUDA.CuDynamicSharedArray(arr_type, (size_sh))
  #end

  dt1 = next_dt_2d(v⃗x[i,j], v⃗y[i,j], cs[i,j], ξx[i,j], ξy[i,j])
  dt2 = next_dt_2d(v⃗x[i,j], v⃗y[i,j], cs[i,j], ηx[i,j], ηy[i,j])

  # sh_dt1[th_id] = next_dt_2d(v⃗x[i,j], v⃗y[i,j], cs[i,j], ξx[i,j], ξy[i,j])
  # #sh_dt1[th_id+blk_size] = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ξx[i1,j1], ξy[i1,j1])
  # sh_dt2[th_id] = next_dt_2d(v⃗x[i,j], v⃗y[i,j], cs[i,j], ηx[i,j], ηy[i,j])
  # #sh_dt2[th_id+blk_size] = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ηx[i1,j1], ηy[i1,j1])
  # sync_threads()

  ## Step 1: Warp level reductions. Each warp stores its value in shared memory.
  #offset = fld(warpsize, 2)
  offset = 16
  while offset > 0
    val1_offset = CUDA.shfl_down_sync(0xffffffff, dt1, offset)
    dt1 = min(dt1, val1_offset)
    val2_offset = CUDA.shfl_down_sync(0xffffffff, dt2, offset)
    dt2 = min(dt2, val2_offset)
    offset = fld(offset,2)
  end

  if lane_id == 1
    @inbounds begin
      sh_dt1[warp_id] = dt1
      sh_dt2[warp_id] = dt2
    end
  end
  sync_threads()

  ## Step2: Read the values from shared memory into warp no. 1 & and do a warp level reduction on that warp.
  if th_id <= (blk_size/warpsize)
    @inbounds begin
      dt1 = sh_dt1[lane_id]
      dt2 = sh_dt2[lane_id]
    end
  else
    dt1 = 0
    dt2 = 0
  end

  if warp_id == 1
    offset = 16
    while offset > 0
      val1_offset = CUDA.shfl_down_sync(0xffffffff, dt1, offset)
      dt1 = min(dt1, val1_offset)
      val2_offset = CUDA.shfl_down_sync(0xffffffff, dt2, offset)
      dt2 = min(dt2, val2_offset)
      offset = fld(offset,2)
    end
  end

  if th_id == 1
    @inbounds begin
      glb_min_dt1[blk_id] = dt1
      glb_min_dt2[blk_id] = dt2
    end
  end
  return 
end

function reduce_dt_shufl_stride((v⃗x, v⃗y)::NTuple{2,AA}, cs, (ξx, ξy, ηx, ηy), CFL, glb_min_dt1, glb_min_dt2) where {AA}
  i = (blockIdx().y - 1) * (blockDim().y) + threadIdx().y
  j = (blockIdx().x - 1) * (blockDim().x) + threadIdx().x
  i1 = i + (blockDim().y * gridDim().y)
  j1 = j + (blockDim().x * gridDim().x)
  # i1 = i + blockDim().y
  # j1 = j + blockDim().x
  blk_size = blockDim().x * blockDim().y
  #warpsize = CUDA.warpsize()
  warpsize = 32
  ## Thread's ID inside its block
  th_id = blockDim().x * (threadIdx().y - 1) + threadIdx().x
  lane_id = (th_id - 1) % warpsize
  lane_id = lane_id + 1
  #lane_id = CUDA.laneid()
  warp_id = fld((th_id - 1), warpsize)
  warp_id = warp_id + 1
  ## Block's ID inside its grid
  blk_id = gridDim().x * (blockIdx().y - 1) + blockIdx().x
  arr_type = eltype(glb_min_dt1)
  #m, n = sizeof(glb_min_dt1)
  m = 2048
  n = 2048
  size_sh = fld(blk_size, warpsize)
  stride_x = blockDim().x * gridDim().x
  stride_y = blockDim().y * gridDim().y
  
  #@inbounds begin 
  sh_dt1 = CUDA.CuDynamicSharedArray(arr_type, (size_sh))
  sh_dt2 = CUDA.CuDynamicSharedArray(arr_type, (size_sh))
  #end

  @inbounds begin
    dt1_1 = next_dt_2d(v⃗x[i,j], v⃗y[i,j], cs[i,j], ξx[i,j], ξy[i,j])
    dt1_2 = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ξx[i1,j1], ξy[i1,j1])
    dt1_3 = next_dt_2d(v⃗x[i1,j], v⃗y[i1,j], cs[i1,j], ξx[i1,j], ξy[i1,j])
    dt1_4 = next_dt_2d(v⃗x[i,j1], v⃗y[i,j1], cs[i,j1], ξx[i,j1], ξy[i,j1])
    dt2_1 = next_dt_2d(v⃗x[i,j], v⃗y[i,j], cs[i,j], ηx[i,j], ηy[i,j])
    dt2_2 = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ηx[i1,j1], ηy[i1,j1])
    dt2_3 = next_dt_2d(v⃗x[i1,j], v⃗y[i1,j], cs[i1,j], ηx[i1,j], ηy[i1,j])
    dt2_4 = next_dt_2d(v⃗x[i,j1], v⃗y[i,j1], cs[i,j1], ηx[i,j1], ηy[i,j1])
  end

  dt1 = min(dt1_1, dt1_2, dt1_3, dt1_4)
  #sh_dt1[th_id+blk_size] = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ξx[i1,j1], ξy[i1,j1])
  dt2 = min(dt2_1, dt2_2, dt2_3, dt2_4) #end
  # sh_dt1[th_id] = next_dt_2d(v⃗x[i,j], v⃗y[i,j], cs[i,j], ξx[i,j], ξy[i,j])
  # #sh_dt1[th_id+blk_size] = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ξx[i1,j1], ξy[i1,j1])
  # sh_dt2[th_id] = next_dt_2d(v⃗x[i,j], v⃗y[i,j], cs[i,j], ηx[i,j], ηy[i,j])
  # #sh_dt2[th_id+blk_size] = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ηx[i1,j1], ηy[i1,j1])
  # sync_threads()

  ## Step 1: Warp level reductions. Each warp stores its value in shared memory.
  #offset = fld(warpsize, 2)
  offset = 16
  #s = blk_size
  while offset > 0
    val1_offset = CUDA.shfl_down_sync(0xffffffff, dt1, offset)
    dt1 = min(dt1, val1_offset)
    val2_offset = CUDA.shfl_down_sync(0xffffffff, dt2, offset)
    dt2 = min(dt2, val2_offset)
    offset = fld(offset,2)
  end

  if lane_id == 1
    @inbounds begin
      sh_dt1[warp_id] = dt1
      sh_dt2[warp_id] = dt2
    end
  end
  sync_threads()

  ## Step2: Read the values from shared memory into warp no. 1 & and do a warp level reduction on that warp.
  if th_id <= (blk_size/warpsize)
    @inbounds begin
      dt1 = sh_dt1[lane_id]
      dt2 = sh_dt2[lane_id]
    end
  else
    dt1 = 0
    dt2 = 0
  end

  if warp_id == 1
    # @inbounds begin
    #   dt1 = sh_dt1[lane_id]
    #   dt2 = sh_dt2[lane_id]
    # end
    #offset = fld(blk_size/warpsize, 2)
    #offset = fld(warpsize, 2)
    offset = 16
    #s = blk_size
    while offset > 0
      val1_offset = CUDA.shfl_down_sync(0xffffffff, dt1, offset)
      dt1 = min(dt1, val1_offset)
      val2_offset = CUDA.shfl_down_sync(0xffffffff, dt2, offset)
      dt2 = min(dt2, val2_offset)
      offset = fld(offset,2)
    end
  end

  if th_id == 1
    @inbounds begin
      glb_min_dt1[blk_id] = dt1
      glb_min_dt2[blk_id] = dt2
    end
  end
  return 
end

## This is actually sequential + stride
function reduce_dt_stride((v⃗x, v⃗y)::NTuple{2,AA}, cs, (ξx, ξy, ηx, ηy), CFL, glb_min_dt1, glb_min_dt2) where {AA}
  #i = (blockIdx().y - 1) * (2*blockDim().y) + threadIdx().y
  #j = (blockIdx().x - 1) * (2*blockDim().x) + threadIdx().x
  i = (blockIdx().y - 1) * (blockDim().y) + threadIdx().y
  j = (blockIdx().x - 1) * (blockDim().x) + threadIdx().x
  i1 = i + (blockDim().y * gridDim().y)
  j1 = j + (blockDim().x * gridDim().x)

  blk_size = blockDim().x * blockDim().y
  th_id = blockDim().x * (threadIdx().y - 1) + threadIdx().x
  blk_id = gridDim().x * (blockIdx().y - 1) + blockIdx().x

  arr_type = eltype(glb_min_dt1)
  #sh_dt1 = CUDA.CuDynamicSharedArray(arr_type, (2*blk_size))
  #sh_dt2 = CUDA.CuDynamicSharedArray(#arr_type, (2*blk_size))
  sh_dt1 = CUDA.CuDynamicSharedArray(arr_type, (blk_size))
  sh_dt2 = CUDA.CuDynamicSharedArray(arr_type, (blk_size))

  @inbounds begin
    dt1_1 = next_dt_2d(v⃗x[i,j], v⃗y[i,j], cs[i,j], ξx[i,j], ξy[i,j])
    dt1_2 = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ξx[i1,j1], ξy[i1,j1])
    dt1_3 = next_dt_2d(v⃗x[i1,j], v⃗y[i1,j], cs[i1,j], ξx[i1,j], ξy[i1,j])
    dt1_4 = next_dt_2d(v⃗x[i,j1], v⃗y[i,j1], cs[i,j1], ξx[i,j1], ξy[i,j1])
    dt2_1 = next_dt_2d(v⃗x[i,j], v⃗y[i,j], cs[i,j], ηx[i,j], ηy[i,j])
    dt2_2 = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ηx[i1,j1], ηy[i1,j1])
    dt2_3 = next_dt_2d(v⃗x[i1,j], v⃗y[i1,j], cs[i1,j], ηx[i1,j], ηy[i1,j])
    dt2_4 = next_dt_2d(v⃗x[i,j1], v⃗y[i,j1], cs[i,j1], ηx[i,j1], ηy[i,j1])
    sh_dt1[th_id] = min(dt1_1, dt1_2, dt1_3, dt1_4)
    #sh_dt1[th_id+blk_size] = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ξx[i1,j1], ξy[i1,j1])
    sh_dt2[th_id] = min(dt2_1, dt2_2, dt2_3, dt2_4)
    #sh_dt2[th_id+blk_size] = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ηx[i1,j1], ηy[i1,j1])
  end
  sync_threads()

  s = fld(blk_size, 2)
  #s = blk_size
  while s > 0
    if (th_id - 1) < s
      @inbounds begin
        sh_dt1[th_id] = min(sh_dt1[th_id], sh_dt1[th_id + s])
        sh_dt2[th_id] = min(sh_dt2[th_id], sh_dt2[th_id + s])
      end
    end
    s = fld(s,2)
    sync_threads()
  end

  if th_id == 1
    @inbounds begin
      glb_min_dt1[blk_id] = sh_dt1[th_id]
      glb_min_dt2[blk_id] = sh_dt2[th_id]
    end
  end

  return 
end

function reduce_dt_stride_unroll((v⃗x, v⃗y)::NTuple{2,AA}, cs, (ξx, ξy, ηx, ηy), CFL, glb_min_dt1, glb_min_dt2) where {AA}
  #i = (blockIdx().y - 1) * (2*blockDim().y) + threadIdx().y
  #j = (blockIdx().x - 1) * (2*blockDim().x) + threadIdx().x
  i = (blockIdx().y - 1) * (blockDim().y) + threadIdx().y
  j = (blockIdx().x - 1) * (blockDim().x) + threadIdx().x
  i1 = i + (blockDim().y * gridDim().y)
  j1 = j + (blockDim().x * gridDim().x)

  blk_size = blockDim().x * blockDim().y
  th_id = blockDim().x * (threadIdx().y - 1) + threadIdx().x
  blk_id = gridDim().x * (blockIdx().y - 1) + blockIdx().x

  arr_type = eltype(glb_min_dt1)
  #sh_dt1 = CUDA.CuDynamicSharedArray(arr_type, (2*blk_size))
  #sh_dt2 = CUDA.CuDynamicSharedArray(#arr_type, (2*blk_size))
  sh_dt1 = CUDA.CuDynamicSharedArray(arr_type, (blk_size))
  sh_dt2 = CUDA.CuDynamicSharedArray(arr_type, (blk_size))

  @inbounds begin
    dt1_1 = next_dt_2d(v⃗x[i,j], v⃗y[i,j], cs[i,j], ξx[i,j], ξy[i,j])
    dt1_2 = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ξx[i1,j1], ξy[i1,j1])
    dt1_3 = next_dt_2d(v⃗x[i1,j], v⃗y[i1,j], cs[i1,j], ξx[i1,j], ξy[i1,j])
    dt1_4 = next_dt_2d(v⃗x[i,j1], v⃗y[i,j1], cs[i,j1], ξx[i,j1], ξy[i,j1])
    dt2_1 = next_dt_2d(v⃗x[i,j], v⃗y[i,j], cs[i,j], ηx[i,j], ηy[i,j])
    dt2_2 = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ηx[i1,j1], ηy[i1,j1])
    dt2_3 = next_dt_2d(v⃗x[i1,j], v⃗y[i1,j], cs[i1,j], ηx[i1,j], ηy[i1,j])
    dt2_4 = next_dt_2d(v⃗x[i,j1], v⃗y[i,j1], cs[i,j1], ηx[i,j1], ηy[i,j1])
    sh_dt1[th_id] = min(dt1_1, dt1_2, dt1_3, dt1_4)
    #sh_dt1[th_id+blk_size] = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ξx[i1,j1], ξy[i1,j1])
    sh_dt2[th_id] = min(dt2_1, dt2_2, dt2_3, dt2_4)
    #sh_dt2[th_id+blk_size] = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ηx[i1,j1], ηy[i1,j1])
  end
  sync_threads()

  s = fld(blk_size, 2)
  #s = blk_size
  while s > 32
    if (th_id - 1) < s
      @inbounds begin
        sh_dt1[th_id] = min(sh_dt1[th_id], sh_dt1[th_id + s])
        sh_dt2[th_id] = min(sh_dt2[th_id], sh_dt2[th_id + s])
      end
    end
    s = fld(s,2)
    sync_threads()
  end

  if(th_id <= 32)
    @inbounds begin
      sh_dt1[th_id] = min(sh_dt1[th_id], sh_dt1[th_id + 32])
      sh_dt2[th_id] = min(sh_dt2[th_id], sh_dt2[th_id + 32])
      sh_dt1[th_id] = min(sh_dt1[th_id], sh_dt1[th_id + 16])
      sh_dt2[th_id] = min(sh_dt2[th_id], sh_dt2[th_id + 16])
      sh_dt1[th_id] = min(sh_dt1[th_id], sh_dt1[th_id + 8])
      sh_dt2[th_id] = min(sh_dt2[th_id], sh_dt2[th_id + 8])
      sh_dt1[th_id] = min(sh_dt1[th_id], sh_dt1[th_id + 4])
      sh_dt2[th_id] = min(sh_dt2[th_id], sh_dt2[th_id + 4])
      sh_dt1[th_id] = min(sh_dt1[th_id], sh_dt1[th_id + 2])
      sh_dt2[th_id] = min(sh_dt2[th_id], sh_dt2[th_id + 2])
      sh_dt1[th_id] = min(sh_dt1[th_id], sh_dt1[th_id + 1])
      sh_dt2[th_id] = min(sh_dt2[th_id], sh_dt2[th_id + 1])
    end
  end

  if th_id == 1
    @inbounds begin
      glb_min_dt1[blk_id] = sh_dt1[th_id]
      glb_min_dt2[blk_id] = sh_dt2[th_id]
    end
  end

  return 
end

function reduce_dt_stride_unroll((v⃗x, v⃗y)::NTuple{2,AA}, cs, (ξx, ξy, ηx, ηy), CFL, glb_min_dt1, glb_min_dt2) where {AA}
  #i = (blockIdx().y - 1) * (2*blockDim().y) + threadIdx().y
  #j = (blockIdx().x - 1) * (2*blockDim().x) + threadIdx().x
  i = (blockIdx().y - 1) * (blockDim().y) + threadIdx().y
  j = (blockIdx().x - 1) * (blockDim().x) + threadIdx().x
  i1 = i + (blockDim().y * gridDim().y)
  j1 = j + (blockDim().x * gridDim().x)

  blk_size = blockDim().x * blockDim().y
  th_id = blockDim().x * (threadIdx().y - 1) + threadIdx().x
  blk_id = gridDim().x * (blockIdx().y - 1) + blockIdx().x

  arr_type = eltype(glb_min_dt1)
  #sh_dt1 = CUDA.CuDynamicSharedArray(arr_type, (2*blk_size))
  #sh_dt2 = CUDA.CuDynamicSharedArray(#arr_type, (2*blk_size))
  sh_dt1 = CUDA.CuDynamicSharedArray(arr_type, (blk_size))
  sh_dt2 = CUDA.CuDynamicSharedArray(arr_type, (blk_size))

  @inbounds begin
    dt1_1 = next_dt_2d(v⃗x[i,j], v⃗y[i,j], cs[i,j], ξx[i,j], ξy[i,j])
    dt1_2 = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ξx[i1,j1], ξy[i1,j1])
    dt1_3 = next_dt_2d(v⃗x[i1,j], v⃗y[i1,j], cs[i1,j], ξx[i1,j], ξy[i1,j])
    dt1_4 = next_dt_2d(v⃗x[i,j1], v⃗y[i,j1], cs[i,j1], ξx[i,j1], ξy[i,j1])
    dt2_1 = next_dt_2d(v⃗x[i,j], v⃗y[i,j], cs[i,j], ηx[i,j], ηy[i,j])
    dt2_2 = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ηx[i1,j1], ηy[i1,j1])
    dt2_3 = next_dt_2d(v⃗x[i1,j], v⃗y[i1,j], cs[i1,j], ηx[i1,j], ηy[i1,j])
    dt2_4 = next_dt_2d(v⃗x[i,j1], v⃗y[i,j1], cs[i,j1], ηx[i,j1], ηy[i,j1])
    sh_dt1[th_id] = min(dt1_1, dt1_2, dt1_3, dt1_4)
    #sh_dt1[th_id+blk_size] = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ξx[i1,j1], ξy[i1,j1])
    sh_dt2[th_id] = min(dt2_1, dt2_2, dt2_3, dt2_4)
    #sh_dt2[th_id+blk_size] = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ηx[i1,j1], ηy[i1,j1])
  end
  sync_threads()

  s = fld(blk_size, 2)
  #s = blk_size
  while s > 32
    if (th_id - 1) < s
      @inbounds begin
        sh_dt1[th_id] = min(sh_dt1[th_id], sh_dt1[th_id + s])
        sh_dt2[th_id] = min(sh_dt2[th_id], sh_dt2[th_id + s])
      end
    end
    s = fld(s,2)
    sync_threads()
  end

  if(th_id <= 32)
    @inbounds begin
      sh_dt1[th_id] = min(sh_dt1[th_id], sh_dt1[th_id + 32])
      sh_dt2[th_id] = min(sh_dt2[th_id], sh_dt2[th_id + 32])
      sh_dt1[th_id] = min(sh_dt1[th_id], sh_dt1[th_id + 16])
      sh_dt2[th_id] = min(sh_dt2[th_id], sh_dt2[th_id + 16])
      sh_dt1[th_id] = min(sh_dt1[th_id], sh_dt1[th_id + 8])
      sh_dt2[th_id] = min(sh_dt2[th_id], sh_dt2[th_id + 8])
      sh_dt1[th_id] = min(sh_dt1[th_id], sh_dt1[th_id + 4])
      sh_dt2[th_id] = min(sh_dt2[th_id], sh_dt2[th_id + 4])
      sh_dt1[th_id] = min(sh_dt1[th_id], sh_dt1[th_id + 2])
      sh_dt2[th_id] = min(sh_dt2[th_id], sh_dt2[th_id + 2])
      sh_dt1[th_id] = min(sh_dt1[th_id], sh_dt1[th_id + 1])
      sh_dt2[th_id] = min(sh_dt2[th_id], sh_dt2[th_id + 1])
    end
  end

  if th_id == 1
    @inbounds begin
      glb_min_dt1[blk_id] = sh_dt1[th_id]
      glb_min_dt2[blk_id] = sh_dt2[th_id]
    end
  end

  return 
end

function reduce_dt_shdouble((v⃗x, v⃗y)::NTuple{2,AA}, cs, (ξx, ξy, ηx, ηy), CFL, glb_min_dt1, glb_min_dt2) where {AA}
  #i = (blockIdx().y - 1) * (2*blockDim().y) + threadIdx().y
  #j = (blockIdx().x - 1) * (2*blockDim().x) + threadIdx().x
  i = (blockIdx().y - 1) * (blockDim().y) + threadIdx().y
  j = (blockIdx().x - 1) * (blockDim().x) + threadIdx().x
  i1 = i + (blockDim().y * gridDim().y)
  j1 = j + (blockDim().x * gridDim().x)

  blk_size = blockDim().x * blockDim().y
  th_id = blockDim().x * (threadIdx().y - 1) + threadIdx().x
  blk_id = gridDim().x * (blockIdx().y - 1) + blockIdx().x

  arr_type = eltype(glb_min_dt1)
  sh_dt1 = CUDA.CuDynamicSharedArray(arr_type, (2*blk_size))
  sh_dt2 = CUDA.CuDynamicSharedArray(arr_type, (2*blk_size))
  # sh_dt1 = CUDA.CuDynamicSharedArray(arr_type, (blk_size))
  # sh_dt2 = CUDA.CuDynamicSharedArray(arr_type, (blk_size))

  @inbounds begin
    dt1_1 = next_dt_2d(v⃗x[i,j], v⃗y[i,j], cs[i,j], ξx[i,j], ξy[i,j])
    dt1_2 = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ξx[i1,j1], ξy[i1,j1])
    dt1_3 = next_dt_2d(v⃗x[i1,j], v⃗y[i1,j], cs[i1,j], ξx[i1,j], ξy[i1,j])
    dt1_4 = next_dt_2d(v⃗x[i,j1], v⃗y[i,j1], cs[i,j1], ξx[i,j1], ξy[i,j1])
    dt2_1 = next_dt_2d(v⃗x[i,j], v⃗y[i,j], cs[i,j], ηx[i,j], ηy[i,j])
    dt2_2 = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ηx[i1,j1], ηy[i1,j1])
    dt2_3 = next_dt_2d(v⃗x[i1,j], v⃗y[i1,j], cs[i1,j], ηx[i1,j], ηy[i1,j])
    dt2_4 = next_dt_2d(v⃗x[i,j1], v⃗y[i,j1], cs[i,j1], ηx[i,j1], ηy[i,j1])

    sh_dt1[th_id] = min(dt1_1, dt1_2)
    sh_dt1[th_id+blk_size] = min(dt1_3,dt1_4)
    # sh_dt1[th_id+(2*blk_size)] = dt1_3
    # sh_dt1[th_id+(3*blk_size)] = dt1_4
    #sh_dt1[th_id+blk_size] = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ξx[i1,j1], ξy[i1,j1])
    sh_dt2[th_id] = min(dt2_1, dt2_2)
    sh_dt2[th_id+blk_size] = min(dt2_3, dt2_4)
    # sh_dt2[th_id+(2*blk_size)] = dt2_3
    # sh_dt2[th_id+(3*blk_size)] = dt2_4
    #sh_dt2[th_id+blk_size] = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ηx[i1,j1], ηy[i1,j1])
  end
  sync_threads()

  #s = fld(blk_size, 2)
  s = 2*blk_size
  while s > 0
    if (th_id - 1) < s
      @inbounds begin
        sh_dt1[th_id] = min(sh_dt1[th_id], sh_dt1[th_id + s], sh_dt1[th_id + 2*s], sh_dt1[th_id + 3*s])
        sh_dt2[th_id] = min(sh_dt2[th_id], sh_dt2[th_id + s], sh_dt2[th_id + 2*s], sh_dt2[th_id + 3*s])
      end
    end
    s = fld(s,2)
    sync_threads()
  end

  if th_id == 1
    @inbounds begin
      glb_min_dt1[blk_id] = sh_dt1[th_id]
      glb_min_dt2[blk_id] = sh_dt2[th_id]
    end
  end

  return 
end

function reduce_dt_stride_shdouble((v⃗x, v⃗y)::NTuple{2,AA}, cs, (ξx, ξy, ηx, ηy), CFL, glb_min_dt1, glb_min_dt2) where {AA}
  i = (blockIdx().y - 1) * (blockDim().y) + threadIdx().y
  j = (blockIdx().x - 1) * (2*blockDim().x) + threadIdx().x
  # i = (blockIdx().y - 1) * (blockDim().y) + threadIdx().y
  # j = (blockIdx().x - 1) * (blockDim().x) + threadIdx().x
  #i_1 = i + blockDim().y
  #j_1 = j + blockDim().x
  i1 = i + (blockDim().y * gridDim().y)
  j1 = j + (2*blockDim().x * gridDim().x)
  #i1_1 = i_1 + (blockDim().y * gridDim().y)
  #j1_1 = j_1 + (2*blockDim().x * gridDim().x)

  blk_size = blockDim().x * blockDim().y
  th_id = blockDim().x * (threadIdx().y - 1) + threadIdx().x
  blk_id = gridDim().x * (blockIdx().y - 1) + blockIdx().x

  arr_type = eltype(glb_min_dt1)
  sh_dt1 = CUDA.CuDynamicSharedArray(arr_type, (2*blk_size))
  sh_dt2 = CUDA.CuDynamicSharedArray(arr_type, (2*blk_size))
  # sh_dt1 = CUDA.CuDynamicSharedArray(arr_type, (blk_size))
  # sh_dt2 = CUDA.CuDynamicSharedArray(arr_type, (blk_size))

  @inbounds begin
    dt1_1 = next_dt_2d(v⃗x[i,j], v⃗y[i,j], cs[i,j], ξx[i,j], ξy[i,j])
    dt1a_1 = next_dt_2d(v⃗x[i,j+blockDim().x], v⃗y[i,j+blockDim().x], cs[i,j+blockDim().x], ξx[i,j+blockDim().x], ξy[i,j+blockDim().x])

    dt1_2 = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ξx[i1,j1], ξy[i1,j1])
    dt1a_2 = next_dt_2d(v⃗x[i1,j1+blockDim().x], v⃗y[i1,j1+blockDim().x], cs[i1,j1+blockDim().x], ξx[i1,j1+blockDim().x], ξy[i1,j1+blockDim().x])

    dt1_3 = next_dt_2d(v⃗x[i1,j], v⃗y[i1,j], cs[i1,j], ξx[i1,j], ξy[i1,j])
    dt1a_3 = next_dt_2d(v⃗x[i1,j+blockDim().x], v⃗y[i1,j+blockDim().x], cs[i1,j+blockDim().x], ξx[i1,j+blockDim().x], ξy[i1,j+blockDim().x])

    dt1_4 = next_dt_2d(v⃗x[i,j1], v⃗y[i,j1], cs[i,j1], ξx[i,j1], ξy[i,j1])
    dt1a_4 = next_dt_2d(v⃗x[i,j1+blockDim().x], v⃗y[i,j1+blockDim().x], cs[i,j1+blockDim().x], ξx[i,j1+blockDim().x], ξy[i,j1+blockDim().x])


    dt2_1 = next_dt_2d(v⃗x[i,j], v⃗y[i,j], cs[i,j], ηx[i,j], ηy[i,j])
    dt2a_1 = next_dt_2d(v⃗x[i,j+blockDim().x], v⃗y[i,j+blockDim().x], cs[i,j+blockDim().x], ηx[i,j+blockDim().x], ηy[i,j+blockDim().x])

    dt2_2 = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ηx[i1,j1], ηy[i1,j1])
    dt2a_2 = next_dt_2d(v⃗x[i1,j1+blockDim().x], v⃗y[i1,j1+blockDim().x], cs[i1,j1+blockDim().x], ηx[i1,j1+blockDim().x], ηy[i1,j1+blockDim().x])

    dt2_3 = next_dt_2d(v⃗x[i1,j], v⃗y[i1,j], cs[i1,j], ηx[i1,j], ηy[i1,j])
    dt2a_3 = next_dt_2d(v⃗x[i1,j+blockDim().x], v⃗y[i1,j+blockDim().x], cs[i1,j+blockDim().x], ηx[i1,j+blockDim().x], ηy[i1,j+blockDim().x])

    dt2_4 = next_dt_2d(v⃗x[i,j1], v⃗y[i,j1], cs[i,j1], ηx[i,j1], ηy[i,j1])
    dt2a_4 = next_dt_2d(v⃗x[i,j1+blockDim().x], v⃗y[i,j1+blockDim().x], cs[i,j1+blockDim().x], ηx[i,j1+blockDim().x], ηy[i,j1+blockDim().x])

    sh_dt1[th_id] = min(dt1_1, dt1_2, dt1_3, dt1_4)
    sh_dt1[th_id+blk_size] = min(dt1a_1, dt1a_2, dt1a_3, dt1a_4)
    sh_dt2[th_id] = min(dt2_1, dt2_2, dt2_3, dt2_4)
    sh_dt2[th_id+blk_size] = min(dt2a_1, dt2a_2, dt2a_3, dt2a_4)

    #sh_dt2[th_id+blk_size] = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ηx[i1,j1], ηy[i1,j1])
  end
  sync_threads()

  #s = fld(blk_size, 2)
  s = blk_size
  while s > 0
    if (th_id - 1) < s
      @inbounds begin
        sh_dt1[th_id] = min(sh_dt1[th_id], sh_dt1[th_id + s])
        sh_dt2[th_id] = min(sh_dt2[th_id], sh_dt2[th_id + s])
      end
    end
    s = fld(s,2)
    sync_threads()
  end

  if th_id == 1
    @inbounds begin
      glb_min_dt1[blk_id] = sh_dt1[th_id]
      glb_min_dt2[blk_id] = sh_dt2[th_id]
    end
  end

  return 
end

function reduce_dt_stride_tp((v⃗x, v⃗y)::NTuple{2,AA}, cs, (ξx, ξy, ηx, ηy), CFL, glb_min_dt1, glb_min_dt2) where {AA}
  #i = (blockIdx().y - 1) * (2*blockDim().y) + threadIdx().y
  #j = (blockIdx().x - 1) * (2*blockDim().x) + threadIdx().x

  # i = (blockIdx().y - 1) * (blockDim().y) + threadIdx().y
  # j = (blockIdx().x - 1) * (blockDim().x) + threadIdx().x
  # i1 = i + (blockDim().y * gridDim().y)
  # j1 = j + (blockDim().x * gridDim().x)
  j = (blockIdx().y - 1) * (blockDim().y) + threadIdx().y
  i = (blockIdx().x - 1) * (blockDim().x) + threadIdx().x
  j1 = j + (blockDim().y * gridDim().y)
  i1 = i + (blockDim().x * gridDim().x)

  blk_size = blockDim().x * blockDim().y
  th_id = blockDim().x * (threadIdx().y - 1) + threadIdx().x
  blk_id = gridDim().x * (blockIdx().y - 1) + blockIdx().x

  arr_type = eltype(glb_min_dt1)
  #sh_dt1 = CUDA.CuDynamicSharedArray(arr_type, (2*blk_size))
  #sh_dt2 = CUDA.CuDynamicSharedArray(#arr_type, (2*blk_size))
  sh_dt1 = CUDA.CuDynamicSharedArray(arr_type, (blk_size))
  sh_dt2 = CUDA.CuDynamicSharedArray(arr_type, (blk_size))

  @inbounds begin
    dt1_1 = next_dt_2d(v⃗x[i,j], v⃗y[i,j], cs[i,j], ξx[i,j], ξy[i,j])
    dt1_2 = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ξx[i1,j1], ξy[i1,j1])
    dt1_3 = next_dt_2d(v⃗x[i1,j], v⃗y[i1,j], cs[i1,j], ξx[i1,j], ξy[i1,j])
    dt1_4 = next_dt_2d(v⃗x[i,j1], v⃗y[i,j1], cs[i,j1], ξx[i,j1], ξy[i,j1])
    dt2_1 = next_dt_2d(v⃗x[i,j], v⃗y[i,j], cs[i,j], ηx[i,j], ηy[i,j])
    dt2_2 = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ηx[i1,j1], ηy[i1,j1])
    dt2_3 = next_dt_2d(v⃗x[i1,j], v⃗y[i1,j], cs[i1,j], ηx[i1,j], ηy[i1,j])
    dt2_4 = next_dt_2d(v⃗x[i,j1], v⃗y[i,j1], cs[i,j1], ηx[i,j1], ηy[i,j1])
    sh_dt1[th_id] = min(dt1_1, dt1_2, dt1_3, dt1_4)
    #sh_dt1[th_id+blk_size] = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ξx[i1,j1], ξy[i1,j1])
    sh_dt2[th_id] = min(dt2_1, dt2_2, dt2_3, dt2_4)
    #sh_dt2[th_id+blk_size] = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ηx[i1,j1], ηy[i1,j1])
  end
  sync_threads()

  s = fld(blk_size, 2)
  #s = blk_size
  while s > 0
    if (th_id - 1) < s
      @inbounds begin
        sh_dt1[th_id] = min(sh_dt1[th_id], sh_dt1[th_id + s])
        sh_dt2[th_id] = min(sh_dt2[th_id], sh_dt2[th_id + s])
      end
    end
    s = fld(s,2)
    sync_threads()
  end

  if th_id == 1
    @inbounds begin
      glb_min_dt1[blk_id] = sh_dt1[th_id]
      glb_min_dt2[blk_id] = sh_dt2[th_id]
    end
  end

  return 
end

function reduce_dt_stride_dimsplit((v⃗x, v⃗y)::NTuple{2,AA}, cs, (ξx, ξy), CFL, glb_min_dt) where {AA}
  #i = (blockIdx().y - 1) * (2*blockDim().y) + threadIdx().y
  #j = (blockIdx().x - 1) * (2*blockDim().x) + threadIdx().x
  i = (blockIdx().y - 1) * (blockDim().y) + threadIdx().y
  j = (blockIdx().x - 1) * (blockDim().x) + threadIdx().x
  i1 = i + (blockDim().y * gridDim().y)
  j1 = j + (blockDim().x * gridDim().x)

  blk_size = blockDim().x * blockDim().y
  th_id = blockDim().x * (threadIdx().y - 1) + threadIdx().x
  blk_id = gridDim().x * (blockIdx().y - 1) + blockIdx().x

  arr_type = eltype(glb_min_dt)
  #sh_dt1 = CUDA.CuDynamicSharedArray(arr_type, (2*blk_size))
  #sh_dt2 = CUDA.CuDynamicSharedArray(#arr_type, (2*blk_size))
  sh_dt = CUDA.CuDynamicSharedArray(arr_type, (blk_size))
  #sh_dt2 = CUDA.CuDynamicSharedArray(arr_type, (blk_size))

  @inbounds begin
    dt1_1 = next_dt_2d(v⃗x[i,j], v⃗y[i,j], cs[i,j], ξx[i,j], ξy[i,j])
    dt1_2 = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ξx[i1,j1], ξy[i1,j1])
    dt1_3 = next_dt_2d(v⃗x[i1,j], v⃗y[i1,j], cs[i1,j], ξx[i1,j], ξy[i1,j])
    dt1_4 = next_dt_2d(v⃗x[i,j1], v⃗y[i,j1], cs[i,j1], ξx[i,j1], ξy[i,j1])
    # dt2_1 = next_dt_2d(v⃗x[i,j], v⃗y[i,j], cs[i,j], ηx[i,j], ηy[i,j])
    # dt2_2 = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ηx[i1,j1], ηy[i1,j1])
    # dt2_3 = next_dt_2d(v⃗x[i1,j], v⃗y[i1,j], cs[i1,j], ηx[i1,j], ηy[i1,j])
    # dt2_4 = next_dt_2d(v⃗x[i,j1], v⃗y[i,j1], cs[i,j1], ηx[i,j1], ηy[i,j1])
    sh_dt[th_id] = min(dt1_1, dt1_2, dt1_3, dt1_4)
    #sh_dt1[th_id+blk_size] = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ξx[i1,j1], ξy[i1,j1])
    #sh_dt2[th_id] = min(dt2_1, dt2_2, dt2_3, dt2_4)
    #sh_dt2[th_id+blk_size] = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ηx[i1,j1], ηy[i1,j1])
  end
  sync_threads()

  s = fld(blk_size, 2)
  #s = blk_size
  while s > 0
    if (th_id - 1) < s
      @inbounds begin
        sh_dt[th_id] = min(sh_dt[th_id], sh_dt[th_id + s])
        #sh_dt2[th_id] = min(sh_dt2[th_id], sh_dt2[th_id + s])
      end
    end
    s = fld(s,2)
    sync_threads()
  end

  if th_id == 1
    @inbounds begin
      glb_min_dt[blk_id] = sh_dt[th_id]
      #glb_min_dt2[blk_id] = sh_dt2[th_id]
    end
  end

  return 
end

function main()
  ni, nj = (2048, 2048)
  T = Float32
  vx = cu(rand(T, (ni, nj)));
  vy = cu(rand(T, (ni, nj)));
  cs = cu(rand(T, (ni, nj)));
  ξx = cu(rand(T, (ni, nj)));
  ξy = cu(rand(T, (ni, nj)));
  ηx = cu(rand(T, (ni, nj)));
  ηy = cu(rand(T, (ni, nj)));

  ## CPU Data Structs
  vx_cpu = rand(T, (ni, nj));
  vy_cpu = rand(T, (ni, nj));
  cs_cpu = rand(T, (ni, nj));
  ξx_cpu = rand(T, (ni, nj));
  ξy_cpu = rand(T, (ni, nj));
  ηx_cpu = rand(T, (ni, nj));
  ηy_cpu = rand(T, (ni, nj));

  #mapreduce(next_dt_2d, min, U, cs, (ξ, η))
  CFL = 0.2
  #CFL = Float32{CFL}

  # next_dt((vx, vy), cs, (ξx, ξy, ηx, ηy), CFL)
  # @show @benchmark next_dt(($vx, $vy), $cs, ($ξx, $ξy, $ηx, $ηy), $CFL)

  # #@show typeof(vx)

  blkdim_x = 4
  blkdim_y = 64 
  warpsize = 32
  stride_x = 2
  stride_y = 2
  blk_size = blkdim_x * blkdim_y
  grddim_x_grdstride = cld(nj, (stride_x*blkdim_x))
  grddim_y_grdstride = cld(ni, (stride_y*blkdim_y))
  grdsize_grdstride = grddim_x_grdstride * grddim_y_grdstride

  grddim_x = cld(nj, (blkdim_x))
  grddim_y = cld(ni, (blkdim_y))
  grdsize = grddim_x * grddim_y

  #glb_min_dt = cu(zeros(grdsize));
  glb_min_dt1_stride = cu(zeros(T, grdsize_grdstride));
  glb_min_dt2_stride = cu(zeros(T, grdsize_grdstride));
  glb_min_dt1_seq = cu(zeros(T, grdsize));
  glb_min_dt1_inter = copy(glb_min_dt1_seq)
  glb_min_dt1_nondiv = copy(glb_min_dt1_seq)
  glb_min_dt2_seq = cu(zeros(T, grdsize));
  glb_min_dt2_inter = copy(glb_min_dt2_seq)
  glb_min_dt2_nondiv = copy(glb_min_dt2_seq)
  glb_min_dt1_shufl = copy(glb_min_dt1_seq)
  glb_min_dt2_shufl = copy(glb_min_dt1_seq)
  glb_min_dt1_unroll = copy(glb_min_dt1_seq)
  glb_min_dt2_unroll = copy(glb_min_dt1_seq)
  glb_min_dt1_shufl_stride = copy(glb_min_dt1_stride)
  glb_min_dt2_shufl_stride = copy(glb_min_dt1_stride)

  el_type = eltype(glb_min_dt1_seq)
  shmem_size_stride = sizeof(el_type) * (blk_size) * 2
  shmem_size = sizeof(el_type) * (blk_size) * 2
  shmem_size_shufl = sizeof(el_type) * cld(blk_size, warpsize) * 2
  println("Running Reduce for CUDA")
  # CUDA.@profile begin
  #   #next_dt((vx, vy), cs, (ξx, ξy, ηx, ηy), CFL)
     CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(nj, (stride_x*blkdim_x)), cld(ni, (stride_y*blkdim_y))) shmem=shmem_size_stride reduce_dt_stride((vx, vy), cs, (ξx, ξy, ηx, ηy), CFL, glb_min_dt1_stride, glb_min_dt2_stride)
    
  #   #CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(nj, (stride_x*blkdim_x)), cld(ni, (stride_y*blkdim_y))) shmem=shmem_size_stride reduce_dt_stride_unroll((vx, vy), cs, (ξx, ξy, ηx, ηy), CFL, glb_min_dt1_stride, glb_min_dt2_stride)

  #   #CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(nj, (stride_x*blkdim_x)), cld(ni, (stride_y*blkdim_y))) shmem=(2*shmem_size_stride) reduce_dt_shdouble((vx, vy), cs, (ξx, ξy, ηx, ηy), CFL, glb_min_dt1_stride, glb_min_dt2_stride)
  #   # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(nj, (blkdim_x)), cld(ni, (blkdim_y))) shmem=shmem_size reduce_dt_sequential((vx, vy), cs, (ξx, ξy, ηx, ηy), CFL, glb_min_dt1_seq, glb_min_dt2_seq)
  #   # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(nj, (blkdim_x)), cld(ni, (blkdim_y))) shmem=shmem_size reduce_dt_nondiv((vx, vy), cs, (ξx, ξy, ηx, ηy), CFL, glb_min_dt1_nondiv, glb_min_dt2_nondiv)

  #   # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(nj, (blkdim_x)), cld(ni, (blkdim_y))) shmem=shmem_size reduce_dt_interleaved((vx, vy), cs, (ξx, ξy, ηx, ηy), CFL, glb_min_dt1_inter, glb_min_dt2_inter)

      CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(nj, (stride_x*blkdim_x)), cld(ni, (stride_y*blkdim_y))) shmem=shmem_size_stride reduce_dt_nondiv_stride((vx, vy), cs, (ξx, ξy, ηx, ηy), CFL, glb_min_dt1_stride, glb_min_dt2_stride)

  #   CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(nj, (blkdim_x)), cld(ni, (blkdim_y))) shmem=shmem_size reduce_dt_unroll((vx, vy), cs, (ξx, ξy, ηx, ηy), CFL, glb_min_dt1_unroll, glb_min_dt2_unroll)

  #   CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(nj, (blkdim_x)), cld(ni, (blkdim_y))) shmem=shmem_size_shufl reduce_dt_shufl((vx, vy), cs, (ξx, ξy, ηx, ηy), CFL, glb_min_dt1_shufl, glb_min_dt2_shufl)

  #   CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(nj, (stride_x*blkdim_x)), cld(ni, (stride_y*blkdim_y))) shmem=shmem_size_shufl reduce_dt_shufl_stride((vx, vy), cs, (ξx, ξy, ηx, ηy), CFL, glb_min_dt1_shufl_stride, glb_min_dt2_shufl_stride)

  #   CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(nj, (stride_x*blkdim_x)), cld(ni, (stride_y*blkdim_y))) shmem=shmem_size reduce_dt_stride_unroll((vx, vy), cs, (ξx, ξy, ηx, ηy), CFL, glb_min_dt1_stride, glb_min_dt2_stride)
  #   # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(nj, (stride_x*blkdim_x)), cld(ni, (stride_y*blkdim_y))) shmem=cld(shmem_size_stride,2) reduce_dt_stride_dimsplit((vx, vy), cs, (ξx, ξy), CFL, glb_min_dt1_stride)

  #   # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(nj, (stride_x*blkdim_x)), cld(ni, (stride_y*blkdim_y))) shmem=cld(shmem_size_stride,2) reduce_dt_stride_dimsplit((vx, vy), cs, (ηx, ηy), CFL, glb_min_dt2_stride)
  #   # shmem_size_stride_shdouble = sizeof(el_type) * (2*blk_size) * 2
  #   # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(nj, (2*stride_x*blkdim_x)), cld(ni, (stride_y*blkdim_y))) shmem=shmem_size_stride_shdouble reduce_dt_stride_shdouble((vx, vy), cs, (ξx, ξy, ηx, ηy), CFL, glb_min_dt1_stride, glb_min_dt2_stride)

  #   # blkdim_x = 64
  #   # blkdim_y = 4
  #   # CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(nj, (stride_x*blkdim_x)), cld(ni, (stride_y*blkdim_y))) shmem=shmem_size_stride reduce_dt_stride_tp((vx, vy), cs, (ξx, ξy, ηx, ηy), CFL, glb_min_dt1_stride, glb_min_dt2_stride)

  # end

  # println("Reduce Seq matches Reduce Inter :", Array(glb_min_dt1_seq) == Array(glb_min_dt1_inter))
  # println("Reduce Inter matches Reduce nondiv :", Array(glb_min_dt1_inter) == Array(glb_min_dt1_nondiv))
  # println("Reduce Inter matches Reduce Seq :", Array(glb_min_dt1_inter) == Array(glb_min_dt1_seq))
  # @show CUDA.@allowscalar glb_min_dt1_seq[2]
  # @show CUDA.@allowscalar glb_min_dt1_inter[2]
  # @show CUDA.@allowscalar glb_min_dt1_nondiv[2]

  #delt = next_dt((vx, vy), cs, (ξx, ξy, ηx, ηy), CFL)
  #delt_gpu = CFL * minimum!(minimum!(Array(glb_min_dt1_inter)), minimum!(Array(glb_min_dt2_inter)))
  #println("CPU version matches GPU:", delt == delt_gpu)

  #next_dt((vx, vy), cs, (ξx, ξy, ηx, ηy), CFL)
  next_dt((vx_cpu, vy_cpu), cs_cpu, (ξx_cpu, ξy_cpu, ηx_cpu, ηy_cpu), CFL)
  #@benchmark next_dt(($vx, $vy), $cs, ($ξx, $ξy, $ηx, $ηy), $CFL)
  @benchmark next_dt(($vx_cpu, $vy_cpu), $cs_cpu, ($ξx_cpu, $ξy_cpu, $ηx_cpu, $ηy_cpu), $CFL)

  #@benchmark CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(ni, blkdim_x), cld(nj, blkdim_y)) shmem=shmem_size reduce_dt(($vx, $vy), $cs, ($ξx, $ξy, $ηx, $ηy), $CFL, $glb_min_dt1, $glb_min_dt2)

  # function stage_l2norm(ϕ1, ϕn)
  #   denom = sqrt(mapreduce(x -> x^2, +, ϕ1))

  #   if isinf(denom) || iszero(denom)
  #     l2norm = -Inf
  #   else
  #     f(x, y) = (x - y)^2
  #     numerator = sqrt(mapreduce(f, +, ϕn, ϕ1))

  #     l2norm = numerator / denom
  #   end

  #   return l2norm
  # end

  # U1 = cu(rand(ni, nj));
  # Un = cu(rand(ni, nj));

  # stage_l2norm(U1, Un)
end

isinteractive() || main()