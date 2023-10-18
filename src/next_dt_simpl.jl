using CUDA
using BenchmarkTools
using Revise
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

function next_dt((v⃗x, v⃗y, v⃗z), cs, (ξx, ξy, ξz, ηx, ηy, ηz, ζx, ζy, ζz), CFL)
  Δt_ξ = mapreduce(next_dt_3d, min, (v⃗x, v⃗y, v⃗z), cs, (ξx, ξy, ξz))
  Δt_η = mapreduce(next_dt_3d, min, (v⃗x, v⃗y, v⃗z), cs, (ηx, ηy, ηz))
  Δt_ζ = mapreduce(next_dt_3d, min, (v⃗x, v⃗y, v⃗z), cs, (ζx, ζy, ζz))

  Δt = CFL * min(Δt_ξ, Δt_η, Δt_ζ)
  return Δt
end

function reduce_dt((v⃗x, v⃗y)::NTuple{2,AA}, cs, (ξx, ξy, ηx, ηy), CFL, glb_min_dt1, glb_min_dt2) where {AA}
  i = (blockIdx().y - 1) * (blockDim().y) + threadIdx().y
  j = (blockIdx().x - 1) * (blockDim().x) + threadIdx().x
  # i1 = i + blockDim().y
  # j1 = j + blockDim().x

  blk_size = blockDim().x * blockDim().y
  th_id = blockDim().x * (threadIdx().y - 1) + threadIdx().x
  blk_id = gridDim().x * (blockIdx().y - 1) + blockIdx().x

  arr_type = eltype(glb_min_dt1)
  sh_dt1 = CUDA.CuDynamicSharedArray(arr_type, (blk_size))
  sh_dt2 = CUDA.CuDynamicSharedArray(arr_type, (blk_size))

  sh_dt1[th_id] = next_dt_2d(v⃗x[i,j], v⃗y[i,j], cs[i,j], ξx[i,j], ξy[i,j])
  #sh_dt1[th_id+blk_size] = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ξx[i1,j1], ξy[i1,j1])
  sh_dt2[th_id] = next_dt_2d(v⃗x[i,j], v⃗y[i,j], cs[i,j], ηx[i,j], ηy[i,j])
  #sh_dt2[th_id+blk_size] = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ηx[i1,j1], ηy[i1,j1])
  sync_threads()

  s = fld(blk_size, 2)
  #s = blk_size
  while s > 0
    if (th_id - 1) < s
      sh_dt1[th_id] = min(sh_dt1[th_id], sh_dt1[th_id + s])
      sh_dt2[th_id] = min(sh_dt2[th_id], sh_dt2[th_id + s])
    end
    s = fld(s,2)
    sync_threads()
  end

  if th_id == 1
    glb_min_dt1[blk_id] = sh_dt1[th_id]
    glb_min_dt2[blk_id] = sh_dt2[th_id]
  end

  return 
end

function reduce_opt_dt((v⃗x, v⃗y)::NTuple{2,AA}, cs, (ξx, ξy, ηx, ηy), CFL, glb_min_dt1, glb_min_dt2) where {AA}
  i = (blockIdx().y - 1) * (2*blockDim().y) + threadIdx().y
  j = (blockIdx().x - 1) * (2*blockDim().x) + threadIdx().x
  i1 = i + blockDim().y
  j1 = j + blockDim().x

  blk_size = blockDim().x * blockDim().y
  th_id = blockDim().x * (threadIdx().y - 1) + threadIdx().x
  blk_id = gridDim().x * (blockIdx().y - 1) + blockIdx().x

  arr_type = eltype(glb_min_dt1)
  sh_dt1 = CUDA.CuDynamicSharedArray(arr_type, (2*blk_size))
  sh_dt2 = CUDA.CuDynamicSharedArray(arr_type, (2*blk_size))

  sh_dt1[th_id] = next_dt_2d(v⃗x[i,j], v⃗y[i,j], cs[i,j], ξx[i,j], ξy[i,j])
  sh_dt1[th_id+blk_size] = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ξx[i1,j1], ξy[i1,j1])
  sh_dt2[th_id] = next_dt_2d(v⃗x[i,j], v⃗y[i,j], cs[i,j], ηx[i,j], ηy[i,j])
  sh_dt2[th_id+blk_size] = next_dt_2d(v⃗x[i1,j1], v⃗y[i1,j1], cs[i1,j1], ηx[i1,j1], ηy[i1,j1])
  sync_threads()

  #s = fld(blk_size, 2)
  s = blk_size
  while s > 0
    if (th_id - 1) < s
      sh_dt1[th_id] = min(sh_dt1[th_id], sh_dt1[th_id + s])
      sh_dt2[th_id] = min(sh_dt2[th_id], sh_dt2[th_id + s])
    end
    s = fld(s,2)
    sync_threads()
  end

  if th_id == 1
    glb_min_dt1[blk_id] = sh_dt1[th_id]
    glb_min_dt2[blk_id] = sh_dt2[th_id]
  end

  return 
end

function main()
  ni, nj = (2048, 2048)
  vx = cu(rand(ni, nj));
  vy = cu(rand(ni, nj));
  cs = cu(rand(ni, nj));
  ξx = cu(rand(ni, nj));
  ξy = cu(rand(ni, nj));
  ηx = cu(rand(ni, nj));
  ηy = cu(rand(ni, nj));
  #mapreduce(next_dt_2d, min, U, cs, (ξ, η))
  CFL = 0.2
  #CFL = Float32{CFL}

  # next_dt((vx, vy), cs, (ξx, ξy, ηx, ηy), CFL)
  # @show @benchmark next_dt(($vx, $vy), $cs, ($ξx, $ξy, $ηx, $ηy), $CFL)

  # #@show typeof(vx)

  blkdim_x = 4
  blkdim_y = 64
  blk_size = blkdim_x * blkdim_y
  grddim_opt_x = cld(nj, (2*blkdim_x))
  grddim_opt_y = cld(ni, (2*blkdim_y))
  grdsize_opt = grddim_opt_x * grddim_opt_y

  grddim_x = cld(nj, (blkdim_x))
  grddim_y = cld(ni, (blkdim_y))
  grdsize = grddim_x * grddim_y

  #glb_min_dt = cu(zeros(grdsize));
  glb_min_dt1_opt = cu(zeros(grdsize_opt));
  glb_min_dt2_opt = cu(zeros(grdsize_opt));
  glb_min_dt1 = cu(zeros(grdsize));
  glb_min_dt2 = cu(zeros(grdsize));

  el_type = eltype(glb_min_dt1)
  shmem_size_opt = sizeof(el_type) * (blk_size * 2) * 2
  shmem_size = sizeof(el_type) * (blk_size) * 2
  println("Running Reduce for CUDA")
  CUDA.@profile begin
    #next_dt((vx, vy), cs, (ξx, ξy, ηx, ηy), CFL)
    CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(nj, (2*blkdim_x)), cld(ni, (2*blkdim_y))) shmem=shmem_size_opt reduce_opt_dt((vx, vy), cs, (ξx, ξy, ηx, ηy), CFL, glb_min_dt1_opt, glb_min_dt2_opt)
    
    CUDA.@sync @cuda threads=(blkdim_x, blkdim_y) blocks=(cld(nj, (blkdim_x)), cld(ni, (blkdim_y))) shmem=shmem_size reduce_dt((vx, vy), cs, (ξx, ξy, ηx, ηy), CFL, glb_min_dt1, glb_min_dt2)
  end

  next_dt((vx, vy), cs, (ξx, ξy, ηx, ηy), CFL)
  @benchmark next_dt(($vx, $vy), $cs, ($ξx, $ξy, $ηx, $ηy), $CFL)
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