using ExpRemez
using GenericLinearAlgebra
using NonlinearSolve
using DoubleFloats
import ParameterSchedulers
using JLD2

using LinearAlgebra
BLAS.set_num_threads(1)

grds_inf = ExpRemez.load_grids_inf()


function gen_from_grid_inf(orig_grd::MinimaxGrid{T}, Rlist, progress=true) where {T}
  if progress
    println("Started grid of size ", orig_grd.n)
  end
  grd = deepcopy(orig_grd)
  R0 = isinf(grd.R) ? grd.extrema[end] : grd.R

  sorted_Rs = T.(sort(Rlist[Rlist.<R0], rev=true))
  results = []
  start_fp64 = true
  for R in sorted_Rs
    try
      grd, dtype = shrink_grid(grd, R, 1.1, verbose=false, start_fp64=start_fp64)
      if dtype != Float64
        if progress && start_fp64
          println("n: ", grd.n, ", Disabling low precision start")
        end
        start_fp64 = false
      end
      push!(results, deepcopy(grd))
    catch e
      println("Error: ", e)
      break
    end
  end
  if progress
    println("Finished grid of size ", grd.n)
  end
  return results
end

kmax = parse(Int64, ARGS[2])
kmin = parse(Int64, ARGS[3])
fname = ARGS[1]

Rlist = collect(1:9) .* exp10.(3:12)'
Rlist = Rlist[:]



# tasks = [Threads.@spawn gen_from_grid_inf(grds_inf[i], Rlist) for i in kmax:-1:kmin]
# results = reduce(vcat, fetch.(tasks))

res = []
mylock = ReentrantLock()

Threads.@threads :greedy for i in kmax:-1:kmin
  res_i = gen_from_grid_inf(grds_inf[i], Rlist)
  lock(mylock)
  try
    append!(res, res_i)
  finally
    unlock(mylock)
  end
end


f = jldopen(fname, "w")
for r in res
  n = r.n
  R = r.R
  if isinf(R)
    grd_name = "$n"*"_inf"
  else
    R = Int64(round(Float64(R)))
    exponent = Int64(floor(log10(R)))
    coef = Int64(R / 10^exponent)
    grd_name = "$n"*"_"*"$coef"*"E"*"$exponent"
  end
  write_grid(f, grd_name, r)
end
close(f)