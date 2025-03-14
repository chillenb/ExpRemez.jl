try
  using MKL
catch
  println("MKL not found, using default BLAS")
end

using ExpRemez
using JLD2
using LinearAlgebra
using Printf
using ArgParse

BLAS.set_num_threads(1)


function parse_commandline()
  s = ArgParseSettings()

  @add_arg_table s begin
      "--grid_type", "-g"
          help = "grid type: 'time', 'freq_even', 'freq_odd'"
          default = "time"
          arg_type = String
      "--kmax", "-k"
          help = "maximum grid size"
          default = 12
          arg_type = Int
      "--kmin", "-m"
          help = "minimum grid size"
          default = 9
          arg_type = Int
      "filename"
          help = "name of JLD2 output file"
          arg_type = String
          required = true
  end

  return parse_args(s)
end



function gen_from_grid_inf(orig_grd::MinimaxGrid{T}, Rlist; funcs, progress=true, verbose=true, shrink_start=1.1) where {T}
  if progress
    println("Started grid of size ", orig_grd.n)
  end
  grd = deepcopy(orig_grd)
  R0 = isinf(grd.R) ? grd.extrema[end] : grd.R

  shrink_ratio = T(shrink_start)

  sorted_Rs = T.(sort(Rlist[Rlist.<R0], rev=true))
  results = []
  start_fp64 = true
  for R in sorted_Rs
    conv = false
    while !conv
      try
        grd, dtype, shrink_ratio_out = shrink_grid(grd, R, shrink_ratio, verbose=verbose, start_fp64=start_fp64, funcs=funcs)
        if dtype != Float64
          if progress && start_fp64
            println("n: ", grd.n, ", Disabling low precision start")
            start_fp64 = false
          end
        end
        conv = true
        push!(results, deepcopy(grd))
        #shrink_ratio = T(shrink_ratio_out)
      catch e
        if e isa InterruptException
          rethrow(e)
        end
        if start_fp64
          println("n: ", grd.n, ", Disabling low precision start")
          start_fp64 = false
          shrink_ratio = T(shrink_ratio_start)
        else
          println("Error: ", e)
          exit(1)
        end
      end
    end
  end
  if progress
    println("Finished grid of size ", grd.n)
  end
  return results
end

function main()

  parsed_args = parse_commandline()

  kmax = parsed_args["kmax"]
  kmin = parsed_args["kmin"]
  fname = parsed_args["filename"]

  key = parsed_args["grid_type"] # "time", "freq_even", "freq_odd"


  grds_inf = ExpRemez.load_grids_inf(key)
  opt_funcs = ExpRemez.funcs_all[key]

  pow_max = 12
  pow_min = 9

  Rlist = logrange(10^pow_min, 10^pow_max, 10*(pow_max - pow_min)+1)
  Rlist = Rlist[:]



  res = []
  mylock = ReentrantLock()

  Threads.@threads :greedy for i in kmax:-1:kmin
    res_i = gen_from_grid_inf(grds_inf[i], Rlist; funcs=opt_funcs)
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
      rstring = @sprintf "%.3E" Float64(R)
      grd_name = "$(key)_$(n)_$(rstring)"
    end
    write_grid(f, grd_name, r)
  end
  close(f)

end

main()