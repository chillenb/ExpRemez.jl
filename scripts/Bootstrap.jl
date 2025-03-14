try
  using MKL
catch
  println("MKL not found, using default BLAS")
end

using ExpRemez
using DoubleFloats
using JLD2
using Symbolics
using ArgParse

function parse_commandline()
  s = ArgParseSettings()

  @add_arg_table s begin
      "--grid_type", "-g"
          help = "grid type: 'time', 'freq_even', 'freq_odd'"
          default = "time"
          arg_type = String
      "--kmax", "-k"
          help = "maximum grid size"
          default = 50
          arg_type = Int
      "filename"
          help = "name of JLD2 output file"
          arg_type = String
          required = true
  end

  return parse_args(s)
end

function main()
  parsed_args = parse_commandline()
  fname = parsed_args["filename"]
  kmax = parsed_args["kmax"]
  key = parsed_args["grid_type"]

  @variables _x, _a, _b

  opt_funcs = ExpRemez.funcs_all[key]
  guess_func = ExpRemez.upgrade_guesses[key]

  grd = convert(MinimaxGrid{Double64}, ExpRemez.seed_grids[key])

  grds = [grd]

  f = jldopen(fname, "w")
  write_grid(f, "1", grds[1])
  close(f)

  for i = 2:kmax
    println(i)
    cur_grd = grds[i-1]
    if cur_grd.err / cur_grd.n > 1e-13
      println("$i: using Float64")
      cur_grd = convert(MinimaxGrid{Float64}, cur_grd)
      new_grd = upgrade_gridsize(cur_grd, funcs=opt_funcs, upgrade_guess=guess_func)
      new_grd = convert(MinimaxGrid{Double64}, new_grd)
      new_grd = compute_minimax_grid(new_grd, Inf, funcs=opt_funcs)
    else
      new_grd = upgrade_gridsize(cur_grd, funcs=opt_funcs, upgrade_guess=guess_func)
    end
    push!(grds, deepcopy(new_grd))
  end

  f = jldopen(fname, "r+")
  for i = 2:kmax
    write_grid(f, "$i", grds[i])
  end
  close(f)
end

main()
