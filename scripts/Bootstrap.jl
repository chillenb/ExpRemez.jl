using MKL
using ExpRemez
using DoubleFloats
using JLD2
using Symbolics

@variables _x, _a, _b
key = ARGS[3] # "time", "freq_even", "freq_odd"
fname = ARGS[1]
kmax = parse(Int64, ARGS[2]) # 100

opt_funcs = ExpRemez.funcs_all[key]
guess_func = ExpRemez.upgrade_guesses[key]

grd = convert(MinimaxGrid{Double64}, ExpRemez.seed_grids[key])

grds = [grd]

f = jldopen(ARGS[1], "w")
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

f = jldopen(ARGS[1], "r+")
for i = 2:kmax
  write_grid(f, "$i", grds[i])
end
