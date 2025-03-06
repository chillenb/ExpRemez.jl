using ExpRemez
using DoubleFloats
using JLD2
using Symbolics

@variables _x, _a, _b
funcs = ExpRemez.gen_funs(1/_x, exp(-_b*_x), _b, _x);

grd = convert(MinimaxGrid{Double64}, ExpRemez._grd1inf)

grds = Dict()
cur_grd = grd

grds[1] = grd
f = jldopen(ARGS[1], "w")
write_grid(f, "1", grds[1])
close(f)

for i = 2:parse(Int64, ARGS[2])
  global cur_grd
  println(cur_grd.n)
  grds[i] = upgrade_gridsize(cur_grd, funcs=funcs, upgrade_guess=ExpRemez.upgrade_time_gridsize_guess)
  cur_grd = grds[i]
  f = jldopen(ARGS[1], "r+")
  write_grid(f, "$i", grds[i])
  close(f)
end


