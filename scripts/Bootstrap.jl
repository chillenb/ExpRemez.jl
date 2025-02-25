using ExpRemez
using GenericLinearAlgebra
using NonlinearSolve
using DoubleFloats
import ParameterSchedulers
using JLD2


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
  grds[i] = upgrade_gridsize(cur_grd)
  cur_grd = grds[i]
  f = jldopen(ARGS[1], "r+")
  write_grid(f, "$i", grds[i])
  close(f)
end


