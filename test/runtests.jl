using ExpRemez
using Test
using DoubleFloats
using Symbolics

@variables _x, _a, _b
funcs = ExpRemez.gen_funs(1/_x, exp(-_b*_x), _b, _x);


time_grids_inf = ExpRemez.load_grids_inf()

function grids_are_close(grd1, grd2; atol)
  return grd1.n == grd2.n && 
  isapprox(grd1.coefs, grd2.coefs; atol=atol) && 
  isapprox(grd1.gridpts, grd2.gridpts; atol=atol) &&
  isapprox(grd1.interp_pts, grd2.interp_pts; atol=atol) &&
  isapprox(grd1.R, grd2.R; atol=atol)
end

@testset "bootstrap" begin
  grd = convert(MinimaxGrid{Double64}, ExpRemez._grd1inf)
  @test grids_are_close(grd, time_grids_inf[1]; atol=1e-15)
  for i = 2:10
    grd = upgrade_gridsize(grd, funcs=funcs, upgrade_guess=ExpRemez.upgrade_time_gridsize_guess)
    @test grids_are_close(grd, time_grids_inf[i]; atol=1e-15)
  end
end