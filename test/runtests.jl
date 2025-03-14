using ExpRemez
using Test
using DoubleFloats
using Symbolics

time_grids_inf = ExpRemez.load_grids_inf()

function grids_are_close(grd1, grd2; atol)
  return grd1.n == grd2.n && 
  isapprox(grd1.coefs, grd2.coefs; atol=atol) && 
  isapprox(grd1.gridpts, grd2.gridpts; atol=atol) &&
  isapprox(grd1.interp_pts, grd2.interp_pts; atol=atol) &&
  isapprox(grd1.R, grd2.R; atol=atol)
end

@testset verbose = true "bootstrap" begin
  grd = convert(MinimaxGrid{Double64}, ExpRemez._grd1inf)
  @test grids_are_close(grd, time_grids_inf[1]; atol=1e-15)
  @testset "upgrade_grid $i" for i = 2:15
    grd_new = upgrade_gridsize(grd, funcs=ExpRemez.funcs_time, upgrade_guess=ExpRemez.upgrade_time_gridsize_guess)
    grd_new = compute_minimax_grid(grd_new, Inf, funcs=ExpRemez.funcs_time)
    @test grids_are_close(grd_new, time_grids_inf[i]; atol=1e-15)
    grd = deepcopy(grd_new)
  end
end