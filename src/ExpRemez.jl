module ExpRemez

using DoubleFloats
using GenericLinearAlgebra
using Roots
using JLD2

using Symbolics
import Base: convert

using PrecompileTools: @setup_workload, @compile_workload    # this is a small dependency

export newton_interp_by_expsum, newton_interp_by_expsum!

export get_extrema_bounded, compute_minimax_grid, merge_params, split_params
export get_extrema_unbounded, bound_extrema, is_R_inf, _grd1inf

export upgrade_gridsize, expand_grid, write_grid, shrink_grid

export MinimaxGrid
export write_grid, seed_grids, funcs_all
export _freqgrd1inf_even
export funcs_freq_even, funcs_time

include("MinimaxGridDef.jl")
using .MinimaxGridDef

include("Autocode.jl")
include("Data.jl")
include("Optimizer.jl")
include("UpgradeGrid.jl")

@variables _x, _a, _b
funcs_time = gen_funs(1/_x, exp(-_b*_x), _b, _x)
funcs_freq_even = gen_funs(1/_x, (_x/(_x^2+_b^2))^2, _b, _x);
funcs_freq_odd = gen_funs(1/_x, (_b/(_x^2+_b^2))^2, _b, _x);

funcs_all = Dict(
    "time" => funcs_time,
    "freq_even" => funcs_freq_even,
    "freq_odd" => funcs_freq_odd
)

@setup_workload begin
  @compile_workload begin
    time_grid_1 = convert(MinimaxGrid{Double64}, _grd1inf)
    load_grids_inf()
    for key in ["time", "freq_even", "freq_odd"]
      for K in [Float64, Double64]
        grd = convert(MinimaxGrid{K}, seed_grids[key])
        compute_minimax_grid(grd, 20, funcs=funcs_all[key])
        upgrade_gridsize(grd, funcs=funcs_all[key], upgrade_guess=upgrade_guesses[key])
      end
    end
    shrink_grid(time_grid_1, 4.0, 1.1; funcs=funcs_time, start_fp64=true, err_min=1e-12)
  end
end

end
