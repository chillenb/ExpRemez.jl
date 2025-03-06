module ExpRemez

using DoubleFloats
using GenericLinearAlgebra
using NonlinearSolve
using JLD2
import ParameterSchedulers
import Base: convert

using PrecompileTools: @setup_workload, @compile_workload    # this is a small dependency

export expsum_eval, dparams_dxi, expsum_jacobian, expsum_xderiv, newton_interp_by_expsum, newton_interp_by_expsum!
export fscalar, fderiv_scalar
export dparams_dxi, phi_grad_ab, phi, phi_grad_xi, phi_fdiff_grad_xi, params_grad_xi

export lorentzsum_eval, lorentzsum_eval!, dparams_dxi_lorentz, lorentzsum_jacobian, lorentzsum_xderiv
export lorentzsum_jacobian!, lorentzsum_jacobian_xderiv, lphi_grad_ab, lphi, lphi_grad_xi
export gscalar, gderiv_scalar

export get_extrema_bounded, compute_minimax_grid, merge_params, split_params
export get_extrema_unbounded, bound_extrema, is_R_inf, _grd1inf

export upgrade_gridsize, expand_grid, write_grid, shrink_grid

export MinimaxGrid
export _freqgrd1inf_even

include("MinimaxGridDef.jl")
using .MinimaxGridDef

include("Autocode.jl")

include("ExpAnsatz.jl")

include("ExpEval.jl")
include("LorentzEval.jl")
include("Data.jl")
include("Optimizer.jl")

# @setup_workload begin
#     @compile_workload begin
#         _grd1inf_db64 = MinimaxGrid{Double64}(
#             1,
#             Double64[1.429099786992801201936113178917134717837857368242504409833627049415743293428176],
#             Double64[0.4464926063120620475458103732318273156556442859543304312839572069496613108979231],
#             Double64[1.190833867525255986065341187399559547919090401090339913338830208954696366557289, 3.77464489561316288484522127216815684440847506987691834099433302905659882508515],
#             Double64[1.0, 1.923202293913185771796446096564028033888656932426095804484079188694164842420845, 8.667029155714755826822433950538620167883090437927102038645594255212853917909915],
#             0.08556407558597321956795594652164825694856719913998882342328298090355234124180788,
#             Inf
#         )
#         compute_minimax_grid(1,
#             merge_params(1, _grd1inf_db64.coefs, _grd1inf_db64.gridpts),
#             _grd1inf_db64.interp_pts,
#             10.0
#         )
#         shrink_grid(_grd1inf_db64, 4.0, 1.1)
#         shrink_grid(convert(MinimaxGrid{Float64}, _grd1inf_db64), 4.0, 1.1)
#     end
# end


end
