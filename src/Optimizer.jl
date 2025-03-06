
struct StepSizeException <: Exception
    var :: String
end
struct MaxIterException <: Exception
    var :: String
end


"""
    newton_interp_by_expsum(n, params, ξ)

Newton's method to interpolate the exponential sum function at points `ξ` by finding appropriate parameters `params`.
"""
function newton_interp!(n::Int64, params::Array{T}, ξ::Array{T};
    funcs,
    maxdiff=T(1.1),
    maxiter=20,
    tol=T(0.0),
    stepsize_min=T(1e-10),
    verbose=false) where {T}

    if tol == T(0.0)
        tol = n * eps(T(1.0)) * 1e3
    end

    iter = 1
    ∇F = similar(ξ, 2 * n, 2 * n)
    dp = similar(ξ, 2 * n)
    new_params = similar(ξ, 2 * n)
    new_resid = similar(ξ, 2 * n)
    while true
        @views resid = funcs.err_eval(params[1:n], params[n+1:end], ξ)

        err = GenericLinearAlgebra.norm(resid)

        if err < tol
            break
        end



        funcs.jacobian!(n, params, ξ, ∇F)
        dp .= ∇F \ resid
        ratio = maximum((dp ./ params) ./ maxdiff)
        ratio = max(1, ratio)
        ratio_stepsize = min(1, 1/ratio)

        if iter > maxiter
            throw(MaxIterException("Maxiter $maxiter reached; err $err;"))
        end

        new_err = err + 1
        stepsize = T(1)
        while new_err >= err
            #println("Inner stepsize: $stepsize")
            new_params .= params .- ratio_stepsize * stepsize * dp
            if any(new_params .< 0)
                stepsize /= 2
                continue
            end
            new_resid .= funcs.err_eval(new_params[1:n], new_params[n+1:end], ξ)
            new_err = GenericLinearAlgebra.norm(new_resid)
            stepsize /= 2
            if stepsize < stepsize_min
                throw(StepSizeException("Stepsize too small"))
            end
        end
        #params .-= ratio_stepsize * rate * dp
        params .= new_params

        iter += 1
    end
    if verbose
        println("inner $iter")
    end
end

newton_interp_by_expsum! = newton_interp!

"""
    newton_interp_by_expsum(n, params, ξ)

Newton's method to interpolate the exponential sum function at points `ξ` by finding appropriate parameters `params`.
"""
function newton_interp(n::Int64, params::Array{T}, ξ::Array{T};kws...) where {T}
    params_copy = copy(params)
    newton_interp!(n, params_copy, ξ; kws...)
    return params_copy
end

newton_interp_by_expsum = newton_interp

"""
    get_extrema_unbounded(n, params, ξ, R)

Compute the extrema of the the error on the intervals [1, ξ[1]], [ξ[1], ξ[2]], ..., [ξ[2n-1], ξ[2n]], [ξ[2n], Inf].
"""
function get_extrema_unbounded(n::Int64, params::Array{T}, ξ::Array{T}; abs_tol=0.0,
    funcs) where {T}
    if abs_tol != 0.0
        tol = T(abs_tol)
    else
        tol = eps(T(1.0)) * 1e3
    end
    pts = [T(1); ξ]
    @views coeffs, exponents = params[1:n], params[n+1:end]
    fder(x) = funcs.errderiv_eval(coeffs, exponents, x)[1]
    f(x) = funcs.err_eval(coeffs, exponents, x)[1]
    fder_compat(x,p) = funcs.errderiv_eval(coeffs, exponents, x)[1]

    fder_function = NonlinearFunction(fder_compat)
    μ = T.(zeros(2 * n + 1))
    for i = 1:2*n
        if sign(fder(pts[i])) != sign(fder(pts[i+1]))
            prob = IntervalNonlinearProblem(fder_function, (pts[i], pts[i+1]))
            μ[i] = solve(prob, abstol=T(tol), Brent()).u
        else
            μ[i] = (abs(f(pts[i])) > abs(f(pts[i+1]))) ? pts[i] : pts[i+1]
        end
    end
    deriv_last_point = fder(pts[2*n+1])
    if sign(deriv_last_point) < 0
        # There's no extremum afterwards
        μ[2*n+1] = Inf
    else
        # Step 1. Find a point where the derivative is negative
        c = pts[2*n+1] * 10
        while sign(fder(c)) >= 0
            c *= 10
        end
        # Step 2. Find the extremum
        prob = IntervalNonlinearProblem(fder_function, (pts[2*n+1], c))
        μ[2*n+1] = solve(prob, abstol=T(tol)).u
    end
    return μ
end

function bound_extrema(n, μ, R)
    return min.(μ, R)
end

function bound_extrema!(n, μ, R)
    μ .= min.(μ, R)
end

"""
    get_extrema(n, params, ξ, R)

Compute the extrema of the the error on the intervals [1, ξ[1]], [ξ[1], ξ[2]], ..., [ξ[2n-1], ξ[2n]], [ξ[2n], R].
"""
function get_extrema_bounded(n::Int64, params::Array{T}, ξ::Array{T}, R; abs_tol=0.0, kws...) where {T}
    μ = get_extrema_unbounded(n, params, ξ; abs_tol=abs_tol, kws...)
    bound_extrema!(n, μ, R)
    return μ
end

"""
    is_R_inf(grd::MinimaxGrid)

Check if the grid has R = Inf. Such a grid has the equioscillation property at the
    very last extremum.
"""
function is_R_inf(grd::MinimaxGrid{T}; funcs) where {T}
    μ = grd.extrema
    fvals_μ = funcs.err_eval(grd.coefs, grd.gridpts, μ)
    fvals_abs = abs.(fvals_μ)
    err_ratio = fvals_abs[end] / fvals_abs[end-1]
    if err_ratio > 1.01 || err_ratio < 0.99
        return false
    end
    return true
end



function compute_minimax_grid(start::MinimaxGrid{T}, R;
    # outersched=ParameterSchedulers.Constant(T(1)),
    # innersched=ParameterSchedulers.Constant(T(1)),
    tol=0.0,
    tol_inner=0.0,
    maxiter_inner=Inf,
    maxiter=Inf,
    verbose=false,
    funcs) where {T}
    tol = T(tol)
    tol_inner = T(tol_inner)

    R = T(R)
    n = start.n
    params = merge_params(start.n, start.coefs, start.gridpts)
    pts = copy(start.interp_pts)
    if tol == T(0.0)
        tol = eps(T(1.0)) * 2 * n * 1e3
    end
    iter = 1
    newton_interp!(n, params, pts, maxiter=maxiter_inner, tol=tol_inner,
    funcs=funcs)
    mu = get_extrema_bounded(n, params, pts, R; funcs=funcs)
    ph = funcs.alternant(n, params, mu)
    conv_err = GenericLinearAlgebra.norm(ph)

    while conv_err > tol && iter <= maxiter
        stepsize = T(1)
        mu = get_extrema_bounded(n, params, pts, R; funcs=funcs)
        ph = funcs.alternant(n, params, mu)
        phg = funcs.alternant_grad_xi(n, params, mu, pts)
        update_vec = phg \ ph
        linesearch_done = false
        phnorm = GenericLinearAlgebra.norm(ph)
        while !linesearch_done
            if stepsize < 1e-10
                raise(StepSizeException("Stepsize too small"))
            end
            try
                pts_new = pts .- stepsize * update_vec
                params_new = copy(params)
                newton_interp!(n, params_new, pts_new, tol=tol_inner,
                    funcs=funcs)
                mu_new = get_extrema_bounded(n, params_new, pts_new, R; funcs=funcs)
                ph_new = funcs.alternant(n, params_new, mu_new)
                if GenericLinearAlgebra.norm(ph_new) < phnorm
                    pts .= pts_new
                    ph .= ph_new
                    params .= params_new
                    linesearch_done = true
                else
                    if verbose
                        println(GenericLinearAlgebra.norm(ph_new), " ", phnorm)
                        println("Decreasing stepsize")
                    end
                    stepsize /= 2
                end
            catch e
                if verbose
                    println("Newton failed, decreasing stepsize")
                end
                stepsize /= 2
            end
        end
        conv_err = GenericLinearAlgebra.norm(ph)
        newton_interp!(n, params, pts, tol=tol_inner,
            funcs=funcs)
        iter += 1
    end
    if conv_err > tol
        error("Failed to converge")
    end
    @views coefs, exponents = params[1:n], params[n+1:end]
    f = (x)->begin funcs.err_eval(coefs, exponents, x) end
    abs_err = maximum(abs, f(mu))
    mu_unbounded = get_extrema_unbounded(n, params, pts; funcs=funcs)
    mu = bound_extrema(n, mu_unbounded, R)
    abs_err = maximum(abs, f(mu))
    grd = MinimaxGrid(n, sort(coefs), sort(exponents), sort(pts), sort(mu_unbounded), abs_err, R)
    if is_R_inf(grd; funcs=funcs)
        grd.R = Inf
    end
    return grd
end

include("UpgradeGrid.jl")

# """
#     upgrade_gridsize(grd::MinimaxGrid)

# Try to create a new grid with a larger size from the existing grid.
# Works best with R large.
# """
# function upgrade_gridsize(grd::MinimaxGrid{T}, funcs) where {T}
#     conv_err = T(1)
#     R = maximum(grd.extrema)

#     if grd.n > 1
#         b_ratio = grd.gridpts[1] / grd.gridpts[2]
#         a_ratio = grd.coefs[1] / grd.coefs[2]
#         xi_ratio = grd.interp_pts[end] / grd.interp_pts[end-1]
#     else
#         b_ratio = 0.5
#         a_ratio = 0.5
#         xi_ratio = 2
#     end

#     a0 = max(a_ratio * grd.coefs[1], grd.err)
#     b0 = min(b_ratio * grd.gridpts[1])

#     pts = [grd.interp_pts; xi_ratio * grd.interp_pts[end]; xi_ratio^2 * grd.interp_pts[end]]
#     coefs = [a0; grd.coefs]
#     exponents = [b0; grd.gridpts]
#     n = grd.n + 1
#     tol = eps(T(1.0)) * 2 * n * 1e3
#     iter = 1

#     params = merge_params(n, coefs, exponents)
#     rates = [0.125, T(1)]
#     times = [5]
#     heavydamping = ParameterSchedulers.Sequence(rates, times)
#     newton_interp!(n, params, pts, sched=heavydamping; funcs=funcs)
#     R = xi_ratio * pts[end]

#     outersched = ParameterSchedulers.Sequence(
#         [0.05, 0.125, 0.5, 1],
#         [2, 4, 2]
#     )

#     while conv_err > tol
#         rate = T(outersched(iter))
#         newton_interp!(n, params, pts; funcs=funcs, maxiter=10)
#         mu = get_extrema_bounded(n, params, pts, R; funcs=funcs)
#         ph = funcs.alternant(n, params, mu)
#         phg = funcs.alternant_grad_xi(n, params, mu, pts)
#         pts = pts .- rate .* (phg \ ph)
#         conv_err = GenericLinearAlgebra.norm(ph)
#         iter += 1
#     end

#     conv_err = T(1)
#     while conv_err > tol
#         rate = T(outersched(iter))
#         newton_interp!(n, params, pts; funcs=funcs, maxiter=10)
#         mu = get_extrema_bounded(n, params, pts, R; funcs=funcs)
#         ph = funcs.alternant(n, params, mu)
#         phg = funcs.alternant_grad_xi(n, params, mu, pts)
#         pts = pts .- rate .* (phg \ ph)
#         conv_err = GenericLinearAlgebra.norm(ph)
#         iter += 1
#     end

#     @views coefs, exponents = params[1:n], params[n+1:end]
#     f = (x)->begin funcs.err_eval(coefs, exponents, x) end

#     abs_err = maximum(abs, f(mu))
#     mu_unbounded = get_extrema_unbounded(n, params, pts; funcs=funcs)
#     mu = bound_extrema(n, mu_unbounded, R)
#     abs_err = maximum(abs, f(mu))
#     newgrd = MinimaxGrid(n, sort(coefs), sort(exponents), sort(pts), sort(mu_unbounded), abs_err, R)
#     if is_R_inf(newgrd; funcs=funcs)
#         newgrd.R = Inf
#     end
#     return newgrd
# end

"""
    expand_grid(grd::MinimaxGrid, R, expand_ratio)

Continuation method for increasing the R of a grid.
"""
function expand_grid(grd::MinimaxGrid{T}, R, expand_ratio; funcs) where {T}
    # If the grid already has R = Inf, return it as is
    if isinf(grd.R)
        return grd
    end

    expand_ratio = T(expand_ratio)
    R0 = grd.R
    n = grd.n
    @assert R > R0
    nsteps = ceil(Int, log(R / R0) / log(expand_ratio))
    Rs = logrange(R0, R, nsteps)[2:end]
    coefs, exponents, pts, mu = grd.coefs, grd.gridpts, grd.interp_pts, grd.extrema
    params = merge_params(n, coefs, exponents)
    newgrd = grd
    for Ri in Rs
        newgrd = compute_minimax_grid(grd, Ri; funcs=funcs)
        if isinf(newgrd.R)
            break
        end
    end
    return newgrd
end

function shrink_grid(grd::MinimaxGrid{T}, R, shrink_ratio; funcs,verbose=false, start_fp64=true) where {T}
    # If the grid already has R = Inf, return it as is
    R0 = isinf(grd.R) ? grd.extrema[end] : grd.R
    @assert R < R0
    shrink_ratio_orig = T(min(shrink_ratio, R0 / R))
    shrink_ratio = shrink_ratio_orig


    n = grd.n


    newgrd = deepcopy(grd)
    if start_fp64
        newgrd = convert(MinimaxGrid{Float64}, newgrd)
        K = Float64
    else
        K = T
    end

    cur_R = R0
    err = newgrd.err
    tol_fac = 0.2
    while cur_R > R
        if verbose
            println("R: ", Float64(cur_R))
        end
        conv = false
        while !conv
            try
                tol = tol_fac * err
                next_R = max(cur_R / shrink_ratio, R)
                next_R = K(next_R)

                if verbose
                    println("Tolerance: ", tol)
                    println("Shrinking ratio: ", Float64(shrink_ratio))
                end

                newgrd.interp_pts .= (newgrd.interp_pts .- 1) ./ K(shrink_ratio) .+ 1
                maxiter_inner = (K == Float64) ? 30 : 15
                newgrd = compute_minimax_grid(newgrd, K(next_R), tol=K(tol), tol_inner=K(0.5 * tol), maxiter_inner=maxiter_inner, maxiter=4; funcs=funcs)
                @assert !isinf(newgrd.err)
                conv = true
                cur_R = next_R
                err = newgrd.err
            catch e
                if shrink_ratio < (1.0 + 1e-6)
                    if verbose
                        print(e)
                    end
                    if K == Float64
                        shrink_ratio = shrink_ratio_orig
                        K = T
                        newgrd = convert(MinimaxGrid{T}, newgrd)
                    else
                        throw("Shrink ratio is too small, giving up")
                    end
                end
                shrink_ratio = 1 + (shrink_ratio - 1) / K(1.1)
            end
        end
    end
    if start_fp64
        newgrd = convert(MinimaxGrid{T}, newgrd)
    end
    newgrd = compute_minimax_grid(newgrd, R, funcs=funcs)
    return newgrd, K
end