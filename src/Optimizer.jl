
"""
    newton_interp_by_expsum(n, params, ξ)

Newton's method to interpolate the exponential sum function at points `ξ` by finding appropriate parameters `params`.
"""
function newton_interp_by_expsum!(n::Int64, params::Array{T}, ξ::Array{T};
    sched=ParameterSchedulers.Constant(T(1)),
    maxdiff=T(1.1),
    maxiter=Inf,
    tol=T(0.0)) where {T}

    if tol == T(0.0)
        tol = eps(T(1.0)) * 1e3
    end

    iter = 1
    ∇F = similar(ξ, 2 * n, 2 * n)
    resid = similar(ξ, 2 * n)
    dp = similar(ξ, 2 * n)
    while true
        rate = T(sched(iter))
        expsum_eval!(params[1:n], params[n+1:end], ξ, resid)
        resid .= resid .- 1 ./ ξ

        if maximum(abs.(resid)) < tol
            break
        end
        if iter > maxiter
            error("Failed to converge")
        end

        expsum_jacobian!(n, params, ξ, ∇F)
        dp .= ∇F \ resid
        ratio = maximum(abs.(dp ./ params) ./ maxdiff)
        if ratio > 1
            rate /= ratio
        end
        params .+= rate * dp

        iter += 1
    end
end

"""
    newton_interp_by_expsum(n, params, ξ)

Newton's method to interpolate the exponential sum function at points `ξ` by finding appropriate parameters `params`.
"""
function newton_interp_by_expsum(n::Int64, params::Array{T}, ξ::Array{T};
    sched=ParameterSchedulers.Constant(T(1)),
    maxdiff=T(1.1),
    maxiter=Inf,
    tol=T(0.0)) where {T}
    params_copy = copy(params)
    newton_interp_by_expsum!(n, params_copy, ξ, sched=sched, maxdiff=maxdiff, maxiter=maxiter, tol=tol)
    return params_copy
end

"""
    get_extrema_unbounded(n, params, ξ, R)

Compute the extrema of the the error on the intervals [1, ξ[1]], [ξ[1], ξ[2]], ..., [ξ[2n-1], ξ[2n]], [ξ[2n], Inf].
"""
function get_extrema_unbounded(n::Int64, params::Array{T}, ξ::Array{T}; abs_tol=0.0) where {T}
    if abs_tol != 0.0
        tol = T(abs_tol)
    else
        tol = eps(T(1.0)) * 1e3
    end
    pts = [T(1); ξ]
    feval = NonlinearFunction(fderiv_scalar)
    μ = T.(zeros(2 * n + 1))
    for i = 1:2*n
        if sign(fderiv_scalar(pts[i], params)) != sign(fderiv_scalar(pts[i+1], params))
            prob = IntervalNonlinearProblem(feval, (pts[i], pts[i+1]), params)
            μ[i] = solve(prob, abstol=T(tol), Brent()).u
        else
            μ[i] = (abs(fscalar(pts[i], params)) > abs(fscalar(pts[i+1], params))) ? pts[i] : pts[i+1]
        end
    end
    deriv_last_point = fderiv_scalar(pts[2*n+1], params)
    if sign(deriv_last_point) < 0
        # There's no extremum afterwards
        μ[2*n+1] = Inf
    else
        # Step 1. Find a point where the derivative is negative
        c = pts[2*n+1] * 10
        while sign(fderiv_scalar(c, params)) >= 0
            c *= 10
        end
        # Step 2. Find the extremum
        prob = IntervalNonlinearProblem(feval, (pts[2*n+1], c), params)
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
function get_extrema_bounded(n::Int64, params::Array{T}, ξ::Array{T}, R; abs_tol=0.0) where {T}
    μ = get_extrema_unbounded(n, params, ξ; abs_tol=abs_tol)
    bound_extrema!(n, μ, R)
    return μ
end

"""
    is_R_inf(grd::MinimaxGrid)

Check if the grid has R = Inf. Such a grid has the equioscillation property at the
    very last extremum.
"""
function is_R_inf(grd::MinimaxGrid{T}) where {T}
    μ = grd.extrema
    fvals_μ = fscalar.(μ, Ref(merge_params(grd.n, grd.coefs, grd.exponents)))
    fvals_abs = abs.(fvals_μ)
    err_ratio = fvals_abs[end] / fvals_abs[end-1]
    if err_ratio > 1.01 || err_ratio < 0.99
        return false
    end
    return true
end



function compute_minimax_grid(n, params::Array{T}, pts::Array{T}, R;
    outersched=ParameterSchedulers.Constant(T(1)),
    innersched=ParameterSchedulers.Constant(T(1)),
    tol=0.0,
    tol_inner=0.0,
    maxiter_inner=Inf,
    maxiter=Inf) where {T}
    tol = T(tol)
    tol_inner = T(tol_inner)
    conv_err = T(1)
    R = T(R)
    if tol == T(0.0)
        tol = eps(T(1.0)) * 2 * n * 1e3
    end
    iter = 1
    while conv_err > tol && iter <= maxiter
        rate = T(outersched(iter))
        newton_interp_by_expsum!(n, params, pts, sched=innersched, maxiter=maxiter_inner, tol=tol_inner)
        mu = get_extrema_bounded(n, params, pts, R)
        ph = phi(n, params, mu)
        phg = phi_grad_xi(n, params, mu, pts)
        pts = pts .- rate .* (phg \ ph)
        conv_err = GenericLinearAlgebra.norm(ph)
        iter += 1
    end
    if conv_err > tol
        error("Failed to converge")
    end
    coefs, exponents = params[1:n], params[n+1:end]
    abs_err = maximum(abs, fscalar.(mu, Ref(params)))
    mu_unbounded = get_extrema_unbounded(n, params, pts)
    mu = bound_extrema(n, mu_unbounded, R)
    abs_err = maximum(abs, fscalar.(mu, Ref(params)))
    grd = MinimaxGrid(n, sort(coefs), sort(exponents), sort(pts), sort(mu_unbounded), abs_err, R)
    if is_R_inf(grd)
        grd.R = Inf
    end
    return grd
end

"""
    upgrade_gridsize(grd::MinimaxGrid)

Try to create a new grid with a larger size from the existing grid.
Works best with R large.
"""
function upgrade_gridsize(grd::MinimaxGrid{T}) where {T}
    conv_err = T(1)
    R = maximum(grd.extrema)

    if grd.n > 1
        b_ratio = grd.exponents[1] / grd.exponents[2]
        a_ratio = grd.coefs[1] / grd.coefs[2]
        xi_ratio = grd.interp_pts[end] / grd.interp_pts[end-1]
    else
        b_ratio = 0.5
        a_ratio = 0.5
        xi_ratio = 2
    end

    a0 = max(a_ratio * grd.coefs[1], grd.err)
    b0 = min(b_ratio * grd.exponents[1])

    pts = [grd.interp_pts; xi_ratio * grd.interp_pts[end]; xi_ratio^2 * grd.interp_pts[end]]
    coefs = [a0; grd.coefs]
    exponents = [b0; grd.exponents]
    n = grd.n + 1
    tol = eps(T(1.0)) * 2 * n * 1e3
    iter = 1

    params = merge_params(n, coefs, exponents)
    rates = [0.125, T(1)]
    times = [5]
    heavydamping = ParameterSchedulers.Sequence(rates, times)
    newton_interp_by_expsum!(n, params, pts, sched=heavydamping)
    R = xi_ratio * pts[end]

    outersched = ParameterSchedulers.Sequence(
        [0.05, 0.125, 0.5, 1],
        [2, 4, 2]
    )

    while conv_err > tol
        rate = T(outersched(iter))
        newton_interp_by_expsum!(n, params, pts; maxiter=10)
        mu = get_extrema_bounded(n, params, pts, R)
        ph = phi(n, params, mu)
        phg = phi_grad_xi(n, params, mu, pts)
        pts = pts .- rate .* (phg \ ph)
        conv_err = GenericLinearAlgebra.norm(ph)
        iter += 1
    end

    conv_err = T(1)
    while conv_err > tol
        rate = T(outersched(iter))
        newton_interp_by_expsum!(n, params, pts)
        mu = get_extrema_bounded(n, params, pts, R)
        ph = phi(n, params, mu)
        phg = phi_grad_xi(n, params, mu, pts)
        pts = pts .- rate .* (phg \ ph)
        conv_err = maximum(abs, ph)
        iter += 1
    end

    coefs, exponents = params[1:n], params[n+1:end]
    abs_err = maximum(abs, fscalar.(mu, Ref(params)))
    mu_unbounded = get_extrema_unbounded(n, params, pts)
    mu = bound_extrema(n, mu_unbounded, R)
    abs_err = maximum(abs, fscalar.(mu, Ref(params)))
    newgrd = MinimaxGrid(n, sort(coefs), sort(exponents), sort(pts), sort(mu_unbounded), abs_err, R)
    if is_R_inf(newgrd)
        newgrd.R = Inf
    end
    return newgrd
end

"""
    expand_grid(grd::MinimaxGrid, R, expand_ratio)

Continuation method for increasing the R of a grid.
"""
function expand_grid(grd::MinimaxGrid{T}, R, expand_ratio) where {T}
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
    coefs, exponents, pts, mu = grd.coefs, grd.exponents, grd.interp_pts, grd.extrema
    params = merge_params(n, coefs, exponents)
    newgrd = nothing
    for Ri in Rs
        newgrd = compute_minimax_grid(n, params, pts, Ri)
        if isinf(newgrd.R)
            break
        end
        params = merge_params(n, newgrd.coefs, newgrd.exponents)
        pts = newgrd.interp_pts
    end
    return newgrd
end

function shrink_grid(grd::MinimaxGrid{T}, R, shrink_ratio; verbose=false, start_fp64=true) where {T}
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
    params = merge_params(n, newgrd.coefs, newgrd.exponents)


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

                params = merge_params(n, newgrd.coefs, newgrd.exponents)
                newpts = (newgrd.interp_pts .- 1) ./ K(shrink_ratio) .+ 1
                maxiter_inner = (K == Float64) ? 30 : 15
                newgrd = compute_minimax_grid(n, params, newpts, K(next_R), tol=K(tol), tol_inner=K(0.5 * tol), maxiter_inner=maxiter_inner, maxiter=4)
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
    params = merge_params(n, newgrd.coefs, newgrd.exponents)
    newgrd = compute_minimax_grid(n, params, newgrd.interp_pts, R,
        innersched=ParameterSchedulers.Sequence([0.1, 1], [4]),
        outersched=ParameterSchedulers.Sequence([0.125, 1], [4])
    )
    return newgrd, K
end