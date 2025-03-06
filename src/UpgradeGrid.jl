
function upgrade_freq_gridsize_guess(coefs, gridpts, interp_pts)
  n = length(coefs)
  m = length(interp_pts)

  if n > 1
      b_ratio = gridpts[end] / gridpts[end-1]
      a_ratio = coefs[end] / coefs[end-1]
      xi_ratio = interp_pts[end] / interp_pts[end-1]
  else
      b_ratio = 2
      a_ratio = 2
      xi_ratio = 2
  end

  a0 = a_ratio * coefs[end]
  b0 = b_ratio * gridpts[end]

  ptsnew = log.((m+1:-1:2))./log(m+1) .* interp_pts
  ptsnew = [ptsnew; xi_ratio * ptsnew[end]; xi_ratio^2 * ptsnew[end]]

  
  coefs = [coefs; a0] .^ ((n+1)/(n+3))
  gridpts = [gridpts; b0] .^ ((n+1)/(n+3))
  return coefs, gridpts, ptsnew
end

function upgrade_time_gridsize_guess(coefs, gridpts, interp_pts)
  n = length(coefs)
  if n > 1
      b_ratio = gridpts[1] / gridpts[2]
      a_ratio = coefs[1] / coefs[2]
      xi_ratio = interp_pts[end] / interp_pts[end-1]
  else
      b_ratio = 0.5
      a_ratio = 0.5
      xi_ratio = 2
  end

  a0 = a_ratio * coefs[1]
  b0 = b_ratio * gridpts[1]

  pts = [interp_pts; xi_ratio * interp_pts[end]; xi_ratio^2 * interp_pts[end]]
  coefs = [a0; coefs]
  exponents = [b0; gridpts]
return coefs, exponents, pts
end

"""
  upgrade_gridsize(grd::MinimaxGrid)

Try to create a new grid with a larger size from the existing grid.
Works best with R large.
"""
function upgrade_gridsize(grd::MinimaxGrid{T}; funcs, upgrade_guess,
  initialdamping=ParameterSchedulers.Sequence([0.01, 0.04, 0.125, 1.0], [10, 10, 5]),
  outersched=ParameterSchedulers.Sequence([0.125, 0.5, 1], [4, 4])
  ) where {T}
  conv_err = T(1)
  R = maximum(grd.extrema)

  coefs, gridpts, interp_pts = upgrade_guess(grd.coefs, grd.gridpts, grd.interp_pts)

  n = grd.n + 1
  tol = eps(T(1.0)) * 2 * n * 1e3
  iter = 1

  params = merge_params(n, coefs, gridpts)

  newton_interp!(n, params, interp_pts, sched=initialdamping; funcs=funcs)
  R = interp_pts[end]^2/interp_pts[end-1]

  # conv_err = T(1)
  # while conv_err > tol
  #     rate = T(outersched(iter))
  #     newton_interp!(n, params, interp_pts; funcs=funcs)
  #     mu = get_extrema_bounded(n, params, interp_pts, R; funcs=funcs)
  #     ph = funcs.alternant(n, params, mu)
  #     phg = funcs.alternant_grad_xi(n, params, mu, interp_pts)
  #     interp_pts = interp_pts .- rate .* (phg \ ph)
  #     conv_err = GenericLinearAlgebra.norm(ph)
  #     iter += 1
  # end


  mu = get_extrema_bounded(n, params, interp_pts, R; funcs=funcs)
  ph = funcs.alternant(n, params, mu)
  conv_err = GenericLinearAlgebra.norm(ph)

  while conv_err > tol
      stepsize = T(1)
      rate = T(outersched(iter))
      newton_interp!(n, params, interp_pts, funcs=funcs)
      mu = get_extrema_bounded(n, params, interp_pts, R; funcs=funcs)
      ph = funcs.alternant(n, params, mu)
      phg = funcs.alternant_grad_xi(n, params, mu, interp_pts)

      update_vec = phg \ ph
      linesearch_done = false
      phnorm = GenericLinearAlgebra.norm(ph)
      while !linesearch_done
        if stepsize < 1e-20
            error("Stepsize too small")
        end
        try
            interp_pts_new = interp_pts .- stepsize * update_vec
            params_new = copy(params)
            newton_interp!(n, params_new, interp_pts_new, funcs=funcs)
            mu_new = get_extrema_bounded(n, params_new, interp_pts_new, R; funcs=funcs)
            ph_new = funcs.alternant(n, params_new, mu_new)
            if GenericLinearAlgebra.norm(ph_new) < phnorm
                interp_pts .= interp_pts_new
                ph .= ph_new
                params .= params_new
                linesearch_done = true
            else
                println(GenericLinearAlgebra.norm(ph_new), " ", phnorm)
                println("Decreasing stepsize")
                stepsize /= 2
            end
        catch e
            println("Newton failed, decreasing stepsize")
            stepsize /= 2
        end
      end
      conv_err = GenericLinearAlgebra.norm(ph)
      iter += 1
    end

  @views coefs, gridpts = params[1:n], params[n+1:end]
  f = (x)->begin funcs.err_eval(coefs, gridpts, x) end

  abs_err = maximum(abs, f(mu))
  mu_unbounded = get_extrema_unbounded(n, params, interp_pts; funcs=funcs)
  mu = bound_extrema(n, mu_unbounded, R)
  abs_err = maximum(abs, f(mu))
  newgrd = MinimaxGrid(n, sort(coefs), sort(gridpts), sort(interp_pts), sort(mu_unbounded), abs_err, R)
  if is_R_inf(newgrd; funcs=funcs)
      newgrd.R = Inf
  end

  return newgrd
end