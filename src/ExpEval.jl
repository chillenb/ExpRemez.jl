exp_eval_err(grd::MinimaxGrid{T}, x) where {T} = begin
  n = grd.n
  params = merge_params(n, grd.coefs, grd.gridpts)
  return abs(fscalar(x, params))
end


"""
  expsum_eval(coeffs, exponents, x)

Evaluate the exponential sum function at points `x` with coefficients `coeffs` and exponents `exponents`.
  e(j) = ∑_{i=1}^{n} coeffs[i] * exp(-x[j] * exponents[i])
"""
function expsum_eval(coeffs, exponents, x)
  return vec(sum(coeffs' .* exp.(-x .* exponents'), dims=2))
end


"""
  expsum_eval(coeffs, exponents, x)

Evaluate the exponential sum function at points `x` with coefficients `coeffs` and exponents `exponents`.
  e(j) = ∑_{i=1}^{n} coeffs[i] * exp(-x[j] * exponents[i])
"""
function expsum_eval!(coeffs, exponents, x, out)
  out .= vec(sum(coeffs' .* exp.(-x .* exponents'), dims=2))
end

"""
  merge_params(n, coeffs, gridpts)

Merge the coefficients and exponents into a single array.
"""
function merge_params(n, coeffs::Array{T}, gridpts::Array{T}) where {T}
  @assert n == length(gridpts)
  @assert n == length(coeffs)
  params = Array{T}(undef, 2n)
  params[1:n] .= coeffs
  params[n+1:end] .= gridpts
  return params
end

function split_params(n, params::Array{T}) where {T}
  @assert 2 * n == length(params)
  @views coeffs = params[1:n]
  @views exponents = params[n+1:end]
  return coeffs, exponents
end

expsum_eval(n::Int64, params::Array{T}, x::Array{T}) where {T} = begin
  @views coeffs, exponents = params[1:n], params[n+1:end]
  return expsum_eval(coeffs, exponents, x)
end

fscalar(x, params) = begin
  n = length(params) ÷ 2
  @views coeffs, exponents = params[1:n], params[n+1:end]
  return 1 ./ x - sum(coeffs .* exp.(-x .* exponents))
end

fderiv_scalar(x, params) = begin
  n = length(params) ÷ 2
  @views coeffs, exponents = params[1:n], params[n+1:end]
  return -1 ./ (x^2) + sum(coeffs .* exp.(-x .* exponents) .* exponents)
end

"""
  expsum_jacobian(n, params, x)

Compute the Jacobian of the exponential sum function at points `x` with respect to the parameters `params`.
"""
expsum_jacobian!(n::Int64, params, x, J) = begin
  @views coeffs, exponents = params[1:n], params[n+1:end]
  expij = -exp.(-exponents' .* x)
  J[:, 1:n] .= expij
  J[:, n+1:end] .= -(coeffs' .* x) .* expij
end

"""
  expsum_jacobian(n, params, x)

Compute the Jacobian of the exponential sum function at points `x` with respect to the parameters `params`.
"""
expsum_jacobian(n::Int64, params, x) = begin
  np = length(x)
  J = similar(x, np, 2 * n)
  expsum_jacobian!(n, params, x, J)
  return J
end

"""
  expsum_jacobian_xderiv(n, params, x)

Compute d/dx[expsum_jacobian(n, params, x)].
"""
expsum_jacobian_xderiv!(n::Int64, params, x, J) = begin
  @views coeffs, exponents = params[1:n], params[n+1:end]
  expij = exp.(-exponents' .* x)
  J[:, 1:n] .= exponents' .* expij
  J[:, n+1:end] .= coeffs' .* (1 .- exponents' .* x) .* expij
  return J
end

"""
  expsum_jacobian_xderiv(n, params, x)

Compute d/dx[expsum_jacobian(n, params, x)].
"""
expsum_jacobian_xderiv(n::Int64, params, x) = begin
  np = length(x)
  J = similar(x, np, 2 * n)
  expsum_jacobian_xderiv!(n, params, x, J)
  return J
end

"""
  expsum_xderiv(n, params, x)

Compute d/dx[expsum_eval(n, params, x)] at the points x.
Formula:
  d/dx[e(j)] = - ∑_{i=1}^{n} coeffs[i] * exponents[i] exp(-x[j] * exponents[i]))
"""
expsum_xderiv(n::Int64, params, x) = begin
  @views coeffs, exponents = params[1:n], params[n+1:end]
  expij = exp.(-exponents' .* x) .* (exponents .* coeffs)'
  return vec(sum(-expij, dims=2))
end

"""
    phi_grad_ab(n, params, μ)

Compute the gradient of the alternants phi(n, params, μ) with respect to the parameters.
"""
function phi_grad_ab(n::Int64, params::Array{T}, μ::Array{T}) where {T}
    ∇F = expsum_jacobian(n, params, μ)
    @assert length(μ) == 2 * n + 1
    @views ∇ϕ = ∇F[1:2*n, :] + ∇F[2:2*n+1, :]
    return ∇ϕ
end

function phi(n::Int64, params::Array{T}, μ::Array{T}) where {T}
    @assert length(μ) == 2 * n + 1
    fvals = 1 ./ μ .- expsum_eval(n, params, μ)
    @views return fvals[1:end-1] .+ fvals[2:end]
end

function phi_grad_xi(n, params, μ, ξ)
    phi_g_ab = phi_grad_ab(n, params, μ)
    f_g_ab = dparams_dxi(n, params, ξ)
    return phi_g_ab * f_g_ab
end

"""
    dparams_dxi(n, params, ξ)

Compute the derivative of the parameters with respect to ξ.
"""
function dparams_dxi(n::Int64, params::Array{T}, ξ::Array{T}) where {T}
    ∇F_params = expsum_jacobian(n, params, ξ)
    npts = length(ξ)
    ∇F_ξ = T.(zeros(npts, npts))
    de_dξ = expsum_xderiv(n, params, ξ)
    for i = 1:2*n
        ∇F_ξ[i, i] = -(1 / ξ[i])^2 - de_dξ[i]
    end
    return -∇F_params \ ∇F_ξ
end