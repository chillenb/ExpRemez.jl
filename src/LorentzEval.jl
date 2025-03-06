
lorentz_eval_err(grd::MinimaxGrid{T}, x) where {T} = begin
  n = grd.n
  params = merge_params(n, grd.coefs, grd.gridpts)
  return abs(gscalar(x, params))
end


function lorentzsum_eval(coeffs, freqs, x)
  return vec(sum(coeffs' .* (x ./ (x.^2 .+ freqs'.^2)).^2, dims=2))
end


function lorentzsum_eval!(coeffs, freqs, x, out)
  out .= vec(sum(coeffs' .* (x ./ (x.^2 .+ freqs'.^2)).^2, dims=2))
end


gscalar(x, params) = begin
  n = length(params) ÷ 2
  @views coeffs, freqs = params[1:n], params[n+1:end]
  return 1 / x - sum(coeffs' .* (x ./ (x ^ 2 .+ freqs' .^ 2)).^2)
end

gderiv_scalar(x, params) = begin
  n = length(params) ÷ 2
  @views coeffs, freqs = params[1:n], params[n+1:end]
  return -1 / (x^2) + 2*x*sum(coeffs' .* (x ^ 2 .- freqs' .^ 2) ./ (x .^ 2 .+ freqs' .^ 2).^3)
end


lorentzsum_jacobian!(n::Int64, params, x, J) = begin
  coeffs, freqs = params[1:n], params[n+1:end]
  lij = x.^2 .+ freqs'.^2
  J[:, 1:n] .= -x.^2 ./ lij.^2
  J[:, n+1:end] .= 4 .* coeffs' .* x .* freqs'.^2 ./ lij.^3
end


lorentzsum_jacobian(n::Int64, params, x) = begin
  np = length(x)
  J = similar(x, np, 2 * n)
  lorentzsum_jacobian!(n, params, x, J)
  return J
end


lorentzsum_jacobian_xderiv!(n::Int64, params, x, J) = begin
  coeffs, freqs = params[1:n], params[n+1:end]
  lij = x.^2 .+ freqs'.^2
  J[:, 1:n] .= 2 .* x .* (x.^2 .- freqs'.^2) ./ lij.^3
  J[:, n+1:end] .= 8 .* coeffs' .* freqs' .* x .* (freqs'.^2 .- 2 .* x.^2) ./ lij.^4
  return J
end

lorentzsum_jacobian_xderiv(n::Int64, params, x) = begin
  np = length(x)
  J = similar(x, np, 2 * n)
  lorentzsum_jacobian_xderiv!(n, params, x, J)
  return J
end

lorentzsum_xderiv(n::Int64, params, x) = begin
  coeffs, freqs = params[1:n], params[n+1:end]
  return vec(sum(2 .* coeffs' .* x .* (freqs' .- x) .* (freqs' .+ x) ./ (x.^2 .+ freqs'.^2).^3, dims=2))
end


function lphi_grad_ab(n::Int64, params::Array{T}, μ::Array{T}) where {T}
    ∇F = lorentzsum_jacobian(n, params, μ)
    @assert length(μ) == 2 * n + 1
    ∇ϕ = ∇F[1:2*n, :] .+ ∇F[2:2*n+1, :]
    return ∇ϕ
end

function lphi_odd(n::Int64, params::Array{T}, μ::Array{T}) where {T}
    @assert length(μ) == 2 * n + 1
    fvals = 1 ./ μ .- lorentzsum_eval(params[1:n], params[n+1:end], μ)
    ϕ = fvals[1:end-1] .+ fvals[2:end]
    return ϕ
end

function lphi_grad_xi(n, params, μ, ξ)
    lphi_g_ab = lphi_grad_ab(n, params, μ)
    g_g_ab = dparams_dxi_lorentz(n, params, ξ)
    return lphi_g_ab * g_g_ab
end


function dparams_dxi_lorentz(n::Int64, params::Array{T}, ξ::Array{T}) where {T}
    ∇G_params = lorentzsum_jacobian(n, params, ξ)
    npts = length(ξ)
    ∇G_ξ = T.(zeros(npts, npts))
    de_dξ = lorentzsum_xderiv(n, params, ξ)
    for i = 1:2*n
        ∇G_ξ[i, i] = -(1 / ξ[i])^2 - de_dξ[i]
    end
    return -∇G_params \ ∇G_ξ
end
