"""
    phi_fdiff_grad_xi(n, params, ξ, R, dξ)

    Finite difference version of phi_grad_xi.
"""
function phi_fdiff_grad_xi(n, params, ξ, R, dξ)
    μ = get_extrema_bounded(n, params, ξ, R)
    f_of_μ = 1 ./ μ .- expsum_eval(n, params, μ)
    J = similar(ξ, length(μ), length(ξ))
    ξ_prime = similar(ξ)
    ξ_prime .= ξ
    @assert length(ξ) == 2*n
    for i = 1:length(ξ)
        ξ_prime[i] = ξ[i] + dξ
        params_prime = newton_interp_by_expsum(n, params, ξ_prime)
        @assert length(params_prime) == length(params)
        μ_prime = get_extrema_bounded(n, params_prime, ξ_prime, R)
        f_of_μ_prime = 1 ./ μ_prime .- expsum_eval(n, params_prime, μ_prime)
        @assert length(μ_prime) == length(μ)
        J[:, i] .= (f_of_μ_prime .- f_of_μ) ./ dξ
        ξ_prime[i] = ξ[i]
    end
    Jphi = similar(ξ, length(μ)-1, length(ξ))
    for j = 1:length(μ)-1
        Jphi[j, :] = J[j, :] .+ J[j+1, :]
    end
    return Jphi, μ
end

"""
    params_grad_xi(n, params, ξ, R, dξ)

    Compute the gradient of the parameters with respect to ξ.
    Finite difference version of dparams_dxi.
"""
function params_grad_xi(n, params, ξ, R, dξ)
    params = newton_interp_by_expsum(n, params, ξ)
    μ = get_extrema_bounded(n, params, ξ, R)
    J = similar(ξ, length(params), length(ξ))
    @assert length(ξ) == 2*n
    ξ_prime = similar(ξ)
    ξ_prime .= ξ
    for i = 1:length(ξ)
        ξ_prime[i] = ξ[i] + dξ
        params_prime = newton_interp_by_expsum(n, params, ξ_prime)

        J[:, i] .= ((params_prime .- params) ./ dξ)
        ξ_prime[i] = ξ[i]
    end
    return J
end