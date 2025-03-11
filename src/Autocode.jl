
using Symbolics
using GenericLinearAlgebra

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


function gen_funs(targetfn, basicexpr, b, x)
  #targetfn is usually 1/x
  #basicf is usually exp(-b*x)

  ex_basic = Symbolics.build_function(basicexpr, b, x, expression=Val{true})
  ex_deriv_x = Symbolics.build_function(Symbolics.derivative(basicexpr, x, simplify=true), b, x, expression=Val{true})
  ex_deriv_b = Symbolics.build_function(Symbolics.derivative(basicexpr, b, simplify=true), b, x, expression=Val{true})


  func_eval = :((a,b,x) -> begin
          $ex_basic.(b',x)*a
        end)

  funcderiv_eval = :((a,b,x) -> begin
          $ex_deriv_x.(b',x)*a
        end)
  
  ex_target = Symbolics.build_function(targetfn, x, expression=Val{true})
  ex_target_deriv_x = Symbolics.build_function(Symbolics.derivative(targetfn, x, simplify=true), x, expression=Val{true})

  target_eval = :((x) -> begin
          $ex_target.(x)
        end)

  targetderiv_eval = :((x) -> begin
          $ex_target_deriv_x.(x)
        end)

  err_eval = :((a,b,x) -> begin
          $ex_target.(x) .- $ex_basic.(b',x)*a
        end)
  errderiv_eval = :((a,b,x) -> begin
          $ex_target_deriv_x.(x) .- $ex_deriv_x.(b',x)*a
        end)

  jacobian! = :((n, params, x, J) -> begin
          a = view(params, 1:n)
          b = view(params, n+1:2*n)
          J[:, 1:n] .= -$ex_basic.(b', x)
          J[:, n+1:end] .= -a' .* $ex_deriv_b.(b', x)
          nothing
        end)
  jacobian = :((n, params, x) -> begin
        J = similar(x, length(x), 2*n)
        $jacobian!(n, params, x, J)
        return J
      end)
  
  alternant = :((n, params, μ) -> begin
          a = view(params, 1:n)
          b = view(params, n+1:2*n)
          fvals = $ex_target.(μ) .- $ex_basic.(b',μ)*a
          @views return fvals[1:end-1] .+ fvals[2:end]
        end)
  
  alternant_grad_ab = :((n, params, μ) -> begin
          ∇F = $jacobian(n, params, μ)
          @assert length(μ) == 2 * n + 1
          @views return ∇F[1:2*n, :] .+ ∇F[2:2*n+1, :]
        end)

  dparams_dxi = :((n, params, ξ) -> begin
          a = view(params, 1:n)
          b = view(params, n+1:2*n)
          ∇F_params = $jacobian(n, params, ξ)
          npts = length(ξ)
          de_dξ = $errderiv_eval(a,b,ξ)
          ∇F_ξ = GenericLinearAlgebra.diagm(0 => de_dξ)
          return -∇F_params \ ∇F_ξ
        end)

  alternant_grad_xi = :((n, params, μ, ξ) -> begin
          phi_g_ab = $alternant_grad_ab(n, params, μ)
          f_g_ab = $dparams_dxi(n, params, ξ)
          return phi_g_ab * f_g_ab
        end)

return (func_eval=eval(func_eval),
        funcderiv_eval=eval(funcderiv_eval),
        target_eval=eval(target_eval),
        targetderiv_eval=eval(targetderiv_eval),
        err_eval=eval(err_eval),
        errderiv_eval=eval(errderiv_eval),
        jacobian=eval(jacobian),
        jacobian! = eval(jacobian!),
        alternant=eval(alternant),
        alternant_grad_ab=eval(alternant_grad_ab),
        alternant_grad_xi=eval(alternant_grad_xi),
        dparams_dxi=eval(dparams_dxi))

end
