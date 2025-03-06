module Autocode

using Symbolics
using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(Autocode)

function gen_funs(targetfn, basicexpr, b, x)
  #targetfn is usually 1/x
  #basicf is usually exp(-b*x)

  ex_basic = Symbolics.build_function(basicexpr, b, x, expression=false)
  ex_deriv_x = Symbolics.build_function(Symbolics.derivative(basicexpr, x, simplify=true), b, x, expression=false)
  ex_deriv_b = Symbolics.build_function(Symbolics.derivative(basicexpr, b, simplify=true), b, x, expression=false)


  func_eval = :((a,b,x) -> begin
          $ex_basic.(b',x)*a
        end)

  funcderiv_eval = :((a,b,x) -> begin
          $ex_deriv_x.(b',x)*a
        end)
  
  ex_target = Symbolics.build_function(targetfn, x, expression=false)
  ex_target_deriv_x = Symbolics.build_function(Symbolics.derivative(targetfn, x), x, expression=false)

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
          @views a, b = params[1:n], params[n+1:end]
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
          @views a, b = params[1:n], params[n+1:end]
          fvals = $ex_target.(μ) .- $ex_basic.(b',μ)*a
          @views return fvals[1:end-1] .+ fvals[2:end]
        end)
  
  alternant_grad_ab = :((n, params, μ) -> begin
          @views a, b = params[1:n], params[n+1:end]
          ∇F = $jacobian(n, params, μ)
          @assert length(μ) == 2 * n + 1
          @views ∇ϕ = ∇F[1:2*n, :] .+ ∇F[2:2*n+1, :]
          return ∇ϕ
        end)

  dparams_dxi = :((n, params, ξ) -> begin
          @views a, b = params[1:n], params[n+1:end]
          ∇F_params = $jacobian(n, params, ξ)
          npts = length(ξ)
          ∇F_ξ = similar(ξ, npts, npts)
          ∇F_ξ .= 0.0
          de_dξ = $errderiv_eval(a,b,ξ)
          @inbounds for i = 1:2*n
              ∇F_ξ[i, i] = de_dξ[i]
          end
          return -∇F_params \ ∇F_ξ
        end)

  alternant_grad_xi = :((n, params, μ, ξ) -> begin
          phi_g_ab = $alternant_grad_ab(n, params, μ)
          f_g_ab = $dparams_dxi(n, params, ξ)
          return phi_g_ab * f_g_ab
        end)

  return (func_eval=@RuntimeGeneratedFunction(func_eval),
          funcderiv_eval=@RuntimeGeneratedFunction(funcderiv_eval),
          target_eval=@RuntimeGeneratedFunction(target_eval),
          targetderiv_eval=@RuntimeGeneratedFunction(targetderiv_eval),
          err_eval=@RuntimeGeneratedFunction(err_eval),
          errderiv_eval=@RuntimeGeneratedFunction(errderiv_eval),
          jacobian=@RuntimeGeneratedFunction(jacobian),
          jacobian! = @RuntimeGeneratedFunction(jacobian!),
          alternant=@RuntimeGeneratedFunction(alternant),
          alternant_grad_ab=@RuntimeGeneratedFunction(alternant_grad_ab),
          alternant_grad_xi=@RuntimeGeneratedFunction(alternant_grad_xi),
          dparams_dxi=@RuntimeGeneratedFunction(dparams_dxi))

end
end