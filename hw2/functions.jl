using Plots, Optim, NLsolve

"""
CRRA uitility that returns -Inf if consumption is negative. ``c^{1-γ}/(1-γ)`` if ``γ > 1`` and ``\\log(c)`` if ``γ = 1``.
"""
function u(c,γ)

    if c < 0
        return -Inf
    end

    if γ == 1
        return log(c)
    else
        return c^(1-γ)/(1-γ)
    end

end

"""
Derivative of CRRA. See help for CRRA utility.
"""
function uprime(c,γ)

    if c < 0
        return -Inf
    end

    if γ == 1
        return 1/c
    else
        return c^(-γ)
    end

end


"""
This is the function ``V`` in the homework. `as` are the assets (which is the parameter to be selected), `ys` is the income process, `r` is the interest rate, etc.
"""
function utility_to_maximize(as,ys,r,β,κ,γ,a_bar,T)

    cs = map(t->(1+r)*as[t] + ys[t] - as[t+1],1:T)
    us = sum(β.^(0:(T-1)).*u.(cs,γ)) + β^T*κ*u(as[T+1] + a_bar,γ)

    return us

end

"""
structure with three fields:

* assets: the array with asset selected for each period
* consumption: implied consumption by interest rate, income process and interest rate
* op: the optimization object or root finding object. Provided for debug purposes. 

"""
struct solution_cons
    assets::Array
    consumption::Array
    op
end

"""
Solves the consumer problem by optimizing. a0 is the starting assets, b_bar is the most debt the agent can take. Should be negative, but this is not enforced. Can be -Inf. Uses LBFGS with box constraints, to guarantee that the agent doesn't die in debt. Otehr arguments, see `utility_to_maximize` function.
"""
function solve_consumer(ys,a0,r,β,κ,γ,a_bar,b_bar,T)

    lw = [fill(b_bar,T-1);0]
    up = fill(Inf,T)
    op2 = optimize(a->-utility_to_maximize([a0;a],ys,r,β,κ,γ,a_bar,T),lw,up,fill(0.01,T),Fminbox(LBFGS()), autodiff = :forward)

    assets = [a0;op2.minimizer]
    consumption = (1+r)*assets[1:T] + ys - assets[2:(T+1)]

    return solution_cons(assets,consumption,op2)

end

"""
first order conditions. Assumes it holds exactly, so you will need to add a multiplier to enforce constraints. Low level constructor. Note that there is a trickery to allow us to recycle this function for each case and the last period in which we have an extra parameter κ.
"""
foc(c,cp,γ,β,κ = 1) = uprime(c,γ) - β*κ*uprime(cp,γ)

"""
This function stacks takes assets `as`, income process `ys` and other parameter, builds and stacks all the FOCs together. The only difference between this and `utility_to_maximize` is μ, which is the multiplier on the last period, which is constraint to end with positive assets. We reparametrize so the assets in the last period are always positive.
"""
function build_focs(as,ys,r,β,κ,γ,a_bar,μ,T)

    as[T+1] = exp(as[T+1])
    cs = map(t->(1+r)*as[t] + ys[t] - as[t+1],1:T)
    focs_alive = map((c,cp)->foc(c,cp,γ,β),cs[1:(T-1)],cs[2:T])
    foc_dead = foc(cs[T],as[T+1] + a_bar,γ,β,κ) + μ

    return [focs_alive;foc_dead;μ*as[T+1]]

end

"""
This is a carbon copy of solve_consumer, but solves the FOCs using NLsolve.
"""
function solve_consumer2(ys,a0,r,β,κ,γ,a_bar,b_bar,T)

    sol = nlsolve(a->build_focs([a0;a[1:T]],ys,r,β,κ,γ,a_bar,a[T+1],T),fill(0.01,T+1), autodiff = :forward)
    assets = [a0;sol.zero[1:(T)]]
    assets[T+1] = exp(assets[T+1])
    consumption = (1+r)*assets[1:T] + ys - assets[2:(T+1)]

    return solution_cons(assets,consumption,sol)

end

"""
Very basic quadratic objective function (GMM with identity weights) that targets the asset values (only) to get an object function (hence the name) to estimate the parameters. Estimating β will be tricky, but κ,γ and a_bar can be estimated. This uses the optimization version of the solution of the consumer problem.
"""
function obj_fun(ys,as,r,β,κ,γ,a_bar,b_bar,T)

    a0 = as[1]
    sol = solve_consumer(ys,a0,r,β,κ,γ,a_bar,b_bar,T)

    Δ_as = as - sol.assets

    return Δ_as'Δ_as

end


"""
Very basic quadratic objective function (GMM with identity weights) that targets the asset values (only) to get an object function (hence the name) to estimate the parameters. Estimating β will be tricky, but κ,γ and a_bar can be estimated. This uses the foc version of the solution of the consumer problem.
"""
function obj_fun2(ys,as,r,β,κ,γ,a_bar,b_bar,T)

    a0 = as[1]
    sol = solve_consumer2(ys,a0,r,β,κ,γ,a_bar,b_bar,T)

    Δ_as = as - sol.assets

    return Δ_as'Δ_as

end
