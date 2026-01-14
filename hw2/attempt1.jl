using Plots, Optim, NLsolve

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

uprime(c,γ) = γ == 1 ? 1/c : c^(-γ)

function utility_to_maximize(as,ys,r,β,κ,γ,a_bar,b_bar,T)

    cs = map(t->(1+r)*as[t] + ys[t] - as[t+1],1:T)
    us = sum(β.^(0:(T-1)).*u.(cs,γ)) + β^T*κ*u(as[T+1] + a_bar,γ)

    return us

end

struct solution_cons
    assets::Array
    consumption::Array
    op::Optim.MultivariateOptimizationResults
end

function solve_consumer(ys,a0,r,β,κ,γ,a_bar,b_bar,T)

    lw = [fill(b_bar,T-1);0]
    up = fill(Inf,T)
    op2 = optimize(a->-utility_to_maximize([a0;a],ys,r,β,κ,γ,a_bar,b_bar,T),lw,up,fill(0.01,T),Fminbox(LBFGS()), autodiff = :forward)

    assets = [a0;op2.minimizer]
    consumption = (1+r)*assets[1:T] + ys - assets[2:(T+1)]

    return solution_cons(assets,consumption,op2)

end

foc(c,cp,γ,β,κ = 1) = uprime(c,γ) - β*κ*uprime(cp,γ)

function build_focs(as,ys,r,β,κ,γ,a_bar,b_bar,T)

    cs = map(t->(1+r)*as[t] + ys[t] - as[t+1],1:T)
    focs_alive = map((c,cp)->foc(c,cp,γ,β),cs[1:(T-1)],cs[2:T])
    foc_dead = foc(cs[T],as[T+1] + a_bar,γ,β,κ)

    return [focs_alive;foc_dead]

end

r = 0.05
a_bar = 1
b_bar = -50
β = 0.99
γ = 1
κ = 0
T = 5

ys = collect(1:T)

tt = solve_consumer(ys,0,r,β,κ,γ,a_bar,b_bar,T)

as = zeros(T+1)
as[1] = 0

for t = 1:5

    as[t+1] = (1+r)*as[t] + ys[t] - as[t]/2

end

build_focs(as,ys,r,β,0.3,1,a_bar,b_bar,T)

nl_sol = nlsolve(a->build_focs([0;a],ys,r,β,0.3,1,a_bar,b_bar,T),fill(0.01,T), autodiff = :forward)
nl_sol2 = mcpsolve(a->build_focs([0;a],ys,r,β,0.3,1,a_bar,b_bar,T),lw,up,fill(0.01,T),autodiff = :forward)



utility_to_maximize(as,ys,r,β,0.3,1,a_bar,b_bar,T)

op = optimize(a->-utility_to_maximize([0;a],ys,r,β,0,1,a_bar,b_bar,T),fill(0.01,T),LBFGS(), autodiff = :forward)
a_op = op.minimizer
cs = (1+r)*[2;a_op[1:(T-1)]] + ys - a_op

lw = [fill(b_bar,T-1);0]
up = fill(Inf,T)
op2 = optimize(a->-utility_to_maximize([0;a],ys,r,β,0,1,a_bar,b_bar,T),lw,up,fill(0.01,T),Fminbox(LBFGS()), autodiff = :forward)
a_op2 = op2.minimizer
cs2 = (1+r)*[0;a_op2[1:(T-1)]] + ys - a_op2
