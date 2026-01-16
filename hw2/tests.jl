include("functions.jl")

using Random

Random.seed!(1539)

# Parametrization with a basic sanity check: no bequest motive

r = 0.06
a_bar = 1
b_bar = -50
β = 0.99
γ = 1
κ = 0
T = 5
a0 = 0

ys = collect(1:T)

tt = solve_consumer(ys,a0,r,β,κ,γ,a_bar,b_bar,T)
tt2 = solve_consumer2(ys,a0,r,β,κ,γ,a_bar,b_bar,T)

## Difference in assets by each method?

maximum(abs.(tt.assets - tt2.assets))
maximum(abs.(tt.assets - tt2.assets))

# Who is the culprit?

argmax(abs.(tt.assets - tt2.assets))

# More interesting parametrization

r = 0.06
a_bar = 1
b_bar = -50
β = 0.99
γ = 1
κ = 0.5
T = 5
a0 = 0.5

ys = collect(1:T)

tt = solve_consumer(ys,a0,r,β,κ,γ,a_bar,b_bar,T)
tt2 = solve_consumer2(ys,a0,r,β,κ,γ,a_bar,b_bar,T)

maximum(abs.(tt.assets - tt2.assets) ./tt.assets)
maximum(abs.(tt.assets - tt2.assets))

argmax(abs.(tt.assets - tt2.assets))

# Non log utility

r = 0.06
a_bar = 1
b_bar = -50
β = 0.99
γ = 2
κ = 0.5
T = 5
a0 = 0.5

ys = collect(1:T)

tt = solve_consumer(ys,a0,r,β,κ,γ,a_bar,b_bar,T)
tt2 = solve_consumer2(ys,a0,r,β,κ,γ,a_bar,b_bar,T)
