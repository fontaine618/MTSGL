v = [1,2,-6,-8,0]'
p,k = size(v)
alpha = 0.4
tau = 1.5
q=2

cvx_begin
    variable x(p)
    minimize tau * ((alpha) * norm(x,1) + (1-alpha) * norm(x,q)) + sum(square(x-v)) / 2
cvx_end

x