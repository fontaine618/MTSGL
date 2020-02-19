v1 = [-2, -1]
v2 = [0, 1]
v3 = [2, 3]
p = size(v1)
alpha = 0.5
tau = 0.2
q = inf

cvx_begin
    variable x1(p)
    minimize tau * ((alpha) * norm(x1,1) + (1-alpha) * norm(x1,q)) + sum(square(x1-v1)) / 2
cvx_end
cvx_begin
    variable x2(p)
    minimize 2*tau * ((alpha) * norm(x2,1) + (1-alpha) * norm(x2,q)) + sum(square(x2-v2)) / 2
cvx_end
cvx_begin
    variable x3(p)
    minimize 3*tau * ((alpha) * norm(x3,1) + (1-alpha) * norm(x3,q)) + sum(square(x3-v3)) / 2
cvx_end

[x1;x2;x3]