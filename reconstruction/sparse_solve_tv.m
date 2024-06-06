function x = sparse_solve_tv(p, q, v, b, solver)
%    solver:
%        0: TwIST
%
    A = sparse(p, q, v);
    if solver == 0
        x = twist_solve_tv(A, b);
    elseif solver == 2
        x = tval3_solve(A, b);
    elseif solver == 1
        x = lasso_tv(A, b, 0.1, 1, 1);
    end

end