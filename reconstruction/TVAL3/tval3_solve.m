function U = tval3_solve(A, b)

opts.mu = 2^8;
opts.beta = 2^5;
opts.tol = 1E-3;
opts.maxit = 300;
opts.TVnorm = 1;
opts.nonneg = true;
opts.disp = false;
% opts.TVL2 = true;

p = 301;
q = 1;

[U,out] = TVAL3(A,b,p,q,opts);