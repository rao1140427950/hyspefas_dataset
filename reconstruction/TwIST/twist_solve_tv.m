function x_twist = twist_solve_tv(A, b)

    %TwIST handles
    % Linear operator handles
    hR = @(x) A*x;
    hRt = @(x) A'*x;

    % regularization parameter
    tau = 0.0002*max(abs(hRt(b)));

    % extreme eigenvalues (TwIST parameter)
    lam1=1e-3;
    % TwIST is not very sensitive to this parameter
    % rule of thumb: lam1=1e-4 for severyly ill-conditioned% problems
    %              : lam1=1e-1 for mildly  ill-conditioned% problems
    %              : lam1=1    when A = Unitary matrix

    % stopping theshold
    tolA = 1e-5;

    % denoising function;
    tv_iters = 5;
    Psi = @(x,th)  tvdenoise(x,2/th,tv_iters);
    % TV regularizer;
    Phi = @(x) TVnorm_1d(x);

    % -- TwIST ---------------------------
    % stop criterium:  the relative change in the objective function
    % falls below 'ToleranceA'
    [x_twist,dummy,obj_twist,...
        times_twist,dummy,mse_twist]= ...
             TwIST(b,hR,tau,...
             'AT', hRt, ...
             'lambda',lam1,...
             'Psi', Psi, ...
             'Phi',Phi, ...
             'Monotone',1,...
             'Initialization',0,...
             'StopCriterion',1,...
             'ToleranceA',tolA,...
             'Verbose', 0);

end