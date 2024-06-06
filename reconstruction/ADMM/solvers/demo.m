clear; close all;
path(path,genpath(pwd));

% problem size
n = 64;
ratio = .3;
p = n; q = n; % p x q is the size of image
m = round(ratio*n^2);

% sensing matrix
A = rand(m,p*q)-.5;

% original image
I = phantom(n);
nrmI = norm(I,'fro');
figure();
subplot(121); imshow(I,[]);
title('Original phantom','fontsize',18); drawnow;

% observation
f = A*I(:);
favg = mean(abs(f));

% add noise
fnoise = .5*favg*randn(m,1);
f = f + fnoise;

[x, his] = lasso_tv(A, f, 100, 1, 1);
