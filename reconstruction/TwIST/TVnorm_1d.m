function y = TVnorm_1d(x)
y = sum(sqrt(diff_1d(x).^2));
