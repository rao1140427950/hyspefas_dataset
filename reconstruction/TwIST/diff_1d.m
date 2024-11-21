function sol=diff_1d(x)
h=[0 1 -1];
sol=conv(x,h);