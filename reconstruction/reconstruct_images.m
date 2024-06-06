clear;
close all;

addpath('ADMM\', 'ADMM\solvers\', 'TwIST\');

load('data\sensing_matrix.mat');

load('..\data\datasets\xyzFcn_interp_d30.mat');
xFcn = xFcn_interp;
yFcn = yFcn_interp;
zFcn = zFcn_interp;

[files, xmins, ymins, xmaxs, ymaxs] = textread('..\data\datasets\labels.txt', '%s %d %d %d %d');
s = size(files);
parfor n = 1:s(1)
    file = files(n);
    xmin = xmins(n);
    ymin = ymins(n);
    xmax = xmaxs(n);
    ymax = ymaxs(n);

    tiff = imread(['..\data\datasets\ssi\', file{1,1}]);
    hsi_savepath = ['..\data\datasets\hsi_reconstructed\', file{1,1}, '.mat'];
    rgb_savepath = ['..\data\datasets\hsi_reconstructed\', file{1,1}, '.png'];
    if exist(hsi_savepath, 'file') && exist(rgb_savepath, 'file')
        disp(n);
        continue
    end
    
    tiff = tiff(ymin + 1:ymax, xmin + 1:xmax);
    masknow = mask(ymin + 1:ymax, xmin + 1:xmax, :);
    [h, w, c] = size(masknow);
    [p, q, v] = build_sensing_matrix(masknow);

    tiff = double(tiff) / 8192;
    ms = flatten_measurements(double(tiff));
    x = sparse_solve_tv(p, q, v, ms, 0);
    spimg = reconstruct_results(x, h, w, c);    
    spimg = single(spimg);
    rgb = sp2rgb(spimg, xFcn, yFcn, zFcn);

    parsave(hsi_savepath, spimg);
    imwrite(rgb, rgb_savepath);

    disp(n);
end