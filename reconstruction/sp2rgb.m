function rgb = sp2rgb(img, xFcn, yFcn, zFcn)

    xFcn = reshape(xFcn, 1, 1, []);
    yFcn = reshape(yFcn, 1, 1, []);
    zFcn = reshape(zFcn, 1, 1, []);
    
    [h, w, ~] = size(img);
    xyz = zeros(h, w, 3);
    xyz(:, :, 1) = sum(xFcn .* img, 3);
    xyz(:, :, 2) = sum(yFcn .* img, 3);
    xyz(:, :, 3) = sum(zFcn .* img, 3);
    
    rgb = xyz2rgb(xyz);
    rgb_max = max(rgb, [], 'all');
    rgb = rgb ./ rgb_max .* 255;
    rgb(rgb < 0) = 0;
    rgb = uint8(rgb);
end