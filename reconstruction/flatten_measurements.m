function img = flatten_measurements(img)
%     img = permute(img, [2, 1]);
    img = reshape(img, [], 1);
end