function [p, q, v] = build_sensing_matrix(mask)
    [h, w, c] = size(mask);
    p = zeros([h * w * c, 1]);
    q = zeros([h * w * c, 1]);
    v = zeros([h * w * c, 1]);

    cnt = 1;
    for cc = 1:c
        for ww = 1:w
            for hh = 1:h
                p(cnt) = hh + (ww - 1) * h;
                q(cnt) = hh + (ww - 1) * h + ((cc - 1) * h * w);
                v(cnt) = mask(hh, ww, cc);
                cnt = cnt + 1;
            end
        end
    end

end