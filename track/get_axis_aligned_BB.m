    function [cx, cy, w, h] = get_axis_aligned_BB(region)
        % this function gets a uniform format of bounding box, formats may
        % vary from benchmark to benchmark.
        num = numel(region);
        assert(num==8 || num==4);

        if num == 8
            % region(1:8) X1, Y1, X2, Y2, X3, Y3, X4, Y4
            %
            cx = mean(region(1:2:end)); % average x
            cy = mean(region(2:2:end)); % average y
            x1 = min(region(1:2:end));  % left x
            x2 = max(region(1:2:end));  % right x
            y1 = min(region(2:2:end));  % top y
            y2 = max(region(2:2:end));  % bottom y
            A1 = norm(region(1:2) - region(3:4)) * norm(region(3:4) - region(5:6));
            % A1 = sqrt[(X1-X2)^2 + (Y1-Y2)^2] * sqrt[(X2-X3)^2 + (Y2-Y3)^2]
            A2 = (x2 - x1) * (y2 - y1);
            s = sqrt(A1 / A2);
            w = s * (x2 - x1) + 1;
            h = s * (y2 - y1) + 1;
        else
            x = region(1);
            y = region(2);
            w = region(3);
            h = region(4);
            cx = x + w/2;
            cy = y + h/2;
        end
    end