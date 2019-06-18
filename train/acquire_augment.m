function acquire_augment(image_size)
        % image_size: for exemplar 127, for instance 239
    
    aug_opts.maxTranslate = [4,4];

    % w = h, either 255 or 127, both are odd numbers
    w = 127;
    h = 127;
    cx = (w+1)/2;  % get the center, cx = 64 or 128
    cy = (h+1)/2;

    aug_opts.stretch = true;
    aug_opts.maxStretch = 0.05;
    aug_opts.translate = true;
    aug_opts.maxTranslate = [4, 4];
    
    if aug_opts.stretch
        % aug_opts.maxStretch = 0.05
        % for example, rand(2,1) = [0.8147; 0.9058]
        % scale = [1.0315; 1.0406]              % rand, 区间(0,1)内均匀分布的随机数
        scale = (1 + aug_opts.maxStretch * (-1 + 2 * rand(2,1)));  %scale，最小值0.95

        % exemplar, image_size(1:2)' = [127;127],for sz min: [121; 121], max: [127 127]
        % instance  image_size(1:2)' = [239;239],for sz min: [227; 227], max [251 251]
        sz = round(min(image_size(1:2)'.*scale, [h;w])); 
    else
        sz = image_size;
    end

    if aug_opts.translate   % true
        if isempty(aug_opts.maxTranslate)
            % dx, dy : a scalar(integer) ranged from w-sz(2)+1 to 1
            dx = randi(w - sz(2) + 1, 1);
            dy = randi(h - sz(1) + 1, 1);
        else 
            % for exemplar: mx, min 0 max 3; my, min 0 max 3 
            mx = min(aug_opts.maxTranslate(2), floor((w-sz(2))/2));  
            my = min(aug_opts.maxTranslate(1), floor((h-sz(1))/2));   
            dx = cx - (sz(2)-1)/2 + randi([-mx,mx],1);  % cx = 64;
            dy = cy - (sz(1)-1)/2 + randi([-my,my],1);
        end
    else
        dx = cx - (sz(2)-1)/2;
        dy = cy - (sz(1)-1)/2;
    end
    sx = round(linspace(dx, dx+sz(2)-1, image_size(2))) ;
    sy = round(linspace(dy, dy+sz(1)-1, image_size(1))) ;

%     if ~aug_opts.color
%         imout = imt(sy,sx,:);
%     else
%         offset = reshape(rgbVariance * randn(3,1),1,1,3);
%         imout = bsxfun(@minus, imt(sy,sx,:), offset);
%     end
end