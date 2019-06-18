function [imout_z, imout_x, labels] = get_random_batch(imdb, batch, imdb_video, data_dir, varargin)
% This function gets a random batch of z and x crops for training.
    % default parameters
    opts.exemplarSize = [];
    opts.instanceSize = [];
    opts.frameRange = 50;
    opts.subMean = false;
    opts.colorRange = 255;    
    opts.stats.rgbMean_z = [];
    opts.stats.rgbVariance_z = [];
    opts.stats.rgbMean_x = [];
    opts.stats.rgbVariance_x = [];
    opts.augment.translate = false;
    opts.augment.maxTranslate = []; 
    opts.augment.stretch = false;
    opts.augment.maxStretch = 0.1;
    opts.augment.color = false;
    opts.augment.grayscale = 0;
    opts.prefetch = false;
    opts.numThreads = 12;
    % above parameters are rewritten here
    opts = vl_argparse(opts, varargin);

    TRAIN_SET = 1; VAL_SET = 2;
    RGB = 1; GRAY = 2;
    batch_set = imdb.images.set((batch(1)));
    assert(all(batch_set == imdb.images.set(batch)));
    batch_size = numel(batch);
    pair_types_rgb = datasample(1:2, batch_size, 'weights', [1-opts.augment.grayscale opts.augment.grayscale]);

    ids_set_index = find(imdb_video.set==batch_set);   % 在训练集或验证集上寻找
    rnd_videos_index = datasample(ids_set_index, batch_size, 'Replace', false); % unique
    ids_pairs_index = rnd_videos_index(1:batch_size);
    objects = struct();
    objects.set = batch_set * uint8(ones(1, batch_size));
    objects.z = cell(1, batch_size);
    objects.x = cell(1, batch_size);
    crops_z_path = cell(1, batch_size);
    crops_x_path = cell(1, batch_size);
    labels = zeros(1, batch_size);
    imout_z = zeros(opts.exemplarSize(1), opts.exemplarSize(1), 3, batch_size, 'single');
    imout_x = zeros(opts.instanceSize(1), opts.instanceSize(1), 3, batch_size, 'single');

    for i = 1:batch_size
        labels(i) = 1;
        [objects.z{i}, objects.x{i}] = choose_pairs(imdb_video, ids_pairs_index(i), opts.frameRange);
    end

    for i = 1:batch_size
        crops_z_path{i} = [strrep(fullfile(data_dir, objects.z{i}.frame_path), '.JPEG','') '.' num2str(objects.z{i}.track_id, '%02d') '.crop.z.jpg'];
        crops_x_path{i} = [strrep(fullfile(data_dir, objects.x{i}.frame_path), '.JPEG','') '.' num2str(objects.x{i}.track_id, '%02d') '.crop.x.jpg'];
    end

    img_paths = [crops_z_path crops_x_path]; % integrate

    if opts.prefetch
        error('prefetch function is not yet implemented');
    end

    crops = vl_imreadjpeg(img_paths, 'numThreads', opts.numThreads);
    crops_z = crops(1:batch_size);
    crops_x = crops(batch_size+1: end);
    clear crops;

    if batch_set == TRAIN_SET
        aug_opts = opts.augment;
    else
        aug_opts = struct('translate', false, ...
            'maxTranslate', 0, ...
            'stretch', false, ...
            'maxStretch', 0, ...
            'color', false);
    end
                % crop 像素矩阵
    aug_z = @(crop)acquire_augment(crop, opts.exemplarSize, opts.stats.rgbVariance_z, aug_opts);
    aug_x = @(crop)acquire_augment(crop, opts.instanceSize, opts.stats.rgbVariance_x, aug_opts);

    for i = 1:batch_size
        tmp_z = aug_z(crops_z{i});
        tmp_x = aug_x(crops_x{i});

        switch pair_types_rgb(i)
            case RGB
                imout_z(:,:,:,i) = tmp_z;
                imout_x(:,:,:,i) = tmp_x;
            case GRAY
                imout_z(:,:,:,i) = repmat(rgb2gray(tmp_z/255)*255, [1 1 3]);
                imout_x(:,:,:,i) = repmat(rgb2gray(tmp_x/255)*255, [1 1 3]);
        end

        if opts.subMean
            means = [opts.stats.rgbMean_z(:); opts.stats.rgbMean_x(:)];
            lower = 0.2 * 255;
            upper = 0.8 * 255;
            if ~all((lower <= means) & (means <= upper))
                error('mean does not seem to for pixels in 0-255');
            end
            imout_z = bsxfun(@minus, imout_z, reshape(opts.stats.rgbMean_z, [1 1 3]));
            imout_x = bsxfun(@minus, imout_x, reshape(opts.stats.rgbMean_x, [1 1 3]));
        end
        imout_z = imout_z / 255 * opts.colorRange;
        imout_x = imout_x / 255 * opts.colorRange;
    end
end


function [z, x] = choose_pairs(imdb_video, rand_vid, frameRange)
    valid_trackids = find(imdb_video.valid_trackids(:, rand_vid) > 1);
    assert(~isempty(valid_trackids), 'no valid trackids for a video in the batch.');
    rand_trackid_z = datasample(valid_trackids, 1);
    rand_z = datasample(imdb_video.valid_per_trackid{rand_trackid_z, rand_vid}, 1);
    possible_x_pos = (1:numel(imdb_video.valid_per_trackid{rand_trackid_z, rand_vid}));
    [~, rand_z_pos] = ismember(rand_z, imdb_video.valid_per_trackid{rand_trackid_z, rand_vid});
    possible_x_pos = possible_x_pos([max(rand_z_pos-frameRange, 1):(rand_z_pos-1), (rand_z_pos+1):min(rand_z_pos+frameRange, numel(possible_x_pos))]);
    possible_x = imdb_video.valid_per_trackid{rand_trackid_z, rand_vid}(possible_x_pos);
    assert(~isempty(possible_x), 'No valid x for the chosen z.');
    rand_x = datasample(possible_x, 1);
    assert(imdb_video.objects{rand_vid}{rand_x}.valid, 'error picking rand x.');
    z = imdb_video.objects{rand_vid}{rand_z};
    x = imdb_video.objects{rand_vid}{rand_x};
end


function imout = acquire_augment(image, image_size, rgbVariance, aug_opts)
               % image_size: for exemplar 127, for instance 239
    if numel(aug_opts.maxTranslate) == 1
        aug_opts.maxTranslate = [aug_opts.maxTranslate, aug_opts.maxTranslate];
    end
    imt = image;
    if size(imt, 3) == 1
        imt = cat(3, imt, imt, imt);
    end

    % w = h, either 255 or 127, both are odd numbers
    w = size(imt, 2);
    h = size(imt, 1);
    cx = (w+1)/2;  % get the center, cx = 64 or 128, 原图像的中心
    cy = (h+1)/2;

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

    if ~aug_opts.color   % aug_opts.color = true
        imout = imt(sy,sx,:);
    else 
        offset = reshape(rgbVariance * randn(3,1),1,1,3); % rgbVariance 3*3
        imout = bsxfun(@minus, imt(sy,sx,:), offset);
    end
end

