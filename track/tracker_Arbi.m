function tracker_Arbi(varargin)
% this function is the main function that performs tracking.
    p.video = 'Tiger1';
    p.dataset = 'OTB-2013';
    p.drawed_rect = [];
    p.gpus = 1;
    p.visualization = true;
    p.siamfc = 0;
    p.siamfc_tri = 1;
   
    % models and prefixes
    p.model_1 = 'SiamFC_c_g.net.mat';
    p.model_2 = 'STRI.net.mat';
    p.prefix_z = 'a_';
    p.prefix_x = 'b_';
    p.prefix_join = 'xcorr';
    p.prefix_adjust = 'adjust';
    p.id_z_feat = 'z_feat';
    p.id_score = 'score';
    
    p.seq_base_path = '../demo_sequences/';
    p.model_base_path = '../data/';

    p.exemplarSize = 127;
    p.instanceSize = 255;
    p.scoreSize = 17;
    p.totalStride = 8;
    p.contextAmount = 0.5;
    
    % default hyper-parameters for both trackers,using 3 scale
    p.numScale = 3;
    p.scaleStep = 1.0375;
    p.scalePenalty = 0.9745;
    p.scaleLR = 0.59;
    p.responseUp = 16;
    p.windowing = 'cosine';
    p.wInfluence = 0.176;
    p = vl_argparse(p, varargin);   
    
    % load model(s)
    if p.siamfc
        net1_z = load_pretrained([p.model_base_path p.model_1], []);
        net1_x = load_pretrained([p.model_base_path p.model_1], []);
        remove_layers_from_prefix(net1_z, p.prefix_x);
        remove_layers_from_prefix(net1_z, p.prefix_join);
        remove_layers_from_prefix(net1_z, p.prefix_adjust);
        remove_layers_from_prefix(net1_x, p.prefix_z);
        z_feat_index = net1_z.getVarIndex(p.id_z_feat);
        score_index = net1_x.getVarIndex(p.id_score);
    end
    
    if p.siamfc_tri
        net2_z = load_pretrained([p.model_base_path p.model_2], []);
        net2_x = load_pretrained([p.model_base_path p.model_2], []);
        remove_layers_from_prefix(net2_z, p.prefix_x);
        remove_layers_from_prefix(net2_z, p.prefix_join);
        remove_layers_from_prefix(net2_z, p.prefix_adjust);
        remove_layers_from_prefix(net2_x, p.prefix_z);
        z_feat_index = net2_z.getVarIndex(p.id_z_feat);
        if ~p.siamfc
            score_index = net2_x.getVarIndex(p.id_score);
        end
    end
    
    % load video info
    [img_files, ~, ~] = load_video_info(p.seq_base_path, p.dataset, p.video);
    nImgs = numel(img_files);
    
    target_size = [p.drawed_rect(4) p.drawed_rect(3)];  % h, w
    target_pos =  [p.drawed_rect(2) + target_size(1)/2, p.drawed_rect(1)+ target_size(2)/2]; % y,x
    
    % calculate the scale factor
    h_z = target_size(1) + p.contextAmount * sum(target_size); % h_z = h + 0.5 * (h + w)
    w_z = target_size(2) + p.contextAmount * sum(target_size); % w_z = w + 0.5 * (h + w)
    s_z = sqrt(h_z * w_z);
    scale_z = p.exemplarSize / s_z;  % scale_z * s_z = p.exemplarSize

    d_search = (p.instanceSize - p.exemplarSize) / 2; % d_search = 64;
    pad = d_search / scale_z;
    s_x = s_z + 2 * pad;  % scale_z * s_x = p.instanceSize
    min_s_x = 0.2 * s_x;  % minimum size of s_x
    max_s_x = 5 * s_x;    % maxmum size of s_x

    exponent = ceil(p.numScale / 2 - p.numScale):floor(p.numScale / 2); % [-1 0 1]
    scales = p.scaleStep .^exponent; %  % scales = [1/1.0375  1  1.0375]

    start_frame = 1;
    im = gpuArray(single(img_files{start_frame}));
    % if grayscale repeat one channel to match filters size
    if(size(im, 3) == 1)  
        im = repmat(im, [1 1 3]);
    end
    % calculate average R G B value 
    avg_chans = gather([mean(mean(im(:,:,1))) mean(mean(im(:,:,2))) mean(mean(im(:,:,3)))]);
    
    % initialize the exemplar
    z_crop = get_subwindow(im, target_pos, p.exemplarSize*[1 1], round(s_z)*[1 1], avg_chans);
    if ~isnumeric(z_crop)
        fprintf('z_crop is not a numeric array.\n');
    end
    
    % perform a forward pass, get exemplar features denoted as z_features
    if p.siamfc
        net1_z.eval({'exemplar', z_crop});
        z_features_1 = net1_z.vars(z_feat_index).value;
        z_features_1 = repmat(z_features_1, [1 1 1 p.numScale]);
        net1_s_x = s_x;
        net1_target_size = target_size;
        net1_target_pos = target_pos;
        net1_boundingboxes = zeros(nImgs, 4);
    end
    if p.siamfc_tri
        net2_z.eval({'exemplar', z_crop});
        z_features_2 = net2_z.vars(z_feat_index).value;
        z_features_2 = repmat(z_features_2, [1 1 1 p.numScale]);
        net2_s_x = s_x;
        net2_target_size = target_size;
        net2_target_pos = target_pos;
        net2_boundingboxes = zeros(nImgs, 4);
    end
    
    switch p.windowing 
        case 'cosine'            % 17 * 16  hanning window
            window = single(hann(p.scoreSize * p.responseUp) * hann(p.scoreSize * p.responseUp)');
        case 'uniform'
            window = single(ones(p.scoreSize * p.responseUp, p.scoreSize * p.responseUp));
    end
    window = window / sum(window(:));
    
    % initialize the video player
    videoPlayer = [];
    if p.visualization && isToolboxAvailable('Computer Vision System Toolbox')
        videoPlayer = vision.VideoPlayer('Position',[100 100 [size(im,2), size(im,1)]+30]);
    end

    tic;
    for i = 1:nImgs
        if i > start_frame
            im = gpuArray(single(img_files{i}));
            if(size(im, 3) == 1)
                im = repmat(im, [1 1 3]);
            end
            % net1
            if p.siamfc
                net1_scaled_instance = net1_s_x .* scales;
                net1_scaled_target = [net1_target_size(1) .* scales; net1_target_size(2) .* scales];
                net1_x_crops = get_scaled_xcrops(im, net1_target_pos, net1_scaled_instance, p.instanceSize, avg_chans, p);
                [new_target_pos, new_scale] = eval_tracking(net1_x, round(net1_s_x), score_index, z_features_1, net1_x_crops, net1_target_pos, window, p);
                net1_target_pos = gather(new_target_pos);

                % according to new_target_pos and new_scale, we obtain new target_size and s_x to calculate the bounding boxes
                % for the current frame and new x_crops for the next frame.
                net1_s_x = max(min_s_x, min(max_s_x, (1-p.scaleLR) *  net1_s_x + p.scaleLR * net1_scaled_instance(new_scale)));
                net1_target_size = (1 - p.scaleLR) * net1_target_size + p.scaleLR * [net1_scaled_target(1,new_scale)  net1_scaled_target(2,new_scale)];
            end

            % net2
            if p.siamfc_tri
                net2_scaled_instance = net2_s_x .* scales;
                net2_scaled_target = [net2_target_size(1) .* scales; net2_target_size(2) .* scales];
                net2_x_crops = get_scaled_xcrops(im, net2_target_pos, net2_scaled_instance, p.instanceSize, avg_chans, p);
                [new_target_pos, new_scale] = eval_tracking(net2_x, round(net2_s_x), score_index, z_features_2, net2_x_crops, net2_target_pos, window, p);
                net2_target_pos = gather(new_target_pos);

                net2_s_x = max(min_s_x, min(max_s_x, (1-p.scaleLR) *  net2_s_x + p.scaleLR * net2_scaled_instance(new_scale)));
                net2_target_size = (1 - p.scaleLR) * net2_target_size + p.scaleLR * [net2_scaled_target(1,new_scale)  net2_scaled_target(2,new_scale)];
            end
        else
        end

        if p.siamfc 
            net1_rect_pos = [net1_target_pos([2, 1]) - net1_target_size([2, 1])/2, net1_target_size([2,1])];
            net1_boundingboxes(i, :) = net1_rect_pos;
        end
        if p.siamfc_tri
            net2_rect_pos = [net2_target_pos([2, 1]) - net2_target_size([2, 1])/2, net2_target_size([2,1])];
            net2_boundingboxes(i, :) = net2_rect_pos;
        end

        if p.visualization
            im = gather(im)/255;
            if p.siamfc
                im = insertShape(im, 'rectangle', net1_rect_pos, 'LineWidth', 4, 'Color', 'red');
            end
            if p.siamfc_tri
                path = ['../res/Bolt2/' num2str(i) '.jpg'];
                im = insertShape(im, 'rectangle', net2_rect_pos, 'LineWidth', 2, 'Color', 'yellow');
                imwrite(im,path);
            end
            step(videoPlayer, im);
        end
    end
    
end