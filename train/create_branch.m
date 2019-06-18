function branch = create_branch(varargin)
% This function creates a simplenn network as an independent branch. 
    opts.exemplarSize = [127 127];
    opts.instanceSize = [239 239];
    opts.batchNormalization = true;
    opts.weightInitMethod = 'xavierimproved';
    opts.scale = 1;
    opts.initBias = 0.1;
    opts.weightDecay = 1;
    opts.strides = [2 2 1 2];
    opts.networkType = 'simplenn';
    opts.cudnnWorkspaceLimit = 1024*1024*1024; % 1GB
    opts = vl_argparse(opts, varargin);
    % make a net
    branch = make_net(struct(), opts);

    branch.meta.normalization.interpolation = 'bicubic' ;
    branch.meta.normalization.averageImage = [] ;
    branch.meta.normalization.keepAspect = true ;
    branch.meta.augmentation.rgbVariance = zeros(0,3) ;
    branch.meta.augmentation.transformation = 'stretch' ;
    % fill in default values
    branch = vl_simplenn_tidy(branch); 
end

% init kernel weights with xavierimproved method
function weights = init_kernel_weights(h, w, c, n)   
    sc = sqrt(2/(h * w * n)) ;
    weights = randn(h, w, c, n, 'single') * sc ;
end


% add conv-bn-relu
% filters: h * w * c * n, where h denotes height, w denotes width, c
% denotes channel and n means the number of filters.
function net = add_conv_bn_relu(net, opts, id, h, w, c, n, stride, pad)
    filter_weights = init_kernel_weights(h, w, c, n);
    convOpts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit};
    net.layers{end+1} = struct('type','conv','name',sprintf('conv%s', id),...
        'weights',{{filter_weights, zeros(n, 1, 'single')}},...
        'stride', stride,...
        'pad', pad,...
        'learningRate', [1 2],...
        'weightDecay', [opts.weightDecay, 0],...
        'opts', {convOpts});
    if opts.batchNormalization
        net.layers{end+1} = struct('type', 'bnorm', 'name', sprintf('bn%s', id), ...
            'weights', {{ones(n, 1, 'single'), zeros(n, 1, 'single'), zeros(n, 2, 'single')}}, ...
            'learningRate', [2 1 0.05], ...
            'weightDecay', [0 0]) ;
    end
    net.layers{end+1} = struct('type', 'relu', 'name', sprintf('relu%s', id)) ;
end

% add conv only
function net = add_conv(net, opts, id, h, w, c, n, stride, pad)
    filter_weights = init_kernel_weights(h, w, c, n);
    convOpts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit};
    net.layers{end+1} = struct('type','conv','name',sprintf('conv%s', id),...
        'weights',{{filter_weights, zeros(n, 1, 'single')}},...
        'stride', stride,...
        'pad', pad,...
        'learningRate', [1 2],...
        'weightDecay', [opts.weightDecay, 0],...
        'opts', {convOpts});
end

% make a net by adding layers.
function net = make_net(net, opts)
    strides = ones(1,7);
    strides(1:numel(opts.strides)) = opts.strides(:);
    net.layers = {};
    net = add_conv_bn_relu(net, opts, '1', 11, 11, 3, 96, strides(1), 0);
    net.layers{end+1} = struct('type','pool', 'name', 'pool1',...
        'method', 'max',...
        'pool', [3, 3],...
        'stride', strides(2),...
        'pad', 0);
    net = add_conv_bn_relu(net, opts, '2', 5, 5, 48, 256, strides(3), 0);
    net.layers{end+1} = struct('type','pool', 'name', 'pool2',...
        'method', 'max',...
        'pool', [3, 3],...
        'stride', strides(4),...
        'pad', 0);
    net = add_conv_bn_relu(net, opts, '3', 3, 3, 256, 384, strides(5), 0) ;
    net = add_conv_bn_relu(net, opts, '4', 3, 3, 192, 384, strides(6), 0) ;
    net = add_conv(net, opts, '5', 3, 3, 192, 256, strides(7), 0) ;

end

