function [net, stats] = STRI()
    % This function is the main function for training, run SiamFC_tri to start training a network.
    opts.exemplarSize = [127 127];
    opts.instanceSize = [239 239];
    opts.init.scale = 1;
    opts.init.weightInitMethod = 'xavierimproved';
    opts.init.initBias = 0.1;
    opts.loss.type = 'triplet';
    opts.loss.rPos = 16;
    opts.loss.labelWeight = 'balanced';

    % Data augmentation settings
    opts.subMean = false;
    opts.colorRange = 255;
    opts.augment.translate = true;
    opts.augment.maxTranslate = 4;
    opts.augment.stretch = true;
    opts.augment.maxStretch = 0.05;
    opts.augment.color = true;
    opts.augment.grayscale = 0.25;  % likelihood of using grayscale pair

    opts.imdbVideoPath='/home/lch/siamese-fc/imdb_video.mat';
    opts.saveModel='/home/lch/siamese-lch/data/Tri_loss/';
    opts.rootDataDir='/home/lch/tracking/ILSVRC2015_VID/ILSVRC2015_lch/Data/VID/train/';
    opts.imageStatsPath='/home/lch/siamese-fc/imageStats.mat';

    opts.pretrain = false;  % the net is not trained at first.
    opts.prefetch = false;  % whether to prefetch data or not
    opts.expDir = opts.saveModel;  % where to save the trained net
    opts.numFetchThreads = 12; % used by vl_imreadjpg when reading dataset
    opts.validation = 0.1; % fraction of all videos reserved to validation
    opts.numPairs =  53200; %5.32e4;  randomly sample 53200 pairs from ILSVRC2015
    opts.frameRange = 100; % in the training phase, randomly selected z and x are at most 100 frames apart. 
    opts.gpus = 1;  
    opts.shuffleDataSet = false;
    opts.batchNormalization = true;
    opts.randomSeed = 0;
    
    % hyper-params for training
    opts.train.numEpochs = 10; % perform training over 10 epochs
    opts.train.learningRate = logspace(-4, -5, opts.train.numEpochs);
    opts.train.weightDecay = 5e-4;
    opts.train.batchSize = 8;
    opts.train.profile = false;
    opts.train.gpus = opts.gpus;
    opts.train.prefetch = opts.prefetch;
    opts.train.expDir = opts.expDir;

    net = construct_net(opts);
    fprintf('construct net complete!\n');
    [resp_size, resp_stride] = get_response_size(net, opts);
    assert(all(mod(resp_size, 2) == 1), 'response size is not odd');

    [net, derOutputs, label_inputs_fn] = setup_loss(net, resp_size, resp_stride, opts.loss);

    imdb_video = load(opts.imdbVideoPath); % load video metadata
    imdb_video = imdb_video.imdb_video;
    fprintf('load data complete!\n');

    [rgbMean_z, rgbVariance_z, rgbMean_x, rgbVariance_x] = load_RGB_Info(opts);

    rng(opts.randomSeed);

    [imdb_video, imdb] = choose_val_set(imdb_video, opts);

    % get batch function
    batch_fn = @(db, batch)get_batch(db, batch, imdb_video, opts.rootDataDir, numel(opts.train.gpus)>=1,...
        struct('exemplarSize', opts.exemplarSize, ...
        'instanceSize', opts.instanceSize, ...
        'frameRange', opts.frameRange, ...
        'subMean', opts.subMean, ...
        'colorRange', opts.colorRange, ...
        'stats', struct('rgbMean_z', rgbMean_z, ...
        'rgbVariance_z', rgbVariance_z, ...
        'rgbMean_x', rgbMean_x, ...
        'rgbVariance_x', rgbVariance_x), ...
        'augment', opts.augment, ...
        'prefetch', opts.train.prefetch, ...
        'numThreads', opts.numFetchThreads), ...
        label_inputs_fn);
    opts.train.derOutputs = derOutputs;
    opts.train.randomSeed = opts.randomSeed;

    % call cnn_train_dag, strat training.
    [net, stats] = cnn_train_dag(net, imdb, batch_fn, opts.train);

end

% get RGB info, rgbMean and rgbCovariance are used to calculate rgbVariance
function [rgbMean_z, rgbVariance_z, rgbMean_x, rgbVariance_x] = load_RGB_Info(opts)
    stats = load(opts.imageStatsPath);
    rgbMean_z = reshape(stats.z.rgbMean, [1,1,3]);
    rgbMean_x = reshape(stats.x.rgbMean, [1,1,3]);
    [v,d] = eig(stats.z.rgbCovariance);  % 特征值的对角矩阵D和矩阵V,A*V = V*D
    rgbVariance_z = 0.1*sqrt(d)*v';
    [v,d] = eig(stats.x.rgbCovariance);
    rgbVariance_x = 0.1*sqrt(d)*v';
end

% choose validation set
function [imdb_video, imdb] = choose_val_set(imdb_video, opts)
    video_size = numel(imdb_video.id);  % number of total videos
    validate_size = round(video_size * opts.validation);
    train_size = video_size - validate_size;
    imdb_video.set = uint8(zeros(1, video_size));  % initialization
    imdb_video.set(1:train_size) = 1;   % train_set is denoted by 1
    imdb_video.set(train_size+1:end) = 2; % validate_set is denoted by 2

    imdb = struct();  % create the imdb struct
    imdb.images = struct();
    imdb.id = 1:opts.numPairs;
    train_pairs = round(opts.numPairs * (1 - opts.validation));
    imdb.images.set = uint8(zeros(1, opts.numPairs));
    imdb.images.set(1:train_pairs) = 1;
    imdb.images.set(train_pairs+1:end) = 2;
end


% get the size of response map and the final stride
function [resp_size, resp_stride] = get_response_size(net, opts)
    sizes = net.getVarSizes({'exemplar', [opts.exemplarSize 3 8], ...
        'instance', [opts.instanceSize 3 8]});
    % resp_size should be [15 15]
    resp_size = sizes{net.getVarIndex('score')}(1:2);
    rfs = net.getVarReceptiveFields('exemplar');

    % resp_stride should be 8
    resp_stride = rfs(net.getVarIndex('score')).stride(1);
    assert(all(rfs(net.getVarIndex('score')).stride == resp_stride));
end


% specifies the losses to minimise
function [net, derOutputs, label_inputs_fn] = setup_loss(net, resp_size, resp_stride, loss)
    % score and eltwise_label are inputs required by the loss layer
    net.addLayer('objective', Tri_Loss('loss','triplet'), ...
        {'score', 'eltwise_label'}, 'objective');
    [eltwise_label, instanceWeight] = create_labels(resp_size, loss.rPos / resp_stride);
    net.layers(end).block.opts = [ net.layers(end).block.opts, {'instanceWeights', instanceWeight}];
    derOutputs = {'objective', 1};
    label_inputs_fn = @(labels) get_label_inputs(labels, eltwise_label);

    net.addLayer('errdisp', centerThrErr(), {'score', 'label'}, 'errdisp');
    net.addLayer('errmax', MaxScoreErr(), {'score', 'label'}, 'errmax');
end


function inputs = get_label_inputs(labels, eltwise_label)
    pos = (labels > 0);
    resp_size = size(eltwise_label);
    eltwise_labels = zeros([resp_size, 1, numel(labels)], 'single');
    eltwise_labels(:,:,:,pos) = repmat(eltwise_label, [1 1 1 sum(pos)]);
    inputs = {'label', labels, 'eltwise_label', eltwise_labels};
end

% get batch function
function inputs = get_batch(db, batch, imdb_video, data_dir, use_gpu, options, label_inputs_fn)
    [imout_z, imout_x, labels] = get_random_batch(db, batch, imdb_video, data_dir, options);
    if use_gpu
        imout_z = gpuArray(imout_z);
        imout_x = gpuArray(imout_x);
    end
    label_inputs = label_inputs_fn(labels);
    inputs = [{'exemplar', imout_z, 'instance', imout_x}, label_inputs];
end