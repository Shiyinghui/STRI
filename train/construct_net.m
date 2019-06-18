function net = construct_net(opts)
% This function constructs our siamese network. 
    opts.strides = [2 2 1 2];
    opts.siamese = true;
    fprintf('construct network...\n');

    branch = create_branch('exemplarSize',opts.exemplarSize,...
        'instanceSize', opts.instanceSize,...
        'batchNormalization', opts.batchNormalization,...
        'networkType', 'simplenn',...
        'weightInitMethod', opts.init.weightInitMethod,...
        'scale', opts.init.scale,...
        'initBias', opts.init.initBias,...
        'strides', opts.strides);

    branch = dagnn.DagNN.fromSimpleNN(branch);
   
    orig_in = branch.getInputs();
    orig_out = branch.getOutputs();
    assert(numel(orig_in) == 1,'two or more inputs!');
    assert(numel(orig_out) == 1,'two or more outputs!');
    branch.renameVar(orig_in{1},'in');
    branch.renameVar(orig_out{1},'out');
    % initialize an empty DagNN object.
    net = dagnn.DagNN();
    net = make_DAG(net, branch, opts.siamese);
   
    % add the cross-correlation layer.
    net.addLayer('xcorr', XCorr(),  {'z_feat', 'x_feat'}, {'xcorr_out'},{});

    % add adjust layer and its some parameters
    net.addLayer('adjust', dagnn.Conv('size',[1, 1, 1, 1]), {'xcorr_out'}, {'score'},{'adjust_f', 'adjust_b'});
    filter_index = net.getParamIndex('adjust_f');
    bias_index = net.getParamIndex('adjust_b');
    net.params(filter_index).value = single(1e-3);
    net.params(bias_index).value = single(0);
    net.params(filter_index).learningRate = 0;
    net.params(bias_index).learningRate = 1;
    
    % initialize the params with the original model
    old_net = load('/home/lch/siamese-fc/2016-08-17_gray025.net.mat');
    old_net = old_net.net;
    for i = 1: numel(old_net.params)
        net.params(i).value = old_net.params(i).value;
    end
    
    inputs={'exemplar',[opts.exemplarSize 3 opts.train.batchSize],...
        'instance',[opts.instanceSize 3 opts.train.batchSize]};
    net_dot=net.print(inputs, 'Format', 'dot');
    if ~exist(opts.saveModel)
        mkdir(opts.saveModel);
    end
    f=fopen(fullfile(opts.saveModel,'arch.dot'),'w');
    fprintf(f,net_dot);
    fclose(f);
end

function net = make_DAG(net, branch, siamese)
    name_z = @(s)['a_',s];
    name_x = @(s)['b_',s];
    % the prefixes of layer names are 'a_' and 'b_' for the exemplar branch
    % and instance branch respectively.
    rename_z = struct('layer', name_z, 'var', name_z, 'param', name_z);
    rename_x = struct('layer', name_x, 'var', name_x, 'param', name_x);
    share = @(s) s;
    if siamese
        rename_z.param = share;
        rename_x.param = share;
    end
    % add two streams to the net
    add_stream(net, branch, rename_z); 
    add_stream(net, branch, rename_x);
    net.renameVar(name_z('in'), 'exemplar');
    net.renameVar(name_x('in'), 'instance');
    net.renameVar(name_z('out'), 'z_feat');
    net.renameVar(name_x('out'), 'x_feat');
end
   
   