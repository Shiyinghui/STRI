function net = add_stream(net, branch, rename)
% This function copies blocks to a new net. 
    for layer = branch.layers
        name = layer.name;
        inputs = layer.inputs;
        outputs = layer.outputs;
        params = layer.params;
        if ~isempty(rename)
            name = rename.layer(name);
            inputs = cellfun(rename.var, inputs, 'UniformOutput', false);
            outputs = cellfun(rename.var, outputs, 'UniformOutput', false);
            params = cellfun(rename.param, params, 'UniformOutput', false);
        end
        net.addLayer(name, layer.block, inputs, outputs, params);
    end

    branch_params = {};
    for layer = branch.layers
        branch_params = [branch_params, layer.params];
    end
    branch_params = unique(branch_params);

    for i = 1:numel(branch_params)
        param_name = branch_params{i};
        param_ind = branch.getParamIndex(param_name);
        name = param_name;
        if ~isempty(rename)
            name = rename.param(name); % avoid possible error
        end
        ind = net.getParamIndex(name);  
        net.params(ind).value = branch.params(param_ind).value;
        net.params(ind).trainMethod = branch.params(param_ind).trainMethod;
        net.params(ind).learningRate = branch.params(param_ind).learningRate;
        net.params(ind).weightDecay = branch.params(param_ind).weightDecay;
    end

end