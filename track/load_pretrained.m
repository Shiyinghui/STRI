    function net = load_pretrained(net_path, gpu)
        % this function loads a pretraind network, given the model path,
        % and sets 'test' the mode of the network.
        if iscell(net_path)
            net_path = net_path{1};
        end
        train_res = load(net_path);
        net = train_res.net;
        % find the layer index of XCorr layer
        for i = 1:numel(net.layers)
            if ~isempty(strfind(net.layers(i).type, 'XCorr'))
                xcorr_ind = i; break;
            end
        end

        % remove specified legacy fields if present
        if isfield(net.layers(xcorr_ind).block, 'expect')
            net.layers(xcorr_ind).block = rmfield(net.layers(xcorr_ind).block, 'expect');

        end
        if isfield(net.layers(xcorr_ind).block, 'visualization_active')
            net.layers(xcorr_ind).block = rmfield(net.layers(xcorr_ind).block, 'visualization_active');
        end
        if isfield(net.layers(xcorr_ind).block, 'visualization_grid_sz')
            net.layers(xcorr_ind).block = rmfield(net.layers(xcorr_ind).block,'visualization_grid_sz');
        end

        net = dagnn.DagNN.loadobj(net);

        % find the Tri_loss block
        for i = 1:numel(net.layers)
            if isa(net.layers(i).block, 'dagnn.Loss')
                layer_name = net.layers(i).name;
                break;
            end
        end
        % remove loss layer
        layer = net.layers(net.getLayerIndex(layer_name));
        net.removeLayer(layer_name);
        net.renameVar(layer.outputs{1}, layer.inputs{1}, 'quiet', true);

        if ~isempty(gpu)
            gpuDevice(gpu)
        end
        net.move('gpu');

        % The setting of mode is very important for batch norm, now we use stats accumulated during training.
        net.mode = 'test';
              
    end
    
   
    
    
    
    
    
    