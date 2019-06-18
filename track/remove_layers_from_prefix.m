    function net = remove_layers_from_prefix(net, prefix)
        % this function removes layers with specified prefix
        layer = net.layers;
        num_layers = numel(layer);
        to_remove = {};
        for i = 1:num_layers
            if strfind(layer(i).name, prefix)
                to_remove{end+1} = layer(i).name;
            end
        end
        net.removeLayer(to_remove);
         
    end