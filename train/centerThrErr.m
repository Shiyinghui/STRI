classdef centerThrErr < dagnn.Loss
    
    methods
        function outputs = forward(obj, inputs, params)
            score_map = inputs{1}; label = inputs{2};
            positive = label(:) > 0;
            score_map = score_map(:,:,:,positive);
            num_pos = sum(positive);
            center = floor(size(score_map,1)/2)+1;
            groundtruth = repmat(center*[1 1], [num_pos 1]);
            positions = zeros(num_pos,2);
            for i = 1:num_pos   % get positions
                score = gather(score_map(:,:,:,i));
                [rmax, cmax] = find(score == max(score(:)), 1);
                positions(i) = [rmax cmax];
            end
            
            n = obj.numAveraged;  % calculate center thresholds error
            m = n + num_pos;
            radiusInPixel = 50; total_stride = 8; n_step = 100;
            radius = radiusInPixel / total_stride;  
            thresholds = linspace(0, radius, n_step);
            errs = zeros(n_step, 1);  % initilization
            distances = sqrt((positions(:,1) - groundtruth(:,1)).^2 + (positions(:,2) - groundtruth(:,2)).^2);
            distances(isnan(distances)) = [];
            for i = 1:n_step
                errs(i) = nnz(distances > thresholds(i)); % nnz: number of nonzero matrix elements
            end
            outputs{1} = trapz(errs); % approximate integration
            obj.average = (n * obj.average + gather(outputs{1})) / m;
            obj.numAveraged = m;
            
        end
        function obj = centerThrErr(varargin)
          obj.load(varargin) ;
        end
    end
end