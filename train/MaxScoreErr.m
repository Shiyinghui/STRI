classdef MaxScoreErr < dagnn.Loss
    
    methods
        function outputs = forward(obj, inputs, params)
            score = inputs{1}; 
            label = inputs{2};
            [d1, d2, k, b] = size(score);
            assert(mod(d1, 2) == 1);  % d1 and d2 are odd numbers
            assert(mod(d2, 2) == 1);
            assert(k == 1); 
            assert(numel(label) == b); % the num of labels should be equal to batch size
            
            label = reshape(label, [1 1 1 b]);
            score = gather(score); % score is a gpu Array.
            
            pos = label > 0;
            center_score = score((d1+1) / 2, (d2+1) / 2, :, :); % size:[1 1 1 8]
            %max_score = max(max(score, [], 1), [], 2);  % size: [1 1 1 8] 4-D array
            error = zeros(b, 1);
            error(pos) = center_score(pos);
            outputs{1} = sum(error < 0); 
        end
        function obj = MaxScoreErr(varargin)
            obj.load(varargin);
        end
       
    end
end