     function [fixedLabel, instanceWeight] = create_labels(fixedLabelSize, pixel_dist)
         % This function creates groundtruth labels and instance weights
         % which are required by the loss function.
         assert(mod(fixedLabelSize(1),2)==1);
         label_size = fixedLabelSize(1);
         center = floor(label_size / 2) + 1; % center should be 8;
         M = 0;  % M denotes the number of positive pairs
         N = 0;  % and N those negative ones
         fixedLabel = single(zeros(label_size));
         center_p = [center center];

         % create fixedLabel, each element is either -1 or 1,depending on
         % its dist from the center
         for i = 1:label_size
             for j = 1:label_size
                 dist_from_center = dist([i j],center_p');
                 if dist_from_center <= pixel_dist
                     fixedLabel(i,j) = 1;
                     M = M + 1;
                 else
                     fixedLabel(i,j) = -1;
                     N = N + 1;
                 end
             end
         end
         % get the instanceWeight, all elements share the same value.
         instanceWeight = ones(label_size);
         instanceWeight(:) = 1 / (M * N);
     end