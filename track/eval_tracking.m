     function [new_pos, best_scale] = eval_tracking(net_x, s_x, score_id, z_features, x_crops, target_pos, window, p)
         %this function calculates new position of the target in a frame and returns 
         % the best scale for further calculation.
         net_x.eval({p.id_z_feat, z_features, 'instance', x_crops}); % do not calculate ders
         % reshape score size to 17*17*3
         responseMaps = reshape(net_x.vars(score_id).value, [p.scoreSize p.scoreSize p.numScale]);

         % initialize responseMapsUP with zeros, size: [17*16 17*16 3]
         responseMapsUP = gpuArray(single(zeros(p.scoreSize * p.responseUp, p.scoreSize * p.responseUp, p.numScale)));
         if p.numScale > 1
             current_scale_id = ceil(p.numScale / 2);  % current_scale_id :2
             best_scale = current_scale_id;
             best_peak = -Inf;
             for s = 1:p.numScale
                 if p.responseUp > 1  % p.responseUp = 16
                     responseMapsUP(:,:,s) = imresize(responseMaps(:,:,s), p.responseUp,'bicubic');
                 else
                     responseMapsUP(:,:,s) = responseMaps(:,:,s);
                 end
                 thisResponse = responseMapsUP(:,:,s);
                 if s~=current_scale_id, thisResponse = thisResponse * p.scalePenalty; end  % p.scalePenalty = 0.9745;
                 this_peak = max(thisResponse(:));
                 if this_peak > best_peak, best_peak = this_peak; best_scale = s; end % get the best scale
             end
             responseMap = responseMapsUP(:,:,best_scale);
         else
             responseMap = responseMapsUP;
             best_scale = 1;
         end
         responseMap = responseMap - min(responseMap(:));
         responseMap = responseMap / sum(responseMap(:));

         % apply windowing   p.wInfluence = 0.176;
         responseMap = (1-p.wInfluence) * responseMap + p.wInfluence * window;
         [r_max, c_max] = find(responseMap == max(responseMap(:)),1);
         [r_max, c_max] = avoid_empty_position(r_max, c_max, p);
         p_corr = [r_max, c_max];
         disp_instanceFinal = p_corr - ceil(p.scoreSize * p.responseUp / 2);
         disp_instanceInput = disp_instanceFinal * p.totalStride / p.responseUp;
         disp_instanceFrame = disp_instanceInput * s_x / p.instanceSize;
         new_pos = target_pos + disp_instanceFrame;
         end

         function [r_max, c_max] = avoid_empty_position(r_max, c_max, params)
         if isempty(r_max)    % if r_max or c_max is empty, then equal to 136
             % r_max = ceil(params.scoreSize/2);
             r_max = ceil(params.scoreSize * params.responseUp / 2);
         end
         if isempty(c_max)
             %c_max = ceil(params.scoreSize/2);
             c_max = ceil(params.scoreSize * params.responseUp / 2);
         end
     end