    function x_crops = get_scaled_xcrops(im, target_pos, scaled_instance, out_x, avg_chans, p)
         % this function obtains sub-windows cropped from its original image and scaled to 255*255.  
         % im: the following frame  target_pos:[cy cx]  out_x: 255
         % scaled instance: instances of three different sizes
         scaled_instance = round(scaled_instance);
         x_crops = gpuArray(zeros(out_x, out_x, 3, p.numScale, 'single')); % initialize
         max_target_side = scaled_instance(end);
         min_target_side = scaled_instance(1);
         beta = out_x / min_target_side;  
         search_side = round(beta * max_target_side); 
         
         % get the search region from an image                                                
         search_region = get_subwindow(im, target_pos, [search_side search_side], [max_target_side max_target_side], avg_chans);         
         assert(round(beta * min_target_side) == out_x);
      
         for s = 1:p.numScale
             target_side = round(beta * scaled_instance(s));             
             [x_crops(:,:,:,s)] = get_subwindow(search_region, (1+search_side*[1 1])/2, [out_x out_x], target_side*[1 1],avg_chans);
         end
    end