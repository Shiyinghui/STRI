function im_patch = get_subwindow(im, pos, model_size, original_size, avg_chans)
     % This function obtains a sub-window of an image.
     % im: image array, the first frame of a video.
     % pos: (cy, cx), target position(the coordinates of the center) of the object.
     % model_size: expected size.
     % avg_chans: for padding use.
     if(isempty(original_size))
         original_size = model_size;
     end
     half = (original_size + 1)/2;
     im_size = size(im);
     
     % context_xmin,context_ymin,context_xmax and context_ymax are all scalars
     context_xmin = round(pos(2) - half(2));
     context_ymin = round(pos(1) - half(1));
     context_xmax = context_xmin + original_size(2) - 1;
     context_ymax = context_ymin + original_size(1) - 1;
     
     % if left_pad, top_pad, right_pad and bottom_pad are all equal to 0, 
     % then it means that an object with context is within the frame.   
     left_pad = max(0, 1-context_xmin);
     top_pad = max(0, 1-context_ymin);
     right_pad = max(0, context_xmax - im_size(2));
     bottom_pad = max(0, context_ymax - im_size(1));
     
     % if context_xmin <= 0, then context_xmin = 1, 
     % context_xmax = (context_xmin + original_size(2) - 1) + (1 - context_xmin) = original_size(2);
     % else context_xmin and context_xmax remain unchanged
     % above analysis also applies to context_ymin and context_ymax
     context_xmin = context_xmin + left_pad;  
     context_xmax = context_xmax + left_pad; 
     context_ymin = context_ymin + top_pad; 
     context_ymax = context_ymax + top_pad;
     
     if top_pad || left_pad
             R = padarray(im(:,:,1),[top_pad left_pad], avg_chans(1), 'pre');
             G = padarray(im(:,:,2),[top_pad left_pad], avg_chans(2), 'pre');
             B = padarray(im(:,:,3),[top_pad left_pad], avg_chans(3), 'pre');
             im = cat(3, R, G, B);
     end
     if bottom_pad || right_pad
             R = padarray(im(:,:,1),[bottom_pad right_pad], avg_chans(1), 'post');
             G = padarray(im(:,:,2),[bottom_pad right_pad], avg_chans(2), 'post');
             B = padarray(im(:,:,3),[bottom_pad right_pad], avg_chans(3), 'post');
             im = cat(3, R, G, B);
     end
     x = context_xmin:context_xmax;  % x and y are indexes.
     y = context_ymin:context_ymax;
     
     im_patch_original = im(y,x,:);
     if ~isequal(model_size, original_size)
         im_patch = imresize(im_patch_original, model_size(1)/original_size(1));
         if size(im_patch, 1) ~= model_size(1)
             im_patch = gpuArray(imresize(gather(im_patch_original), model_size));
         end
     else
         im_patch = im_patch_original;
     end
end