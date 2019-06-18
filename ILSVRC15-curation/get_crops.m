
function get_crops(im,cy,cx,w,h, size_z, size_x, context_amount)

    % take bbox with context for the exemplar
    wc_z = w + context_amount*(w+h);
    hc_z = h + context_amount*(w+h);
    s_z = sqrt(single(wc_z*hc_z));
    scale_z = size_z / s_z;
    im_crop_z = get_subwindow_avg(im, [cy cx], [size_z size_z], [round(s_z) round(s_z)]);
    imwrite(im_crop_z, 'z.jpg');
 
    d_search = (size_x - size_z)/2;
    pad = d_search/scale_z;
    s_x = s_z + 2*pad;

    im_crop_x = get_subwindow_avg(im, [cy cx], [size_x size_x], [round(s_x) round(s_x)]);
    imwrite(im_crop_x, 'x.jpg');
  
end


function im_patch = get_subwindow_avg(im, pos, model_sz, original_sz)

    avg_chans = [mean(mean(im(:,:,1))) mean(mean(im(:,:,2))) mean(mean(im(:,:,3)))];

    if isempty(original_sz)
        original_sz = model_sz;
    end
    sz = original_sz;
    im_sz = size(im);
    %make sure the size is not too small
    assert(all(im_sz(1:2) > 2));
    c = (sz+1) / 2; 

    %check out-of-bounds coordinates, and set them to avg_chans
    context_xmin = round(pos(2) - c(2));
    context_xmax = context_xmin + sz(2) - 1;
    context_ymin = round(pos(1) - c(1));
    context_ymax = context_ymin + sz(1) - 1;
    left_pad = double(max(0, 1-context_xmin));
    top_pad = double(max(0, 1-context_ymin));
    right_pad = double(max(0, context_xmax - im_sz(2)));
    bottom_pad = double(max(0, context_ymax - im_sz(1)));

    context_xmin = context_xmin + left_pad;
    context_xmax = context_xmax + left_pad;
    context_ymin = context_ymin + top_pad;
    context_ymax = context_ymax + top_pad;

    if top_pad || left_pad
        R = padarray(im(:,:,1), [top_pad left_pad], avg_chans(1), 'pre');
        G = padarray(im(:,:,2), [top_pad left_pad], avg_chans(2), 'pre');
        B = padarray(im(:,:,3), [top_pad left_pad], avg_chans(3), 'pre');
        im = cat(3, R, G, B);
    end

    if bottom_pad || right_pad
        R = padarray(im(:,:,1), [bottom_pad right_pad], avg_chans(1), 'post');
        G = padarray(im(:,:,2), [bottom_pad right_pad], avg_chans(2), 'post');
        B = padarray(im(:,:,3), [bottom_pad right_pad], avg_chans(3), 'post');
        im = cat(3, R, G, B);
    end

    xs = context_xmin : context_xmax;
    ys = context_ymin : context_ymax;

    im_patch_original = im(ys, xs, :);
    if ~isequal(model_sz, original_sz)
        im_patch = imresize(im_patch_original, model_sz);
    else
        im_patch = im_patch_original;
    end
end
