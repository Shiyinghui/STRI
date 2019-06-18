function [imgs, pos, target_size] = load_video_info(base_path, dataset, video)
     % This function returns the image array of a specified video, the center
     % of the bounding box in the first frame and the size of the
     % bounding box.
     if base_path(end) ~= '/' && base_path(end) ~= '\',base_path(end+1) = '/';
     end
     if strcmp(dataset, 'OTB-2013'), dataset_id = 1; 
     elseif strcmp(dataset, 'VOT-2015'), dataset_id = 2;
     else
         fprintf('unknown dataset!\n');
     end
     video_path = [base_path dataset '/' video];
     if dataset_id == 1
         ground_truth = csvread([video_path '/groundtruth_rect.txt']);
         img_path = [video_path '/img/'];
     else
         ground_truth = csvread([video_path '/groundtruth.txt']);
         img_path = [video_path '/imgs/'];
     end
     region = ground_truth(1,:);
     [cx, cy, w, h] = get_axis_aligned_BB(region);
     pos = [cy cx];
     target_size = [h w];
     
     images_path = [img_path '*.jpg'];
     img_files = dir(images_path);
     assert(~isempty(img_files), 'No images to read.');
     img_files = sort({img_files.name});
     img_files = strcat(img_path, img_files);
     imgs = vl_imreadjpeg(img_files, 'numThreads', 12);  % imgs is a cell.
     
end