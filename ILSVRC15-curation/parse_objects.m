
function parse_objects(root, folder_index)
    % 读取每一帧的bndbox信息,将一个视频的所有目标整合到一个文件中
    % folder_index: 0,1,2,3,4, 即a, b, c, d,e
	% e.g. parse_objects('/path/to/ILSVRC15/', 0)
% -------------------------------------------------------------------------------
    addpath(genpath('../..'));
    slash = '/';
    i = folder_index + 3;

    root_imgs = [root 'Data/VID/train'];
    dir_root_imgs = dir(root_imgs);
    dir_root_imgs = {dir_root_imgs.name};   % a, b, c, d,e，只保留路径
    % we always start from 3 because at 1 we have '.' and at 2 '..'
    level1 = [root_imgs slash dir_root_imgs{i}];  % 比如  D:/ILSVRC15/Data/VID/train/a
    dir_level1 = dir(level1);
    dir_level1 = {dir_level1.name};   % a文件夹里所有的视频序列
    dir_root_imgs_i = dir_root_imgs{i};
    % j iterates across videos
    for j=3:numel(dir_level1)  
        level2 = [root_imgs slash dir_root_imgs_i slash dir_level1{j}];  %level2 当前视频的路径
        dir_level2 = dir(level2);
        dir_level2 = {dir_level2.name}; % 视频里所有的文件，只保留文件名,包括帧和其注释
        if isdir(level2)
            new_file_video = [level2 '.txt']; % 取完整路径
            if exist(new_file_video,'file')~=2
            % create new file per video which lists all the objects
                fv = fopen(new_file_video, 'w'); % 没有则创建，fv, file identifier
                % k iterates across frames
                for k=3:numel(dir_level2)   % i:文件夹，j:视频，k:视频里的文件
                    fprintf('Processing i: %d j: %d k: %d\n', i-2, j-2, k-2); 
                    path = [root_imgs slash dir_root_imgs_i slash dir_level1{j} slash dir_level2{k}];
                    if ismember('.txt', path)
                        fin = fopen(path,'r');
                        % reading i, j, k, o, trackid, o_class, frame_sz(2), frame_sz(1), o_xmin, o_ymin, o_sz(2), o_sz(1), im_path
                        C = textscan(fin, '%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s');  % o: 一个注释文件中的目标信息行数，1 到 n 行
                        fclose(fin);

                        if ~isempty(C{8})
                            n_lines = numel(C{1});
                            trackids = C{5};
                            classes = C{6};
                            frame_w = C{7}(1);
                            frame_h = C{8}(1);
                            o_xmins = C{9};
                            o_ymins = C{10};
                            o_ws = C{11};
                            o_hs = C{12};
                            frame_paths = C{13};

                            for o=1:n_lines
                                if ~isempty(C{4}(o))  % 写入文件
                                    fprintf(fv, '%d,%d,%d,%d,%d,%d,%d,%d,%s\n', trackids(o), classes(o), frame_w, frame_h, o_xmins(o), o_ymins(o), o_ws(o), o_hs(o), frame_paths{o});
                                end
                            end
                        end
                    end
                end
                fclose(fv);
            end
        end
    end
end
