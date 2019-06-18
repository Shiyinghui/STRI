function simple_gui3
    % this function creates a simple GUI for object tracking.
    f = figure('Visible','off','Position',[360,500,600,340]);
    f.Units = 'normalized';
    f.Name = 'object tracking';
    f.NumberTitle = 'off';
    f.MenuBar = 'none';
    f.Color = 'w';
   
    hsurf = uicontrol('Style','pushbutton',...
        'String','choose sequences','Position',[500,0,100,20],...
        'enable','on', 'Callback',@surfbutton_Callback);
    hsurf.Units = 'normalized';
    
    ha=axes('units','normalized','position',[0 0 1 1]);
    uistack(ha,'down');
    x=imread('gui_bg.jpg');
    image(x)
    colormap gray
    set(ha,'handlevisibility','off','visible','off');
    % Move the window to the center of the screen.
    movegui(f,'center')
    % Make the window visible.
    f.Visible = 'on';
end

function surfbutton_Callback(src, event)
    selected_dir = uigetdir('../demo_sequences');
    if isequal(selected_dir,0)
        disp('User selected Cancel');
    else
        %disp(['User selected ', selected_dir]);
        get_frame_path(selected_dir);
    end
   
end
 
function get_frame_path(selected_dir)
    strs = split(selected_dir, '\');
    seq_name = strs(end);  % sequence name
    dataset_name = strs(end-1);
    if strcmp(dataset_name, 'VOT-2015')
        imgs_path = fullfile(selected_dir, 'imgs');
        groundtruth_path = fullfile(selected_dir, 'groundtruth.txt');
    elseif strcmp(dataset_name, 'OTB-2013')
        imgs_path = fullfile(selected_dir, 'img');
        groundtruth_path = fullfile(selected_dir, 'groundtruth_rect.txt');
    else
        f = errordlg('sequence not found','Operation Error');
        return;
    end

    assert(~isempty(imgs_path) || ~isempty(groundtruth_path));
    imgs = dir(imgs_path);
    img_name = imgs(3).name;
    frame_path = fullfile(imgs_path, img_name);
    assert(~isempty(frame_path));
    show_frame(frame_path, groundtruth_path, dataset_name, seq_name);

end

function show_frame(frame_path, bbox_path, dataset_name,seq_name)
    img = imread(frame_path);
    ground_truth = csvread(bbox_path);
    region = ground_truth(1,:);
    total_frames = int2str(size(ground_truth, 1));
    [cx, cy, w, h] = get_axis_aligned_BB(region);
    target_pos = [cy cx];
    target_size = [h w];
    rect_pos = [target_pos([2 1]) - target_size([2 1])/2, target_size([2 1])];
    %----show original target
    img_original = img;
    img_original = insertShape(img_original, 'Rectangle', rect_pos, 'LineWidth',2, 'Color', 'green');
   % imwrite(img_original, 'Bolt.jpg');
    %imshow(img_original);
    %---syh
    title = ['initial frame:1/' total_frames];
    f1 = figure('Name',title,'NumberTitle','off', 'MenuBar','none',...
        'Position', [360,500,600,340]);
  
    figure(f1);
   
    text = ['video sequence: ', seq_name];
    h_text = uicontrol('Style','text','String', text,...
        'Position',[0,0,100,30]);

    h_track = uicontrol('Style','pushbutton',...
        'String','start tracking','Position',[500,0,100,20],...
        'enable','on','Callback',{@run, dataset_name, seq_name});
    h_locate_object = uicontrol('Style','pushbutton',...
        'String','locate object','Position',[5,288,70,20],...
         'Tooltip', 'after selection click '' locate done '' button',...
        'enable','on','Callback',@locate_object);
    h_locate_done =  uicontrol('Style','pushbutton',...
        'String','locate done','Position',[5,265,70,20],...
        'enable','on','Callback',{@locate_done, img});
    h_show_target =  uicontrol('Style','pushbutton',...
        'String','show target','Position',[5,242,70,20],...
         'Tooltip', 'show original target',...
        'enable','on','Callback',{@show_original, img, rect_pos});

%     h_locate_object = uicontrol('Style','pushbutton',...
%         'String','locate object','Position',[3,288,70,20],...
%         'enable','on' ,'Callback',@locate_object);
%     h_locate_done =  uicontrol('Style','pushbutton',...
%         'String','locate done','Position',[3,260,70,20],...
%         'enable','on' ,'Callback',{@locate_done, img});
%     h_show_target =  uicontrol('Style','pushbutton',...
%         'String','show target','Position',[3,232,70,20],...
%         'enable','on' ,'Callback',{@show_original, img, rect_pos});

    checkbox1 = uicontrol('Style','checkbox',...
        'String',{'SiamFC'},...
        'Position',[200,0,80,20],...
        'enable','on',...
        'Tooltip', 'tracking with SiamFC,bounding box: red rectangle',...
        'Callback',@cbox1_Callback)
    checkbox2 = uicontrol('Style','checkbox',...
        'String',{'STRI'},...
        'Position',[280,0,80,20],...
        'enable','on',...
        'Tooltip', 'tracking with STRI,bounding box: yellow rectangle',...
        'Callback',@cbox2_Callback)
    
    global handle_rect;
    global tracker1_enable;
    global tracker2_enable;
    global draw_rect;
    handle_rect = [];
    tracker1_enable = 0; 
    tracker2_enable = 0;
    draw_rect = [];
    draw_flag = 0;
    
    movegui(f1, 'center');
    imshow(img_original);
   
    function locate_object(src, event)
        if draw_flag == 0
            draw_flag = draw_flag + 1;
            %---add tips
           % helpdlg('After selection click '' locate done '' button');
            %---syh
            handle_rect = imrect();
        else
            return;
        end
    end

    function locate_done(src,event, img)
        if isempty(handle_rect) || ~isvalid(handle_rect)
            return;
        else
            if ~isempty(handle_rect)
                pos = handle_rect.getPosition();
                select_size = [round(pos(3)), round(pos(4))]; % w, h
                select_pos = [round(pos(1)), round(pos(2))];  % x, y
              %  fprintf('%d %d %d %d\n', select_pos(1), select_pos(2), select_size(1), select_size(2));
                handle_rect.delete();
                draw_flag = 0;
                draw_rect = [select_pos(1), select_pos(2), select_size(1), select_size(2)];
                img = insertShape(img, 'Rectangle', draw_rect, 'LineWidth',2, 'Color', 'green');
                imshow(img);
            end
        end
    end

    function show_original(src, event, img, rect)
        if draw_flag ~= 0
            return;
        end
        img = insertShape(img, 'Rectangle', rect, 'LineWidth',2, 'Color', 'green');
        imshow(img);
        draw_rect = [];
    end
    
end

function cbox1_Callback(src, event)
    global tracker1_enable ;
    tracker1_enable = src.Value;   
end

function cbox2_Callback(src, event)
    global tracker2_enable;
    tracker2_enable = src.Value;
end


function run(src,event,dataset_name,seq_name)
    global tracker1_enable;
    global tracker2_enable;
    global draw_rect;
    t1_enable = tracker1_enable;
    t2_enable = tracker2_enable;
    draw_rectangle = draw_rect;
   % fprintf('%d %d\n', t1_enable, t2_enable);
    if ~t1_enable && ~t2_enable
        f = warndlg('You must select one of the trackers', 'warning');
    end
    dataset_name = dataset_name{1};
    seq_name = seq_name{1};
    assert(~iscell(seq_name));
    if t1_enable || t2_enable
        if isempty(draw_rectangle)
            run_tracker(t1_enable,t2_enable, dataset_name, seq_name, true);
        else
          run_tracker_Arbi(t1_enable,t2_enable, dataset_name, seq_name, true, draw_rectangle);
        end
    end
end