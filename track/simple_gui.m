function simple_gui
    % this function creates a simple GUI for object tracking.
    f = figure('Visible','off','Position',[360,500,600,340]);
    hsurf    = uicontrol('Style','pushbutton',...
        'String','choose sequences','Position',[500,0,100,20],...
        'enable','on' ,'Callback',@surfbutton_Callback);

    f.Units = 'normalized';
    hsurf.Units = 'normalized';
    f.Name = 'object tracking';
    f.NumberTitle = 'off';
   % f.MenuBar = 'none';
    f.Color = 'w';
    % Move the window to the center of the screen.

    ha=axes('units','normalized','position',[0 0 1 1]);
    uistack(ha,'down');
    x=imread('gui_bg.jpg');
    image(x)
    colormap gray
    set(ha,'handlevisibility','off','visible','off');
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
   else 
       imgs_path = fullfile(selected_dir, 'img');
       groundtruth_path = fullfile(selected_dir, 'groundtruth_rect.txt');
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
    [cx, cy, w, h] = get_axis_aligned_BB(region);
    target_pos = [cy cx];
    target_size = [h w];
    rect_pos = [target_pos([2 1]) - target_size([2 1])/2, target_size([2 1])];
    img = insertShape(img, 'Rectangle', rect_pos, 'LineWidth',4, 'Color', 'green');
    f1 = figure('Name','initial frame','NumberTitle','off', 'MenuBar','none',...
        'Position', [360,500,600,340]);
    figure(f1);
    text = ['video sequence: ', seq_name];
    htext = uicontrol('Style','text','String', text,...
        'Position',[0,0,100,30]);

    hsurf = uicontrol('Style','pushbutton',...
        'String','start tracking','Position',[500,0,100,28],...
        'enable','on' ,'Callback',{@run, dataset_name, seq_name});

    bg = uibuttongroup('Visible','off',...
        'Title', 'select tracker',...
        'Position',[0.3, 0, 0.4, 0.1],...
        'SelectionChangedFcn',@bselection);

    % Create three radio buttons in the button group.
    r1 = uicontrol(bg,'Style',...
        'radiobutton',...
        'String','Default',...
        'Position',[25 0 80 20],...
        'HandleVisibility','off');

    r2 = uicontrol(bg,'Style','radiobutton',...
        'String','SiamFC',...
        'Position',[85 0 100 20],...
        'HandleVisibility','off');

    r3 = uicontrol(bg,'Style','radiobutton',...
        'String','SiamFC-tri',...
        'Position',[150 0 100 20],...
        'HandleVisibility','off');

    % Make the uibuttongroup visible after creating child objects.
    bg.Visible = 'on';

    imshow(img);
end

function bselection(src, event)
    global trackerID,
    if strcmp(event.NewValue.String, 'Default')
        trackerID = 1;
    elseif strcmp(event.NewValue.String, 'SiamFC')
        trackerID = 2;
    else
        trackerID = 3;
    end
    %fprintf('%d\n', trackerID);
end


function run(src,event,dataset_name,seq_name)
    global trackerID;
    tracker_id = trackerID;
    fprintf('%d\n', tracker_id);
    dataset_name = dataset_name{1};
    seq_name = seq_name{1};
    assert(~iscell(seq_name))
    run_tracker(tracker_id, dataset_name, seq_name, true);
end


