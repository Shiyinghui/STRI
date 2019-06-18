function run_tracker(t1_enable,t2_enable, dataset, video, visualization)
    % this function receives arguments for tracking from GUI, and adds environment 
    % paths and necessary settings to use MatConvNet.
    addpath('D:/Software/MATLAB/matconvnet/matlab');
    addpath('../train');
    root = vl_rootnn();
    addpath(genpath(root));
    vl_setupnn;
    params.video = video;
    params.dataset = dataset;
    params.visualization = visualization;
    params.gpus = 1;
    params.siamfc = t1_enable;
    params.siamfc_tri = t2_enable;

    tracker(params);
end