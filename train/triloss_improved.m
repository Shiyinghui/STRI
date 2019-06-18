function outputs = triloss_improved(x,c,varargin)
% This function calculates the triplet loss of a batch, with x as the input
% response maps of the batch and c as the groundtruth labels. And if dzdy is
% not empty in a case of backward pass, the derivatives are also calculated.
    if ~isempty(varargin) && ~ischar(varargin{1})
        dzdy = varargin{1};
        varargin(1) = [];
    else
        dzdy = [];
    end
   opts.loss = 'logistic';
   opts.instanceWeights = [];
   opts = vl_argparse(opts,varargin);
   
   instanceWeights = [] ;

   if isa(x, 'gpuArray')
      switch classUnderlying(x) 
       case 'single', cast = @(z) single(z) ;
       case 'double', cast = @(z) double(z) ;
      end
   else
      switch class(x)
       case 'single', cast = @(z) single(z) ;
       case 'double', cast = @(z) double(z) ;
      end
   end
   
   hasIgnoreLabel = any(c(:) == 0);
   if hasIgnoreLabel
      % null labels denote instances that should be skipped
      instanceWeights = cast(c ~= 0) ;
   end

   if ~isempty(opts.instanceWeights)
  % important: this code needs to broadcast opts.instanceWeights to
  % an array of the same size as c
      if isempty(instanceWeights)
         instanceWeights = bsxfun(@times, onesLike(c), opts.instanceWeights) ;
      else
         instanceWeights = bsxfun(@times, instanceWeights, opts.instanceWeights);
      end
   end
  
   %assert((opts.loss == 'triplet'),'wrong loss type');
   scoreSize = [size(x,1) size(x,2) size(x,3) size(x,4)];
   labelSize = [size(c,1) size(c,2) size(c,3) size(c,4)];
   assert(isequal(scoreSize(1:2), labelSize(1:2)));
   assert(scoreSize(4)==labelSize(4));
   pos_index = find(c(:,:,:,1)==1);  % 正例索引， 一列(一张得分图中)
   neg_index = find(c(:,:,:,1)==-1); % 反例索引， 一列
   M = numel(pos_index);  % 正例索引个数
   N = numel(neg_index);  % 反例索引个数
   
   if nargin <=2 || isempty(dzdy)  % forward, compute the triplet loss
       triplet_loss = 0;
       for num = 1:scoreSize(4)
           x_col = x(:,:,:,num); % 取第num个 score map
           x_col = x_col(:);   % 排成一列  
           pos_score = deal(x_col(pos_index));  % 所有的正例值， 排成一列，M行
           neg_score = deal(x_col(neg_index));  % 所有的负例值， 排成一列，N行
           POS1 = repmat(pos_score, [1 N]); % 重复N列， 得到 M * N
           NEG1 = repmat(neg_score', [M 1]); % 先转置排成 1行N列， 重复M行，得到 M * N;
           temp_loss = log(1 + exp(NEG1 - POS1)); % temp_loss是 M * N 矩阵
           temp_loss = sum(temp_loss(:));  % 转化为数值
           triplet_loss = triplet_loss + temp_loss;
       end
       outputs = triplet_loss / (M * N);  
   else                            %backward, compute the derivatives
        dzdy = dzdy * instanceWeights;
        outputs = single(gpuArray(zeros(scoreSize(1), scoreSize(2), scoreSize(3),scoreSize(4)))); % 初始化
        for num = 1:scoreSize(4)
           x_col = x(:,:,:,num); % 取第num个 score map
           x_col = x_col(:);   % 排成一列  
           pos_score = deal(x_col(pos_index));  % 所有的正例值， 排成一列，M行
           neg_score = deal(x_col(neg_index));  % 所有的负例值， 排成一列，N行
           POS1 = repmat(pos_score, [1 N]); % 重复N列， 得到 M * N
           NEG1 = repmat(neg_score', [M 1]); % 先转置排成 1行N列， 重复M行，得到 M * N;
           POS2 = repmat(pos_score', [N 1]); % 先转置排成 1行M列， 重复N行，得到 N * M;
           NEG2 = repmat(neg_score, [1 M]); % 重复M列， 得到 N * M;
           pos_der = -1 * exp(NEG1) ./ (exp(NEG1) + exp(POS1));  % 计算对正例的导数
           pos_der = sum(pos_der, 2); % 以行为对象，求一行的和， 得到 M行1列
           neg_der = exp(NEG2) ./ (exp(NEG2) + exp(POS2));  % 计算对负例的导数
           neg_der = sum(neg_der, 2); % 以行为对象，求一行的和， 得到 N行1列
           der = single(gpuArray(zeros(M+N, 1))); 
           der(pos_index) = deal(pos_der);
           der(neg_index) = deal(neg_der);
           der = reshape(der, [scoreSize(1), scoreSize(2), scoreSize(3)]);
           outputs(:,:,:,num) = der;
       end
        outputs = dzdy .* outputs;
   end
end
  
 function y = onesLike(x)
   if isa(x,'gpuArray')
    y = gpuArray.ones(size(x),classUnderlying(x)) ; 
   else
    y = ones(size(x),'like',x) ;
   end
 end
