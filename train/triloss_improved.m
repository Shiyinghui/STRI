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
   pos_index = find(c(:,:,:,1)==1);  % ���������� һ��(һ�ŵ÷�ͼ��)
   neg_index = find(c(:,:,:,1)==-1); % ���������� һ��
   M = numel(pos_index);  % ������������
   N = numel(neg_index);  % ������������
   
   if nargin <=2 || isempty(dzdy)  % forward, compute the triplet loss
       triplet_loss = 0;
       for num = 1:scoreSize(4)
           x_col = x(:,:,:,num); % ȡ��num�� score map
           x_col = x_col(:);   % �ų�һ��  
           pos_score = deal(x_col(pos_index));  % ���е�����ֵ�� �ų�һ�У�M��
           neg_score = deal(x_col(neg_index));  % ���еĸ���ֵ�� �ų�һ�У�N��
           POS1 = repmat(pos_score, [1 N]); % �ظ�N�У� �õ� M * N
           NEG1 = repmat(neg_score', [M 1]); % ��ת���ų� 1��N�У� �ظ�M�У��õ� M * N;
           temp_loss = log(1 + exp(NEG1 - POS1)); % temp_loss�� M * N ����
           temp_loss = sum(temp_loss(:));  % ת��Ϊ��ֵ
           triplet_loss = triplet_loss + temp_loss;
       end
       outputs = triplet_loss / (M * N);  
   else                            %backward, compute the derivatives
        dzdy = dzdy * instanceWeights;
        outputs = single(gpuArray(zeros(scoreSize(1), scoreSize(2), scoreSize(3),scoreSize(4)))); % ��ʼ��
        for num = 1:scoreSize(4)
           x_col = x(:,:,:,num); % ȡ��num�� score map
           x_col = x_col(:);   % �ų�һ��  
           pos_score = deal(x_col(pos_index));  % ���е�����ֵ�� �ų�һ�У�M��
           neg_score = deal(x_col(neg_index));  % ���еĸ���ֵ�� �ų�һ�У�N��
           POS1 = repmat(pos_score, [1 N]); % �ظ�N�У� �õ� M * N
           NEG1 = repmat(neg_score', [M 1]); % ��ת���ų� 1��N�У� �ظ�M�У��õ� M * N;
           POS2 = repmat(pos_score', [N 1]); % ��ת���ų� 1��M�У� �ظ�N�У��õ� N * M;
           NEG2 = repmat(neg_score, [1 M]); % �ظ�M�У� �õ� N * M;
           pos_der = -1 * exp(NEG1) ./ (exp(NEG1) + exp(POS1));  % ����������ĵ���
           pos_der = sum(pos_der, 2); % ����Ϊ������һ�еĺͣ� �õ� M��1��
           neg_der = exp(NEG2) ./ (exp(NEG2) + exp(POS2));  % ����Ը����ĵ���
           neg_der = sum(neg_der, 2); % ����Ϊ������һ�еĺͣ� �õ� N��1��
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
