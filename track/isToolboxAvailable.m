    function result = isToolboxAvailable(toolbox_name, action)
        % this function checks whether a toolbox is installed or not.
         if nargin < 1 || nargin > 2
             error('usage: isToolboxAvailable(toolbox_name,action)');
         elseif nargin == 1
             action = 'warning';
         end
         v_ = ver;
         [installedToolboxes{1:length(v_)}] = deal(v_.Name);
         result = all(ismember(toolbox_name, installedToolboxes));
         switch action
             case 'error'
                 assert(result, ['error!' toolbox_name 'is not installed!']);
             case 'warning'
                 if ~result
                     warning([toolbox_name 'is not installed!']);
                 end
             otherwise
         end        
    end