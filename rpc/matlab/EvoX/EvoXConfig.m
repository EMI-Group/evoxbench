classdef EvoXConfig
    methods(Static)
        function r = fromfile(name)
            CallStack = dbstack('-completenames');
            FullPath = fullfile(fileparts(CallStack(1).file), 'configs', name);
            content = fileread(FullPath, Encoding = "UTF-8");
            r = jsondecode(content);
        end
    end
end
