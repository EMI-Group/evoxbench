classdef C10MOP10 < EvoXBaseProblem % < PROBLEM
% <multi> <real> <large/none> <expensive/none>
    methods
        %% Default settings of the problem
        function Setting(obj)
            config.name = 'darts';
            config.args.objs = 'err&flops';
            config.args.normalized_objectives = false;
            obj.Setting@EvoXBaseProblem(config);
        end
    end
end

