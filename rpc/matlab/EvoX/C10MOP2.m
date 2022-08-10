classdef C10MOP2 < EvoXBaseProblem % < PROBLEM
% <multi> <real> <large/none> <expensive/none>
    methods
        %% Default settings of the problem
        function Setting(obj)
            config.name = 'nb101';
            config.args.objs = 'err&params&flops';
            config.args.normalized_objectives = true;
            obj.Setting@EvoXBaseProblem(config);
        end
    end
end

