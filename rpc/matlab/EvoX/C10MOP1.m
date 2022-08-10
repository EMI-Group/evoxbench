classdef C10MOP1 < EvoXBaseProblem % < PROBLEM
% <multi> <real> <large/none> <expensive/none>
% EvoXBench
    methods
        %% Default settings of the problem
        function Setting(obj)
            config.name = 'nb101';
            config.args.objs = 'err&params';
            config.args.normalized_objectives = true;
            obj.Setting@EvoXBaseProblem(config);
        end
    end
end

