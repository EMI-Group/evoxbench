classdef IN1KMOP3 < EvoXBaseProblem % < PROBLEM
% <multi> <real> <large/none> <expensive/none>
    methods
        %% Default settings of the problem
        function Setting(obj)
            config.name = 'resnet';
            config.args.objs = 'err&params&flops';
            config.args.normalized_objectives = false;
            obj.Setting@EvoXBaseProblem(config);
        end
    end
end
