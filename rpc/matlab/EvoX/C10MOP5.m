classdef C10MOP5 < EvoXBaseProblem % < PROBLEM
% <multi> <real> <large/none> <expensive/none>
    methods
        %% Default settings of the problem
        function Setting(obj)
            config.name = 'nb201';
            config.args.fidelity = 200;
            config.args.objs = 'err&params&flops&edgegpu_latency';
            config.args.dataset = 'cifar10';
            config.args.normalized_objectives = true;
            obj.Setting@EvoXBaseProblem(config);
        end
    end
end

