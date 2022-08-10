classdef C10MOP3 < EvoXBaseProblem % < PROBLEM
% <multi> <real> <large/none> <expensive/none>
    methods
        %% Default settings of the problem
        function Setting(obj)
            config.name = 'nats';
            config.args.fidelity = 90;
            config.args.objs = 'err&params&flops';
            config.args.dataset = 'cifar10';
            config.args.normalized_objectives = true;
            obj.Setting@EvoXBaseProblem(config);
        end
    end
end

