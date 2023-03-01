addpath(genpath(cd));
clear;
clc;

problem_list = {@IN1KMOP1, @IN1KMOP2, @IN1KMOP3, @IN1KMOP4, @IN1KMOP5, ...
    @IN1KMOP6, @IN1KMOP7, @IN1KMOP8, @IN1KMOP9};
algorithms = {@NSGAII, @IBEA, @MOEAD, @NSGAIII, @HypE, @RVEAa};
algorithm_names = {'nsga2', 'ibea', 'moead', 'nsga3', 'hype', 'RVEAa'};
pop_size = {100, 100, 105, 100, 100, 105, 100, 105, 120};
num_runs = 2;

for algidx = 1 : 6
    alg = algorithms{algidx};
    algname = algorithm_names{algidx};
    for proidx = 1 : 9
        fileID = fopen(sprintf("in1kmop%d_%s.json", proidx, algname),'w');
        pro = problem_list{proidx};
        N = pop_size{proidx};
        result = [];
        for runs = 1 :num_runs
            [decs, objs, cons] = platemo('algorithm', alg, 'problem', pro,'N', N, 'maxFE', 10000);
            problem = pro();
            problem.SetTest();
            objs = problem.CalObj(decs);
            result(runs).run = runs;
            result(runs).X = decs;
            result(runs).F = objs;
            population = SOLUTION(decs, objs, cons);
            hv = problem.CalMetric('HV', population);
            result(runs).HV = hv;
        end
        fprintf(fileID, '%s\n', jsonencode(result));
        fclose(fileID);
    end
end


