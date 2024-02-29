addpath(genpath(cd));
clear;
clc;

problem_list = {@CITYSEGMOP1, @CITYSEGMOP2, @CITYSEGMOP3, @CITYSEGMOP4, @CITYSEGMOP5, @CITYSEGMOP6, @CITYSEGMOP7, @CITYSEGMOP8, @CITYSEGMOP9, @CITYSEGMOP10, @CITYSEGMOP11, @CITYSEGMOP12, @CITYSEGMOP13, @CITYSEGMOP14, @CITYSEGMOP15};
algorithms = {@NSGAII, @NSGAIII, @MOEAD, @RVEAa, @SMSEMOA};
algorithm_names = {'nsga2', 'nsga3', 'moead', 'RVEAa', 'smsemoa'};
pop_size = {100, 105, 105, 120, 126, 100, 105, 105, 120, 126, 105, 126, 132, 132, 217};
num_runs = 31;  % the number of runs

for algidx = 1 : 5
    alg = algorithms{algidx};
    algname = algorithm_names{algidx};
    for proidx = 1 : 15
        fileID = fopen(sprintf("citysegmop%d_%s.json", proidx, algname),'w');
        pro = problem_list{proidx};
        N = pop_size{proidx};
        result = [];
        for runs = 1 : num_runs
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

