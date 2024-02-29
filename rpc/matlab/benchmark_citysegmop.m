addpath(genpath(cd));
clear;
clc;

problem_list = {@CITYMOP1, @CITYMOP2, @CITYMOP3, @CITYMOP4, @CITYMOP5, @CITYMOP6, @CITYMOP7, @CITYMOP8, @CITYMOP9, @CITYMOP10, @CITYMOP11, @CITYMOP12, @CITYMOP13, @CITYMOP14, @CITYMOP15};
algorithms = {@NSGAII, @NSGAIII, @MOEAD, @RVEAa, @SMSEMOA};
algorithm_names = {'nsga2', 'nsga3', 'moead', 'RVEAa', 'smsemoa'};
pop_size = {100, 105, 105, 120, 126, 100, 105, 105, 120, 126, 105, 126, 132, 132, 217};
num_runs = 31;  % the number of runs

for algidx = 1 : 6
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

