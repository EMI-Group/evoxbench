addpath(genpath(cd));
clear;
clc;

problem_list = {@C10MOP1, @C10MOP2, @C10MOP3, @C10MOP4, @C10MOP5, ...
    @C10MOP6, @C10MOP7, @C10MOP8, @C10MOP9};
algorithms = {@NSGAII, @IBEA, @MOEAD, @NSGAIII, @HypE, @RVEAa};
algorithm_names = {'nsga2', 'ibea', 'moead', 'nsga3', 'hype', 'RVEAa'};
pop_size = {100, 105, 105, 120, 126, 132, 156, 100, 105};
num_runs = 2;  % the number of runs

for algidx = 1 : 6
    alg = algorithms{algidx};
    algname = algorithm_names{algidx};
    for proidx = 1 : 9
        fileID = fopen(sprintf("c10mop%d_%s.json", proidx, algname),'w');
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
            if proidx < 8
                % We only calculate IGD for C-10/MOP1 - C-10/MOP7 since 
                % the true Pareto Fronts are available.
                igd = problem.CalMetric('IGD', population);
                result(runs).IGD = igd;
            end
            hv = problem.CalMetric('HV', population);
            result(runs).HV = hv;
        end
        fprintf(fileID, '%s\n', jsonencode(result));
        fclose(fileID);
    end
end

