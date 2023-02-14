% Preparation steps:
%   1. Follow the steps in the README.md file to download the materials and 
%      install and configure EvoXBench.
%   2. Download PlatEMO (https://github.com/BIMK/PlatEMO), the EvoXBench 
%      code in matlab is embedded in PlatEMO platform.
%   3. Run the command "evoxbenchrpc” in the terminal (In Window system, 
%      the evoxbench.exe executable file will be created and stored in the 
%      "Scripts" folder in your python environment).
%   4. Run PlatEMO. You can run this script directly or use the GUI to run.

addpath(genpath(cd));
clear;
clc;

pro = @C10MOP1;
problem = pro();
fileID = fopen("Data/NSGAII/c10mop1.json",'w'); % Folders need to be created in advance
result = [];
num_runs = 2;
for runs = 1 : num_runs
    [decs, objs, cons] = platemo('algorithm',@NSGAII,'problem',pro,'N',100,'maxFE',10000);
    problem.SetTest();                          % Set to test phase
    objs = problem.CalObj(decs);                % Calculate objective value on test set 
    result(runs).run = runs;
    result(runs).X = decs;
    result(runs).F = objs;
    population = SOLUTION(decs, objs, cons);
    igd = problem.CalMetric('IGD', population); % IGD can be caculated on C10MOP1-C10MOP7.
    hv = problem.CalMetric('HV', population);
    result(runs).IGD = igd;
    result(runs).HV = hv;
end
fprintf(fileID, '%s\n', jsonencode(result));
fclose(fileID);


