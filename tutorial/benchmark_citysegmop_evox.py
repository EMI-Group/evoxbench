import evox
from evox import workflows, problems
from evox.monitors import StdMOMonitor
import jax
import jax.numpy as jnp
import numpy as np
import time
import argparse
import os
import json
from evoxbench.test_suites.citysegmop import citysegmop
import math

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

@jax.jit
def pop_transform(pop):
    return jnp.around(pop).astype(int)


def get_algorithms(moea, lb, ub, n_objs, pop_size):
    if moea == "nsga2":
        return evox.algorithms.NSGA2(
            lb=lb,
            ub=ub,
            n_objs=n_objs,
            pop_size=pop_size,
        )
    elif moea == "nsga3":
        return evox.algorithms.NSGA3(
            lb=lb,
            ub=ub,
            n_objs=n_objs,
            pop_size=pop_size,
        )
    elif moea == "moead":
        return evox.algorithms.MOEAD(
            lb=lb,
            ub=ub,
            n_objs=n_objs,
            pop_size=pop_size,
        )
    elif moea == "rvea":
        return evox.algorithms.RVEA(
            lb=lb,
            ub=ub,
            n_objs=n_objs,
            pop_size=pop_size,
        )
    elif moea == 'ibea':
        return evox.algorithms.IBEA(
            lb=lb,
            ub=ub,
            n_objs=n_objs,
            pop_size=pop_size,
        )
    elif moea == 'hype':
        return evox.algorithms.HypE(
            lb=lb,
            ub=ub,
            n_objs=n_objs,
            pop_size=pop_size,
        )
    else:
        raise NotImplementedError


def get_moea_settings(pid):
    if pid == 1 or pid == 6:
        return 2, 100
    elif pid == 2 or pid == 3 or pid == 7 or pid == 8 or pid == 11:
        return 3, 105
    elif pid == 4 or pid == 9:
        return 4, 120
    elif pid == 5 or pid == 10 or pid == 12:
        return 5, 126
    elif pid == 13 or pid == 14:
        return 6, 132
    elif pid == 15:
        return 7, 217


def run_moea(algorithm, problem, key, benchmark):
    monitor = StdMOMonitor()

    workflow = workflows.StdWorkflow(
        algorithm=algorithm,
        problem=problem,
        monitor=monitor,
        sol_transforms=[pop_transform],
    )

    state = workflow.init(key)

    for i in range(math.ceil((1e4 / algorithm.pop_size))):
        key, subkey = jax.random.split(key)
        state = workflow.step(state)

    pop = state.get_child_state("algorithm").population
    fit = state.get_child_state("algorithm").fitness
    pop = jnp.clip(pop, lb, ub)
    pop = np.array(pop, dtype=np.float32)
    pop = np.round(pop).astype(int)
    hv = benchmark.calc_perf_indicator(pop, 'hv')
    return hv, pop, fit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark CitySeg/MOP")
    parser.add_argument("--moea", type=str, default="nsga2", help="MOEA algorithm")
    parser.add_argument("--runs", type=int, default=31, help="number of runs to repeat")
    parser.add_argument("--problem", type=int, default=15, help="max problem range from 1 to --problem")
    args = parser.parse_args()
    key = jax.random.PRNGKey(42)

    for pid in range(1, args.problem + 1):
        problem = problems.evoxbench.CitySegMOP(pid)
        lb = jnp.array(problem.lb)
        ub = jnp.array(problem.ub)
        benchmark = citysegmop(pid)

        for r in range(1, args.runs + 1):
            folder_path = r'../data/{}/{}/'.format(args.moea, pid)
            file_name = r'citymop_problem{}_full_record_{}_{}th_run.json'.format(pid, args.moea, r)
            print("open({})".format(folder_path))
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            if os.path.exists(os.path.join(folder_path, file_name)):
                print("The file exists in:{}".format(os.path.join(folder_path, file_name)))
                continue
            file = open(os.path.join(folder_path, file_name), 'w')
            n_objs, pop_size = get_moea_settings(pid)
            algorithm = get_algorithms(args.moea, lb, ub, n_objs, pop_size)
            start = time.time()
            results = {}
            hv, pop, fit = run_moea(algorithm, problem, key, benchmark)
            end = time.time()
            time_elapsed = end - start
            key, subkey = jax.random.split(key)
            pop = pop.tolist()
            fit = fit.tolist()
            results.update({'hv': hv})
            results.update({'time': time_elapsed})
            results.update({'pop': pop})
            results.update({'fit': fit})
            results.update({'r': r})
            results.update({'n_objs': n_objs})
            results.update({'pop_size': pop_size})
            print(results)

            json.dump(results, file, indent=4, separators=(', ', ': '))
            file.close()
