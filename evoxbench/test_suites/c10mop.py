from evoxbench.benchmarks import NASBench101Benchmark, NASBench201Benchmark, NATSBenchmark, DARTSBenchmark

__all__ = ['c10mop']


def c10mop(problem_id):
    if problem_id == 1:
        return NASBench101Benchmark(
            objs='err&params', normalized_objectives=True)
    elif problem_id == 2:
        return NASBench101Benchmark(
            objs='err&params&flops', normalized_objectives=True)
    elif problem_id == 3:
        return NATSBenchmark(
            90, objs='err&params&flops', dataset='cifar10', normalized_objectives=True)
    elif problem_id == 4:
        return NATSBenchmark(
            90, objs='err&params&flops&latency', dataset='cifar10', normalized_objectives=True)
    elif problem_id == 5:
        return NASBench201Benchmark(
            200, objs='err&params&flops&edgegpu_latency&edgegpu_energy', dataset='cifar10',
            normalized_objectives=True)
    elif problem_id == 6:
        return NASBench201Benchmark(
            200, objs='err&params&flops&eyeriss_latency&eyeriss_energy&eyeriss_arithmetic_intensity', dataset='cifar10',
            normalized_objectives=True)
    elif problem_id == 7:
        return NASBench201Benchmark(
            200, objs='err&params&flops&edgegpu_latency&edgegpu_energy'
                      '&eyeriss_latency&eyeriss_energy&eyeriss_arithmetic_intensity', dataset='cifar10',
            normalized_objectives=True)
    elif problem_id == 8:
        return DARTSBenchmark(
            objs='err&params', normalized_objectives=False)
    elif problem_id == 9:
        return DARTSBenchmark(
            objs='err&params&flops', normalized_objectives=False)
    else:
        raise ValueError("the requested problem id does not exist")
