from evoxbench.benchmarks import ResNet50DBenchmark, MobileNetV3Benchmark, TransformerBenchmark

__all__ = ['in1kmop']


def in1kmop(problem_id):
    if problem_id == 1:
        return ResNet50DBenchmark(
            objs='err&params', normalized_objectives=False)
    elif problem_id == 2:
        return ResNet50DBenchmark(
            objs='err&flops', normalized_objectives=False)
    elif problem_id == 3:
        return ResNet50DBenchmark(
            objs='err&params&flops', normalized_objectives=False)
    elif problem_id == 4:
        return TransformerBenchmark(       
            objs='err&params', normalized_objectives=False)
    elif problem_id == 5:
        return TransformerBenchmark(
            objs='err&params', normalized_objectives=False)
    elif problem_id == 6:
        return TransformerBenchmark(
            objs='err&params', normalized_objectives=False)
    elif problem_id == 7:
        return MobileNetV3Benchmark(
            objs='err&params', normalized_objectives=False)
    elif problem_id == 8:
        return MobileNetV3Benchmark(
            objs='err&params&flops', normalized_objectives=False)
    elif problem_id == 9:
        return MobileNetV3Benchmark(
            objs='err&params&flops&latency', normalized_objectives=False)
    else:
        raise ValueError("the requested problem id does not exist")
