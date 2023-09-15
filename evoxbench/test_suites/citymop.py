from evoxbench.benchmarks import MoSegNASBenchmark

__all__ = ['citymop']


def citymop(problem_id):
    if problem_id == 1:
        return MoSegNASBenchmark(
            objs = 'params&flops&mIoU&latency&FPS',
            normalized_objectives=False
        )
    elif problem_id == 2:
        return MoSegNASBenchmark(
            objs = 'params&mIoU&latency&FPS',
            normalized_objectives=False
        )
    elif problem_id == 3:
        return MoSegNASBenchmark(
            objs = 'flops&mIoU&latency&FPS',
            normalized_objectives=False
        )
    elif problem_id == 4:
        return MoSegNASBenchmark(
            objs = 'params&flops&mIoU',
            normalized_objectives=False
        )
    elif problem_id == 5:
        return MoSegNASBenchmark(
            objs = 'mIoU&latency&FPS',
            normalized_objectives=False
        )
    elif problem_id == 6:
        return MoSegNASBenchmark(
            objs = 'mIoU',
            normalized_objectives=False
        )
    else:
        raise ValueError("the requested problem id does not exist")
