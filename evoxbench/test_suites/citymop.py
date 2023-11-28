from evoxbench.benchmarks import MoSegNASBenchmark

__all__ = ["citymop"]

# params | latency | FPS | flops | mIoU
def citymop(problem_id):
    if problem_id == 1:
        return MoSegNASBenchmark(objs="mIoU&energy_consumption&flops", normalized_objectives=False)
    elif problem_id == 2:
        return MoSegNASBenchmark(
            objs="mIoU&energy_consumption&utilmIoU", normalized_objectives=False
        )
    elif problem_id == 3:
        return MoSegNASBenchmark(
            objs="mIoU&flops&latency", normalized_objectives=False
        )
    elif problem_id == 4:
        return MoSegNASBenchmark(
            objs="mIoU&latency&params", normalized_objectives=False
        )
    elif problem_id == 5:
        return MoSegNASBenchmark(
            objs="mIoU&params&flops", normalized_objectives=False
        )
    elif problem_id == 6:
        return MoSegNASBenchmark(
            objs="mIoU&energy_consumption&temperature&util", normalized_objectives=False
        )
    elif problem_id == 7:
        return MoSegNASBenchmark(
            objs="mIoU&params&flops&latency", normalized_objectives=False
        )
    elif problem_id == 8:
        return MoSegNASBenchmark(
            objs="mIoU&params&latency&energy_consumption&util", normalized_objectives=False
        )
    elif problem_id == 9:
        return MoSegNASBenchmark(
            objs="mIoU&params&flops&latency&energy_consumption&temperature&util", normalized_objectives=False
        )
    else:
        raise ValueError("the requested problem id does not exist")
