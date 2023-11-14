from evoxbench.benchmarks import MoSegNASBenchmark

__all__ = ["citymop"]

# params | latency | FPS | flops | mIoU
def citymop(problem_id):
    if problem_id == 1:
        # Balance the accuracy and reference speed.
        return MoSegNASBenchmark(objs="mIoU&FPS", normalized_objectives=False)
    elif problem_id == 2:
        # Balance the accuracy and consumption of the model.
        return MoSegNASBenchmark(
            objs="mIoU&flops", normalized_objectives=False
        )
    elif problem_id == 3:
        # Find the relationship between accuracy and size of the model.
        return MoSegNASBenchmark(
            objs="mIoU&params", normalized_objectives=False
        )
    elif problem_id == 4:
        # Find the relationship between size and speed of the model (including accuracy).
        return MoSegNASBenchmark(
            objs="mIoU&latency&params", normalized_objectives=False
        )
    elif problem_id == 5:
        # Find the relationship between speed and consumption of the model (including accuracy).
        return MoSegNASBenchmark(
            objs="mIoU&latency&flops", normalized_objectives=False
        )
    elif problem_id == 6:
        # Find the relationship between accuracy, size and consumption.
        return MoSegNASBenchmark(
            objs="mIoU&params&flops", normalized_objectives=False
        )
    elif problem_id == 7:
        # Balance all the objectives.
        return MoSegNASBenchmark(
            objs="mIoU&params&flops&latency", normalized_objectives=False
        )
    else:
        raise ValueError("the requested problem id does not exist")
