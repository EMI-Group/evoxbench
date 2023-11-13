from evoxbench.benchmarks import MoSegNASBenchmark

__all__ = ["citymop"]

# TODO: 需要一个硬件相关的指标（模型采集 功耗，是否指定GPU？）
# params | latency | FPS | flops | mIoU
def citymop(problem_id):
    if problem_id == 1:
        # Balance the speed and accuracy of the model.
        return MoSegNASBenchmark(objs="FPS&mIoU", normalized_objectives=False)
    elif problem_id == 2:
        # Balance the consumption and accuracy of the model.
        return MoSegNASBenchmark(
            objs="flops&mIoU", normalized_objectives=False
        )
    elif problem_id == 3:
        # Find the relationship between size and accuracy of the model.
        return MoSegNASBenchmark(
            objs="params&mIoU", normalized_objectives=False
        )
    elif problem_id == 4:
        # Find the relationship between size and speed of the model (including accuracy).
        return MoSegNASBenchmark(
            objs="latency&params&mIoU", normalized_objectives=False
        )
    elif problem_id == 5:
        # Find the relationship between speed and consumption of the model (including accuracy).
        return MoSegNASBenchmark(
            objs="latency&flops&mIoU", normalized_objectives=False
        )
    elif problem_id == 6:
        # Find the relationship between size, consumption and accuracy.
        return MoSegNASBenchmark(
            objs="params&flops&mIoU", normalized_objectives=False
        )
    elif problem_id == 7:
        # Balance all the hyperparameters.
        return MoSegNASBenchmark(
            objs="params&flops&latency&mIoU", normalized_objectives=False
        )
    else:
        raise ValueError("the requested problem id does not exist")
