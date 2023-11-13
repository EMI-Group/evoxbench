from evoxbench.benchmarks import MoSegNASBenchmark

__all__ = ["citymop"]

# params | latency | FPS | flops | mIoU
def citymop(problem_id):
    if problem_id == 1:
        # Balance the accuracy and the speed of the model.
        return MoSegNASBenchmark(objs="FPS&mIoU", normalized_objectives=False)
    elif problem_id == 2:
        # Find the relationship between the amount of computation and the size of the model.
        return MoSegNASBenchmark(objs="params&flops", normalized_objectives=False)

    elif problem_id == 3:
        # Balance the consumption of computational resources and the speed of the model.
        return MoSegNASBenchmark(
            objs="flops&latency", normalized_objectives=False
        )
    elif problem_id == 4:
        # Balance the size of the model and the reference speed.
        return MoSegNASBenchmark(
            objs="params&latency", normalized_objectives=False
        )
    elif problem_id == 5:
        # Find the relationship between size and accuracy of the model.
        return MoSegNASBenchmark(
            objs="params&mIoU", normalized_objectives=False
        )
    elif problem_id == 6:
        # Find the relationship between size and speed of the model.
        return MoSegNASBenchmark(
            objs="latency&FPS&mIoU", normalized_objectives=False
        )
    elif problem_id == 7:
        # Find the relationship between size, consumption and speed.
        return MoSegNASBenchmark(
            objs="params&flops&latency", normalized_objectives=False
        )
    elif problem_id == 8:
        # Find the relationship between size, consumption and accuracy.
        return MoSegNASBenchmark(
            objs="params&flops&mIoU", normalized_objectives=False
        )
    elif problem_id == 9:
        # Balance all the hyperparameters.
        return MoSegNASBenchmark(
            objs="params&flops&mIoU&latency", normalized_objectives=False
        )
    else:
        raise ValueError("the requested problem id does not exist")
