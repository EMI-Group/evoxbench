from evoxbench.benchmarks import MoSegNASBenchmark

__all__ = ["citymop"]


def citymop(problem_id):
    if problem_id == 1:
        return MoSegNASBenchmark(objs="err&h1_latency", normalized_objectives=False)
    elif problem_id == 2:
        return MoSegNASBenchmark(
            objs="err&h1_latency&flops", normalized_objectives=False
        )
    elif problem_id == 3:
        return MoSegNASBenchmark(
            objs="err&h1_latency&params", normalized_objectives=False
        )
    elif problem_id == 4:
        return MoSegNASBenchmark(
            objs="err&h1_latency&h1_energy_consumption&flops",
            normalized_objectives=False,
        )
    elif problem_id == 5:
        return MoSegNASBenchmark(
            objs="err&h1_latency&h1_energy_consumption&flops&params",
            normalized_objectives=False,
        )
    elif problem_id == 6:
        return MoSegNASBenchmark(objs="err&h2_latency", normalized_objectives=False)
    elif problem_id == 7:
        return MoSegNASBenchmark(
            objs="err&h2_latency&flops", normalized_objectives=False
        )
    elif problem_id == 8:
        return MoSegNASBenchmark(
            objs="err&h2_latency&params", normalized_objectives=False
        )
    elif problem_id == 9:
        return MoSegNASBenchmark(
            objs="err&h2_latency&h2_energy_consumption&flops",
            normalized_objectives=False,
        )
    elif problem_id == 10:
        return MoSegNASBenchmark(
            objs="err&h1_latency&h1_energy_consumption&flops&params",
            normalized_objectives=False,
        )
    elif problem_id == 11:
        return MoSegNASBenchmark(
            objs="err&h1_latency&h2_latency", normalized_objectives=False
        )
    elif problem_id == 12:
        return MoSegNASBenchmark(
            objs="err&h1_latency&h2_latency&h1_energy_consumption&h2_energy_consumption",
            normalized_objectives=False,
        )
    elif problem_id == 13:
        return MoSegNASBenchmark(
            objs="err&h1_latency&h2_latency&h1_energy_consumption&h2_energy_consumption&flops",
            normalized_objectives=False,
        )
    elif problem_id == 14:
        return MoSegNASBenchmark(
            objs="err&h1_latency&h2_latency&h1_energy_consumption&h2_energy_consumption&params",
            normalized_objectives=False,
        )
    elif problem_id == 15:
        return MoSegNASBenchmark(
            objs="err&h1_latency&h2_latency&h1_energy_consumption&h2_energy_consumption&flops&params",
            normalized_objectives=False,
        )
    else:
        raise ValueError(
            "the requested problem id does not exist! Please choose a problem id between 1 and 15."
        )
