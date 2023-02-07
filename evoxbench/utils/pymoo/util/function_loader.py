import importlib


def get_functions():
    from evoxbench.utils.pymoo.util.nds.fast_non_dominated_sort import fast_non_dominated_sort
    from evoxbench.utils.pymoo.util.nds.efficient_non_dominated_sort import efficient_non_dominated_sort
    from evoxbench.utils.pymoo.util.nds.tree_based_non_dominated_sort import tree_based_non_dominated_sort
    from evoxbench.utils.pymoo.decomposition.util import calc_distance_to_weights
    from evoxbench.utils.pymoo.util.misc import calc_perpendicular_distance
    from evoxbench.utils.pymoo.util.hv import hv
    from evoxbench.utils.pymoo.util.stochastic_ranking import stochastic_ranking

    FUNCTIONS = {
        "fast_non_dominated_sort": {
            "python": fast_non_dominated_sort
        },
        "efficient_non_dominated_sort": {
            "python": efficient_non_dominated_sort
        },
        "tree_based_non_dominated_sort": {
            "python": tree_based_non_dominated_sort
        },
        "calc_distance_to_weights": {
            "python": calc_distance_to_weights
        },
        "calc_perpendicular_distance": {
            "python": calc_perpendicular_distance
        },
        "stochastic_ranking": {
            "python": stochastic_ranking
        },
        "hv": {
            "python": hv
        },

    }

    return FUNCTIONS


class FunctionLoader:
    # -------------------------------------------------
    # Singleton Pattern
    # -------------------------------------------------
    __instance = None

    @staticmethod
    def get_instance():
        if FunctionLoader.__instance is None:
            FunctionLoader.__instance = FunctionLoader()
        return FunctionLoader.__instance

    # -------------------------------------------------

    def __init__(self) -> None:
        super().__init__()

    def load(self, func_name=None, mode="python"):

        FUNCTIONS = get_functions()

        if func_name not in FUNCTIONS:
            raise Exception("Function %s not found: %s" % (func_name, FUNCTIONS.keys()))

        func = FUNCTIONS[func_name]
        if mode not in func:
            raise Exception("Module not available in %s." % mode)
        func = func[mode]

        # either provide a function or a string to the module (used for cython)
        if not callable(func):
            module = importlib.import_module(func)
            func = getattr(module, func_name)

        return func


def load_function(func_name=None, _type="python"):
    return FunctionLoader.get_instance().load(func_name, mode=_type)

