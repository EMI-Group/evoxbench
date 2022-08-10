'''
Codes from [pymoo](https://pymoo.org/).
Use Performance Indicator only for IGD and HV metrics in Benchmark
'''

import re


# =========================================================================================================
# Generic
# =========================================================================================================


def get_from_list(l, name, args, kwargs):
    i = None

    for k, e in enumerate(l):
        if e[0] == name:
            i = k
            break

    if i is None:
        for k, e in enumerate(l):
            if re.match(e[0], name):
                i = k
                break

    if i is not None:

        if len(l[i]) == 2:
            name, clazz = l[i]

        elif len(l[i]) == 3:
            name, clazz, default_kwargs = l[i]

            # overwrite the default if provided
            for key, val in kwargs.items():
                default_kwargs[key] = val
            kwargs = default_kwargs

        return clazz(*args, **kwargs)
    else:
        raise Exception("Object '%s' for not found in %s" % (name, [e[0] for e in l]))


# =========================================================================================================
# Performance Indicator
# =========================================================================================================


def get_performance_indicator_options():
    from evoxbench.utils.pymoo.indicators.gd import GD
    from evoxbench.utils.pymoo.indicators.gd_plus import GDPlus
    from evoxbench.utils.pymoo.indicators.igd import IGD
    from evoxbench.utils.pymoo.indicators.igd_plus import IGDPlus
    from evoxbench.utils.pymoo.indicators.hv import Hypervolume
    from evoxbench.utils.pymoo.indicators.rmetric import RMetric

    PERFORMANCE_INDICATOR = [
        ("gd", GD),
        ("gd+", GDPlus),
        ("igd", IGD),
        ("igd+", IGDPlus),
        ("hv", Hypervolume),
        ("rmetric", RMetric)
    ]
    return PERFORMANCE_INDICATOR


def get_performance_indicator(name, *args, d={}, **kwargs):
    return get_from_list(get_performance_indicator_options(), name, args, {**d, **kwargs})
