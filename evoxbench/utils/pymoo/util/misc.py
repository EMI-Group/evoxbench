import numpy as np
import scipy
import scipy.spatial


def parameter_less(F, CV, fmax=None, inplace=False):
    assert len(F) == len(CV)

    if not inplace:
        F = np.copy(F)

    if fmax is None:
        fmax = np.max(F)

    param_less = fmax + CV

    infeas = (CV > 0).flatten()
    F[infeas] = param_less[infeas]

    return F


def swap(M, a, b):
    tmp = M[a]
    M[a] = M[b]
    M[b] = tmp


# repairs a numpy array to be in bounds
def repair(X, xl, xu):
    larger_than_xu = X[0, :] > xu
    X[0, larger_than_xu] = xu[larger_than_xu]

    smaller_than_xl = X[0, :] < xl
    X[0, smaller_than_xl] = xl[smaller_than_xl]

    return X


def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)] * a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


def parameter_less_constraints(F, CV, F_max=None):
    if F_max is None:
        F_max = np.max(F)
    has_constraint_violation = CV > 0
    F[has_constraint_violation] = CV[has_constraint_violation] + F_max
    return F


def random_permuations(n, l):
    perms = []
    for i in range(n):
        perms.append(np.random.permutation(l))
    P = np.concatenate(perms)
    return P


def get_duplicates(M):
    res = []
    I = np.lexsort([M[:, i] for i in reversed(range(0, M.shape[1]))])
    S = M[I, :]

    i = 0

    while i < S.shape[0] - 1:
        l = []
        while np.all(S[i, :] == S[i + 1, :]):
            l.append(I[i])
            i += 1
        if len(l) > 0:
            l.append(I[i])
            res.append(l)
        i += 1

    return res


# -----------------------------------------------
# Euclidean Distance
# -----------------------------------------------

def func_euclidean_distance(a, b):
    return np.sqrt(((a - b) ** 2).sum(axis=1))


def func_norm_euclidean_distance(xl, xu):
    return lambda a, b: np.sqrt((((a - b) / (xu - xl)) ** 2).sum(axis=1))


def norm_eucl_dist_by_bounds(A, B, xl, xu, **kwargs):
    return vectorized_cdist(A, B, func_dist=func_norm_euclidean_distance(xl, xu), **kwargs)


def norm_eucl_dist(problem, A, B, **kwargs):
    return norm_eucl_dist_by_bounds(A, B, *problem.bounds(), **kwargs)


# -----------------------------------------------
# Manhatten Distance
# -----------------------------------------------

def func_manhatten_distance(a, b):
    return np.abs(a - b).sum(axis=1)


def func_norm_manhatten_distance(xl, xu):
    return lambda a, b: np.abs((a - b) / (xu - xl)).sum(axis=1)


def norm_manhatten_dist_by_bounds(A, B, xl, xu, **kwargs):
    return vectorized_cdist(A, B, func_dist=func_norm_manhatten_distance(xl, xu), **kwargs)


def norm_manhatten_dist(problem, A, B, **kwargs):
    return norm_manhatten_dist_by_bounds(A, B, *problem.bounds(), **kwargs)


# -----------------------------------------------
# Tchebychev Distance
# -----------------------------------------------


def func_tchebychev_distance(a, b):
    return np.abs(a - b).max(axis=1)


def func_norm_tchebychev_distance(xl, xu):
    return lambda a, b: np.abs((a - b) / (xu - xl)).max(axis=1)


def norm_tchebychev_dist_by_bounds(A, B, xl, xu, **kwargs):
    return vectorized_cdist(A, B, func_dist=func_norm_tchebychev_distance(xl, xu), **kwargs)


def norm_tchebychev_dist(problem, A, B, **kwargs):
    return norm_tchebychev_dist_by_bounds(A, B, *problem.bounds(), **kwargs)


# -----------------------------------------------
# Others
# -----------------------------------------------


def cdist(A, B, **kwargs):
    return scipy.spatial.distance.cdist(A.astype(float), B.astype(float), **kwargs)


def vectorized_cdist(A, B, func_dist=func_euclidean_distance, fill_diag_with_inf=False, **kwargs) -> object:
    assert A.ndim <= 2 and B.ndim <= 2

    A, only_row = at_least_2d_array(A, extend_as="row", return_if_reshaped=True)
    B, only_column = at_least_2d_array(B, extend_as="row", return_if_reshaped=True)

    u = np.repeat(A, B.shape[0], axis=0)
    v = np.tile(B, (A.shape[0], 1))

    D = func_dist(u, v, **kwargs)
    M = np.reshape(D, (A.shape[0], B.shape[0]))

    if fill_diag_with_inf:
        np.fill_diagonal(M, np.inf)

    if only_row and only_column:
        M = M[0, 0]
    elif only_row:
        M = M[0]
    elif only_column:
        M = M[:, [0]]

    return M


def at_least_2d_array(x, extend_as="row", return_if_reshaped=False):
    if x is None:
        return x
    elif not isinstance(x, np.ndarray):
        x = np.array([x])

    has_been_reshaped = False

    if x.ndim == 1:
        if extend_as == "row":
            x = x[None, :]
        elif extend_as == "column":
            x = x[:, None]

        has_been_reshaped = True

    if return_if_reshaped:
        return x, has_been_reshaped
    else:
        return x


def all_except(x, *args):
    if len(args) == 0:
        return x
    else:
        H = set(args) if len(args) > 5 else args
        I = [k for k in range(len(x)) if k not in H]
        return x[I]


def all_combinations(A, B):
    u = np.repeat(A, B.shape[0], axis=0)
    v = np.tile(B, A.shape[0])
    return np.column_stack([u, v])


def evaluate_if_not_done_yet(evaluator, problem, pop, algorithm=None):
    I = np.where(pop.get("F") == None)[0]
    if len(I) > 0:
        pop[I] = evaluator.process(problem, pop[I], algorithm=algorithm)


def set_if_none(kwargs, str, val):
    if str not in kwargs:
        kwargs[str] = val


def set_if_none_from_tuples(kwargs, *args):
    for key, val in args:
        if key not in kwargs:
            kwargs[key] = val


def calc_perpendicular_distance(N, ref_dirs):
    u = np.tile(ref_dirs, (len(N), 1))
    v = np.repeat(N, len(ref_dirs), axis=0)

    norm_u = np.linalg.norm(u, axis=1)

    scalar_proj = np.sum(v * u, axis=1) / norm_u
    proj = scalar_proj[:, None] * u / norm_u[:, None]
    val = np.linalg.norm(proj - v, axis=1)
    matrix = np.reshape(val, (len(N), len(ref_dirs)))

    return matrix


def distance_of_closest_points_to_others(X):
    D = vectorized_cdist(X, X)
    np.fill_diagonal(D, np.inf)
    return D.argmin(axis=1), D.min(axis=1)
