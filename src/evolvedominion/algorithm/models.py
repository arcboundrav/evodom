import numpy as np

from itertools import product
from evolvedominion.params import EPSILON, N_PIECE_IDENTITIES
from evolvedominion.types import FLOAT


class ModelCreator:
    """
    Subclass to introduce alternative model spaces for preference pmfs.
    """
    def __init__(self):
        pass

    def create_cache(self, models):
        assert models
        CACHE = dict()
        for i, model in enumerate(models):
            CACHE[i] = model
        return CACHE

    def get_full_model_space(self):
        raise NotImplementedError("Subclasses must define their method for generating a model space.")


class SimpleModelSpace(ModelCreator):
    """
    Produce the acquisition preference distributions used by evolved strategies.
    See the README for more details.
    """
    def __init__(self):
        self.basis = dict()
        self.basis[3] = [
            np.arange(35, 95, 5),
            np.arange(5, 50, 5),
            np.arange(5, 35, 5)
        ]
        self.basis[4] = [
            np.arange(25, 90, 5),
            np.arange(5, 50, 5),
            np.arange(5, 35, 5),
            np.arange(5, 30, 5)
        ]
        self.basis[5] = [
            np.arange(20, 85, 5),
            np.arange(5, 45, 5),
            np.arange(5, 35, 5),
            np.arange(5, 29, 5),
            np.arange(5, 25, 5)
        ]

    def _make_search_space(self, k):
        assert k in self.basis.keys()
        return product(*self.basis[k])

    def _pmf_predicate(self, combination):
        return sum(i for i in combination) == 100

    def _order_predicate(self, combination):
        return all((combination[i] >= combination[i+1]) for i in range(len(combination)-1))

    def _is_legal_pmf(self, combination):
        return self._pmf_predicate(combination) and self._order_predicate(combination)

    def _extract_legal_pmfs(self, k):
        return filter(self._is_legal_pmf, self._make_search_space(k))

    def _transform_legal_pmf(self, legal_pmf, N=N_PIECE_IDENTITIES):
        """\
            Ascending sort legal_pmf and embed it in an array of size N,
            shifted maximally to the right. Redistribute weight from the
            smallest element of legal_pmf, assigning a minimal value to
            each zero weight element in the new array so that it remains
            a valid pmf---i.e., sum(result) == sum(legal_pmf) == 1.0 and
            result[i] > 0.0 for all i in the closed interval [0:N-1].
        """
        k = len(legal_pmf)
        result = np.zeros(N, dtype=FLOAT)
        N_ = N - 1
        E = N - k

        for i in range(k):
            weight = legal_pmf[i] / 100.0
            j = N_ - i
            result[j] = weight

        result[j] = result[j] - (E * EPSILON)

        for i in range(E):
            result[i] = EPSILON
        return result

    def _get_model_space(self, k):
        return list(map(self._transform_legal_pmf, self._extract_legal_pmfs(k)))

    def get_full_model_space(self):
        models = []
        for model_size in self.basis:
            models.extend(self._extract_legal_pmfs(model_size))
        models.sort(key=lambda t: (t[0], t[2], t[1]))
        return [self._transform_legal_pmf(legal_pmf) for legal_pmf in models]
