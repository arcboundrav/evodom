from functools import reduce
from itertools import chain, combinations, tee, filterfalse


def section(X, f):
    """
    A function f: X -> Y defines an equivalence relation R on the set X
    under which x[i] and x[j] are equivalent when f(x[i]) == f(x[j]).
    R is known as the equivalence kernel of f.

    R can be used to partition X into Q, the quotient set of X by R,
    where Q is a set of disjoint subsets of X which are known as
    equivalence classes. Each equivalence class contains all of the
    elements of X which are mapped to the same value in Y by f.

    Each equivalence class in Q can be represented by any of its
    elements. To choose a representative element for each equivalence
    class is to define a function, s: Q -> X, known as a section.

    A subset of X containing one representative element for each
    equivalence class in Q is therefore the image of Q under some
    section s, and can be identified with that section.

    This routine returns the image of Q under an arbitrary section s
    i.e., a set containing one arbitrary representative element for
    each equivalence class in Q.

    Note: Expects f to return a hashable value.
    """
    Y = set()

    def add_representative(representatives, x):
        y = f(x)
        if (y not in Y):
            Y.add(y)
            representatives.append(x)
        return representatives

    return list(reduce(add_representative, X, []))


def unique_piece_combinations(piece_combos):
    """
    Filter out functionally equivalent combinations using the
    equivalence relation defined by a function M from a tuple
    of distinct Piece instances to a sorted tuple of integers
    representing their type.
    """
    def M(piece_combo):
        return tuple(sorted(piece.total_order_index for piece in piece_combo))
    result = section(X=piece_combos, f=M)
    return result


def partition(predicate, iterable):
    """
    Based on an example in the Python itertools documentation.

    Returns a 2-tuple P where:
        P[0] is a list of the elements in iterable for which
             predicate evaluates to True; and,
        P[1] is a list of the elements in iterable for which
             predicate evaluates to False.

    Note: The relative order of elements in iterable is preserved
          in both P[0] and P[1]. At most, only one of P[0] and P[1]
          can be an empty list.
    """
    t1, t2 = tee(iterable)
    return list(filter(predicate, t1)), list(filterfalse(predicate, t2))


def classify(types, instances):
    """
    Iteratively partition instances into P, a list containing
    1 <= n <= len(types) subsets such that P[i] contains each
    element in instances of type types[i].
    """
    P = list()
    remainder = list(instances)
    for type_ in types:
        match, remainder = partition(lambda x: isinstance(x, type_), remainder)
        P.append(match)
    return P


def _unique(pieces):
    toi = set()
    result = []
    for piece in pieces:
        if not(piece.total_order_index in toi):
            toi.add(piece.total_order_index)
            result.append(piece)
    return result


def get_pieces(iterable, unique=False, predicate=lambda x: True):
    """
    Uniqueness is defined in terms of Piece type.
    Expects predicate to be a unary function that returns a boolean.
    """
    toi = set()
    result = []
    source = filter(predicate, iterable)
    if unique:
        return _unique(source)
    return list(source)


def get_piece_combinations(pieces, kmin, kmax=None):
    """
    Return the union of the k-combinations of pieces for
    k in the closed interval [kmin, kmax]. By default, kmax
    resolves to len(pieces). Expects pieces is non-empty and
    that 1 <= kmin <= kmax.

    Example 1

    pieces0 = [Gold0, Gold1, Gold2]
    get_piece_combinations(pieces0, kmin=1, kmax=2) ->

    [[Gold0], [Gold0, Gold1], [Gold0, Gold1, Gold2]]

    Note: Only [Gold0] is included since [Gold1] and [Gold2]
    are equivalent from a gameplay perspective. [Gold0, Gold1]
    is included---not [Gold0, Gold2], nor [Gold1, Gold2]---for
    the same reason.

    Example 2

    pieces1 = [Curse, Moat0, Moat1, Gold]
    get_piece_combinations(pieces1, kmin=2, kmax=4) ->

    [[Curse, Moat0], [Curse, Gold], [Moat0, Moat1],
     [Moat0, Gold], [Curse, Moat0, Moat1], [Curse, Moat0, Gold],
     [Moat0, Moat1, Gold], [Curse, Moat0, Moat1, Gold]]
    """
    kmax = (len(pieces) + 1) if (kmax is None) else (kmax + 1)
    assert pieces and (1 <= kmin <= kmax)
    combos = set(chain.from_iterable(combinations(pieces, k) for k in range(kmin, kmax)))
    return unique_piece_combinations(combos)
