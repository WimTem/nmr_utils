from itertools import combinations


def find_pairs(atoms, cutoff, element1=None, element2=None):
    """
    Find all unique atom pairs within the cutoff.

    If element1 and element2 are given, only those pairs are returned.
    """

    distances = atoms.get_all_distances(mic=True)

    pairs = []

    for i, j in combinations(range(len(atoms)), 2):

        if distances[i, j] > cutoff:
            continue

        s1 = atoms[i].symbol
        s2 = atoms[j].symbol

        if element1 is not None and element2 is not None:

            if not (
                (s1 == element1 and s2 == element2)
                or
                (s1 == element2 and s2 == element1)
            ):
                continue

        pairs.append((i, j))

    return pairs