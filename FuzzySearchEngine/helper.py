from rapidfuzz import fuzz


_fuzzratio = fuzz.ratio
_int = int


def fuzzy_ratio_distance(s1, s2):
    """
    Calculate the inverse similarity score between two strings using fuzzy ratio.

    Parameters
    ----------
    s1 : str
        First string to compare.
    s2 : str
        Second string to compare.

    Returns
    -------
    int
        The distance score between 0 and 100.
    """
    return 100 - _int(_fuzzratio(s1, s2))