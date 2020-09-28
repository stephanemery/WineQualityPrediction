from hypothesis import given, strategies as st

Values = st.integers()
SortedLists = st.lists(Values).map(sorted)

@given(ls=SortedLists, v=Values)
def test_insert_is_sorted(ls, v):
    """
    This is the inspiration from the original tutorial documention on Hypothesis.works

    We test the first invariant: binary_search should return an index such
    that inserting the value provided at that index would result in a sorted
    set.
    """
    ls.insert(binary_search(ls, v), v)
    assert is_sorted(ls)

