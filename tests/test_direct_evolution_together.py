from direct_evolution_together import hamming_distance


def test_nim():
    assert hamming_distance([1, 2, 3], [1, 0, 3]) == 1
