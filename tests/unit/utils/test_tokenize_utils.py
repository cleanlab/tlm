from tlm.utils.tokenize_utils import round_max_words


def test_round_max_words() -> None:
    # Less than 10
    assert round_max_words(9) == 9
    assert round_max_words(10) == 10
    # Less than 100
    assert round_max_words(11) == 10
    assert round_max_words(70) == 70
    assert round_max_words(99) == 90
    # Greater than 100
    assert round_max_words(100) == 100
    assert round_max_words(150) == 150
    assert round_max_words(151) == 150
    assert round_max_words(199) == 150
    assert round_max_words(200) == 200
    assert round_max_words(201) == 200
