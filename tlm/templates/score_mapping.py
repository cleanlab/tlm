import logging

logger = logging.getLogger(__name__)


def score_5_mapping(x: str) -> float:
    return {
        "5": 1.0,
        "4": 0.75,
        "3": 0.5,
        "2": 0.25,
        "1": 0,
        "five": 1.0,
        "four": 0.75,
        "three": 0.5,
        "two": 0.25,
        "one": 0,
    }.get(x, 0.5)


def score_10_mapping(x: str) -> float:
    return {
        "10": 1.0,
        "9": 0.9,
        "8": 0.8,
        "7": 0.7,
        "6": 0.6,
        "5": 0.5,
        "4": 0.4,
        "3": 0.3,
        "2": 0.2,
        "1": 0.1,
        "0": 0.0,
        "ten": 1.0,
        "nine": 0.9,
        "eight": 0.8,
        "seven": 0.7,
        "six": 0.6,
        "five": 0.5,
        "four": 0.4,
        "three": 0.3,
        "two": 0.2,
        "one": 0.1,
        "zero": 0.0,
    }.get(x, 0.5)


def score_100_mapping(x: str) -> float:
    try:
        return float(x) / 100
    except ValueError:
        logger.info(f"Error mapping {x} to a float")
        return 0.5


def true_false_mapping(x: str | bool) -> float:
    return {
        "True": 1.0,
        "False": 0.0,
        "true": 1.0,
        "false": 0.0,
        True: 1.0,
        False: 0.0,
    }.get(x, 0.5)


def yes_no_mapping(x: str) -> float:
    return {
        "Yes": 1.0,
        "No": 0.0,
        "yes": 1.0,
        "no": 0.0,
    }.get(x, 0.5)


def ab_mapping(x: str) -> float:
    return {
        "A": 1.0,
        "B": 0.0,
        "a": 1.0,
        "b": 0.0,
    }.get(x, 0.5)


def certainty_mapping(x: str) -> float:
    try:
        x = x.lower()
    except ValueError:
        logger.info(f"Error getting .lower() of {x}")
        return 0.5

    return {
        "certain": 1.0,
        "mostly certain": 0.75,
        "somewhat certain": 0.5,
        "uncertain": 0.25,
        "likely incorrect": 0.0,
    }.get(x, 0.5)
