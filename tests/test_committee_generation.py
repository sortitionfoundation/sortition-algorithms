"""Tests for the committee generation algorithms using modern data structures."""

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations

import pytest

from sortition_algorithms import errors
from sortition_algorithms.committee_generation import (
    find_distribution_maximin,
    find_distribution_nash,
)
from sortition_algorithms.features import FeatureCollection, FeatureValueMinMax
from sortition_algorithms.people import People
from sortition_algorithms.utils import StrippedDict


def convert_categories_to_features(
    categories_dict: dict[str, dict[str, dict[str, int]]],
) -> FeatureCollection:
    """Convert old categories format to FeatureCollection."""
    features: FeatureCollection = defaultdict(dict)

    for feature_name, values_dict in categories_dict.items():
        for value_name, constraints in values_dict.items():
            min_val = constraints["min"]
            max_val = constraints["max"]
            min_flex = constraints.get("min_flex", min_val)
            max_flex = constraints.get("max_flex", max_val)

            features[feature_name][value_name] = FeatureValueMinMax(
                min=min_val,
                max=max_val,
                min_flex=min_flex,
                max_flex=max_flex,
            )

    return features


def convert_people_data(
    people_dict: dict[str, dict[str, str]],
    columns_data: dict[str, dict[str, str]],
    features: FeatureCollection,
) -> People:
    """Convert old people format to People object."""
    # Determine what columns to keep from the columns_data
    all_columns = set()
    for person_columns in columns_data.values():
        all_columns.update(person_columns.keys())
    columns_to_keep = list(all_columns)

    people = People(columns_to_keep)

    for person_id, features_dict in people_dict.items():
        # Merge feature data with columns data
        person_data = {**features_dict}
        if person_id in columns_data:
            person_data.update(columns_data[person_id])

        people.add(person_key=person_id, data=StrippedDict(person_data), features=features, row_number=0)

    return people


@dataclass
class Example:
    features: FeatureCollection
    people: People
    no_address_columns: list[str]
    address_columns: list[str]
    number_people_wanted: int


def create_example_from_old_format(
    categories: dict[str, dict[str, dict[str, int]]],
    people_dict: dict[str, dict[str, str]],
    columns_data: dict[str, dict[str, str]],
    number_people_wanted: int,
) -> Example:
    """Create Example from old test data format."""
    features = convert_categories_to_features(categories)
    people = convert_people_data(people_dict, columns_data, features)
    no_address_columns = []
    address_columns = ["home", "street"]  # Need exactly 2 columns for address checking

    return Example(
        features=features,
        people=people,
        no_address_columns=no_address_columns,
        address_columns=address_columns,
        number_people_wanted=number_people_wanted,
    )


# Create all test examples using the conversion utilities
example1 = create_example_from_old_format(
    {
        "age": {"child": {"min": 1, "max": 2}, "adult": {"min": 1, "max": 2}},
        "franchise": {
            "simpsons": {"min": 1, "max": 2},
            "ducktales": {"min": 1, "max": 2},
        },
    },
    {
        "lisa": {"age": "child", "franchise": "simpsons"},
        "marge": {"age": "adult", "franchise": "simpsons"},
        "louie": {"age": "child", "franchise": "ducktales"},
        "dewey": {"age": "child", "franchise": "ducktales"},
        "scrooge": {"age": "adult", "franchise": "ducktales"},
    },
    {
        "lisa": {"home": "1", "street": "A"},
        "marge": {"home": "3", "street": "A"},
        "louie": {"home": "2", "street": "A"},
        "dewey": {"home": "2", "street": "A"},
        "scrooge": {"home": "1", "street": "A"},
    },
    2,
)

example2 = create_example_from_old_format(
    {
        "age": {"child": {"min": 1, "max": 2}, "adult": {"min": 1, "max": 2}},
        "franchise": {
            "simpsons": {"min": 1, "max": 2},
            "ducktales": {"min": 1, "max": 2},
        },
    },
    {
        "lisa": {"age": "child", "franchise": "simpsons"},
        "marge": {"age": "adult", "franchise": "simpsons"},
        "louie": {"age": "child", "franchise": "ducktales"},
        "dewey": {"age": "child", "franchise": "ducktales"},
        "scrooge": {"age": "adult", "franchise": "ducktales"},
    },
    {
        "lisa": {"home": "1", "street": "A"},
        "marge": {"home": "3", "street": "A"},
        "louie": {"home": "1", "street": "A"},
        "dewey": {"home": "2", "street": "A"},
        "scrooge": {"home": "1", "street": "A"},
    },
    2,
)

example3 = create_example_from_old_format(
    {
        "f1": {"v1": {"min": 1, "max": 2}, "v2": {"min": 0, "max": 2}},
        "f2": {"v1": {"min": 1, "max": 2}, "v2": {"min": 0, "max": 2}},
        "f3": {"v1": {"min": 1, "max": 2}, "v2": {"min": 0, "max": 2}},
    },
    {
        "a": {"f1": "v1", "f2": "v1", "f3": "v1"},
        "b": {"f1": "v1", "f2": "v2", "f3": "v2"},
        "c": {"f1": "v2", "f2": "v1", "f3": "v2"},
        "d": {"f1": "v2", "f2": "v2", "f3": "v1"},
    },
    {
        "a": {"home": "1", "street": "A"},
        "b": {"home": "2", "street": "A"},
        "c": {"home": "3", "street": "A"},
        "d": {"home": "3", "street": "A"},
    },
    2,
)

example4 = create_example_from_old_format(
    {
        "f1": {
            "v1": {"min": 1, "max": 1, "min_flex": 0, "max_flex": 0},
            "v2": {"min": 0, "max": 2, "min_flex": 0, "max_flex": 0},
        },
        "f2": {
            "v1": {"min": 1, "max": 1, "min_flex": 0, "max_flex": 0},
            "v2": {"min": 0, "max": 2, "min_flex": 0, "max_flex": 0},
        },
        "f3": {
            "v1": {"min": 1, "max": 1, "min_flex": 0, "max_flex": 0},
            "v2": {"min": 0, "max": 2, "min_flex": 0, "max_flex": 0},
        },
    },
    {
        "a": {"f1": "v1", "f2": "v1", "f3": "v1"},
        "b": {"f1": "v1", "f2": "v2", "f3": "v2"},
        "c": {"f1": "v2", "f2": "v1", "f3": "v2"},
        "d": {"f1": "v2", "f2": "v2", "f3": "v1"},
    },
    {
        "a": {"home": "1", "street": "A"},
        "b": {"home": "2", "street": "A"},
        "c": {"home": "3", "street": "A"},
        "d": {"home": "3", "street": "A"},
    },
    2,
)

example5 = create_example_from_old_format(
    {
        "gender": {"female": {"min": 1, "max": 1}, "male": {"min": 4, "max": 4}},
        "political": {
            "liberal": {"min": 4, "max": 4},
            "conservative": {"min": 1, "max": 1},
        },
    },
    {
        "adam": {"gender": "male", "political": "liberal"},
        "brian": {"gender": "male", "political": "liberal"},
        "cameron": {"gender": "male", "political": "liberal"},
        "dave": {"gender": "male", "political": "liberal"},
        "elinor": {"gender": "female", "political": "liberal"},
        "frank": {"gender": "male", "political": "conservative"},
        "grace": {"gender": "female", "political": "conservative"},
    },
    {
        "adam": {"home": "1", "street": "A"},
        "brian": {"home": "2", "street": "A"},
        "cameron": {"home": "3", "street": "A"},
        "dave": {"home": "4", "street": "A"},
        "elinor": {"home": "5", "street": "A"},
        "frank": {"home": "6", "street": "A"},
        "grace": {"home": "7", "street": "A"},
    },
    5,
)

# Example 6 - complex case with many people
example6_categories: dict[str, dict[str, dict[str, int]]] = {
    "f1": {
        "v1": {"min": 15, "max": 46},  # at least 15 people from A & B together
        "v2": {"min": 15, "max": 46},  # at least 15 people from D & E together
        "v3": {"min": 0, "max": 46},
    },
    "f2": {
        "v1": {"min": 15, "max": 46},  # at least 15 people from A & C together
        "v2": {"min": 15, "max": 46},  # at least 15 people from D & F together
        "v3": {"min": 0, "max": 46},
    },
    "f3": {
        "v1": {"min": 15, "max": 46},  # at least 15 people from B & C together
        "v2": {"min": 15, "max": 46},  # at least 15 people from E & F together
        "v3": {"min": 0, "max": 46},
    },
}
example6_people: dict[str, dict[str, str]] = {}
for i in range(1, 11):
    example6_people["p" + str(i)] = {
        "f1": "v1",
        "f2": "v1",
        "f3": "v3",
    }  # 10 people of kind A
    example6_people["p" + str(i + 10)] = {
        "f1": "v1",
        "f2": "v3",
        "f3": "v1",
    }  # 10 people of kind B
    example6_people["p" + str(i + 20)] = {
        "f1": "v3",
        "f2": "v1",
        "f3": "v1",
    }  # 10 people of kind C
    example6_people["p" + str(i + 30)] = {
        "f1": "v2",
        "f2": "v2",
        "f3": "v3",
    }  # 10 people of kind D
    example6_people["p" + str(i + 40)] = {
        "f1": "v2",
        "f2": "v3",
        "f3": "v2",
    }  # 10 people of kind E
    example6_people["p" + str(i + 50)] = {
        "f1": "v3",
        "f2": "v2",
        "f3": "v2",
    }  # 10 people of kind F
example6_people["p61"] = {"f1": "v3", "f2": "v3", "f3": "v3"}  # 1 extra person
example6_columns_data: dict[str, dict[str, str]] = {
    pid: {"home": str(i % 10), "street": "A"} for i, pid in enumerate(example6_people.keys())
}
example6 = create_example_from_old_format(example6_categories, example6_people, example6_columns_data, 46)


def calculate_marginals(
    people: People, committees: list[frozenset[str]], probabilities: list[float]
) -> dict[str, float]:
    """Calculate marginal selection probabilities for each person."""
    marginals = dict.fromkeys(people, 0)
    for committee, prob in zip(committees, probabilities, strict=False):
        for pid in committee:
            marginals[pid] += prob
    return marginals


def probabilities_well_formed(probabilities: list[float], precision: int = 5) -> None:
    """Check that probabilities are well-formed."""
    assert len(probabilities) >= 1
    for prob in probabilities:
        assert prob >= 0
        assert prob <= 1
    prob_sum = sum(probabilities)
    assert abs(prob_sum - 1) < 10 ** (-precision)


def allocation_feasible(
    committee: frozenset[str],
    features: FeatureCollection,
    people: People,
    number_people_wanted: int,
    check_same_address_columns: list[str],
) -> None:
    """Check that an allocation is feasible."""
    assert len(committee) == len(set(committee))
    assert len(committee) == number_people_wanted
    for pid in committee:
        assert pid in people

    # Check feature constraints
    for feature_name, fvalues in features.items():
        for fvalue_name, fv_minmax in fvalues.items():
            num_value = sum(1 for pid in committee if people.get_person_dict(pid)[feature_name] == fvalue_name)
            assert num_value >= fv_minmax.min
            assert num_value <= fv_minmax.max

    # Check household constraints
    if check_same_address_columns:
        for id1, id2 in combinations(committee, r=2):
            person1_data = people.get_person_dict(id1)
            person2_data = people.get_person_dict(id2)
            address1 = [person1_data.get(col, "") for col in check_same_address_columns]
            address2 = [person2_data.get(col, "") for col in check_same_address_columns]
            assert address1 != address2


def distribution_okay(
    committees: list[frozenset[str]],
    probabilities: list[float],
    features: FeatureCollection,
    people: People,
    number_people_wanted: int,
    check_same_address_columns: list[str],
    precision: int = 5,
) -> None:
    """Check that a distribution is valid."""
    probabilities_well_formed(probabilities, precision)
    for committee in committees:
        allocation_feasible(
            committee,
            features,
            people,
            number_people_wanted,
            check_same_address_columns,
        )


# Maximin Algorithm Tests


@pytest.mark.slow
def test_maximin_no_address_example1() -> None:
    """Test maximin without address constraints on example1."""
    features = deepcopy(example1.features)
    people = example1.people
    address_columns = []
    number_people_wanted = example1.number_people_wanted

    committees, probabilities, _ = find_distribution_maximin(
        features,
        people,
        number_people_wanted,
        address_columns,
    )

    distribution_okay(
        committees,
        probabilities,
        features,
        people,
        number_people_wanted,
        address_columns,
    )

    # maximin is 1/3, can be achieved uniquely by
    # 1/3: {louie, marge}, 1/3: {dewey, marge}, 1/3: {scrooge, lisa}
    marginals = calculate_marginals(people, committees, probabilities)
    precision = 5
    assert abs(marginals["lisa"] - 1 / 3) < 10 ** (-precision)
    assert abs(marginals["scrooge"] - 1 / 3) < 10 ** (-precision)
    assert abs(marginals["louie"] - 1 / 3) < 10 ** (-precision)
    assert abs(marginals["dewey"] - 1 / 3) < 10 ** (-precision)
    assert abs(marginals["marge"] - 2 / 3) < 10 ** (-precision)


@pytest.mark.slow
def test_maximin_with_address_example1() -> None:
    """Test maximin with address constraints on example1."""
    features = deepcopy(example1.features)
    people = example1.people
    number_people_wanted = example1.number_people_wanted

    committees, probabilities, _ = find_distribution_maximin(
        features, people, number_people_wanted, example1.address_columns
    )

    distribution_okay(committees, probabilities, features, people, number_people_wanted, example1.address_columns)

    # Scrooge and Lisa can no longer be included. E.g. if Scrooge is included, we need a simpsons child for the
    # second position. Only Lisa qualifies, but lives in the same household. Unique maximin among everyone else is:
    # 1/2: {louie, marge}, 1/2: {dewey, marge}
    marginals = calculate_marginals(people, committees, probabilities)
    precision = 5
    assert abs(marginals["lisa"] - 0) < 10 ** (-precision)
    assert abs(marginals["scrooge"] - 0) < 10 ** (-precision)
    assert abs(marginals["louie"] - 1 / 2) < 10 ** (-precision)
    assert abs(marginals["dewey"] - 1 / 2) < 10 ** (-precision)
    assert abs(marginals["marge"] - 1) < 10 ** (-precision)


@pytest.mark.slow
def test_maximin_no_address_example3() -> None:
    """Test maximin without address constraints on example3."""
    features = deepcopy(example3.features)
    people = example3.people
    number_people_wanted = example3.number_people_wanted

    committees, probabilities, _ = find_distribution_maximin(
        features,
        people,
        number_people_wanted,
        example3.no_address_columns,
    )

    distribution_okay(
        committees,
        probabilities,
        features,
        people,
        number_people_wanted,
        example3.no_address_columns,
    )

    # maximin is 1/3, can be achieved uniquely by
    # 1/3: {a, b}, 1/3: {a, c}, 1/3: {a, d}
    marginals = calculate_marginals(people, committees, probabilities)
    precision = 5
    assert abs(marginals["a"] - 1) < 10 ** (-precision)
    assert abs(marginals["b"] - 1 / 3) < 10 ** (-precision)
    assert abs(marginals["c"] - 1 / 3) < 10 ** (-precision)
    assert abs(marginals["d"] - 1 / 3) < 10 ** (-precision)


@pytest.mark.slow
def test_maximin_no_address_example4_infeasible() -> None:
    """Test maximin on example4 which should be infeasible."""
    features = deepcopy(example4.features)
    people = example4.people
    number_people_wanted = example4.number_people_wanted

    # There are no feasible committees at all.
    with pytest.raises(errors.InfeasibleQuotasCantRelaxError):
        find_distribution_maximin(features, people, number_people_wanted, example4.no_address_columns)


@pytest.mark.slow
def test_maximin_no_address_example5() -> None:
    """Test maximin without address constraints on example5."""
    features = deepcopy(example5.features)
    people = example5.people
    number_people_wanted = example5.number_people_wanted

    committees, probabilities, _ = find_distribution_maximin(
        features,
        people,
        number_people_wanted,
        example5.no_address_columns,
    )

    distribution_okay(
        committees,
        probabilities,
        features,
        people,
        number_people_wanted,
        example5.no_address_columns,
    )

    # maximin is 1/2 (for individuals)
    marginals = calculate_marginals(people, committees, probabilities)
    precision = 5
    assert marginals["adam"] >= 1 / 2 - 1e-5
    assert marginals["brian"] >= 1 / 2 - 1e-5
    assert marginals["cameron"] >= 1 / 2 - 1e-5
    assert marginals["dave"] >= 1 / 2 - 1e-5
    assert marginals["frank"] >= 1 / 2 - 1e-5
    assert abs(marginals["elinor"] - 1 / 2) < 10 ** (-precision)
    assert abs(marginals["grace"] - 1 / 2) < 10 ** (-precision)


@pytest.mark.slow
def test_maximin_no_address_example6() -> None:
    """Test maximin without address constraints on example6."""
    features = deepcopy(example6.features)
    people = example6.people
    number_people_wanted = example6.number_people_wanted

    committees, probabilities, _ = find_distribution_maximin(
        features, people, number_people_wanted, example6.no_address_columns
    )

    distribution_okay(committees, probabilities, features, people, number_people_wanted, example6.no_address_columns)

    # The full maximin is 0 because p61 cannot be selected. But our algorithm should aim for the maximin among the
    # remaining agents, which means choosing everyone else with probability 46/60.
    marginals = calculate_marginals(people, committees, probabilities)
    precision = 5
    assert marginals["p61"] == 0
    for i in range(1, 61):
        assert abs(marginals["p" + str(i)] - 46 / 60) < 10 ** (-precision)


# Nash Algorithm Tests


@pytest.mark.slow
def test_nash_no_address_example4_infeasible() -> None:
    """Test Nash on example4 which should be infeasible."""
    features = deepcopy(example4.features)
    people = example4.people
    number_people_wanted = example4.number_people_wanted

    # There are no feasible committees at all.
    with pytest.raises(errors.InfeasibleQuotasCantRelaxError):
        find_distribution_nash(features, people, number_people_wanted, [])


@pytest.mark.slow
def test_nash_no_address_example5() -> None:
    """Test Nash without address constraints on example5."""
    features = deepcopy(example5.features)
    people = example5.people
    number_people_wanted = example5.number_people_wanted

    committees, probabilities, _ = find_distribution_nash(features, people, number_people_wanted, [])

    probabilities_well_formed(probabilities, precision=3)
    for committee in committees:
        allocation_feasible(
            committee,
            features,
            people,
            number_people_wanted,
            [],
        )

    # hand-calculated unique nash optimum
    marginals = calculate_marginals(people, committees, probabilities)
    precision = 3
    assert abs(marginals["adam"] - 6 / 7) < 10 ** (-precision)
    assert abs(marginals["brian"] - 6 / 7) < 10 ** (-precision)
    assert abs(marginals["cameron"] - 6 / 7) < 10 ** (-precision)
    assert abs(marginals["dave"] - 6 / 7) < 10 ** (-precision)
    assert abs(marginals["frank"] - 4 / 7) < 10 ** (-precision)
    assert abs(marginals["elinor"] - 4 / 7) < 10 ** (-precision)
    assert abs(marginals["grace"] - 3 / 7) < 10 ** (-precision)


@pytest.mark.slow
def test_nash_no_address_example6() -> None:
    """Test Nash without address constraints on example6."""
    features = deepcopy(example6.features)
    people = example6.people
    number_people_wanted = example6.number_people_wanted

    committees, probabilities, _ = find_distribution_nash(
        features, people, number_people_wanted, example6.no_address_columns
    )

    distribution_okay(
        committees,
        probabilities,
        features,
        people,
        number_people_wanted,
        example6.no_address_columns,
        precision=3,
    )

    # The full maximin is -∞ because p61 cannot be selected. But our algorithm should maximize the Nash welfare of
    # the remaining agents, which means choosing everyone else with probability 46/60.
    marginals = calculate_marginals(people, committees, probabilities)
    precision = 2
    assert marginals["p61"] == 0
    for i in range(1, 61):
        assert abs(marginals["p" + str(i)] - 46 / 60) < 10 ** (-precision)
