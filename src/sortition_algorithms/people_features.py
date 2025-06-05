import csv
import random
import typing
from collections import defaultdict
from copy import deepcopy

from sortition_algorithms.features import FeatureCollection
from sortition_algorithms.people import People
from sortition_algorithms.settings import Settings


class PeopleFeatures:
    """
    This class manipulates people and features together, making a deepcopy on init.
    """

    # TODO: consider naming: maybe SelectionState

    def __init__(self, people: People, features: FeatureCollection) -> None:
        self.people = deepcopy(people)
        self.features = deepcopy(features)

    def update_features_remaining(self, person_key: str) -> None:
        # this will blow up if the person does not exist
        person = self.people.get_person_dict(person_key)
        for feature_name in self.features.feature_names:
            self.features.add_remaining(feature_name, person[feature_name])

    def update_all_features_remaining(self) -> None:
        for person_key in self.people:
            self.update_features_remaining(person_key)

    def delete_all_with_feature_value(
        self,
        feature_name: str,
        feature_value: str,
    ) -> tuple[int, int]:
        """
        When a feature is full we want to delete everyone in it.
        Returns count of those deleted, and count of those left
        """
        people_to_delete: list[str] = []
        for pkey in self.people:
            person = self.people.get_person_dict(pkey)
            if person[feature_name] == feature_value:
                people_to_delete.append(pkey)
                # adjust the features "remaining" values for each feature in turn
                for feature in self.features.feature_names:
                    self.features.remove_remaining(feature, person[feature])
        for p in people_to_delete:
            self.people.remove(p)
        # return the number of people deleted and the number of people left
        return len(people_to_delete), self.people.count

    def prune_for_feature_max_0(self) -> list[str]:
        """
        Check if any feature_value.max is set to zero. if so delete everyone with that feature value
        NOT DONE: could then check if anyone is left.
        """
        # TODO: when do we want to do this?
        msg: list[str] = []
        msg.append(f"Number of people: {self.people.count}.")
        total_num_deleted = 0
        for (
            feature_name,
            feature_value,
            fv_counts,
        ) in self.features.feature_values_counts():
            if fv_counts.max == 0:  # we don't want any of these people
                # pass the message in as deleting them might throw an exception
                msg.append(f"Feature/value {feature_name}/{feature_value} full - deleting people...")
                num_deleted, num_left = self.delete_all_with_feature_value(feature_name, feature_value)
                # if no exception was thrown above add this bit to the end of the previous message
                msg[-1] += f" Deleted {num_deleted}, {num_left} left."
                total_num_deleted += num_deleted
        # if the total number of people deleted is lots then we're probably doing a replacement selection, which means
        # the 'remaining' file will be useless - remind the user of this!
        if total_num_deleted >= self.people.count / 2:
            msg.append(
                ">>> WARNING <<< That deleted MANY PEOPLE - are you doing a "
                "replacement? If so your REMAINING FILE WILL BE USELESS!!!"
            )
        return msg


class WeightedSample:
    def __init__(self, features: FeatureCollection) -> None:
        """
        This produces a set of lists of feature values for each feature.  Each value
        is in the list `fv_counts.max` times - so a random choice with represent the max.

        So if we had feature "ethnicity", value "white" w max 4, "asian" w max 3 and
        "black" with max 2 we'd get:

        ["white", "white", "white", "white", "asian", "asian", "asian", "black", "black"]

        Then making random choices from that list produces a weighted sample.
        """
        self.weighted = defaultdict(list)
        for feature_name, value, fv_counts in features.feature_values_counts():
            self.weighted[feature_name] += [value] * fv_counts.max

    def value_for(self, feature_name: str) -> str:
        # S311 is random numbers for crypto - but this is just for a sample file
        return random.choice(self.weighted[feature_name])  # noqa: S311


def create_readable_sample_file(
    features: FeatureCollection,
    people_file: typing.TextIO,
    number_people_example_file: int,
    settings: Settings,
):
    example_people_writer = csv.writer(
        people_file,
        delimiter=",",
        quotechar='"',
        quoting=csv.QUOTE_MINIMAL,
    )
    example_people_writer.writerow([settings.id_column, *settings.columns_to_keep, *features.feature_names])
    weighted = WeightedSample(features)
    for x in range(number_people_example_file):
        row = [
            f"p{x}",
            *[f"{col} {x}" for col in settings.columns_to_keep],
            *[weighted.value_for(f) for f in features.feature_names],
        ]
        example_people_writer.writerow(row)
