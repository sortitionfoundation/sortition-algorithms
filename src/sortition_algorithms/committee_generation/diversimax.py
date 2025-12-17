import itertools
import logging
from collections.abc import Iterable
from dataclasses import dataclass

import mip
import numpy as np
import pandas as pd
from mip import minimize, xsum
from sklearn.preprocessing import OneHotEncoder  # type: ignore[import-untyped]

from sortition_algorithms import errors
from sortition_algorithms.committee_generation.common import _relax_infeasible_quotas
from sortition_algorithms.features import FeatureCollection, iterate_feature_collection
from sortition_algorithms.people import People
from sortition_algorithms.utils import RunReport

InteractionNamesTuple = tuple[str, ...]


@dataclass
class IntersectionData:
    intersections_names: list[str]
    intersection_member_values: np.ndarray


@dataclass
class AllIntersectionsData:
    data: dict[InteractionNamesTuple, IntersectionData]
    all_dims_combs: list[InteractionNamesTuple]


def find_distribution_diversimax(
    features: FeatureCollection,
    people: People,
    number_people_wanted: int,
    check_same_address_columns: list[str],
) -> tuple[frozenset[str], RunReport]:
    """
    Find a committee using the Diversimax algorithm.
    """
    report = RunReport()
    report.add_line_and_log("Using Diversimax algorithm.", log_level=logging.INFO)
    optimizer = DiversityOptimizer(people, features, number_people_wanted, check_same_address_columns)
    model, selected_ids = optimizer.optimize()
    if model.status == mip.OptimizationStatus.OPTIMAL:
        report.add_line_and_log(
            f"Diversimax optimization successful. Selected {len(selected_ids)} participants.",
            log_level=logging.INFO,
        )
        return selected_ids, report
    elif model.status == mip.OptimizationStatus.INFEASIBLE:
        relaxed_features, output_lines = _relax_infeasible_quotas(
            features, people, number_people_wanted, check_same_address_columns
        )
        raise errors.InfeasibleQuotasError(relaxed_features, output_lines)
    else:
        msg = (
            f"No feasible committees found, solver returns code {model.status} (see "
            "https://docs.python-mip.com/en/latest/classes.html#optimizationstatus)."
        )
        raise errors.SelectionError(msg)


class DiversityOptimizer:
    def __init__(
        self, people: People, features: FeatureCollection, panel_size: int, check_same_address_columns: list[str]
    ):
        self.people = people
        # convert people to dataframe
        people_df = pd.DataFrame.from_dict(dict(people.items()), orient="index")
        self.pool_members_df = people_df[features.keys()]  # keep only feature columns
        self.features = features
        self.panel_size = panel_size
        self.check_same_address_columns = check_same_address_columns
        self.intersections_data: AllIntersectionsData = self.prepare_all_data()
        self.all_ohe = self.create_all_one_hot_encodings()

    @staticmethod
    def _intersection_name(category_names: Iterable[str]) -> str:
        return "__".join(category_names)

    def _prepare_features_data(self, features_names: tuple) -> IntersectionData:
        """
        Prepares the data for a given set of features.
        Makes all the possible intersections between the features' category names (e.g. male__18-24__highschool)
        Then for each person, figures out which intersection they belong to.
        """
        # Take all the combinations between the features' category names
        all_combinations = itertools.product(*[self.pool_members_df[d].unique() for d in features_names])
        intersections_names = [self._intersection_name(combination) for combination in all_combinations]
        # make the member value (join with __) for each person
        each_member_value = (
            self.pool_members_df.loc[:, features_names]
            .apply(lambda row: self._intersection_name([row[d] for d in features_names]), axis=1)
            .to_numpy()
        )
        return IntersectionData(intersections_names=intersections_names, intersection_member_values=each_member_value)

    def prepare_all_data(self) -> AllIntersectionsData:
        """
        For each combination of features, prepares the intersection data.
        The data is a dict where key is the combination of category names (e.g. (age, income, education level))
        and value is the IntersectionData for that combination.
        all_dims_combs is a list of all the intersections of all categories of all features.
        (e.g. [(age,), (income,), (education level,), (income, education level), (age, income, education level)])
        """
        all_dims_combs_iterators = [
            itertools.combinations(self.pool_members_df.columns, r=i)
            for i in range(1, self.pool_members_df.shape[1] + 1)
        ]
        all_dims_combs: list[InteractionNamesTuple] = [x for y in all_dims_combs_iterators for x in y]
        data = {}
        for features_intersections in all_dims_combs:
            try:
                data[features_intersections] = self._prepare_features_data(features_intersections)
            except Exception as e:
                raise Exception(f"Error with {features_intersections}") from e
        return AllIntersectionsData(data=data, all_dims_combs=all_dims_combs)

    def create_all_one_hot_encodings(self) -> list[np.ndarray]:
        """
        For every intersection of features - one hot encode who is in which intersection.
        Rows are the different people. Columns are the different possible intersections.
        The values are 0/1 if the person is in that intersection or not.
        i.e. The number of columns is the number of combinations we have between the features: product of their sizes.
        """
        all_ohe = []
        for dims in self.intersections_data.all_dims_combs:
            intersection_data: IntersectionData = self.intersections_data.data[dims]
            possible_profiles = intersection_data.intersections_names
            # When there are too many intersections, each will have 0/1 people, and optimization is pointless + too slow
            if len(possible_profiles) > self.panel_size:
                continue

            ohe = OneHotEncoder(categories=[possible_profiles], sparse_output=False)
            ohe_values = ohe.fit_transform(intersection_data.intersection_member_values.reshape(-1, 1))
            all_ohe.append(ohe_values)
        return all_ohe

    def optimize(self) -> tuple[mip.Model, frozenset]:
        """
        Uses MIP to optimize based on the categories constraints

        For the optimization goal, for every dims intersection:
        Take the one hot encoded of who is in which intersection for these dims
        Take the binary vector of who is selected and multiply and sum to get the sizes of each intersection of categories
        Figure out the "best" value - if all intersections were of equal size
        Take the abs for each intersection from that value
        Minimize sum of abs
        """
        df = self.pool_members_df
        m = mip.Model()
        # binary variable for each person - if they are selected or not
        model_variables = []
        for person_id in df.index:
            var = m.add_var(var_type="B", name=str(person_id))
            model_variables.append(var)
        model_variables_series = pd.Series(model_variables, index=df.index)

        # Household constraints: at most 1 person per household
        if self.check_same_address_columns:
            for housemates in self.people.households(self.check_same_address_columns).values():
                if len(housemates) > 1:
                    m.add_constr(mip.xsum(model_variables_series[member_id] for member_id in housemates) <= 1)

        # the sum of all people in each category must be between the min and max specified
        for feature_name, value_name, fv_minmax in iterate_feature_collection(self.features):
            relevant_members = model_variables_series[df[feature_name] == value_name]
            rel_sum = xsum(relevant_members)
            m.add_constr(rel_sum >= fv_minmax.min)
            m.add_constr(rel_sum <= fv_minmax.max)
        m.add_constr(xsum(model_variables_series) == self.panel_size)  # cannot exceed panel size

        # define the optimization goal
        all_objectives = []
        for ohe in self.all_ohe:  # for every set of intersections (one hot encoded) of features
            vals = np.asarray(model_variables_series.values)  # ensures it's ndarray
            # how many selected to each intersection
            intersection_sizes = (vals.reshape(-1, 1) * ohe).sum(axis=0)
            # the best value is if all intersections were equal size
            best_val = self.panel_size / ohe.shape[1]

            # set support variables that are the diffs from each intersection size to the best_val
            diffs_from_best_val = [m.add_var(var_type="C") for x in intersection_sizes]
            # constrain these support variables to be the abs diff from the intersection size
            for abs_diff, intersection_size in zip(diffs_from_best_val, intersection_sizes, strict=False):
                m.add_constr(abs_diff >= (intersection_size - best_val))
                m.add_constr(abs_diff >= (best_val - intersection_size))

            support_vars_sum = xsum(diffs_from_best_val)  # we will minimize the abs diffs
            all_objectives.append(support_vars_sum)

        obj = xsum(all_objectives)
        m.objective = minimize(obj)
        m.optimize()
        selected_ids = []
        for person_id in df.index:
            var = m.var_by_name(str(person_id))
            if var.x == 1:
                selected_ids.append(person_id)
        selected_ids_set = frozenset(selected_ids)

        return m, selected_ids_set
