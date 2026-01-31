# ABOUTME: Diversimax algorithm for committee generation.
# ABOUTME: Maximizes diversity across intersections of demographic features.

import itertools
import logging
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder  # type: ignore[import-untyped]

from sortition_algorithms import errors
from sortition_algorithms.committee_generation.common import _relax_infeasible_quotas
from sortition_algorithms.committee_generation.solver import SolverSense, SolverStatus, create_solver, solver_sum
from sortition_algorithms.features import FeatureCollection, iterate_feature_collection
from sortition_algorithms.people import People
from sortition_algorithms.utils import RunReport, logger, random_provider

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
    max_seconds: int = 30,
    solver_backend: str = "highspy",
) -> tuple[frozenset[str], RunReport]:
    """
    Find a committee using the Diversimax algorithm.

    Args:
        features: FeatureCollection with min/max quotas
        people: People object with pool members
        number_people_wanted: desired size of the panel
        check_same_address_columns: columns to check for same address, or empty list if
                                    not checking addresses.
        max_seconds: maximum seconds to spend searching
        solver_backend: solver backend to use ("highspy" or "mip")

    Returns:
        tuple of (selected_ids, report)
    """
    report = RunReport()
    report.add_line_and_log("Using Diversimax algorithm.", log_level=logging.INFO)
    optimizer = DiversityOptimizer(people, features, number_people_wanted, check_same_address_columns, solver_backend)
    optimizer.log_problem_stats()

    status, selected_ids, gap = optimizer.optimize(max_seconds=max_seconds)

    # ===== HANDLE BOTH OPTIMAL AND FEASIBLE AS SUCCESS =====
    if status in [SolverStatus.OPTIMAL, SolverStatus.FEASIBLE]:
        if status == SolverStatus.OPTIMAL:
            report.add_line_and_log(
                f"Diversimax optimization successful (optimal). Selected {len(selected_ids)} participants.",
                log_level=logging.INFO,
            )
        else:  # FEASIBLE
            report.add_line_and_log(
                f"Diversimax optimization successful (feasible, gap {gap:.1%}). Selected {len(selected_ids)} participants.",
                log_level=logging.INFO,
            )
        return selected_ids, report

    elif status == SolverStatus.INFEASIBLE:
        relaxed_features, output_lines = _relax_infeasible_quotas(
            features, people, number_people_wanted, check_same_address_columns, solver_backend=solver_backend
        )
        raise errors.InfeasibleQuotasError(relaxed_features, output_lines)

    else:
        msg = f"No feasible committees found, solver returns status {status}."
        raise errors.SelectionError(msg)


class DiversityOptimizer:
    def __init__(
        self,
        people: People,
        features: FeatureCollection,
        panel_size: int,
        check_same_address_columns: list[str],
        solver_backend: str = "highspy",
    ):
        self.people = people
        # convert people to dataframe
        people_df = pd.DataFrame.from_dict(dict(people.items()), orient="index")
        people_df = people_df.rename(columns=str.lower)
        people_df = people_df.map(lambda x: x.lower() if isinstance(x, str) else x)  # normalize to lower case

        self.pool_members_df = people_df[(k.lower() for k in features)]  # keep only feature columns
        self.features = features
        self.panel_size = panel_size
        self.check_same_address_columns = check_same_address_columns
        self.solver_backend = solver_backend
        self.intersections_data: AllIntersectionsData = self.prepare_all_data()
        self.all_ohe = self.create_all_one_hot_encodings()

    @staticmethod
    def _intersection_name(category_names: Iterable[str]) -> str:
        return "__".join(category_names)

    def _prepare_features_data(self, features_names: InteractionNamesTuple) -> IntersectionData:
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
        # randomize order within each size, so if we cutoff intersections at len(all_ohe) >= 50, we get a random sample
        rng = np.random.default_rng(random_provider().randint())
        rng.shuffle(all_dims_combs)
        all_dims_combs = sorted(all_dims_combs, key=len)
        data: dict[InteractionNamesTuple, IntersectionData] = {}
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
        all_ohe: list[np.ndarray] = []
        for dims in self.intersections_data.all_dims_combs:
            intersection_data: IntersectionData = self.intersections_data.data[dims]
            possible_profiles = intersection_data.intersections_names
            # When there are too many intersections, each will have 0/1 people, and optimization is pointless + too slow
            if len(possible_profiles) > self.panel_size:
                continue
            if len(self.pool_members_df) / len(possible_profiles) < 2:
                continue
            if len(all_ohe) >= 50:  # limit to 50 different intersections to avoid too much computation
                break

            ohe = OneHotEncoder(categories=[possible_profiles], sparse_output=False)
            ohe_values = ohe.fit_transform(intersection_data.intersection_member_values.reshape(-1, 1))
            all_ohe.append(ohe_values)
        return all_ohe

    def optimize(self, max_seconds: int = 30, accepted_gap: float = 0.1) -> tuple[SolverStatus, frozenset[str], float]:
        """
        Uses solver to optimize based on the categories constraints

        For the optimization goal, for every dims intersection:
        Take the one hot encoded of who is in which intersection for these dims
        Take the binary vector of who is selected and multiply and sum to get the sizes of each intersection of categories
        Figure out the "best" value - if all intersections were of equal size
        Take the abs for each intersection from that value
        Minimize sum of abs

        Returns:
            tuple of (status, selected_ids, gap)
        """
        df = self.pool_members_df
        solver = create_solver(backend=self.solver_backend, time_limit=max_seconds, mip_gap=accepted_gap)

        # binary variable for each person - if they are selected or not
        # We store them by name so we can look them up later
        model_variables = {}
        for person_id in df.index:
            model_variables[str(person_id)] = solver.add_binary_var(name=str(person_id))
        model_variables_list = list(model_variables.values())

        # Household constraints: at most 1 person per household
        if self.check_same_address_columns:
            for housemates in self.people.households(self.check_same_address_columns).values():
                if len(housemates) > 1:
                    solver.add_constr(solver_sum(model_variables[str(member_id)] for member_id in housemates) <= 1)

        # the sum of all people in each category must be between the min and max specified
        for feature_name, value_name, fv_minmax in iterate_feature_collection(self.features):
            relevant_member_vars = [
                model_variables[str(person_id)]
                for person_id in df.index
                if df.loc[person_id, feature_name.lower()] == value_name.lower()
            ]
            rel_sum = solver_sum(relevant_member_vars)
            solver.add_constr(rel_sum >= fv_minmax.min)
            solver.add_constr(rel_sum <= fv_minmax.max)
        solver.add_constr(solver_sum(model_variables_list) == self.panel_size)  # cannot exceed panel size

        # define the optimization goal
        all_objectives = []
        for ohe in self.all_ohe:  # for every set of intersections (one hot encoded) of features
            # how many selected to each intersection
            # We need to build expressions for each intersection column
            for col_idx in range(ohe.shape[1]):
                # For this intersection column, sum up the binary vars weighted by ohe membership
                intersection_size = solver_sum(
                    model_variables_list[row_idx] * ohe[row_idx, col_idx]
                    for row_idx in range(len(model_variables_list))
                )
                # the best value is if all intersections were equal size
                best_val = self.panel_size / ohe.shape[1]

                # set support variable that is the abs diff from intersection size to best_val
                abs_diff = solver.add_continuous_var(lb=0.0, ub=float("inf"))
                # constrain this support variable to be the abs diff from the intersection size
                solver.add_constr(abs_diff >= (intersection_size - best_val))
                solver.add_constr(abs_diff >= (best_val - intersection_size))

                all_objectives.append(abs_diff)

        obj = solver_sum(all_objectives)
        solver.set_objective(obj, SolverSense.MINIMIZE)

        logger.info(f"Diversimax: {len(model_variables)} binary vars, {len(all_objectives)} abs-diff vars")

        status = solver.optimize()
        gap = solver.get_gap()

        if status in [SolverStatus.OPTIMAL, SolverStatus.FEASIBLE]:
            selected_ids = []
            for person_id in df.index:
                var = solver.get_var_by_name(str(person_id))
                if var is not None and solver.get_value(var) >= 0.99:  # ==1, for numerical stability
                    selected_ids.append(person_id)
            selected_ids_set = frozenset(selected_ids)
            return status, selected_ids_set, gap
        return status, frozenset(), gap

    def log_problem_stats(self) -> None:
        logger.info(f"Features: {len(self.pool_members_df.columns)}")
        logger.info(f"Feature combinations considered: {len(self.intersections_data.all_dims_combs)}")
        logger.info(f"  Number of OHE matrices: {len(self.all_ohe)}")

        total_intersections = 0
        for i, (dims, ohe) in enumerate(zip(self.intersections_data.all_dims_combs, self.all_ohe, strict=False)):
            if i < len(self.all_ohe):  # Only show those that passed filtering
                n_intersections = ohe.shape[1]
                total_intersections += n_intersections
                logger.info(f"  {dims}: {n_intersections} intersections")

        logger.info(f"TOTAL INTERSECTIONS: {total_intersections}")
        logger.info(f"BINARY VARIABLES: {len(self.pool_members_df)}")
