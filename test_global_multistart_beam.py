"""Focused global-search tests with no real Lambert or Kaggle run."""

from contextlib import redirect_stdout
from io import StringIO
import json
import math
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import global_multistart_beam as experiment
from test_anchored_two_stop_search import load_route_helpers


class GlobalBeamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = load_route_helpers()

    def test_distinct_esa_starts_without_credit(self):
        catalog = [(i, 400 - i) for i in range(400)]
        roots = experiment.initial_states(catalog, self.engine.BeamState, self.engine.RouteStop)
        self.assertEqual(len({root.current_id for root in roots}), 12)
        self.assertIn(300, {root.current_id for root in roots})
        for root in roots:
            self.assertIn(root.current_id, range(len(catalog)))
            self.assertEqual(root.current_day, 1)
            self.assertEqual(root.remaining_dv, 10_000)
            self.assertEqual(root.collected_materials, (0, 0, 0))
            self.assertEqual(root.visited_asteroids, {root.current_id})
            self.assertEqual(root.route, (self.engine.RouteStop(root.current_id, 0, 1),))

    def test_beam_retains_origins_and_only_dominates_within_context(self):
        roots = experiment.initial_states([(i, i) for i in range(20)],
                                          self.engine.BeamState, self.engine.RouteStop)
        first, second = roots[:2]
        from_first = self.engine.BeamState(25, 80, 9000, (1, 0, 0), first.visited_asteroids | {25},
                                           first.route + (self.engine.RouteStop(25, 50, 30),))
        weaker = self.engine.BeamState(25, 80, 8000, (1, 0, 0), from_first.visited_asteroids,
                                       from_first.route)
        from_second = self.engine.BeamState(25, 80, 7000, (1, 0, 0), second.visited_asteroids | {25},
                                            second.route + (self.engine.RouteStop(25, 50, 30),))
        stats = experiment.Counter()
        with patch.object(experiment, "BEAM_WIDTH", 2):
            selected = experiment.select_beam([weaker, from_second, from_first], {}, stats)
        self.assertEqual(set(selected), {from_first, from_second})
        self.assertEqual(stats["pareto"], 1)
        self.assertEqual(experiment.projected_minimum(SimpleNamespace(
            current_day=1827, collected_materials=(8, 2, 3),
        )), 2)

    def test_budget_reserves_validation_and_caches_failed_solves(self):
        calculate = Mock(side_effect=[100, RuntimeError("no Lambert")])
        budget = experiment.TransferBudget(calculate, math.inf)
        with patch.object(experiment, "MAX_SOLVES", 2 * experiment.MAX_LEVELS + 2):
            self.assertEqual(budget.transfer(0, 1, 1, 9), 100)
            self.assertEqual(budget.transfer(0, 1, 1, 9), 100)
            self.assertTrue(math.isnan(budget.transfer(0, 1, 2, 9)))
            self.assertTrue(math.isnan(budget.transfer(0, 1, 2, 9)))
            with self.assertRaises(experiment.SearchLimit):
                budget.transfer(0, 1, 2, 15)
        self.assertEqual(calculate.call_count, 2)
        self.assertEqual(calculate.call_args_list[0].args, (0, 1, 1, 9))

    def test_builds_new_routes_charges_first_leg_and_checks_after_refueling(self):
        engine = SimpleNamespace(
            apply_candidate=self.engine.apply_candidate,
            udp=SimpleNamespace(asteroid_materials_types=[3, 3, 0, 1, 2],
                                asteroid_masses=[1.0, 0.8, 0.9, 0.9, 0.9]),
        )
        root = self.engine.BeamState(0, 1, 10_000, (0, 0, 0), frozenset({0}),
                                     (self.engine.RouteStop(0, 0, 1),))

        def calculate(source, target, departure, arrival):
            return {(0, 1): 7000, (0, 2): 10_001, (1, 2): 8500,
                    (2, 3): 1000, (3, 4): 100}[source, target]

        def shortlist(state, arrival, **options):
            return {0: [1, 2], 1: [2], 2: [3], 3: [4]}[state.current_id]

        budget = experiment.TransferBudget(Mock(side_effect=calculate), math.inf)
        search = experiment.GlobalSearch(engine, shortlist, budget)
        with patch.object(experiment, "MAX_LEVELS", 4):
            best, = search.run([root])
        self.assertEqual([stop.asteroid_id for stop in best.route], [0, 1, 2, 3, 4])
        self.assertEqual(best.collected_materials, (0.9, 0.9, 0.9))
        self.assertGreaterEqual(best.remaining_dv, 0)
        self.assertGreater(search.stats["insufficient fuel"], 0)
        self.assertGreater(search.stats["fuel continuation found"], 0)
        self.assertEqual(experiment.stays_for(1, 0, 1826, 10_000), [1])
        self.assertEqual(experiment.stays_for(0.8, 3, 50, 3000), [10, 15, 21])
        self.assertEqual(len(budget.calculate.call_args_list), len(set(
            call.args for call in budget.calculate.call_args_list
        )))

    def test_official_result_written_exclusively_and_only_above_record(self):
        root = experiment.initial_states([(i, i) for i in range(12)],
                                         self.engine.BeamState, self.engine.RouteStop)[0]
        validator = Mock(side_effect=[ValueError("incomplete"), experiment.INCUMBENT_SCORE, 4.5, 4.6])
        engine = SimpleNamespace(validate_route=validator)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / experiment.OUTPUT.name
            with patch.object(experiment, "OUTPUT", output), redirect_stdout(StringIO()):
                for _ in range(2):
                    experiment.validate_and_save(engine, root, math.inf)
                    self.assertFalse(output.exists())
                experiment.validate_and_save(engine, root, math.inf)
                experiment.validate_and_save(engine, root, math.inf)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["official_score"], 4.5)
        self.assertTrue(all(call.kwargs == {"require_full_route": True}
                            for call in validator.call_args_list))
