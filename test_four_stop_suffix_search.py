"""Exercise suffix search with synthetic Lambert costs and the saved itinerary."""

import json
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import four_stop_suffix_search as suffix
from test_anchored_two_stop_search import load_route_helpers


class FourStopSuffixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.beam = load_route_helpers()

    def test_reconstructs_saved_anchor_and_uses_its_own_fuel_balance(self):
        saved = json.loads(suffix.SOURCE.read_text(encoding="utf-8"))
        catalog = suffix.SOURCE.parent / "data/spoc/mining/candidates.txt"
        with catalog.open(encoding="utf-8") as file:
            deposits = [line.split()[-2:] for line in file if line.strip()]
        udp = SimpleNamespace(
            asteroids=range(len(deposits)),
            asteroid_masses=[float(mass) for mass, _ in deposits],
            asteroid_materials_types=[int(float(kind)) for _, kind in deposits],
        )
        calculate = Mock(side_effect=lambda source, target, departure, arrival:
                         suffix.TANK - saved["remaining_dv"] if target == 294 else 0)
        validate = Mock(return_value=suffix.BASELINE)
        engine = SimpleNamespace(
            udp=udp, BeamState=self.beam.BeamState, RouteStop=self.beam.RouteStop,
            calculate_transfer_dv=calculate, apply_candidate=self.beam.apply_candidate,
            validate_route=validate,
        )
        anchor, final, minimum, official = suffix.reconstruct_anchor(saved, engine)
        self.assertEqual((anchor.current_id, anchor.current_day), suffix.ANCHOR)
        self.assertEqual([stop.asteroid_id for stop in anchor.route], saved["visited_asteroids"][:-4])
        self.assertEqual(anchor.visited_asteroids, frozenset(saved["visited_asteroids"][:-4]))
        self.assertEqual(suffix.mining_requirements(anchor.collected_materials),
                         {0: 29, 1: 17, 2: 27, 3: 1})
        self.assertEqual(anchor.remaining_dv, suffix.TANK)
        self.assertAlmostEqual(final.remaining_dv, saved["remaining_dv"])
        self.assertAlmostEqual(minimum, saved["remaining_dv"])
        self.assertEqual(official, suffix.BASELINE)
        validate.assert_called_once_with(final, require_full_route=True)
        saved["mining_times"][-5] += 1
        with self.assertRaisesRegex(ValueError, "exactly four stops"):
            suffix.reconstruct_anchor(saved, engine)

    def test_budget_stops_before_calling_lambert_after_deadline_or_solve_limit(self):
        lambert = Mock(return_value=42)
        budget = suffix.Budget(lambert, deadline=10)
        budget.solve_limit = 1
        with patch.object(suffix.time, "perf_counter", return_value=9):
            self.assertEqual(budget(), 42)
            with self.assertRaisesRegex(suffix.SearchLimit, "solve limit"):
                budget()
        with patch.object(suffix.time, "perf_counter", return_value=10):
            with self.assertRaisesRegex(suffix.SearchLimit, "time limit"):
                budget()
        lambert.assert_called_once()
        self.assertEqual(budget.solves, 1)

    def test_supervisor_enforces_total_runtime(self):
        with (patch.object(suffix.time, "perf_counter", side_effect=[100, 102]),
              patch.object(suffix.subprocess, "run",
                           side_effect=subprocess.TimeoutExpired("worker", 598)) as run,
              patch("builtins.print")):
            self.assertEqual(suffix.main(), 1)
        self.assertEqual(run.call_args.kwargs["timeout"], 598)

    def tiny_search(self, departure, fuel, costs):
        udp = SimpleNamespace(
            asteroid_materials_types=[0, 3, 0, 1, 2, 0],
            asteroid_masses=[0, 1, 1, 1, 1, 0.04],
        )
        engine = SimpleNamespace(udp=udp, apply_candidate=self.beam.apply_candidate,
                                 calculate_transfer_dv=costs)
        anchor = self.beam.BeamState(0, departure, fuel, (8.96, 8.96, 8.96),
                                     frozenset({0}), (self.beam.RouteStop(0, 0, 1),))

        class Shortlist:
            np = SimpleNamespace(asarray=lambda values, dtype: list(values))
            masses = udp.asteroid_masses
            groups = [[0, 2, 5], [3], [4], [1]]

            def __call__(self, state, arrival, *, kinds, per_material):
                kind, = kinds
                return list(self.groups[kind])

        return suffix.search(engine, Shortlist(), Mock(), anchor, anchor)

    def test_search_can_refuel_first_and_rejects_infeasible_arrival(self):
        costs = Mock(side_effect=lambda source, target, departure, arrival: 50 if target == 1 else 100)
        best, _ = self.tiny_search(1816, 50, costs)
        self.assertEqual(best.route[1].asteroid_id, 1)
        self.assertEqual(best.current_day, 1827)
        self.assertGreater(min(best.collected_materials), 9)
        self.assertAlmostEqual(best.remaining_dv, suffix.TANK / 30 - 300)
        self.assertTrue(all(call.args[1] != 5 for call in costs.call_args_list))
        self.assertTrue(all(call.args[3] - call.args[2] == 1 for call in costs.call_args_list))
        best, _ = self.tiny_search(1816, 49, costs)
        self.assertIsNone(best)

    def test_material_and_time_bounds_keep_strict_score_threshold(self):
        stays = suffix.mining_requirements((9 - 1 / 30, 9.01, 7))
        self.assertEqual(stays, {0: 2, 1: 1, 2: 61, 3: 1})
        costs = Mock()
        self.assertEqual(self.tiny_search(1817, 50, costs), (None, "material or time bound"))
        costs.assert_not_called()

    def test_frontier_retains_distinct_epochs_and_fuel_states_at_same_asteroid(self):
        states = [
            self.beam.BeamState(1, day, fuel, (9.1, 8.2, 8.3), frozenset({0, 1}), ())
            for day, fuel in [(1700, 2000), (1700, 3500), (1703, 2000)]
        ]
        remaining = frozenset({1, 2, 3})
        self.assertCountEqual(suffix.select_frontier((state, remaining) for state in states),
                              [(state, remaining) for state in states])


if __name__ == "__main__":
    unittest.main()
