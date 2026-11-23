"""Exercise suffix search with synthetic Lambert costs and the saved itinerary."""

from contextlib import redirect_stdout
from io import StringIO
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

    def tiny_search(self, departure, fuel, costs, solve_limit=suffix.MAX_SOLVES):
        udp = SimpleNamespace(
            asteroid_materials_types=[0, 3, 0, 1, 2, 0, 0],
            asteroid_masses=[0, 1, 1, 1, 1, 0.04, 1],
        )
        budget = suffix.Budget(costs, suffix.time.perf_counter() + 60)
        budget.solve_limit = solve_limit
        engine = SimpleNamespace(udp=udp, apply_candidate=self.beam.apply_candidate,
                                 calculate_transfer_dv=budget)
        anchor = self.beam.BeamState(0, departure, fuel, (8.96, 8.96, 8.96),
                                     frozenset({0}), (self.beam.RouteStop(0, 0, 1),))

        class Shortlist:
            np = SimpleNamespace(asarray=lambda values, dtype: list(values))
            masses = udp.asteroid_masses
            groups = [[0, 2, 5, 6], [3], [4], [1]]

            def __call__(self, state, arrival, *, kinds, per_material):
                kind, = kinds
                return list(self.groups[kind][:per_material])

        output = StringIO()
        with redirect_stdout(output):
            best, reason = suffix.search(engine, Shortlist(), budget, anchor, anchor)
        self.assertEqual(budget.solve_limit, solve_limit)
        return best, reason, output.getvalue()

    def test_search_can_refuel_first_and_rejects_infeasible_arrival(self):
        costs = Mock(side_effect=lambda source, target, departure, arrival: 50 if target == 1 else 100)
        best, _, _ = self.tiny_search(1816, 50, costs)
        self.assertEqual(best.route[1].asteroid_id, 1)
        self.assertEqual(best.current_day, 1827)
        self.assertGreater(min(best.collected_materials), 9)
        self.assertAlmostEqual(best.remaining_dv, suffix.TANK / 30 - 300)
        self.assertTrue(all(call.args[1] != 5 for call in costs.call_args_list))
        self.assertTrue(all(call.args[3] - call.args[2] == 1 for call in costs.call_args_list))
        best, _, output = self.tiny_search(1816, 49, costs)
        self.assertIsNone(best)
        self.assertIn("frontier empty after layer 1 (finished)", output)
        self.assertNotIn("layer 2:", output)

    def test_material_and_time_bounds_keep_strict_score_threshold(self):
        stays = suffix.mining_requirements((9 - 1 / 30, 9.01, 7))
        self.assertEqual(stays, {0: 2, 1: 1, 2: 61, 3: 1})
        costs = Mock()
        self.assertEqual(self.tiny_search(1817, 50, costs), (None, "material or time bound", ""))
        costs.assert_not_called()

    def test_frontier_retains_distinct_epochs_and_fuel_states_at_same_asteroid(self):
        states = [
            self.beam.BeamState(1, day, fuel, (material, 8.2, 8.3), frozenset({0, 1}), ())
            for day, fuel, material in [(1701, 2000, 9.2), (1701, 2100, 9.1), (1702, 2000, 9.2)]
        ]
        dominated = self.beam.BeamState(1, 1701, 2000, (9.1, 8.2, 8.3),
                                       frozenset({0, 1}), ())
        remaining = frozenset({1, 2, 3})
        children = [(state, remaining) for state in states + [dominated]]
        self.assertCountEqual(suffix.select_frontier(children, 24),
                              [(state, remaining) for state in states])

    def test_wider_pass_finds_hidden_asteroid_and_reuses_exact_solves(self):
        def calculate(source, target, departure, arrival):
            return {1: 50, 2: 500}.get(target, 100)

        costs = Mock(side_effect=calculate)
        with patch.object(suffix, "WIDENING_PASSES", ((2, 1, 0.4), (8, 2, 1.0))):
            best, reason, output = self.tiny_search(1816, 50, costs, solve_limit=100)
        self.assertIn(6, best.visited_asteroids)
        self.assertNotIn(2, best.visited_asteroids)
        self.assertEqual(reason, "widening passes finished")
        self.assertIn("Pass 1: beam width 2, shortlist 1", output)
        self.assertIn("Pass 1, layer 4:", output)
        self.assertRegex(output, r"Pass 1, layer 4:[^\n]*completed routes 0")
        self.assertIn("Pass 2: beam width 8, shortlist 2", output)
        self.assertIn("Pass 2: terminal layer finished", output)
        self.assertRegex(output, r"Pass 2, layer 4: exact Lambert probes \d+, "
                                 r"fuel-feasible children \d+, retained states 0, completed routes [1-9]")
        self.assertEqual(costs.call_count, len({call.args for call in costs.call_args_list}))

    def test_small_pass_budget_reserves_solves_for_wider_terminal_layer(self):
        costs = Mock(side_effect=lambda source, target, departure, arrival: 50 if target == 1 else 100)
        best, _, output = self.tiny_search(1816, 50, costs, solve_limit=20)
        self.assertIn("Pass 1: frontier empty after layer 1 (Lambert solve limit reached)", output)
        self.assertIn("Pass 3, layer 4:", output)
        self.assertIsNotNone(best)
        self.assertLessEqual(costs.call_count, 20)


if __name__ == "__main__":
    unittest.main()
