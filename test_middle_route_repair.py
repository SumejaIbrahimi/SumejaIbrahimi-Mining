"""Lightweight checks of the middle repair using mocked exact transfer costs."""

from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import json
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import middle_route_repair as repair
from test_anchored_two_stop_search import load_route_helpers


class MiddleRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.beam = load_route_helpers()
        cls.saved = json.loads(repair.SOURCE.read_text(encoding="utf-8"))
        with (repair.SOURCE.parent / "data/spoc/mining/candidates.txt").open(encoding="utf-8") as file:
            deposits = [line.split()[-2:] for line in file if line.strip()]
        cls.udp = SimpleNamespace(
            asteroids=range(len(deposits)),
            asteroid_masses=[float(mass) for mass, _ in deposits],
            asteroid_materials_types=[int(float(kind)) for _, kind in deposits],
        )

    def setup_search(self, calculate):
        baseline_transfer = Mock(side_effect=lambda source, target, departure, arrival:
                                 repair.TANK - self.saved["remaining_dv"] if target == 6791 else 0)
        engine = SimpleNamespace(
            udp=self.udp, BeamState=self.beam.BeamState, RouteStop=self.beam.RouteStop,
            calculate_transfer_dv=baseline_transfer, apply_candidate=self.beam.apply_candidate,
            validate_route=Mock(return_value=repair.BASELINE),
        )
        reference, indices, _, _ = repair.load_reference(self.saved, engine)
        budget = repair.Budget(calculate, repair.time.perf_counter() + 60)
        engine.calculate_transfer_dv = budget
        masses = self.udp.asteroid_masses

        class Shortlist:
            def __init__(self):
                self.masses = masses
                self.groups = [[6170], [1033], [6791], [5028, 7002, 6459]]

            def __call__(self, state, arrival, *, kinds, per_material):
                kind, = kinds
                return [target for target in self.groups[kind]
                        if target not in state.visited_asteroids][:per_material]

        return repair.RepairSearch(engine, Shortlist(), budget, reference, indices)

    def test_verified_prefix_and_completed_route_structure(self):
        search = self.setup_search(Mock(return_value=100))
        anchor = search.reference[search.start]
        self.assertEqual((anchor.current_id, anchor.current_day), repair.ANCHOR)
        for key, values in repair.route_arrays(anchor).items():
            self.assertEqual(values, self.saved[key][:search.start + 1])
        final = search.reference[-1]
        indices = (search.start, search.rejoin, search.tail)
        repair.check_complete(search.reference, indices, final, self.udp)
        one_refuel = replace(final, route=final.route[:search.start + 1] + final.route[search.start + 2:])
        repair.check_complete(search.reference, indices, one_refuel, self.udp)
        changed = replace(final.route[0], mining_days=2)
        with self.assertRaisesRegex(ValueError, "verified day-1328 prefix"):
            repair.check_complete(search.reference, indices,
                                  replace(final, route=(changed, *final.route[1:])), self.udp)

    def test_repair_keeps_one_and_two_refuel_patterns_and_distinct_rejoin_days(self):
        search = self.setup_search(Mock(return_value=100))
        with patch.object(repair, "FUEL_WIDTH", 4), redirect_stdout(StringIO()):
            states = search.repair()
        root = search.reference[search.start]
        counts = set()
        for state in states:
            self.assertEqual(state.route[:len(root.route)], root.route)
            self.assertEqual(state.current_id, repair.REJOIN)
            self.assertEqual(state.route[-1].mining_days, 24)
            self.assertTrue(repair.REJOIN_WINDOW[0] <= state.route[-1].arrival_day <= repair.REJOIN_WINDOW[1])
            counts.add(len(state.route) - len(root.route) - 1)
        self.assertEqual(counts, {1, 2})
        self.assertGreater(len({state.current_day for state in states}), 1)
        self.assertLessEqual(len(states), repair.REJOIN_WIDTH)

    def test_refill_requires_fuel_on_arrival_and_caps_the_tank(self):
        calculate = Mock(return_value=101)
        search = self.setup_search(calculate)
        root = replace(search.reference[search.start], remaining_dv=100)
        self.assertFalse(any(search.fuel_children(root, 0)))
        calculate.return_value = 25
        search.transfer.cache_clear()
        root = replace(root, remaining_dv=9900)
        children = [child for batch in search.fuel_children(root, 0) for child in batch]
        self.assertTrue(children)
        self.assertTrue(all(child.remaining_dv == repair.TANK for child in children))

    def test_downstream_retiming_checks_fuel_at_each_exact_epoch(self):
        calculate = Mock(side_effect=lambda source, target, departure, arrival:
                         900 if arrival < 1428 else 100)
        search = self.setup_search(calculate)
        previous, destination = search.reference[search.rejoin:search.rejoin + 2]
        earlier = replace(previous, current_day=previous.current_day - 12, remaining_dv=150)
        children = [child for batch in search.downstream_children(earlier, previous, destination)
                    for child in batch]
        self.assertIn((5694, 4525, 1411, 1425), [call.args for call in calculate.call_args_list])
        self.assertTrue(children)
        self.assertTrue(all(child.route[-1].arrival_day >= 1428 for child in children))
        self.assertTrue(all(child.remaining_dv == 50 for child in children))

    def test_fallback_survives_repair_and_downstream_with_no_alternatives(self):
        search = self.setup_search(Mock(return_value=repair.TANK + 1))
        search.budget.solve_limit = 100
        output = StringIO()
        with redirect_stdout(output):
            best = search.run()
        self.assertIs(best, search.reference[-1])
        self.assertIn("Downstream 7230: retained 1", output.getvalue())
        self.assertIn("Final stop 4:", output.getvalue())
        self.assertLessEqual(search.budget.solves, 100)
        self.assertEqual(search.budget.solve_limit, 100)

    def test_repaired_state_can_reach_a_complete_synthetic_improvement(self):
        search = self.setup_search(Mock(return_value=100))
        search.budget.solve_limit = 5000
        indices = (search.start, search.rejoin, search.tail)
        with (patch.object(repair, "FUEL_WIDTH", 4),
              patch.object(repair, "REJOIN_WIDTH", 8),
              patch.object(repair, "CONTINUATION_WIDTH", 8),
              redirect_stdout(StringIO())):
            best = search.run()
        repair.check_complete(search.reference, indices, best, self.udp)
        self.assertGreater(min(best.collected_materials), repair.BASELINE)
        self.assertLessEqual(best.current_day, repair.MISSION_END)
        self.assertLessEqual(search.budget.solves, 5000)

    def test_stage_budget_preserves_later_share_and_cached_probes_are_free(self):
        calculate = Mock(return_value=100)
        search = self.setup_search(calculate)
        now = repair.time.perf_counter()
        with patch.object(repair.time, "perf_counter", return_value=now):
            repair.set_stage_budget(search.budget, 4, now + 30, 4)
            self.assertEqual(search.budget.deadline, now + 7.5)
            self.assertEqual(search.budget.solve_limit, 1)
            self.assertEqual(search.transfer(5231, 5028, 1328, 1342), 100)
            self.assertEqual(search.transfer(5231, 5028, 1328, 1342), 100)
            with self.assertRaisesRegex(repair.SearchLimit, "solve limit"):
                search.transfer(5231, 5028, 1328, 1343)
        calculate.assert_called_once()

    def test_wall_clock_supervisor(self):
        with (patch.object(repair.time, "perf_counter", side_effect=[100, 102]),
              patch.object(repair.subprocess, "run",
                           side_effect=subprocess.TimeoutExpired("worker", 598)) as run,
              patch("builtins.print")):
            self.assertEqual(repair.main(), 1)
        self.assertEqual(run.call_args.kwargs["timeout"], 598)


if __name__ == "__main__":
    unittest.main()
