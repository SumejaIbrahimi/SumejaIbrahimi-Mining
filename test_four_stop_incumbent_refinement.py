"""Focused refinement tests with deterministic Lambert costs."""

from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import json
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import four_stop_incumbent_refinement as refinement
from test_anchored_two_stop_search import load_route_helpers


class IncumbentRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.beam = load_route_helpers()
        cls.saved = json.loads(refinement.SOURCE.read_text(encoding="utf-8"))
        with (refinement.SOURCE.parent / "data/spoc/mining/candidates.txt").open(encoding="utf-8") as file:
            entries = [line.split()[-2:] for line in file if line.strip()]
        cls.udp = SimpleNamespace(
            asteroids=range(len(entries)),
            asteroid_masses=[float(mass) for mass, _ in entries],
            asteroid_materials_types=[int(float(kind)) for _, kind in entries],
        )

    def reference(self):
        saved = self.saved
        calculate = Mock(side_effect=lambda source, target, departure, arrival:
                         refinement.TANK - saved["remaining_dv"] if target == 3094 else 0)
        engine = SimpleNamespace(
            udp=self.udp, BeamState=self.beam.BeamState, RouteStop=self.beam.RouteStop,
            calculate_transfer_dv=calculate, apply_candidate=self.beam.apply_candidate,
            validate_route=Mock(return_value=refinement.BASELINE),
        )
        reference, _, _ = refinement.reconstruct_anchor(saved, engine)
        return engine, reference

    def search(self, cost):
        engine, reference = self.reference()
        visited = reference[0].visited_asteroids
        masses = self.udp.asteroid_masses
        rich = {
            kind: max((target for target, material in enumerate(self.udp.asteroid_materials_types)
                       if material == kind and target not in visited), key=masses.__getitem__)
            for kind in (1, 2)
        }

        class Shortlist:
            def __init__(self):
                self.masses = masses
                self.groups = [[6170], [2787, rich[1]], [3094, rich[2]], [5137]]

            def __call__(self, state, arrival, *, kinds, per_material):
                kind, = kinds
                return [target for target in self.groups[kind]
                        if target not in state.visited_asteroids][:per_material]

        budget = refinement.Budget(cost, refinement.time.perf_counter() + 60)
        engine.calculate_transfer_dv = budget
        return refinement.Refinement(engine, Shortlist(), budget, reference), rich

    def test_replay_preserves_exact_prefix(self):
        engine, reference = self.reference()
        anchor, final = reference[0], reference[-1]
        self.assertEqual((anchor.current_id, anchor.current_day), refinement.ANCHOR)
        self.assertEqual(tuple(state.current_id for state in reference[1:]), refinement.INCUMBENT_IDS)
        self.assertEqual(refinement.route_arrays(anchor),
                         {key: self.saved[key][:-4] for key in refinement.route_arrays(anchor)})
        engine.validate_route.assert_called_once_with(final, require_full_route=True)
        changed = replace(final.route[0], mining_days=2)
        with self.assertRaisesRegex(ValueError, "verified prefix"):
            refinement.check_prefix(anchor, replace(final, route=(changed, *final.route[1:])), self.udp)

    def test_gold_probes_neighboring_days_and_whole_day_stays_through_29(self):
        cost = Mock(return_value=100)
        search, _ = self.search(cost)
        children = [state for batch in search.expand(search.anchor, 0) for state in batch]
        center = [state for state in children if state.route[-1].arrival_day == 1694]
        self.assertTrue(set(range(22, 30)) <= {state.route[-1].mining_days for state in center})
        self.assertTrue({1693, 1694, 1695} <= {state.route[-1].arrival_day for state in children})
        longer = next(state for state in center if state.route[-1].mining_days == 29)
        self.assertEqual(longer.current_day, 1723)
        self.assertGreater(longer.collected_materials[0], 9)
        self.assertTrue(any(option["arrival_day"] == 1740 for option, _, _ in search.edges(longer, 1)))
        self.assertIn((6170, 5137, 1723, 1740), [call.args for call in cost.call_args_list])

    def test_refuel_requires_arrival_fuel_and_caps_tank(self):
        cost = Mock(return_value=2001)
        search, _ = self.search(cost)
        gold = replace(search.reference[1], remaining_dv=2000)
        self.assertFalse(any(search.expand(gold, 1)))
        self.assertEqual(search.lookahead(gold, 1), -1)
        cost.return_value = 200
        search.transfer.cache_clear()
        children = [state for batch in search.expand(gold, 1) for state in batch]
        self.assertTrue(any(state.remaining_dv > gold.remaining_dv for state in children))
        self.assertTrue(all(0 <= state.remaining_dv <= refinement.TANK for state in children))

    def test_protected_incumbent_reaches_terminal_layer_even_with_no_alternatives(self):
        search, _ = self.search(Mock(return_value=refinement.TANK + 1))
        output = StringIO()
        with redirect_stdout(output):
            best = search.run()
        self.assertIs(best, search.reference[-1])
        self.assertIn("children 0, retained 1", output.getvalue())
        self.assertIn("Layer 4:", output.getvalue())

    def test_bounded_search_finds_complete_synthetic_improvement(self):
        search, rich = self.search(Mock(return_value=100))
        search.budget.solve_limit = 5000
        with (patch.object(refinement, "BEAM_WIDTH", 12), redirect_stdout(StringIO())):
            best = search.run()
        refinement.check_prefix(search.anchor, best, self.udp)
        self.assertGreater(min(best.collected_materials), refinement.BASELINE)
        self.assertTrue(rich[1] in best.visited_asteroids or rich[2] in best.visited_asteroids)
        self.assertLessEqual(search.budget.solves, 5000)
        self.assertEqual(search.budget.solve_limit, 5000)

    def test_cache_and_solve_budget(self):
        cost = Mock(return_value=100)
        search, _ = self.search(cost)
        search.budget.solve_limit = 1
        self.assertEqual(search.transfer(7230, 6170, 1670, 1694), 100)
        self.assertEqual(search.transfer(7230, 6170, 1670, 1694), 100)
        with self.assertRaisesRegex(refinement.SearchLimit, "solve limit"):
            search.transfer(7230, 6170, 1670, 1695)
        cost.assert_called_once()

    def test_supervisor_uses_remaining_wall_clock_budget(self):
        with (patch.object(refinement.time, "perf_counter", side_effect=[100, 102]),
              patch.object(refinement.subprocess, "run",
                           side_effect=subprocess.TimeoutExpired("worker", 598)) as run,
              patch("builtins.print")):
            self.assertEqual(refinement.main(), 1)
        self.assertEqual(run.call_args.kwargs["timeout"], 598)


if __name__ == "__main__":
    unittest.main()
