"""Lightweight tests for the six-stop experiment with synthetic transfer costs."""

from dataclasses import replace
from contextlib import redirect_stdout
from io import StringIO
import json
import math
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import six_stop_suffix_search as suffix
from test_anchored_two_stop_search import load_route_helpers


class SixStopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.beam = load_route_helpers()

    def test_reconstructs_saved_prefix_and_checks_six_stop_counts(self):
        saved = json.loads(suffix.SOURCE.read_text(encoding="utf-8"))
        catalog = suffix.SOURCE.parent / "data/spoc/mining/candidates.txt"
        with catalog.open(encoding="utf-8") as file:
            entries = [line.split()[-2:] for line in file if line.strip()]
        udp = SimpleNamespace(
            asteroids=range(len(entries)),
            asteroid_masses=[float(mass) for mass, _ in entries],
            asteroid_materials_types=[int(float(kind)) for _, kind in entries],
        )
        calculate = Mock(side_effect=lambda source, target, departure, arrival:
                         suffix.TANK - saved["remaining_dv"] if target == 294 else 0)
        engine = SimpleNamespace(
            udp=udp, BeamState=self.beam.BeamState, RouteStop=self.beam.RouteStop,
            calculate_transfer_dv=calculate, apply_candidate=self.beam.apply_candidate,
            validate_route=Mock(return_value=suffix.BASELINE),
        )
        reference, _, score = suffix.reconstruct_anchor(saved, engine)
        anchor, final = reference[0], reference[-1]
        self.assertEqual((anchor.current_id, anchor.current_day), suffix.ANCHOR)
        self.assertEqual(score, suffix.BASELINE)
        for key, values in suffix.route_arrays(anchor).items():
            self.assertEqual(values, saved[key][:-6])
        suffix.check_suffix(anchor, final, udp)
        with self.assertRaisesRegex(ValueError, "six|two platinum"):
            suffix.check_suffix(anchor, replace(final, route=final.route[:-1]), udp)
        with self.assertRaisesRegex(ValueError, "verified prefix"):
            changed = replace(final.route[0], mining_days=2)
            suffix.check_suffix(anchor, replace(final, route=(changed, *final.route[1:])), udp)

    def toy_search(self, cost):
        udp = SimpleNamespace(
            asteroid_masses=[0, 0.60, 0.60, 1, 1, 0.8, 0.8],
            asteroid_materials_types=[0, 1, 1, 0, 2, 3, 3],
        )
        budget = suffix.Budget(cost, math.inf)
        engine = SimpleNamespace(udp=udp, apply_candidate=self.beam.apply_candidate,
                                 calculate_transfer_dv=budget)
        state = self.beam.BeamState(0, 1596, 3000, (8.5, 7.7, 8.5), frozenset({0}),
                                    (self.beam.RouteStop(0, 0, 1),))

        class Shortlist:
            masses = udp.asteroid_masses

            def __init__(self):
                self.groups = [[3], [1, 2], [4], [5, 6]]

            def __call__(self, state, arrival, *, kinds, per_material):
                kind, = kinds
                return [target for target in self.groups[kind]
                        if target not in state.visited_asteroids][:per_material]

        return suffix.SuffixSearch(engine, Shortlist(), budget, [state]), state

    def test_platinum_deposits_are_evaluated_jointly_and_cannot_be_reused(self):
        search, state = self.toy_search(Mock(return_value=100))
        self.assertIsNone(search.bounds(state, (0, 1, 0, 0)))
        self.assertIsNotNone(search.bounds(state, suffix.PATTERN))
        options = list(search.edges(state, (0, 2, 0, 0), (14,), 2))
        self.assertEqual({option["id"] for _, _, option, _, _ in options}, {1, 2})
        self.assertTrue(all(minimum == 14 for _, _, _, minimum, _ in options))
        self.assertIsNone(search.bounds(replace(state, visited_asteroids=frozenset({0, 1})),
                                        suffix.PATTERN))

    def test_fuel_must_cover_arrival_and_refill_respects_tank_cap(self):
        calculate = Mock(return_value=3001)
        search, state = self.toy_search(calculate)
        already_mined = replace(state, collected_materials=(8.8, 8.8, 8.8))
        self.assertFalse(any(search.expand(already_mined, (0, 0, 0, 1))))
        self.assertEqual(search.reachability(already_mined, (0, 0, 0, 1)), -1.0)
        calculate.return_value = 500
        search.transfer.cache_clear()
        children = [child for batch in search.expand(already_mined, (0, 0, 0, 1))
                    for child, _ in batch]
        self.assertTrue(children)
        self.assertTrue(all(child.remaining_dv == suffix.TANK for child in children))
        self.assertEqual(search.reachability(already_mined, (0, 0, 0, 1)), 1.0)

    def test_cached_transfers_do_not_consume_reserved_solves(self):
        calculate = Mock(return_value=100)
        search, _ = self.toy_search(calculate)
        search.budget.solve_limit = 1
        self.assertEqual(search.transfer(0, 1, 1596, 1610), 100)
        self.assertEqual(search.transfer(0, 1, 1596, 1610), 100)
        with self.assertRaisesRegex(suffix.SearchLimit, "solve limit"):
            search.transfer(0, 1, 1596, 1611)
        calculate.assert_called_once()

    def test_bounded_beam_completes_a_synthetic_improvement(self):
        search, anchor = self.toy_search(Mock(return_value=100))
        incumbent = anchor
        for target, stay in [(5, 8), (1, 12), (3, 24), (6, 8), (4, 24), (2, 18)]:
            option = suffix.candidate(search.engine.udp, target, incumbent.current_day + 8, 100)
            incumbent = search.engine.apply_candidate(incumbent, option, stay)
            search.reference.append(incumbent)
        search = suffix.SuffixSearch(search.engine, search.shortlist, search.budget, search.reference)
        search.budget.solve_limit = 4000
        search.budget.deadline = suffix.time.perf_counter() + 30
        output = StringIO()
        with (patch.object(suffix, "BEAM_WIDTH", 8),
              patch.object(suffix, "FLIGHTS", (8, 14)),
              redirect_stdout(output)):
            best = search.run()
        suffix.check_suffix(anchor, best, search.engine.udp)
        self.assertGreater(min(best.collected_materials), suffix.BASELINE)
        self.assertIn("Layer 6:", output.getvalue())
        self.assertLessEqual(search.budget.solves, 4000)
        self.assertEqual(search.budget.solve_limit, 4000)

    def test_supervisor_enforces_wall_clock_limit(self):
        with (patch.object(suffix.time, "perf_counter", side_effect=[100, 102]),
              patch.object(suffix.subprocess, "run",
                           side_effect=subprocess.TimeoutExpired("worker", 598)) as run,
              patch("builtins.print")):
            self.assertEqual(suffix.main(), 1)
        self.assertEqual(run.call_args.kwargs["timeout"], 598)


if __name__ == "__main__":
    unittest.main()
