"""Exercise completion ranking and limits without solving real transfers."""

from collections import Counter
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import subprocess
import unittest
from unittest.mock import Mock, patch

import completion_aware_downstream as experiment
import test_middle_route_repair as repair_fixtures


class CompletionAwareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = repair_fixtures.MiddleRepairTests()
        cls.fixture.setUpClass()

    def search(self, cost):
        fixture = self.fixture.setup_search(cost)
        masses = fixture.engine.udp.asteroid_masses
        types = fixture.engine.udp.asteroid_materials_types
        visited = fixture.reference[fixture.tail].visited_asteroids
        rich = {kind: max((target for target, material in enumerate(types)
                          if material == kind and target not in visited), key=masses.__getitem__)
                for kind in (1, 2)}
        for kind, target in rich.items():
            fixture.shortlist.groups[kind].append(target)
        indices = (fixture.start, fixture.rejoin, fixture.tail)
        return (experiment.CompletionSearch(fixture.engine, fixture.shortlist, fixture.budget,
                                            fixture.reference, indices), rich)

    def test_dynamic_window_uses_parent_and_nickel_mining_deadline(self):
        search, _ = self.search(Mock(return_value=100))
        parent = replace(search.reference[-2], current_day=1765)
        days, minimum, reserve = search.schedule(parent, 3)
        self.assertIn(1784, days)  # Parent-relative incumbent flight duration.
        self.assertIn(1800, days)  # Leaves 27 days for Nickel above 9.0.
        self.assertTrue(all(parent.current_day < day <= experiment.MISSION_END - minimum for day in days))
        self.assertEqual(reserve, 0)
        too_late = replace(parent, current_day=1826)
        self.assertFalse(search.schedule(too_late, 3)[0])

    def test_exact_completion_prefers_mined_yield_over_fuel(self):
        def cost(source, target, departure, arrival):
            return 100 if target == 6791 else 500

        search, rich = self.search(Mock(side_effect=cost))
        parent = replace(search.reference[-2], current_day=1785, remaining_dv=800,
                         collected_materials=(9.1, 9.1, 8.119630904530917))
        stats = Counter()
        children = [state for batch in search.arrival_children(parent, 3, 1800, 22, 0, stats)
                    for state in batch]
        cheap = next(child for child in children if child.current_id == 6791)
        best = max(children, key=experiment.result_rank)
        self.assertEqual(best.current_id, rich[2])
        self.assertGreater(min(best.collected_materials), 9)
        self.assertLess(best.remaining_dv, cheap.remaining_dv)
        self.assertEqual(best.current_day, 1827)

    def test_terminal_diagnostics_distinguish_invalid_fuel_and_time(self):
        def cost(source, target, departure, arrival):
            return float("nan") if target == 6791 else 1000

        search, _ = self.search(Mock(side_effect=cost))
        parent = replace(search.reference[-2], current_day=1785, remaining_dv=100)
        stats = Counter()
        self.assertFalse(any(search.children(parent, 3, stats)))
        self.assertGreater(stats["invalid Lambert"], 0)
        self.assertGreater(stats["insufficient fuel"], 0)
        self.assertFalse(any(search.children(replace(parent, current_day=1826), 3, stats)))
        self.assertEqual(stats["timing window empty"], 1)

    def test_diversity_keeps_distinct_days_and_repair_lineages(self):
        search, _ = self.search(Mock(return_value=100))
        original = search.reference[search.tail]
        first_fuel = search.start + 1
        repaired_stop = replace(original.route[first_fuel], asteroid_id=6459)
        route = (*original.route[:first_fuel], repaired_stop, *original.route[first_fuel + 1:])
        repaired = replace(original, route=route, current_day=1660,
                           visited_asteroids=(original.visited_asteroids - {5028}) | {6459})
        delayed = replace(repaired, current_day=1661, remaining_dv=5000)
        retained = search.diverse([repaired, delayed], 3, lambda state: state.remaining_dv, original)
        self.assertCountEqual(retained, [original, repaired, delayed])
        self.assertNotEqual(search.describe_ancestry(original), search.describe_ancestry(repaired))

    def test_synthetic_repair_finds_a_complete_improvement_within_budget(self):
        search, _ = self.search(Mock(return_value=100))
        search.budget.solve_limit = 6000
        with (patch.object(experiment, "EARLY_WIDTH", 8),
              patch.object(experiment, "PLATINUM_POOL", 32),
              redirect_stdout(StringIO())):
            best = search.run()
        indices = (search.start, search.rejoin, search.tail)
        experiment.check_complete(search.reference, indices, best, search.engine.udp)
        self.assertGreater(min(best.collected_materials), experiment.BASELINE)
        self.assertLessEqual(search.budget.solves, 6000)

    def test_fallback_and_budget_survive_failed_continuations(self):
        cost = Mock(return_value=experiment.TANK + 1)
        search, _ = self.search(cost)
        search.budget.solve_limit = 100
        output = StringIO()
        with redirect_stdout(output):
            best = search.run()
        self.assertIs(best, search.reference[-1])
        self.assertLessEqual(cost.call_count, 100)
        self.assertEqual(search.budget.solve_limit, 100)
        self.assertIn("complete candidates 0", output.getvalue())

    def test_worker_timeout_uses_remaining_wall_clock_budget(self):
        with (patch.object(experiment.time, "perf_counter", side_effect=[100, 102]),
              patch.object(experiment.subprocess, "run",
                           side_effect=subprocess.TimeoutExpired("worker", 598)) as run,
              patch("builtins.print")):
            self.assertEqual(experiment.main(), 1)
        self.assertEqual(run.call_args.kwargs["timeout"], 598)


if __name__ == "__main__":
    unittest.main()
