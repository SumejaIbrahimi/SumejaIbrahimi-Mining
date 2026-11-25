"""Exercise the joint suffix with the real catalog and mocked Lambert costs."""

from collections import Counter
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import unittest
import re
from unittest.mock import Mock, patch

import joint_fuel_platinum_nickel as joint
import test_middle_route_repair as repair_fixtures


class JointSuffixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repair_fixtures.MiddleRepairTests.setUpClass()

    def search(self, cost):
        fixture = repair_fixtures.MiddleRepairTests().setup_search(cost)
        for kind in (1, 2):
            available = (target for target, material in
                         enumerate(fixture.engine.udp.asteroid_materials_types)
                         if material == kind and target not in
                         fixture.reference[fixture.tail].visited_asteroids)
            richest = max(available, key=fixture.shortlist.masses.__getitem__)
            fixture.shortlist.groups[kind].append(richest)
        return joint.JointSearch(fixture.engine, fixture.shortlist, fixture.budget,
                                 fixture.reference, (fixture.start, fixture.rejoin, fixture.tail))

    def test_actual_deposit_and_strict_goal_deadline(self):
        self.assertEqual(joint.required_stay(8, 1.0, joint.BASELINE), 26)
        self.assertIsNone(joint.required_stay(8, 1.0, joint.GOAL))
        self.assertEqual(joint.required_stay(8.1, 1.0, joint.GOAL), 28)
        cost = Mock(return_value=100)
        search = self.search(cost)
        parent = replace(search.reference[-2], current_day=1799,
                         collected_materials=(9.01, 9.02, 8.11963090453092))
        outcome = self.probe(search, parent)
        self.assertGreater(outcome["goal"], 0)
        self.assertTrue(all(call.args[3] <= 1805 for call in cost.call_args_list))
        goal_dates = [call.args[3] for call in cost.call_args_list if call.args[3] == 1800]
        self.assertTrue(goal_dates)
        self.assertIsNone(self.probe(search, replace(parent, current_day=1805))["best"])

    def test_gold_keeps_29_day_stay(self):
        search = self.search(Mock(return_value=100))
        seed = search.reference[search.tail]
        gold = [state for batch in search.children(seed, 0, Counter()) for state in batch]
        retained = search.select_departures(gold, search.reference[search.tail + 1])
        self.assertTrue(any(state.route[-1].mining_days == 29
                            and state.current_id == 6170
                            and state.collected_materials[0] > 9 for state in retained))

    def test_rank_rewards_complete_yield_before_fuel(self):
        search = self.search(Mock(return_value=100))
        incumbent = search.reference[-1]
        higher = replace(incumbent, collected_materials=(9.01, 9.02, 9.03), remaining_dv=0)
        same = replace(higher, remaining_dv=1)
        self.assertGreater(joint.completion_rank(higher), joint.completion_rank(incumbent))
        self.assertGreater(joint.completion_rank(same), joint.completion_rank(higher))

    def test_refill_accounts_for_inbound_burn_and_clipping(self):
        self.assertEqual(joint.refill_details(9000, 0.6, 15, 1000),
                         {"refill": 2000, "clipped": 3000, "fuel_burn": 1000})
        self.assertEqual(joint.refill_details(4000, 0.3, 20, 1000)["refill"], 3000)

    @staticmethod
    def probe(search, parent):
        search.nickel_solves = 0
        outcome = dict(margin=None, margin_flight=None, floor=0, goal=0,
                       best=None, best_flight=None)
        search.probe_nickel(parent, Counter(), outcome)
        return outcome

    def test_probe_bands_and_refines_integer_dates(self):
        cost = Mock(side_effect=lambda source, target, departure, arrival:
                    100 + abs(arrival - 1788))
        search = self.search(cost)
        parent = replace(search.reference[-2], current_day=1770, remaining_dv=200,
                         collected_materials=(9.01, 9.02, 8.11963090453092))
        outcome = self.probe(search, parent)
        dates = [call.args[-1] for call in cost.call_args_list]
        coarse = dates[:joint.PROBES_PER_PARENT // 2]
        self.assertTrue(all(any(day in band for day in coarse)
                            for band in joint.flight_bands(1771, 1800)))
        self.assertTrue(any(abs(day - earlier) == 1 and day not in coarse
                            for day in dates[len(coarse):] for earlier in coarse))
        self.assertLessEqual(len(dates), joint.PROBES_PER_PARENT)
        self.assertGreaterEqual(outcome["margin"], 0)

    def test_nickel_fresh_cap_allows_cached_transfer_and_restores_limit(self):
        search = self.search(Mock(return_value=100))
        search.nickel_solves = 0
        parent = search.reference[-2]
        end = search.budget.solve_limit
        with patch.object(joint, "NICKEL_SOLVES", 1):
            self.assertEqual(search.nickel_transfer(parent, 6791, 1805), 100)
            self.assertEqual(search.nickel_transfer(parent, 6791, 1805), 100)
            with self.assertRaises(joint.SearchLimit):
                search.nickel_transfer(parent, 6791, 1804)
        self.assertEqual(search.budget.solve_limit, end)
        self.assertEqual(search.nickel_solves, 1)

    def test_small_joint_run_completes_before_platinum_pruning(self):
        search = self.search(Mock(return_value=100))
        search.budget.solve_limit = 3000
        with (patch.object(joint, "JOINT_SOLVES", 2000),
              patch.object(joint, "NICKEL_SOLVES", 1100),
              patch.object(joint, "EARLY_WIDTH", 8),
              redirect_stdout(StringIO()) as output):
            best = search.run()
        joint.check_complete(search.reference, (search.start, search.rejoin, search.tail),
                             best, search.engine.udp)
        self.assertGreater(joint.completion_rank(best)[0], joint.GOAL)
        self.assertLessEqual(search.budget.solves, 3000)
        self.assertLessEqual(search.nickel_solves, 1100)
        solves = re.search(r"\| fresh solves (\d+)", output.getvalue())
        self.assertIsNotNone(solves)
        self.assertLessEqual(int(solves[1]), 2000)
        self.assertIn('"platinum_burn"', output.getvalue())
        self.assertIn('"control": true', output.getvalue())

    def test_fallback_with_unaffordable_transfers(self):
        search = self.search(Mock(return_value=joint.TANK + 1))
        search.budget.solve_limit = 100
        with redirect_stdout(StringIO()):
            self.assertIs(search.run(), search.reference[-1])
        self.assertEqual(search.budget.solve_limit, 100)
        self.assertLessEqual(search.budget.solves, 100)

    def test_validation_requires_both_official_runs_to_improve(self):
        search = self.search(Mock(return_value=100))
        incumbent = search.reference[-1]
        best = replace(incumbent, collected_materials=(9.01, 9.02, 9.03))
        indices = search.start, search.rejoin, search.tail
        with (patch.object(joint, "replay", return_value=([best], 100)),
              patch.object(joint, "check_result"),
              redirect_stdout(StringIO())):
            search.engine.validate_route.side_effect = [9.01, joint.BASELINE, 9.01, 9.01]
            self.assertIsNone(joint.validate_improvement(
                best, search.reference, indices, search.engine, search.budget))
            self.assertEqual(joint.validate_improvement(
                best, search.reference, indices, search.engine, search.budget), (best, 9.01, 100))


if __name__ == "__main__":
    unittest.main()
