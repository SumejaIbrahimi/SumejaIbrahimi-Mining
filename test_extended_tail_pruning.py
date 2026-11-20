"""Fast, dependency-free tests for the extended tail's optimistic bound."""

from types import SimpleNamespace
import unittest

from extended_tail_search import can_still_improve


class OptimisticPruningTests(unittest.TestCase):
    def setUp(self):
        self.state = SimpleNamespace(
            collected_materials=(4.2, 3.0, 4.1),
            visited_asteroids=frozenset(),
            current_day=1700,
        )
        self.deposits = ((), ((101, 0.6), (102, 0.5)), ())

    def can_improve(self, stops=2, deadline=1827):
        return can_still_improve(
            self.state, 4.0, stops, self.deposits, deadline, 20, 30
        )

    def test_two_platinum_deposits_can_beat_target(self):
        self.assertTrue(self.can_improve())
        self.assertFalse(self.can_improve(stops=1))

    def test_visited_deposit_cannot_contribute_again(self):
        self.state.visited_asteroids = frozenset({101})
        self.assertFalse(self.can_improve())

    def test_bound_includes_mining_and_both_flights(self):
        self.assertFalse(self.can_improve(deadline=1769))
        self.assertTrue(self.can_improve(deadline=1770))

    def test_different_ores_share_the_stop_budget(self):
        self.state.collected_materials = (3.9, 3.0, 4.1)
        self.deposits = (((103, 0.2),), self.deposits[1], ())
        self.assertFalse(self.can_improve(stops=2))
        self.assertTrue(self.can_improve(stops=3))
