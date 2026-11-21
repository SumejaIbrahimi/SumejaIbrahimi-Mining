"""Fast tests that do not import PyKEP or run any transfers."""

from dataclasses import dataclass
import json
import unittest

from middle_route_experiment import SOURCE, replace_stop, select_candidates


@dataclass(frozen=True)
class Stop:
    asteroid_id: int
    arrival_day: float
    mining_days: int


class MiddleRouteExperimentTests(unittest.TestCase):
    def test_filter_and_rank_at_most_twenty_eligible_candidates(self):
        catalog = [(i, 100, 0, 0, 0, 0) for i in range(832)]
        for i in range(30):
            catalog[i] = (i, 101 + i, 0, 0, 0.91125, 1)
        catalog[1] = (1, 100, 0, 0, 0.91124, 1)
        catalog[2] = (2, 100, 0, 0, 1.0, 2)
        catalog[831] = (831, 100, 0, 0, 1.0, 1)
        self.assertEqual(select_candidates(catalog, {0, 831}), list(range(3, 23)))

    def test_replacement_preserves_entire_saved_schedule(self):
        saved = json.loads(SOURCE.read_text(encoding="utf-8"))
        route = tuple(Stop(*stop) for stop in zip(
            saved["visited_asteroids"], saved["arrival_times"], saved["mining_times"],
        ))
        index = saved["visited_asteroids"].index(831)
        updated = replace_stop(route, 42)
        self.assertEqual(updated[:index], route[:index])
        self.assertEqual(updated[index], Stop(42, 1495, 28))
        self.assertEqual(updated[index + 1], Stop(332, 1583, 23))
        self.assertEqual(updated[index + 1:], route[index + 1:])
        self.assertEqual(route[index], Stop(831, 1495, 28))
