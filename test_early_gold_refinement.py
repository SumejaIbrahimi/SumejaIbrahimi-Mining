"""Fast checks for the refinement configuration and candidate shortlist."""

import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from early_gold_refinement import gold_candidates, run


class EarlyGoldRefinementTests(unittest.TestCase):
    def test_refinement_uses_unchanged_anchors_and_bounded_search(self):
        with patch("early_gold_refinement.search") as search:
            run()
        search.assert_called_once()
        source, output = search.call_args.args
        options = search.call_args.kwargs
        saved = json.loads(source.read_text(encoding="utf-8"))
        route = list(zip(saved["visited_asteroids"], saved["arrival_times"], saved["mining_times"]))

        self.assertEqual(source.name, "middle_gold_best_result.json")
        self.assertEqual(output.name, "early_gold_refinement_best_result.json")
        self.assertEqual(saved["official_score"], options["expected_score"])
        self.assertEqual(route[1:5], [
            (410, 101, 29), (4745, 210, 27), (772, 294, 8), (1850, 382, 24),
        ])
        self.assertEqual(options["inner_stops"], tuple(route[2:4]))
        self.assertEqual(options["schedules"], ((207, 30), (208, 29)))
        self.assertTrue(all(day + stay == 237 for day, stay in options["schedules"]))
        self.assertEqual(options["candidate_limit"], 12)
        self.assertEqual(options["solve_limit"], 22 + 2 * 12 * 2 + 44)
        self.assertIs(options["select_candidates"], gold_candidates)

    def test_shortlist_filters_and_reserves_slots_for_high_yield(self):
        def orbit(offset):
            return SimpleNamespace(eph=lambda day: ((0, 0, 0), (offset * day, 0, 0)))

        asteroids = [orbit(i) for i in range(4746)]
        asteroids[4745] = orbit(0)
        materials = [0] * len(asteroids)
        masses = [0.0] * len(asteroids)
        masses[:20] = [0.92] * 20
        masses[1] = 0.901
        materials[2] = 1
        masses[17:20] = [0.98, 0.99, 1.0]
        udp = SimpleNamespace(
            asteroids=asteroids, asteroid_materials_types=materials, asteroid_masses=masses,
        )

        selected = gold_candidates(udp, {0, 4745}, 207, 30)
        self.assertEqual(selected, list(range(3, 12)) + [19, 18, 17])
        self.assertEqual(gold_candidates(udp, {0, 4745}, 208, 29),
                         list(range(3, 12)) + [17, 18, 19])
