"""Fast checks for the timing grid and deadline, without importing PyKEP."""

import json
import subprocess
import unittest
from unittest.mock import patch

from middle_route_timing_experiment import SOURCE, TRIALS, main, trial_route
from test_middle_route_experiment import Stop


class MiddleRouteTimingTests(unittest.TestCase):
    def test_four_trials_preserve_fixed_endpoints_and_unchanged_stops(self):
        saved = json.loads(SOURCE.read_text(encoding="utf-8"))
        route = tuple(Stop(*stop) for stop in zip(
            saved["visited_asteroids"], saved["arrival_times"], saved["mining_times"],
        ))
        index = saved["visited_asteroids"].index(831)
        self.assertEqual(TRIALS, ((2985, 1490), (2985, 1500), (9419, 1490), (9419, 1500)))
        for asteroid_id, day in TRIALS:
            with self.subTest(asteroid_id=asteroid_id, day=day):
                trial = trial_route(route, index, asteroid_id, day)
                self.assertEqual(trial[:index], route[:index])
                self.assertEqual(trial[index], Stop(asteroid_id, day, 28))
                self.assertEqual(trial[index + 1:], route[index + 1:])
                self.assertEqual(trial[index - 1], Stop(375, 1387, 28))
                self.assertEqual(trial[index + 1], Stop(332, 1583, 23))

    def test_timeout_stops_the_experiment(self):
        with patch("middle_route_timing_experiment.subprocess.run") as process:
            process.side_effect = subprocess.TimeoutExpired("worker", 15)
            with patch("builtins.print") as output:
                self.assertEqual(main(), 1)
        process.assert_called_once()
        self.assertEqual(process.call_args.kwargs["timeout"], 15)
        output.assert_called_once()
