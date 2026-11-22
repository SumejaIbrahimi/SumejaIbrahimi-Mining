"""Test the anchored experiment without calling PyKEP or the ESA evaluator."""

from contextlib import redirect_stdout
import importlib
from io import StringIO
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import anchored_two_stop_search as experiment


def load_route_helpers():
    # Load the actual route transition while replacing its orbital dependencies.
    modules = {
        "numpy": SimpleNamespace(),
        "pykep": SimpleNamespace(),
        "find_reachable": SimpleNamespace(search_multiple_transfers=None, esa=None, udp=None),
    }
    with patch.dict("sys.modules", modules):
        return importlib.import_module("beam_search")


class AnchoredTwoStopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.beam = load_route_helpers()

    def test_real_transition_requires_arrival_fuel_and_caps_refill(self):
        stop = self.beam.RouteStop(772, 294, 8)
        state = self.beam.BeamState(772, 302, 4500, (1, 2, 3), frozenset({772}), (stop,))
        candidate = {"id": 1850, "arrival_day": 390, "dv": 4501, "material": 3, "mass": 0.767}
        self.assertIsNone(self.beam.apply_candidate(state, candidate, 16))
        candidate["dv"] = 500
        partial = self.beam.apply_candidate(state, candidate, 16)
        self.assertEqual(partial.current_day, 406)
        self.assertAlmostEqual(partial.remaining_dv, 4000 + 10000 * 16 / 30)
        self.assertEqual(partial.collected_materials, state.collected_materials)
        candidate["arrival_day"] = 382
        full = self.beam.apply_candidate(state, candidate, 24)
        self.assertEqual(full.current_day, 406)
        self.assertEqual(full.remaining_dv, 10000)

    def test_cache_reuses_success_and_failure_and_reserves_validation(self):
        calculate = Mock(side_effect=[42, ValueError("Lambert failed")])
        cache = experiment.TransferCache(calculate, math.inf)
        with patch.object(experiment, "SOLVE_LIMIT", experiment.VALIDATION_SOLVES + 2):
            for _ in range(2):
                self.assertEqual(cache.transfer(410, 130, 4745, 210), 42)
                self.assertTrue(math.isnan(cache.transfer(410, 130, 4745, 207)))
            with self.assertRaises(experiment.BudgetExceeded):
                cache.transfer(410, 130, 4745, 213)
        self.assertEqual(calculate.call_count, 2)
        self.assertEqual(calculate.call_args_list[0].args, (410, 4745, 130, 210))

    def test_rank_uses_minimum_then_fuel(self):
        states = [
            SimpleNamespace(collected_materials=(4.3, 9, 9), remaining_dv=10000),
            SimpleNamespace(collected_materials=(4.5, 4.4, 4.6), remaining_dv=1),
            SimpleNamespace(collected_materials=(4.4, 4.5, 4.6), remaining_dv=2),
        ]
        self.assertIs(max(states, key=experiment.route_rank), states[-1])

    def test_shortlist_keeps_incumbent_and_excludes_used_wrong_type_and_low_yield(self):
        orbit = lambda i: SimpleNamespace(eph=lambda _: ((i, 0, 0), (0, 0, 0)))
        udp = SimpleNamespace(
            asteroids=[orbit(i) for i in range(6)],
            asteroid_materials_types=[0, 0, 0, 1, 0, 0],
            asteroid_masses=[0.8, 0.7, 1, 1, 1, 1],
        )
        ids = experiment.shortlist(udp, {0, 2}, 0, 0, 0, ((0, 50), (3, 50)),
                                   (204, 210), 31, 0.9, 2)
        self.assertEqual(ids, [0, 4])

    def test_one_validation_preserves_anchors_and_suffix_and_saves_subtarget_gain(self):
        saved = json.loads(experiment.SOURCE.read_text(encoding="utf-8"))
        original = tuple(self.beam.RouteStop(*fields) for fields in zip(
            saved["visited_asteroids"], saved["arrival_times"], saved["mining_times"], strict=True,
        ))
        catalog = Path("data/spoc/mining/candidates.txt").read_text().splitlines()
        masses = [float(row.split()[-2]) for row in catalog]
        materials = [int(float(row.split()[-1])) for row in catalog]
        masses[9999], materials[9999] = 0.9, 0
        score = experiment.INCUMBENT_SCORE + (0.9 - masses[4745])
        validate = Mock(return_value=score)
        # Only the late refuel arrival is reachable for the challenger, and
        # the resulting partial tank suffices for the unchanged suffix.
        def calculate(source, target, departure, arrival):
            if source == 410 and target == 9999:
                return 9600 if arrival != 213 else 3000
            if source == 772 and target == 1850:
                return {378: 8200, 382: 6500, 386: 6200, 390: 2000}[arrival]
            return 500

        solver = SimpleNamespace(
            BeamState=self.beam.BeamState, RouteStop=self.beam.RouteStop,
            apply_candidate=self.beam.apply_candidate, calculate_transfer_dv=Mock(side_effect=calculate),
            validate_route=validate, esa={"T_START": SimpleNamespace(mjd2000=0)},
            udp=SimpleNamespace(asteroid_materials_types=materials, asteroid_masses=masses),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / experiment.OUTPUT.name
            with (patch.dict("sys.modules", {"beam_search": solver}),
                  patch.object(experiment, "shortlist", side_effect=([4745, 9999], [772])),
                  patch.object(experiment, "OUTPUT", output),
                  redirect_stdout(StringIO()) as report):
                experiment.run(time.monotonic() + 10)
            result = json.loads(output.read_text(encoding="utf-8"))
            with self.assertRaises(FileExistsError):
                experiment.export_result(validate.call_args.args[0], score, output=output)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), result)

        validate.assert_called_once()
        self.assertEqual(validate.call_args.kwargs, {"require_full_route": True})
        route = validate.call_args.args[0].route
        self.assertEqual(route[:2], original[:2])
        self.assertEqual(route[5:], original[5:])
        self.assertEqual(route[2].arrival_day, 213)
        self.assertEqual(route[4], self.beam.RouteStop(1850, 390, 16))
        self.assertLessEqual(solver.calculate_transfer_dv.call_count + 44, 2500)
        self.assertEqual(result["official_score"], score)
        self.assertIn("target not met", report.getvalue())

    def test_worker_timeout_includes_validation(self):
        with patch.object(experiment.subprocess, "run") as worker, redirect_stdout(StringIO()):
            worker.side_effect = subprocess.TimeoutExpired("worker", 60)
            self.assertEqual(experiment.main(), 1)
        self.assertGreater(worker.call_args.kwargs["timeout"], 0)
        self.assertLessEqual(worker.call_args.kwargs["timeout"], 60)
