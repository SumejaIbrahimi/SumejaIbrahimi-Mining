"""Fast, mocked checks for the early gold experiment."""

from collections import namedtuple
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from early_gold_experiment import OUTPUT, SOURCE, export_result, main, run


class EarlyGoldExperimentTests(unittest.TestCase):
    def test_preserves_schedule_and_reports_first_failure(self):
        saved = json.loads(SOURCE.read_text(encoding="utf-8"))
        stop_type = namedtuple("Stop", "asteroid_id arrival_day mining_days")
        state_type = namedtuple(
            "State", "current_id current_day remaining_dv collected_materials visited_asteroids route",
        )
        original = tuple(map(stop_type._make, zip(
            saved["visited_asteroids"], saved["arrival_times"], saved["mining_times"],
        )))
        catalog = [
            line.split()[-2:]
            for line in Path("data/spoc/mining/candidates.txt").read_text().splitlines()
        ]
        masses = [float(mass) for mass, _ in catalog]
        materials = [int(float(material)) for _, material in catalog]
        for i in range(9000, 9008):
            masses[i], materials[i] = 1.0, 0

        transfer_calls = []

        def transfer(start_id, target_id, departure_day, arrival_day):
            transfer_calls.append((start_id, target_id, departure_day, arrival_day))
            if target_id == 9000:
                raise RuntimeError("test failure")
            return {(410, 9001): 11000, (410, 9002): 5000, (9002, 772): 3000}.get(
                (start_id, target_id), 1000,
            )

        def apply(state, candidate, stay):
            stop = stop_type(candidate["id"], candidate["arrival_day"], stay)
            mined = min(candidate["mass"], stay / 30)
            totals = list(state.collected_materials)
            fuel = state.remaining_dv - candidate["dv"]
            if candidate["material"] == 3:
                fuel = min(10000, fuel + 10000 * mined)
            else:
                totals[candidate["material"]] += mined
            return state_type(
                stop.asteroid_id, stop.arrival_day + stay, fuel, tuple(totals),
                state.visited_asteroids | {stop.asteroid_id}, state.route + (stop,),
            )

        solver = SimpleNamespace(
            BeamState=state_type, RouteStop=stop_type, apply_candidate=apply,
            calculate_transfer_dv=transfer, validate_route=Mock(return_value=4.3899),
            esa={"T_START": SimpleNamespace(mjd2000=0)},
            udp=SimpleNamespace(asteroid_materials_types=materials, asteroid_masses=masses),
        )
        with TemporaryDirectory() as directory:
            destination = Path(directory) / OUTPUT
            incumbent_copy = Path(directory) / SOURCE
            incumbent_bytes = SOURCE.read_bytes()
            incumbent_copy.write_bytes(incumbent_bytes)

            def validate(state, *, require_full_route):
                self.assertTrue(require_full_route)
                self.assertFalse(destination.exists())
                return 4.3899

            solver.validate_route.side_effect = validate
            with patch.dict("sys.modules", {"beam_search": solver}), redirect_stdout(StringIO()) as output:
                with (
                    patch("early_gold_experiment.SOURCE", incumbent_copy),
                    patch("early_gold_experiment.OUTPUT", destination),
                    patch("early_gold_experiment.gold_candidates", return_value=range(9000, 9009)) as select,
                ):
                    run()
                    exported = destination.read_bytes()
                    state = solver.validate_route.call_args.args[0]
                    with self.assertRaises(FileExistsError):
                        export_result(state, 4.39)
                    self.assertEqual(destination.read_bytes(), exported)
            self.assertEqual(incumbent_copy.read_bytes(), incumbent_bytes)
        self.assertEqual(json.loads(exported), {
            "algorithm": "Early gold local search",
            "candidate_strategy": "catalog-wide",
            "score": min(state.collected_materials),
            "official_score": 4.3899,
            "visited_asteroids": [stop.asteroid_id for stop in state.route],
            "arrival_times": [stop.arrival_day for stop in state.route],
            "mining_times": [stop.mining_days for stop in state.route],
            "collected_materials": dict(zip(("0", "1", "2"), state.collected_materials)),
            "remaining_dv": state.remaining_dv,
            "final_day": state.current_day,
        })

        self.assertEqual([call.args[-2:] for call in select.call_args_list], [(205, 29), (210, 27)])
        self.assertLessEqual(len(transfer_calls) + 44, 98)
        self.assertIn((410, 9000, 130, 205), transfer_calls)
        solver.validate_route.assert_called_once()
        self.assertEqual(solver.validate_route.call_args.kwargs, {"require_full_route": True})
        route = solver.validate_route.call_args.args[0].route
        self.assertEqual(route[:2], original[:2])
        self.assertEqual(route[2:4], (stop_type(9003, 205, 29), stop_type(772, 294, 8)),
                         output.getvalue())
        self.assertEqual(route[4:], original[4:])
        self.assertIn("410 -> 9000: Lambert solver error", output.getvalue())
        self.assertIn("410 -> 9001: insufficient fuel", output.getvalue())
        self.assertIn("772 -> 1850: refuel did not restore a full tank", output.getvalue())
        self.assertIn("official ESA score 4.3899", output.getvalue())

    def test_worker_timeout(self):
        with patch("early_gold_experiment.subprocess.run") as worker, redirect_stdout(StringIO()):
            worker.side_effect = subprocess.TimeoutExpired("worker", 30)
            self.assertEqual(main(), 1)
        self.assertEqual(worker.call_args.kwargs["timeout"], 30)
