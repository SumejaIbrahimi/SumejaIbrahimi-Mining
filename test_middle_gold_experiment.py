"""Fast checks for the fixed schedule, solve budget and validated export."""

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

from middle_gold_experiment import OUTPUT, SOURCE, export_result, main, run


class MiddleGoldExperimentTests(unittest.TestCase):
    def test_full_budget_preserves_schedule_and_exports_only_after_validation(self):
        saved_bytes = SOURCE.read_bytes()
        saved = json.loads(saved_bytes)
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
        for i in range(9000, 9040):
            masses[i], materials[i] = 1.0, 0

        def transfer(*, start_id, target_id, departure_day, arrival_day):
            if start_id == 9000:
                raise RuntimeError("test Lambert failure")
            return 10001.0 if start_id == 9001 else 100.0

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
            calculate_transfer_dv=Mock(side_effect=transfer),
            esa={"T_START": SimpleNamespace(mjd2000=0)},
            udp=SimpleNamespace(asteroid_materials_types=materials, asteroid_masses=masses),
        )
        with TemporaryDirectory() as directory:
            destination = Path(directory) / OUTPUT
            incumbent = Path(directory) / SOURCE
            incumbent.write_bytes(saved_bytes)

            def validate(state, *, require_full_route):
                self.assertTrue(require_full_route)
                self.assertFalse(destination.exists())
                self.assertEqual(solver.calculate_transfer_dv.call_count, 94)
                return min(state.collected_materials)

            solver.validate_route = Mock(side_effect=validate)
            selections = [range(start, start + 13) for start in (9000, 9013, 9026)]
            with (
                patch.dict("sys.modules", {"beam_search": solver}),
                patch("middle_gold_experiment.SOURCE", incumbent),
                patch("middle_gold_experiment.OUTPUT", destination),
                patch("middle_gold_experiment.gold_candidates", side_effect=selections) as select,
                redirect_stdout(StringIO()) as output,
            ):
                run()
                state = solver.validate_route.call_args.args[0]
                exported = destination.read_bytes()
                with self.assertRaises(FileExistsError):
                    export_result(state, 4.5)
                self.assertEqual(destination.read_bytes(), exported)
            self.assertEqual(incumbent.read_bytes(), saved_bytes)

        solver.validate_route.assert_called_once()
        self.assertEqual([call.args[-3:] for call in select.call_args_list],
                         [(604, 634, 30), (605, 634, 29), (606, 634, 28)])
        self.assertEqual(state.route[:6], original[:6])
        self.assertEqual(state.route[6], stop_type(9002, 604, 30))
        self.assertEqual(state.route[7:], original[7:])
        for call in solver.calculate_transfer_dv.call_args_list[22:]:
            leg = call.kwargs
            if leg["start_id"] == 13:
                self.assertEqual(leg["departure_day"], 527)
            else:
                self.assertEqual((leg["departure_day"], leg["target_id"], leg["arrival_day"]),
                                 (634, 276, 694))
        data = json.loads(exported)
        self.assertEqual(set(data), set(saved))
        self.assertEqual(data["official_score"], min(state.collected_materials))
        self.assertEqual(data["visited_asteroids"], [stop.asteroid_id for stop in state.route])
        self.assertEqual(data["arrival_times"], [stop.arrival_day for stop in state.route])
        self.assertEqual(data["mining_times"], [stop.mining_days for stop in state.route])
        self.assertEqual(data["remaining_dv"], state.remaining_dv)
        self.assertIn("9000 -> 276: Lambert solver error", output.getvalue())
        self.assertIn("9001 -> 276: insufficient fuel", output.getvalue())

    def test_worker_timeout(self):
        with patch("middle_gold_experiment.subprocess.run") as worker, redirect_stdout(StringIO()):
            worker.side_effect = subprocess.TimeoutExpired("worker", 30)
            self.assertEqual(main(), 1)
        self.assertEqual(worker.call_args.kwargs["timeout"], 30)
