"""Fast checks for route preservation and both search limits."""

from collections import namedtuple
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from two_stop_local_search import SOURCE, main, run


class TwoStopLocalSearchTests(unittest.TestCase):
    def test_fixed_segment_and_bounded_full_validation(self):
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
        for i in range(9000, 9017):
            masses[i], materials[i] = 1.0, 0 if i < 9009 else 1
        calculate = Mock(return_value=1.0)

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
            calculate_transfer_dv=calculate, validate_route=Mock(return_value=4.395),
            esa={"T_START": SimpleNamespace(mjd2000=0)},
            udp=SimpleNamespace(asteroid_materials_types=materials, asteroid_masses=masses),
        )
        with patch.dict("sys.modules", {"beam_search": solver}), redirect_stdout(StringIO()) as output:
            with patch("two_stop_local_search.shortlist", return_value=range(9000, 9009)) as gold:
                with patch("two_stop_local_search.platinum_options", return_value=[8719, 9009, 9010, 9011]) as platinum:
                    run()

        self.assertEqual([call.args[-2:] for call in gold.call_args_list], [(1611, 24), (1613, 22)])
        self.assertEqual(platinum.call_count, 32)
        self.assertEqual({call.args[3:5] for call in platinum.call_args_list}, {(1650, 25), (1652, 23)})
        self.assertLessEqual(calculate.call_count + 44, 274)
        solver.validate_route.assert_called_once()
        self.assertEqual(solver.validate_route.call_args.kwargs, {"require_full_route": True})
        route = solver.validate_route.call_args.args[0].route
        self.assertEqual(route[:18], original[:18])
        self.assertEqual(route[20:], original[20:])
        self.assertEqual(route[18], stop_type(9000, 1611, 24))
        self.assertIn("Screened 96 pairs", output.getvalue())
        self.assertIn("Official ESA score 4.395", output.getvalue())

    def test_worker_timeout(self):
        with patch("two_stop_local_search.subprocess.run") as worker, redirect_stdout(StringIO()):
            worker.side_effect = subprocess.TimeoutExpired("worker", 30)
            self.assertEqual(main(), 1)
        self.assertEqual(worker.call_args.kwargs["timeout"], 30)
