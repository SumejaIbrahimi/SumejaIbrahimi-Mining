"""Exercise all four gold-stop schedules with a fake transfer calculator."""

from collections import namedtuple
from contextlib import redirect_stdout
from io import StringIO
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from gold_stop_experiment import SOURCE, main


class GoldStopExperimentTests(unittest.TestCase):
    def test_trial_grid_tail_replay_and_total_budget(self):
        saved = json.loads(SOURCE.read_text(encoding="utf-8"))
        ids = saved["visited_asteroids"]
        index = ids.index(4088)
        candidates = tuple(range(9000, 9008))
        stop_type = namedtuple("Stop", "asteroid_id arrival_day mining_days")
        state_type = namedtuple(
            "State", "current_id current_day remaining_dv collected_materials visited_asteroids route",
        )
        calculate = Mock(return_value=1.0)

        def apply(state, candidate, mining):
            asteroid_id = candidate["id"]
            stop = stop_type(asteroid_id, candidate["arrival_day"], mining)
            total = 4.2 if asteroid_id in candidates else state.collected_materials[0]
            return state_type(
                asteroid_id, stop.arrival_day + mining, state.remaining_dv - candidate["dv"],
                (total, 4.4, 4.4), state.visited_asteroids | {asteroid_id}, state.route + (stop,),
            )

        def validate(state, *, require_full_route):
            self.assertTrue(require_full_route)
            self.assertEqual(calculate.call_count, 86)
            for _ in range(2 * (len(state.route) - 1)):
                calculate()
            return 4.199

        solver = SimpleNamespace(
            BeamState=state_type, RouteStop=stop_type, apply_candidate=Mock(side_effect=apply),
            calculate_transfer_dv=calculate, validate_route=Mock(side_effect=validate),
            esa={"T_START": SimpleNamespace(mjd2000=0)},
            udp=SimpleNamespace(
                asteroid_materials_types=[0] * 10000, asteroid_masses=[1.0] * 10000,
            ),
        )
        with patch.dict("sys.modules", {"beam_search": solver}):
            with patch("gold_stop_experiment.shortlist", return_value=(*candidates, 9008)) as select:
                with redirect_stdout(StringIO()) as output:
                    main()

        schedules = [(1613, 17), (1611, 19), (1611, 21), (1613, 21)]
        self.assertEqual([call.args[-2:] for call in select.call_args_list], schedules)
        expected = [
            leg for day, mining in schedules for asteroid_id in candidates
            for leg in ((831, asteroid_id, 1593, day), (asteroid_id, 8719, day + mining, 1650))
        ]
        self.assertEqual([call.args for call in calculate.call_args_list[22:86]], expected)
        self.assertEqual(calculate.call_count, 130)
        self.assertEqual(solver.apply_candidate.call_count, 22 + 32 * (len(ids) - index))
        solver.validate_route.assert_called_once()
        route = solver.validate_route.call_args.args[0].route
        original = tuple(map(stop_type._make, zip(ids, saved["arrival_times"], saved["mining_times"])))
        self.assertEqual(route[:index], original[:index])
        self.assertEqual(route[index + 1:], original[index + 1:])
        self.assertIn("official ESA score 4.199", output.getvalue())
