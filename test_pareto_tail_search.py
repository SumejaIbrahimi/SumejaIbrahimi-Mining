"""Check tail search behavior with synthetic Lambert costs."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pareto_tail_search as experiment
from test_anchored_two_stop_search import load_route_helpers


class ParetoTailTests(unittest.TestCase):
    def test_dominance_is_local_to_context_and_preserves_material_tradeoffs(self):
        def state(fuel=10, totals=(4, 5, 6), asteroid=1266, day=1753, visited=frozenset({1, 1266})):
            return SimpleNamespace(current_id=asteroid, current_day=day, visited_asteroids=visited,
                                   remaining_dv=fuel, collected_materials=totals)

        dominated = state(fuel=9)
        stronger = state()
        tradeoff = state(fuel=8, totals=(5, 5, 6))
        contexts = [state(day=1754), state(asteroid=1), state(visited=frozenset({2, 1266}))]
        self.assertEqual(
            experiment.pareto_front([dominated, stronger, state(), tradeoff, *contexts]),
            [stronger, tradeoff, *contexts],
        )
        with patch.object(experiment, "BEAM_WIDTH", 2):
            beam = experiment.choose_beam([stronger, tradeoff, *contexts], (0, 0, 0))
        self.assertEqual(len(beam), 2)
        self.assertEqual(len({(item.current_id, item.current_day) for item in beam}), 2)

    def test_terminal_mining_stays_respect_deadline(self):
        self.assertIn(13, experiment.mining_stays(10, 0.31, 1814, terminal=True))
        self.assertEqual(experiment.mining_stays(10, 0.31, 1826, terminal=True), [1])
        self.assertEqual(experiment.mining_stays(10, 0.31, 1827, terminal=True), [])

    def test_control_parent_survives_empty_beam(self):
        beam = load_route_helpers()
        stops = [beam.RouteStop(5420, 1455, 30), beam.RouteStop(831, 1565, 28),
                 beam.RouteStop(7735, 1814, 10)]
        control = [
            beam.BeamState(stop.asteroid_id, stop.arrival_day + stop.mining_days, 500,
                           (4, 4, 4), frozenset(s.asteroid_id for s in stops[:index + 1]),
                           tuple(stops[:index + 1]))
            for index, stop in enumerate(stops)
        ]
        advance = Mock(return_value=None)
        best = experiment.search_tail(control, [(831,), (7735,)], [(0, 0, 0)] * 2,
                                      lambda _: 1.0, beam.RouteStop, advance, lambda: None)
        self.assertIs(best, control[-1])
        self.assertTrue(any(call.args[0] is control[1] for call in advance.call_args_list))

    def test_replays_prefix_validates_once_and_saves_only_an_official_gain(self):
        beam = load_route_helpers()
        saved = json.loads(experiment.SOURCE.read_text(encoding="utf-8"))
        original = tuple(beam.RouteStop(*fields) for fields in zip(
            saved["visited_asteroids"], saved["arrival_times"], saved["mining_times"], strict=True,
        ))
        catalog = Path("data/spoc/mining/candidates.txt").read_text().splitlines()
        deposits = [row.split()[-2:] for row in catalog]
        masses = [float(mass) for mass, _ in deposits]
        materials = [int(float(material)) for _, material in deposits]
        masses[9999], materials[9999] = 1.0, 0

        def calculate(source, target, departure, arrival):
            # The richer asteroid requires extra time on arrival and exhausts
            # its incoming fuel budget on all other dates.
            return 11000 if target == 9999 and arrival != 1621 else 200

        def shortlist(_, excluded, epoch, material, asteroid_id, *rest):
            return [asteroid_id, 9999] if material == 0 else [asteroid_id]

        def validate(state, **kwargs):
            return min(state.collected_materials)

        solver = SimpleNamespace(
            BeamState=beam.BeamState, RouteStop=beam.RouteStop, apply_candidate=beam.apply_candidate,
            calculate_transfer_dv=Mock(side_effect=calculate),
            validate_route=Mock(side_effect=validate), esa={"T_START": SimpleNamespace(mjd2000=0)},
            udp=SimpleNamespace(asteroid_materials_types=materials, asteroid_masses=masses),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / experiment.OUTPUT.name
            with (patch.dict("sys.modules", {"beam_search": solver}),
                  patch.object(experiment, "shortlist", side_effect=shortlist),
                  patch.object(experiment, "OUTPUT", output), redirect_stdout(StringIO())):
                experiment.run(time.monotonic() + 10)
            result = json.loads(output.read_text(encoding="utf-8"))

        solver.validate_route.assert_called_once()
        self.assertEqual(solver.validate_route.call_args.kwargs, {"require_full_route": True})
        winner = solver.validate_route.call_args.args[0]
        self.assertEqual(winner.route[:17], original[:17])
        self.assertEqual(winner.route[16].arrival_day + winner.route[16].mining_days, 1485)
        self.assertEqual((winner.route[18].asteroid_id, winner.route[18].arrival_day), (9999, 1621))
        self.assertLessEqual(winner.current_day, 1827)
        self.assertGreater(result["official_score"], experiment.INCUMBENT_SCORE)
        calls = [call.args for call in solver.calculate_transfer_dv.call_args_list]
        self.assertEqual(len(calls), len(set(calls)))
        self.assertLessEqual(len(calls) + 44, experiment.SOLVE_LIMIT)
