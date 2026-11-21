"""Fast candidate selection tests with synthetic orbits and a stub Lambert solver."""

import importlib.util
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch


class CandidateShortlistTests(unittest.TestCase):
    def setUp(self):
        self.seen = set()

        def ephemeris(asteroid_id):
            def eph(epoch):
                self.seen.add(asteroid_id)
                offset = min(asteroid_id, 10000 - asteroid_id) * 1000
                return (offset, 0, 0), (offset, 0, 0)
            return eph

        udp = SimpleNamespace(
            asteroids=[SimpleNamespace(eph=ephemeris(i)) for i in range(10000)],
            asteroid_materials_types=[0] * 10000,
            asteroid_masses=[1.0] * 10000,
            MU=1,
        )
        self.lambert = Mock(return_value=Mock(
            get_v1=Mock(return_value=[(0, 0, 0)]),
            get_v2=Mock(return_value=[(0, 0, 0)]),
        ))
        pykep = SimpleNamespace(DAY2SEC=86400, lambert_problem=self.lambert)
        numpy = SimpleNamespace(array=MagicMock(), linalg=SimpleNamespace(norm=lambda _: 0))
        catalog = {"udp": udp, "T_START": SimpleNamespace(mjd2000=0)}
        spec = importlib.util.spec_from_file_location("shortlist_under_test", "find_reachable.py")
        self.reachable = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {"pykep": pykep, "numpy": numpy}):
            with patch("runpy.run_path", return_value=catalog):
                spec.loader.exec_module(self.reachable)

    def test_full_catalog_coverage_and_deterministic_diversity(self):
        args = (0, 20, 20 * 86400, 3000, {0, 1}, 10, (1, 0, 0))
        selected = self.reachable.shortlist_asteroids(*args)
        self.assertEqual(self.seen, set(range(10000)) - {1})
        self.assertEqual(selected, self.reachable.shortlist_asteroids(*args))
        self.assertEqual(len(set(selected)), 10)
        self.assertIn(9999, selected)
        self.assertGreater(min(selected[-1], 10000 - selected[-1]), 5)
        self.assertTrue({0, 1}.isdisjoint(selected))

    def test_both_modes_use_same_lambert_budget_across_valid_dates(self):
        for deficits in (None, (1, 0, 0)):
            with self.subTest(deficits=deficits):
                self.lambert.reset_mock()
                probes = []
                candidates = self.reachable.search_multiple_transfers(
                    0, 1787, 3000, {0, 1, 9998}, limit=10,
                    deficits=deficits, on_probes=probes.append,
                )
                self.assertEqual(probes, [8, 8])
                self.assertEqual(self.lambert.call_count, 16)
                self.assertEqual({row["arrival_day"] for row in candidates}, {1807, 1827})
                ids = {row["id"] for row in candidates}
                self.assertTrue({0, 1, 9998}.isdisjoint(ids))
                self.assertEqual(any(asteroid_id >= 10 for asteroid_id in ids), deficits is not None)
