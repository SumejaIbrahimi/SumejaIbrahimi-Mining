"""Test the catalog-wide JSON export without loading PyKEP or running a search."""

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class CatalogWideExportTests(unittest.TestCase):
    def test_export_schema_and_exclusive_creation(self):
        spec = importlib.util.spec_from_file_location(
            "beam_export_under_test", Path(__file__).with_name("beam_search.py"),
        )
        beam = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {
            spec.name: beam,
            "numpy": SimpleNamespace(),
            "pykep": SimpleNamespace(),
            "find_reachable": SimpleNamespace(search_multiple_transfers=None, esa=None, udp=None),
        }):
            spec.loader.exec_module(beam)

        state = beam.BeamState(
            current_id=5025,
            current_day=1806,
            remaining_dv=898.93,
            collected_materials=(3.87, 3.86, 3.88),
            visited_asteroids=frozenset({300, 5025}),
            route=(beam.RouteStop(300, 0, 1), beam.RouteStop(5025, 1776, 30)),
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / beam.CATALOG_WIDE_RESULT
            protected = [root / name for name in (
                "three_stop_best_result.json", "beam_best_result.json", "baseline_result.json",
            )]
            for file in protected:
                file.write_text("original", encoding="utf-8")
            with patch.object(beam, "CATALOG_WIDE_RESULT", output):
                beam.export_catalog_wide_result(state, 3.860000001)
                with self.assertRaises(FileExistsError):
                    beam.export_catalog_wide_result(state, 3.869)

            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {
                "algorithm": "Beam Search",
                "candidate_strategy": "catalog-wide",
                "score": 3.86,
                "official_score": 3.860000001,
                "visited_asteroids": [300, 5025],
                "arrival_times": [0, 1776],
                "mining_times": [1, 30],
                "collected_materials": {"0": 3.87, "1": 3.86, "2": 3.88},
                "remaining_dv": 898.93,
                "final_day": 1806,
            })
            for file in protected:
                self.assertEqual(file.read_text(encoding="utf-8"), "original")
