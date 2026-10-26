"""Small regression for report serialization; no model/dependency initialization."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("release_check_utils", Path(__file__).resolve().parents[1] / "scripts/release_check_utils.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)

class ReportSerializationTest(unittest.TestCase):
    def test_supported_and_failed_serialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            value = {"set": {3, 1, 2}, "frozen": frozenset({"b", "a"}), "path": Path("relative/model")}
            helpers.dump(path, value)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"set": [1, 2, 3], "frozen": ["a", "b"], "path": str(value["path"])})
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                helpers.dump(path, value)
            self.assertEqual(path.read_bytes(), original)
            for value in ({"bad": object()}, {"bad": float("nan")}):
                failed = Path(tmp) / "failed.json"
                with self.assertRaises((TypeError, ValueError)):
                    helpers.dump(failed, value)
                self.assertFalse(failed.exists())
            self.assertEqual(json.dumps({"x": {3, 1, 2}}, default=helpers._report_json_default), json.dumps({"x": {2, 3, 1}}, default=helpers._report_json_default))

if __name__ == "__main__":
    unittest.main()
