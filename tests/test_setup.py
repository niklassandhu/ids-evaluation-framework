import importlib.util
import subprocess
import sys
import unittest


class TestSetup(unittest.TestCase):
    def test_numpy_import(self):
        numpy_spec = importlib.util.find_spec("numpy")
        if numpy_spec is None:
            self.fail("numpy is not installed")

    def test_package_import(self):
        pkg_spec = importlib.util.find_spec("ids_eval")
        if pkg_spec is None:
            self.fail("ids_eval package is not importable")
        import ids_eval  # noqa: F401

    def test_cli_version(self):
        # Use module invocation to avoid relying on entry point installation in the environment
        proc = subprocess.run([sys.executable, "-m", "ids_eval.main", "version"], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("IDS-EVAL Version", proc.stdout)


if __name__ == "__main__":
    unittest.main()
