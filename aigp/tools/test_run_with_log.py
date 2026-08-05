import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from run_with_log import _metadata, _parse_args


class ObserveOnlyRunnerTests(unittest.TestCase):
    def test_observe_only_flag_is_parsed(self):
        with patch.object(sys, "argv", ["run_with_log.py", "--observe-only"]):
            args = _parse_args()

        self.assertTrue(args.observe_only)

    def test_child_observe_only_environment_is_recorded_in_metadata(self):
        with (
            patch("run_with_log._run_git", return_value=None),
            patch("run_with_log._load_runtime_config", return_value={}),
        ):
            metadata = _metadata(
                ["python", "main.py"],
                Path.cwd(),
                Path.cwd(),
                "observe-test",
                {"OBSERVE_ONLY": "true"},
            )

        self.assertEqual(metadata["env"]["OBSERVE_ONLY"], "true")


if __name__ == "__main__":
    unittest.main()
