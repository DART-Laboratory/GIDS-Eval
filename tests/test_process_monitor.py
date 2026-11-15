from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gids_eval.process_monitor import (
    derive_phase_profile_path,
    load_phase_profile_usage,
    maybe_profile_phase,
    run_command_logged,
)

class ProcessMonitorTests(unittest.TestCase):
    def test_run_command_logged_can_disable_resource_monitoring(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            usage = run_command_logged(
                ["/bin/sh", "-lc", "sleep 0.05"],
                cwd=tmp_dir,
                env=dict(os.environ),
                stdout_path=tmp_path / "stdout.log",
                stderr_path=tmp_path / "stderr.log",
                monitor_resources=False,
            )

        self.assertEqual(usage.exit_code, 0)
        self.assertGreater(usage.wall_time_sec, 0.0)
        self.assertEqual(usage.sample_count, 0)
        self.assertEqual(usage.cpu_sample_count, 0)
        self.assertEqual(usage.peak_rss_kb, 0)
        self.assertEqual(usage.peak_cpu_percent, 0.0)

    def test_maybe_profile_phase_writes_aggregate_usage(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            metrics_path = Path(tmp_dir) / "metrics.csv"
            profile_path = Path(derive_phase_profile_path(metrics_path))
            previous = os.environ.get("GIDS_PHASE_PROFILE_PATH")
            os.environ["GIDS_PHASE_PROFILE_PATH"] = str(profile_path)
            try:
                with maybe_profile_phase("train", poll_interval=0.01):
                    blob = bytearray(2_000_000)
                    time.sleep(0.03)
                    self.assertGreater(len(blob), 0)
                with maybe_profile_phase("test_0", poll_interval=0.01):
                    blob = bytearray(1_000_000)
                    time.sleep(0.03)
                    self.assertGreater(len(blob), 0)
            finally:
                if previous is None:
                    os.environ.pop("GIDS_PHASE_PROFILE_PATH", None)
                else:
                    os.environ["GIDS_PHASE_PROFILE_PATH"] = previous

            usage = load_phase_profile_usage(profile_path)

        self.assertIsNotNone(usage)
        assert usage is not None
        self.assertGreater(usage.wall_time_sec, 0.0)
        self.assertGreater(usage.sample_count, 0)
        self.assertGreater(usage.peak_rss_kb, 0)

if __name__ == "__main__":
    unittest.main()
