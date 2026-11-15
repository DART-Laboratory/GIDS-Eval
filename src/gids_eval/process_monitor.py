from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass
class ProcessUsage:
    exit_code: int
    wall_time_sec: float
    peak_rss_kb: int
    mean_rss_kb: float
    peak_cpu_percent: float
    mean_cpu_percent: float
    peak_process_count: int
    sample_count: int
    cpu_sample_count: int

def run_command_logged(
    command: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    stdout_path: str | Path,
    stderr_path: str | Path,
    poll_interval: float = 0.5,
    monitor_resources: bool = True,
) -> ProcessUsage:
    stdout_target = Path(stdout_path)
    stderr_target = Path(stderr_path)
    stdout_target.parent.mkdir(parents=True, exist_ok=True)
    stderr_target.parent.mkdir(parents=True, exist_ok=True)

    start_t = time.perf_counter()
    with stdout_target.open("w", encoding="utf-8") as stdout_handle:
        with stderr_target.open("w", encoding="utf-8") as stderr_handle:
            proc = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
            )
            monitor = _ProcessTreeMonitor(proc.pid) if monitor_resources else None
            while True:
                if monitor is not None:
                    monitor.sample()
                exit_code = proc.poll()
                if exit_code is not None:
                    break
                time.sleep(poll_interval)
            if monitor is not None:
                monitor.sample()

    wall_time_sec = round(time.perf_counter() - start_t, 6)
    if monitor is None:
        usage = ProcessUsage(
            exit_code=proc.returncode or 0,
            wall_time_sec=wall_time_sec,
            peak_rss_kb=0,
            mean_rss_kb=0.0,
            peak_cpu_percent=0.0,
            mean_cpu_percent=0.0,
            peak_process_count=0,
            sample_count=0,
            cpu_sample_count=0,
        )
    else:
        usage = monitor.finish(exit_code=proc.returncode or 0, wall_time_sec=wall_time_sec)
    return usage

def derive_phase_profile_path(metrics_path: str | Path | None) -> str | None:
    if not metrics_path:
        return None
    return str(Path(metrics_path).with_name("resource_profile.json"))

def load_phase_profile_usage(path: str | Path | None) -> ProcessUsage | None:
    if not path:
        return None
    target = Path(path)
    if not target.exists():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    aggregate = payload.get("aggregate") or {}
    return ProcessUsage(
        exit_code=int(aggregate.get("exit_code", 0) or 0),
        wall_time_sec=float(aggregate.get("wall_time_sec", 0.0) or 0.0),
        peak_rss_kb=int(aggregate.get("peak_rss_kb", 0) or 0),
        mean_rss_kb=float(aggregate.get("mean_rss_kb", 0.0) or 0.0),
        peak_cpu_percent=float(aggregate.get("peak_cpu_percent", 0.0) or 0.0),
        mean_cpu_percent=float(aggregate.get("mean_cpu_percent", 0.0) or 0.0),
        peak_process_count=int(aggregate.get("peak_process_count", 0) or 0),
        sample_count=int(aggregate.get("sample_count", 0) or 0),
        cpu_sample_count=int(aggregate.get("cpu_sample_count", 0) or 0),
    )

@contextmanager
def maybe_profile_phase(name: str, *, poll_interval: float = 0.2) -> Any:
    profile_path = os.environ.get("GIDS_PHASE_PROFILE_PATH")
    if not profile_path:
        yield
        return
    collector = PhaseProfileCollector(profile_path, poll_interval=poll_interval)
    with collector.profile(name):
        yield

class PhaseProfileCollector:
    def __init__(self, path: str | Path, *, poll_interval: float = 0.2) -> None:
        self.path = Path(path)
        self.poll_interval = poll_interval
        self.phases = self._load_existing_phases()

    @contextmanager
    def profile(self, name: str) -> Any:
        phase_name = self._reserve_name(name)
        monitor = _ProcessTreeMonitor(os.getpid())
        stop_event = threading.Event()

        def _sample_loop() -> None:
            while not stop_event.is_set():
                monitor.sample()
                stop_event.wait(self.poll_interval)
            monitor.sample()

        start_t = time.perf_counter()
        worker = threading.Thread(target=_sample_loop, daemon=True)
        worker.start()
        try:
            yield
        finally:
            stop_event.set()
            worker.join()
            usage = monitor.finish(exit_code=0, wall_time_sec=round(time.perf_counter() - start_t, 6))
            self.phases[phase_name] = {
                "wall_time_sec": usage.wall_time_sec,
                "peak_rss_kb": usage.peak_rss_kb,
                "mean_rss_kb": usage.mean_rss_kb,
                "peak_cpu_percent": usage.peak_cpu_percent,
                "mean_cpu_percent": usage.mean_cpu_percent,
                "peak_process_count": usage.peak_process_count,
                "sample_count": usage.sample_count,
                "cpu_sample_count": usage.cpu_sample_count,
                "exit_code": usage.exit_code,
            }
            self._write()

    def _load_existing_phases(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        phases = payload.get("phases")
        if not isinstance(phases, dict):
            return {}
        return {str(key): dict(value) for key, value in phases.items()}

    def _reserve_name(self, preferred: str) -> str:
        if preferred not in self.phases:
            return preferred
        suffix = 1
        while f"{preferred}_{suffix}" in self.phases:
            suffix += 1
        return f"{preferred}_{suffix}"

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "phases": self.phases,
                    "aggregate": _aggregate_phase_usage(self.phases.values()),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

def _aggregate_phase_usage(rows: Any) -> dict[str, Any]:
    rows = [dict(row) for row in rows]
    if not rows:
        return {
            "exit_code": 0,
            "wall_time_sec": 0.0,
            "peak_rss_kb": 0,
            "mean_rss_kb": 0.0,
            "peak_cpu_percent": 0.0,
            "mean_cpu_percent": 0.0,
            "peak_process_count": 0,
            "sample_count": 0,
            "cpu_sample_count": 0,
        }
    sample_count = sum(int(row.get("sample_count", 0) or 0) for row in rows)
    cpu_sample_count = sum(int(row.get("cpu_sample_count", 0) or 0) for row in rows)
    mean_rss_num = sum(float(row.get("mean_rss_kb", 0.0) or 0.0) * int(row.get("sample_count", 0) or 0) for row in rows)
    mean_cpu_num = sum(
        float(row.get("mean_cpu_percent", 0.0) or 0.0) * int(row.get("cpu_sample_count", 0) or 0)
        for row in rows
    )
    return {
        "exit_code": max(int(row.get("exit_code", 0) or 0) for row in rows),
        "wall_time_sec": round(sum(float(row.get("wall_time_sec", 0.0) or 0.0) for row in rows), 6),
        "peak_rss_kb": max(int(row.get("peak_rss_kb", 0) or 0) for row in rows),
        "mean_rss_kb": round(mean_rss_num / sample_count, 3) if sample_count else 0.0,
        "peak_cpu_percent": round(max(float(row.get("peak_cpu_percent", 0.0) or 0.0) for row in rows), 3),
        "mean_cpu_percent": round(mean_cpu_num / cpu_sample_count, 3) if cpu_sample_count else 0.0,
        "peak_process_count": max(int(row.get("peak_process_count", 0) or 0) for row in rows),
        "sample_count": sample_count,
        "cpu_sample_count": cpu_sample_count,
    }

class _ProcessTreeMonitor:
    def __init__(self, root_pid: int) -> None:
        self.root_pid = int(root_pid)
        self.clock_ticks = max(int(os.sysconf("SC_CLK_TCK")), 1)
        self.cpu_count = max(os.cpu_count() or 1, 1)
        self.sample_count = 0
        self.cpu_sample_count = 0
        self.total_rss_kb = 0
        self.total_cpu_percent = 0.0
        self.peak_rss_kb = 0
        self.peak_cpu_percent = 0.0
        self.peak_process_count = 0
        self._previous_sample_at: float | None = None
        self._previous_cpu_ticks: int | None = None

    def sample(self) -> None:
        sample_at = time.perf_counter()
        tree_pids = _collect_tree_pids(self.root_pid)
        process_count = len(tree_pids)
        rss_kb = sum(_read_rss_kb(pid) for pid in tree_pids)
        cpu_ticks = sum(_read_cpu_ticks(pid) for pid in tree_pids)
        self.sample_count += 1
        self.total_rss_kb += rss_kb
        self.peak_rss_kb = max(self.peak_rss_kb, rss_kb)
        self.peak_process_count = max(self.peak_process_count, process_count)
        if self._previous_sample_at is not None and self._previous_cpu_ticks is not None:
            elapsed_sec = sample_at - self._previous_sample_at
            delta_ticks = max(0, cpu_ticks - self._previous_cpu_ticks)
            if elapsed_sec > 0:
                cpu_percent = (delta_ticks / self.clock_ticks) / (elapsed_sec * self.cpu_count) * 100.0
                self.cpu_sample_count += 1
                self.total_cpu_percent += cpu_percent
                self.peak_cpu_percent = max(self.peak_cpu_percent, cpu_percent)
        self._previous_sample_at = sample_at
        self._previous_cpu_ticks = cpu_ticks

    def finish(self, *, exit_code: int, wall_time_sec: float) -> ProcessUsage:
        mean_rss_kb = (self.total_rss_kb / self.sample_count) if self.sample_count else 0.0
        mean_cpu_percent = (
            self.total_cpu_percent / self.cpu_sample_count if self.cpu_sample_count else 0.0
        )
        return ProcessUsage(
            exit_code=exit_code,
            wall_time_sec=wall_time_sec,
            peak_rss_kb=self.peak_rss_kb,
            mean_rss_kb=round(mean_rss_kb, 3),
            peak_cpu_percent=round(self.peak_cpu_percent, 3),
            mean_cpu_percent=round(mean_cpu_percent, 3),
            peak_process_count=self.peak_process_count,
            sample_count=self.sample_count,
            cpu_sample_count=self.cpu_sample_count,
        )

def _collect_tree_pids(root_pid: int) -> set[int]:
    children_by_ppid: dict[int, list[int]] = {}
    for pid, ppid in _iter_pid_ppid_pairs():
        children_by_ppid.setdefault(ppid, []).append(pid)

    tree_pids: set[int] = set()
    pending = [root_pid]
    while pending:
        current = pending.pop()
        if current in tree_pids:
            continue
        tree_pids.add(current)
        pending.extend(children_by_ppid.get(current, []))
    return tree_pids

def _iter_pid_ppid_pairs() -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        stat_path = Path(entry.path) / "stat"
        try:
            content = stat_path.read_text(encoding="utf-8")
        except OSError:
            continue
        right_paren = content.rfind(")")
        if right_paren == -1:
            continue
        parts = content[right_paren + 2 :].split()
        if len(parts) < 2:
            continue
        try:
            pid = int(entry.name)
            ppid = int(parts[1])
        except ValueError:
            continue
        pairs.append((pid, ppid))
    return pairs

def _read_rss_kb(pid: int) -> int:
    status_path = Path("/proc") / str(pid) / "status"
    try:
        with status_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith("VmRSS:"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1])
    except (OSError, ValueError):
        return 0
    return 0

def _read_cpu_ticks(pid: int) -> int:
    stat_path = Path("/proc") / str(pid) / "stat"
    try:
        content = stat_path.read_text(encoding="utf-8")
    except OSError:
        return 0
    right_paren = content.rfind(")")
    if right_paren == -1:
        return 0
    parts = content[right_paren + 2 :].split()
    if len(parts) <= 12:
        return 0
    try:
        return int(parts[11]) + int(parts[12])
    except ValueError:
        return 0
