"""Host-memory acquisition and logging share one measurement."""

import logging
from collections import Counter
from pathlib import Path

from vrl.utils.memory import HostMemoryMonitor, HostMemorySnapshot


def test_capture_reads_each_proc_table_once_and_logs_same_values(tmp_path, monkeypatch, caplog):
    (tmp_path / "self").mkdir()
    status = tmp_path / "self/status"
    meminfo = tmp_path / "meminfo"
    status.write_text("Name:\ttrainer\nVmRSS:\t2048 kB\nThreads:\t4\n")
    meminfo.write_text("MemTotal: 8192 kB\nMemAvailable: 2048 kB\nHugePages_Total: 0\n")
    opened = Counter()
    original_open = Path.open

    def track_open(path, *args, **kwargs):
        opened[path] += 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", track_open)
    logger = logging.getLogger("test.host_memory")
    monitor = HostMemoryMonitor(logger=logger, proc_root=tmp_path)
    with caplog.at_level(logging.INFO, logger=logger.name):
        snapshot = monitor.log("after_load")

    assert snapshot == HostMemorySnapshot(rss_mb=2.0, available_mb=2.0, total_mb=8.0)
    assert snapshot.used_fraction == 0.75
    assert opened == {status: 1, meminfo: 1}
    assert (
        "host_memory[after_load]: rss=2.0MiB available=2.0MiB total=8.0MiB used=0.750"
        in caplog.text
    )


def test_missing_tables_and_fields_remain_unknown(tmp_path):
    monitor = HostMemoryMonitor(proc_root=tmp_path)
    assert str(monitor.capture()) == "unavailable"
    (tmp_path / "meminfo").write_text("MemTotal: 4096 kB\nMemAvailable:\n")
    snapshot = monitor.capture()
    assert snapshot == HostMemorySnapshot(rss_mb=None, available_mb=None, total_mb=4.0)
    assert snapshot.used_fraction is None
