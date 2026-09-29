"""Verify scripts/run_tests_parallel.py kills test-spawned grandchildren.

Setup
-----
A test in this file spawns a long-lived Python grandchild that writes
its PID + a nonce to a tempfile, then exits without cleaning up.
With the old ``subprocess.run`` runner, that grandchild would orphan
and outlive the test (and the whole runner). With the current Popen +
``start_new_session`` + ``_kill_tree`` runner, the grandchild gets
SIGKILL'd via process-group kill when its file's pytest exits.

The leaker test always passes — its only job is to spawn a grandchild
and walk away. The verifier runs the runner over the leaker file in a
subprocess, then waits for the grandchild PID to disappear from the
kernel's process table.

POSIX-only: Windows has its own grandchild lifecycle (no shared session,
``taskkill /F /T`` semantics). Marked accordingly.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


# Both tests share the same handoff file: the leaker writes here, the
# verifier reads here. We park it in $TMPDIR with a unique-per-run name
# so concurrent invocations of the suite don't clobber each other.
_HANDOFF_DIR = Path(os.environ.get("TMPDIR", "/tmp")) / "hermes-isolation-probe"
_HANDOFF_DIR.mkdir(exist_ok=True)


def _handoff_path_for(nonce: str) -> Path:
    return _HANDOFF_DIR / f"grandchild-{nonce}.json"


def _pid_alive(pid: int) -> bool:
    """POSIX: send signal 0 to probe whether ``pid`` is still alive.

    ``os.kill(pid, 0)`` raises ``ProcessLookupError`` if the process is
    gone, ``PermissionError`` if it exists but we can't signal it
    (someone else's pid). We treat PermissionError as "alive" because
    the process exists and that's all we need to know.
    """
    if sys.platform == "win32":  # pragma: no cover — POSIX-only test
        # On Windows we'd use OpenProcess + GetExitCodeProcess; this
        # test is skipped on Windows so the path is unreachable.
        raise RuntimeError("_pid_alive POSIX-only")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only probe")
@pytest.mark.live_system_guard_bypass
def test_grandchild_leak_is_killed_by_runner(tmp_path: Path) -> None:
    """Run the parallel runner over a probe file and verify cleanup.

    1. Materialize a probe file that spawns a long-lived grandchild and
       writes its PID to disk before exiting.
    2. Invoke ``scripts/run_tests_parallel.py`` against the probe file.
    3. Wait for the grandchild PID to vanish (poll for ~5s).
    4. Assert the runner exited cleanly AND the grandchild is dead.
    """
    repo_root = Path(__file__).resolve().parent.parent
    runner = repo_root / "scripts" / "run_tests_parallel.py"
    assert runner.exists(), f"runner missing at {runner}"

    # Probe lives in a temp dir, NOT under tests/, so the regular suite
    # never picks it up — only our explicit invocation does.
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    probe = probe_dir / "test_probe_leaker.py"
    nonce = f"{os.getpid()}-{int(time.time() * 1000)}"
    handoff = _handoff_path_for(nonce)
    if handoff.exists():
        handoff.unlink()

    probe_src = textwrap.dedent(f"""
        import json, os, subprocess, sys, time
        from pathlib import Path

        HANDOFF = Path({str(handoff)!r})

        def test_spawns_grandchild_and_walks_away():
            # Long-lived grandchild: detached, ignores SIGTERM (we want
            # SIGKILL or process-group kill to be the only thing that
            # works, simulating a misbehaving server).
            child = subprocess.Popen(
                [
                    sys.executable, "-c",
                    "import os, signal, sys, time; "
                    "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                    "sys.stdout.write(f'gc-pgid={{os.getpgid(0)}} gc-pid={{os.getpid()}}\\\\n'); "
                    "sys.stdout.flush(); "
                    "time.sleep(600)",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                # IMPORTANT: do NOT pass start_new_session here. We want
                # the grandchild to inherit the pytest subprocess's
                # process group, so when the runner kills the group the
                # grandchild dies too.
            )
            # Read the first line so we can record gc's pgid in the
            # handoff, then walk away — don't close the pipe (would
            # signal EOF and let the child see SIGPIPE on next write).
            first_line = child.stdout.readline().decode().strip()
            HANDOFF.write_text(json.dumps({{
                "pid": child.pid,
                "diag": first_line,
                "test_pid": os.getpid(),
                "test_pgid": os.getpgid(0),
            }}))
            assert child.pid > 0
    """).strip()
    probe.write_text(probe_src + "\n")

    # Run the parallel runner against just the probe file. The runner
    # discovers under ``tests/`` by default, so we override via --paths.
    proc = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--paths",
            str(probe_dir),
            "-j",
            "1",
            # Tight per-file timeout: the probe finishes in <1s, no
            # need for 10min.
            "--file-timeout",
            "30",
        ],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
    )

    assert handoff.exists(), (
        f"probe never wrote handoff file; runner output:\n{proc.stdout}"
    )
    handoff_data = json.loads(handoff.read_text())
    grandchild_pid = handoff_data["pid"]
    diag = handoff_data.get("diag", "(no diag)")
    test_pid = handoff_data.get("test_pid")
    test_pgid = handoff_data.get("test_pgid")
    handoff.unlink()

    # The runner must have exited cleanly (probe test passes).
    assert proc.returncode == 0, (
        f"runner exited {proc.returncode}; output:\n{proc.stdout}"
    )

    # The grandchild must be gone. Poll for a bit because process-group
    # SIGKILL + reaping isn't synchronous; on a loaded box it can take
    # a beat.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not _pid_alive(grandchild_pid):
            break
        time.sleep(0.05)
    else:
        # Test cleanup: kill the leaked grandchild ourselves so a
        # FAILED assertion doesn't leave a sleep(600) running.
        try:
            os.kill(grandchild_pid, 9)
        except ProcessLookupError:
            pass
        pytest.fail(
            f"grandchild PID {grandchild_pid} survived runner exit; "
            f"diag={diag!r} test_pid={test_pid} test_pgid={test_pgid}; "
            f"runner output:\n{proc.stdout}"
        )


# ── Bare pytest-flag passthrough ─────────────────────────────────────────────
#
# The runner routes any token starting with ``-`` that isn't one of its own
# options (``-j``/``--jobs``, ``--paths``, ``--slice``, ``--file-timeout``,
# ``--generate-slices``, ``--files``, ``--include-integration``) straight
# through to each per-file pytest invocation — no ``--`` separator required.
# Before this, a bare ``-q`` errored out with "unrecognized arguments",
# forcing a retry on every run. These tests are behavior contracts, not
# snapshots: they assert that bare flags reach pytest and that value-taking
# flags (``-k expr``) keep their value instead of having it stolen by the
# positional-path discovery.


def _make_probe_dir(tmp_path: Path) -> Path:
    """Two trivial passing tests, one named test_alpha, one test_beta."""
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    (probe_dir / "test_flagprobe.py").write_text(
        "def test_alpha():\n    assert True\n\n"
        "def test_beta():\n    assert True\n"
    )
    return probe_dir


def _run_runner(probe_dir: Path, *extra: str) -> subprocess.CompletedProcess:
    repo_root = Path(__file__).resolve().parent.parent
    runner = repo_root / "scripts" / "run_tests_parallel.py"
    return subprocess.run(
        [sys.executable, str(runner), "--paths", str(probe_dir),
         "-j", "1", "--file-timeout", "30", *extra],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
    )


def test_bare_q_flag_passes_through(tmp_path: Path) -> None:
    """A bare ``-q`` (no ``--``) runs clean instead of erroring out."""
    probe_dir = _make_probe_dir(tmp_path)
    proc = _run_runner(probe_dir, "-q")
    assert proc.returncode == 0, proc.stdout
    assert "unrecognized arguments" not in proc.stdout


def test_bare_value_flag_keeps_its_value(tmp_path: Path) -> None:
    """``-k test_alpha`` reaches pytest as a selector, not as a path.

    The value token (``test_alpha``) must NOT be swallowed by the runner's
    positional-path discovery — if it were, discovery would look for a path
    named ``test_alpha``, find nothing, and the run would degrade. We assert
    the run succeeds AND only one of the two tests was selected (proving the
    ``-k`` filter actually applied inside pytest).
    """
    probe_dir = _make_probe_dir(tmp_path)
    proc = _run_runner(probe_dir, "-k", "test_alpha")
    assert proc.returncode == 0, proc.stdout
    # Exactly one test selected: the per-file summary shows "1✓" (1 passed).
    # test_beta is deselected by the -k filter.
    assert "1✓" in proc.stdout or "1 passed" in proc.stdout, proc.stdout
    assert "2✓" not in proc.stdout, (
        f"both tests ran — -k filter did not apply:\n{proc.stdout}"
    )


def test_explicit_double_dash_still_works(tmp_path: Path) -> None:
    """The legacy ``--`` separator keeps working alongside bare flags."""
    probe_dir = _make_probe_dir(tmp_path)
    proc = _run_runner(probe_dir, "-q", "--", "--tb=short")
    assert proc.returncode == 0, proc.stdout
    assert "unrecognized arguments" not in proc.stdout


def test_positional_path_not_treated_as_flag(tmp_path: Path) -> None:
    """A positional path arg still overrides discovery (not routed to pytest)."""
    probe_dir = _make_probe_dir(tmp_path)
    repo_root = Path(__file__).resolve().parent.parent
    runner = repo_root / "scripts" / "run_tests_parallel.py"
    # Pass the probe dir positionally (no --paths), plus a bare -q.
    proc = subprocess.run(
        [sys.executable, str(runner), str(probe_dir), "-j", "1",
         "--file-timeout", "30", "-q"],
        cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stdout
    # Discovery found the probe file (2 tests), proving the positional path
    # was consumed as a root, not forwarded to pytest as a bad flag.
    assert "test_flagprobe.py" in proc.stdout, proc.stdout


def test_file_retry_self_heals_and_prints_both_attempts(tmp_path: Path) -> None:
    """A pass-on-retry is green, loud, and retains the failing traceback."""
    repo_root = Path(__file__).resolve().parent.parent
    runner = repo_root / "scripts" / "run_tests_parallel.py"
    marker = tmp_path / "ran-once"
    probe = tmp_path / "test_flaky_probe.py"
    probe.write_text(
        textwrap.dedent(
            f"""
            from pathlib import Path

            def test_flaky_once():
                marker = Path({str(marker)!r})
                if not marker.exists():
                    marker.write_text("failed once")
                    assert False, "simulated first-attempt flake"
                assert True
            """
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--files",
            str(probe),
            "--file-retries",
            "1",
            "-j",
            "1",
            "-q",
        ],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout
    assert "FLAKY file" in proc.stdout
    assert "simulated first-attempt flake" in proc.stdout
    assert "first-attempt output" in proc.stdout
    assert "retry output" in proc.stdout


def test_file_retry_does_not_launder_deterministic_failure(tmp_path: Path) -> None:
    """A real regression fails both attempts and the runner remains red."""
    repo_root = Path(__file__).resolve().parent.parent
    runner = repo_root / "scripts" / "run_tests_parallel.py"
    probe = tmp_path / "test_red_probe.py"
    probe.write_text(
        "def test_always_red():\n    assert False, 'deterministic regression'\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--files",
            str(probe),
            "--file-retries",
            "1",
            "-j",
            "1",
            "-q",
        ],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 1, proc.stdout
    assert "deterministic regression" in proc.stdout
    assert "FLAKY file" not in proc.stdout


# ---------------------------------------------------------------------------
# Zero-collection is not a pass; node ids are translated, not dropped.
#
# Both behaviors were real foot-guns: a run where NOTHING was collected printed
# "0 tests passed, 0 failed (100% complete)" (reads green), and a pytest node id
# (`file.py::Class::test`) was silently discarded by path discovery so the run
# ended with "No test files to run" while looking like an accepted selector.


def test_zero_collected_across_run_fails_and_says_so(tmp_path: Path) -> None:
    """A -k that matches nothing must FAIL, not report a green summary."""
    probe_dir = _make_probe_dir(tmp_path)
    proc = _run_runner(probe_dir, "-k", "zzz_matches_nothing")
    assert proc.returncode == 1, proc.stdout
    assert "NO TESTS RAN" in proc.stdout
    assert "NOT a pass" in proc.stdout


def test_all_skipped_file_is_still_a_pass(tmp_path: Path) -> None:
    """Per-file zero-collection stays tolerated.

    A platform-gated file (every test skipped) reports "N skipped" — collected,
    just not executed — and must NOT trip the nothing-ran guard.
    """
    probe_dir = tmp_path / "skipprobe"
    probe_dir.mkdir()
    (probe_dir / "test_allskipped.py").write_text(
        "import pytest\n\n"
        "pytestmark = pytest.mark.skip(reason='platform-gated')\n\n"
        "def test_one():\n    assert True\n\n"
        "def test_two():\n    assert True\n"
    )
    proc = _run_runner(probe_dir)
    assert proc.returncode == 0, proc.stdout
    assert "NO TESTS RAN" not in proc.stdout


def test_node_id_selector_runs_the_named_test(tmp_path: Path) -> None:
    """``file.py::test_alpha`` runs that test instead of discovering nothing."""
    probe_dir = _make_probe_dir(tmp_path)
    target = probe_dir / "test_flagprobe.py"
    repo_root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "run_tests_parallel.py"),
         f"{target}::test_alpha", "-j", "1", "--file-timeout", "30"],
        cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stdout
    assert "No test files to run" not in proc.stdout
    assert "node id" in proc.stdout  # explains the translation
    # Ran exactly the one selected test, not both in the file.
    assert "1 tests passed" in proc.stdout


def test_explicit_k_wins_over_node_id_inference(tmp_path: Path) -> None:
    """A caller's own ``-k`` is not overridden by the node-id translation."""
    probe_dir = _make_probe_dir(tmp_path)
    target = probe_dir / "test_flagprobe.py"
    repo_root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "run_tests_parallel.py"),
         f"{target}::test_alpha", "-k", "test_beta",
         "-j", "1", "--file-timeout", "30"],
        cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=60,
    )
    # -k test_beta wins: one test ran, and it wasn't filtered to nothing.
    assert proc.returncode == 0, proc.stdout
    assert "1 tests passed" in proc.stdout


# ── Duration-aware timeout scaling, retry eligibility, cold-cache gate ───────
#
# These are fast, deterministic unit tests against the runner module's
# internals (loaded directly, not via subprocess) rather than integration
# tests over real multi-minute pytest files. Each loads its own fresh copy
# of the module so module-level state (``_FLAKY_RESULTS``) never leaks
# between tests.
#
# Context: a known-large test file (e.g. tests/test_hermes_state.py, ~435
# tests) passes in ~113-115s solo but can be SIGKILL'd under a cold-cache,
# multi-worker run before that — the flat --file-timeout cap has no notion
# of "this file is normally slow", and the automatic flake-retry then
# quietly reruns (and passes) a file that was never actually broken,
# manufacturing a green FLAKY result instead of surfacing the real risk.


def _load_runner_module():
    """Import scripts/run_tests_parallel.py as a fresh, isolated module."""
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / "scripts" / "run_tests_parallel.py"
    spec = importlib.util.spec_from_file_location(
        "run_tests_parallel_under_test", path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_effective_timeout_floor_when_uncached_or_zero() -> None:
    """No cache entry (or a 0.0 entry, which can't be a real run) keeps the flat cap."""
    mod = _load_runner_module()
    repo_root = Path(__file__).resolve().parent.parent
    f = repo_root / "tests" / "test_example_uncached.py"
    assert mod._effective_file_timeout(f, repo_root, 300.0, None) == 300.0
    assert mod._effective_file_timeout(f, repo_root, 300.0, {}) == 300.0
    zero_cached = {mod._format_file(f, repo_root): 0.0}
    assert mod._effective_file_timeout(f, repo_root, 300.0, zero_cached) == 300.0


def test_effective_timeout_headroom_and_floor() -> None:
    """3x a slow file's cached duration wins; a fast file's 3x stays under the cap."""
    mod = _load_runner_module()
    repo_root = Path(__file__).resolve().parent.parent
    slow = repo_root / "tests" / "test_example_slow.py"
    durations = {mod._format_file(slow, repo_root): 115.0}
    # 115s observed (matches the tests/test_hermes_state.py-style evidence)
    # -> 345s bound: headroom over the flat 300s cap.
    assert mod._effective_file_timeout(slow, repo_root, 300.0, durations) == 345.0

    fast = repo_root / "tests" / "test_example_fast.py"
    fast_durations = {mod._format_file(fast, repo_root): 4.0}
    # 4s * 3 = 12s, well under the flat cap -> the flat cap is the floor.
    assert mod._effective_file_timeout(fast, repo_root, 300.0, fast_durations) == 300.0


def test_clean_pass_durations_excludes_failed_and_flaky() -> None:
    """Only a first-attempt-clean duration feeds the cache.

    A timed-out (SIGKILL'd) or retry-healed (FLAKY) file must not inflate
    its own future timeout budget — see _clean_pass_durations.
    """
    mod = _load_runner_module()
    repo_root = Path(__file__).resolve().parent.parent
    clean = repo_root / "tests" / "test_example_clean.py"
    timed_out = repo_root / "tests" / "test_example_timed_out.py"
    flaky_file = repo_root / "tests" / "test_example_flaky.py"
    file_times = [(clean, 12.0), (timed_out, 300.4), (flaky_file, 250.0)]
    failures = [(timed_out, "(300s exceeded; process tree SIGKILL'd)", {})]
    flaky = [(flaky_file, "⚠ FLAKY: failed on attempt 1, passed on retry")]

    kept = mod._clean_pass_durations(file_times, failures, flaky)

    assert kept == [(clean, 12.0)]


def test_timeout_rc_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """rc 124 (this runner's SIGKILL-on-timeout convention) is never retried."""
    mod = _load_runner_module()
    calls: list[float] = []

    def fake_once(file, pytest_args, repo_root, file_timeout):
        calls.append(file_timeout)
        return file, 124, "(300s exceeded; process tree SIGKILL'd)", {}, 300.0

    monkeypatch.setattr(mod, "_run_one_file_once", fake_once)
    _file, rc, _output, _summary, _wall = mod._run_one_file(
        Path("tests/test_would_be_slow.py"), [], Path("."), 300.0, retries=1
    )

    assert rc == 124
    assert len(calls) == 1, "timed-out file must not be retried"


def test_signal_killed_rc_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """A negative (signal-killed) rc is never retried either."""
    mod = _load_runner_module()
    calls: list[int] = []

    def fake_once(file, pytest_args, repo_root, file_timeout):
        calls.append(1)
        return file, -9, "killed by signal 9", {}, 5.0

    monkeypatch.setattr(mod, "_run_one_file_once", fake_once)
    _file, rc, _output, _summary, _wall = mod._run_one_file(
        Path("tests/test_would_be_oom_killed.py"), [], Path("."), 300.0, retries=1
    )

    assert rc == -9
    assert len(calls) == 1, "signal-killed file must not be retried"


def test_ordinary_failure_still_retries_and_reports_flaky(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An everyday nonzero pytest exit keeps the existing retry + FLAKY behavior."""
    mod = _load_runner_module()
    calls: list[int] = []

    def fake_once(file, pytest_args, repo_root, file_timeout):
        calls.append(1)
        if len(calls) == 1:
            return file, 1, "assertion failed", {"failed": 1}, 1.0
        return file, 0, "passed on retry", {"passed": 1}, 1.0

    monkeypatch.setattr(mod, "_run_one_file_once", fake_once)
    _file, rc, output, _summary, _wall = mod._run_one_file(
        Path("tests/test_would_be_flaky.py"), [], Path("."), 300.0, retries=1
    )

    assert rc == 0
    assert len(calls) == 2, "ordinary nonzero exits remain retry-eligible"
    assert "FLAKY" in output


def test_is_large_file_threshold() -> None:
    mod = _load_runner_module()
    counts = {
        Path("a.py"): mod._LARGE_FILE_TEST_COUNT_THRESHOLD - 1,
        Path("b.py"): mod._LARGE_FILE_TEST_COUNT_THRESHOLD,
        Path("c.py"): mod._LARGE_FILE_TEST_COUNT_THRESHOLD + 100,
    }
    assert mod._is_large_file(Path("a.py"), counts) is False
    assert mod._is_large_file(Path("b.py"), counts) is True
    assert mod._is_large_file(Path("c.py"), counts) is True
    assert mod._is_large_file(Path("missing.py"), counts) is False


def test_large_file_gate_serializes_large_files_leaves_normal_files_ungated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At most one large file runs at a time; normal (ungated) files never wait on it.

    Uses threading.Event handoffs (not sleeps) so the ordering is asserted
    deterministically: the 2nd/3rd large file must NOT have started while
    the 1st still holds the gate, and normal files complete immediately
    regardless of the gate's state. The "second/third large file hasn't
    started yet" check itself is a handoff on a second Event set by a
    watcher thread that blocks on the semaphore, rather than a sleep-and-hope
    window — see ``blocked_on_gate`` below.
    """
    mod = _load_runner_module()
    gate = threading.Semaphore(1)

    large_files = [Path(f"tests/test_large_{i}.py") for i in range(3)]
    normal_files = [Path(f"tests/test_normal_{i}.py") for i in range(2)]

    concurrent_large = 0
    max_concurrent_large = 0
    lock = threading.Lock()
    entered = {f: threading.Event() for f in large_files}
    release = {f: threading.Event() for f in large_files}

    def fake_run_one_file(file, pytest_args, repo_root, file_timeout, retries):
        nonlocal concurrent_large, max_concurrent_large
        if file not in entered:
            # Normal file: no synchronization — must return immediately.
            return file, 0, "", {}, 0.0
        with lock:
            concurrent_large += 1
            max_concurrent_large = max(max_concurrent_large, concurrent_large)
        entered[file].set()
        held = release[file].wait(timeout=5)
        with lock:
            concurrent_large -= 1
        assert held, f"{file} was never released — test deadlocked"
        return file, 0, "", {}, 0.0

    monkeypatch.setattr(mod, "_run_one_file", fake_run_one_file)

    # No thread involved: a non-blocking acquire on the SAME semaphore
    # either succeeds immediately (capacity available) or fails immediately
    # (fully held), so this is deterministic with no sleep or wait needed.
    # A successful acquire is released right away so it doesn't itself
    # consume the capacity we're trying to observe.
    def gate_is_exhausted() -> bool:
        acquired = gate.acquire(blocking=False)
        if acquired:
            gate.release()
            return False
        return True

    with ThreadPoolExecutor(max_workers=len(large_files) + len(normal_files)) as pool:
        large_futures = [
            pool.submit(
                mod._run_one_file_gated, f, [], Path("."), 30.0, 0, gate
            )
            for f in large_files
        ]

        assert entered[large_files[0]].wait(timeout=2), "first large file never started"
        # Deterministic, no sleep: the semaphore (capacity 1) is held by
        # large_files[0], so a non-blocking acquire attempt must fail —
        # proving no other large file could be inside the gate right now.
        assert gate_is_exhausted(), "gate was not held while first large file ran"
        assert not entered[large_files[1]].is_set(), (
            "second large file started while the gate was held"
        )
        assert not entered[large_files[2]].is_set(), (
            "third large file started while the gate was held"
        )

        # Normal files are ungated: they must complete promptly even while
        # a large file is still holding the gate open.
        normal_futures = [
            pool.submit(
                mod._run_one_file_gated, f, [], Path("."), 30.0, 0, None
            )
            for f in normal_files
        ]
        for fut in normal_futures:
            fut.result(timeout=2)

        release[large_files[0]].set()
        assert entered[large_files[1]].wait(timeout=2)
        release[large_files[1]].set()
        assert entered[large_files[2]].wait(timeout=2)
        release[large_files[2]].set()

        for fut in large_futures:
            fut.result(timeout=5)

    assert max_concurrent_large == 1


def test_submission_order_normal_files_before_large_files() -> None:
    """Normal files precede statically-large files; relative order within
    each group is preserved (see _submission_order).

    Regression coverage for the controller finding: with FIFO
    ThreadPoolExecutor submission, a gated large file submitted ahead of a
    normal file could occupy a worker thread waiting on the semaphore
    before the normal file's future is even queued — silently stealing a
    worker slot from ordinary work despite the semaphore's mutual
    exclusion being correct on its own.
    """
    mod = _load_runner_module()
    threshold = mod._LARGE_FILE_TEST_COUNT_THRESHOLD

    small = Path("tests/test_small.py")
    normal_a = Path("tests/test_normal_a.py")
    normal_b = Path("tests/test_normal_b.py")
    large_a = Path("tests/test_large_a.py")
    large_b = Path("tests/test_large_b.py")

    test_counts = {
        large_a: threshold,
        normal_a: threshold - 1,
        large_b: threshold + 50,
        normal_b: 3,
        small: 0,
    }

    # Discovery/slice order deliberately interleaves large and normal so
    # the helper's reordering (not accidental input order) is what's
    # under test.
    files = [large_a, normal_a, large_b, normal_b, small]

    ordered = mod._submission_order(files, test_counts)

    assert ordered == [normal_a, normal_b, small, large_a, large_b]
    # Every normal file's index precedes every large file's index.
    normal_set = {normal_a, normal_b, small}
    large_set = {large_a, large_b}
    last_normal_idx = max(ordered.index(f) for f in normal_set)
    first_large_idx = min(ordered.index(f) for f in large_set)
    assert last_normal_idx < first_large_idx
