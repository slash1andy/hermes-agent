from __future__ import annotations

import threading

from cron import delivery_queue
from cron.scheduler_provider import _profile_cron_scope


def test_native_active_delivery_alias_is_profile_local(tmp_path):
    home_a = tmp_path / "home-a"
    home_b = tmp_path / "home-b"
    home_a.mkdir()
    home_b.mkdir()
    execution_id = "shared-execution-id"
    sends: list[tuple[str, str]] = []
    entered_b = threading.Event()
    release_b = threading.Event()
    worker_errors: list[BaseException] = []

    with _profile_cron_scope(home_a):
        delivery_queue.enqueue(execution_id, {"profile": "a"}, "content-a")
    with _profile_cron_scope(home_b):
        delivery_queue.enqueue(execution_id, {"profile": "b"}, "content-b")

    def send_b(job, content, _for_failure):
        sends.append((job["profile"], content))
        entered_b.set()
        release_b.wait()
        return None

    def drain_b():
        try:
            with _profile_cron_scope(home_b):
                assert delivery_queue.drain(send_b) == 1
        except BaseException as exc:
            worker_errors.append(exc)

    worker = threading.Thread(target=drain_b)
    worker.start()
    try:
        assert entered_b.wait(timeout=5)

        with _profile_cron_scope(home_a):
            assert delivery_queue.drain(
                lambda job, content, _for_failure: sends.append(
                    (job["profile"], content)
                )
                or None
            ) == 1

        with _profile_cron_scope(home_b):
            assert delivery_queue.recover_abandoned() == 0
            status_b = delivery_queue.get_status(execution_id)
            assert status_b is not None
            assert status_b["status"] == "delivering"
    finally:
        release_b.set()
        worker.join(timeout=5)

    assert not worker.is_alive()
    assert worker_errors == []
    with _profile_cron_scope(home_a):
        status_a = delivery_queue.get_status(execution_id)
        assert status_a is not None
        assert status_a["status"] == "delivered"
    with _profile_cron_scope(home_b):
        status_b = delivery_queue.get_status(execution_id)
        assert status_b is not None
        assert status_b["status"] == "delivered"
    assert sorted(sends) == [("a", "content-a"), ("b", "content-b")]
