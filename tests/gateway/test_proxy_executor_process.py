"""Real two-OS-process required-proxy admission test with seeded executor continuity.

``test_proxy_memory_identity.py`` and ``test_proxy_background.py`` host the client dispatch AND
the native ``_create_agent`` receiver in the SAME process, so they exercise real in-process
wire/media/profile-identity forwarding but cannot exercise the required-proxy admission POLICY
itself -- a single pinned process root can't be both required and not-required at once. This
module is process-policy integration, not filesystem/OS isolation: it spawns the native receiver
as a genuinely separate OS process (``tests/gateway/proxy_executor_fixture.py``, real
``subprocess``, no invented IPC broker) whose own root ``config.yaml`` declares
``gateway.proxy_required: false``, while THIS test process pins its own root
``gateway.proxy_required: true``. Only the final model/provider resolution and the child's
``AIAgent`` construction/execution are a labelled synthetic seam (reusing the child fixture's
recording agent); the real ``GatewayRunner`` proxy dispatch (foreground A/B/A + a background image
handler), authenticated per-profile API keys, and the real native ``/p/<profile>/`` receiver all
run unmodified across the two processes. The fixture also seeds the native owner-A SessionDB
before child startup; that proves executor continuity, not request-history forwarding/import.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

import hermes_constants
from agent import secret_scope as ss
from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.proxy_admission import gateway_proxy_required
from gateway.run import _profile_runtime_scope
from gateway.session import build_session_key
from hermes_state import SessionDB

from tests.gateway.test_proxy_background import _RecordingAdapter, _TINY_PNG, _make_background_runner
from tests.gateway.test_proxy_memory_identity import _client_runner, _profile_home, _source

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHILD_STARTUP_TIMEOUT = 20.0
_CHILD_TERMINATE_TIMEOUT = 10.0


@pytest.fixture(autouse=True)
def _reset_multiplex():
    ss.set_multiplex_active(False)
    hermes_constants.pin_process_hermes_home(None)
    yield
    ss.set_multiplex_active(False)
    hermes_constants.pin_process_hermes_home(None)


def _spawn_child(
    child_root: Path, evidence_path: Path, owner_a_home: Path, owner_b_home: Path, stderr_path: Path,
) -> subprocess.Popen:
    """Real ``subprocess.Popen``, runner-sanitized env (never ``os.environ.copy()``): only what
    the interpreter and imports need, no inherited credentials. Stderr goes to a file, never a
    second pipe this test doesn't drain (would risk a full-buffer deadlock)."""
    env = {
        "PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(_REPO_ROOT), "TZ": "UTC", "LANG": "C.UTF-8",
        "HERMES_HOME": str(child_root), "HOME": str(child_root),
    }
    with open(stderr_path, "wb") as stderr_handle:
        return subprocess.Popen(
            [sys.executable, "-m", "tests.gateway.proxy_executor_fixture",
             str(child_root), str(evidence_path), str(owner_a_home), str(owner_b_home)],
            cwd=str(_REPO_ROOT), env=env, stdout=subprocess.PIPE, stderr=stderr_handle, text=True)


async def _await_child_ready(proc: subprocess.Popen, stderr_path: Path) -> int:
    """Bounded wait on the child's one readiness line -- never an unbounded blocking read."""
    line = await asyncio.wait_for(asyncio.to_thread(proc.stdout.readline), timeout=_CHILD_STARTUP_TIMEOUT)
    assert line.startswith("READY "), (
        f"child fixture did not report readiness: {line!r}; stderr: "
        f"{stderr_path.read_text(encoding='utf-8', errors='replace')}")
    return int(line.split()[1])


def _read_evidence(evidence_path: Path) -> List[Dict[str, Any]]:
    if not evidence_path.exists():
        return []
    lines = [line for line in evidence_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [json.loads(line) for line in lines]


@pytest.mark.asyncio
async def test_required_proxy_admission_real_child_process(tmp_path, monkeypatch):
    """Genuinely separate OS processes with seeded continuity: the CHILD's own root policy is
    false (native execution
    permitted there) while the PARENT (this test process) pins its own root required -- proving
    the admission gate reads each process's OWN identity, never the other's, and that the
    required-proxy parent exclusively uses the configured proxy for both a foreground A/B/A and a
    background image dispatch, never falling back to local construction. The owner-A executor
    transcript is explicitly seeded in its native SessionDB before startup."""
    evidence_path = tmp_path / "evidence.jsonl"

    key_owner_a, key_owner_b = "child-owner-a-key", "child-owner-b-key"
    home_owner_a = _profile_home(tmp_path, "owner-a", extra_env={"API_SERVER_KEY": key_owner_a})
    home_owner_b = _profile_home(tmp_path, "owner-b", extra_env={"API_SERVER_KEY": key_owner_b})

    child_root = _profile_home(tmp_path, "child-root")
    (child_root / "config.yaml").write_text(
        yaml.safe_dump({"gateway": {"proxy_required": False}}), encoding="utf-8")

    authored_a1_history = [
        {"role": "user" if index % 2 == 0 else "assistant",
         "content": f"A1 history message {index:02d}"}
        for index in range(20)
    ]
    # Seed native continuity, not a bootstrap import: the authenticated Session-ID is owned by SessionDB.
    owner_a_db = SessionDB(db_path=home_owner_a / "state.db")
    try:
        owner_a_db.create_session("session-fixture", "api_server", profile_name="owner-a")
        for message in authored_a1_history:
            owner_a_db.append_message("session-fixture", message["role"], message["content"])
        expected_persisted_history = owner_a_db.get_messages_as_conversation("session-fixture")
    finally:
        owner_a_db.close()

    stderr_path = tmp_path / "child-stderr.log"
    proc = _spawn_child(child_root, evidence_path, home_owner_a, home_owner_b, stderr_path)
    try:
        port = await _await_child_ready(proc, stderr_path)
        base_url = f"http://127.0.0.1:{port}"

        home_client_a = _profile_home(
            tmp_path, "client-a",
            extra_env={"GATEWAY_PROXY_URL": f"{base_url}/p/owner-a", "GATEWAY_PROXY_KEY": key_owner_a})
        for image_home in (home_owner_a, home_client_a):
            (image_home / "config.yaml").write_text(
                yaml.safe_dump({"agent": {"image_input_mode": "text"}}), encoding="utf-8")
        home_client_b = _profile_home(
            tmp_path, "client-b",
            extra_env={"GATEWAY_PROXY_URL": f"{base_url}/p/owner-b", "GATEWAY_PROXY_KEY": key_owner_b})
        home_client_a_wrongkey = _profile_home(
            tmp_path, "client-a-wrongkey",
            extra_env={"GATEWAY_PROXY_URL": f"{base_url}/p/owner-a", "GATEWAY_PROXY_KEY": "wrong-key"})

        # The PARENT's own launch config (pinned, real root config.yaml): required=true.
        parent_root = _profile_home(tmp_path, "parent-root")
        (parent_root / "config.yaml").write_text(
            yaml.safe_dump({
                "gateway": {"proxy_required": True},
                "compression": {"enabled": True, "hygiene_hard_message_limit": 10},
                "agent": {"image_input_mode": "text"},
            }), encoding="utf-8")
        hermes_constants.pin_process_hermes_home(str(parent_root))

        profiles = {
            "client-a": home_client_a, "client-b": home_client_b,
            "client-a-wrongkey": home_client_a_wrongkey,
        }
        monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: profiles[name])
        monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name in profiles)
        monkeypatch.setattr("hermes_cli.profiles.profiles_to_serve", lambda multiplex: list(profiles.items()))
        ss.set_multiplex_active(True)

        # Real per-process policy reader, exercised directly on the PARENT side: true.
        assert gateway_proxy_required() is True

        parent_constructed: List[Any] = []
        parent_runtime_calls: List[Any] = []
        parent_vision_calls: List[Any] = []
        import run_agent
        from tools import vision_tools
        monkeypatch.setattr(run_agent, "AIAgent", lambda **kw: parent_constructed.append(kw) or None)
        monkeypatch.setattr(
            "gateway.run._resolve_runtime_agent_kwargs",
            lambda: (parent_runtime_calls.append(True), {})[1])

        async def forbidden_parent_vision(*args, **kwargs):
            parent_vision_calls.append((args, kwargs))
            raise AssertionError("required-proxy image admission must not run parent vision")

        monkeypatch.setattr(vision_tools, "vision_analyze_tool", forbidden_parent_vision)

        client_runner = _client_runner()
        source = _source()
        key_a = build_session_key(source, profile="owner-a")
        key_b = build_session_key(source, profile="owner-b")

        source.profile = "client-a"
        # Keep this as a real hygiene call under the owning client profile. Required-proxy
        # admission must preserve the exact payload rather than locally bounding it.
        a1_history_bytes = json.dumps(authored_a1_history, ensure_ascii=False, separators=(",", ":")).encode()
        with _profile_runtime_scope(home_client_a):
            hygienic_a1_history = await client_runner._hmwa_run_session_hygiene(
                SimpleNamespace(source=source), source,
                SimpleNamespace(session_id="session-fixture"), key_a,
                authored_a1_history, "quick-key", 1,
            )
        assert hygienic_a1_history is authored_a1_history
        assert json.dumps(hygienic_a1_history, ensure_ascii=False, separators=(",", ":")).encode() == a1_history_bytes
        result_a1 = await client_runner._run_agent(
            "current A1", "fixture system", hygienic_a1_history, source, "session-fixture", session_key=key_a)
        source.profile = "client-b"
        result_b = await client_runner._run_agent(
            "current B", "fixture system", [], source, "session-fixture", session_key=key_b)
        source.profile = "client-a"
        result_a2 = await client_runner._run_agent(
            "current A2", "fixture system", [], source, "session-fixture-next", session_key=key_a)

        assert [result["final_response"] for result in (result_a1, result_b, result_a2)] == [
            f"echo:{key_a}:current A1", f"echo:{key_b}:current B", f"echo:{key_a}:current A2",
        ]

        source.profile = "client-a-wrongkey"
        wrong_auth_result = await client_runner._run_agent(
            "must reject", "fixture system", [], source, "session-fixture", session_key=key_a)
        assert "401" in wrong_auth_result["final_response"]
        assert len(_read_evidence(evidence_path)) == 3, "wrong auth must be refused before any construction"

        # -- foreground image admission: preparation itself must cross the required proxy --
        image_path = home_client_a / "cache" / "images" / "img_native_test.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(_TINY_PNG)
        expected_data_url = f"data:image/png;base64,{base64.b64encode(_TINY_PNG).decode('ascii')}"

        source.profile = "client-a"
        image_event = MessageEvent(
            text="describe this image", source=source, message_id="foreground-image",
            media_urls=[str(image_path)], media_types=["image/png"],
        )
        prepared_image_text = await client_runner._prepare_profile_scoped_inbound_message_text(
            event=image_event, source=source, history=[], session_key=key_a,
        )
        assert prepared_image_text is not None
        image_result = await client_runner._run_agent(
            prepared_image_text, "fixture system", [], source, "fixture-image", session_key=key_a)
        assert image_result["final_response"] == f"echo:{key_a}:image:fixture-image"
        assert image_result["completed"] is True
        assert image_result["final_response"] != f"echo:{key_a}:{prepared_image_text}"
        assert len(_read_evidence(evidence_path)) == 4

        # -- background dispatch with an attached image, separate (None) memory identity --

        bg_runner = _make_background_runner()
        bg_runner.config = GatewayConfig(multiplex_profiles=True)
        bg_adapter = _RecordingAdapter(platform=Platform.MATRIX)
        bg_runner.adapters[Platform.MATRIX] = bg_adapter
        bg_runner._profile_adapters["client-a"] = {Platform.MATRIX: bg_adapter}
        bg_source = _source()
        bg_source.profile = "client-a"

        generated_background_ids: List[str] = []
        original_run_background_task = bg_runner._run_background_task

        async def record_background_id(prompt, source, task_id, *args, **kwargs):
            generated_background_ids.append(task_id)
            return await original_run_background_task(prompt, source, task_id, *args, **kwargs)

        bg_runner._run_background_task = record_background_id

        started = await bg_runner._handle_background_command(MessageEvent(
            text="/bg describe this image", source=bg_source, message_id="bg-command-image",
            media_urls=[str(image_path)], media_types=["image/png"],
        ))
        assert started
        tasks = tuple(bg_runner._background_tasks)
        assert tasks, "the real /bg handler must track the spawned task"
        await asyncio.gather(*tasks)
        assert bg_adapter.sent and "✅ Background task complete" in bg_adapter.sent[0]

        # -- evidence written by the CHILD process (genuinely separate pid) --
        records = _read_evidence(evidence_path)
        assert len(records) == 5, "3 foreground A/B/A + 1 foreground + 1 background image dispatch"
        agent_a1, agent_b, agent_a2, agent_image, agent_bg = records

        authored_at_final_model = [
            message for message in agent_a1["conversation_history"]
            if message.get("role") in {"user", "assistant"}
        ]
        assert agent_a1["conversation_history"] == expected_persisted_history
        assert [
            {"role": message["role"], "content": message["content"]}
            for message in authored_at_final_model
        ] == authored_a1_history, (
            "the final model seam must receive every authored history message in order and in full; "
            "native persisted fields are compared above"
        )

        child_pid = agent_a1["pid"]
        assert child_pid == proc.pid
        assert child_pid != os.getpid(), "child evidence must come from a genuinely separate OS process"
        assert {r["pid"] for r in records} == {child_pid}
        assert all(r["get_process_hermes_home"] == str(child_root) for r in records)

        assert [r["session_id"] for r in (agent_a1, agent_b, agent_a2)] == [
            "session-fixture", "session-fixture", "session-fixture-next",
        ]
        assert agent_image["session_id"] == "fixture-image"
        assert agent_image["gateway_session_key"] == key_a
        assert agent_image["home"] == str(home_owner_a)
        assert agent_image["pid"] == child_pid
        assert agent_image["session_db_id"] == agent_a1["session_db_id"]
        assert agent_image["content_kind"] == "image"
        assert agent_image["data_urls"] == [expected_data_url]
        assert str(image_path) not in json.dumps(agent_image, ensure_ascii=False)
        assert len(generated_background_ids) == 1
        assert agent_bg["session_id"] == agent_bg["task_id"] == generated_background_ids[0]
        assert agent_bg["session_id"] not in {agent_a1["session_id"], agent_b["session_id"], agent_a2["session_id"]}
        assert agent_a1["home"] == agent_a2["home"] == str(home_owner_a)
        assert agent_b["home"] == str(home_owner_b)
        assert agent_a1["home"] != agent_b["home"]
        assert agent_a1["session_db_id"] == agent_a2["session_db_id"]
        assert agent_a1["session_db_id"] != agent_b["session_db_id"]
        assert [r["gateway_session_key"] for r in (agent_a1, agent_b, agent_a2)] == [key_a, key_b, key_a]

        # -- the real per-request admission recheck on the CHILD's own process root: false --
        assert all(r["own_proxy_required"] is False for r in records)

        # -- background dispatch: separate (no foreground) memory identity, exact native
        #    content-part data URI reaches the child, never the local file path --
        assert agent_bg["gateway_session_key"] is None
        assert agent_bg["content_kind"] == "image"
        assert agent_bg["data_urls"] == [expected_data_url]
        assert agent_bg["home"] == str(home_owner_a)

        # -- wrong auth never reached agent construction on the child: it was still exactly 3 records --
        assert len(_read_evidence(evidence_path)) == 5

        # -- the PARENT (required=true) never constructs a native agent or resolves local
        #    runtime kwargs: the configured proxy is exclusively used --
        assert parent_constructed == []
        assert parent_runtime_calls == []
        assert parent_vision_calls == []
    finally:
        proc.terminate()
        try:
            await asyncio.wait_for(asyncio.to_thread(proc.wait), timeout=_CHILD_TERMINATE_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            await asyncio.wait_for(asyncio.to_thread(proc.wait), timeout=_CHILD_TERMINATE_TIMEOUT)
        finally:
            assert proc.stdout is not None
            proc.stdout.close()
        assert proc.poll() is not None, "the child process must be reaped, never left as a zombie/orphan"
