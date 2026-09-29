"""Child-process entry point for the real two-process required-proxy admission test.

Run as ``python -m tests.gateway.proxy_executor_fixture <child_root> <evidence_path>
<owner_a_home> <owner_b_home>``. Starts a genuine ``APIServerAdapter`` aiohttp listener in a
SEPARATE OS process from the pytest test process, serving ``owner-a``/``owner-b`` through the
real ``/p/<profile>/`` prefix middleware (reusing
``tests.gateway.test_proxy_memory_identity._start_shared_native_server`` -- the exact mechanism
``connect()`` wires). This process's own root ``config.yaml`` under ``child_root`` is the process
this test proves stays UNREQUIRED (``gateway.proxy_required: false``), independent of whatever
policy the parent test process pins for itself.

Only the final model/provider resolution and the ``AIAgent`` constructor/execution are a labelled
synthetic seam (mirrors ``test_proxy_memory_identity._SyntheticAIAgent`` /
``_patch_deterministic_provider_resolver``, reimplemented as direct attribute patches since this
module runs with no pytest ``monkeypatch``); auth, the real profile-prefix middleware, home
scoping, and the real ``_create_agent`` body all run unmodified.

Every constructed agent appends one JSON evidence record to ``evidence_path`` (one JSON object per
line under a lock, so concurrent requests can't corrupt each other's record): the observed
model inputs, the owning home, session id, gateway session key, this process's own pid, the
SessionDB instance id, and this process's own real ``gateway_proxy_required()`` reading.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Any, Dict

_EVIDENCE_LOCK = threading.Lock()


def _append_evidence(evidence_path: Path, record: Dict[str, Any]) -> None:
    with _EVIDENCE_LOCK:
        with open(evidence_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _install_synthetic_seam(evidence_path: Path) -> None:
    """Patch ONLY the final model/provider resolution and ``AIAgent`` construction/execution;
    auth, profile-prefix middleware and the real ``_create_agent`` body stay untouched."""
    import run_agent
    import gateway.run as gateway_run
    import hermes_cli.tools_config as tools_config
    from hermes_constants import get_hermes_home, get_process_hermes_home
    from gateway.proxy_admission import gateway_proxy_required

    class _ChildRecordingAIAgent:
        def __init__(self, **kwargs):
            self.session_id = kwargs.get("session_id")
            self.platform = kwargs.get("platform")
            self.gateway_session_key = kwargs.get("gateway_session_key")
            self.session_db = kwargs.get("session_db")
            self._stream_delta_callback = kwargs.get("stream_delta_callback")
            self._base_record = {
                "pid": os.getpid(),
                "home": str(get_hermes_home()),
                "get_process_hermes_home": str(get_process_hermes_home()),
                "session_id": self.session_id,
                "gateway_session_key": self.gateway_session_key,
                "session_db_id": id(self.session_db),
                "platform": self.platform,
                "provider": kwargs.get("provider"),
                "base_url": kwargs.get("base_url"),
                "api_mode": kwargs.get("api_mode"),
                # The real per-checkpoint recheck this test proves stays FALSE on this process.
                "own_proxy_required": gateway_proxy_required(),
            }

        def run_conversation(self, *, user_message, conversation_history, task_id, **_kwargs):
            content_kind = "image" if isinstance(user_message, list) else "text"
            data_urls = [
                part["image_url"]["url"] for part in user_message
                if isinstance(part, dict) and part.get("type") == "image_url"
            ] if isinstance(user_message, list) else []
            record = dict(
                self._base_record, content_kind=content_kind, data_urls=data_urls, task_id=task_id,
                user_message=user_message if isinstance(user_message, str) else None,
                conversation_history=conversation_history,
            )
            _append_evidence(evidence_path, record)
            text = f"echo:{self.gateway_session_key or ''}:{user_message}" if content_kind == "text" \
                else f"echo:{self.gateway_session_key or ''}:image:{task_id}"
            if self._stream_delta_callback:
                self._stream_delta_callback(text)
            return {"final_response": text, "messages": [], "api_calls": 1, "tools": [], "completed": True}

    run_agent.AIAgent = lambda **kwargs: _ChildRecordingAIAgent(**kwargs)
    gateway_run._resolve_runtime_agent_kwargs = lambda: {
        "provider": "synthetic-child-provider", "base_url": "https://example.test/v1", "api_mode": "chat"}
    gateway_run._resolve_gateway_model = lambda user_config=None: "hermes-agent"
    gateway_run.GatewayRunner._load_reasoning_config = staticmethod(lambda model="": None)
    gateway_run.GatewayRunner._load_fallback_model = staticmethod(lambda: None)
    tools_config._get_platform_tools = lambda *_a, **_k: set()


async def _run_server(child_root: Path, owner_homes: Dict[str, Path], evidence_path: Path) -> None:
    import hermes_constants
    hermes_constants.pin_process_hermes_home(str(child_root))
    _install_synthetic_seam(evidence_path)

    import hermes_cli.profiles as profiles_mod
    from agent import secret_scope as ss
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from tests.gateway.test_proxy_memory_identity import _start_shared_native_server

    profiles_mod.get_profile_dir = lambda name: owner_homes[name]
    profiles_mod.profile_exists = lambda name: name in owner_homes
    profiles_mod.profiles_to_serve = lambda multiplex: list(owner_homes.items())
    ss.set_multiplex_active(True)

    server_runner = object.__new__(GatewayRunner)
    server_runner.config = GatewayConfig(multiplex_profiles=True)
    server_runner._primary_profile_name = "default"
    app_runner, adapter, base_url = await _start_shared_native_server(server_runner)

    # Bounded startup line: the parent reads exactly this one line to learn the real listening
    # port before issuing any request.
    port = base_url.rsplit(":", 1)[1]
    print(f"READY {port}", flush=True)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _handle_signal(*_a) -> None:
        loop.call_soon_threadsafe(stop_event.set)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _handle_signal)

    try:
        await stop_event.wait()
    finally:
        await app_runner.cleanup()
        await adapter.disconnect()


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    child_root, evidence_path, owner_a_home, owner_b_home = (Path(a) for a in argv[:4])
    owner_homes = {"owner-a": owner_a_home, "owner-b": owner_b_home}
    asyncio.run(_run_server(child_root, owner_homes, evidence_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
