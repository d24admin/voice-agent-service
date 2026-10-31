from __future__ import annotations

import asyncio

from app.conversation.call_session import CallSession


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, CallSession] = {}

    def register(self, session: CallSession) -> None:
        if session.session_id in self._sessions:
            raise ValueError(f"session already registered: {session.session_id}")
        self._sessions[session.session_id] = session

    def unregister(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    async def close_all(self, timeout_s: float) -> None:
        sessions = tuple(self._sessions.values())
        if not sessions:
            return
        try:
            async with asyncio.timeout(timeout_s):
                await asyncio.gather(
                    *(session.request_shutdown() for session in sessions)
                )
                await asyncio.gather(*(session.wait_closed() for session in sessions))
        except TimeoutError:
            for session in sessions:
                session.force_cancel()
        finally:
            self._sessions.clear()
