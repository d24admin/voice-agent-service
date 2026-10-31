from __future__ import annotations

import re


class SentenceBuffer:
    def __init__(self) -> None:
        self._buffer = ""

    def push(self, token: str) -> list[str]:
        normalized = " ".join(token.split())
        if not normalized:
            return []

        if not self._buffer:
            self._buffer = normalized
        elif normalized[0] in ".?!,;:":
            self._buffer += normalized
        else:
            self._buffer += f" {normalized}"

        sentences: list[str] = []
        while match := re.search(r"[.?!]", self._buffer):
            boundary = match.end()
            sentence = self._buffer[:boundary].strip()
            self._buffer = self._buffer[boundary:].strip()
            if sentence:
                sentences.append(sentence)
        return sentences

    def flush(self) -> str | None:
        trailing = self._buffer.strip()
        self._buffer = ""
        return trailing or None
