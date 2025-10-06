"""Parser for Cursor Agent CLI JSON output."""

from __future__ import annotations

import json
from typing import Any

from .base import BaseParser, ParsedCLIResponse, ParserError


class CursorAgentJSONParser(BaseParser):
    """Parse stdout produced by `cursor-agent --output-format json`."""

    name = "cursor_json"

    def parse(self, stdout: str, stderr: str) -> ParsedCLIResponse:
        if not stdout.strip():
            raise ParserError("Cursor Agent CLI returned empty stdout while JSON output was expected")

        try:
            payload: dict[str, Any] = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise ParserError(f"Failed to decode Cursor Agent CLI JSON output: {exc}") from exc

        # Cursor Agent returns content in "result" field, not "response"
        result = payload.get("result")
        if not isinstance(result, str) or not result.strip():
            raise ParserError("Cursor Agent CLI response is missing a textual 'result' field")

        metadata: dict[str, Any] = {"raw": payload}

        # Extract cursor-agent specific metadata
        if "type" in payload:
            metadata["type"] = payload["type"]
        if "subtype" in payload:
            metadata["subtype"] = payload["subtype"]
        if "is_error" in payload:
            metadata["is_error"] = payload["is_error"]
        if "duration_ms" in payload:
            metadata["duration_ms"] = payload["duration_ms"]
        if "duration_api_ms" in payload:
            metadata["duration_api_ms"] = payload["duration_api_ms"]
        if "session_id" in payload:
            metadata["session_id"] = payload["session_id"]
        if "request_id" in payload:
            metadata["request_id"] = payload["request_id"]

        # Extract model information if available
        model_used = payload.get("model")
        if model_used:
            metadata["model_used"] = model_used

        # Extract token usage if available
        if "usage" in payload and isinstance(payload["usage"], dict):
            metadata["token_usage"] = payload["usage"]

        if stderr and stderr.strip():
            metadata["stderr"] = stderr.strip()

        return ParsedCLIResponse(content=result.strip(), metadata=metadata)
