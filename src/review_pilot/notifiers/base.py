from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class NotificationMessage:
    channel: str
    payload: dict[str, Any]
    summary: str


@dataclass(frozen=True)
class NotificationResult:
    channel: str
    mode: str
    delivered: bool
    payload: dict[str, Any]
    response: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "mode": self.mode,
            "delivered": self.delivered,
            "payload": self.payload,
            "response": self.response,
        }


class Notifier(Protocol):
    def notify(
        self,
        message: NotificationMessage,
        *,
        dry_run: bool = False,
    ) -> NotificationResult:
        raise NotImplementedError
