from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class MqttSettings:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 1883
    username: str | None = None
    password: str | None = None
    client_id: str | None = None
    topic_prefix: str = "eleccia"
    qos: int = 0
    retain: bool = False


class ElecciaMqttService:
    def __init__(self, settings: MqttSettings) -> None:
        self._settings = settings
        self._client: Any | None = None
        self._running = False
        self._last_error: str | None = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def start(self) -> None:
        if self._running:
            return
        if not self._settings.enabled:
            self._last_error = None
            return

        try:
            import paho.mqtt.client as mqtt

            protocol = getattr(mqtt, "MQTTv311", None)
            client = mqtt.Client(client_id=self._client_id(), protocol=protocol)
            if self._settings.username:
                client.username_pw_set(
                    username=self._settings.username,
                    password=self._settings.password,
                )
            client.connect(self._settings.host, int(self._settings.port), keepalive=60)
            client.loop_start()
            self._client = client
            self._running = True
            self._last_error = None
        except Exception as exc:
            self._client = None
            self._running = False
            self._last_error = str(exc)

    def stop(self) -> None:
        client = self._client
        self._client = None
        self._running = False
        if client is None:
            return
        try:
            client.loop_stop()
        except Exception:
            pass
        try:
            client.disconnect()
        except Exception:
            pass

    def publish_intent(self, *, text: str, intent: str, confidence: float, slots: dict[str, str]) -> bool:
        client = self._client
        if client is None or not self._running:
            return False
        try:
            topic = f"{self._topic_prefix()}/events/intent"
            payload = {
                "source": "eleccia.listen",
                "text": text,
                "intent": intent,
                "confidence": float(confidence),
                "slots": slots,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "hostname": os.uname().nodename,
            }
            client.publish(
                topic,
                json.dumps(payload, ensure_ascii=True),
                qos=int(self._settings.qos),
                retain=bool(self._settings.retain),
            )
            return True
        except Exception as exc:
            self._last_error = str(exc)
            return False

    def _topic_prefix(self) -> str:
        prefix = (self._settings.topic_prefix or "eleccia").strip().strip("/")
        return prefix or "eleccia"

    def _client_id(self) -> str:
        configured = (self._settings.client_id or "").strip()
        if configured:
            return configured
        return f"eleccia-core-{os.getpid()}"
