"""맥 알림 (osascript). 실패해도 조용히 넘어간다."""
from __future__ import annotations

import logging
import subprocess

log = logging.getLogger("tickerfinder.notify")


def notify(title: str, msg: str) -> None:
    log.info("%s: %s", title, msg)
    try:
        q = lambda s: s.replace('"', "'")  # noqa: E731
        subprocess.run(["osascript", "-e", f'display notification "{q(msg)}" with title "{q(title)}"'], timeout=5, capture_output=True)
    except Exception:  # noqa: BLE001
        pass
