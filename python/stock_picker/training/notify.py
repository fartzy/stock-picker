"""Publish this morning's pick list where we can actually reach it.

Primary: a plain-text file under picks/YYYY/MM/DD.txt plus picks/latest.txt,
committed and pushed so it opens in the GitHub app. Email is optional and
often stuck in local postfix -- do not rely on it.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

NOTIFY_EMAIL_ENV = "STOCK_PICKER_NOTIFY_EMAIL"
NOTIFY_EMAIL_FILE = Path.home() / ".config" / "api" / "notify-email.txt"
TOP_N = 20
PICKS_DIR_NAME = "picks"


def notify_address(key_file: Path | None = None) -> str | None:
    env = os.environ.get(NOTIFY_EMAIL_ENV, "").strip()
    if env:
        return env
    path = key_file if key_file is not None else NOTIFY_EMAIL_FILE
    if not path.is_file():
        return None
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "@" in stripped:
            return stripped
    return None


def format_picks_email(payload: dict) -> tuple[str, str]:
    as_of = payload.get("as_of", "")
    signals = payload.get("signals") or []
    scored = payload.get("scored_count", 0)
    skipped = payload.get("skipped") or []
    freshness = payload.get("freshness") or {}
    top = signals[:TOP_N]
    lines = [
        f"stockpicker picks for {as_of}",
        freshness.get("detail") or "",
        f"scored {scored} · {len(signals)} cleared 0.5% · showing top {len(top)}",
        "",
    ]
    if not top:
        lines.append("No tickers cleared the 0.5% threshold.")
    else:
        lines.append(f"{'ticker':8}  {'pred%':>7}  {'open':>10}")
        for row in top:
            pred = float(row["predicted_return"]) * 100
            lines.append(f"{row['ticker']:8}  {pred:6.2f}%  {float(row['open_price']):10.2f}")
    earnings = [s.get("ticker") for s in skipped if "earnings" in (s.get("reason") or "")]
    earnings = [t for t in earnings if t]
    if earnings:
        lines.extend(["", "Skipped earnings: " + ", ".join(earnings)])
    subject = (
        f"stockpicker {as_of}: {len(signals)} picks" if signals else f"stockpicker {as_of}: no picks"
    )
    return subject, "\n".join(lines).strip() + "\n"


def repo_root() -> Path:
    return Path(os.environ.get("BUILD_WORKING_DIRECTORY", Path.cwd()))


def picks_paths(as_of: str, root: Path | None = None) -> tuple[Path, Path]:
    """Partitioned file plus a one-tap latest.txt for the GitHub app."""
    base = (root or repo_root()) / PICKS_DIR_NAME
    year, month, day = as_of.split("-", 2)
    return base / year / month / f"{day}.txt", base / "latest.txt"


def write_picks_files(as_of: str, body: str, root: Path | None = None) -> Path:
    dated, latest = picks_paths(as_of, root)
    dated.parent.mkdir(parents=True, exist_ok=True)
    dated.write_text(body)
    latest.write_text(body)
    return dated


def publish_picks(as_of: str, body: str, root: Path | None = None) -> bool:
    """Write picks/ and push only that path. Other dirty files stay unstaged."""
    checkout = root or repo_root()
    dated = write_picks_files(as_of, body, checkout)
    picks = checkout / PICKS_DIR_NAME
    add = subprocess.run(
        ["git", "-C", str(checkout), "add", "--", "picks"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if add.returncode != 0:
        return False
    status = subprocess.run(
        ["git", "-C", str(checkout), "status", "--porcelain", "--", str(picks)],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if not status.stdout.strip():
        return True
    commit = subprocess.run(
        ["git", "-C", str(checkout), "commit", "-m", f"Morning picks {as_of}"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if commit.returncode != 0:
        return False
    pushed = subprocess.run(
        ["git", "-C", str(checkout), "push", "origin", "HEAD"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return pushed.returncode == 0


def format_not_ready_email(as_of: str, detail: str) -> tuple[str, str]:
    subject = f"stockpicker {as_of}: not ready"
    body = (
        f"{detail}\n\n"
        "Nightly prices/features/retrain did not finish in time. "
        "No score this morning -- check ~/Library/Logs/stock-picker/nightly.log.\n"
    )
    return subject, body


def send_email(subject: str, body: str, to: str | None = None) -> bool:
    address = to or notify_address()
    if not address:
        return False
    mailed = subprocess.run(
        ["/usr/bin/mail", "-s", subject, address],
        input=body,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if mailed.returncode == 0:
        return True
    return _send_via_mail_app(subject, body, address)


def _send_via_mail_app(subject: str, body: str, address: str) -> bool:
    """Mail.app fallback when /usr/bin/mail has no local SMTP."""
    escaped_subject = subject.replace("\\", "\\\\").replace('"', '\\"')
    escaped_body = body.replace("\\", "\\\\").replace('"', '\\"')
    script = (
        'tell application "Mail"\n'
        f'  set newMessage to make new outgoing message with properties '
        f'{{subject:"{escaped_subject}", content:"{escaped_body}", visible:false}}\n'
        "  tell newMessage\n"
        f'    make new to recipient at end of to recipients with properties {{address:"{address}"}}\n'
        "    send\n"
        "  end tell\n"
        "end tell\n"
    )
    apple = subprocess.run(
        ["osascript", "-e", script], capture_output=True, text=True, timeout=30, check=False
    )
    return apple.returncode == 0
