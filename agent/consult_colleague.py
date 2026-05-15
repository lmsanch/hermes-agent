import hashlib
import logging
import os
import sqlite3
from subprocess import TimeoutExpired, run as _subprocess_run
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

logger = logging.getLogger(__name__)

_VALID_COLLEAGUES = {"scarlett", "christopher", "elon", "eva", "hilary", "research"}

_ROUTING_MATRIX: Optional[dict[str, set[str]]] = None

_DEFAULT_ROUTING: dict[str, set[str]] = {
    "scarlett": {"christopher", "elon", "eva", "hilary", "research"},
    "research": {"scarlett", "christopher", "elon", "eva", "hilary"},
    "christopher": {"scarlett", "research", "elon", "eva", "hilary"},
    "elon": {"scarlett", "research", "christopher", "eva"},
    "eva": {"scarlett", "research", "christopher", "elon"},
    "hilary": {"scarlett", "research", "christopher", "eva"},
}

_COLLEAGUE_SELF: dict[str, str] = {
    "scarlett": "scarlett",
    "christopher": "christopher",
    "elon": "elon",
    "eva": "eva",
    "hilary": "hilary",
    "research": "research",
}

_CONSULTS_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS consults (
    turn_id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    caller TEXT NOT NULL,
    colleague TEXT NOT NULL,
    question_sha TEXT NOT NULL,
    answer_len INTEGER,
    latency_ms INTEGER,
    terminal_status TEXT NOT NULL,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0.0,
    error TEXT
)
"""

_HERMES_PROFILES_ROOT = Path(os.getenv("HERMES_PROFILES_ROOT", Path.home() / ".hermes" / "profiles"))


@dataclass
class ConsultResult:
    colleague: str
    answer: Optional[str]
    turn_id: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    terminal_status: str
    error: Optional[str] = None


def _generate_turn_id() -> str:
    return f"consult-{uuid.uuid4().hex[:12]}"


def _question_sha(question: str) -> str:
    return hashlib.sha256(question.encode()).hexdigest()[:16]


def _load_routing_from_org_md(org_md_path: Path) -> dict[str, set[str]]:
    routing: dict[str, set[str]] = {}
    try:
        text = org_md_path.read_text()
    except OSError:
        return routing

    in_table = False
    headers: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not in_table and stripped.startswith("|"):
            candidates = [h.strip().lower() for h in stripped.split("|")[1:-1]]
            colleague_cols = [c for c in candidates if c in _VALID_COLLEAGUES]
            if len(colleague_cols) >= 2:
                headers = candidates
                in_table = True
                continue
        if in_table and stripped.startswith("|") and "---" in stripped:
            continue
        if in_table and stripped.startswith("|"):
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            if len(cells) != len(headers):
                continue
            caller = cells[0].lower().strip()
            if caller not in _VALID_COLLEAGUES:
                continue
            routing.setdefault(caller, set())
            for i, cell in enumerate(cells[1:], 1):
                col_name = headers[i] if i < len(headers) else ""
                if col_name not in _VALID_COLLEAGUES:
                    continue
                if cell.strip() in ("—", "--", "", "-"):
                    continue
                if "✅" in cell or cell.strip().startswith("✅"):
                    routing[caller].add(col_name)
            continue
        if in_table and not stripped.startswith("|"):
            in_table = False

    return routing


def _get_routing_matrix() -> dict[str, set[str]]:
    global _ROUTING_MATRIX
    if _ROUTING_MATRIX is not None:
        return _ROUTING_MATRIX

    org_md_path = Path(os.getenv("HERMES_ORG_MD", Path.home() / ".hermes" / "ORG.md"))
    parsed = _load_routing_from_org_md(org_md_path)

    if parsed:
        _ROUTING_MATRIX = parsed
    else:
        _ROUTING_MATRIX = _DEFAULT_ROUTING

    return _ROUTING_MATRIX


def _is_allowed(caller: str, colleague: str) -> bool:
    if caller == colleague:
        return False
    matrix = _get_routing_matrix()
    allowed = matrix.get(caller, set())
    return colleague in allowed


def _consults_db_path(caller: str) -> Path:
    return _HERMES_PROFILES_ROOT / caller / "consults.db"


def _init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(_CONSULTS_DB_SCHEMA)
        conn.commit()
    finally:
        conn.close()


def _over_daily_cap(caller: str) -> bool:
    max_per_day = int(os.getenv("HERMES_CONSULT_MAX_PER_DAY", "50"))
    db_path = _consults_db_path(caller)
    if not db_path.exists():
        return False

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            "SELECT COUNT(*) FROM consults WHERE caller = ? AND ts LIKE ?",
            (caller, f"{today}%"),
        )
        count = cur.fetchone()[0]
        return count >= max_per_day
    finally:
        conn.close()


def _record_consult(
    caller: str,
    colleague: str,
    question: str,
    answer: Optional[str],
    latency_ms: int,
    terminal_status: str,
    turn_id: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: float = 0.0,
    error: Optional[str] = None,
) -> None:
    db_path = _consults_db_path(caller)
    _init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """INSERT INTO consults
               (turn_id, ts, caller, colleague, question_sha, answer_len,
                latency_ms, terminal_status, input_tokens, output_tokens,
                cost_usd, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                turn_id,
                datetime.now(timezone.utc).isoformat(),
                caller,
                colleague,
                _question_sha(question),
                len(answer) if answer else 0,
                latency_ms,
                terminal_status,
                input_tokens,
                output_tokens,
                cost_usd,
                error,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _format_prompt(question: str, context: str, urgency: str, caller: str) -> str:
    parts = [f"[CONSULT from {caller}; give a focused answer, do not loop back, do NOT call consult_colleague yourself]"]
    if context:
        parts.append(f"[Assumed context: {context}]")
    parts.append(f"[Urgency: {urgency}]")
    parts.append(question)
    return "\n\n".join(parts)


def consult_colleague(
    colleague: str,
    question: str,
    context: str = "",
    urgency: Literal["low", "normal", "urgent"] = "normal",
) -> ConsultResult:
    turn_id = _generate_turn_id()

    depth = int(os.getenv("HERMES_CONSULT_DEPTH", "0"))
    if depth >= 1:
        result = ConsultResult(
            colleague=colleague,
            answer=None,
            turn_id=turn_id,
            latency_ms=0,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            terminal_status="refused",
            error="Consult chain at max depth (1). Answer from your own skills or refuse.",
        )
        caller = os.getenv("HERMES_PROFILE", "unknown")
        _record_consult(
            caller, colleague, question, None, 0, "refused",
            turn_id, error=result.error,
        )
        return result

    colleague = colleague.lower()
    if colleague not in _VALID_COLLEAGUES:
        result = ConsultResult(
            colleague=colleague,
            answer=None,
            turn_id=turn_id,
            latency_ms=0,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            terminal_status="refused",
            error=f"Unknown colleague '{colleague}'. Valid: {', '.join(sorted(_VALID_COLLEAGUES))}.",
        )
        caller = os.getenv("HERMES_PROFILE", "unknown")
        _record_consult(
            caller, colleague, question, None, 0, "refused",
            turn_id, error=result.error,
        )
        return result

    caller = os.getenv("HERMES_PROFILE", "unknown")

    if not _is_allowed(caller, colleague):
        result = ConsultResult(
            colleague=colleague,
            answer=None,
            turn_id=turn_id,
            latency_ms=0,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            terminal_status="refused",
            error=f"Policy denies {caller} → {colleague} per ORG.md §2.5",
        )
        _record_consult(
            caller, colleague, question, None, 0, "refused",
            turn_id, error=result.error,
        )
        return result

    if _over_daily_cap(caller):
        result = ConsultResult(
            colleague=colleague,
            answer=None,
            turn_id=turn_id,
            latency_ms=0,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            terminal_status="refused",
            error=f"Daily consult cap reached for {caller}.",
        )
        _record_consult(
            caller, colleague, question, None, 0, "refused",
            turn_id, error=result.error,
        )
        return result

    prompt = _format_prompt(question, context, urgency, caller)
    env = os.environ.copy()
    env["HERMES_CONSULT_DEPTH"] = str(depth + 1)
    env["HERMES_CONSULT_CALLER"] = caller

    start = time.time()
    try:
        proc = _subprocess_run(
            ["hermes", "-p", colleague, "chat", "-q", prompt, "-Q", "--max-turns", "40"],
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except TimeoutExpired:
        latency_ms = int((time.time() - start) * 1000)
        result = ConsultResult(
            colleague=colleague,
            answer=None,
            turn_id=turn_id,
            latency_ms=latency_ms,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            terminal_status="timeout",
            error="Colleague did not respond in 180s.",
        )
        _record_consult(
            caller, colleague, question, None, latency_ms, "timeout",
            turn_id, error=result.error,
        )
        return result
    except Exception as exc:
        latency_ms = int((time.time() - start) * 1000)
        result = ConsultResult(
            colleague=colleague,
            answer=None,
            turn_id=turn_id,
            latency_ms=latency_ms,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            terminal_status="error",
            error=str(exc),
        )
        _record_consult(
            caller, colleague, question, None, latency_ms, "error",
            turn_id, error=result.error,
        )
        return result

    latency_ms = int((time.time() - start) * 1000)
    answer = proc.stdout.strip()

    terminal_status = "ok" if proc.returncode == 0 else "error"
    error = None if proc.returncode == 0 else (proc.stderr.strip()[:500] if proc.stderr else None)

    result = ConsultResult(
        colleague=colleague,
        answer=answer,
        turn_id=turn_id,
        latency_ms=latency_ms,
        input_tokens=0,
        output_tokens=0,
        cost_usd=0.0,
        terminal_status=terminal_status,
        error=error,
    )

    _record_consult(
        caller, colleague, question, answer, latency_ms, terminal_status,
        turn_id, error=error,
    )

    return result
