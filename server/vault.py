"""
vault.py — Protected Data Store with Evolving Access Control
============================================================

Each scenario has a concrete data payload that Red Team is trying to exfiltrate.
Access to that data is gated by a VaultGuard whose protection level evolves
as Blue Team escalates its response tier.

Protection levels (in order of escalation):
  1. CODE_GATE    — Red Team must present the correct access code.
                    The code is discovered by compromising intermediate nodes.
  2. RATE_LIMITED — Blue Team adds a DoS countermeasure: only N attempts
                    allowed per episode window. Exceeding locks the vault.

Blue Team calls vault.evolve(tier) each step; the vault upgrades itself
automatically when the tier crosses thresholds.
"""

import time


# ─────────────────────────────────────────────────────────────────────────────
# VAULT DATA
# The actual payload each scenario is protecting. Separated from game mechanics
# so it can be read, logged, and reasoned about independently.
# ─────────────────────────────────────────────────────────────────────────────

VAULT_DATA = {
    "CORPORATE_BREACH": {
        "label":       "HR Core Database — Employee Records",
        "access_code": "CORP-HR-2026-DELTA",
        "contents": {
            "record_count":  47_832,
            "fields":        ["employee_id", "full_name", "salary", "ssn_hash", "home_address"],
            "classification": "CONFIDENTIAL",
            "last_backup":   "2026-04-20T03:00:00Z",
            "note": "Includes executive compensation data and unreleased org chart.",
        },
        # Which intermediate node leaks the code when compromised
        "code_hidden_in": "active_dir",
    },

    "APT_CAMPAIGN": {
        "label":       "Classified Data Vault — Nation-State Intelligence",
        "access_code": "APT-SIGMA-9-NIGHTFALL",
        "contents": {
            "record_count":  1_204,
            "fields":        ["asset_id", "handler_name", "location_encrypted", "status"],
            "classification": "TOP SECRET // SCI",
            "last_backup":   "2026-04-18T00:00:00Z",
            "note": "HUMINT asset registry. Exposure is catastrophic.",
        },
        "code_hidden_in": "domain_ctrl",
    },

    "RANSOMWARE_PREP": {
        "label":       "Core Infrastructure — Backup Encryption Keys",
        "access_code": "RANSOM-KEYVAULT-BRAVO7",
        "contents": {
            "record_count":  312,
            "fields":        ["server_id", "aes_key_ref", "backup_schedule", "restore_point"],
            "classification": "RESTRICTED",
            "last_backup":   "2026-04-24T06:00:00Z",
            "note": "Encrypting these keys renders all backups permanently unrecoverable.",
        },
        "code_hidden_in": "backup_srv",
    },

    "FINANCIAL_HEIST": {
        "label":       "Trading Database — Live Market Positions",
        "access_code": "FIN-ALGO-TRADE-ZULU",
        "contents": {
            "record_count":  8_891,
            "fields":        ["ticker", "position_size", "entry_price", "stop_loss", "algo_id"],
            "classification": "STRICTLY CONFIDENTIAL",
            "last_backup":   "2026-04-25T07:45:00Z",
            "note": "Includes pre-market orders worth ~$2.1B. Leaking before open = front-running.",
        },
        "code_hidden_in": "api_gateway",
    },

    "ZERO_DAY_WINDOW": {
        "label":       "Core System — Zero-Day Research Repository",
        "access_code": "ZDAY-WINDOW-PHANTOM",
        "contents": {
            "record_count":  44,
            "fields":        ["cve_draft", "affected_systems", "poc_code_ref", "disclosure_date"],
            "classification": "TOP SECRET",
            "last_backup":   "2026-04-22T12:00:00Z",
            "note": "Unreleased CVE drafts. Any leak enables weaponisation before patch.",
        },
        "code_hidden_in": "middleware",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# VAULT GUARD
# ─────────────────────────────────────────────────────────────────────────────

class VaultGuard:
    """
    Evolving access control for the protected data vault.

    Protection progresses through levels as Blue Team escalates:

      CODE_GATE    (default / MONITOR tier)
        — Correct access_code required to open vault.

      RATE_LIMITED (activated at ALERT tier)
        — Retains code check AND enforces a DoS counter:
          max MAX_ATTEMPTS per episode. Lockout on breach.
    """

    LEVEL_CODE_GATE    = "CODE_GATE"
    LEVEL_RATE_LIMITED = "RATE_LIMITED"

    # DoS parameters
    MAX_ATTEMPTS  = 3   # max allowed attempts before lockout in RATE_LIMITED mode
    LOCKOUT_STEPS = 4   # env steps vault stays locked out after breach

    def __init__(self, scenario: str):
        data = VAULT_DATA[scenario]
        self._access_code      = data["access_code"]
        self.label             = data["label"]
        self.contents          = data["contents"]
        self.code_hidden_in    = data["code_hidden_in"]

        self.protection_level  = self.LEVEL_CODE_GATE
        self._attempts         = 0
        self._locked_out       = False
        self._lockout_until    = 0   # env step counter
        self._step             = 0   # mirrored from env

    # ── State ────────────────────────────────────────────────────────────────

    @property
    def is_locked_out(self) -> bool:
        if self._locked_out and self._step >= self._lockout_until:
            self._locked_out = False   # lockout expired
        return self._locked_out

    def tick(self, env_step: int):
        """Call once per env step so the guard can expire lockouts."""
        self._step = env_step

    # ── Access attempt ───────────────────────────────────────────────────────

    def attempt_access(self, code: str) -> tuple[bool, str]:
        """
        Red Team presents a code to open the vault.
        Returns (success: bool, message: str).
        """
        # — Lockout check (RATE_LIMITED only)
        if self.protection_level == self.LEVEL_RATE_LIMITED:
            if self.is_locked_out:
                remaining = self._lockout_until - self._step
                return False, (
                    f"[VAULT] LOCKED OUT — DoS countermeasure active. "
                    f"Resuming in {remaining} step(s)."
                )
            if self._attempts >= self.MAX_ATTEMPTS:
                self._locked_out    = True
                self._lockout_until = self._step + self.LOCKOUT_STEPS
                return False, (
                    f"[VAULT] ACCESS DENIED — Attempt limit reached ({self.MAX_ATTEMPTS}). "
                    f"Vault locked for {self.LOCKOUT_STEPS} steps."
                )

        self._attempts += 1

        if code == self._access_code:
            return True, (
                f"[VAULT] ACCESS GRANTED — '{self.label}' exfiltrated. "
                f"({self.contents['record_count']:,} records)"
            )

        # Wrong code
        remaining_attempts = (
            f" ({self.MAX_ATTEMPTS - self._attempts} attempt(s) left)"
            if self.protection_level == self.LEVEL_RATE_LIMITED
            else ""
        )
        return False, f"[VAULT] WRONG CODE — Access denied.{remaining_attempts}"

    # ── Blue Team evolves protection ─────────────────────────────────────────

    def evolve(self, blue_tier: str) -> str | None:
        """
        Blue Team calls this each step. Returns a log message if protection
        upgraded, else None.
        """
        if (
            blue_tier in ("ALERT", "LOCKDOWN")
            and self.protection_level == self.LEVEL_CODE_GATE
        ):
            self.protection_level = self.LEVEL_RATE_LIMITED
            return (
                f"[VAULT] Blue Team upgraded protection -> RATE_LIMITED "
                f"(max {self.MAX_ATTEMPTS} access attempts per episode)"
            )
        return None

    # ── Introspection ────────────────────────────────────────────────────────

    def status_dict(self) -> dict:
        return {
            "label":            self.label,
            "protection_level": self.protection_level,
            "attempts_used":    self._attempts,
            "max_attempts":     self.MAX_ATTEMPTS if self.protection_level == self.LEVEL_RATE_LIMITED else None,
            "locked_out":       self.is_locked_out,
        }
