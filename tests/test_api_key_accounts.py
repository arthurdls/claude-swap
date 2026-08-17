"""Tests for managed API-key (``/login`` key) account support.

Covers kind detection, ``--add-token`` auto-detection, the cross-kind collision
guard, the ``add_account`` live-key guard, kind+platform-aware active credential
read/write with OAuth↔API-key mutual exclusion, the "API key — no quota" usage
display, the ``cswap run`` session guard, and export/import of raw keys.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

from claude_swap import macos_keychain
from claude_swap import session as session_mod
from claude_swap.api_key_helper import ApiKeyHelperChannel
from claude_swap.credentials import (
    CLAUDE_CODE_KEYCHAIN_SERVICE,
    CLAUDE_CODE_MANAGED_KEYCHAIN_SERVICE,
    approved_form,
    looks_like_api_key,
)
from claude_swap.exceptions import SessionError, ValidationError
from claude_swap.json_output import USAGE_API_KEY, usage_fields
from claude_swap.models import Platform
from claude_swap.paths import get_credentials_path, get_global_config_path
from claude_swap.session import SessionManager
from claude_swap.switcher import ClaudeAccountSwitcher
from claude_swap.transfer import export_accounts, import_accounts

API_KEY = "sk-ant-api03-" + "a1b2c3d4e5" * 4  # 53 chars
OTHER_KEY = "sk-ant-api03-" + "z9y8x7w6v5" * 4
OAUTH_JSON = json.dumps(
    {"claudeAiOauth": {"accessToken": "tok", "refreshToken": "rtok", "expiresAt": 9}}
)


def _linux_switcher() -> ClaudeAccountSwitcher:
    s = ClaudeAccountSwitcher()
    s.platform = Platform.LINUX
    s._setup_directories()
    s._init_sequence_file()
    return s


def _macos_switcher() -> ClaudeAccountSwitcher:
    s = ClaudeAccountSwitcher()
    s.platform = Platform.MACOS
    s._setup_directories()
    s._init_sequence_file()
    return s


def _read_global_config() -> dict:
    return json.loads(get_global_config_path().read_text(encoding="utf-8"))


def _helper() -> ApiKeyHelperChannel:
    """A fresh channel: it resolves every path at call time, so it reads
    whatever the switcher under test just wrote under the patched HOME."""
    return ApiKeyHelperChannel(logging.getLogger("test"))


# ---------------------------------------------------------------------------
# Kind detection helpers
# ---------------------------------------------------------------------------


class TestKindDetection:
    def test_api_key_detected(self):
        assert looks_like_api_key(API_KEY) is True

    @pytest.mark.parametrize(
        "value",
        [
            "",
            None,
            "sk-ant-oat01-abcdef",  # setup-token, not a key
            OAUTH_JSON,  # OAuth JSON blob
            '{"x": "sk-ant-api03-inside-json"}',  # JSON that merely contains a key
        ],
    )
    def test_non_api_key(self, value):
        assert looks_like_api_key(value) is False

    def test_approved_form_is_last_20(self):
        assert approved_form(API_KEY) == API_KEY[-20:]
        assert len(approved_form(API_KEY)) == 20


# ---------------------------------------------------------------------------
# --add-token auto-detection
# ---------------------------------------------------------------------------


class TestAddTokenApiKey:
    def test_adds_api_key_account(self, temp_home: Path, capsys):
        s = _linux_switcher()
        s.add_account_from_token(API_KEY)

        assert s._account_kind("1") == "api_key"
        # default synthesized label
        data = s._get_sequence_data()
        assert data["accounts"]["1"]["email"] == "api-key-1@token.local"
        # the raw key is stored verbatim as the backup credential
        assert s._read_account_credentials("1", "api-key-1@token.local") == API_KEY
        out = capsys.readouterr().out
        assert "Added" in out and "API key" in out

    def test_setup_token_stays_oauth(self, temp_home: Path):
        s = _linux_switcher()
        s.add_account_from_token("sk-ant-oat01-abc")
        assert s._account_kind("1") == "oauth"
        email = s._get_sequence_data()["accounts"]["1"]["email"]
        assert email == "setup-token-1@token.local"
        blob = json.loads(s._read_account_credentials("1", email))
        assert blob["claudeAiOauth"]["accessToken"] == "sk-ant-oat01-abc"

    def test_refresh_in_place_same_api_key_account(self, temp_home: Path):
        s = _linux_switcher()
        s.add_account_from_token(API_KEY, email="me@example.com")
        s.add_account_from_token(OTHER_KEY, email="me@example.com")
        data = s._get_sequence_data()
        assert len(data["accounts"]) == 1
        assert s._read_account_credentials("1", "me@example.com") == OTHER_KEY


class TestCrossKindCollision:
    def test_api_key_rejected_when_email_is_oauth(self, temp_home: Path):
        s = _linux_switcher()
        s.add_account_from_token("sk-ant-oat01-abc", email="dup@example.com")
        with pytest.raises(ValidationError, match="already exists as an OAuth account"):
            s.add_account_from_token(API_KEY, email="dup@example.com")

    def test_oauth_rejected_when_email_is_api_key(self, temp_home: Path):
        s = _linux_switcher()
        s.add_account_from_token(API_KEY, email="dup@example.com")
        with pytest.raises(ValidationError, match="already exists as an API-key account"):
            s.add_account_from_token("sk-ant-oat01-abc", email="dup@example.com")


# ---------------------------------------------------------------------------
# Active credential read/write + mutual exclusion
# ---------------------------------------------------------------------------


class TestWriteCredentialsLinux:
    def test_activate_key_then_oauth(self, temp_home: Path):
        s = _linux_switcher()
        cred_file = get_credentials_path()
        cred_file.parent.mkdir(parents=True, exist_ok=True)
        cred_file.write_text(OAUTH_JSON, encoding="utf-8")

        # Activate the API key: the helper holds it, approved is recorded, the
        # OAuth file is cleared — and primaryApiKey is deliberately NOT written,
        # because a session that starts alongside one memoizes it for its whole
        # lifetime and can never be switched off it again.
        s._write_credentials(API_KEY)
        cfg = _read_global_config()
        assert "primaryApiKey" not in cfg
        assert _helper().active_key() == API_KEY
        assert API_KEY[-20:] in cfg["customApiKeyResponses"]["approved"]
        assert not cred_file.exists()
        # cswap must still be able to read back the credential it just activated.
        assert s._read_credentials() == API_KEY

        # Switch back to OAuth: file restored, helper unregistered, approved kept.
        s._write_credentials(OAUTH_JSON)
        assert cred_file.read_text(encoding="utf-8") == OAUTH_JSON
        cfg = _read_global_config()
        assert "primaryApiKey" not in cfg
        assert _helper().active_key() == ""
        assert API_KEY[-20:] in cfg["customApiKeyResponses"]["approved"]

    def test_activate_key_falls_back_to_config_without_the_helper(
        self, temp_home: Path
    ):
        """A foreign apiKeyHelper is left alone, so the key needs its old home.

        Those users keep the restart-to-pick-up behaviour rather than silently
        losing their own helper — and must still get a working switch.
        """
        s = _linux_switcher()
        settings = _helper().settings_path
        settings.parent.mkdir(parents=True, exist_ok=True)
        settings.write_text(
            json.dumps({"apiKeyHelper": "/opt/mine/print-key.sh"}), encoding="utf-8"
        )

        s._write_credentials(API_KEY)

        cfg = _read_global_config()
        assert cfg["primaryApiKey"] == API_KEY
        assert s._read_credentials() == API_KEY
        # Their helper is untouched, and ours never claimed the key.
        assert json.loads(settings.read_text(encoding="utf-8"))["apiKeyHelper"] == (
            "/opt/mine/print-key.sh"
        )
        assert _helper().active_key() == ""

    def test_read_credentials_returns_active_key(self, temp_home: Path):
        s = _linux_switcher()
        get_global_config_path().write_text(
            json.dumps({"primaryApiKey": API_KEY}), encoding="utf-8"
        )
        assert s._read_credentials() == API_KEY

    def test_oauth_file_not_misread_as_key(self, temp_home: Path):
        s = _linux_switcher()
        cred_file = get_credentials_path()
        cred_file.parent.mkdir(parents=True, exist_ok=True)
        cred_file.write_text(OAUTH_JSON, encoding="utf-8")
        # primaryApiKey also present, but the OAuth file wins (read first).
        get_global_config_path().write_text(
            json.dumps({"primaryApiKey": API_KEY}), encoding="utf-8"
        )
        assert s._read_credentials() == OAUTH_JSON


class TestWriteCredentialsMacOS:
    def test_activate_key_uses_keychain_not_config(self, temp_home, block_real_keychain):
        store = block_real_keychain
        s = _macos_switcher()
        acct = macos_keychain.keychain_account_name()
        store.set_password(CLAUDE_CODE_KEYCHAIN_SERVICE, acct, OAUTH_JSON)

        s._write_credentials(API_KEY)

        # Key in the managed keychain service; OAuth keychain item cleared.
        assert store.get_password(CLAUDE_CODE_MANAGED_KEYCHAIN_SERVICE, acct) == API_KEY
        assert store.get_password(CLAUDE_CODE_KEYCHAIN_SERVICE, acct) is None
        # approved recorded, but the full key stays OUT of plaintext config.
        cfg = _read_global_config()
        assert API_KEY[-20:] in cfg["customApiKeyResponses"]["approved"]
        assert "primaryApiKey" not in cfg

    def test_switch_back_to_oauth_clears_key(self, temp_home, block_real_keychain):
        store = block_real_keychain
        s = _macos_switcher()
        s._write_credentials(API_KEY)
        acct = macos_keychain.keychain_account_name()
        assert store.get_password(CLAUDE_CODE_MANAGED_KEYCHAIN_SERVICE, acct) == API_KEY

        s._write_credentials(OAUTH_JSON)
        # managed keychain cleared, OAuth keychain populated, approved kept.
        assert store.get_password(CLAUDE_CODE_MANAGED_KEYCHAIN_SERVICE, acct) is None
        assert store.get_password(CLAUDE_CODE_KEYCHAIN_SERVICE, acct) == OAUTH_JSON
        cfg = _read_global_config()
        assert API_KEY[-20:] in cfg["customApiKeyResponses"]["approved"]

    def test_read_credentials_from_managed_keychain(self, temp_home, block_real_keychain):
        store = block_real_keychain
        s = _macos_switcher()
        acct = macos_keychain.keychain_account_name()
        store.set_password(CLAUDE_CODE_MANAGED_KEYCHAIN_SERVICE, acct, API_KEY)
        assert s._read_credentials() == API_KEY


# ---------------------------------------------------------------------------
# Usage display ("API key — no quota")
# ---------------------------------------------------------------------------


class TestUsageDisplay:
    def test_usage_fields_maps_api_key(self):
        assert usage_fields(USAGE_API_KEY) == ("api_key", None)

    def test_collect_usage_short_circuits(self, temp_home: Path):
        s = _linux_switcher()
        info = [(2, "api-key-2@token.local", "", "", False, API_KEY, "")]
        entries = s._collect_usage_entries(info)
        assert entries["2"].sentinel == USAGE_API_KEY
        assert entries["2"].decision_value() == USAGE_API_KEY

    def test_active_account_usage_short_circuits(self, temp_home: Path):
        s = _linux_switcher()
        get_global_config_path().write_text(
            json.dumps({"primaryApiKey": API_KEY}), encoding="utf-8"
        )
        entry = s._active_account_usage("2", "api-key-2@token.local", "")
        assert entry.sentinel == USAGE_API_KEY
        assert entry.decision_value() == USAGE_API_KEY


class TestStrategyBehaviour:
    """API-key accounts are never *rate-limited* (next-available can fall back to
    them), but `best` must NOT auto-prefer them — they have no measurable quota and
    jumping to one would silently spend paid per-token credits."""

    def test_api_key_headroom_is_unknown(self):
        # None headroom == "unknown" == never auto-skipped by next-available.
        from claude_swap import oauth

        assert oauth.account_headroom(USAGE_API_KEY) is None

    def test_best_does_not_jump_to_api_key_even_when_exhausted(
        self, temp_home: Path, monkeypatch
    ):
        s = _linux_switcher()
        s.add_account_from_token("sk-ant-oat01-x", slot=1)  # OAuth, switchable
        s.add_account_from_token(API_KEY, slot=2)  # API key, switchable
        # Current OAuth account (1) is fully exhausted; the only other account is
        # the no-quota API key. `best` must stay put rather than burn API credits.
        monkeypatch.setattr(
            s,
            "_usage_by_account",
            lambda: {"1": {"five_hour": {"pct": 100.0}}, "2": USAGE_API_KEY},
        )
        target, _ = s._select_best_switchable("1")
        assert target is None


# ---------------------------------------------------------------------------
# add_account guard against capturing a live API-key login
# ---------------------------------------------------------------------------


class TestAddAccountGuard:
    def test_rejects_live_api_key_login(self, temp_home: Path):
        s = _linux_switcher()
        # Lingering oauthAccount identity + an active managed key in config.
        get_global_config_path().write_text(
            json.dumps(
                {
                    "oauthAccount": {"emailAddress": "stale@example.com"},
                    "primaryApiKey": API_KEY,
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValidationError, match="Active login is an API-key account"):
            s.add_account()


# ---------------------------------------------------------------------------
# Session-mode guard
# ---------------------------------------------------------------------------


class TestSessionGuard:
    def _seed_api_key_account(self) -> ClaudeAccountSwitcher:
        s = _linux_switcher()
        s.add_account_from_token(API_KEY, slot=2)
        return s

    def test_setup_session_rejects(self, temp_home: Path):
        mgr = SessionManager(self._seed_api_key_account())
        with pytest.raises(SessionError, match="does not support API-key accounts"):
            mgr.setup_session("2", share=True)

    def test_run_rejects_before_exec(self, temp_home: Path, monkeypatch):
        mgr = SessionManager(self._seed_api_key_account())
        monkeypatch.setattr(session_mod.shutil, "which", lambda name: "/fake/claude")
        with pytest.raises(SessionError, match="does not support API-key accounts"):
            mgr.run("2", [], share=True)


# ---------------------------------------------------------------------------
# Export / import of raw keys
# ---------------------------------------------------------------------------


class TestExportImport:
    def test_round_trip_preserves_key_and_kind(self, tmp_path: Path):
        src_home = tmp_path / "src"
        (src_home / ".claude").mkdir(parents=True)
        with _patched_home(src_home):
            src = _linux_switcher()
            src.add_account_from_token(API_KEY, slot=1)
            out = tmp_path / "b.cswap"
            export_accounts(src, str(out))
            payload = json.loads(out.read_text(encoding="utf-8"))
            # exported as a raw string, tagged api_key — not a JSON object.
            assert payload["accounts"][0]["credentials"] == API_KEY
            assert payload["accounts"][0]["kind"] == "api_key"

        dst_home = tmp_path / "dst"
        (dst_home / ".claude").mkdir(parents=True)
        with _patched_home(dst_home):
            dst = _linux_switcher()
            import_accounts(dst, str(out))
            assert dst._account_kind("1") == "api_key"
            assert dst._read_account_credentials("1", "api-key-1@token.local") == API_KEY


# ---------------------------------------------------------------------------
# Live-credential resolution: WHICH key is live, and which slot is it
# ---------------------------------------------------------------------------

# Distinct keys, one per door, so a precedence assertion names the winner rather
# than merely proving "some key came back".
ENV_KEY = "sk-ant-api03-" + "e0e0e0e0e0" * 4
FD_KEY = "sk-ant-api03-" + "f1f1f1f1f1" * 4
HELPER_KEY = "sk-ant-api03-" + "h2h2h2h2h2" * 4
LOGIN_KEY = "sk-ant-api03-" + "l3l3l3l3l3" * 4

# An OAuth identity matching no managed slot. This is the state a key login
# leaves behind: ``/login`` with an ``sk-ant-api…`` key does not clear
# ``oauthAccount``, and a running Claude Code rewrites that block on every
# re-login — so the identity there drifts to whichever account touched it last.
STALE_OAUTH = {
    "emailAddress": "someone-else@example.com",
    "organizationUuid": "org-uuid-matching-no-slot",
}


def _arm_helper(api_key: str) -> None:
    """Register the ``apiKeyHelper`` hook on ``api_key``, as a switch would."""
    assert _helper().install(api_key) is True


def _write_global_config(**keys) -> None:
    get_global_config_path().write_text(json.dumps(keys), encoding="utf-8")


def _key_slot_switcher(live_config: dict) -> ClaudeAccountSwitcher:
    """One API-key slot, plus a stale ``oauthAccount`` over ``live_config``.

    ``live_config`` supplies the API-key door under test (e.g.
    ``{"primaryApiKey": …}``). ``activeAccountNumber`` is pinned to the slot on
    purpose: the resolver must never *use* it, so the tests that expect ``None``
    have to be able to fail if it ever became a fallback.
    """
    s = _linux_switcher()
    s.add_account_from_token(API_KEY)
    data = s._get_sequence_data()
    data["activeAccountNumber"] = 1
    s._write_json(s.sequence_file, data)
    _write_global_config(oauthAccount=STALE_OAUTH, **live_config)
    return s


class TestLiveApiKeySlotDetection:
    """An API-key login is resolved by its KEY, never by ``oauthAccount``.

    Reading ``oauthAccount`` alone is what made ``cswap status`` print
    ``(not managed)`` for a slot cswap owns, and made the auto engine burn ticks
    on ``unmanaged-active-account`` while an API-key slot was live.
    """

    def test_live_key_resolves_its_own_slot(self, temp_home: Path):
        s = _key_slot_switcher({"primaryApiKey": API_KEY})
        assert s.current_account_number() == "1"
        assert s.has_live_login() is True

    def test_live_key_matching_no_slot_is_none_not_a_guess(self, temp_home: Path):
        """The invariant ``current_account_number`` exists to protect.

        A slot returned here would be evaluated for usage and then switched onto
        by ``_perform_switch``'s no-backup direct-activation path — overwriting a
        login cswap does not own. ``activeAccountNumber`` says 1 and the OAuth
        block names an account; neither may stand in for a key match.
        """
        s = _key_slot_switcher({"primaryApiKey": OTHER_KEY})
        assert s._get_sequence_data()["activeAccountNumber"] == 1
        assert s.current_account_number() is None
        # It is still a live login, so the engine says "unmanaged", not "absent".
        assert s.has_live_login() is True

    def test_truncated_key_is_a_miss(self, temp_home: Path):
        """No prefix or partial credit — a near-miss key is an unmanaged login."""
        s = _key_slot_switcher({"primaryApiKey": API_KEY[:-4]})
        assert s.current_account_number() is None

    def test_kindless_slot_holding_the_live_key_is_not_matched(self, temp_home: Path):
        """Only a slot that *declares* ``kind == "api_key"`` may be resolved.

        A slot with no ``kind`` reads as OAuth everywhere else in the class
        (``_account_kind``'s back-compat default, which the session guard, export
        and cross-kind collision checks all key off), and only a pre-``kind``
        install could have stored a raw key on one. Matching it here would report
        a slot the rest of cswap treats as OAuth as a live API-key account — so it
        is a miss even though its stored bytes *are* the live key. The safe
        failure: "unmanaged" blocks a switch, a wrong slot invites one. Re-adding
        it with ``cswap add-token`` is the fix, not a looser match here.
        """
        s = _linux_switcher()
        s.add_account_from_token(API_KEY)
        data = s._get_sequence_data()
        del data["accounts"]["1"]["kind"]
        s._write_json(s.sequence_file, data)
        assert s._account_kind("1") == "oauth"
        assert s._read_account_credentials("1", "api-key-1@token.local") == API_KEY

        _write_global_config(oauthAccount=STALE_OAUTH, primaryApiKey=API_KEY)
        assert s.current_account_number() is None
        assert s.has_live_login() is True

    def test_oauth_login_matching_a_slot_is_unchanged(self, temp_home: Path):
        s = _linux_switcher()
        s.add_account_from_token("sk-ant-oat01-abc", email="me@example.com")
        _write_global_config(oauthAccount={"emailAddress": "me@example.com"})
        assert s.current_account_number() == "1"
        assert s.has_live_login() is True

    def test_unmanaged_oauth_login_is_unchanged(self, temp_home: Path):
        s = _linux_switcher()
        s.add_account_from_token(API_KEY)
        _write_global_config(oauthAccount=STALE_OAUTH)
        assert s.current_account_number() is None
        assert s.has_live_login() is True

    def test_no_credential_anywhere_has_no_live_login(self, temp_home: Path):
        s = _linux_switcher()
        s.add_account_from_token(API_KEY)
        _write_global_config()
        assert s.current_account_number() is None
        assert s.has_live_login() is False

    def test_helper_held_key_resolves_its_slot(self, temp_home: Path):
        """The live-switch channel is where an activated key actually lives.

        ``_write_credentials`` stores the key ONLY in the helper key file, so a
        resolver that reads ``primaryApiKey`` alone sees nothing at all on the
        machine state cswap itself produces.
        """
        s = _key_slot_switcher({})
        _arm_helper(API_KEY)
        assert s.current_account_number() == "1"


class TestLiveApiKeyPrecedence:
    """Claude Code's own resolution order, honored exactly.

    ``ANTHROPIC_API_KEY`` (approved only) -> key file descriptor -> apiKeyHelper
    -> the ``/login``-managed key -> claude.ai OAuth -> none. Each test arms every
    door *below* the one under test, so a wrong order fails rather than passing on
    an empty lower door.
    """

    def test_approved_env_key_outranks_every_other_door(
        self, temp_home: Path, monkeypatch
    ):
        s = _linux_switcher()
        _arm_helper(HELPER_KEY)
        _write_global_config(
            primaryApiKey=LOGIN_KEY,
            customApiKeyResponses={"approved": [approved_form(ENV_KEY)]},
        )
        monkeypatch.setenv("ANTHROPIC_API_KEY", ENV_KEY)
        assert s._resolve_live_api_key() == (ENV_KEY, True, "env")

    def test_unapproved_env_key_is_not_live(self, temp_home: Path, monkeypatch):
        """Claude Code prompts for an unrecognized key and does not use it.

        So an unapproved value must fall through — the ``/login`` key below is
        still what gets billed.
        """
        s = _linux_switcher()
        _write_global_config(primaryApiKey=LOGIN_KEY)
        monkeypatch.setenv("ANTHROPIC_API_KEY", ENV_KEY)
        assert s._resolve_live_api_key() == (LOGIN_KEY, True, "login")

    def test_unapproved_env_key_leaves_the_oauth_login_live(
        self, temp_home: Path, monkeypatch
    ):
        s = _linux_switcher()
        s.add_account_from_token("sk-ant-oat01-abc", email="me@example.com")
        _write_global_config(oauthAccount={"emailAddress": "me@example.com"})
        monkeypatch.setenv("ANTHROPIC_API_KEY", ENV_KEY)
        assert s._resolve_live_api_key() == ("", False, "none")
        assert s.current_account_number() == "1"

    def test_env_key_approved_under_a_different_key_is_not_live(
        self, temp_home: Path, monkeypatch
    ):
        """``approved`` holds ``key[-20:]``; another key's entry must not count."""
        s = _linux_switcher()
        _write_global_config(
            primaryApiKey=LOGIN_KEY,
            customApiKeyResponses={"approved": [approved_form(LOGIN_KEY)]},
        )
        monkeypatch.setenv("ANTHROPIC_API_KEY", ENV_KEY)
        assert s._resolve_live_api_key() == (LOGIN_KEY, True, "login")

    @pytest.mark.skipif(
        not hasattr(os, "pread"),
        reason="no os.pread on Windows, so a descriptor is never read there",
    )
    def test_descriptor_outranks_helper_and_login_key(
        self, temp_home: Path, tmp_path: Path, monkeypatch
    ):
        s = _linux_switcher()
        _arm_helper(HELPER_KEY)
        _write_global_config(primaryApiKey=LOGIN_KEY)
        key_file = tmp_path / "fd-key"
        key_file.write_text(FD_KEY + "\n", encoding="utf-8")
        fd = os.open(str(key_file), os.O_RDONLY)
        try:
            monkeypatch.setenv("CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR", str(fd))
            assert s._resolve_live_api_key() == (FD_KEY, True, "fd")
            # pread, not read: the descriptor's owner must still see byte 0.
            assert os.lseek(fd, 0, os.SEEK_CUR) == 0
        finally:
            os.close(fd)

    def test_unreadable_descriptor_is_live_but_unidentified(
        self, temp_home: Path, monkeypatch
    ):
        """A pipe is left alone: reading it would steal the owner's key.

        Live-but-unidentified, so no slot is resolved and nothing is consumed.
        """
        s = _linux_switcher()
        _write_global_config(primaryApiKey=LOGIN_KEY)
        read_fd, write_fd = os.pipe()
        try:
            os.write(write_fd, FD_KEY.encode("utf-8"))
            monkeypatch.setenv("CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR", str(read_fd))
            assert s._resolve_live_api_key() == ("", True, "fd")
            assert s.current_account_number() is None
            assert s.has_live_login() is True
            # Every byte the owner is waiting on is still in the pipe.
            assert os.read(read_fd, 4096).decode("utf-8") == FD_KEY
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_garbage_descriptor_is_live_but_unidentified(
        self, temp_home: Path, monkeypatch
    ):
        s = _linux_switcher()
        _write_global_config(primaryApiKey=LOGIN_KEY)
        monkeypatch.setenv("CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR", "not-a-number")
        assert s._resolve_live_api_key() == ("", True, "fd")

    def test_helper_outranks_the_login_key(self, temp_home: Path):
        s = _linux_switcher()
        _arm_helper(HELPER_KEY)
        _write_global_config(primaryApiKey=LOGIN_KEY)
        assert s._resolve_live_api_key() == (HELPER_KEY, True, "helper")

    def test_foreign_helper_does_not_supply_a_key(self, temp_home: Path):
        """A key file left behind by an older cswap is not what Claude Code reads.

        Ownership is decided by ``settings.json``, not by the key file existing —
        so the leftover file below must be ignored while a foreign helper holds
        the hook, and the ``/login`` key is what is really live.
        """
        s = _linux_switcher()
        _arm_helper(HELPER_KEY)
        assert _helper().key_path.exists()
        settings = _helper().settings_path
        settings.write_text(
            json.dumps({"apiKeyHelper": "/opt/mine/print-key.sh"}), encoding="utf-8"
        )
        _write_global_config(primaryApiKey=LOGIN_KEY)
        assert s._resolve_live_api_key() == (LOGIN_KEY, True, "login")

    def test_login_key_when_no_higher_door_is_armed(self, temp_home: Path):
        s = _linux_switcher()
        _write_global_config(primaryApiKey=LOGIN_KEY)
        assert s._resolve_live_api_key() == (LOGIN_KEY, True, "login")

    def test_oauth_only_is_not_a_live_key(self, temp_home: Path):
        s = _linux_switcher()
        cred_file = get_credentials_path()
        cred_file.parent.mkdir(parents=True, exist_ok=True)
        cred_file.write_text(OAUTH_JSON, encoding="utf-8")
        _write_global_config(oauthAccount={"emailAddress": "me@example.com"})
        assert s._resolve_live_api_key() == ("", False, "none")

    def test_macos_managed_keychain_is_the_login_door(
        self, temp_home: Path, block_real_keychain
    ):
        store = block_real_keychain
        s = _macos_switcher()
        store.set_password(
            CLAUDE_CODE_MANAGED_KEYCHAIN_SERVICE,
            macos_keychain.keychain_account_name(),
            LOGIN_KEY,
        )
        assert s._resolve_live_api_key() == (LOGIN_KEY, True, "login")


class TestStatusAndListWithLiveApiKey:
    """``status`` / ``list`` name the API-key slot instead of ``(not managed)``."""

    def test_status_names_the_slot(self, temp_home: Path, capsys):
        s = _key_slot_switcher({"primaryApiKey": API_KEY})
        s.status()
        out = capsys.readouterr().out
        assert "Account-1" in out
        assert "api-key-1@token.local" in out
        assert "not managed" not in out

    def test_list_marks_the_slot_active(self, temp_home: Path, capsys):
        s = _key_slot_switcher({"primaryApiKey": API_KEY})
        s.list_accounts()
        out = capsys.readouterr().out
        assert "1: api-key-1@token.local" in out
        assert "(active)" in out

    def test_status_json_reports_it_managed(self, temp_home: Path):
        s = _key_slot_switcher({"primaryApiKey": API_KEY})
        payload = s.status(json_output=True)
        assert payload["active"]["managed"] is True
        assert payload["active"]["number"] == 1
        assert payload["active"]["usageStatus"] == "api_key"

    def test_active_slot_is_attributed_to_the_key_not_a_stale_oauth_blob(
        self, temp_home: Path
    ):
        """Usage must read "API key", not a subscription's numbers.

        A key login leaves an OAuth credential on disk. Reading the *store*
        (OAuth-first, by design, for the switch paths) would hand the active row
        another account's token and fetch its quota for a slot billing per token.
        """
        s = _key_slot_switcher({"primaryApiKey": API_KEY})
        cred_file = get_credentials_path()
        cred_file.parent.mkdir(parents=True, exist_ok=True)
        cred_file.write_text(OAUTH_JSON, encoding="utf-8")

        num, _email, _org_name, _org_uuid, is_active, creds, _alias = (
            s._build_accounts_info()[0]
        )
        assert (num, is_active) == (1, True)
        assert creds == API_KEY
        assert s._collect_usage_entries(s._build_accounts_info())["1"].sentinel == (
            USAGE_API_KEY
        )

    def test_unmanaged_live_key_is_reported_without_leaking_it(
        self, temp_home: Path, capsys
    ):
        s = _key_slot_switcher({"primaryApiKey": OTHER_KEY})
        s.status()
        out = capsys.readouterr().out
        assert "not managed" in out
        assert OTHER_KEY not in out
        assert approved_form(OTHER_KEY) not in out
        # ...and the stale OAuth identity is not passed off as the live one.
        assert STALE_OAUTH["emailAddress"] not in out


class _patched_home:
    """Redirect HOME/Path.home() to ``home`` for export/import on two homes."""

    def __init__(self, home: Path):
        self.home = home
        self._patches: list = []

    def __enter__(self):
        import os
        from unittest.mock import patch

        self._patches = [
            patch.dict(os.environ, {"HOME": str(self.home), "USERPROFILE": str(self.home)}),
            patch("pathlib.Path.home", return_value=self.home),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()
        return False
