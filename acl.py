"""ACL enforcement layer for telegram-mcp.

Pure functions — no Telegram client dependency.  Policy is loaded once from
JSON; all checks are synchronous dict lookups.

Public API
----------
acl_check(policy, tool, sender_id, target_chat_id=None) -> AclResult
handle_group_message(policy, chat_id, sender_id, text)   -> Optional[str]
resolve_alias(policy, alias)                             -> int
load_policy(path=None)                                   -> dict
check_command_prefix(policy, text)                       -> bool
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate-limit state for unknown-sender group replies (PRD §10 Q2)
# ---------------------------------------------------------------------------
_RATE_LIMIT_DEFAULT: int = 86400  # 24 hours
_unknown_reply_times: dict[tuple[int, int], float] = {}

# ---------------------------------------------------------------------------
# Tool classification sets (PRD §7.3 + acl-enforcement.md)
# ---------------------------------------------------------------------------

# ADMIN-only state-changing ops (~40 tools)
ACT_TOOLS: frozenset[str] = frozenset(
    {
        # privacy
        "get_privacy_settings",
        "set_privacy_settings",
        # contacts
        "add_contact",
        "delete_contact",
        "import_contacts",
        "export_contacts",
        # group-admin
        "ban_user",
        "unban_user",
        "promote_admin",
        "demote_admin",
        "create_group",
        "create_channel",
        "set_group_description",
        "set_group_title",
        "set_group_username",
        "set_group_photo",
        "delete_group_photo",
        "set_channel_description",
        "set_channel_title",
        "set_channel_username",
        "set_channel_photo",
        "delete_channel_photo",
        "get_admins",
        "get_members",
        # join/leave
        "join_chat_by_link",
        "import_chat_invite",
        "leave_chat",
        "subscribe_public_channel",
        # message-admin
        "delete_message",
        "edit_message",
        "forward_message",
        # profile
        "update_profile",
        "delete_profile_photo",
        "set_profile_photo",
        # block
        "block_user",
        "unblock_user",
        # bot
        "set_bot_commands",
        # chat-state
        "archive_chat",
        "unarchive_chat",
        "mute_chat",
        "unmute_chat",
        # polls
        "create_poll",
        "stop_poll",
        # folders
        "create_folder",
        "delete_folder",
        "update_folder",
        "add_to_folder",
        "remove_from_folder",
        # invite
        "create_invite_link",
        "revoke_invite_link",
        # inline
        "send_inline_message",
    }
)

# Media / file operations
FILES_TOOLS: frozenset[str] = frozenset(
    {
        "download_media",
        "send_file",
        "send_voice",
        "get_media_info",
        "get_user_photos",
    }
)

# Messaging / non-admin write ops
WRITE_TOOLS: frozenset[str] = frozenset(
    {
        "send_message",
        "reply_to_message",
        "send_sticker",
        "send_gif",
        "send_reaction",
        "remove_reaction",
        "save_draft",
        "clear_draft",
        "mark_as_read",
    }
)

# History / listing — denied for trusted contacts
BROAD_READ_TOOLS: frozenset[str] = frozenset(
    {
        "get_history",
        "get_chats",
        "list_chats",
        "search_messages",
        "search_public_chats",
        "get_recent_actions",
        "resolve_username",
        "list_folders",
        "get_folder",
        "get_drafts",
        "get_sticker_sets",
        "get_gif_search",
        "get_bot_info",
        "list_contacts",
        "search_contacts",
        "get_contact_ids",
    }
)

# No target, always allowed — exempt from acl_guard entirely.
# get_me: returns own account info, no sensitive data.
# get_runtime_config: kept here for parity but returns only topology hints
# (proxy enabled/type/host/port, no credentials).  If the MCP transport layer
# is unauthenticated this leaks reconnaissance data; move to ACT_TOOLS and
# gate behind ADMIN if that becomes a concern.
# reload_policy: re-reads secrets/policy.json; must not be gated itself
# (chicken-and-egg after revocation).
SELF_TOOLS: frozenset[str] = frozenset(
    {
        "get_me",
        "get_runtime_config",
        "reload_policy",
    }
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class AclResult:
    allowed: bool
    reason: str = field(default="")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _classify_tool(tool: str) -> str:
    """Return capability label for a tool name."""
    if tool in ACT_TOOLS:
        return "ACT"
    if tool in FILES_TOOLS:
        return "FILES"
    if tool in WRITE_TOOLS:
        return "WRITE"
    if tool in BROAD_READ_TOOLS:
        return "BROAD_READ"
    return "NARROW_READ"


def _classify_sender(policy: dict, sender_id: int) -> str:
    """Return 'ADMIN', 'TRUSTED', or 'UNKNOWN'."""
    if sender_id in policy.get("admin_user_ids", []):
        return "ADMIN"
    if str(sender_id) in policy.get("trusted_contacts", {}):
        return "TRUSTED"
    return "UNKNOWN"


def _is_allowed_channel(policy: dict, target: int) -> bool:
    return target in policy.get("allowed_channel_ids", [])


def _is_allowed_chat(policy: dict, target: int) -> bool:
    return target in policy.get("allowed_chat_ids", [])


def _is_known_target(policy: dict, target: int) -> bool:
    """True if target is a trusted contact, allowed chat, or allowed channel."""
    if str(target) in policy.get("trusted_contacts", {}):
        return True
    if _is_allowed_chat(policy, target):
        return True
    if _is_allowed_channel(policy, target):
        return True
    return False


# ---------------------------------------------------------------------------
# Core check
# ---------------------------------------------------------------------------


def _check_admin(
    policy: dict, capability: str, target_chat_id: Optional[int]
) -> AclResult:
    """ADMIN branch of the decision matrix."""
    if target_chat_id is None:
        return AclResult(allowed=True, reason="admin, no target")

    if not _is_known_target(policy, target_chat_id):
        return AclResult(allowed=False, reason="target not in policy")

    if _is_allowed_channel(policy, target_chat_id) and capability != "NARROW_READ":
        return AclResult(allowed=False, reason="channels are read-only")

    return AclResult(allowed=True, reason="admin action on known target")


def _check_trusted(
    policy: dict, capability: str, sender_id: int, target_chat_id: Optional[int]
) -> AclResult:
    """TRUSTED sender branch of the decision matrix."""
    if capability in ("ACT", "FILES"):
        return AclResult(
            allowed=False,
            reason="ACT/FILES not allowed for trusted contacts",
        )

    if capability == "BROAD_READ":
        return AclResult(
            allowed=False,
            reason="broad reads not allowed for trusted contacts",
        )

    if capability == "WRITE":
        if target_chat_id == sender_id:
            return AclResult(allowed=True, reason="write to own DM")
        if _is_allowed_chat(policy, target_chat_id):
            return AclResult(allowed=True, reason="write in approved group")
        return AclResult(allowed=False, reason="write target not permitted")

    # NARROW_READ (default capability)
    if target_chat_id == sender_id:
        return AclResult(allowed=True, reason="read own DM")
    return AclResult(allowed=False, reason="read target not permitted")


def acl_check(
    policy: dict,
    tool: str,
    sender_id: int,
    target_chat_id: Optional[int] = None,
) -> AclResult:
    """Evaluate whether *tool* is allowed for *sender_id* targeting *target_chat_id*.

    Decision matrix follows PRD §7.3 + acl-enforcement.md verbatim.
    Dispatches to ``_check_admin`` / ``_check_trusted`` after the fast-path
    checks (enforce kill-switch, self-tools, unknown-sender deny).
    """
    # 1. Enforcement kill-switch
    if not policy.get("enforce", True):
        return AclResult(allowed=True, reason="enforcement disabled")

    # 2. Self-info tools — always allowed
    if tool in SELF_TOOLS:
        return AclResult(allowed=True, reason="self-info tool")

    # 3. Classify sender and capability
    role = _classify_sender(policy, sender_id)
    capability = _classify_tool(tool)

    # --- UNKNOWN sender → deny immediately ---
    if role == "UNKNOWN":
        return AclResult(
            allowed=False,
            reason="sender not in admin_user_ids or trusted_contacts",
        )

    # --- role-specific dispatch ---
    if role == "ADMIN":
        return _check_admin(policy, capability, target_chat_id)

    # role == "TRUSTED" (only remaining option after UNKNOWN guard)
    return _check_trusted(policy, capability, sender_id, target_chat_id)


# ---------------------------------------------------------------------------
# Group-message short-circuit
# ---------------------------------------------------------------------------


def handle_group_message(
    policy: dict,
    chat_id: int,
    sender_id: int,
    text: str,
) -> Optional[str]:
    """Return a fixed reply string if the sender should be short-circuited.

    Returns None  → normal processing (ADMIN / trusted contact).
    Returns str   → send this reply, call no tools.  Non-empty = not yet sent;
                    caller is responsible for sending it.
    Returns ""    → rate-limited; caller must stay silent.
    """
    # Not one of our groups → ignore entirely
    if not _is_allowed_chat(policy, chat_id):
        return None

    role = _classify_sender(policy, sender_id)
    if role in ("ADMIN", "TRUSTED"):
        return None

    # Rate-limit: one reply per (chat, sender) per rate_limit window
    now = time.time()
    rate_limit = policy.get("unknown_reply_rate_limit_secs", _RATE_LIMIT_DEFAULT)
    key = (chat_id, sender_id)
    if key in _unknown_reply_times and (now - _unknown_reply_times[key]) < rate_limit:
        return ""  # rate-limited — caller must NOT send anything
    _unknown_reply_times[key] = now

    return policy.get(
        "unknown_group_reply",
        "Sender not recognised. Contact the administrator.",
    )


# ---------------------------------------------------------------------------
# Alias resolution
# ---------------------------------------------------------------------------


def resolve_alias(policy: dict, alias: str) -> int:
    """Look up a trusted contact by alias string.

    Returns the numeric user_id.
    Raises ValueError if no contact matches.
    """
    for uid_str, contact in policy.get("trusted_contacts", {}).items():
        if contact.get("alias") == alias:
            return int(uid_str)
    raise ValueError(f"No trusted contact with alias {alias!r}")


def check_command_prefix(policy: dict, text: str) -> bool:
    """Return True if *text* starts with an allowed command prefix.

    If ``command_prefixes`` is missing or empty the gate is considered open
    (no prefix required) and the function returns True unconditionally.
    """
    prefixes = policy.get("command_prefixes", [])
    if not prefixes:
        return True
    return any(text.strip().startswith(p) for p in prefixes)


# ---------------------------------------------------------------------------
# Policy loader
# ---------------------------------------------------------------------------

_DEFAULT_RELATIVE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "secrets", "policy.json"
)


_DENY_ALL: dict = {
    "enforce": True,
    "admin_user_ids": [],
    "trusted_contacts": {},
    "allowed_chat_ids": [],
    "allowed_channel_ids": [],
}
"""Fail-closed fallback: enforcement ON, every non-SELF tool denied."""


def load_policy(path: Optional[str] = None) -> dict:
    """Load and return the policy dict.

    Resolution order:
      1. Explicit *path* argument
      2. TG_ACL_POLICY_PATH env var
      3. Default: ../../secrets/policy.json relative to this file

    Returns a deny-all policy (enforcement ON, empty allow-lists) on
    FileNotFoundError or JSON decode error so that a missing or corrupt
    config file does **not** silently open the system.
    """
    resolved = path or os.environ.get("TG_ACL_POLICY_PATH") or _DEFAULT_RELATIVE
    try:
        with open(resolved, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        logger.error(
            "ACL policy file not found at %s — using deny-all fallback", resolved
        )
        return dict(_DENY_ALL)
    except json.JSONDecodeError as exc:
        logger.error(
            "ACL policy file is not valid JSON (%s) — using deny-all fallback: %s",
            resolved,
            exc,
        )
        return dict(_DENY_ALL)
