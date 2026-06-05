#!/usr/bin/env python3
"""mkv_renamer.py — TUI/CLI for normalizing anime video filenames.

This tool keeps episode naming consistent across mixed release styles
(different token formats, tags, and numbering gaps). It exposes the same
rename engine through two entry points:

* TUI mode for iterative preview/filter/sort workflows.
* CLI mode for scriptable preview/apply runs.

Design choices:
* Plans are generated first and validated before any filesystem mutation.
* Apply uses a two-phase temp rename to avoid in-folder name collisions.
* Scan results are cached by file signature to keep repeated previews fast.

Usage examples:
    python mkv_renamer.py
    python mkv_renamer.py --path D:/Anime --mode cli --dry-run
    python mkv_renamer.py --path D:/Anime --mode apply --yes
    python mkv_renamer.py --path D:/Anime --mode tui
"""
# pylint: disable=too-many-lines
# pylint: disable=too-many-instance-attributes,too-many-return-statements
# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=too-many-branches,too-many-locals

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, cast

from rich import box
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.prompt import Confirm
from rich.table import Table
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.reactive import reactive
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    RichLog,
    Select,
    Static,
    Switch,
)


VIDEO_EXTENSIONS_DEFAULT = ("*.mkv", "*.mp4", "*.avi")
CONSOLE = Console()

_CLI_STATUS_STYLES = {
    "Pending": "bold bright_cyan",
    "WouldRename": "bold yellow",
    "Renamed": "bold green",
    "Failed": "bold red",
    "Skipped": "bold yellow",
    "Unchanged": "dim",
}
_CLI_ACTION_STYLES = {
    "Movie": "bright_magenta",
    "Skip": "yellow",
    "Format": "cyan",
    "Renumber": "green",
}

_SCAN_CACHE_MAX_ENTRIES = 1000
_SCAN_CACHE_TTL_SECONDS = 14 * 24 * 60 * 60


EpisodeMode = Literal["NumericEpisode", "SpecialEpisode"]
PlanStatus = Literal[
    "Pending", "Skipped", "Unchanged", "WouldRename", "Renamed", "Failed"
]
PlanAction = Literal["Movie", "Skip", "Format", "Renumber"]

_APPLYABLE_STATUSES = {"Pending", "WouldRename"}

EPISODE_PATTERNS = {
    "SeasonEpisode": re.compile(
        r"\bS(?P<Season>\d{1,2})E(?P<Episode>\d{1,3}(?:\.\d+)?)(?:v\d+)?\b", re.I
    ),
    "LeadingEpisode": re.compile(
        r"^(?:E|EP|Episode)\s*(?P<Episode>\d{1,3}(?:\.\d+)?)(?:v\d+)?(?:\b|[-\s_])",
        re.I,
    ),
    "ExplicitEpisode": re.compile(
        r"(?:^|[-\s])(?:E|EP|Episode)\s*(?P<Episode>\d{1,3}(?:\.\d+)?)(?:v\d+)?$", re.I
    ),
    "SpecialEpisode": re.compile(
        (
            r"(?:^|[-\s])(?P<Label>OVA|OAD|SP|SPECIAL|NCOP|NCED|OP|ED|PV)"
            r"\s*(?P<Number>\d{0,3})(?:v\d+)?$"
        ),
        re.I,
    ),
    "BareEpisode": re.compile(
        r"(?:^|[-\s])(?P<Episode>\d{1,3}(?:\.\d+)?)(?:v\d+)?$", re.I
    ),
}


def _empty_rename_plan_items() -> list[RenamePlanItem]:
    """Return a new list for dataclass defaults.

    Using a factory keeps each `FolderPlan` isolated and avoids accidental
    cross-instance sharing from mutable defaults.
    """
    return []


def _empty_folder_plans() -> list[FolderPlan]:
    """Return a new list for dataclass defaults.

    This prevents `ScanResult` instances from sharing the same underlying list.
    """
    return []


def _config_path() -> Path:
    """Return the on-disk path for persisted UI settings."""
    return Path(__file__).with_name(".mkv-renamer-config.json")


def _scan_cache_path() -> Path:
    """Return the on-disk path for cached scan results."""
    return Path(__file__).with_name(".mkv-renamer-scan-cache.json")


def _save_json(path: Path, payload: dict[str, object]) -> None:
    """Write a JSON object to disk, ignoring best-effort file errors."""
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        # Config/cache writes are intentionally non-fatal so a temporary
        # permission or lock issue does not block the main rename flow.
        pass


def _load_json(path: Path) -> dict[str, object]:
    """Load a JSON object from disk or return an empty mapping."""
    try:
        obj: object = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            # JSON keys can be non-string after deserialization in loose
            # data, so this normalizes keys to strings for type stability.
            raw = cast(dict[object, object], obj)
            out: dict[str, object] = {}
            for key, value in raw.items():
                out[str(key)] = value
            return out
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _load_config() -> dict[str, object]:
    """Load the saved TUI settings payload."""
    return _load_json(_config_path())


def _save_config(cfg: dict[str, object]) -> None:
    """Persist the TUI settings payload."""
    _save_json(_config_path(), cfg)


def _item_to_payload(item: RenamePlanItem) -> dict[str, object]:
    """Serialize one plan item into cache-friendly JSON data."""
    return {
        "folder_path": item.folder_path,
        "series_name": item.series_name,
        "original_name": item.original_name,
        "original_path": item.original_path,
        "extension": item.extension,
        "detected_type": item.detected_type,
        "original_token": item.original_token,
        "target_token": item.target_token,
        "target_name": item.target_name,
        "target_path": item.target_path,
        "action": item.action,
        "will_rename": item.will_rename,
        "will_renumber": item.will_renumber,
        "status": item.status,
        "reason": item.reason,
    }


def _item_from_payload(payload: dict[str, object]) -> RenamePlanItem:
    """Restore one plan item from cached JSON data."""
    return RenamePlanItem(
        folder_path=str(payload.get("folder_path", "")),
        series_name=str(payload.get("series_name", "")),
        original_name=str(payload.get("original_name", "")),
        original_path=str(payload.get("original_path", "")),
        extension=str(payload.get("extension", "")),
        detected_type=str(payload.get("detected_type", "Unknown")),
        original_token=str(payload.get("original_token", "")),
        target_token=str(payload.get("target_token", "")),
        target_name=str(payload.get("target_name", "")),
        target_path=str(payload.get("target_path", "")),
        action=cast(PlanAction, str(payload.get("action", "Skip"))),
        will_rename=bool(payload.get("will_rename", False)),
        will_renumber=bool(payload.get("will_renumber", False)),
        status=cast(PlanStatus, str(payload.get("status", "Pending"))),
        reason=str(payload.get("reason", "")),
    )


def _scan_result_to_payload(result: ScanResult) -> dict[str, object]:
    """Serialize a scan result into a JSON-friendly payload."""
    folders: list[dict[str, object]] = []
    for folder in result.folder_plans:
        folders.append(
            {
                "folder_path": folder.folder_path,
                "series_name": folder.series_name,
                "status": folder.status,
                "items": [_item_to_payload(item) for item in folder.items],
            }
        )
    return {"media_path": str(result.media_path), "folder_plans": folders}


def _scan_result_from_payload(payload: dict[str, object]) -> ScanResult | None:
    """Restore a scan result from cached JSON data when valid."""
    media_path = Path(str(payload.get("media_path", ".")))
    folders_raw = payload.get("folder_plans")
    if not isinstance(folders_raw, list):
        return None

    folder_plans: list[FolderPlan] = []
    for folder_obj in cast(list[object], folders_raw):
        if not isinstance(folder_obj, dict):
            continue
        folder_map = cast(dict[str, object], folder_obj)
        items_raw = folder_map.get("items")
        items: list[RenamePlanItem] = []
        if isinstance(items_raw, list):
            for item_obj in cast(list[object], items_raw):
                if isinstance(item_obj, dict):
                    items.append(_item_from_payload(cast(dict[str, object], item_obj)))
        folder_plans.append(
            FolderPlan(
                folder_path=str(folder_map.get("folder_path", "")),
                series_name=str(folder_map.get("series_name", "")),
                items=items,
                status=str(folder_map.get("status", "UNCHANGED")),
            )
        )
    return ScanResult(media_path=media_path, folder_plans=folder_plans)


def _scan_signature(file_records: list[FileRecord]) -> str:
    """Build a fast-change signature from path, size, and modified time.

    This intentionally avoids file-content hashing so large media libraries
    can be rescanned quickly while still invalidating stale cache entries.
    """
    # A SHA-1 digest is used here as a fast change fingerprint, not for
    # security; collisions are acceptable for a best-effort cache key.
    h = hashlib.sha1()
    for record in file_records:
        path = record.file
        try:
            stat = path.stat()
            token = f"{path}|{stat.st_size}|{stat.st_mtime_ns}\n"
        except OSError:
            token = f"{path}|missing\n"
        h.update(token.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def _scan_cache_key(
    media_path: Path,
    extensions: Iterable[str],
    recursive: bool,
    renumber_enabled: bool,
) -> str:
    """Build the cache key for a specific scan configuration."""
    ext_key = ",".join(sorted(e.lower() for e in extensions))
    return (
        f"{media_path.resolve()}|{ext_key}|r={int(recursive)}|n={int(renumber_enabled)}"
    )


def _compact_scan_cache_payload(cache_payload: dict[str, object]) -> bool:
    """Prune stale/invalid cache entries and enforce the entry cap.

    Keeping this logic in one place avoids scattered cache assumptions and
    ensures read/write paths share the same retention behavior.
    """
    entries_obj = cache_payload.get("entries")
    if not isinstance(entries_obj, dict):
        if entries_obj is None:
            return False
        cache_payload["entries"] = {}
        return True

    entries = cast(dict[str, object], entries_obj)
    now_ts = int(time.time())
    changed = False

    valid_entries: dict[str, object] = {}
    for cache_key, entry_obj in entries.items():
        if not isinstance(entry_obj, dict):
            changed = True
            continue
        entry = cast(dict[str, object], entry_obj)
        raw_saved_at = entry.get("saved_at")
        if isinstance(raw_saved_at, (int, float)):
            saved_at = int(raw_saved_at)
        elif isinstance(raw_saved_at, str) and raw_saved_at.isdigit():
            saved_at = int(raw_saved_at)
            entry["saved_at"] = saved_at
            changed = True
        else:
            saved_at = now_ts
            entry["saved_at"] = saved_at
            changed = True

        if now_ts - saved_at > _SCAN_CACHE_TTL_SECONDS:
            changed = True
            continue
        valid_entries[cache_key] = entry

    if len(valid_entries) > _SCAN_CACHE_MAX_ENTRIES:

        def _entry_saved_at(entry_obj: object) -> int:
            if not isinstance(entry_obj, dict):
                return 0
            entry_map = cast(dict[str, object], entry_obj)
            raw = entry_map.get("saved_at", 0)
            if isinstance(raw, (int, float)):
                return int(raw)
            if isinstance(raw, str) and raw.isdigit():
                return int(raw)
            return 0

        sorted_keys = sorted(
            valid_entries,
            key=lambda key: _entry_saved_at(valid_entries.get(key)),
            reverse=True,
        )
        keep_keys = set(sorted_keys[:_SCAN_CACHE_MAX_ENTRIES])
        valid_entries = {
            key: value for key, value in valid_entries.items() if key in keep_keys
        }
        changed = True

    if set(valid_entries) != set(entries):
        changed = True
    if changed:
        cache_payload["entries"] = valid_entries
    return changed


@dataclass(slots=True)
class EpisodeInfo:
    """Normalized episode metadata extracted from a file name."""

    type: EpisodeMode
    token: str
    sort_group: int
    sort_value: float
    raw_token: str


@dataclass(slots=True)
class FileRecord:
    """Source file plus the metadata needed for planning."""

    file: Path
    folder_path: str
    series_name: str
    extension: str
    episode_info: EpisodeInfo | None
    renumber_token: str | None = None


@dataclass(slots=True)
class RenamePlanItem:
    """One planned rename action for a single file."""

    folder_path: str
    series_name: str
    original_name: str
    original_path: str
    extension: str
    detected_type: str
    original_token: str
    target_token: str
    target_name: str
    target_path: str
    temp_name: str | None = None
    temp_path: str | None = None
    action: PlanAction = "Skip"
    will_rename: bool = False
    will_renumber: bool = False
    status: PlanStatus = "Pending"
    reason: str = ""


@dataclass(slots=True)
class FolderPlan:
    """All rename decisions for one folder."""

    folder_path: str
    series_name: str
    items: list[RenamePlanItem] = field(default_factory=_empty_rename_plan_items)
    status: str = "UNCHANGED"


@dataclass(slots=True)
class ScanResult:
    """Top-level scan output for the current path and options."""

    media_path: Path
    folder_plans: list[FolderPlan] = field(default_factory=_empty_folder_plans)

    @property
    def items(self) -> list[RenamePlanItem]:
        """Return all plan items flattened across every folder."""
        return [item for folder in self.folder_plans for item in folder.items]


def remove_leading_release_tags(value: str) -> str:
    """Remove release-group tags from the start of a name fragment."""
    result = value.strip()
    pattern = re.compile(r"^\s*(\[[^\]]+\]|\([^\)]+\)|\{[^\}]+\})\s*")
    while True:
        updated = pattern.sub("", result).strip()
        if updated == result:
            break
        result = updated
    return result


def remove_trailing_release_tags(value: str) -> str:
    """Remove release-group tags from the end of a name fragment."""
    result = value.strip()
    pattern = re.compile(r"\s*(\[[^\]]+\]|\([^\)]+\)|\{[^\}]+\})\s*$")
    while True:
        updated = pattern.sub("", result).strip()
        if updated == result:
            break
        result = updated
    return result


def format_padded_episode_part(episode_value: str) -> str:
    """Pad the integer part of an episode number while keeping fractions."""
    match = re.fullmatch(r"(?P<whole>\d+)(?P<fraction>\.\d+)?", episode_value)
    if not match:
        return episode_value
    whole = f"{int(match.group('whole')):02d}"
    fraction = match.group("fraction") or ""
    return f"{whole}{fraction}"


def new_numeric_episode_token(episode_value: str) -> str:
    """Build the normalized numeric episode token like `E03`."""
    return f"E{format_padded_episode_part(episode_value)}"


def new_special_episode_token(label: str, number_value: str = "") -> str:
    """Build the normalized token used for specials and extras."""
    normalized_label = label.upper()
    if not number_value.strip():
        return normalized_label
    return f"{normalized_label}{int(number_value):02d}"


def get_episode_info(filename: str) -> EpisodeInfo | None:
    """Extract a normalized episode token from a filename stem.

    Matching intentionally follows a strict precedence order so ambiguous names
    resolve predictably (season/episode first, bare trailing numbers last).
    Returns ``None`` when no trustworthy token is found so callers can skip
    risky renames instead of guessing.
    """
    working_name = re.sub(r"[._]+", " ", filename)
    working_name = remove_leading_release_tags(working_name)
    working_name = remove_trailing_release_tags(working_name)
    working_name = re.sub(r"\s+", " ", working_name).strip()
    if not working_name:
        return None

    if match := EPISODE_PATTERNS["SeasonEpisode"].search(working_name):
        episode = match.group("Episode")
        return EpisodeInfo(
            "NumericEpisode",
            new_numeric_episode_token(episode),
            1,
            float(episode),
            match.group(0),
        )
    if match := EPISODE_PATTERNS["LeadingEpisode"].search(working_name):
        episode = match.group("Episode")
        return EpisodeInfo(
            "NumericEpisode",
            new_numeric_episode_token(episode),
            2,
            float(episode),
            match.group(0),
        )
    if match := EPISODE_PATTERNS["ExplicitEpisode"].search(working_name):
        episode = match.group("Episode")
        return EpisodeInfo(
            "NumericEpisode",
            new_numeric_episode_token(episode),
            2,
            float(episode),
            match.group(0),
        )
    if match := EPISODE_PATTERNS["SpecialEpisode"].search(working_name):
        number_value = match.group("Number") or ""
        sort_number = int(number_value) if number_value.strip() else 0
        return EpisodeInfo(
            "SpecialEpisode",
            new_special_episode_token(match.group("Label"), number_value),
            3,
            float(sort_number),
            match.group(0),
        )
    if match := EPISODE_PATTERNS["BareEpisode"].search(working_name):
        episode = match.group("Episode")
        return EpisodeInfo(
            "NumericEpisode",
            new_numeric_episode_token(episode),
            2,
            float(episode),
            match.group(0),
        )
    return None


def format_episode_filename(
    series_name: str, episode_token: str, extension: str
) -> str:
    """Return the final filename expected for one media item."""
    return (
        f"{series_name}{extension}"
        if not episode_token
        else f"{series_name} - {episode_token}{extension}"
    )


def get_supported_extension_set(extensions: Iterable[str]) -> set[str]:
    """Normalize extension globs into lowercase file suffixes."""
    normalized: set[str] = set()
    for extension in extensions:
        if not extension or not extension.strip():
            continue
        token = extension.strip()
        if token.startswith("*."):
            token = f".{token[2:]}"
        elif not token.startswith("."):
            token = f".{token.lstrip('*')}"
        normalized.add(token.lower())
    return normalized


def get_video_files(
    root_path: Path, extension_set: set[str], recursive: bool = True
) -> list[Path]:
    """Collect matching video files from the selected root path."""
    if not root_path.exists():
        return []
    iterator = root_path.rglob("*") if recursive else root_path.iterdir()
    files = [
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in extension_set
    ]
    return sorted(files, key=lambda path: str(path).lower())


def new_file_record(file_object: Path) -> FileRecord:
    """Create the scan-time record for one media file."""
    episode_info = get_episode_info(file_object.stem)
    return FileRecord(
        file=file_object,
        folder_path=str(file_object.parent),
        series_name=file_object.parent.name.strip(),
        extension=file_object.suffix,
        episode_info=episode_info,
    )


def new_rename_plan_item(
    file_record: FileRecord,
    target_name: str,
    target_token: str,
    action: PlanAction,
    will_renumber: bool,
    status: PlanStatus,
    reason: str,
) -> RenamePlanItem:
    """Create one rename-plan row from a scanned file record."""
    return RenamePlanItem(
        folder_path=file_record.folder_path,
        series_name=file_record.series_name,
        original_name=file_record.file.name,
        original_path=str(file_record.file),
        extension=file_record.extension,
        detected_type=file_record.episode_info.type
        if file_record.episode_info
        else "Unknown",
        original_token=file_record.episode_info.token
        if file_record.episode_info
        else "",
        target_token=target_token,
        target_name=target_name,
        target_path=str(file_record.file.with_name(target_name)),
        action=action,
        will_rename=file_record.file.name != target_name,
        will_renumber=will_renumber,
        status=status,
        reason=reason,
    )


def describe_plan_reason(
    file_record: FileRecord, target_token: str, action: PlanAction
) -> str:
    """Return a human-readable explanation for a planned rename."""
    if action == "Movie":
        return "Rename movie to match the folder title."

    if not file_record.episode_info:
        return "Rename to match the expected series format."

    original_token = file_record.episode_info.token
    if action == "Renumber" and target_token and target_token != original_token:
        return f"Renumber in folder order: {original_token} -> {target_token}."
    if target_token and target_token != original_token:
        return f"Normalize episode token: {original_token} -> {target_token}."
    if target_token:
        return f"Format filename using episode token {target_token}."
    return "Rename to match the expected series format."


def is_applyable_item(item: RenamePlanItem) -> bool:
    """Return whether a plan row should be applied during the rename pass."""
    return item.status in _APPLYABLE_STATUSES and item.will_rename


def get_plan_counts(plan_items: Iterable[RenamePlanItem]) -> dict[str, int]:
    """Summarize plan items into changed, skipped, and failed counts."""
    items = list(plan_items)
    return {
        "Changed": sum(
            1
            for item in items
            if item.will_rename and item.status not in {"Skipped", "Failed"}
        ),
        "Renumbered": sum(
            1
            for item in items
            if item.will_rename
            and item.will_renumber
            and item.status not in {"Skipped", "Failed"}
        ),
        "Unchanged": sum(
            1 for item in items if not item.will_rename and item.status == "Unchanged"
        ),
        "Skipped": sum(1 for item in items if item.status == "Skipped"),
        "Failed": sum(1 for item in items if item.status == "Failed"),
    }


def get_folder_status_label(plan_items: Iterable[RenamePlanItem]) -> str:
    """Collapse plan counts into a short folder status label."""
    counts = get_plan_counts(plan_items)
    if counts["Renumbered"] > 0:
        return "RENUMBER"
    if counts["Changed"] > 0:
        return "FORMAT"
    if counts["Skipped"] > 0:
        return "SKIP"
    return "UNCHANGED"


def new_folder_plan(
    folder_path: str, file_records: list[FileRecord], renumber_enabled: bool
) -> FolderPlan:
    """Build rename actions for one folder using safe fallback heuristics.

    Folder-level planning keeps decisions local so mixed series in other
    folders cannot affect numbering or conflict checks here.

    Heuristics:
    * Single unparsed file -> treat as a movie and use folder title.
    * Multi-file folder with no tokens -> skip, avoid risky guesses.
    * Parsed episodes -> format names and optionally renumber in sort order.
    """
    sorted_files = sorted(
        file_records,
        key=lambda record: (
            record.episode_info.sort_group if record.episode_info else 9,
            record.episode_info.sort_value if record.episode_info else float("inf"),
            record.file.name.lower(),
        ),
    )
    series_name = (
        sorted_files[0].series_name if sorted_files else Path(folder_path).name
    )
    parsed_count = sum(1 for record in sorted_files if record.episode_info)
    plans: list[RenamePlanItem] = []

    if len(sorted_files) == 1 and parsed_count == 0:
        single_file = sorted_files[0]
        target_name = format_episode_filename(
            single_file.series_name, "", single_file.extension
        )
        plans.append(
            new_rename_plan_item(
                single_file,
                target_name,
                "",
                "Movie",
                False,
                "Pending",
                describe_plan_reason(single_file, "", "Movie"),
            )
        )
    elif parsed_count == 0:
        for file_record in sorted_files:
            plans.append(
                new_rename_plan_item(
                    file_record,
                    file_record.file.name,
                    "",
                    "Skip",
                    False,
                    "Skipped",
                    "Multi-file folder without detectable episode tokens.",
                )
            )
    else:
        if renumber_enabled:
            renumber_index = 1
            for file_record in sorted_files:
                if file_record.episode_info:
                    file_record.renumber_token = new_numeric_episode_token(
                        str(renumber_index)
                    )
                    renumber_index += 1

        for file_record in sorted_files:
            if not file_record.episode_info:
                plans.append(
                    new_rename_plan_item(
                        file_record,
                        file_record.file.name,
                        "",
                        "Skip",
                        False,
                        "Skipped",
                        "Could not detect an episode token.",
                    )
                )
                continue

            target_token = (
                file_record.renumber_token
                if renumber_enabled
                else file_record.episode_info.token
            )
            action: PlanAction = "Renumber" if renumber_enabled else "Format"
            target_name = format_episode_filename(
                file_record.series_name, target_token or "", file_record.extension
            )
            plans.append(
                new_rename_plan_item(
                    file_record,
                    target_name,
                    target_token or "",
                    action,
                    renumber_enabled,
                    "Pending",
                    describe_plan_reason(file_record, target_token or "", action),
                )
            )

    for plan in plans:
        if plan.status == "Pending" and not plan.will_rename:
            plan.status = "Unchanged"
            plan.reason = "Already matches the expected name."

    folder_plan = FolderPlan(
        folder_path=folder_path, series_name=series_name, items=plans
    )
    folder_plan.status = get_folder_status_label(folder_plan.items)
    return folder_plan


def validate_folder_plan(folder_plan: FolderPlan) -> FolderPlan:
    """Mark rename rows unsafe to apply as `Skipped`.

    Validation is a separate pass so plan generation can stay readable and
    safety checks remain explicit and testable.
    """
    change_items = [
        item
        for item in folder_plan.items
        if item.status == "Pending" and item.will_rename
    ]
    duplicate_targets: dict[str, list[RenamePlanItem]] = defaultdict(list)
    for item in change_items:
        duplicate_targets[item.target_name.lower()].append(item)

    for duplicate_group in duplicate_targets.values():
        if len(duplicate_group) < 2:
            continue
        for item in duplicate_group:
            item.status = "Skipped"
            item.reason = f"Duplicate target name would be created: {item.target_name}"

    active_change_items = [
        item
        for item in folder_plan.items
        if item.status == "Pending" and item.will_rename
    ]
    source_paths = {item.original_path.lower() for item in active_change_items}
    for item in active_change_items:
        if (
            Path(item.target_path).exists()
            and item.target_path.lower() not in source_paths
        ):
            item.status = "Skipped"
            item.reason = "Target path already exists outside this rename set."

    folder_plan.status = get_folder_status_label(folder_plan.items)
    return folder_plan


def collect_file_records(
    media_path: Path, extensions: Iterable[str], recursive: bool = True
) -> list[FileRecord]:
    """Scan the filesystem and convert matches into file records."""
    extension_set = get_supported_extension_set(extensions)
    video_files = get_video_files(media_path, extension_set, recursive=recursive)
    return [new_file_record(path) for path in video_files]


def group_file_records(
    file_records: list[FileRecord],
) -> list[tuple[str, list[FileRecord]]]:
    """Group records by parent folder with deterministic folder ordering.

    Sorting by folder path keeps scans stable across runs, which helps both
    previews and cache signatures remain predictable.
    """
    grouped: dict[str, list[FileRecord]] = defaultdict(list)
    for record in file_records:
        grouped[record.folder_path].append(record)
    return [(folder_path, grouped[folder_path]) for folder_path in sorted(grouped)]


def build_scan_result(
    media_path: Path,
    extensions: Iterable[str],
    recursive: bool = True,
    renumber_enabled: bool = True,
    folder_callback: Callable[[FolderPlan, int, int], None] | None = None,
) -> ScanResult:
    """Build a full plan snapshot for the current scan settings.

    The optional ``folder_callback`` is invoked as each folder completes so the
    TUI can stream incremental progress instead of waiting for a full-library
    scan to finish.
    """
    file_records = collect_file_records(media_path, extensions, recursive=recursive)
    grouped_records = group_file_records(file_records)

    folder_plans: list[FolderPlan] = []
    total_folders = len(grouped_records)
    for folder_index, (folder_path, records) in enumerate(grouped_records, start=1):
        folder_plan = validate_folder_plan(
            new_folder_plan(folder_path, records, renumber_enabled=renumber_enabled)
        )
        folder_plans.append(folder_plan)
        if folder_callback is not None:
            folder_callback(folder_plan, folder_index, total_folders)

    return ScanResult(media_path=media_path, folder_plans=folder_plans)


def invoke_two_phase_rename(plan_items: list[RenamePlanItem]) -> None:
    """Apply pending renames using a temp hop, then finalize target names.

    Why two phases: direct A->B renames can fail when B already exists in the
    same batch (for example swapping names). Temp names make the operation
    collision-safe and allow best-effort rollback on failure.
    """
    pending_changes = [item for item in plan_items if is_applyable_item(item)]
    if not pending_changes:
        return

    phase_one_completed: list[RenamePlanItem] = []
    for item in pending_changes:
        item.temp_name = f".rename-temp-{os.urandom(8).hex()}{item.extension}"
        item.temp_path = str(Path(item.folder_path) / item.temp_name)

    try:
        for item in pending_changes:
            assert item.temp_path is not None
            Path(item.original_path).rename(item.temp_path)
            phase_one_completed.append(item)

        for item in pending_changes:
            assert item.temp_path is not None
            Path(item.temp_path).rename(Path(item.folder_path) / item.target_name)
            item.status = "Renamed"
    except OSError as exc:
        failure_message = str(exc)
        for item in pending_changes:
            if item.status in _APPLYABLE_STATUSES:
                item.status = "Failed"
                item.reason = failure_message

        for item in phase_one_completed:
            if (
                item.temp_path
                and Path(item.temp_path).exists()
                and not Path(item.original_path).exists()
            ):
                try:
                    Path(item.temp_path).rename(item.original_path)
                except OSError as rollback_exc:
                    item.status = "Failed"
                    item.reason = f"Rollback failed: {rollback_exc}"


def _cli_bool_label(flag: bool) -> str:
    """Return a colored yes/no label for CLI summary panels."""
    return "[green]Yes[/green]" if flag else "[red]No[/red]"


def _cli_info_panel(
    title: str, rows: list[tuple[str, str]], border_style: str
) -> Panel:
    """Build a small key/value panel for CLI headers and summaries."""
    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(style="bold white", no_wrap=True)
    grid.add_column(style="white", ratio=1)
    for label, value in rows:
        grid.add_row(f"{label}", value)
    return Panel(
        grid,
        title=title,
        border_style=border_style,
        box=box.ROUNDED,
        padding=(0, 1),
    )


def _cli_metric_panel(
    title: str,
    value: str,
    border_style: str,
    subtitle: str = "",
) -> Panel:
    """Build a compact metric panel for CLI summary totals."""
    body = Text(justify="center")
    body.append(f"{value}\n", style=f"bold {border_style}")
    if subtitle:
        body.append(subtitle, style="dim")
    return Panel(
        body,
        title=title,
        border_style=border_style,
        box=box.ROUNDED,
        padding=(0, 1),
    )


def _cli_build_notice_panel(
    message: str,
    border_style: str,
    *,
    title: str | None = None,
) -> Panel:
    """Build a compact notice panel for live CLI updates."""
    return Panel(
        message,
        title=title,
        border_style=border_style,
        box=box.ROUNDED,
        padding=(0, 1),
    )


def _cli_build_operation_progress() -> Progress:
    """Build a cleaner-style counted progress display for renamer CLI runs."""
    return Progress(
        SpinnerColumn(style="bold cyan", finished_text="✓"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(
            bar_width=None,
            complete_style="bright_magenta",
            finished_style="magenta",
        ),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.fields[details]}", style="white"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=CONSOLE,
        transient=False,
        expand=True,
    )


def _effective_plan_status(plan_item: RenamePlanItem, dry_run: bool) -> PlanStatus:
    """Return the display status without mutating the underlying plan item."""
    if dry_run and plan_item.status == "Pending" and plan_item.will_rename:
        return "WouldRename"
    return plan_item.status


def _cli_display_counts(folder_plan: FolderPlan, dry_run: bool) -> dict[str, int]:
    """Return folder counts using display statuses for the current phase."""
    status_counts = Counter(
        _effective_plan_status(item, dry_run) for item in folder_plan.items
    )
    plan_counts = get_plan_counts(folder_plan.items)
    return {
        "Files": len(folder_plan.items),
        "Changed": plan_counts["Changed"],
        "Renumbered": plan_counts["Renumbered"],
        "Unchanged": plan_counts["Unchanged"],
        "Skipped": status_counts["Skipped"],
        "Failed": status_counts["Failed"],
    }


def _cli_status_badge(status: PlanStatus) -> Text:
    """Return a styled status badge for one plan row."""
    label_map = {
        "WouldRename": "DRY-RUN",
        "Renamed": "DONE",
        "Failed": "ERROR",
        "Skipped": "SKIP",
        "Unchanged": "UNCHANGED",
        "Pending": "PENDING",
    }
    label = label_map.get(status, status.upper())
    return Text(f" {label} ", style=_CLI_STATUS_STYLES.get(status, "white"))


def _cli_action_badge(action: PlanAction) -> Text:
    """Return a styled action badge for one plan row."""
    return Text(f" {action.upper()} ", style=_CLI_ACTION_STYLES.get(action, "white"))


def _cli_build_folder_metrics(folder_plan: FolderPlan, dry_run: bool) -> Table:
    """Build compact folder metrics for the live current-folder panel."""
    counts = _cli_display_counts(folder_plan, dry_run)
    grid = Table.grid(expand=True)
    for _ in range(3):
        grid.add_column(ratio=1)
    grid.add_row(
        _cli_metric_panel("Files", str(counts["Files"]), "bright_blue"),
        _cli_metric_panel("Changed", str(counts["Changed"]), "bright_green"),
        _cli_metric_panel("Renumbered", str(counts["Renumbered"]), "green"),
    )
    grid.add_row(
        _cli_metric_panel("Unchanged", str(counts["Unchanged"]), "bright_black"),
        _cli_metric_panel("Skipped", str(counts["Skipped"]), "yellow"),
        _cli_metric_panel("Failed", str(counts["Failed"]), "red"),
    )
    return grid


def _cli_build_folder_items_table(folder_plan: FolderPlan, dry_run: bool) -> Table:
    """Build the per-file table for one folder in the live CLI view."""
    table = Table(box=box.HEAVY_HEAD, show_lines=True, expand=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Action", no_wrap=True)
    table.add_column("Original", overflow="fold")
    table.add_column("Target", overflow="fold")
    table.add_column("Reason", overflow="fold")

    for item in folder_plan.items:
        display_status = _effective_plan_status(item, dry_run)
        table.add_row(
            _cli_status_badge(display_status),
            _cli_action_badge(item.action),
            item.original_name,
            item.target_name if item.will_rename else "—",
            item.reason or "—",
        )
    return table


def _cli_build_current_folder_panel(
    *,
    phase_label: str,
    folder_index: int,
    total_folders: int,
    folder_plan: FolderPlan | None,
    dry_run: bool,
) -> Panel:
    """Build the single current-folder live panel for CLI scan/apply runs."""
    header = Text(style="dim")
    if folder_plan is None:
        header.append(f"{phase_label} 0/{total_folders} • waiting for first folder")
        body: list[RenderableType] = [
            header,
            _cli_build_notice_panel(
                "[dim]No folder details available yet.[/dim]",
                "bright_black",
            ),
        ]
    else:
        header.append(f"{phase_label} {folder_index}/{total_folders} • ")
        header.append(folder_plan.series_name, style="bold white")
        header.append(f"  ({folder_plan.status})", style="bold magenta")
        counts = _cli_display_counts(folder_plan, dry_run)
        body = [header, _cli_build_folder_metrics(folder_plan, dry_run)]
        if counts["Changed"] == 0 and counts["Skipped"] == 0 and counts["Failed"] == 0:
            body.append(
                _cli_build_notice_panel(
                    "[bold green]Everything in this folder already matches the expected naming scheme.[/bold green]",
                    "green",
                )
            )
        else:
            body.append(_cli_build_folder_items_table(folder_plan, dry_run))

    return Panel(
        Group(*body),
        title="[bold white]Current Folder[/bold white]",
        border_style="bright_blue",
        box=box.DOUBLE,
        padding=(0, 1),
    )


def _render_cli_runtime_header(
    media_path: Path,
    extensions: list[str],
    recursive: bool,
    renumber_enabled: bool,
    dry_run: bool,
    mode: str,
    *,
    folder_count: int,
    file_count: int,
) -> None:
    """Render a run header before the live CLI progress begins."""
    header_grid = Table.grid(expand=True)
    header_grid.add_column(ratio=2)
    header_grid.add_column(ratio=1)
    header_grid.add_column(ratio=1)
    header_grid.add_row(
        _cli_info_panel(
            "Path",
            [
                ("Root", str(media_path)),
                ("Extensions", ", ".join(extensions or list(VIDEO_EXTENSIONS_DEFAULT))),
            ],
            "cyan",
        ),
        _cli_info_panel(
            "Run",
            [
                ("Mode", mode.upper()),
                ("Recursive", _cli_bool_label(recursive)),
                ("Renumber", _cli_bool_label(renumber_enabled)),
                ("Dry-run", _cli_bool_label(dry_run)),
            ],
            "magenta",
        ),
        _cli_info_panel(
            "Inventory",
            [
                ("Folders", str(folder_count)),
                ("Files", str(file_count)),
                ("Current", "pending"),
            ],
            "green",
        ),
    )
    CONSOLE.print(
        Panel(
            header_grid,
            title="[bold bright_white]Anime File Renamer[/bold bright_white]",
            border_style="bright_blue",
            box=box.DOUBLE,
            padding=(0, 1),
        )
    )


def _render_folder_breakdown(folder_plans: list[FolderPlan]) -> None:
    """Render a per-folder summary table for the CLI preview."""
    table = Table(title="Folder Breakdown", box=box.HEAVY_HEAD, show_lines=True)
    table.add_column("Folder", style="bold white", overflow="fold")
    table.add_column("Status", style="bold magenta", no_wrap=True)
    table.add_column("Changed", justify="right", style="cyan")
    table.add_column("Renumbered", justify="right", style="green")
    table.add_column("Unchanged", justify="right", style="dim")
    table.add_column("Skipped", justify="right", style="yellow")
    for folder_plan in folder_plans:
        counts = get_plan_counts(folder_plan.items)
        table.add_row(
            folder_plan.series_name,
            folder_plan.status,
            str(counts["Changed"]),
            str(counts["Renumbered"]),
            str(counts["Unchanged"]),
            str(counts["Skipped"]),
        )
    CONSOLE.print(table)


def _render_plan_items_table(scan_result: ScanResult, dry_run: bool) -> None:
    """Render the detailed per-file action table for CLI runs."""
    table = Table(title="File Actions", box=box.HEAVY_HEAD, show_lines=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Action", no_wrap=True)
    table.add_column("Original", style="white", overflow="fold")
    table.add_column("Target", style="cyan", overflow="fold")
    table.add_column("Folder", style="dim", overflow="fold")
    table.add_column("Reason", style="white", overflow="fold")

    for item in scan_result.items:
        status = _effective_plan_status(item, dry_run)
        target = item.target_name if item.will_rename else "—"
        table.add_row(
            _cli_status_badge(status),
            _cli_action_badge(item.action),
            item.original_name,
            target,
            item.series_name,
            item.reason or "—",
        )

    CONSOLE.print(table)


def _render_cli_summary(scan_result: ScanResult) -> None:
    """Render summary metrics for CLI preview/apply runs."""
    all_items = scan_result.items
    counts = get_plan_counts(all_items)
    status_counts = Counter(item.status for item in all_items)
    metrics = [
        _cli_metric_panel("Files", str(len(all_items)), "bright_cyan"),
        _cli_metric_panel("Changed", str(counts["Changed"]), "bright_green"),
        _cli_metric_panel("Renumbered", str(counts["Renumbered"]), "green"),
        _cli_metric_panel("Unchanged", str(counts["Unchanged"]), "bright_black"),
        _cli_metric_panel("Skipped", str(counts["Skipped"]), "yellow"),
        _cli_metric_panel("Failed", str(status_counts["Failed"]), "red"),
    ]
    CONSOLE.print(
        Panel(
            Group(*metrics),
            title="Summary",
            border_style="bright_blue",
            box=box.DOUBLE,
            padding=(0, 1),
        )
    )


def render_cli_plan(scan_result: ScanResult, dry_run: bool) -> None:
    """Render the CLI preview or applied-results report."""
    all_items = scan_result.items
    if not all_items:
        CONSOLE.print(
            Panel(
                "[bold yellow]No files found to process.[/bold yellow]",
                border_style="yellow",
                box=box.ROUNDED,
            )
        )
        return

    counts = get_plan_counts(all_items)
    if counts["Changed"] == 0 and counts["Skipped"] == 0:
        CONSOLE.print(
            Panel(
                "[bold green]Everything already matches the expected naming scheme.[/bold green]",
                border_style="green",
                box=box.ROUNDED,
            )
        )
        _render_cli_summary(scan_result)
        return

    _render_folder_breakdown(scan_result.folder_plans)
    _render_plan_items_table(scan_result, dry_run)
    _render_cli_summary(scan_result)


def print_preview_table(scan_result: ScanResult) -> None:
    """Print a compact preview table before detailed per-folder output."""
    table = Table(title="Rename Preview", box=box.HEAVY_HEAD, show_lines=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Action", no_wrap=True)
    table.add_column("Original", overflow="fold")
    table.add_column("Target", overflow="fold")
    table.add_column("Folder", overflow="fold")
    table.add_column("Reason", overflow="fold")
    for item in scan_result.items:
        table.add_row(
            _cli_status_badge(item.status),
            _cli_action_badge(item.action),
            item.original_name,
            item.target_name if item.will_rename else "—",
            item.series_name,
            item.reason,
        )
    CONSOLE.print(table)


class HelpScreen(ModalScreen[None]):
    """Modal help dialog describing controls and shortcuts."""

    CSS = """
    HelpScreen {
        align: center middle;
        background: $background 60%;
    }

    #help-dialog {
        width: 94%;
        height: 88%;
        background: $surface-darken-1;
        border: thick $accent;
        padding: 1 2;
    }

    #help-title {
        text-style: bold;
        color: $accent;
        text-align: center;
        margin-bottom: 1;
    }

    #help-content {
        height: 1fr;
        overflow-y: auto;
        margin-bottom: 1;
    }

    #help-close {
        width: 100%;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("ctrl+c", "dismiss", "Close"),
        Binding("ctrl+q", "dismiss", "Close"),
    ]

    def compose(self) -> ComposeResult:
        """Build the help modal widgets."""
        with Vertical(id="help-dialog"):
            yield Label("MKV Renamer Help", id="help-title")
            yield Static(
                """
PATH
  • Enter a file or folder path to scan.
  • Use Explore to pick a directory visually.

FILTERS / SORTING
  • Global filter: search across all table columns.
  • Sort by: choose the column to sort.
  • Desc: reverse the current sort order.

WORKFLOW
  • Scan: build a live preview of rename actions.
  • Apply: perform the rename pass after previewing.
  • Dry-run: keep the scan as preview only.

SHORTCUTS
  • Ctrl+R: Scan      • Ctrl+Enter: Apply
  • Ctrl+H: Help      • Ctrl+E: Explore
  • Ctrl+L: Clear log • Ctrl+Q: Quit
""",
                id="help-content",
            )
            yield Button("Close", id="help-close", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Dismiss the help dialog when the close button is pressed."""
        if event.button.id == "help-close":
            self.dismiss()


class DirectoryPickerScreen(ModalScreen[str | None]):
    """Modal folder picker used by the Explore action."""

    CSS = """
    DirectoryPickerScreen {
        align: center middle;
        background: $background 60%;
    }

    #picker-dialog {
        width: 80%;
        height: 78%;
        background: $surface;
        border: thick $accent;
        padding: 1 2;
    }

    #picker-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }

    #picker-current,
    #picker-hint {
        color: $text-muted;
        margin-bottom: 1;
    }

    #picker-table {
        height: 1fr;
        margin-bottom: 1;
    }

    #picker-buttons {
        height: auto;
        layout: horizontal;
    }

    #picker-buttons Button {
        width: 1fr;
        margin-right: 1;
    }

    #picker-buttons Button:last-child {
        margin-right: 0;
    }
    """

    BINDINGS = [
        Binding("left", "go_up", "Up"),
        Binding("right", "drill_down", "Open"),
        Binding("enter", "drill_down", "Open"),
        Binding("backspace", "go_up", "Up"),
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, start_path: str) -> None:
        """Initialize the picker with a safe starting directory."""
        super().__init__()
        start = Path(start_path).expanduser() if start_path else Path.cwd()
        if start.exists() and start.is_file():
            start = start.parent
        if not start.exists() or not start.is_dir():
            start = Path.cwd()
        self._current_dir = start
        self._entries: list[tuple[Path, str]] = []

    def compose(self) -> ComposeResult:
        """Build the directory picker layout."""
        with Vertical(id="picker-dialog"):
            yield Label("Select Scan Folder", id="picker-title")
            yield Static("", id="picker-current")
            yield DataTable(id="picker-table", cursor_type="row", zebra_stripes=True)
            yield Static("Enter/Open: choose folder  •  Esc: cancel", id="picker-hint")
            with Horizontal(id="picker-buttons"):
                yield Button("Use Folder", id="picker-use", variant="success")
                yield Button("Cancel", id="picker-cancel", variant="error")

    def on_mount(self) -> None:
        """Create the table columns and load the initial directory."""
        table = cast(DataTable[Any], self.query_one("#picker-table", DataTable))
        table.add_columns("Name", "Type")
        self._refresh_directory()

    def _table(self) -> DataTable[Any]:
        """Return the directory table with the expected type."""
        return cast(DataTable[Any], self.query_one("#picker-table", DataTable))

    def _refresh_directory(self) -> None:
        """Reload the current directory listing into the picker table."""
        table = self._table()
        self.query_one("#picker-current", Static).update(
            f"Current: {self._current_dir}"
        )
        # The synthetic '.' and '..' rows mimic file-manager semantics,
        # reducing cognitive load for keyboard-first navigation.
        entries: list[tuple[Path, str]] = [(self._current_dir, "current")]
        parent = self._current_dir.parent
        if parent != self._current_dir:
            entries.append((parent, "parent"))
        try:
            children = sorted(
                self._current_dir.iterdir(),
                key=lambda p: (not p.is_dir(), p.name.lower()),
            )
        except OSError:
            children = []
        for child in children:
            if child.is_dir():
                entries.append((child, "dir"))
        self._entries = entries
        table.clear()
        for path, kind in entries:
            if kind == "current":
                table.add_row(".", "current dir")
            elif kind == "parent":
                table.add_row("..", "parent dir")
            else:
                table.add_row(path.name, "directory")

    def _current_entry(self) -> tuple[Path, str] | None:
        """Return the currently selected picker entry, if any."""
        table = self._table()
        row = table.cursor_row
        if row < 0 or row >= len(self._entries):
            return None
        return self._entries[row]

    def action_drill_down(self) -> None:
        """Open the selected directory entry."""
        entry = self._current_entry()
        if entry is None:
            return
        path, kind = entry
        if kind in {"dir", "parent"}:
            self._current_dir = path
            self._refresh_directory()

    def action_go_up(self) -> None:
        """Move the picker to the parent directory."""
        parent = self._current_dir.parent
        if parent != self._current_dir:
            self._current_dir = parent
            self._refresh_directory()

    def action_cancel(self) -> None:
        """Close the picker without returning a folder."""
        self.dismiss(None)

    def action_use_folder(self) -> None:
        """Return the current directory as the chosen folder."""
        self.dismiss(str(self._current_dir))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle Use Folder and Cancel button presses."""
        if event.button.id == "picker-use":
            self.action_use_folder()
        elif event.button.id == "picker-cancel":
            self.action_cancel()


class MkvRenamerApp(App[None]):
    """Main Textual application for previewing and applying renames."""

    TITLE = "MKV Renamer"
    SUB_TITLE = "Episode Rename Assistant"

    _SEL_PATH_INPUT = "#path-input"
    _SEL_RECURSIVE_SWITCH = "#recursive-switch"
    _SEL_RENUMBER_SWITCH = "#renumber-switch"
    _SEL_DRY_RUN_SWITCH = "#dry-run-switch"
    is_busy = reactive(False)

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+r", "scan", "Scan", show=True),
        Binding("ctrl+enter", "apply", "Apply", show=True),
        Binding("ctrl+e", "explorer", "Explore", show=True),
        Binding("ctrl+h", "help", "Help", show=True),
        Binding("ctrl+l", "clear_log", "Clear Log"),
        Binding("[", "narrow_col", "Narrow col", show=False),
        Binding("]", "widen_col", "Widen col", show=False),
    ]

    CSS = """
    Screen {
        layout: vertical;
    }

    #controls {
        height: auto;
        padding: 1;
        border: round $panel-lighten-1;
        background: $surface-darken-1;
    }

    #controls-row,
    #filter-row,
    #options-row {
        height: auto;
        layout: horizontal;
        margin-bottom: 1;
        align: left middle;
    }

    #path-input {
        width: 1fr;
    }

    #btn-explorer {
        margin-left: 0;
        margin-right: 2;
    }

    #global-filter {
        width: 1fr;
        margin-right: 1;
    }

    #sort-by {
        width: 18;
        margin-right: 1;
    }

    #table {
        height: 1fr;
        border: round $panel-lighten-1;
    }

    #resize-hint {
        height: auto;
        color: $text-muted;
        margin: 0 0 1 0;
    }

    #progress-row {
        height: 3;
        layout: horizontal;
        background: $surface-darken-1;
        border-top: tall $panel-lighten-1;
        border-bottom: tall $panel-lighten-1;
        padding: 0 1;
        align: left middle;
        display: none;
    }

    #progress-row.visible {
        display: block;
        layout: horizontal;
    }

    #progress-bar {
        width: 1fr;
        margin: 0;
        padding: 0;
    }

    #progress-bar Bar {
        width: 1fr;
    }

    #progress-label,
    #progress-eta {
        width: auto;
        color: $text-muted;
        margin-left: 1;
    }

    .switch-item {
        layout: horizontal;
        align: left middle;
        height: 3;
        width: auto;
        margin-right: 2;
    }

    .switch-item Label {
        width: auto;
        margin-right: 1;
        padding-top: 1;
    }

    #log {
        height: 12;
        border-top: tall $panel-lighten-1;
        background: $surface-darken-1;
    }
    """

    _COLUMN_SPECS: list[tuple[str, str, float, int]] = [
        ("Reason", "reason", 0.30, 12),
        ("Status", "status", 0.12, 8),
        ("Action", "action", 0.12, 8),
        ("Target", "target", 0.18, 14),
        ("Original", "original", 0.16, 14),
        ("Folder", "folder", 0.12, 14),
    ]

    def __init__(
        self,
        initial_path: str,
        initial_extensions: str,
        recursive: bool,
        renumber_enabled: bool,
    ) -> None:
        """Initialize app state from CLI-provided defaults and saved config."""
        super().__init__()
        self._initial_path = initial_path
        self._initial_extensions = initial_extensions
        self._recursive = recursive
        self._renumber_enabled = renumber_enabled
        self._scan_result: ScanResult | None = None
        self._global_filter = ""
        self._sort_by = "reason"
        self._sort_desc = False
        self._progress_total = 0
        self._progress_done = 0
        self._progress_label = ""
        self._progress_started_at = 0.0
        self._saved_dry_run = True
        self._column_keys: list[Any] = []
        self._col_widths: list[int] = [min_w for *_rest, min_w in self._COLUMN_SPECS]
        self._user_fixed_cols: set[int] = set()
        self._fixed_col_widths: dict[int, int] = {}
        self._scan_cache = _load_json(_scan_cache_path())
        if _compact_scan_cache_payload(self._scan_cache):
            _save_json(_scan_cache_path(), self._scan_cache)

        saved = _load_config()
        saved_path = str(saved.get("path", "")).strip()
        if self._initial_path in {"", "."} and saved_path:
            self._initial_path = saved_path
        if isinstance(saved.get("recursive"), bool):
            self._recursive = bool(saved.get("recursive"))
        if isinstance(saved.get("renumber_enabled"), bool):
            self._renumber_enabled = bool(saved.get("renumber_enabled"))
        if isinstance(saved.get("dry_run"), bool):
            self._saved_dry_run = bool(saved.get("dry_run"))
        self._global_filter = str(saved.get("global_filter", self._global_filter))
        saved_sort = str(saved.get("sort_by", self._sort_by))
        if saved_sort in {"reason", "status", "action", "target", "original", "folder"}:
            self._sort_by = saved_sort
        if isinstance(saved.get("sort_desc"), bool):
            self._sort_desc = bool(saved.get("sort_desc"))

        saved_fixed_cols = saved.get("user_fixed_cols")
        if isinstance(saved_fixed_cols, list):
            for col in cast(list[object], saved_fixed_cols):
                if isinstance(col, int):
                    self._user_fixed_cols.add(col)

        saved_fixed_widths = saved.get("fixed_col_widths")
        if isinstance(saved_fixed_widths, dict):
            for key, value in cast(dict[object, object], saved_fixed_widths).items():
                if not isinstance(value, int):
                    continue
                try:
                    idx = int(str(key))
                except ValueError:
                    continue
                if 0 <= idx < len(self._COLUMN_SPECS):
                    self._fixed_col_widths[idx] = max(8, value)

    def _save_current_config(self) -> None:
        """Persist the current control values and column settings."""
        cfg: dict[str, object] = {
            "path": self.query_one(self._SEL_PATH_INPUT, Input).value.strip(),
            "recursive": self.query_one(self._SEL_RECURSIVE_SWITCH, Switch).value,
            "renumber_enabled": self.query_one(self._SEL_RENUMBER_SWITCH, Switch).value,
            "dry_run": self.query_one(self._SEL_DRY_RUN_SWITCH, Switch).value,
            "global_filter": self._global_filter,
            "sort_by": self._sort_by,
            "sort_desc": self._sort_desc,
            "user_fixed_cols": sorted(self._user_fixed_cols),
            "fixed_col_widths": {
                str(col_idx): width
                for col_idx, width in sorted(self._fixed_col_widths.items())
            },
        }
        _save_config(cfg)

    def _scan_cache_lookup(self, cache_key: str, signature: str) -> ScanResult | None:
        """Return cached scan results only when config and file signature agree.

        Cache misses are intentionally cheap and silent; stale or malformed
        entries simply behave as a miss so scans remain resilient.
        """
        if _compact_scan_cache_payload(self._scan_cache):
            _save_json(_scan_cache_path(), self._scan_cache)
        # A cache hit requires both matching config key and signature,
        # guarding against stale plans when files changed on disk.
        entries_obj = self._scan_cache.get("entries")
        if not isinstance(entries_obj, dict):
            return None
        entries = cast(dict[str, object], entries_obj)
        entry_obj = entries.get(cache_key)
        if not isinstance(entry_obj, dict):
            return None
        entry = cast(dict[str, object], entry_obj)
        if str(entry.get("signature", "")) != signature:
            return None
        result_obj = entry.get("result")
        if not isinstance(result_obj, dict):
            return None
        return _scan_result_from_payload(cast(dict[str, object], result_obj))

    def _scan_cache_store(
        self, cache_key: str, signature: str, result: ScanResult
    ) -> None:
        """Store one scan result in the bounded on-disk cache.

        The cache is compacted before and after writes so retention limits are
        enforced even when older app versions wrote looser payloads.
        """
        _compact_scan_cache_payload(self._scan_cache)
        entries_obj = self._scan_cache.get("entries")
        if not isinstance(entries_obj, dict):
            entries_obj = {}
            self._scan_cache["entries"] = entries_obj
        entries = cast(dict[str, object], entries_obj)
        # Saving timestamp with each entry enables deterministic LRU-ish
        # pruning when the entry cap is exceeded.
        entries[cache_key] = {
            "signature": signature,
            "saved_at": int(time.time()),
            "result": _scan_result_to_payload(result),
        }

        _compact_scan_cache_payload(self._scan_cache)

        _save_json(_scan_cache_path(), self._scan_cache)

    def compose(self) -> ComposeResult:
        """Build the application layout and widgets."""
        yield Header()
        with Vertical(id="controls"):
            with Horizontal(id="controls-row"):
                yield Input(
                    value=self._initial_path, placeholder="Scan path", id="path-input"
                )
                yield Button("Explore", id="btn-explorer", variant="default")
                yield Button("Scan", id="btn-scan", variant="primary")
                yield Button("Apply", id="btn-apply", variant="success")
            with Horizontal(id="filter-row"):
                yield Input(
                    value=self._global_filter,
                    placeholder="Filter all columns…",
                    id="global-filter",
                )
                yield Select(
                    [
                        ("Reason", "reason"),
                        ("Status", "status"),
                        ("Action", "action"),
                        ("Target", "target"),
                        ("Original", "original"),
                        ("Folder", "folder"),
                    ],
                    value=self._sort_by,
                    allow_blank=False,
                    prompt="Sort by",
                    id="sort-by",
                )
                with Horizontal(classes="switch-item"):
                    yield Switch(value=self._sort_desc, id="sort-desc")
                    yield Label("Desc")
            with Horizontal(id="options-row"):
                with Horizontal(classes="switch-item"):
                    yield Label("Recursive")
                    yield Switch(value=self._recursive, id="recursive-switch")
                with Horizontal(classes="switch-item"):
                    yield Label("Renumber")
                    yield Switch(value=self._renumber_enabled, id="renumber-switch")
                with Horizontal(classes="switch-item"):
                    yield Label("Dry-run")
                    yield Switch(value=self._saved_dry_run, id="dry-run-switch")
        yield DataTable(id="table", cursor_type="row", zebra_stripes=True)
        yield Static("[ / ] resize selected table column", id="resize-hint")
        with Horizontal(id="progress-row"):
            yield ProgressBar(id="progress-bar", show_eta=False, show_percentage=False)
            yield Static("", id="progress-label")
            yield Static("", id="progress-eta")
        yield RichLog(id="log", markup=True, highlight=True)
        yield Footer()

    def on_mount(self) -> None:
        """Create table columns, restore state, and trigger the initial scan."""
        table = self._table()
        self._column_keys = [
            table.add_column(name)
            for name, _key, _ratio, _min_width in self._COLUMN_SPECS
        ]
        table.cursor_type = "cell"
        self._apply_column_widths()
        self._write_log("Ready. Press Scan to build a preview.")
        self._save_current_config()
        if self._initial_path:
            self.call_after_refresh(self.action_scan)

    def _table(self) -> DataTable[Any]:
        """Return the main results table with the expected type."""
        return cast(DataTable[Any], self.query_one("#table", DataTable))

    def _progress_bar(self) -> ProgressBar:
        """Return the progress bar widget."""
        return self.query_one("#progress-bar", ProgressBar)

    def _progress_label_widget(self) -> Static:
        """Return the progress label widget."""
        return self.query_one("#progress-label", Static)

    def _progress_eta_widget(self) -> Static:
        """Return the ETA label widget."""
        return self.query_one("#progress-eta", Static)

    def _set_busy(self, busy: bool) -> None:
        """Update the reactive busy state."""
        self.is_busy = busy

    def watch_is_busy(self, busy: bool) -> None:
        """Show progress UI and disable controls while background work runs."""
        row = self.query_one("#progress-row")
        if busy:
            row.add_class("visible")
        else:
            row.remove_class("visible")
        for selector in [
            "#btn-scan",
            "#btn-apply",
            "#btn-explorer",
            "#global-filter",
            "#sort-by",
            "#sort-desc",
            self._SEL_RECURSIVE_SWITCH,
            self._SEL_RENUMBER_SWITCH,
            self._SEL_DRY_RUN_SWITCH,
            self._SEL_PATH_INPUT,
        ]:
            try:
                widget = self.query_one(selector)
                widget.disabled = busy  # type: ignore[attr-defined]
            except NoMatches:
                pass

    def _collect_controls(self) -> tuple[Path, list[str], bool, bool, bool]:
        """Read the current scan settings from the live widgets."""
        path = Path(
            self.query_one(self._SEL_PATH_INPUT, Input).value.strip() or "."
        ).expanduser()
        extensions = [
            token.strip()
            for token in self._initial_extensions.split(",")
            if token.strip()
        ]
        recursive = self.query_one(self._SEL_RECURSIVE_SWITCH, Switch).value
        renumber_enabled = self.query_one(self._SEL_RENUMBER_SWITCH, Switch).value
        dry_run = self.query_one(self._SEL_DRY_RUN_SWITCH, Switch).value
        return path, extensions, recursive, renumber_enabled, dry_run

    def _write_log(self, message: str) -> None:
        """Append one message to the on-screen log."""
        self.query_one("#log", RichLog).write(message)

    def _set_progress(self, done: int, total: int, label: str) -> None:
        """Update the progress row with counts and a best-effort ETA."""
        self._progress_done = done
        self._progress_total = max(total, 1)
        self._progress_label = label
        self._progress_bar().update(total=self._progress_total, progress=done)
        percent = (done / self._progress_total) * 100
        self._progress_label_widget().update(
            f"{done}/{self._progress_total} •{percent:5.1f}% •{label}"
        )

        eta_text = "ETA --:--"
        if 0 < done < self._progress_total and self._progress_started_at > 0:
            elapsed = max(0.001, time.monotonic() - self._progress_started_at)
            rate = done / elapsed
            if rate > 0:
                remaining = int(round((self._progress_total - done) / rate))
                minutes, seconds = divmod(remaining, 60)
                if minutes >= 60:
                    hours, minutes = divmod(minutes, 60)
                    eta_text = f"ETA {hours:02d}:{minutes:02d}:{seconds:02d}"
                else:
                    eta_text = f"ETA {minutes:02d}:{seconds:02d}"
        elif done >= self._progress_total:
            eta_text = "ETA 00:00"
        self._progress_eta_widget().update(eta_text)

    def _matches_filter(self, item: RenamePlanItem) -> bool:
        """Return whether a plan item matches the global text filter."""
        needle = self._global_filter.strip().lower()
        if not needle:
            return True
        haystack = " ".join(
            [
                item.reason,
                item.status,
                item.action,
                item.target_name,
                item.original_name,
                item.folder_path,
            ]
        ).lower()
        return needle in haystack

    def _sort_key_for_item(self, item: RenamePlanItem) -> str:
        """Return the active sort key for one table row."""
        mapping = {
            "reason": item.reason,
            "status": item.status,
            "action": item.action,
            "target": item.target_name,
            "original": item.original_name,
            "folder": item.folder_path,
        }
        return mapping.get(self._sort_by, item.reason).lower()

    def _filtered_sorted_rows(self) -> list[RenamePlanItem]:
        """Return the visible rows after filtering and sorting."""
        if self._scan_result is None:
            return []
        rows = [item for item in self._scan_result.items if self._matches_filter(item)]
        rows.sort(key=self._sort_key_for_item, reverse=self._sort_desc)
        return rows

    def _table_content_width(self) -> int:
        """Return the usable table width after accounting for widget chrome."""
        try:
            table = self._table()
            if table.size.width > 0:
                return max(20, table.size.width + 5)
        except (NoMatches, AttributeError):
            pass
        return max(40, self.size.width - 6)

    def _measure_column_content(
        self, rows: list[RenamePlanItem], col_idx: int, header: str
    ) -> int:
        """Measure the width needed for one column's header and visible cells."""
        width = len(header)
        for item in rows:
            if col_idx == 0:
                width = max(width, len(item.reason))
            elif col_idx == 1:
                width = max(width, len(item.status))
            elif col_idx == 2:
                width = max(width, len(item.action))
            elif col_idx == 3:
                width = max(width, len(item.target_name))
            elif col_idx == 4:
                width = max(width, len(item.original_name))
            else:
                width = max(width, len(item.folder_path))
        return width + 2

    def _rendered_columns_width(self) -> int:
        """Return the table's current rendered column width without borders."""
        table = self._table()
        total = 0
        for col_key in self._column_keys:
            total += table.columns[col_key].get_render_width(table)
        return total

    def _apply_column_widths(self, rows: list[RenamePlanItem] | None = None) -> None:
        """Fit table columns to content while honoring manual overrides."""
        table = self._table()
        if not self._column_keys:
            return
        if rows is None:
            rows = self._filtered_sorted_rows()

        available = self._table_content_width()
        separators = len(self._COLUMN_SPECS) + 1
        usable = max(20, available - separators)

        hard_min_widths = [14, 12, 8, 16, 14, 14]
        soft_min_widths = [20, 14, 10, 32, 28, 28]
        max_widths = [60, 18, 14, 90, 80, 80]
        preferred = [
            self._measure_column_content(rows, idx, header)
            for idx, (header, _key, _ratio, _min_width) in enumerate(self._COLUMN_SPECS)
        ]
        widths = [
            max(
                soft_min_widths[i],
                min(max_widths[i], preferred[i]),
            )
            for i in range(len(preferred))
        ]

        for idx, fixed_width in self._fixed_col_widths.items():
            if idx in range(len(widths)):
                widths[idx] = max(
                    hard_min_widths[idx], min(max_widths[idx], fixed_width)
                )

        total = sum(widths)

        shrink_order = [5, 4, 3, 2, 1, 0]
        while total > usable:
            overflow = total - usable
            changed = False
            for idx in shrink_order:
                if idx in self._user_fixed_cols:
                    continue
                shrink_cap = widths[idx] - hard_min_widths[idx]
                if shrink_cap <= 0:
                    continue
                reduction = min(shrink_cap, overflow)
                widths[idx] -= reduction
                total -= reduction
                overflow -= reduction
                changed = True
                if overflow <= 0:
                    break
            if not changed:
                break

        self._col_widths = list(widths)
        for col_key, width in zip(self._column_keys, widths, strict=False):
            col = table.columns[col_key]
            col.auto_width = False
            col.width = width
            col.content_width = width

        rendered_total = self._rendered_columns_width() + separators
        while rendered_total > available:
            overflow = rendered_total - available
            changed = False
            for idx in shrink_order:
                if idx in self._user_fixed_cols:
                    continue
                shrink_cap = widths[idx] - hard_min_widths[idx]
                if shrink_cap <= 0:
                    continue
                reduction = min(shrink_cap, overflow)
                if reduction <= 0:
                    continue
                widths[idx] -= reduction
                col = table.columns[self._column_keys[idx]]
                col.width = widths[idx]
                col.content_width = widths[idx]
                changed = True
                rendered_total -= reduction
                overflow -= reduction
                if overflow <= 0:
                    break
            if not changed:
                break

        self._col_widths = list(widths)
        table.refresh(layout=True, repaint=True)

    def _selected_col_index(self) -> int:
        """Return the currently selected table column index."""
        table = self._table()
        col_idx = table.cursor_column
        if col_idx < 0:
            return 0
        return min(col_idx, len(self._COLUMN_SPECS) - 1)

    def _set_fixed_column_width(self, col_idx: int, width: int) -> None:
        """Apply and persist a manual width for the selected column."""
        clamped_width = max(8, min(120, width))
        self._fixed_col_widths[col_idx] = clamped_width
        self._user_fixed_cols.add(col_idx)
        self._col_widths[col_idx] = clamped_width

        table = self._table()
        saved_row = table.cursor_row
        saved_col = table.cursor_column
        col_key = self._column_keys[col_idx]
        col = table.columns[col_key]
        col.auto_width = False
        col.width = clamped_width
        col.content_width = clamped_width

        self._refresh_table()

        if table.row_count > 0:
            table.move_cursor(
                row=max(0, min(saved_row, table.row_count - 1)),
                column=max(0, min(saved_col, len(self._column_keys) - 1)),
                animate=False,
            )
        table.focus()
        self._save_current_config()

    def action_narrow_col(self) -> None:
        """Shrink the currently selected table column."""
        if not self._column_keys:
            return
        col_idx = self._selected_col_index()
        current = self._col_widths[col_idx]
        self._set_fixed_column_width(col_idx, current - 2)

    def action_widen_col(self) -> None:
        """Widen the currently selected table column."""
        if not self._column_keys:
            return
        col_idx = self._selected_col_index()
        current = self._col_widths[col_idx]
        self._set_fixed_column_width(col_idx, current + 2)

    def _refresh_table(self) -> None:
        """Rebuild the visible results table from the current scan result."""
        table = self._table()
        table.clear()
        if self._scan_result is None:
            self._apply_column_widths()
            return

        rows = self._filtered_sorted_rows()
        for item in rows:
            table.add_row(
                item.reason,
                item.status,
                item.action,
                item.target_name,
                item.original_name,
                item.folder_path,
            )
        self._apply_column_widths(rows)

    def _folder_scanned(
        self, folder_plan: FolderPlan, index: int, total: int, dry_run: bool
    ) -> None:
        """Append one folder's plan to the live preview while scanning.

        This keeps the UI responsive on large libraries and gives users early
        feedback before the full scan is complete.
        """
        if self._scan_result is None:
            self._scan_result = ScanResult(
                media_path=Path(
                    self.query_one(self._SEL_PATH_INPUT, Input).value.strip() or "."
                ),
                folder_plans=[],
            )
        if dry_run:
            for item in folder_plan.items:
                if item.status == "Pending" and item.will_rename:
                    item.status = "WouldRename"
        self._scan_result.folder_plans.append(folder_plan)
        self._set_progress(index, total, "")
        self._refresh_table()

    def _scan_finished(self, result: ScanResult, dry_run: bool) -> None:
        """Finalize scan state and update the UI after scanning completes."""
        self._scan_result = result
        if dry_run:
            for folder_plan in self._scan_result.folder_plans:
                for item in folder_plan.items:
                    if item.status == "Pending" and item.will_rename:
                        item.status = "WouldRename"
        self._refresh_table()
        self._write_log(
            (
                f"Scanned {len(self._scan_result.items)} file(s) across "
                f"{len(self._scan_result.folder_plans)} folder(s)."
            )
        )
        self._set_progress(self._progress_total, self._progress_total, "Scan complete")
        self._set_busy(False)

    def _apply_folder_done(self, index: int, total: int, folder_name: str) -> None:
        """Update progress after one folder finishes applying renames."""
        self._set_progress(index, total, f"Applying {index}/{total}: {folder_name}")
        self._refresh_table()

    @work(thread=True)
    def action_scan(self) -> None:
        """Run scan planning in a worker thread and stream progress to the UI.

        Heavy filesystem work stays off the main event loop to keep controls
        interactive; cached results short-circuit repeated scans.
        """
        path, extensions, recursive, renumber_enabled, dry_run = (
            self._collect_controls()
        )
        if not path.exists():
            self.call_from_thread(
                self._write_log, f"[red]Media path does not exist:[/red] {path}"
            )
            return

        self._scan_result = ScanResult(media_path=path, folder_plans=[])
        self._progress_started_at = time.monotonic()
        self.call_from_thread(self._set_busy, True)
        self.call_from_thread(self._set_progress, 0, 1, "")

        file_records = collect_file_records(
            path,
            extensions or list(VIDEO_EXTENSIONS_DEFAULT),
            recursive=recursive,
        )
        signature = _scan_signature(file_records)
        cache_key = _scan_cache_key(
            path,
            extensions or list(VIDEO_EXTENSIONS_DEFAULT),
            recursive,
            renumber_enabled,
        )
        cached = self._scan_cache_lookup(cache_key, signature)
        if cached is not None:
            self.call_from_thread(self._write_log, "Loaded scan plan from cache.")
            self.call_from_thread(self._set_progress, 1, 1, "Loaded from cache")
            self.call_from_thread(self._scan_finished, cached, dry_run)
            return

        result = build_scan_result(
            path,
            extensions or list(VIDEO_EXTENSIONS_DEFAULT),
            recursive=recursive,
            renumber_enabled=renumber_enabled,
            folder_callback=lambda plan, idx, total: self.call_from_thread(
                self._folder_scanned, plan, idx, total, dry_run
            ),
        )
        self._scan_cache_store(cache_key, signature, result)
        self.call_from_thread(self._scan_finished, result, dry_run)

    @work(thread=True)
    def action_apply(self) -> None:
        """Apply pending renames in a worker thread from the current preview.

        This path is guarded to prevent accidental writes when the target path
        changed or when dry-run is still enabled.
        """
        if self._scan_result is None:
            self.call_from_thread(self._write_log, "[yellow]Run Scan first.[/yellow]")
            return

        path, _extensions, _recursive, _renumber_enabled, dry_run = (
            self._collect_controls()
        )
        if path != self._scan_result.media_path:
            self.call_from_thread(
                self._write_log,
                "[yellow]Settings changed since the last scan. Scan again first.[/yellow]",
            )
            return
        if dry_run:
            self.call_from_thread(
                self._write_log,
                "[yellow]Dry-run is enabled. Turn it off to apply changes.[/yellow]",
            )
            return

        folders = [
            folder
            for folder in self._scan_result.folder_plans
            if any(is_applyable_item(item) for item in folder.items)
        ]
        if not folders:
            self.call_from_thread(self._write_log, "[yellow]Nothing to apply.[/yellow]")
            return

        self.call_from_thread(self._set_busy, True)
        self._progress_started_at = time.monotonic()
        self.call_from_thread(self._set_progress, 0, len(folders), "Applying…")
        for index, folder_plan in enumerate(folders, start=1):
            invoke_two_phase_rename(folder_plan.items)
            folder_plan.status = get_folder_status_label(folder_plan.items)
            self.call_from_thread(
                self._apply_folder_done, index, len(folders), folder_plan.series_name
            )

        self.call_from_thread(self._refresh_table)
        self.call_from_thread(self._write_log, "[green]Rename pass completed.[/green]")
        self.call_from_thread(self._set_busy, False)

    def action_explorer(self) -> None:
        """Open the folder picker modal."""
        current = self.query_one(self._SEL_PATH_INPUT, Input).value.strip()
        self.push_screen(DirectoryPickerScreen(current), self._on_directory_chosen)

    def _on_directory_chosen(self, path: str | None) -> None:
        """Handle the folder returned by the picker modal."""
        if not path:
            return
        self.query_one(self._SEL_PATH_INPUT, Input).value = path
        self._write_log(f"Explorer selected: {path}")
        self._save_current_config()

    def action_help(self) -> None:
        """Open the help modal."""
        self.push_screen(HelpScreen())

    def action_clear_log(self) -> None:
        """Clear the on-screen log widget."""
        self.query_one("#log", RichLog).clear()

    def on_input_changed(self, event: Input.Changed) -> None:
        """React to text-input changes and persist the new state."""
        if event.input.id == "global-filter":
            self._global_filter = event.value
            self._refresh_table()
        self._save_current_config()

    def on_select_changed(self, event: Select.Changed) -> None:
        """React to select-widget changes and persist the new state."""
        if event.select.id == "sort-by":
            self._sort_by = str(event.value)
            self._refresh_table()
        self._save_current_config()

    def on_switch_changed(self, event: Switch.Changed) -> None:
        """React to switch changes and persist the new state."""
        if event.switch.id == "sort-desc":
            self._sort_desc = bool(event.value)
            self._refresh_table()
        self._save_current_config()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Dispatch top-level button presses to their matching actions."""
        button_id = event.button.id
        if button_id == "btn-scan":
            self.action_scan()
        elif button_id == "btn-apply":
            self.action_apply()
        elif button_id == "btn-explorer":
            self.action_explorer()

    def on_resize(self) -> None:
        """Recompute automatic column widths when the terminal resizes."""
        self._apply_column_widths()


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser for the renamer."""
    parser = argparse.ArgumentParser(
        description="Rename anime video files using episode tokens."
    )
    parser.add_argument("--path", default=".", help="Root media folder to scan.")
    parser.add_argument(
        "--extensions",
        default=", ".join(VIDEO_EXTENSIONS_DEFAULT),
        help="Comma-separated list of extensions or glob-like patterns.",
    )
    parser.add_argument(
        "--no-recursive", action="store_true", help="Only scan the top-level folder."
    )
    parser.add_argument(
        "--skip-renumbering",
        action="store_true",
        help="Keep detected episode numbers instead of renumbering.",
    )
    parser.add_argument(
        "--mode",
        choices=("tui", "cli", "apply"),
        default="tui",
        help="Choose the interactive TUI, a CLI preview, or direct apply mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Force preview-only output in CLI/apply mode.",
    )
    parser.add_argument(
        "--yes", action="store_true", help="Skip confirmation when using apply mode."
    )
    return parser


def parse_extensions(raw_value: str) -> list[str]:
    """Split a comma-separated extension list while preserving user intent.

    This function only trims empty/whitespace tokens. Pattern normalization
    happens later in ``get_supported_extension_set``.
    """
    return [token.strip() for token in raw_value.split(",") if token.strip()]


def confirm_apply() -> bool:
    """Ask the CLI user to confirm that renames should be applied."""
    if not sys.stdin.isatty():
        return False
    try:
        return Confirm.ask("Apply these rename changes now?", default=False)
    except EOFError:
        return False


def run_cli(args: argparse.Namespace) -> int:
    """Run the non-interactive flow used by scripts and terminal workflows.

    `cli` mode always previews. `apply` mode still previews first so users can
    audit changes before mutation, then applies after confirmation unless
    `--yes` is provided.
    """
    media_path = Path(args.path).expanduser().resolve()
    if not media_path.exists():
        CONSOLE.print(f"[red]Media path does not exist:[/red] {media_path}")
        return 1

    extensions = parse_extensions(args.extensions)
    renumber_enabled = not args.skip_renumbering
    dry_run = args.dry_run or args.mode == "cli"

    recursive = not args.no_recursive
    file_records = collect_file_records(
        media_path,
        extensions,
        recursive=recursive,
    )
    grouped_records = group_file_records(file_records)

    _render_cli_runtime_header(
        media_path,
        extensions,
        recursive,
        renumber_enabled,
        dry_run,
        args.mode,
        folder_count=len(grouped_records),
        file_count=len(file_records),
    )

    if not file_records:
        CONSOLE.print(
            Panel(
                "[bold yellow]No files found to process.[/bold yellow]",
                border_style="yellow",
                box=box.ROUNDED,
            )
        )
        return 0

    folder_plans: list[FolderPlan] = []
    scan_progress = _cli_build_operation_progress()
    scan_task = scan_progress.add_task(
        "Scanning folders",
        total=max(1, len(grouped_records)),
        completed=0,
        details="queued",
    )
    current_panel = _cli_build_current_folder_panel(
        phase_label="Scanning",
        folder_index=0,
        total_folders=len(grouped_records),
        folder_plan=None,
        dry_run=dry_run,
    )
    live_group = Group(scan_progress, current_panel)
    with Live(
        live_group, console=CONSOLE, refresh_per_second=10, transient=False
    ) as live:
        for folder_index, (folder_path, records) in enumerate(grouped_records, start=1):
            folder_plan = validate_folder_plan(
                new_folder_plan(
                    folder_path,
                    records,
                    renumber_enabled=renumber_enabled,
                )
            )
            folder_plans.append(folder_plan)
            current_panel = _cli_build_current_folder_panel(
                phase_label="Scanning",
                folder_index=folder_index,
                total_folders=len(grouped_records),
                folder_plan=folder_plan,
                dry_run=dry_run,
            )
            scan_progress.update(scan_task, details=folder_plan.series_name)
            live_group = Group(scan_progress, current_panel)
            live.update(live_group)
            scan_progress.advance(scan_task)

        scan_progress.update(
            scan_task,
            description="Scanned folders",
            details=f"{len(grouped_records)} folder(s) scanned",
        )
        live_group = Group(scan_progress, current_panel)
        live.update(live_group)

    scan_result = ScanResult(media_path=media_path, folder_plans=folder_plans)
    _render_cli_summary(scan_result)

    if dry_run or args.mode == "cli":
        return 0
    if not args.yes and not confirm_apply():
        CONSOLE.print(
            Panel(
                "[bold yellow]Cancelled. No rename changes were applied.[/bold yellow]",
                border_style="yellow",
                box=box.ROUNDED,
            )
        )
        return 1

    apply_progress = _cli_build_operation_progress()
    apply_task = apply_progress.add_task(
        "Applying folders",
        total=max(1, len(scan_result.folder_plans)),
        completed=0,
        details="queued",
    )
    current_panel = _cli_build_current_folder_panel(
        phase_label="Applying",
        folder_index=0,
        total_folders=len(scan_result.folder_plans),
        folder_plan=None,
        dry_run=False,
    )
    live_group = Group(apply_progress, current_panel)
    with Live(
        live_group, console=CONSOLE, refresh_per_second=10, transient=False
    ) as live:
        for folder_index, folder_plan in enumerate(scan_result.folder_plans, start=1):
            current_panel = _cli_build_current_folder_panel(
                phase_label="Applying",
                folder_index=folder_index,
                total_folders=len(scan_result.folder_plans),
                folder_plan=folder_plan,
                dry_run=False,
            )
            apply_progress.update(apply_task, details=folder_plan.series_name)
            live_group = Group(apply_progress, current_panel)
            live.update(live_group)

            invoke_two_phase_rename(folder_plan.items)
            folder_plan.status = get_folder_status_label(folder_plan.items)

            current_panel = _cli_build_current_folder_panel(
                phase_label="Applying",
                folder_index=folder_index,
                total_folders=len(scan_result.folder_plans),
                folder_plan=folder_plan,
                dry_run=False,
            )
            apply_progress.advance(apply_task)
            live_group = Group(apply_progress, current_panel)
            live.update(live_group)

        apply_progress.update(
            apply_task,
            description="Applied folders",
            details=f"{len(scan_result.folder_plans)} folder(s) finished",
        )
        live_group = Group(apply_progress, current_panel)
        live.update(live_group)

    CONSOLE.print(
        Panel(
            "[bold green]Rename pass finished.[/bold green]",
            border_style="green",
            box=box.ROUNDED,
        )
    )
    _render_cli_summary(scan_result)
    return 0


def run_tui(args: argparse.Namespace) -> int:
    """Run the interactive Textual application."""
    app = MkvRenamerApp(
        initial_path=args.path,
        initial_extensions=args.extensions,
        recursive=not args.no_recursive,
        renumber_enabled=not args.skip_renumbering,
    )
    app.run()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and dispatch to TUI or CLI mode."""
    args = build_arg_parser().parse_args(argv)
    if args.mode == "tui":
        return run_tui(args)
    return run_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
