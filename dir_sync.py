#!/usr/bin/env python3
"""dir_sync.py — recursively mirror one directory into another.

This tool treats the source directory as the truth and mutates the target so the
target becomes a recursive match.

Modes:
- fast   (default): metadata-first sync using size + mtime
- exact: compare file contents with full hashing, reusing a persistent cache
- rebuild: clear target contents and recopy the source tree from scratch

Examples:
    python dir_sync.py D:/Anime A:/Anime
    python dir_sync.py D:/Anime A:/Anime --mode exact --jobs 2
    python dir_sync.py D:/Anime A:/Anime --mode rebuild --yes
"""
# pylint: disable=too-many-lines

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, cast

from rich.console import Console, Group, RenderableType
from rich.align import Align
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    TaskProgressColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.prompt import Confirm
from rich.panel import Panel
from rich.table import Column, Table
from rich.text import Text
from rich import box


ActionKind = Literal[
    "ADD_DIR",
    "COPY_FILE",
    "REPLACE_FILE",
    "REMOVE_FILE",
    "REMOVE_DIR",
    "REPLACE_WITH_DIR",
    "REPLACE_WITH_FILE",
]
SyncMode = Literal["fast", "exact", "rebuild"]
CompareMode = Literal["fast", "exact", "verify"]
CacheRecord = dict[str, object]
CacheEntries = dict[str, CacheRecord]

DEFAULT_HASH_ALGORITHM = "sha256"
DEFAULT_CHUNK_SIZE = 1024 * 1024
DEFAULT_JOBS = 1
CACHE_VERSION = 1
CONSOLE = Console()

_interrupt_requested: bool = False  # pylint: disable=invalid-name
"""Set to ``True`` by the signal handler; functions should poll via
:func:`check_interrupt` before starting long operations."""

_original_sigint: Any = None  # pylint: disable=invalid-name
"""Holds the original SIGINT handler so we can restore it on shutdown."""


def _handle_sigint(signum: int, frame: object) -> None:  # pylint: disable=unused-argument
    """Signal handler for SIGINT (Ctrl+C).

    On the *first* interrupt, sets the global flag so ongoing operations can
    finish their current unit of work before exiting cleanly.  On a *second*
    interrupt, restores the default handler and lets Python's normal
    ``KeyboardInterrupt`` propagate immediately.
    """
    global _interrupt_requested  # pylint: disable=global-statement
    if _interrupt_requested:
        # Second Ctrl+C – restore original handler and re-raise
        if _original_sigint is not None:
            signal.signal(signal.SIGINT, _original_sigint)
        raise KeyboardInterrupt()
    _interrupt_requested = True
    CONSOLE.print(
        "\n[bold yellow]Interrupt requested – finishing current action "
        "before stopping… (press Ctrl+C again to force)[/bold yellow]"
    )


def install_interrupt_handler() -> None:
    """Install the graceful SIGINT handler.

    Must be called at the start of :func:`main` after the Console is created.
    """
    global _original_sigint  # pylint: disable=global-statement
    _original_sigint = signal.signal(signal.SIGINT, _handle_sigint)


def check_interrupt() -> None:
    """Poll the interrupt flag; raise ``KeyboardInterrupt`` if set.

    Call this before starting any action that could leave a partial artifact
    (e.g. scanning a directory, starting a hash, beginning a copy).
    """
    if _interrupt_requested:
        raise KeyboardInterrupt()


def cleanup_stale_temp_files(target_root: Path) -> None:
    """Remove any leftover ``.dir-sync-*.tmp`` files from a previous run."""
    if not target_root.exists():
        return
    removed = 0
    for temp_path in target_root.rglob(".dir-sync-*.tmp"):
        try:
            temp_path.unlink(missing_ok=True)
            removed += 1
        except OSError:
            pass
    if removed:
        CONSOLE.print(
            f"[dim]Cleaned up {removed} stale temp file(s) from target.[/dim]"
        )


def safe_save_cache(path: Path, cache_obj: dict[str, Any]) -> None:
    """Atomically write the cache file via a temp file so it cannot be
    corrupted by an untimely interrupt."""
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=".dir-sync-cache-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
        temp_path.write_text(
            json.dumps(cache_obj, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temp_path, path)
    except OSError:
        pass
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


ACTION_STYLES: dict[str, str] = {
    "ADD_DIR": "bold green",
    "COPY_FILE": "bold bright_green",
    "REPLACE_FILE": "bold yellow",
    "REMOVE_FILE": "bold red",
    "REMOVE_DIR": "bold bright_red",
    "REPLACE_WITH_DIR": "bold magenta",
    "REPLACE_WITH_FILE": "bold bright_magenta",
}

REASON_STYLES: list[tuple[str, str]] = [
    ("rebuild", "bold red"),
    ("missing", "bright_green"),
    ("type mismatch", "bold magenta"),
    ("directory", "magenta"),
    ("file", "cyan"),
    ("content differs", "bold yellow"),
    ("hash", "yellow"),
    ("size", "bright_yellow"),
    ("metadata", "bright_blue"),
    ("mtime", "blue"),
    ("verification", "bold red"),
    ("copy", "green"),
    ("remove", "red"),
]


@dataclass(slots=True)
class FileEntry:
    """One file discovered during inventory scanning."""

    path: Path
    relative_path: str
    size: int
    mtime_ns: int


@dataclass(slots=True)
class Inventory:
    """Recursive directory inventory used for planning."""

    root: Path
    directories: set[str]
    files: dict[str, FileEntry]
    total_bytes: int = 0


@dataclass(slots=True)
class SyncAction:
    """One filesystem action required to mirror the target to the source."""

    kind: ActionKind
    relative_path: str
    source: Path | None = None
    target: Path | None = None
    size: int = 0
    reason: str = ""
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.reasons and self.reason:
            self.reasons = (self.reason,)
        elif self.reasons and not self.reason:
            self.reason = self.reasons[0]


@dataclass(slots=True)
class CompareOutcome:
    """Comparison result for a single source-side file path."""

    relative_path: str
    action: SyncAction | None = None
    cache_record: CacheRecord | None = None
    compared_file_pairs: int = 0
    metadata_equal_pairs: int = 0
    cached_equal_pairs: int = 0
    hashed_file_pairs: int = 0
    identical_file_pairs: int = 0


def new_action_list() -> list[SyncAction]:
    """Return a typed empty list for dataclass defaults."""
    return []


def new_cache_entries() -> CacheEntries:
    """Return a typed empty cache mapping for dataclass defaults."""
    return {}


@dataclass(slots=True)
class PlanResult:
    """Planning output plus useful counters for summaries."""

    mode: str
    actions: list[SyncAction] = field(default_factory=new_action_list)
    cache_updates: CacheEntries = field(default_factory=new_cache_entries)
    compared_file_pairs: int = 0
    metadata_equal_pairs: int = 0
    cached_equal_pairs: int = 0
    hashed_file_pairs: int = 0
    identical_file_pairs: int = 0
    source_dir_count: int = 0
    source_file_count: int = 0
    target_dir_count: int = 0
    target_file_count: int = 0


class SyncError(RuntimeError):
    """Raised when the sync cannot proceed safely."""


def positive_int(value: str) -> int:
    """Parse a positive integer for CLI options."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Recursively mirror target directory B so it matches source A. "
            "Fast mode is the default."
        )
    )
    parser.add_argument("source_dir", help="Source directory A")
    parser.add_argument("target_dir", help="Target directory B to modify")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing anything",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt for real runs",
    )
    parser.add_argument(
        "--mode",
        choices=["fast", "exact", "rebuild"],
        default="fast",
        help="Sync strategy to use (default: fast)",
    )
    parser.add_argument(
        "--algorithm",
        default=DEFAULT_HASH_ALGORITHM,
        help=f"Hash algorithm for exact mode (default: {DEFAULT_HASH_ALGORITHM})",
    )
    parser.add_argument(
        "--chunk-size",
        type=positive_int,
        default=DEFAULT_CHUNK_SIZE,
        help=(
            "Chunk size in bytes used while hashing/copying files "
            f"(default: {DEFAULT_CHUNK_SIZE})"
        ),
    )
    parser.add_argument(
        "--jobs",
        type=positive_int,
        default=DEFAULT_JOBS,
        help=(
            f"Number of worker threads for exact-mode hashing, parallel file copying, "
            f"and parallel scanning/verification (default: {DEFAULT_JOBS})"
        ),
    )
    return parser.parse_args()


def validate_hash_algorithm(name: str) -> str:
    """Return a normalized hash algorithm name or raise on invalid input."""
    normalized = name.strip().lower()
    if not normalized:
        raise SyncError("Hash algorithm cannot be empty.")
    try:
        hashlib.new(normalized)
    except ValueError as exc:
        raise SyncError(f"Unsupported hash algorithm: {name}") from exc
    return normalized


def validate_chunk_size(chunk_size: int) -> int:
    """Ensure chunk size is a positive integer."""
    if chunk_size <= 0:
        raise SyncError("Chunk size must be a positive integer.")
    return chunk_size


def validate_jobs(jobs: int) -> int:
    """Ensure the worker count is a positive integer."""
    if jobs <= 0:
        raise SyncError("Jobs must be a positive integer.")
    return jobs


def resolve_root(path_text: str, label: str) -> Path:
    """Resolve a path and ensure it is an existing directory."""
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise SyncError(f"{label} does not exist: {path}")
    if not path.is_dir():
        raise SyncError(f"{label} is not a directory: {path}")
    return path


def is_relative_to(path: Path, other: Path) -> bool:
    """Return whether ``path`` is inside ``other``."""
    try:
        path.relative_to(other)
        return True
    except ValueError:
        return False


def validate_roots(source_root: Path, target_root: Path) -> None:
    """Refuse unsafe root combinations that could cause self-mutation."""
    if source_root == target_root:
        raise SyncError("Source and target must be different directories.")
    if is_relative_to(source_root, target_root):
        raise SyncError("Source cannot be inside target.")
    if is_relative_to(target_root, source_root):
        raise SyncError("Target cannot be inside source.")


def format_bytes(size: int) -> str:
    """Format bytes as a compact human-readable string."""
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{int(size)} B"


def format_count(value: int) -> str:
    """Format an integer with grouping for stable progress text."""
    return f"{value:,}"


def action_style(kind: str) -> str:
    """Return the style used for one action kind."""
    return ACTION_STYLES.get(kind, "bold white")


def action_badge(kind: str) -> Text:
    """Render a styled action label."""
    return Text(kind, style=action_style(kind))


def reason_style(reason: str) -> str:
    """Return a style for a reason string based on its meaning."""
    lowered = reason.lower()
    for needle, style in REASON_STYLES:
        if needle in lowered:
            return style
    return "white"


def reason_lines(action: SyncAction) -> Text:
    """Render all reasons for an action as colored bullet lines."""
    text = Text()
    reasons = action.reasons or ((action.reason,) if action.reason else ())
    for index, item in enumerate(reasons):
        if index:
            text.append("\n")
        text.append("● ", style=reason_style(item))
        text.append(item, style=reason_style(item))
    return text


def build_metric_panel(title: str, value: str, style: str, subtitle: str = "") -> Panel:
    """Build a small metric card for the dashboard summary."""
    value_text = Text(value, style=style, justify="center")
    subtitle_text = Text(subtitle if subtitle else " ", style="dim", justify="center")
    return Panel(
        Align.center(Group(value_text, subtitle_text), vertical="middle"),
        title=title,
        border_style=style,
        box=box.ROUNDED,
        padding=(0, 1),
        height=7,
        expand=True,
    )


def build_info_panel(
    title: str,
    rows: list[tuple[str, str]],
    border_style: str,
) -> Panel:
    """Build one fixed-height info panel for the header."""
    grid = Table.grid(padding=(0, 1), expand=True)
    grid.add_column(style="bold cyan", ratio=2)
    grid.add_column(style="white", ratio=5)
    for label, value in rows:
        grid.add_row(label, value)
    return Panel(
        grid,
        title=title,
        border_style=border_style,
        box=box.ROUNDED,
        padding=(0, 1),
        height=7,
        expand=True,
    )


def build_panel_grid(
    panels: list[Panel],
    columns: int,
    *,
    padding: tuple[int, int] = (0, 0),
) -> Table:
    """Lay out panels in a stable fixed-column grid."""
    grid = Table.grid(expand=True, padding=padding)
    for _ in range(columns):
        grid.add_column(ratio=1)

    for start in range(0, len(panels), columns):
        row: list[RenderableType] = list(panels[start : start + columns])
        if len(row) < columns:
            row.extend(Text("") for _ in range(columns - len(row)))
        grid.add_row(*row)
    return grid


def inventory_stats_text(
    directory_count: int, file_count: int, total_bytes: int
) -> str:
    """Render stable inventory stats for progress output."""
    return (
        f"dirs={format_count(directory_count)} "
        f"files={format_count(file_count)} "
        f"size={format_bytes(total_bytes)}"
    )


def apply_stats_text(
    removed_files: int,
    removed_dirs: int,
    added_dirs: int,
    copied_files: int,
    copied_bytes: int,
) -> str:
    """Render stable apply-phase counters."""
    return (
        f"rm_f={format_count(removed_files)} "
        f"rm_d={format_count(removed_dirs)} "
        f"add_d={format_count(added_dirs)} "
        f"copy_f={format_count(copied_files)} "
        f"bytes={format_bytes(copied_bytes)}"
    )


def build_status_progress() -> Progress:
    """Build a progress display for indeterminate scan/status work."""
    return Progress(
        SpinnerColumn(style="bold cyan", finished_text="✓"),
        TextColumn(
            f"[{'bold cyan'}]{{task.description}}",
            table_column=Column(width=18),
        ),
        TextColumn(
            "{task.fields[details]}",
            style="white",
            table_column=Column(ratio=1),
        ),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=CONSOLE,
        transient=False,
        expand=True,
    )


def build_operation_progress() -> Progress:
    """Build a progress display for counted compare/apply work."""
    return Progress(
        SpinnerColumn(style="bold cyan", finished_text="✓"),
        TextColumn(
            f"[{'bold cyan'}]{{task.description}}",
            table_column=Column(width=18),
        ),
        BarColumn(
            bar_width=None, complete_style="bright_magenta", finished_style="magenta"
        ),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn(
            "{task.fields[details]}",
            style="white",
            table_column=Column(ratio=1),
        ),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=CONSOLE,
        transient=False,
        expand=True,
    )


def build_current_action_panel(
    action: SyncAction | None,
    *,
    title: str,
    note: str,
    active_jobs: int = 0,
) -> Panel:
    """Build the current-action panel shown during apply."""
    if action is None:
        body = Table.grid(expand=True, padding=(0, 1))
        body.add_column(style="white")
        body.add_row(note)
        if active_jobs > 0:
            body.add_row(f"Active jobs: {format_count(active_jobs)}")
        return Panel(
            body,
            title=title,
            border_style="bright_blue",
            box=box.DOUBLE,
            padding=(0, 1),
        )

    body = Table.grid(expand=True, padding=(0, 1))
    body.add_column(style="bold cyan", no_wrap=True)
    body.add_column(style="white", ratio=1)
    body.add_row("Action", str(action_badge(action.kind)))
    body.add_row("Path", action.relative_path)
    body.add_row("Details", note)
    if action.reason:
        body.add_row("Reason", action.reason)
    if active_jobs > 0:
        body.add_row("Active jobs", format_count(active_jobs))
    if action.size:
        body.add_row("Size", format_bytes(action.size))

    return Panel(
        body,
        title=title,
        border_style="bright_blue",
        box=box.DOUBLE,
        padding=(0, 1),
    )


def build_job_action_panels(
    slot_actions: list[SyncAction | None],
    slot_notes: list[str],
) -> list[Panel]:
    """Build one live panel per worker slot during parallel copy."""
    panels: list[Panel] = []
    for slot_index, action in enumerate(slot_actions, start=1):
        panels.append(
            build_current_action_panel(
                action,
                title=f"[bold white]Job {slot_index}[/bold white]",
                note=slot_notes[slot_index - 1],
            )
        )
    return panels


def build_compare_progress() -> Progress:
    """Build a progress display for compare-only work without extra detail padding."""
    return Progress(
        SpinnerColumn(style="bold cyan", finished_text="✓"),
        TextColumn(
            f"[{'bold cyan'}]{{task.description}}",
            table_column=Column(width=18),
        ),
        BarColumn(
            bar_width=None, complete_style="bright_magenta", finished_style="magenta"
        ),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=CONSOLE,
        transient=False,
        expand=True,
    )


def cache_file_path() -> Path:
    """Return the on-disk cache file path for exact-mode comparisons."""
    return Path(__file__).with_name(".dir-sync-cache.json")


def cache_pair_key(source_root: Path, target_root: Path) -> str:
    """Return a stable key for one source/target root pair."""
    payload = f"{source_root}\n{target_root}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def load_pair_cache(source_root: Path, target_root: Path) -> CacheEntries:
    """Load cache entries for the current source/target pair."""
    path = cache_file_path()
    try:
        raw_obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(raw_obj, dict):
        return {}
    raw_obj = cast(dict[str, Any], raw_obj)
    pairs_obj = raw_obj.get("pairs")
    if not isinstance(pairs_obj, dict):
        return {}
    pairs_obj = cast(dict[str, Any], pairs_obj)

    pair_payload = pairs_obj.get(cache_pair_key(source_root, target_root))
    if not isinstance(pair_payload, dict):
        return {}
    pair_payload = cast(dict[str, Any], pair_payload)
    if pair_payload.get("source_root") != str(source_root):
        return {}
    if pair_payload.get("target_root") != str(target_root):
        return {}

    entries_obj = pair_payload.get("entries")
    if not isinstance(entries_obj, dict):
        return {}
    entries_obj = cast(dict[str, Any], entries_obj)

    entries: CacheEntries = {}
    for relative_path, record in entries_obj.items():
        if isinstance(record, dict):
            entries[relative_path] = cast(CacheRecord, record)
    return entries


def save_pair_cache(
    source_root: Path, target_root: Path, entries: CacheEntries
) -> None:
    """Persist cache entries for the current source/target pair."""
    path = cache_file_path()
    try:
        raw_obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw_obj, dict):
            raw_obj = {}
    except (OSError, json.JSONDecodeError):
        raw_obj = {}
    raw_obj = cast(dict[str, Any], raw_obj)

    pairs_obj = raw_obj.get("pairs")
    if not isinstance(pairs_obj, dict):
        pairs_obj = {}
    pairs_obj = cast(dict[str, Any], pairs_obj)
    raw_obj["version"] = CACHE_VERSION
    raw_obj["pairs"] = pairs_obj

    key = cache_pair_key(source_root, target_root)
    sorted_entries = {
        relative_path: entries[relative_path] for relative_path in sorted(entries)
    }
    pairs_obj[key] = {
        "source_root": str(source_root),
        "target_root": str(target_root),
        "entries": sorted_entries,
    }

    safe_save_cache(path, raw_obj)


def build_cache_record(
    source_entry: FileEntry,
    target_entry: FileEntry,
    algorithm: str,
) -> CacheRecord:
    """Build a persistent cache record for one known-identical file pair."""
    return {
        "algorithm": algorithm,
        "source_size": source_entry.size,
        "source_mtime_ns": source_entry.mtime_ns,
        "target_size": target_entry.size,
        "target_mtime_ns": target_entry.mtime_ns,
    }


def cache_record_matches(
    record: CacheRecord | None,
    source_entry: FileEntry,
    target_entry: FileEntry,
    algorithm: str,
) -> bool:
    """Return whether a cache record still matches both files exactly by metadata."""
    if not isinstance(record, dict):
        return False
    return (
        record.get("algorithm") == algorithm
        and record.get("source_size") == source_entry.size
        and record.get("source_mtime_ns") == source_entry.mtime_ns
        and record.get("target_size") == target_entry.size
        and record.get("target_mtime_ns") == target_entry.mtime_ns
    )


def file_entry_from_disk(path: Path, relative_path: str) -> FileEntry:
    """Build a FileEntry from a live path on disk."""
    stat = path.stat()
    return FileEntry(
        path=path,
        relative_path=relative_path,
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def drop_cache_for_path(cache_entries: CacheEntries, relative_path: str) -> None:
    """Drop one file cache entry, if present."""
    cache_entries.pop(relative_path, None)


def drop_cache_for_prefix(cache_entries: CacheEntries, relative_path: str) -> None:
    """Drop cache entries for all files inside a removed/replaced directory."""
    prefix = f"{relative_path}/"
    for cached_path in tuple(cache_entries):
        if cached_path == relative_path or cached_path.startswith(prefix):
            del cache_entries[cached_path]


def scan_inventory(
    root: Path,
    *,
    progress: Progress | None = None,
    task_id: TaskID | None = None,
    label: str = "directory",
) -> Inventory:
    """Build a recursive inventory of directories and regular files.

    Symlinks are rejected because mirroring their intent safely is ambiguous for
    this script's first version.
    """
    directories: set[str] = set()
    files: dict[str, FileEntry] = {}
    scanned_dirs = 0
    scanned_files = 0
    scanned_bytes = 0

    if progress is not None and task_id is not None:
        progress.update(
            task_id,
            description=f"Scanning {label}",
            details=f"{root} • starting",
        )

    for current_root, dir_names, file_names in os.walk(root, topdown=True):
        check_interrupt()
        current_path = Path(current_root)
        dir_names.sort()
        file_names.sort()

        for dir_name in dir_names:
            full_dir = current_path / dir_name
            if full_dir.is_symlink():
                raise SyncError(f"Symlinks are not supported: {full_dir}")
            rel_dir = full_dir.relative_to(root).as_posix()
            directories.add(rel_dir)
            scanned_dirs += 1

        for file_name in file_names:
            full_file = current_path / file_name
            if full_file.is_symlink():
                raise SyncError(f"Symlinks are not supported: {full_file}")
            try:
                stat = full_file.stat()
            except OSError as exc:
                raise SyncError(f"Failed to stat file: {full_file} :: {exc}") from exc
            rel_file = full_file.relative_to(root).as_posix()
            files[rel_file] = FileEntry(
                path=full_file,
                relative_path=rel_file,
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
            )
            scanned_files += 1
            scanned_bytes += stat.st_size

        if progress is not None and task_id is not None:
            progress.update(
                task_id,
                details=(
                    f"{root} • "
                    f"{inventory_stats_text(scanned_dirs, scanned_files, scanned_bytes)}"
                ),
            )

    if progress is not None and task_id is not None:
        progress.update(
            task_id,
            description=f"Scanned {label}",
            details=(
                f"{root} • "
                f"{inventory_stats_text(scanned_dirs, scanned_files, scanned_bytes)}"
            ),
            completed=1,
            total=1,
        )

    return Inventory(
        root=root,
        directories=directories,
        files=files,
        total_bytes=scanned_bytes,
    )


def hash_file(path: Path, algorithm: str, chunk_size: int) -> str:
    """Hash a file by streaming its contents."""
    check_interrupt()
    digest = hashlib.new(algorithm)
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise SyncError(f"Failed to hash file: {path} :: {exc}") from exc
    return digest.hexdigest()


def compare_source_file(
    relative_path: str,
    source_entry: FileEntry,
    target_inventory: Inventory,
    *,
    compare_mode: CompareMode,
    algorithm: str,
    chunk_size: int,
    cache_entries: CacheEntries,
) -> CompareOutcome:
    """Compare one source file against the corresponding target path."""
    if relative_path in target_inventory.directories:
        target_dir = target_inventory.root / Path(relative_path)
        return CompareOutcome(
            relative_path=relative_path,
            action=SyncAction(
                kind="REPLACE_WITH_FILE",
                relative_path=relative_path,
                source=source_entry.path,
                target=target_dir,
                size=source_entry.size,
                reasons=("type mismatch (target is a directory, source is a file)",),
            ),
        )

    target_entry = target_inventory.files.get(relative_path)
    if target_entry is None:
        return CompareOutcome(
            relative_path=relative_path,
            action=SyncAction(
                kind="COPY_FILE",
                relative_path=relative_path,
                source=source_entry.path,
                target=target_inventory.root / Path(relative_path),
                size=source_entry.size,
                reasons=("missing in target",),
            ),
        )

    outcome = CompareOutcome(relative_path=relative_path, compared_file_pairs=1)
    if source_entry.size != target_entry.size:
        outcome.action = SyncAction(
            kind="REPLACE_FILE",
            relative_path=relative_path,
            source=source_entry.path,
            target=target_entry.path,
            size=source_entry.size,
            reasons=("content differs (size mismatch)",),
        )
        return outcome

    if compare_mode == "verify":
        if source_entry.mtime_ns != target_entry.mtime_ns:
            outcome.action = SyncAction(
                kind="REPLACE_FILE",
                relative_path=relative_path,
                source=source_entry.path,
                target=target_entry.path,
                size=source_entry.size,
                reasons=("verification mismatch (mtime differs)",),
            )
            return outcome
        outcome.metadata_equal_pairs = 1
        outcome.identical_file_pairs = 1
        return outcome

    if compare_mode == "fast":
        if source_entry.mtime_ns == target_entry.mtime_ns:
            outcome.metadata_equal_pairs = 1
            outcome.identical_file_pairs = 1
            return outcome

        outcome.action = SyncAction(
            kind="REPLACE_FILE",
            relative_path=relative_path,
            source=source_entry.path,
            target=target_entry.path,
            size=source_entry.size,
            reasons=("metadata differs (mtime mismatch)",),
        )
        return outcome

    record = cache_entries.get(relative_path)
    if cache_record_matches(record, source_entry, target_entry, algorithm):
        outcome.cached_equal_pairs = 1
        outcome.identical_file_pairs = 1
        if isinstance(record, dict):
            outcome.cache_record = dict(record)
        return outcome

    outcome.hashed_file_pairs = 1
    source_hash = hash_file(source_entry.path, algorithm, chunk_size)
    target_hash = hash_file(target_entry.path, algorithm, chunk_size)
    if source_hash == target_hash:
        outcome.identical_file_pairs = 1
        outcome.cache_record = build_cache_record(source_entry, target_entry, algorithm)
        return outcome

    outcome.action = SyncAction(
        kind="REPLACE_FILE",
        relative_path=relative_path,
        source=source_entry.path,
        target=target_entry.path,
        size=source_entry.size,
        reasons=("content differs (hash mismatch)",),
    )
    return outcome


def parent_directories(relative_path: str) -> list[str]:
    """Return parent directory paths for a relative path, closest first."""
    parts = Path(relative_path).parts
    parents: list[str] = []
    for end in range(len(parts) - 1, 0, -1):
        parents.append(Path(*parts[:end]).as_posix())
    return parents


def is_within_any_directory(relative_path: str, directories: set[str]) -> bool:
    """Return whether a path is inside any directory in ``directories``."""
    return any(parent in directories for parent in parent_directories(relative_path))


def display_action_sort_key(action: SyncAction) -> tuple[str, str]:
    """Sort actions stably for plan output."""
    return action.relative_path, action.kind


def build_compare_plan(
    source_inventory: Inventory,
    target_inventory: Inventory,
    *,
    compare_mode: CompareMode,
    algorithm: str,
    chunk_size: int,
    jobs: int,
    cache_entries: CacheEntries,
    progress: Progress | None = None,
    task_id: TaskID | None = None,
) -> PlanResult:
    """Build a plan for fast, exact, or verify comparisons."""
    plan = PlanResult(
        mode=compare_mode,
        source_dir_count=len(source_inventory.directories),
        source_file_count=len(source_inventory.files),
        target_dir_count=len(target_inventory.directories),
        target_file_count=len(target_inventory.files),
    )
    target_dirs_replaced_with_file: set[str] = set()
    target_extra_dir_roots: set[str] = set()
    file_items = sorted(source_inventory.files.items())

    if progress is not None and task_id is not None:
        progress.update(
            task_id,
            description=f"Comparing files ({compare_mode})",
            total=max(1, len(file_items)),
            completed=0,
            details="",
        )

    use_threads = compare_mode == "exact" and jobs > 1 and len(file_items) > 1
    if use_threads:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            future_to_path = {
                executor.submit(
                    compare_source_file,
                    relative_path,
                    source_entry,
                    target_inventory,
                    compare_mode=compare_mode,
                    algorithm=algorithm,
                    chunk_size=chunk_size,
                    cache_entries=cache_entries,
                ): relative_path
                for relative_path, source_entry in file_items
            }
            for future in as_completed(future_to_path):
                check_interrupt()
                relative_path = future_to_path[future]
                try:
                    outcome = future.result()
                except Exception as exc:  # pragma: no cover
                    raise SyncError(
                        f"Failed while comparing {relative_path}: {exc}"
                    ) from exc
                plan.compared_file_pairs += outcome.compared_file_pairs
                plan.metadata_equal_pairs += outcome.metadata_equal_pairs
                plan.cached_equal_pairs += outcome.cached_equal_pairs
                plan.hashed_file_pairs += outcome.hashed_file_pairs
                plan.identical_file_pairs += outcome.identical_file_pairs
                if outcome.action is not None:
                    plan.actions.append(outcome.action)
                    if outcome.action.kind == "REPLACE_WITH_FILE":
                        target_dirs_replaced_with_file.add(outcome.relative_path)
                if outcome.cache_record is not None:
                    plan.cache_updates[outcome.relative_path] = outcome.cache_record
                if progress is not None and task_id is not None:
                    progress.advance(task_id)
                    progress.update(task_id, details="")
    else:
        for relative_path, source_entry in file_items:
            check_interrupt()
            outcome = compare_source_file(
                relative_path,
                source_entry,
                target_inventory,
                compare_mode=compare_mode,
                algorithm=algorithm,
                chunk_size=chunk_size,
                cache_entries=cache_entries,
            )
            plan.compared_file_pairs += outcome.compared_file_pairs
            plan.metadata_equal_pairs += outcome.metadata_equal_pairs
            plan.cached_equal_pairs += outcome.cached_equal_pairs
            plan.hashed_file_pairs += outcome.hashed_file_pairs
            plan.identical_file_pairs += outcome.identical_file_pairs
            if outcome.action is not None:
                plan.actions.append(outcome.action)
                if outcome.action.kind == "REPLACE_WITH_FILE":
                    target_dirs_replaced_with_file.add(outcome.relative_path)
            if outcome.cache_record is not None:
                plan.cache_updates[outcome.relative_path] = outcome.cache_record
            if progress is not None and task_id is not None:
                progress.advance(task_id)
                progress.update(task_id, details="")

    for relative_path in sorted(source_inventory.directories):
        if relative_path in target_inventory.files:
            target_file = target_inventory.files[relative_path]
            plan.actions.append(
                SyncAction(
                    kind="REPLACE_WITH_DIR",
                    relative_path=relative_path,
                    target=target_file.path,
                    reasons=(
                        "type mismatch (target is a file, source is a directory)",
                    ),
                )
            )
        elif relative_path not in target_inventory.directories:
            plan.actions.append(
                SyncAction(
                    kind="ADD_DIR",
                    relative_path=relative_path,
                    target=target_inventory.root / Path(relative_path),
                    reasons=("missing in target",),
                )
            )

    extra_directories = sorted(
        target_inventory.directories - source_inventory.directories
    )
    for relative_path in extra_directories:
        if relative_path in source_inventory.files:
            continue
        if relative_path in target_dirs_replaced_with_file:
            continue
        if is_within_any_directory(relative_path, target_extra_dir_roots):
            continue
        plan.actions.append(
            SyncAction(
                kind="REMOVE_DIR",
                relative_path=relative_path,
                target=target_inventory.root / Path(relative_path),
                reasons=("missing in source",),
            )
        )
        target_extra_dir_roots.add(relative_path)

    for relative_path, target_entry in sorted(target_inventory.files.items()):
        if (
            relative_path in source_inventory.files
            or relative_path in source_inventory.directories
        ):
            continue
        if is_within_any_directory(relative_path, target_extra_dir_roots):
            continue
        plan.actions.append(
            SyncAction(
                kind="REMOVE_FILE",
                relative_path=relative_path,
                target=target_entry.path,
                size=target_entry.size,
                reasons=("missing in source",),
            )
        )

    plan.actions.sort(key=display_action_sort_key)
    if progress is not None and task_id is not None:
        progress.update(
            task_id,
            description=f"Compared files ({compare_mode})",
            details="",
        )
    return plan


def build_rebuild_plan(
    source_inventory: Inventory,
    target_inventory: Inventory,
    *,
    progress: Progress | None = None,
    task_id: TaskID | None = None,
) -> PlanResult:
    """Build a plan that clears the target and recopies the source."""
    plan = PlanResult(
        mode="rebuild",
        source_dir_count=len(source_inventory.directories),
        source_file_count=len(source_inventory.files),
        target_dir_count=len(target_inventory.directories),
        target_file_count=len(target_inventory.files),
    )

    target_top_level_dirs = sorted(
        relative_path
        for relative_path in target_inventory.directories
        if len(Path(relative_path).parts) == 1
    )
    target_top_level_files = sorted(
        relative_path
        for relative_path in target_inventory.files
        if len(Path(relative_path).parts) == 1
    )
    total_steps = len(target_top_level_dirs) + len(target_top_level_files)
    total_steps += len(source_inventory.directories) + len(source_inventory.files)

    if progress is not None and task_id is not None:
        progress.update(
            task_id,
            description="Planning rebuild",
            total=max(1, total_steps),
            completed=0,
            details="clearing target and scheduling full copy",
        )

    for relative_path in target_top_level_files:
        target_entry = target_inventory.files[relative_path]
        plan.actions.append(
            SyncAction(
                kind="REMOVE_FILE",
                relative_path=relative_path,
                target=target_entry.path,
                size=target_entry.size,
                reasons=("rebuild mode (clear target contents first)",),
            )
        )
        if progress is not None and task_id is not None:
            progress.advance(task_id)

    for relative_path in target_top_level_dirs:
        plan.actions.append(
            SyncAction(
                kind="REMOVE_DIR",
                relative_path=relative_path,
                target=target_inventory.root / Path(relative_path),
                reasons=("rebuild mode (clear target contents first)",),
            )
        )
        if progress is not None and task_id is not None:
            progress.advance(task_id)

    for relative_path in sorted(source_inventory.directories):
        plan.actions.append(
            SyncAction(
                kind="ADD_DIR",
                relative_path=relative_path,
                target=target_inventory.root / Path(relative_path),
                reasons=("rebuild mode (recreate source directory tree)",),
            )
        )
        if progress is not None and task_id is not None:
            progress.advance(task_id)

    for relative_path, source_entry in sorted(source_inventory.files.items()):
        plan.actions.append(
            SyncAction(
                kind="COPY_FILE",
                relative_path=relative_path,
                source=source_entry.path,
                target=target_inventory.root / Path(relative_path),
                size=source_entry.size,
                reasons=("rebuild mode (recopy source file)",),
            )
        )
        if progress is not None and task_id is not None:
            progress.advance(task_id)

    plan.actions.sort(key=display_action_sort_key)
    if progress is not None and task_id is not None:
        progress.update(
            task_id,
            description="Planned rebuild",
            details=f"actions={format_count(len(plan.actions))}",
        )
    return plan


def render_plan(
    plan: PlanResult,
    source_root: Path,
    target_root: Path,
    *,
    dry_run: bool,
    algorithm: str,
    jobs: int,
) -> None:
    """Print a readable plan summary."""
    mode_label = "DRY RUN" if dry_run else "APPLY"
    header_grid = Table.grid(expand=True)
    header_grid.add_column(ratio=2)
    header_grid.add_column(ratio=1)
    header_grid.add_column(ratio=1)
    header_grid.add_row(
        build_info_panel(
            "Paths",
            [("Source", str(source_root)), ("Target", str(target_root))],
            "cyan",
        ),
        build_info_panel(
            "Run",
            [
                ("Run", mode_label),
                ("Strategy", plan.mode),
                ("Hash", algorithm),
                ("Jobs", str(jobs)),
            ],
            "magenta",
        ),
        build_info_panel(
            "Inventory",
            [
                ("Source dirs", format_count(plan.source_dir_count)),
                ("Source files", format_count(plan.source_file_count)),
                ("Target dirs", format_count(plan.target_dir_count)),
                ("Target files", format_count(plan.target_file_count)),
            ],
            "green",
        ),
    )

    CONSOLE.print(
        Panel(
            header_grid,
            title="[bold bright_white]Directory Sync[/bold bright_white]",
            border_style="bright_blue",
            box=box.DOUBLE,
            padding=(0, 1),
        )
    )

    if not plan.actions:
        CONSOLE.print(
            Panel(
                "[bold green]Target already matches source. No changes needed.[/bold green]",
                border_style="green",
                box=box.ROUNDED,
            )
        )
        return

    legend_text = Text()
    for index, kind in enumerate(sorted(ACTION_STYLES)):
        if index:
            legend_text.append("   ")
        legend_text.append(kind, style=action_style(kind))
    CONSOLE.print(
        Panel(
            legend_text,
            title="Action Legend",
            border_style="bright_black",
            box=box.ROUNDED,
        )
    )

    table = Table(title="Planned Actions", show_lines=True, box=box.HEAVY_HEAD)
    table.add_column("Action", style="bold cyan", no_wrap=True)
    table.add_column("Path", style="white", overflow="fold")
    table.add_column("Reasons", style="dim")
    table.add_column("Size", justify="right")

    for action in plan.actions:
        size_label = format_bytes(action.size) if action.size else ""
        table.add_row(
            action_badge(action.kind),
            action.relative_path,
            reason_lines(action),
            size_label,
        )

    CONSOLE.print(table)

    counts = Counter(action.kind for action in plan.actions)
    reason_notes = sum(
        len(action.reasons or ((action.reason,) if action.reason else ()))
        for action in plan.actions
    )
    copied_bytes = sum(
        action.size
        for action in plan.actions
        if action.kind in {"COPY_FILE", "REPLACE_FILE", "REPLACE_WITH_FILE"}
    )
    removed_bytes = sum(
        action.size for action in plan.actions if action.kind == "REMOVE_FILE"
    )

    metrics = [
        build_metric_panel(
            "Actions",
            format_count(len(plan.actions)),
            "bright_cyan",
            f"{format_count(reason_notes)} reasons",
        ),
        build_metric_panel(
            "Copy / Replace", format_bytes(copied_bytes), "bright_green"
        ),
        build_metric_panel("Remove", format_bytes(removed_bytes), "bright_red"),
    ]
    if plan.mode != "rebuild":
        metrics.extend(
            [
                build_metric_panel(
                    "Compared",
                    format_count(plan.compared_file_pairs),
                    "bright_blue",
                    "file pairs",
                ),
                build_metric_panel(
                    "Cache Hits", format_count(plan.cached_equal_pairs), "magenta"
                ),
                build_metric_panel(
                    "Hashed", format_count(plan.hashed_file_pairs), "yellow"
                ),
            ]
        )

    breakdown_panels = [
        build_metric_panel(kind, format_count(counts[kind]), action_style(kind))
        for kind in sorted(counts)
    ]

    summary_group = Group(
        build_panel_grid(metrics, columns=3),
        Panel(
            build_panel_grid(breakdown_panels, columns=4),
            title="Action Breakdown",
            border_style="bright_blue",
            box=box.ROUNDED,
            padding=(0, 1),
        ),
    )
    CONSOLE.print(
        Panel(
            summary_group,
            title="[bold white]Summary[/bold white]",
            border_style="bright_blue",
            box=box.DOUBLE,
            padding=(0, 1),
        )
    )


def _validate_source_read_only(
    source_root: Path,
    target_root: Path,
    action: SyncAction,
) -> None:
    """Assert that an action's target is under target_root and source is under source_root.

    This is a safety guard to guarantee the source directory is **never** mutated,
    even in the face of bugs or future code changes.
    """
    if action.target is not None:
        try:
            action.target.relative_to(target_root)
        except ValueError as exc:
            raise SyncError(
                f"SAFETY GUARD: target path {action.target} is not inside the "
                f"target root {target_root}. Refusing to mutate outside the "
                "designated target directory."
            ) from exc
    if action.source is not None:
        try:
            action.source.relative_to(source_root)
        except ValueError as exc:
            raise SyncError(
                f"SAFETY GUARD: source path {action.source} is not inside the "
                f"source root {source_root}. Refusing to read from outside the "
                "designated source directory."
            ) from exc


def copy_file_atomic(
    source: Path,
    target: Path,
    source_root: Path,
    target_root: Path,
    chunk_size: int,
    progress_callback: Callable[[int], None] | None = None,
) -> None:
    """Copy a file via a temp file, then replace the target atomically."""
    try:
        target.relative_to(target_root)
    except ValueError as exc:
        raise SyncError(
            f"SAFETY GUARD: target path {target} is not inside the target root "
            f"{target_root}. Refusing to write outside the designated target directory."
        ) from exc
    try:
        source.relative_to(source_root)
    except ValueError as exc:
        raise SyncError(
            f"SAFETY GUARD: source path {source} is not inside the source root "
            f"{source_root}. Refusing to read outside the designated source directory."
        ) from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=".dir-sync-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
        with source.open("rb") as src_handle, temp_path.open("wb") as dst_handle:
            while True:
                chunk = src_handle.read(chunk_size)
                if not chunk:
                    break
                dst_handle.write(chunk)
                if progress_callback is not None:
                    progress_callback(len(chunk))
        shutil.copystat(source, temp_path)
        os.replace(temp_path, target)
    except OSError as exc:
        raise SyncError(f"Failed to copy {source} -> {target} :: {exc}") from exc
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def _copy_single_file(
    action: SyncAction,
    *,
    source_root: Path,
    target_root: Path,
    chunk_size: int,
    progress_callback: Callable[[int], None] | None = None,
) -> None:
    """Copy a single file action via atomic copy (used for parallel threads)."""
    assert action.source is not None
    assert action.target is not None
    check_interrupt()
    _validate_source_read_only(source_root, target_root, action)
    action.target.parent.mkdir(parents=True, exist_ok=True)
    copy_file_atomic(
        action.source,
        action.target,
        source_root,
        target_root,
        chunk_size,
        progress_callback=progress_callback,
    )


def apply_plan(
    plan: PlanResult,
    *,
    source_root: Path,
    target_root: Path,
    chunk_size: int,
    algorithm: str,
    cache_entries: CacheEntries,
    jobs: int = 1,
) -> None:
    """Apply planned actions in a conflict-safe order.

    Shows a progress bar above live action panels. Serial phases use one focused
    panel; the parallel copy phase expands to one panel per worker slot so the
    apply UI accurately reflects multi-job runs.
    """
    remove_file_actions = sorted(
        [
            action
            for action in plan.actions
            if action.kind in {"REMOVE_FILE", "REPLACE_WITH_DIR"}
        ],
        key=display_action_sort_key,
    )
    remove_dir_actions = sorted(
        [
            action
            for action in plan.actions
            if action.kind in {"REMOVE_DIR", "REPLACE_WITH_FILE"}
        ],
        key=lambda item: len(Path(item.relative_path).parts),
        reverse=True,
    )
    add_dir_actions = sorted(
        [
            action
            for action in plan.actions
            if action.kind in {"ADD_DIR", "REPLACE_WITH_DIR"}
        ],
        key=display_action_sort_key,
    )
    copy_file_actions = sorted(
        [
            action
            for action in plan.actions
            if action.kind in {"COPY_FILE", "REPLACE_FILE", "REPLACE_WITH_FILE"}
        ],
        key=display_action_sort_key,
    )
    ordered_actions = [
        *remove_file_actions,
        *remove_dir_actions,
        *add_dir_actions,
        *copy_file_actions,
    ]

    progress = build_operation_progress()
    task_id = progress.add_task(
        "Applying sync",
        total=max(1, len(ordered_actions)),
        completed=0,
        details="",
    )

    active_jobs = 0
    current_panel = build_current_action_panel(
        None,
        title="[bold white]Current Action[/bold white]",
        note="Waiting for the first planned action.",
    )
    renderable = Group(progress, current_panel)

    live = Live(renderable, console=CONSOLE, refresh_per_second=10, transient=False)
    live.start()

    def _update_renderable(panels: list[Panel]) -> None:
        """Refresh the live layout with one or more panels."""
        nonlocal renderable
        if len(panels) == 1:
            renderable = Group(progress, panels[0])
        else:
            renderable = Group(
                progress,
                build_panel_grid(
                    panels,
                    columns=min(2, len(panels)),
                    padding=(0, 1),
                ),
            )
        live.update(renderable)

    def _set_current_action(
        action: SyncAction | None,
        *,
        note: str,
        active_count: int,
    ) -> None:
        """Update the single current-action panel and refresh the live view."""
        nonlocal current_panel
        current_panel = build_current_action_panel(
            action,
            title="[bold white]Current Action[/bold white]",
            note=note,
            active_jobs=active_count,
        )
        _update_renderable([current_panel])

    def _copy_interrupt_callback(_: int) -> None:
        """Poll for cancellation between copied chunks."""
        check_interrupt()

    for action in remove_file_actions:
        check_interrupt()
        _validate_source_read_only(source_root, target_root, action)
        assert action.target is not None
        _set_current_action(action, note="Removing file…", active_count=1)
        try:
            action.target.unlink(missing_ok=True)
        except OSError as exc:
            raise SyncError(f"Failed to remove file: {action.target} :: {exc}") from exc
        drop_cache_for_path(cache_entries, action.relative_path)
        progress.advance(task_id)

    for action in remove_dir_actions:
        check_interrupt()
        _validate_source_read_only(source_root, target_root, action)
        assert action.target is not None
        _set_current_action(action, note="Removing directory…", active_count=1)
        if action.target.exists():
            try:
                shutil.rmtree(action.target)
            except OSError as exc:
                raise SyncError(
                    f"Failed to remove directory: {action.target} :: {exc}"
                ) from exc
        drop_cache_for_prefix(cache_entries, action.relative_path)
        progress.advance(task_id)

    for action in add_dir_actions:
        check_interrupt()
        _validate_source_read_only(source_root, target_root, action)
        assert action.target is not None
        _set_current_action(action, note="Creating directory…", active_count=1)
        try:
            action.target.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise SyncError(
                f"Failed to create directory: {action.target} :: {exc}"
            ) from exc
        drop_cache_for_path(cache_entries, action.relative_path)
        progress.advance(task_id)

    for action in copy_file_actions:
        check_interrupt()

    use_parallel_copy = jobs > 1 and len(copy_file_actions) > 1
    if use_parallel_copy:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            future_to_work: dict[Any, tuple[SyncAction, int]] = {}
            pending_actions = iter(copy_file_actions)
            slot_count = min(jobs, len(copy_file_actions))
            slot_actions: list[SyncAction | None] = [None] * slot_count
            slot_notes: list[str] = ["Waiting for work." for _ in range(slot_count)]

            def _render_job_panels() -> None:
                """Render the current per-job panel grid."""
                _update_renderable(build_job_action_panels(slot_actions, slot_notes))

            def _submit_next(slot: int) -> bool:
                """Submit the next copy action for one slot, if any remain."""
                nonlocal active_jobs
                try:
                    next_action = next(pending_actions)
                except StopIteration:
                    slot_actions[slot] = None
                    slot_notes[slot] = "Completed — no more files assigned."
                    _render_job_panels()
                    return False

                check_interrupt()
                _validate_source_read_only(source_root, target_root, next_action)
                active_jobs += 1
                slot_actions[slot] = next_action
                slot_notes[slot] = "Copying file…"
                _render_job_panels()
                future = executor.submit(
                    _copy_single_file,
                    next_action,
                    source_root=source_root,
                    target_root=target_root,
                    chunk_size=chunk_size,
                    progress_callback=_copy_interrupt_callback,
                )
                future_to_work[future] = (next_action, slot)
                return True

            _render_job_panels()
            for slot in range(slot_count):
                _submit_next(slot)

            while future_to_work:
                check_interrupt()
                completed_future = next(as_completed(tuple(future_to_work)))
                action, slot = future_to_work.pop(completed_future)
                try:
                    completed_future.result()
                except Exception as exc:
                    raise SyncError(
                        f"Failed while copying {action.relative_path}: {exc}"
                    ) from exc
                source_entry = file_entry_from_disk(
                    cast(Path, action.source), action.relative_path
                )
                target_entry = file_entry_from_disk(
                    cast(Path, action.target), action.relative_path
                )
                cache_entries[action.relative_path] = build_cache_record(
                    source_entry,
                    target_entry,
                    algorithm,
                )
                active_jobs = max(0, active_jobs - 1)
                slot_actions[slot] = None
                slot_notes[slot] = "Completed copy."
                _render_job_panels()
                progress.advance(task_id)
                _submit_next(slot)
    else:
        for action in copy_file_actions:
            check_interrupt()
            _validate_source_read_only(source_root, target_root, action)
            assert action.source is not None
            assert action.target is not None
            _set_current_action(action, note="Copying file…", active_count=1)

            copy_file_atomic(
                action.source,
                action.target,
                source_root,
                target_root,
                chunk_size,
                progress_callback=_copy_interrupt_callback,
            )

            source_entry = file_entry_from_disk(action.source, action.relative_path)
            target_entry = file_entry_from_disk(action.target, action.relative_path)
            cache_entries[action.relative_path] = build_cache_record(
                source_entry,
                target_entry,
                algorithm,
            )

            progress.advance(task_id)

    progress.update(task_id, description="Applied sync")
    if not use_parallel_copy:
        _set_current_action(
            None,
            note="All planned sync actions completed.",
            active_count=0,
        )
    live.stop()


def confirm_apply(target_root: Path, *, assume_yes: bool) -> bool:
    """Confirm a destructive run unless the caller opted out."""
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        raise SyncError("Non-interactive apply requires --yes.")
    return Confirm.ask(
        f"Apply these changes to [bold]{target_root}[/bold]?",
        default=False,
        console=CONSOLE,
    )


def build_plan_for_mode(
    mode: SyncMode,
    source_inventory: Inventory,
    target_inventory: Inventory,
    *,
    algorithm: str,
    chunk_size: int,
    jobs: int,
    cache_entries: CacheEntries,
    progress: Progress | None = None,
    task_id: TaskID | None = None,
) -> PlanResult:
    """Dispatch to the appropriate planner for the selected mode."""
    if mode == "rebuild":
        return build_rebuild_plan(
            source_inventory,
            target_inventory,
            progress=progress,
            task_id=task_id,
        )
    return build_compare_plan(
        source_inventory,
        target_inventory,
        compare_mode=mode,
        algorithm=algorithm,
        chunk_size=chunk_size,
        jobs=jobs,
        cache_entries=cache_entries,
        progress=progress,
        task_id=task_id,
    )


def main() -> int:
    """Program entry point."""
    try:
        install_interrupt_handler()
        args = parse_args()
        mode: SyncMode = args.mode
        algorithm = validate_hash_algorithm(args.algorithm)
        chunk_size = validate_chunk_size(args.chunk_size)
        jobs = validate_jobs(args.jobs)
        source_root = resolve_root(args.source_dir, "Source")
        target_root = resolve_root(args.target_dir, "Target")
        validate_roots(source_root, target_root)
        cleanup_stale_temp_files(target_root)

        cache_entries = load_pair_cache(source_root, target_root)

        if args.dry_run:
            CONSOLE.print(
                "[bold yellow]DRY RUN — no files will be copied or "
                "removed. Use the plan output to preview changes, then "
                "re-run without --dry-run to apply them.[/bold yellow]"
            )

        if jobs > 1:
            # pylint: disable=consider-using-f-string
            CONSOLE.print(
                "[dim]--jobs {} — parallelism enabled for scanning, copying, "
                "exact-mode hashing, and verification. "
                "I/O bandwidth may limit gains when both directories are "
                "on the same drive or a slow drive.[/dim]".format(jobs)
            )

        with build_status_progress() as progress:
            source_task = progress.add_task(
                "Scanning source", total=1, completed=0, details="queued"
            )
            target_task = progress.add_task(
                "Scanning target", total=1, completed=0, details="queued"
            )
            if jobs > 1:
                with ThreadPoolExecutor(max_workers=jobs) as scan_executor:
                    source_future = scan_executor.submit(
                        scan_inventory,
                        source_root,
                        progress=progress,
                        task_id=source_task,
                        label="source",
                    )
                    target_future = scan_executor.submit(
                        scan_inventory,
                        target_root,
                        progress=progress,
                        task_id=target_task,
                        label="target",
                    )
                    source_inventory = source_future.result()
                    target_inventory = target_future.result()
            else:
                source_inventory = scan_inventory(
                    source_root,
                    progress=progress,
                    task_id=source_task,
                    label="source",
                )
                target_inventory = scan_inventory(
                    target_root,
                    progress=progress,
                    task_id=target_task,
                    label="target",
                )

        with build_compare_progress() as progress:
            plan_task = progress.add_task(
                "Planning sync",
                total=max(1, len(source_inventory.files)),
                completed=0,
                details="queued",
            )
            plan = build_plan_for_mode(
                mode,
                source_inventory,
                target_inventory,
                algorithm=algorithm,
                chunk_size=chunk_size,
                jobs=jobs,
                cache_entries=cache_entries,
                progress=progress,
                task_id=plan_task,
            )

        cache_entries.update(plan.cache_updates)
        save_pair_cache(source_root, target_root, cache_entries)

        render_plan(
            plan,
            source_root,
            target_root,
            dry_run=args.dry_run,
            algorithm=algorithm,
            jobs=jobs,
        )

        if args.dry_run or not plan.actions:
            return 0

        if not confirm_apply(target_root, assume_yes=args.yes):
            CONSOLE.print("[yellow]Cancelled. No changes were written.[/yellow]")
            return 0

        apply_plan(
            plan,
            source_root=source_root,
            target_root=target_root,
            chunk_size=chunk_size,
            algorithm=algorithm,
            cache_entries=cache_entries,
            jobs=jobs,
        )

        save_pair_cache(source_root, target_root, cache_entries)
        CONSOLE.print("\n[bold green]Sync applied. Verifying target...[/bold green]")

        with build_status_progress() as progress:
            verify_source_task = progress.add_task(
                "Rescanning source",
                total=1,
                completed=0,
                details="queued",
            )
            verify_target_task = progress.add_task(
                "Rescanning target",
                total=1,
                completed=0,
                details="queued",
            )
            if jobs > 1:
                with ThreadPoolExecutor(max_workers=jobs) as verify_scan_executor:
                    verify_source_future = verify_scan_executor.submit(
                        scan_inventory,
                        source_root,
                        progress=progress,
                        task_id=verify_source_task,
                        label="source",
                    )
                    verify_target_future = verify_scan_executor.submit(
                        scan_inventory,
                        target_root,
                        progress=progress,
                        task_id=verify_target_task,
                        label="target",
                    )
                    verified_source = verify_source_future.result()
                    verified_target = verify_target_future.result()
            else:
                verified_source = scan_inventory(
                    source_root,
                    progress=progress,
                    task_id=verify_source_task,
                    label="source",
                )
                verified_target = scan_inventory(
                    target_root,
                    progress=progress,
                    task_id=verify_target_task,
                    label="target",
                )

        with build_compare_progress() as progress:
            verify_mode: CompareMode = "exact" if mode == "exact" else "verify"
            verify_task = progress.add_task(
                "Verifying target",
                total=max(1, len(verified_source.files)),
                completed=0,
                details="queued",
            )
            verified_plan = build_compare_plan(
                verified_source,
                verified_target,
                compare_mode=verify_mode,
                algorithm=algorithm,
                chunk_size=chunk_size,
                jobs=jobs,
                cache_entries=(cache_entries if verify_mode == "exact" else {}),
                progress=progress,
                task_id=verify_task,
            )

        if verified_plan.actions:
            CONSOLE.print(
                "[red]Verification failed: target still differs from source.[/red]"
            )
            render_plan(
                verified_plan,
                source_root,
                target_root,
                dry_run=False,
                algorithm=algorithm,
                jobs=1,
            )
            return 2

        CONSOLE.print("[green]Verification passed. Target now matches source.[/green]")
        return 0
    except SyncError as exc:
        CONSOLE.print(f"[red]Error:[/red] {exc}")
        return 1
    except KeyboardInterrupt:
        CONSOLE.print("\n[yellow]Cancelled by user.[/yellow]")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
