#!/usr/bin/env python3
"""
mkv_cleaner.py — TUI/CLI for cleaning MKV audio/subtitle tracks safely.

Why this tool exists:
    Batch anime releases often ship with extra commentary/sign tracks and
    inconsistent default-track flags. This script makes those cleanups repeatable
    across many files, with previews and guardrails to reduce accidental data loss.

Design notes:
    - Uses mkvmerge JSON as the source of truth for track metadata.
    - Supports both inspect-first TUI workflow and automation-friendly CLI mode.
    - Defaults favor safety (dry-run support, single-track protection, optional
        in-place replacement).

Requirements:
  pip install textual rich
  MKVToolNix installed (provides mkvmerge)
    macOS:   brew install mkvtoolnix
    Ubuntu:  sudo apt install mkvtoolnix
    Windows: https://mkvtoolnix.download/downloads.html

Usage:
    # Interactive TUI (default)
    python mkv_cleaner.py
    python mkv_cleaner.py /path/to/file.mkv
    python mkv_cleaner.py /path/to/folder

    # Non-interactive CLI mode
    python mkv_cleaner.py --cli --inspect /path/to/file.mkv
    python mkv_cleaner.py --cli --dry-run --keep-audio eng jpn /path/to/folder
"""
# pylint: disable=too-many-lines
# pylint: disable=too-many-instance-attributes,too-many-arguments
# pylint: disable=too-many-positional-arguments,too-many-locals
# pylint: disable=too-many-branches,too-many-statements

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from operator import attrgetter
from pathlib import Path
from typing import TypedDict, cast

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.command import CommandPalette, DiscoveryHit, Hit, Hits, Provider
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.css.query import NoMatches
from textual.events import Key
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets._data_table import ColumnKey, RowKey
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
    TabbedContent,
    TabPane,
)


# ── Data types ────────────────────────────────────────────────────────────────


ConfigDict = dict[str, object]
JsonRichCell = dict[str, str | bool]
TableCell = str | Text
InspectorRow = list[TableCell]

_SUBPROCESS_PROBE_TIMEOUT_SECONDS = 60.0
_SUBPROCESS_WRITE_TIMEOUT_SECONDS = 300.0
_SUBPROCESS_MAX_ATTEMPTS = 2


class FormConfig(TypedDict):
    """Persisted form values stored in the local JSON config."""

    # TypedDict is a static-typing aid only: at runtime this is still
    # an ordinary dict. The benefit is editor/type-checker guidance for
    # expected keys and value shapes when reading or writing config.

    path: str
    recursive: bool
    keep_audio_langs: str
    keep_sub_langs: str
    no_subs: bool
    remove_named: str
    default_audio: str
    default_subs: str
    auto_default: bool
    fix_missing_default: bool
    sync_title_to_filename: bool
    protect_single_audio: bool
    protect_single_sub: bool
    dry_run: bool
    in_place: bool
    save_log_file: bool
    jobs: int
    auto_scroll: bool
    use_selection: bool
    selected_paths: list[str]
    upper_panel_height: int
    log_panel_height: int


class RunConfig(TypedDict):
    """Normalized runtime options used by inspect/run workers."""

    # Keeping a dedicated runtime config shape (separate from FormConfig)
    # lets the app convert loose UI strings (e.g., "eng jpn") into
    # canonical structures (e.g., ["eng", "jpn"]) before worker threads use them.

    path: str
    recursive: bool
    keep_audio_langs: list[str]
    keep_sub_langs: list[str]
    no_subs: bool
    remove_named: list[str]
    default_audio: str | None
    default_subs: str | None
    auto_default: bool
    fix_missing_default: bool
    sync_title_to_filename: bool
    protect_single_audio: bool
    protect_single_sub: bool
    dry_run: bool
    in_place: bool
    save_log_file: bool
    jobs: int
    use_selection: bool
    selected_paths: list[str]


def _empty_removed() -> list[tuple[str, int, str, str]]:
    """Return an empty default list for removed-track tuples."""
    return []


def _empty_default_changes() -> list[tuple[str, int, str, str, bool, bool]]:
    """Return an empty default list for default-flag change tuples."""
    return []


@dataclass
class TrackInfo:
    """Track metadata parsed from mkvmerge JSON output."""

    tid: int
    ttype: str  # video / audio / subtitles
    lang: str
    name: str
    codec: str
    default: bool
    forced: bool


@dataclass
class FileSummary:
    """Per-file processing outcome used by the run summary view."""

    path: Path
    skipped: bool = False
    errored: bool = False
    src_size: int = 0
    dst_size: int = 0
    removed: list[tuple[str, int, str, str]] = field(default_factory=_empty_removed)
    default_changes: list[tuple[str, int, str, str, bool, bool]] = field(
        default_factory=_empty_default_changes
    )


class RunSummary(TypedDict):
    """Computed keep/remove/default decisions for one source file."""

    audio_keep: list[TrackInfo]
    audio_removed: list[TrackInfo]
    subs_keep: list[TrackInfo]
    subs_removed: list[TrackInfo]
    audio_defaults: dict[int, bool]
    sub_defaults: dict[int, bool]
    defaults_changed: bool
    title_target: str | None
    title_changed: bool


def _empty_track_list() -> list[TrackInfo]:
    """Return an empty default list for TrackInfo values."""
    return []


def _empty_log_messages() -> list[tuple[str, str]]:
    """Return an empty default list for log message tuples."""
    return []


@dataclass
class RunWorkerOutcome:
    """Thread worker result payload consumed by the UI thread."""

    path: Path
    result: FileSummary
    summary: RunSummary
    state: str
    before_size: int
    after_size: int | None = None
    error: str | None = None
    display_tracks: list[TrackInfo] = field(default_factory=_empty_track_list)
    display_size: str = "?"
    display_prefix: str = ""
    log_messages: list[tuple[str, str]] = field(default_factory=_empty_log_messages)


# ── Config persistence ────────────────────────────────────────────────────────


def _config_path() -> Path:
    """Return the path to the JSON config file for persisting options."""
    return Path(__file__).with_name(".mkv-cleaner-config.json")


def _track_cache_path() -> Path:
    """Return the path to the persistent track metadata cache file."""
    return Path(__file__).with_name(".mkv-cleaner-track-cache.json")


def _track_to_payload(track: TrackInfo) -> dict[str, object]:
    """Serialize TrackInfo for disk caching."""
    return {
        "tid": track.tid,
        "ttype": track.ttype,
        "lang": track.lang,
        "name": track.name,
        "codec": track.codec,
        "default": track.default,
        "forced": track.forced,
    }


def _track_from_payload(payload: Mapping[str, object]) -> TrackInfo:
    """Deserialize TrackInfo from disk cache payload."""
    raw_tid = payload.get("tid", 0)
    if isinstance(raw_tid, bool):
        # bool is a subclass of int in Python, so we branch first to keep
        # intent explicit and avoid surprising acceptance of True/False.
        tid = int(raw_tid)
    elif isinstance(raw_tid, int):
        tid = raw_tid
    elif isinstance(raw_tid, str):
        tid = int(raw_tid)
    else:
        raise ValueError("Invalid track id in cache payload")

    return TrackInfo(
        tid=tid,
        ttype=str(payload.get("ttype", "")),
        lang=str(payload.get("lang", "und")),
        name=str(payload.get("name", "")),
        codec=str(payload.get("codec", "")),
        default=bool(payload.get("default", False)),
        forced=bool(payload.get("forced", False)),
    )


def _save_config(config: ConfigDict) -> None:
    """Write the config dict to the JSON config file."""
    path = _config_path()
    try:
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    except OSError:
        pass  # Silently ignore write failures


def _load_config() -> ConfigDict:
    """Read and return the config dict from the JSON config file."""
    path = _config_path()
    try:
        data: object = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            raw = cast(dict[object, object], data)
            cfg: ConfigDict = {}
            for key, value in raw.items():
                cfg[str(key)] = value
            return cfg
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _cell_to_json(cell: TableCell) -> str | JsonRichCell:
    """Serialize a DataTable cell for JSON config storage."""
    if isinstance(cell, Text):
        style = cell.style
        if not isinstance(style, str):
            style = str(style)
        return {"__rich__": True, "text": str(cell), "style": style or ""}
    return str(cell)


def _cell_from_json(cell: object) -> TableCell:
    """Restore a DataTable cell from JSON config storage."""
    if isinstance(cell, dict):
        raw = cast(dict[object, object], cell)
        cell_map: dict[str, object] = {}
        for key, value in raw.items():
            cell_map[str(key)] = value
        if cell_map.get("__rich__"):
            # Sentinel key keeps round-tripping robust: plain dict-looking
            # strings are not mistaken for rich-styled payload objects.
            raw_text = cell_map.get("text")
            raw_style = cell_map.get("style")
            text = "" if raw_text is None else str(raw_text)
            style = "" if raw_style is None else str(raw_style)
            return Text(text, style=style)
        return str(cell_map)
    if isinstance(cell, str):
        return cell
    return str(cell)


def _inspect_config_hash(cfg: Mapping[str, object]) -> str:
    """Hash filter/options (excluding path) to detect stale inspector data."""
    payload = {k: v for k, v in cfg.items() if k != "path"}
    return json.dumps(payload, sort_keys=True)


def default_jobs() -> int:
    """Return a safe default worker count for metadata-heavy operations."""
    cpu = os.cpu_count() or 4
    return max(1, min(8, cpu))


def resolve_jobs(raw: str | int | None, fallback: int | None = None) -> int:
    """Parse and clamp configured worker count."""
    default = fallback if fallback is not None else default_jobs()
    if isinstance(raw, int):
        parsed = raw
    elif isinstance(raw, str):
        token = raw.strip()
        if not token:
            return default
        try:
            parsed = int(token)
        except ValueError:
            return default
    else:
        return default
    return max(1, min(16, parsed))


def check_optional_tools() -> dict[str, bool]:
    """Return availability of optional tools used for fast-path operations."""
    return {
        "mkvpropedit": shutil.which("mkvpropedit") is not None,
    }


def _track_selector_map(tracks: list[TrackInfo]) -> dict[int, str]:
    """Map mkvmerge track IDs to mkvpropedit selectors (e.g., track:a1)."""
    selector_by_tid: dict[int, str] = {}
    counts = {"video": 0, "audio": 0, "subtitles": 0}
    for t in tracks:
        # mkvpropedit selectors are positional per track type, so we
        # count each media kind separately instead of reusing global track IDs.
        if t.ttype not in counts:
            continue
        counts[t.ttype] += 1
        if t.ttype == "video":
            selector = f"track:v{counts['video']}"
        elif t.ttype == "audio":
            selector = f"track:a{counts['audio']}"
        else:
            selector = f"track:s{counts['subtitles']}"
        selector_by_tid[t.tid] = selector
    return selector_by_tid


def build_mkvpropedit_cmd(
    target: Path,
    tracks: list[TrackInfo],
    summary: RunSummary,
    title: str | None = None,
) -> list[str]:
    """Build mkvpropedit command for default-flag-only updates."""
    selector_by_tid = _track_selector_map(tracks)
    cmd = ["mkvpropedit", str(target)]

    if title is not None:
        cmd += ["--edit", "info", "--set", f"title={title}"]

    for t in summary["audio_keep"]:
        new_default = summary["audio_defaults"].get(t.tid, t.default)
        # Skipping unchanged flags keeps command lines shorter and
        # reduces no-op writes that can still touch file metadata timestamps.
        if new_default == t.default:
            continue
        selector = selector_by_tid.get(t.tid)
        if selector is None:
            continue
        cmd += [
            "--edit",
            selector,
            "--set",
            f"flag-default={'1' if new_default else '0'}",
        ]

    for t in summary["subs_keep"]:
        new_default = summary["sub_defaults"].get(t.tid, t.default)
        if new_default == t.default:
            continue
        selector = selector_by_tid.get(t.tid)
        # Missing selectors are ignored defensively so a partial map
        # never crashes the run loop for one imperfect file.
        if selector is None:
            continue
        cmd += [
            "--edit",
            selector,
            "--set",
            f"flag-default={'1' if new_default else '0'}",
        ]

    return cmd


# ── Core logic ─────────────────────────────────────


def check_dependencies() -> bool:
    """Return True when mkvmerge is available on PATH."""
    return shutil.which("mkvmerge") is not None


def _run_subprocess_with_timeout(
    cmd: list[str],
    *,
    timeout: float,
    max_attempts: int = _SUBPROCESS_MAX_ATTEMPTS,
) -> subprocess.CompletedProcess[str]:
    """Run subprocess calls defensively so one bad probe cannot stall the session.

    We retry a small number of times because mkvtoolnix calls can fail
    transiently on busy disks/network shares, but we keep retries bounded so
    batch runs still fail fast on persistent errors.
    """
    attempts = max(1, max_attempts)
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            last_error = RuntimeError(
                (
                    f"Command timed out after {timeout:.0f}s "
                    f"(attempt {attempt}/{attempts}): {' '.join(cmd)}"
                )
            )
        except OSError as exc:
            last_error = RuntimeError(
                (
                    f"Command launch failed (attempt {attempt}/{attempts}): "
                    f"{' '.join(cmd)} :: {exc}"
                )
            )

        if attempt < attempts:
            continue

    assert last_error is not None
    raise last_error


def _extract_title_from_identification(data: Mapping[str, object]) -> str | None:
    """Best-effort extraction of the Matroska segment title from mkvmerge JSON."""
    candidate_roots: list[object] = [
        data.get("container"),
        data.get("properties"),
        data.get("segment_info"),
        data.get("segmentInfo"),
        data.get("info"),
    ]

    for root in candidate_roots:
        if not isinstance(root, Mapping):
            continue

        root_map = cast(Mapping[str, object], root)

        raw_title = root_map.get("title")
        if isinstance(raw_title, str):
            title = raw_title.strip()
            if title:
                return title

        raw_props = root_map.get("properties")
        if isinstance(raw_props, Mapping):
            props = cast(Mapping[str, object], raw_props)
            raw_title = props.get("title")
            if isinstance(raw_title, str):
                title = raw_title.strip()
                if title:
                    return title

    return None


def get_tracks_and_title(mkv_path: Path) -> tuple[list[TrackInfo], str | None]:
    """Read track info and title from an MKV file using mkvmerge JSON output."""
    result = _run_subprocess_with_timeout(
        ["mkvmerge", "-J", str(mkv_path)],
        timeout=_SUBPROCESS_PROBE_TIMEOUT_SECONDS,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr.strip())
    data = json.loads(result.stdout)
    tracks: list[TrackInfo] = []
    for t in data.get("tracks", []):
        props = t.get("properties", {})
        tracks.append(
            TrackInfo(
                tid=t["id"],
                ttype=t["type"],
                lang=props.get("language", "und"),
                name=props.get("track_name", ""),
                codec=props.get("codec_id", t.get("codec", "")),
                default=bool(props.get("default_track", False)),
                forced=bool(props.get("forced_track", False)),
            )
        )
    return tracks, _extract_title_from_identification(cast(Mapping[str, object], data))


def get_tracks(mkv_path: Path) -> list[TrackInfo]:
    """Read track info from an MKV file using mkvmerge JSON output."""
    tracks, _title = get_tracks_and_title(mkv_path)
    return tracks


def fmt_size(b: int) -> str:
    """Format bytes as MB/GB text for logs and tables."""
    mb = b / 1_048_576
    return f"{mb / 1024:.2f} GB" if mb >= 1024 else f"{mb:.1f} MB"


def fmt_delta(saved: int, src: int) -> str:
    """Format signed size delta and percent saved vs source size."""
    sign = "-" if saved >= 0 else "+"
    pct = abs(saved) / src * 100 if src else 0
    return f"{sign}{fmt_size(abs(saved))} ({pct:.1f}%)"


def name_matches(track_name: str, needles: list[str]) -> bool:
    """Return True if any case-insensitive token appears in track_name."""
    tl = track_name.lower()
    return any(n.lower() in tl for n in needles)


def lang_matches(track_lang: str, wanted: list[str]) -> bool:
    """Return True if track_lang matches any wanted language code."""
    return track_lang.lower() in [w.lower() for w in wanted]


def collect_files(path: Path, recursive: bool) -> list[Path]:
    """Collect MKV files from a file or directory input path."""
    if path.is_file():
        if path.suffix.lower() != ".mkv":
            raise ValueError(f"Not an MKV file: {path}")
        return [path]
    if path.is_dir():
        pattern = "**/*.mkv" if recursive else "*.mkv"
        files = sorted(path.glob(pattern))
        if not files:
            raise ValueError(f"No .mkv files found in: {path}")
        return files
    raise ValueError(f"Path not found: {path}")


def collect_files_from_selection(paths: list[Path], recursive: bool) -> list[Path]:
    """Collect MKV files from a mixed list of selected files/directories."""
    if not paths:
        raise ValueError("No selected files or folders.")

    resolved: set[Path] = set()
    for selected in paths:
        if selected.is_file():
            if selected.suffix.lower() == ".mkv":
                resolved.add(selected)
            continue
        if selected.is_dir():
            pattern = "**/*.mkv" if recursive else "*.mkv"
            for mkv in selected.glob(pattern):
                if mkv.is_file() and mkv.suffix.lower() == ".mkv":
                    resolved.add(mkv)

    files = sorted(resolved)
    if not files:
        raise ValueError("Selection contains no .mkv files.")
    return files


def resolve_default_flags(
    tracks: list[TrackInfo],
    keep_ids: set[int],
    ttype: str,
    auto_default: bool,
    fix_missing_default: bool,
    default_lang: str | None,
    default_name: str | None,
) -> dict[int, bool]:
    """Resolve post-filter default flags while preserving player-friendly behavior.

    Priority order:
      1) Explicit user override by name/language.
      2) Existing defaults when still valid.
      3) Optional auto-repair when a default was removed or missing.

    This keeps the user's intent first, while preventing files from ending up
    with no default audio/subtitle when auto-fix options are enabled.
    """
    type_tracks = [t for t in tracks if t.ttype == ttype]
    kept = [t for t in type_tracks if t.tid in keep_ids]
    if not kept:
        return {}
    flags: dict[int, bool] = {t.tid: t.default for t in kept}

    explicit_default_tid: int | None = None
    if default_name:
        match = next((t for t in kept if default_name.lower() in t.name.lower()), None)
        if match:
            explicit_default_tid = match.tid
    if explicit_default_tid is None and default_lang:
        match = next((t for t in kept if lang_matches(t.lang, [default_lang])), None)
        if match:
            explicit_default_tid = match.tid

    if explicit_default_tid is not None:
        for tid in flags:
            flags[tid] = tid == explicit_default_tid
        return flags

    original_default = next((t for t in type_tracks if t.default), None)
    default_removed = (
        original_default is not None and original_default.tid not in keep_ids
    )
    no_default_ever = original_default is None

    if (auto_default and default_removed) or (fix_missing_default and no_default_ever):
        if not any(flags[t.tid] for t in kept):
            flags[kept[0].tid] = True

    return flags


def split_default(val: str | None) -> tuple[str | None, str | None]:
    """Split default selector into language-code or name matching mode."""
    if val and len(val) <= 3 and val.isalpha():
        return val, None
    return None, val


def _desired_output_title(src: Path, dst: Path, in_place: bool) -> str | None:
    """Return the title that should match the final on-disk filename."""
    title = (src if in_place else dst).stem.strip()
    return title or None


def build_mkvmerge_cmd(
    src: Path,
    dst: Path,
    tracks: list[TrackInfo],
    options: Mapping[str, object],
    title: str | None = None,
) -> tuple[list[str], RunSummary]:
    """Build the mkvmerge command plus a UI/CLI-readable change plan.

    Returning both artifacts keeps execution and reporting in sync: the same
    decision set drives the actual command, dry-run output, and run summaries.
    That avoids "preview says X, run did Y" drift.
    """
    keep_audio_langs = cast(list[str], options.get("keep_audio_langs", []))
    keep_sub_langs = cast(list[str], options.get("keep_sub_langs", []))
    no_subs = bool(options.get("no_subs", False))
    remove_named = cast(list[str], options.get("remove_named", []))
    auto_default = bool(options.get("auto_default", True))
    fix_missing_default = bool(options.get("fix_missing_default", True))
    default_audio = cast(str | None, options.get("default_audio"))
    default_subs = cast(str | None, options.get("default_subs"))
    protect_single_audio = bool(options.get("protect_single_audio", True))
    protect_single_sub = bool(options.get("protect_single_sub", True))

    audio_keep: list[TrackInfo] = []
    sub_keep: list[TrackInfo] = []
    video_keep: list[TrackInfo] = []
    skipped_audio: list[TrackInfo] = []
    skipped_subs: list[TrackInfo] = []

    for t in tracks:
        if remove_named and t.name and name_matches(t.name, remove_named):
            (skipped_audio if t.ttype == "audio" else skipped_subs).append(t)
            continue
        if t.ttype == "video":
            video_keep.append(t)
        elif t.ttype == "audio":
            if not keep_audio_langs or lang_matches(t.lang, keep_audio_langs):
                audio_keep.append(t)
            else:
                skipped_audio.append(t)
        elif t.ttype == "subtitles":
            if no_subs:
                skipped_subs.append(t)
            elif not keep_sub_langs or lang_matches(t.lang, keep_sub_langs):
                sub_keep.append(t)
            else:
                skipped_subs.append(t)

    if protect_single_audio:
        all_audio = [t for t in tracks if t.ttype == "audio"]
        if len(all_audio) == 1:
            sole = all_audio[0]
            if sole in skipped_audio:
                skipped_audio.remove(sole)
                audio_keep.append(sole)

    if protect_single_sub:
        all_subs = [t for t in tracks if t.ttype == "subtitles"]
        if len(all_subs) == 1:
            sole = all_subs[0]
            if sole in skipped_subs:
                skipped_subs.remove(sole)
                sub_keep.append(sole)

    da_lang, da_name = split_default(default_audio)
    ds_lang, ds_name = split_default(default_subs)

    audio_keep_ids = {t.tid for t in audio_keep}
    sub_keep_ids = {t.tid for t in sub_keep}

    audio_defaults = resolve_default_flags(
        tracks,
        audio_keep_ids,
        "audio",
        auto_default,
        fix_missing_default,
        da_lang,
        da_name,
    )
    sub_defaults = resolve_default_flags(
        tracks,
        sub_keep_ids,
        "subtitles",
        auto_default,
        fix_missing_default,
        ds_lang,
        ds_name,
    )

    cmd = ["mkvmerge", "-o", str(dst)]
    if video_keep:
        cmd += ["--video-tracks", ",".join(str(t.tid) for t in video_keep)]
    else:
        cmd += ["--no-video"]
    if audio_keep:
        cmd += ["--audio-tracks", ",".join(str(t.tid) for t in audio_keep)]
        for t in audio_keep:
            flag = "1" if audio_defaults.get(t.tid, t.default) else "0"
            cmd += ["--default-track-flag", f"{t.tid}:{flag}"]
    else:
        cmd += ["--no-audio"]
    if sub_keep:
        cmd += ["--subtitle-tracks", ",".join(str(t.tid) for t in sub_keep)]
        for t in sub_keep:
            flag = "1" if sub_defaults.get(t.tid, t.default) else "0"
            cmd += ["--default-track-flag", f"{t.tid}:{flag}"]
    else:
        cmd += ["--no-subtitles"]
    if title is not None:
        cmd += ["--title", title]
    cmd.append(str(src))

    audio_flags_changed = any(
        audio_defaults.get(t.tid, t.default) != t.default for t in audio_keep
    )
    sub_flags_changed = any(
        sub_defaults.get(t.tid, t.default) != t.default for t in sub_keep
    )

    summary: RunSummary = {
        "audio_keep": audio_keep,
        "audio_removed": skipped_audio,
        "subs_keep": sub_keep,
        "subs_removed": skipped_subs,
        "audio_defaults": audio_defaults,
        "sub_defaults": sub_defaults,
        "defaults_changed": audio_flags_changed or sub_flags_changed,
        "title_target": title,
        "title_changed": title is not None,
    }
    return cmd, summary


def build_preview_tracks_from_summary(
    tracks: list[TrackInfo],
    summary: RunSummary,
) -> list[TrackInfo]:
    """Build projected post-run tracks from an input file and run summary plan."""
    audio_keep_ids = {t.tid for t in summary["audio_keep"]}
    sub_keep_ids = {t.tid for t in summary["subs_keep"]}
    audio_defaults = summary["audio_defaults"]
    sub_defaults = summary["sub_defaults"]

    preview: list[TrackInfo] = []
    for track in tracks:
        if track.ttype == "audio":
            if track.tid not in audio_keep_ids:
                continue
            new_default = audio_defaults.get(track.tid, track.default)
            if new_default == track.default:
                preview.append(track)
                continue
            preview.append(
                TrackInfo(
                    tid=track.tid,
                    ttype=track.ttype,
                    lang=track.lang,
                    name=track.name,
                    codec=track.codec,
                    default=new_default,
                    forced=track.forced,
                )
            )
            continue

        if track.ttype == "subtitles":
            if track.tid not in sub_keep_ids:
                continue
            new_default = sub_defaults.get(track.tid, track.default)
            if new_default == track.default:
                preview.append(track)
                continue
            preview.append(
                TrackInfo(
                    tid=track.tid,
                    ttype=track.ttype,
                    lang=track.lang,
                    name=track.name,
                    codec=track.codec,
                    default=new_default,
                    forced=track.forced,
                )
            )
            continue

        preview.append(track)

    return preview


# ── Command palette ───────────────────────────────────────────────────────────


class LabeledThemeProvider(Provider):
    """Theme list that marks the active theme in the command palette."""

    def _theme_entries(self) -> list[tuple[str, Callable[[], None], bool]]:
        current = self.app.theme
        themes = self.app.available_themes

        def set_app_theme(name: str) -> None:
            """Switch app theme from palette callbacks without leaking provider state."""
            self.app.theme = name

        return [
            (theme.name, partial(set_app_theme, theme.name), theme.name == current)
            for theme in sorted(themes.values(), key=attrgetter("name"))
        ]

    @staticmethod
    def _display_name(name: str, is_current: bool) -> str:
        return f"✓ {name} (current)" if is_current else name

    async def discover(self) -> Hits:
        """List every theme and label the active one for quick visual confirmation."""
        for name, callback, is_current in self._theme_entries():
            yield DiscoveryHit(
                self._display_name(name, is_current),
                callback,
                text=name,
                help=f"Active theme — {name}" if is_current else f"Switch to {name}",
            )

    async def search(self, query: str) -> Hits:
        """Search themes by both raw and decorated names so active-theme labels still match."""
        matcher = self.matcher(query)
        for name, callback, is_current in self._theme_entries():
            display = self._display_name(name, is_current)
            match = max(matcher.match(name), matcher.match(display))
            if match > 0:
                yield Hit(
                    match,
                    matcher.highlight(display),
                    callback,
                    help=f"Active theme — {name}"
                    if is_current
                    else f"Switch to {name}",
                )


# ── TUI App ───────────────────────────────────────────────────────────────────


class StatusBar(Static):
    """Bottom status bar for short, high-signal run and validation feedback."""

    def set_status(self, msg: str, style: str = "dim") -> None:
        """Render a styled status message in the bottom status bar."""
        self.update(Text(msg, style=style))


class HelpScreen(ModalScreen[None]):
    """A modal help screen with a close button."""

    CSS = """
    HelpScreen {
        align: center middle;
        background: $background 60%;
    }

    #help-dialog {
        width: auto;
        max-width: 92%;
        height: auto;
        max-height: 80%;
        background: $surface;
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
        padding: 0 1;
        margin-bottom: 1;
    }

    #help-close {
        width: 100%;
        dock: bottom;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("ctrl+c", "dismiss", "Close"),
        Binding("ctrl+q", "dismiss", "Close"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="help-dialog"):
            yield Label("MKV Cleaner Help", id="help-title")
            with ScrollableContainer(id="help-content"):
                yield Static("""
  PATH & RECURSIVE
    • Enter a file or folder path to process
    • Enable 'Search subfolders' to scan recursively

  FILTERS
    • Keep audio (langs):  Space-separated language codes to keep
    • Keep subs (langs):   Space-separated language codes to keep
    • Strip all subtitles: Remove ALL subtitle tracks
    • Remove tracks named: Remove tracks whose name contains text

  DEFAULTS
    • Force default audio: Set default track by language or name
    • Force default sub:   Same for subtitle tracks
    • Auto-promote:        Promote next track to default when
                           the original default is removed
    • Fix missing:         Set first track as default when no
                           existing default track is found

    METADATA
        • Sync title to filename: Keep the MKV title aligned with
                                                            the file name on disk

  TRACK PROTECTION
    • Protect single audio: Keep the only audio track even if
                            it would otherwise be removed
    • Protect single sub:   Same for the only subtitle track

  OUTPUT
    • Dry run:  Preview changes without writing files
    • In-place: Replace originals (use with caution)
    • Save log: Write detailed logs to .mkv-cleaner-tui.log

  KEYBOARD SHORTCUTS
    • Ctrl+I: Inspect tracks     • Ctrl+R: Run processing
    • Ctrl+L: Clear log          • Ctrl+O: Reset options
    • Ctrl+H: Open this help     • Ctrl+Q: Quit
    • Ctrl+E: Open file explorer
    • Ctrl+↑/↓: Move split (↑ grow log, ↓ grow upper)
    • Esc:    Cancel run/inspect when active
    • Esc:    Close this dialog  • [ / ]: Narrow/widen column

  BUTTONS
    • Inspect:  Analyse files without modifying
    • Run:      Process files based on current settings
    • Explorer: Select files/folders with multi-select
    • Cancel:   Stop active run/inspect
    • Reset:    Restore all options to defaults
    • Help:     Open this help dialog""")
            yield Button("Close (Esc)", id="help-close", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Dismiss the help modal when the close button is pressed."""
        if event.button.id == "help-close":
            self.dismiss()


class PathExplorerScreen(ModalScreen[tuple[str, list[str]] | None]):
    """Explorer-like picker for drilling directories and multi-selecting paths."""

    _SEL_TABLE = "#explorer-table"

    CSS = """
    PathExplorerScreen {
        align: center middle;
        background: $background 60%;
    }

    #explorer-dialog {
        width: 76%;
        height: 74%;
        background: $surface;
        border: thick $accent;
        padding: 1 2;
    }

    #explorer-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }

    #explorer-current {
        margin-bottom: 1;
        color: $text-muted;
    }

    #explorer-table {
        height: 1fr;
        margin-bottom: 1;
    }

    #explorer-selected {
        margin-bottom: 0;
        color: $text-muted;
    }

    #explorer-keybinds {
        margin-top: 1;
        margin-bottom: 1;
        color: $text-muted;
    }

    #explorer-buttons {
        height: auto;
        layout: horizontal;
    }

    #explorer-buttons Button {
        width: 1fr;
        margin-right: 1;
    }

    #explorer-buttons Button:last-child {
        margin-right: 0;
    }
    """

    BINDINGS = [
        Binding("left", "go_up", "Up Dir"),
        Binding("right", "drill_down", "Open Dir"),
        Binding("enter", "drill_down", "Open Dir"),
        Binding("backspace", "go_up", "Up Dir"),
        Binding("space", "mark_selection", "Toggle Select"),
        Binding("ctrl+a", "select_all_visible", "Select All Visible"),
        Binding("ctrl+d", "clear_visible", "Clear Visible"),
        Binding("ctrl+enter", "apply", "Apply Selection"),
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(
        self,
        start_path: str,
        initial_selected: list[str] | None = None,
    ) -> None:
        """Start the explorer at a sensible directory and preload selections."""
        super().__init__()
        start = Path(start_path).expanduser() if start_path else Path.cwd()
        if start.exists() and start.is_file():
            start = start.parent
        if not start.exists() or not start.is_dir():
            start = Path.cwd()
        self._current_dir = start
        self._selected: set[str] = {
            str(Path(p).expanduser()) for p in (initial_selected or [])
        }
        self._entries: list[tuple[Path, str, bool]] = []

    def compose(self) -> ComposeResult:
        """Build the explorer modal with the table, legend, and action buttons."""
        with Vertical(id="explorer-dialog"):
            yield Label("Path Explorer", id="explorer-title")
            yield Static("", id="explorer-current")
            yield DataTable(id="explorer-table", cursor_type="row", zebra_stripes=True)
            yield Static("", id="explorer-selected")
            yield Static(
                "↑/↓ Move  •  ← Up dir  •  → Open dir  •  Space Toggle select  •  "
                "Ctrl+A Select all visible  •  "
                "Ctrl+D Clear visible  •  Ctrl+Enter Apply  •  Esc Cancel",
                id="explorer-keybinds",
            )
            with Horizontal(id="explorer-buttons"):
                yield Button("Toggle", id="explorer-toggle")
                yield Button("Apply", id="explorer-apply", variant="success")
                yield Button("Cancel", id="explorer-cancel", variant="error")

    def on_mount(self) -> None:
        """Create explorer table columns and load the first directory listing."""
        table = cast(DataTable[TableCell], self.query_one(self._SEL_TABLE, DataTable))
        table.add_column("Sel", width=4)
        table.add_column("Name", width=56)
        table.add_column("Type", width=14)
        self._refresh_directory()

    def _set_cursor_row(self, row: int) -> None:
        """Best-effort cursor restoration after DataTable refresh."""
        table = cast(DataTable[TableCell], self.query_one(self._SEL_TABLE, DataTable))
        if not self._entries:
            return
        target = max(0, min(row, len(self._entries) - 1))
        try:
            table.move_cursor(row=target, column=0, animate=False)
            return
        except (TypeError, AttributeError):
            pass

    def _refresh_directory(self, focus_path: Path | None = None) -> None:
        """Reload visible entries for the current directory and restore focus."""
        table = cast(DataTable[TableCell], self.query_one(self._SEL_TABLE, DataTable))
        previous_entry = self._current_entry()
        previous_focus = previous_entry[0] if previous_entry is not None else None
        self.query_one("#explorer-current", Static).update(
            f"Current: {self._current_dir}"
        )

        entries: list[tuple[Path, str, bool]] = []
        entries.append((self._current_dir, "current", False))
        parent = self._current_dir.parent
        if parent != self._current_dir:
            entries.append((parent, "parent", True))

        try:
            children = sorted(
                self._current_dir.iterdir(),
                key=lambda p: (not p.is_dir(), p.name.lower()),
            )
        except OSError:
            children = []
        for child in children:
            kind = "dir" if child.is_dir() else "file"
            entries.append((child, kind, False))

        self._entries = entries
        table.clear()
        target_path = focus_path or previous_focus
        target_row = 0
        for path, kind, is_nav in self._entries:
            selected = "✓" if str(path) in self._selected and not is_nav else ""
            if kind == "current":
                name = "."
                type_label = "current dir"
            elif kind == "parent":
                name = ".."
                type_label = "parent dir"
            else:
                name = path.name
                type_label = "directory" if kind == "dir" else "file"
            table.add_row(selected, name, type_label)
            if target_path is not None and path == target_path:
                target_row = len(table.rows) - 1

        self._update_selected_label()
        self._set_cursor_row(target_row)

    def _current_entry(self) -> tuple[Path, str, bool] | None:
        """Return the currently highlighted explorer row, if any."""
        table = cast(DataTable[TableCell], self.query_one(self._SEL_TABLE, DataTable))
        row = table.cursor_row
        if row < 0 or row >= len(self._entries):
            return None
        return self._entries[row]

    def _update_selected_label(self) -> None:
        """Refresh the short selected-item count shown under the explorer table."""
        count = len(self._selected)
        self.query_one("#explorer-selected", Static).update(
            (
                "Selected paths: none"
                if count == 0
                else f"Selected paths: {count} (files and/or directories)"
            )
        )

    def _toggle_path_selection(self, path: Path, is_nav: bool) -> None:
        """Toggle one file or directory in the explorer selection set."""
        if is_nav:
            return
        key = str(path)
        if key in self._selected:
            self._selected.remove(key)
        else:
            self._selected.add(key)
        self._refresh_directory(focus_path=path)

    def action_drill_down(self) -> None:
        """Open the highlighted directory-like row inside the explorer."""
        entry = self._current_entry()
        if entry is None:
            return
        path, kind, _ = entry
        if kind in {"dir", "parent"}:
            self._current_dir = path
            self._refresh_directory()

    def action_go_up(self) -> None:
        """Move to the parent directory and reselect the folder we just left."""
        previous_dir = self._current_dir
        parent = self._current_dir.parent
        if parent != self._current_dir:
            self._current_dir = parent
            self._refresh_directory(focus_path=previous_dir)

    def action_mark_selection(self) -> None:
        """Toggle selection for the currently highlighted non-navigation row."""
        entry = self._current_entry()
        if entry is None:
            return
        path, _kind, is_nav = entry
        self._toggle_path_selection(path, is_nav)

    def action_select_all_visible(self) -> None:
        """Add every visible file and directory in the current view to the selection."""
        for path, kind, is_nav in self._entries:
            if is_nav:
                continue
            if kind in {"file", "dir", "current"}:
                self._selected.add(str(path))
        self._refresh_directory()

    def action_clear_visible(self) -> None:
        """Remove every visible file and directory in the current view from selection."""
        visible = {str(path) for path, _kind, is_nav in self._entries if not is_nav}
        self._selected.difference_update(visible)
        self._refresh_directory()

    def action_cancel(self) -> None:
        """Close the explorer without applying any selection changes."""
        self.dismiss(None)

    def action_apply(self) -> None:
        """Apply the current explorer selection and close the modal."""
        self._apply()

    def _apply(self) -> None:
        """Return the current directory plus selected paths to the parent screen."""
        selected = sorted(self._selected)
        self.dismiss((str(self._current_dir), selected))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Map explorer button presses to the matching modal actions."""
        button_id = event.button.id
        if button_id == "explorer-toggle":
            self.action_mark_selection()
        elif button_id == "explorer-apply":
            self._apply()
        elif button_id == "explorer-cancel":
            self.action_cancel()


class MkvCleanerApp(App[None]):
    """MKV Cleaner — TUI for removing unwanted audio/subtitle tracks."""

    TITLE = "MKV Cleaner"
    SUB_TITLE = "Audio & Subtitle Track Manager"
    _SUMMARY_LABEL_STYLE = "bold white"
    _SUMMARY_SECTION_STYLE = "bold magenta"
    _SUMMARY_ERROR_STYLE = "bold red"
    _SUMMARY_SUCCESS_STYLE = "bold green"
    _SUMMARY_MUTED_STYLE = "dim"

    _SEL_TRACK_TABLE = "#track-table"
    _SEL_TABS = "#tabs"
    _SEL_SUMMARY_LOG = "#summary-log"
    _SEL_INPUT_PATH = "#inp-path"
    _SEL_INPUT_KEEP_SUBS = "#inp-keep-subs"
    _SEL_SWITCH_USE_SELECTION = "#sw-use-selection"
    _SEL_SWITCH_AUTO_SCROLL = "#sw-auto-scroll"
    _SEL_SELECTION_SUMMARY = "#selection-summary"
    _SEL_SELECTION_DETAILS = "#selection-details"
    _SEL_UPPER_PANE = "#upper-pane"
    _SEL_LOG_PANE = "#log-pane"
    _SEL_INS_FILTER_SEARCH = "#ins-filter-search"
    _SEL_INS_FILTER_TYPES = "#ins-filter-types"
    _SEL_INS_FILTER_LANGS = "#ins-filter-langs"
    _SEL_INS_SORT_BY = "#ins-sort-by"
    _SEL_INS_SORT_DESC = "#sw-ins-sort-desc"
    _SEL_INS_FILTER_DEFAULT = "#sw-ins-default"
    _SEL_INS_FILTER_FORCED = "#sw-ins-forced"
    _SEL_INS_FILTER_SUMMARY = "#ins-filter-summary"
    _INS_SORT_KEYS = {"file", "track", "type", "lang", "codec", "name", "flags"}
    _INS_SORT_OPTIONS: list[tuple[str, str]] = [
        ("File", "file"),
        ("Track", "track"),
        ("Type", "type"),
        ("Lang", "lang"),
        ("Codec", "codec"),
        ("Name", "name"),
        ("Flags", "flags"),
    ]

    CSS = """
    Screen {
        background: $surface;
    }

    #dep-warning {
        background: $error-darken-2;
        color: $error;
        text-style: bold;
        padding: 0 2;
        display: none;
    }

    #dep-warning.visible {
        display: block;
    }

    #workspace {
        height: 1fr;
        layout: horizontal;
    }

    #sidebar {
        width: 44;
        min-width: 36;
        height: 1fr;
        layout: vertical;
        border-right: tall $panel-lighten-1;
        background: $panel-darken-1;
        padding: 1 0 0 1;
        overflow-x: hidden;
        overflow-y: auto;
    }

    .control-card {
        width: 100%;
        height: auto;
        border: round $panel-lighten-1;
        background: $surface;
        padding: 0 1 1 1;
        margin-bottom: 1;
        margin-right: 1;
        margin-left: 0;
    }

    .section-title {
        text-style: bold;
        color: $accent;
        margin-top: 1;
        margin-bottom: 0;
    }

    .field-label {
        color: $text-muted;
        margin-top: 1;
        margin-bottom: 0;
    }

    .switch-row {
        layout: horizontal;
        height: 3;
        align: left middle;
    }

    .switch-row Label {
        width: 1fr;
        padding-top: 1;
    }

    #selection-summary,
    #selection-details {
        color: $text-muted;
    }

    #selection-details {
        margin-top: 1;
    }

    #btn-explorer {
        width: 100%;
        margin-top: 1;
    }

    #btn-row {
        layout: vertical;
        height: auto;
        margin-top: 1;
    }

    #btn-row Button {
        width: 100%;
        margin-bottom: 1;
        text-align: center;
    }

    #btn-row Button:last-child {
        margin-bottom: 0;
    }

    #right-pane {
        width: 1fr;
        height: 1fr;
        background: $surface;
    }

    #upper-pane {
        height: 1fr;
        padding: 0;
    }

    TabbedContent {
        height: 1fr;
        border: round $panel-lighten-1;
        background: $surface-darken-1;
    }

    TabPane {
        padding: 0 1;
    }

    #inspect-header {
        height: auto;
        padding: 0 1;
        color: $text-muted;
        text-style: bold;
        align: left middle;
        border: round $panel-lighten-1;
    }

    #inspect-header.live {
        color: $success;
    }

    #inspect-filter-row {
        height: auto;
        layout: horizontal;
    }

    #ins-filter-search {
        width: 3fr;
        margin-right: 1;
    }

    #ins-filter-types,
    #ins-filter-langs {
        width: 1fr;
        margin-right: 1;
    }

    #ins-sort-by {
        width: 20;
        margin-right: 1;
    }

    #inspect-sort-desc-row {
        width: auto;
        height: 3;
        layout: horizontal;
        align: left middle;
        margin-right: 1;
    }

    #inspect-sort-desc-row Label {
        width: auto;
        padding-top: 1;
        margin-right: 1;
    }

    #inspect-filter-default-row,
    #inspect-filter-forced-row {
        width: auto;
        height: 3;
        layout: horizontal;
        align: left middle;
        margin-right: 1;
    }

    #inspect-filter-default-row Label,
    #inspect-filter-forced-row Label {
        width: auto;
        padding-top: 1;
        margin-right: 1;
    }

    #btn-ins-filter-clear {
        min-width: 14;
    }

    #ins-filter-summary {
        color: $text-muted;
        margin-top: 1;
    }

    #resize-hint {
        height: auto;
        margin-top: 1;
        color: $text-muted;
        border: round $panel-lighten-1;
        padding: 0 1;
        align: left middle;
    }

    #track-table {
        height: 1fr;
        border: round $panel-lighten-1;
    }

    #tab-summary {
        padding: 1;
    }

    #summary-log {
        height: 1fr;
        border: round $panel-lighten-1;
        background: $surface-darken-1;
        padding: 1 2;
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

    #progress-label {
        margin-left: 0;
    }

    #log-pane {
        height: 12;
        border-top: tall $panel-lighten-1;
        background: $surface-darken-1;
    }

    #log-label {
        height: 1;
        background: $panel;
        padding: 0 1;
        text-style: bold;
        color: $text-muted;
    }

    #main-log {
        height: 1fr;
        background: $surface-darken-1;
        padding: 0 1;
    }

    #status-bar {
        height: 1;
        background: $panel-darken-1;
        padding: 0 1;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+i", "inspect", "Inspect", show=True),
        Binding("ctrl+r", "run", "Run", show=True),
        Binding("ctrl+e", "explorer", "Explorer", show=True),
        Binding("ctrl+l", "clear_log", "Clear Log"),
        Binding("ctrl+o", "reset", "Reset", show=True),
        Binding("ctrl+h", "help", "Help", show=True),
        Binding("ctrl+up", "move_split_up", "Move split up", show=False, priority=True),
        Binding(
            "ctrl+down",
            "move_split_down",
            "Move split down",
            show=False,
            priority=True,
        ),
        Binding("[", "narrow_col", "Narrow col", show=False),
        Binding("]", "widen_col", "Widen col", show=False),
    ]

    # ── Reactive state ─────────────────────────────────────────────────────────

    is_busy: reactive[bool] = reactive(False)

    _COL_NAMES = ["File", "Track", "Type", "Lang", "Codec", "Name", "Flags"]
    _COL_DEFAULTS = [40, 5, 9, 4, 16, 16, 10]  # File width overridden on mount/resize

    def __init__(
        self,
        initial_path: str = "",
        initial_overrides: dict[str, object] | None = None,
    ) -> None:
        super().__init__()
        self._initial_path = initial_path
        self._initial_overrides = initial_overrides or {}
        self._files: list[Path] = []
        self._results: list[FileSummary] = []
        self._file_logger = self._build_logger()
        self._save_log_file = False
        self._optional_tools = check_optional_tools()
        self._track_cache: dict[
            str, tuple[int, int, list[TrackInfo], str | None, int]
        ] = {}
        self._track_cache_lock = threading.Lock()
        self._track_cache_dirty = False
        self._track_cache_max_entries = 5000
        self._track_cache_ttl_seconds = 14 * 24 * 60 * 60
        self._load_track_cache()
        self._cache_stats_lock = threading.Lock()
        self._cache_hits = 0
        self._cache_misses = 0
        self._active_jobs = default_jobs()
        self._progress_update_interval = 0.2
        self._pending_inspector_autofit = 0
        self._selected_paths: set[str] = set()
        self._run_detail_batch_size = 10
        self._row_batch_size = 16
        self._row_flush_interval = 0.35
        self._live_detail_interval = 1.0
        self._live_log_interval = 2.0

        self._col_keys: list[ColumnKey] = []
        self._col_widths: list[int] = list(self._COL_DEFAULTS)
        self._user_fixed_cols: set[int] = set()
        self._inspect_rows: list[InspectorRow] = []
        self._inspect_source_rows: list[InspectorRow] = []
        self._file_row_offsets: dict[str, list[int]] = {}
        self._file_row_keys: dict[str, list[RowKey]] = {}
        self._last_inspected_path: Path | None = None
        self._last_inspected_source_label: str | None = None
        self._last_inspected_time: float | None = None
        self._last_inspect_config_hash: str | None = None
        self._cancel_event = threading.Event()
        self._active_processes: set[subprocess.Popen[str]] = set()
        self._active_process_lock = threading.Lock()
        self._upper_panel_height = 18
        self._log_panel_height = 12

    @staticmethod
    def _log_path() -> Path:
        return Path(__file__).with_name(".mkv-cleaner-tui.log")

    def _build_logger(self) -> logging.Logger:
        """Create a per-app logger used when log-file output is enabled."""
        logger_name = f"mkv_cleaner.{id(self)}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        return logger

    def _ensure_file_logger(self) -> None:
        """Attach the file handler the first time log-file output is used."""
        if any(
            isinstance(handler, logging.FileHandler)
            for handler in self._file_logger.handlers
        ):
            return

        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        try:
            handler = logging.FileHandler(self._log_path(), encoding="utf-8")
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(formatter)
            self._file_logger.addHandler(handler)
        except OSError:
            # If file logging fails, the TUI log widget still works.
            pass

    # ── Compose ────────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        """Build a split layout that keeps controls visible while runs stream live updates."""
        yield Header()

        yield Static(
            "⚠  mkvmerge not found — install MKVToolNix first",
            id="dep-warning",
        )

        with Horizontal(id="workspace"):
            with ScrollableContainer(id="sidebar"):
                with Vertical(classes="control-card"):
                    yield Label("INPUT SOURCE", classes="section-title")
                    yield Input(
                        placeholder="/path/to/file.mkv or folder",
                        id="inp-path",
                        value=self._initial_path,
                    )
                    with Container(classes="switch-row"):
                        yield Label("Use explorer selection")
                        yield Switch(id="sw-use-selection", value=False)
                    yield Static("Mode: path input", id="selection-summary")
                    yield Static("Explorer selection: none", id="selection-details")
                    yield Button(
                        "Explorer / Select Files", id="btn-explorer", variant="primary"
                    )
                    yield Label("Recursive", classes="field-label")
                    with Container(classes="switch-row"):
                        yield Label("Search subfolders")
                        yield Switch(id="sw-recursive", value=False)

                with Vertical(classes="control-card"):
                    yield Label("FILTERS", classes="section-title")
                    yield Label("Keep audio (langs)", classes="field-label")
                    yield Input(placeholder="e.g. jpn eng", id="inp-keep-audio")
                    yield Label("Keep subs (langs)", classes="field-label")
                    yield Input(placeholder="e.g. eng", id="inp-keep-subs")
                    yield Label("Strip all subtitles", classes="field-label")
                    with Container(classes="switch-row"):
                        yield Label("No subtitles")
                        yield Switch(id="sw-no-subs", value=False)
                    yield Label("Remove tracks named", classes="field-label")
                    yield Input(
                        placeholder="e.g. Signs Commentary", id="inp-remove-named"
                    )

                with Vertical(classes="control-card"):
                    yield Label("DEFAULTS", classes="section-title")
                    yield Label("Force default audio", classes="field-label")
                    yield Input(
                        placeholder="lang or track name", id="inp-default-audio"
                    )
                    yield Label("Force default sub", classes="field-label")
                    yield Input(placeholder="lang or track name", id="inp-default-subs")
                    yield Label("Auto-promote default", classes="field-label")
                    with Container(classes="switch-row"):
                        yield Label("When default removed")
                        yield Switch(id="sw-auto-default", value=True)
                    yield Label("Fix missing default", classes="field-label")
                    with Container(classes="switch-row"):
                        yield Label("Assign if none set")
                        yield Switch(id="sw-fix-defaults", value=True)

                with Vertical(classes="control-card"):
                    yield Label("METADATA", classes="section-title")
                    yield Label("Sync title to filename", classes="field-label")
                    with Container(classes="switch-row"):
                        yield Label("Keep MKV title aligned")
                        yield Switch(id="sw-sync-title", value=True)

                with Vertical(classes="control-card"):
                    yield Label("SAFETY", classes="section-title")
                    yield Label("Protect single audio", classes="field-label")
                    with Container(classes="switch-row"):
                        yield Label("Keep if only 1 track")
                        yield Switch(id="sw-protect-audio", value=True)
                    yield Label("Protect single sub", classes="field-label")
                    with Container(classes="switch-row"):
                        yield Label("Keep if only 1 track")
                        yield Switch(id="sw-protect-sub", value=True)

                with Vertical(classes="control-card"):
                    yield Label("OUTPUT", classes="section-title")
                    yield Label("Dry run", classes="field-label")
                    with Container(classes="switch-row"):
                        yield Label("No files written")
                        yield Switch(id="sw-dry-run", value=False)
                    yield Label("In-place", classes="field-label")
                    with Container(classes="switch-row"):
                        yield Label("Replace originals")
                        yield Switch(id="sw-in-place", value=False)
                    yield Label("Save log file", classes="field-label")
                    with Container(classes="switch-row"):
                        yield Label("Write .mkv-cleaner-tui.log")
                        yield Switch(id="sw-save-log", value=False)
                    yield Label("Parallel jobs", classes="field-label")
                    yield Input(
                        placeholder="e.g. 6",
                        id="inp-jobs",
                        value=str(default_jobs()),
                    )
                    yield Label("Auto-scroll", classes="field-label")
                    with Container(classes="switch-row"):
                        yield Label("Scroll to new content")
                        yield Switch(id="sw-auto-scroll", value=True)

                with Vertical(classes="control-card"):
                    with Horizontal(id="btn-row"):
                        yield Button("Inspect", id="btn-inspect", variant="default")
                        yield Button("Run", id="btn-run", variant="success")
                        yield Button("Cancel (Esc/Q)", id="btn-cancel", variant="error")
                        yield Button("Reset", id="btn-reset", variant="default")
                        yield Button("Help", id="btn-help", variant="default")

            with Vertical(id="right-pane"):
                with Vertical(id="upper-pane"):
                    with TabbedContent(id="tabs"):
                        with TabPane("Track Inspector", id="tab-inspect"):
                            yield Static("", id="inspect-header")
                            with Horizontal(id="inspect-filter-row"):
                                yield Input(
                                    placeholder="Filter text (file, type, id, codec, name)",
                                    id="ins-filter-search",
                                )
                                yield Input(
                                    placeholder="Types (audio subtitles video)",
                                    id="ins-filter-types",
                                )
                                yield Input(
                                    placeholder="Langs (jpn eng)",
                                    id="ins-filter-langs",
                                )
                                with Container(id="inspect-filter-default-row"):
                                    yield Label("Default")
                                    yield Switch(id="sw-ins-default", value=False)
                                with Container(id="inspect-filter-forced-row"):
                                    yield Label("Forced")
                                    yield Switch(id="sw-ins-forced", value=False)
                                yield Select(
                                    self._INS_SORT_OPTIONS,
                                    value="file",
                                    allow_blank=False,
                                    prompt="Sort by",
                                    id="ins-sort-by",
                                )
                                with Container(id="inspect-sort-desc-row"):
                                    yield Label("Desc")
                                    yield Switch(id="sw-ins-sort-desc", value=False)
                                yield Button(
                                    "Clear Filters",
                                    id="btn-ins-filter-clear",
                                    variant="default",
                                )
                            yield Static(
                                "Filters: showing all rows", id="ins-filter-summary"
                            )
                            yield Static(
                                "← → move cell   [ ] narrow/widen columns   "
                                "Ctrl+↑/↓ move split (↑ log bigger, ↓ upper bigger)",
                                id="resize-hint",
                                markup=False,
                            )
                            yield DataTable(
                                id="track-table", zebra_stripes=True, cursor_type="cell"
                            )

                        with TabPane("Summary", id="tab-summary"):
                            yield RichLog(id="summary-log", highlight=True, markup=True)

                with Horizontal(id="progress-row"):
                    yield ProgressBar(
                        id="progress-bar", show_eta=False, show_percentage=False
                    )
                    yield Static("", id="progress-label")
                    yield Static("", id="progress-eta")

                with Vertical(id="log-pane"):
                    yield Static("OUTPUT LOG", id="log-label")
                    yield RichLog(id="main-log", highlight=True, markup=True, wrap=True)

        yield StatusBar(
            "Ready — configure options and press Inspect or Run", id="status-bar"
        )
        yield Footer()

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        """Initialize dependency state, config restore, and inspector startup."""
        if not check_dependencies():
            self.query_one("#dep-warning").add_class("visible")
            self.query_one("#btn-run").disabled = True
        self.query_one("#btn-cancel").disabled = True

        self._restore_config()
        self._apply_initial_overrides()

        self.query_one(self._SEL_INPUT_PATH, Input).focus()
        self._update_selection_summary()

        self._log_event(
            (
                "Application ready. "
                f"optional_tools={self._optional_tools} "
                f"initial_path={'<none>' if not self._initial_path else self._initial_path}"
            ),
            "INFO",
        )

        self.call_after_refresh(self._init_inspector)
        self.call_after_refresh(self._apply_panel_heights)

        # CLI path overrides cache — always re-inspect
        if self._initial_path:
            self.call_after_refresh(self.action_inspect)

    def on_unmount(self) -> None:
        """Flush persistent caches when the app is closed."""
        self._persist_track_cache(force=True)

    def _apply_initial_overrides(self) -> None:
        """Apply CLI-provided field overrides on top of restored config values."""
        if not self._initial_overrides:
            return

        input_mapping = {
            "path": self._SEL_INPUT_PATH,
            "keep_audio_langs": "inp-keep-audio",
            "keep_sub_langs": self._SEL_INPUT_KEEP_SUBS,
            "remove_named": "inp-remove-named",
            "default_audio": "inp-default-audio",
            "default_subs": "inp-default-subs",
            "jobs": "inp-jobs",
        }
        switch_mapping = {
            "recursive": "sw-recursive",
            "no_subs": "sw-no-subs",
            "auto_default": "sw-auto-default",
            "fix_missing_default": "sw-fix-defaults",
            "sync_title_to_filename": "sw-sync-title",
            "protect_single_audio": "sw-protect-audio",
            "protect_single_sub": "sw-protect-sub",
            "dry_run": "sw-dry-run",
            "in_place": "sw-in-place",
            "save_log_file": "sw-save-log",
            "auto_scroll": self._SEL_SWITCH_AUTO_SCROLL,
        }

        for key, selector in input_mapping.items():
            if key not in self._initial_overrides:
                continue
            raw = self._initial_overrides.get(key)
            if raw is None:
                continue
            if selector.startswith("#"):
                input_selector = selector
            else:
                input_selector = f"#{selector}"
            self.query_one(input_selector, Input).value = str(raw)

        for key, selector in switch_mapping.items():
            if key not in self._initial_overrides:
                continue
            raw = self._initial_overrides.get(key)
            if raw is None:
                continue
            if selector.startswith("#"):
                switch_selector = selector
            else:
                switch_selector = f"#{selector}"
            self.query_one(switch_selector, Switch).value = bool(raw)

        no_subs = bool(self._initial_overrides.get("no_subs", False))
        self.query_one(self._SEL_INPUT_KEEP_SUBS, Input).disabled = no_subs

        if "save_log_file" in self._initial_overrides:
            self._save_log_file = bool(self._initial_overrides.get("save_log_file"))

        self._save_current_config()

    # ── Config persistence ─────────────────────────────────────────────────────

    def _get_config(self) -> FormConfig:
        """Collect all form values into a config dict."""
        return self._snapshot_form_state()

    def _snapshot_form_state(self) -> FormConfig:
        """Return a plain dict with all current form values for saving."""
        return {
            "path": self.query_one(self._SEL_INPUT_PATH, Input).value.strip(),
            "recursive": self.query_one("#sw-recursive", Switch).value,
            "keep_audio_langs": self.query_one("#inp-keep-audio", Input).value.strip(),
            "keep_sub_langs": self.query_one(
                self._SEL_INPUT_KEEP_SUBS, Input
            ).value.strip(),
            "no_subs": self.query_one("#sw-no-subs", Switch).value,
            "remove_named": self.query_one("#inp-remove-named", Input).value.strip(),
            "default_audio": self.query_one("#inp-default-audio", Input).value.strip(),
            "default_subs": self.query_one("#inp-default-subs", Input).value.strip(),
            "auto_default": self.query_one("#sw-auto-default", Switch).value,
            "fix_missing_default": self.query_one("#sw-fix-defaults", Switch).value,
            "sync_title_to_filename": self.query_one("#sw-sync-title", Switch).value,
            "protect_single_audio": self.query_one("#sw-protect-audio", Switch).value,
            "protect_single_sub": self.query_one("#sw-protect-sub", Switch).value,
            "dry_run": self.query_one("#sw-dry-run", Switch).value,
            "in_place": self.query_one("#sw-in-place", Switch).value,
            "save_log_file": self.query_one("#sw-save-log", Switch).value,
            "jobs": resolve_jobs(
                self.query_one("#inp-jobs", Input).value,
                fallback=default_jobs(),
            ),
            "auto_scroll": self.query_one(self._SEL_SWITCH_AUTO_SCROLL, Switch).value,
            "use_selection": self.query_one(
                self._SEL_SWITCH_USE_SELECTION, Switch
            ).value,
            "selected_paths": sorted(self._selected_paths),
            "upper_panel_height": self._upper_panel_height,
            "log_panel_height": self._log_panel_height,
        }

    def _restore_config(self) -> None:
        """Apply saved config values to the form widgets."""
        saved = _load_config()
        if not saved:
            return

        try:
            inp_path = self.query_one(self._SEL_INPUT_PATH, Input)
            saved_path = saved.get("path")
            if not self._initial_path and isinstance(saved_path, str) and saved_path:
                inp_path.value = saved_path

            switch_mapping = {
                "sw-recursive": ("recursive", False),
                "sw-no-subs": ("no_subs", False),
                "sw-auto-default": ("auto_default", True),
                "sw-fix-defaults": ("fix_missing_default", True),
                "sw-sync-title": ("sync_title_to_filename", True),
                "sw-protect-audio": ("protect_single_audio", True),
                "sw-protect-sub": ("protect_single_sub", True),
                "sw-dry-run": ("dry_run", False),
                "sw-in-place": ("in_place", False),
                "sw-save-log": ("save_log_file", False),
                "sw-use-selection": ("use_selection", False),
                "sw-auto-scroll": ("auto_scroll", True),
            }
            for widget_id, (key, default) in switch_mapping.items():
                sw = self.query_one(f"#{widget_id}", Switch)
                sw.value = bool(saved.get(key, default))

            self._save_log_file = bool(saved.get("save_log_file", False))

            saved_upper = saved.get("upper_panel_height")
            saved_log = saved.get("log_panel_height")
            if isinstance(saved_upper, int):
                self._upper_panel_height = saved_upper
            if isinstance(saved_log, int):
                self._log_panel_height = saved_log

            saved_selected = saved.get("selected_paths")
            if isinstance(saved_selected, list):
                saved_selected_list = cast(list[object], saved_selected)
                self._selected_paths = {
                    str(Path(str(p)).expanduser())
                    for p in saved_selected_list
                    if isinstance(p, str) and p
                }
                self._update_selection_summary()

            input_mapping = {
                "inp-keep-audio": "keep_audio_langs",
                "inp-keep-subs": "keep_sub_langs",
                "inp-remove-named": "remove_named",
                "inp-default-audio": "default_audio",
                "inp-default-subs": "default_subs",
                "inp-jobs": "jobs",
            }
            for widget_id, key in input_mapping.items():
                val = saved.get(key)
                if val is not None and (val != "" or key == "jobs"):
                    inp = self.query_one(f"#{widget_id}", Input)
                    inp.value = str(val)

            no_subs = bool(saved.get("no_subs", False))
            self.query_one(self._SEL_INPUT_KEEP_SUBS, Input).disabled = no_subs
            self._apply_panel_heights()

        except (LookupError, TypeError, ValueError, AttributeError):
            pass  # Silently ignore restore errors; default values will be used

    # ── Column management ──────────────────────────────────────────────────────

    @staticmethod
    def _plain_cell(cell: object) -> str:
        """Convert a plain or rich table cell into plain text for width checks."""
        if isinstance(cell, Text):
            plain = getattr(cell, "plain", None)
            return plain if plain is not None else str(cell)
        return str(cell)

    def _table_content_width(self) -> int:
        """Best-effort width of the track table content area."""
        try:
            table = self._track_table()
            if table.size.width > 0:
                return table.size.width
        except (NoMatches, AttributeError):
            pass
        sidebar = 36
        chrome = 4
        return max(40, self.size.width - sidebar - chrome)

    def _measure_col_content(self, col_idx: int) -> int:
        """Cell count needed to fit header + all cached values in a column."""
        width = len(self._COL_NAMES[col_idx])
        for row in self._inspect_rows:
            width = max(width, len(self._plain_cell(row[col_idx])))
        return width + 2

    @staticmethod
    def _tokenize_filter(raw: str) -> list[str]:
        """Split a filter input into lowercase space-separated tokens."""
        return [token.lower() for token in raw.split() if token]

    def _current_inspector_filters(self) -> dict[str, object]:
        """Read live filter controls from the Track Inspector filter row."""
        sort_value_obj = cast(
            object,
            getattr(self.query_one(self._SEL_INS_SORT_BY), "value", None),
        )
        sort_value = sort_value_obj if isinstance(sort_value_obj, str) else "file"
        sort_by = sort_value if sort_value in self._INS_SORT_KEYS else "file"
        return {
            "search": self.query_one(self._SEL_INS_FILTER_SEARCH, Input)
            .value.strip()
            .lower(),
            "types": self._tokenize_filter(
                self.query_one(self._SEL_INS_FILTER_TYPES, Input).value.strip()
            ),
            "langs": self._tokenize_filter(
                self.query_one(self._SEL_INS_FILTER_LANGS, Input).value.strip()
            ),
            "sort_by": sort_by,
            "sort_desc": self.query_one(self._SEL_INS_SORT_DESC, Switch).value,
            "default_only": self.query_one(self._SEL_INS_FILTER_DEFAULT, Switch).value,
            "forced_only": self.query_one(self._SEL_INS_FILTER_FORCED, Switch).value,
        }

    def _inspector_sort_key(
        self, row: InspectorRow, sort_by: str
    ) -> tuple[object, ...]:
        """Return sort tuple for one inspector row using the selected sort field."""
        path_key = str(row[-1])
        file_name = Path(path_key).name.lower()
        track_raw = self._plain_cell(row[1])
        try:
            track_num = int(track_raw)
        except ValueError:
            track_num = 10**9
        track_type = self._plain_cell(row[2]).lower()
        lang = self._plain_cell(row[3]).lower()
        codec = self._plain_cell(row[4]).lower()
        name = self._plain_cell(row[5]).lower()
        flags = self._plain_cell(row[6]).lower()

        primary_map: dict[str, object] = {
            "file": file_name,
            "track": track_num,
            "type": track_type,
            "lang": lang,
            "codec": codec,
            "name": name,
            "flags": flags,
        }
        primary = primary_map.get(sort_by, file_name)
        return (primary, file_name, track_num, track_type, lang, codec, name, flags)

    def _row_matches_inspector_filters(
        self,
        row: InspectorRow,
        filters: Mapping[str, object],
    ) -> bool:
        """Return True when an inspector row matches the current filter set."""
        file_name = self._plain_cell(row[0]).lower()
        track_id = self._plain_cell(row[1]).lower()
        track_type = self._plain_cell(row[2]).lower()
        lang = self._plain_cell(row[3]).lower()
        codec = self._plain_cell(row[4]).lower()
        track_name = self._plain_cell(row[5]).lower()
        flags = self._plain_cell(row[6]).lower()

        search = filters.get("search")
        if isinstance(search, str) and search:
            haystack = " ".join(
                [file_name, track_id, track_type, lang, codec, track_name]
            )
            if search not in haystack:
                return False

        type_tokens = filters.get("types")
        if isinstance(type_tokens, list) and type_tokens:
            lowered = [token.lower() for token in cast(list[str], type_tokens)]
            if track_type not in lowered:
                return False

        lang_tokens = filters.get("langs")
        if isinstance(lang_tokens, list) and lang_tokens:
            lowered = [token.lower() for token in cast(list[str], lang_tokens)]
            if lang not in lowered:
                return False

        if bool(filters.get("default_only")) and "default" not in flags:
            return False
        if bool(filters.get("forced_only")) and "forced" not in flags:
            return False
        return True

    def _update_inspector_filter_summary(self) -> None:
        """Render a compact summary of active filters and row count."""
        total = len(self._inspect_source_rows)
        shown = len(self._inspect_rows)
        if total == 0:
            self.query_one(self._SEL_INS_FILTER_SUMMARY, Static).update(
                "Filters: no rows loaded"
            )
            return

        filters = self._current_inspector_filters()
        active_parts: list[str] = []
        search = filters.get("search")
        if isinstance(search, str) and search:
            active_parts.append(f"text='{search}'")
        types = filters.get("types")
        if isinstance(types, list) and types:
            active_parts.append("types=" + ",".join(cast(list[str], types)))
        langs = filters.get("langs")
        if isinstance(langs, list) and langs:
            active_parts.append("langs=" + ",".join(cast(list[str], langs)))
        if bool(filters.get("default_only")):
            active_parts.append("default only")
        if bool(filters.get("forced_only")):
            active_parts.append("forced only")
        sort_by = str(filters.get("sort_by", "file"))
        sort_dir = "desc" if bool(filters.get("sort_desc")) else "asc"
        active_parts.append(f"sort={sort_by}:{sort_dir}")

        details = "; ".join(active_parts) if active_parts else "none"
        self.query_one(self._SEL_INS_FILTER_SUMMARY, Static).update(
            f"Filters: {details}  •  showing {shown}/{total} rows"
        )

    def _refresh_filtered_inspector_rows(self) -> None:
        """Rebuild visible inspector rows from source rows and active filters."""
        table = self._track_table()
        filters = self._current_inspector_filters()
        auto_scroll = self.query_one(self._SEL_SWITCH_AUTO_SCROLL, Switch).value
        saved_scroll_y = table.scroll_y

        table.clear()
        self._inspect_rows.clear()
        self._file_row_keys.clear()
        self._file_row_offsets.clear()

        matched_rows: list[InspectorRow] = []
        for source_row in self._inspect_source_rows:
            if not self._row_matches_inspector_filters(source_row, filters):
                continue
            matched_rows.append(source_row)

        sort_by = str(filters.get("sort_by", "file"))
        sort_desc = bool(filters.get("sort_desc"))
        matched_rows.sort(
            key=lambda row: self._inspector_sort_key(row, sort_by),
            reverse=sort_desc,
        )

        for source_row in matched_rows:
            path_key = str(source_row[-1])
            visible_cells: list[TableCell] = source_row[:-1]
            rk = table.add_row(*visible_cells)
            idx = len(self._inspect_rows)
            self._inspect_rows.append(list(source_row))
            self._file_row_keys.setdefault(path_key, []).append(rk)
            self._file_row_offsets.setdefault(path_key, []).append(idx)

        if self._inspect_rows:
            self._auto_fit_data_columns()
            if auto_scroll:
                table.scroll_end(animate=False)
            else:
                table.scroll_to(y=saved_scroll_y, animate=False)
        self._update_inspector_filter_summary()

    def _track_table(self) -> DataTable[TableCell]:
        """Return the main track inspector table with the expected cell type."""
        return cast(
            DataTable[TableCell],
            self.query_one(self._SEL_TRACK_TABLE, DataTable),
        )

    def _other_columns_render_width(self, table: DataTable[TableCell]) -> int:
        """Total render width of every column except File."""
        total = 0
        for col_key in self._col_keys[1:]:
            total += table.columns[col_key].get_render_width(table)
        return total

    def _flush_table_layout(self) -> None:
        """Recalculate column layout and repaint without waiting for idle/unfocus."""
        table = self._track_table()
        table.refresh(layout=True, repaint=True)

    def _update_file_column_width(self) -> None:
        """Size File to content when it fits, else use remaining table width."""
        if not self._col_keys or not self._inspect_rows:
            return
        if 0 in self._user_fixed_cols:
            self._flush_table_layout()
            return
        table = self._track_table()
        file_content_w = self._measure_col_content(0)
        other_w = self._other_columns_render_width(table)
        separators = len(self._COL_NAMES) + 1
        remaining = self._table_content_width() - other_w - separators

        if file_content_w <= max(8, remaining):
            file_w = file_content_w
        else:
            file_w = max(8, remaining)

        col = table.columns[self._col_keys[0]]
        col.auto_width = False
        col.width = file_w
        col.content_width = max(col.content_width, file_content_w)
        self._col_widths[0] = file_w
        self._flush_table_layout()

    def _auto_fit_data_columns(self) -> None:
        """Fit non-File columns to content; user-fixed columns keep manual width."""
        if not self._col_keys:
            return
        table = self._track_table()
        for col_idx in range(1, len(self._COL_NAMES)):
            col_key = self._col_keys[col_idx]
            col = table.columns[col_key]
            if col_idx in self._user_fixed_cols:
                continue
            content_w = self._measure_col_content(col_idx)
            col.auto_width = False
            col.width = content_w
            col.content_width = content_w
            self._col_widths[col_idx] = content_w
        self._update_file_column_width()

    def _set_fixed_column_width(self, col_idx: int, width: int) -> None:
        """Pin a column to a manual width (user resized with [ ])."""
        table = self._track_table()
        col_key = self._col_keys[col_idx]
        col = table.columns[col_key]
        col.auto_width = False
        col.width = width
        col.content_width = width
        self._col_widths[col_idx] = width
        self._user_fixed_cols.add(col_idx)
        if col_idx == 0:
            self._flush_table_layout()
        else:
            self._update_file_column_width()

    def _setup_table_columns(self) -> None:
        """Add columns: File fixed (computed), others auto-fit to content."""
        saved_obj = _load_config().get("inspect_cache")
        saved = (
            cast(dict[str, object], saved_obj) if isinstance(saved_obj, dict) else {}
        )
        fixed_obj = saved.get("user_fixed_cols")
        fixed = cast(list[int], fixed_obj) if isinstance(fixed_obj, list) else []
        self._user_fixed_cols = {int(i) for i in fixed}
        fixed_widths_obj = saved.get("fixed_col_widths")
        fixed_widths = (
            cast(dict[str, int], fixed_widths_obj)
            if isinstance(fixed_widths_obj, dict)
            else {}
        )

        table = self._track_table()
        self._col_keys = []
        if 0 in self._user_fixed_cols:
            file_w = int(fixed_widths.get("0", self._COL_DEFAULTS[0]))
            self._col_widths[0] = file_w
            self._col_keys.append(table.add_column("File", width=file_w))
        else:
            self._col_keys.append(table.add_column("File", width=20))
        for idx, name in enumerate(self._COL_NAMES[1:], start=1):
            if idx in self._user_fixed_cols:
                w = int(fixed_widths.get(str(idx), self._COL_DEFAULTS[idx]))
                self._col_widths[idx] = w
                self._col_keys.append(table.add_column(name, width=w))
            else:
                self._col_keys.append(table.add_column(name, width=None))

    def _init_inspector(self) -> None:
        """Create columns, restore cached rows, and refresh header."""
        self._setup_table_columns()
        self._restore_inspect_cache()
        self._update_inspect_header()

    def on_resize(self) -> None:
        """Recompute File column width when terminal is resized."""
        if self._col_keys and self._inspect_rows:
            self._auto_fit_data_columns()
        self._apply_panel_heights()

    def _available_pane_height(self) -> int | None:
        """Return available vertical space to split between upper and log panes."""
        try:
            right = self.query_one("#right-pane", Vertical)
            total = right.size.height
        except (NoMatches, AttributeError):
            return None
        if total <= 0:
            return None
        progress_height = 3 if self.is_busy else 0
        return max(6, total - progress_height)

    def _apply_panel_heights(self, changed: str | None = None) -> None:
        """Apply and clamp the upper/log pane heights to fit current terminal size."""
        available = self._available_pane_height()
        if available is None:
            return

        min_upper = 7
        min_log = 5
        if available < (min_upper + min_log):
            min_upper = max(3, available - 3)
            min_log = max(3, available - min_upper)

        upper = max(min_upper, int(self._upper_panel_height))
        log = max(min_log, int(self._log_panel_height))

        total = upper + log
        if total > available:
            if changed == "upper":
                # Keep requested upper size; shrink log to make room.
                log = max(min_log, available - upper)
            elif changed == "log":
                # Keep requested log size; shrink upper to make room.
                upper = max(min_upper, available - log)
            else:
                upper = max(min_upper, available - log)

        total = upper + log
        if total > available:
            # Fallback rebalance when clamping still overflows.
            if changed == "upper":
                upper = max(min_upper, available - log)
            elif changed == "log":
                log = max(min_log, available - upper)
            else:
                upper = max(min_upper, available - log)

        total = upper + log
        if total < available:
            # Preserve the pane the user changed; grow the opposite pane.
            if changed == "upper":
                log = available - upper
            elif changed == "log":
                upper = available - log
            else:
                upper = available - log

        self._upper_panel_height = upper
        self._log_panel_height = log

        self.query_one(self._SEL_UPPER_PANE, Vertical).styles.height = upper
        self.query_one(self._SEL_LOG_PANE, Vertical).styles.height = log
        self.query_one("#right-pane", Vertical).refresh(layout=True, repaint=True)

    def _resize_upper_panel(self, delta: int) -> None:
        """Resize the upper pane by delta rows and persist the new split."""
        self._upper_panel_height += delta
        self._apply_panel_heights("upper")
        self._save_current_config()
        self._set_status(
            (
                f"Upper panel: {self._upper_panel_height} rows  •  "
                f"Output log: {self._log_panel_height} rows"
            ),
            "dim",
        )

    def _resize_log_panel(self, delta: int) -> None:
        """Resize the log pane by delta rows and persist the new split."""
        self._log_panel_height += delta
        self._apply_panel_heights("log")
        self._save_current_config()
        self._set_status(
            (
                f"Output log: {self._log_panel_height} rows  •  "
                f"Upper panel: {self._upper_panel_height} rows"
            ),
            "dim",
        )

    # ── Column resize actions ──────────────────────────────────────────────────

    def action_widen_col(self) -> None:
        """Increase width of the currently focused table column."""
        table = self._track_table()
        col_idx = table.cursor_column
        if col_idx < 0 or col_idx >= len(self._col_widths):
            return
        cur = table.columns[self._col_keys[col_idx]].width
        cap = 120 if col_idx == 0 else 80
        new_w = min(cap, cur + 3)
        self._set_fixed_column_width(col_idx, new_w)
        self._persist_inspect_state()
        self._set_status(
            f"Column '{self._COL_NAMES[col_idx]}' width → {new_w}",
            "dim",
        )

    def action_narrow_col(self) -> None:
        """Decrease width of the currently focused table column."""
        table = self._track_table()
        col_idx = table.cursor_column
        if col_idx < 0 or col_idx >= len(self._col_widths):
            return
        cur = table.columns[self._col_keys[col_idx]].width
        new_w = max(3, cur - 3)
        self._set_fixed_column_width(col_idx, new_w)
        self._persist_inspect_state()
        self._set_status(
            f"Column '{self._COL_NAMES[col_idx]}' width → {new_w}",
            "dim",
        )

    def action_move_split_down(self) -> None:
        """Move split downward: grow upper pane, shrink output log."""
        self._resize_upper_panel(2)

    def action_move_split_up(self) -> None:
        """Move split upward: grow output log, shrink upper pane."""
        self._resize_log_panel(2)

    def search_themes(self) -> None:
        """Open theme picker with the active theme labeled."""
        self.push_screen(
            CommandPalette(
                providers=[LabeledThemeProvider],
                placeholder="Search for themes…",
            ),
        )

    # ── Inspect cache persistence ──────────────────────────────────────────────

    def _inspect_cache_payload(self) -> dict[str, object]:
        return {
            "path": str(self._last_inspected_path)
            if self._last_inspected_path
            else None,
            "inspected_at": self._last_inspected_time,
            "config_hash": self._last_inspect_config_hash,
            "col_widths": self._col_widths,
            "user_fixed_cols": sorted(self._user_fixed_cols),
            "fixed_col_widths": {
                str(i): self._col_widths[i] for i in self._user_fixed_cols
            },
            "rows": [
                [_cell_to_json(c) for c in row] for row in self._inspect_source_rows
            ],
        }

    def _persist_inspect_state(self) -> None:
        """Save inspector rows and metadata into the config file."""
        if not self._inspect_source_rows and self._last_inspected_path is None:
            return
        cfg = _load_config()
        cfg.update(self._snapshot_form_state())
        cfg["inspect_cache"] = self._inspect_cache_payload()
        _save_config(cfg)

    def _restore_inspect_cache(self) -> None:
        """Restore the last inspector table from disk, if available."""
        if self._initial_path or not self._col_keys:
            return
        cache_obj = _load_config().get("inspect_cache")
        if not isinstance(cache_obj, dict):
            return
        cache = cast(dict[str, object], cache_obj)
        rows_obj = cache.get("rows")
        if not isinstance(rows_obj, list) or not rows_obj:
            return
        rows = cast(list[object], rows_obj)

        path_raw = cache.get("path")
        if path_raw:
            self._last_inspected_path = Path(str(path_raw)).expanduser()
        inspected_at = cache.get("inspected_at")
        self._last_inspected_time = (
            float(inspected_at) if isinstance(inspected_at, (int, float)) else None
        )
        cfg_hash = cache.get("config_hash")
        self._last_inspect_config_hash = (
            str(cfg_hash) if isinstance(cfg_hash, str) else None
        )

        self._inspect_source_rows = []
        for row_obj in rows:
            if isinstance(row_obj, list):
                row_cells = cast(list[object], row_obj)
                restored_row: InspectorRow = [_cell_from_json(c) for c in row_cells]
                self._inspect_source_rows.append(restored_row)

        self._refresh_filtered_inspector_rows()

        self._update_inspect_header()

    # ── Inspect header ────────────────────────────────────────────────────────

    def _update_inspect_header(self, live: bool = False) -> None:
        """Refresh the inspect-header bar above the track table."""
        header = self.query_one("#inspect-header", Static)
        if self._last_inspected_path is None:
            header.update("  No data — press Ctrl+I to inspect a file or folder")
            header.remove_class("live")
            return

        time_str = "unknown time"
        if self._last_inspected_time is not None:
            dt = datetime.fromtimestamp(self._last_inspected_time)
            time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        path_str = self._last_inspected_source_label or str(self._last_inspected_path)

        if live:
            header.update(f"  ⟳  Updating live — {path_str}")
            header.add_class("live")
        else:
            header.update(f"Last inspected: {time_str} — {path_str}")
            header.remove_class("live")

    # ── Row helpers ───────────────────────────────────────────────────────────

    def _make_track_row(
        self,
        file_cell: TableCell,
        track: TrackInfo,
        path_key: str,
    ) -> InspectorRow:
        """Build a single row list for the DataTable (last element is hidden path key)."""
        flags: list[str] = []
        if track.default:
            flags.append("default")
        if track.forced:
            flags.append("forced")
        flag_str = ", ".join(flags) if flags else ""
        type_color = {
            "video": "green",
            "audio": "yellow",
            "subtitles": "cyan",
        }.get(track.ttype, "white")
        return [
            file_cell,
            str(track.tid),
            Text(track.ttype, style=type_color),
            track.lang,
            track.codec,
            track.name or "—",
            Text(flag_str, style="bold magenta") if flag_str else Text(""),
            path_key,
        ]

    def _append_new_inspector_rows(self, new_source_rows: list[InspectorRow]) -> None:
        """Append rows directly to the table without clearing — no scroll jump."""
        table = self._track_table()
        filters = self._current_inspector_filters()
        sort_by = str(filters.get("sort_by", "file"))
        sort_desc = bool(filters.get("sort_desc"))
        if sort_by != "file" or sort_desc:
            self._refresh_filtered_inspector_rows()
            return
        auto_scroll = self.query_one(self._SEL_SWITCH_AUTO_SCROLL, Switch).value

        for source_row in new_source_rows:
            if not self._row_matches_inspector_filters(source_row, filters):
                continue
            path_key = str(source_row[-1])
            visible_cells: list[TableCell] = source_row[:-1]
            rk = table.add_row(*visible_cells)
            idx = len(self._inspect_rows)
            self._inspect_rows.append(list(source_row))
            self._file_row_keys.setdefault(path_key, []).append(rk)
            self._file_row_offsets.setdefault(path_key, []).append(idx)

        if self._inspect_rows:
            # Column auto-fit scans all rows; doing it per file append is very costly
            # on large runs. While busy, batch this work and fit periodically.
            if self.is_busy:
                self._pending_inspector_autofit += len(new_source_rows)
                if self._pending_inspector_autofit >= 40:
                    self._auto_fit_data_columns()
                    self._pending_inspector_autofit = 0
            else:
                self._auto_fit_data_columns()
            if auto_scroll:
                table.scroll_end(animate=False)
        self._update_inspector_filter_summary()

    @staticmethod
    def _format_file_cell(src: Path, size_str: str, prefix: str) -> TableCell:
        """Build a file-cell value, styling preview prefixes for visibility."""
        body = f"{src.name} ({size_str})"
        if not prefix:
            return body
        if prefix.strip().startswith("◌ PREVIEW"):
            cell = Text()
            cell.append(prefix, style="bold blue")
            cell.append(body)
            return cell
        return f"{prefix}{body}"

    def _add_file_rows(
        self,
        src: Path,
        size_str: str,
        tracks: list[TrackInfo],
        prefix: str = "",
    ) -> None:
        """Add (or replace) track rows for one file. Must run on the main thread."""
        path_key = str(src)
        file_cell = self._format_file_cell(src, size_str, prefix)

        is_reinspect = any(
            str(row[-1]) == path_key for row in self._inspect_source_rows
        )

        # Remove previous source rows for this file if re-inspecting during a run
        self._inspect_source_rows = [
            row for row in self._inspect_source_rows if str(row[-1]) != path_key
        ]

        new_rows: list[InspectorRow] = []
        if not tracks:
            new_rows.append(
                [file_cell, "—", "(no tracks)", "—", "—", "—", Text(""), path_key]
            )
        else:
            for t in tracks:
                new_rows.append(self._make_track_row(file_cell, t, path_key))

        self._inspect_source_rows.extend(new_rows)

        if is_reinspect:
            self._refresh_filtered_inspector_rows()
        else:
            self._append_new_inspector_rows(new_rows)

    def _update_file_prefix(self, src: Path, size_str: str, prefix: str) -> None:
        """Update the File-column cell for all rows belonging to src. Main thread only."""
        table = self._track_table()
        path_key = str(src)
        new_cell = f"{prefix}{src.name} ({size_str})"
        file_col_key = self._col_keys[0] if self._col_keys else None
        if file_col_key is None:
            return
        for rk in self._file_row_keys.get(path_key, []):
            try:
                table.update_cell(rk, file_col_key, new_cell)
            except (LookupError, TypeError, AttributeError):
                pass
        for idx in self._file_row_offsets.get(path_key, []):
            if 0 <= idx < len(self._inspect_rows):
                self._inspect_rows[idx][0] = new_cell
        for source_row in self._inspect_source_rows:
            if str(source_row[-1]) == path_key:
                source_row[0] = new_cell
        self._update_file_column_width()

    def _save_current_config(self) -> None:
        """Snapshot current form state and persist to disk (keeps inspect cache)."""
        cfg = _load_config()
        cfg.update(self._snapshot_form_state())
        _save_config(cfg)

    @staticmethod
    def _display_path(path: Path) -> str:
        """Return a readable label for a file or directory selection."""
        label = str(path)
        return f"{label}{'/' if path.is_dir() else ''}"

    def _selected_path_objects(self) -> list[Path]:
        """Return existing selected paths in a stable order."""
        return [
            Path(raw).expanduser()
            for raw in sorted(self._selected_paths)
            if Path(raw).expanduser().exists()
        ]

    def _selection_source_enabled(self) -> bool:
        """Return True when explorer selection mode is enabled."""
        return self.query_one(self._SEL_SWITCH_USE_SELECTION, Switch).value

    def _apply_input_mode_ui(self) -> None:
        """Reflect the active input mode in the path widget state."""
        self.query_one(
            self._SEL_INPUT_PATH, Input
        ).disabled = self._selection_source_enabled()

    def _update_selection_summary(self) -> None:
        """Refresh the selected-paths summary label in the sidebar."""
        summary = self.query_one(self._SEL_SELECTION_SUMMARY, Static)
        details = self.query_one(self._SEL_SELECTION_DETAILS, Static)
        use_selection = self._selection_source_enabled()
        selected_paths = self._selected_path_objects()

        summary.update(
            "Mode: explorer selection" if use_selection else "Mode: path input"
        )

        if not selected_paths:
            details.update("Explorer selection: none")
            self._apply_input_mode_ui()
            return

        lines = [
            (
                "Explorer selection (active):"
                if use_selection
                else "Explorer selection (saved, inactive):"
            )
        ]
        max_items = 8
        for selected in selected_paths[:max_items]:
            lines.append(f"• {self._display_path(selected)}")
        if len(selected_paths) > max_items:
            lines.append(f"• … +{len(selected_paths) - max_items} more")
        details.update("\n".join(lines))
        self._apply_input_mode_ui()

    def _open_explorer_at(self) -> str:
        """Choose the explorer's starting folder from the current path input."""
        raw = self.query_one(self._SEL_INPUT_PATH, Input).value.strip()
        if raw:
            candidate = Path(raw).expanduser()
            if candidate.exists() and candidate.is_file():
                return str(candidate.parent)
            if candidate.exists() and candidate.is_dir():
                return str(candidate)
        return str(Path.cwd())

    def _on_explorer_closed(self, result: tuple[str, list[str]] | None) -> None:
        """Apply explorer selection result when the modal closes."""
        if result is None:
            self._set_status("Explorer closed without changes.", "dim")
            return
        current_dir, selected = result
        self.query_one(self._SEL_INPUT_PATH, Input).value = current_dir
        self._selected_paths = {
            str(Path(path).expanduser())
            for path in selected
            if Path(path).expanduser().exists()
        }
        if self._selected_paths:
            self.query_one(self._SEL_SWITCH_USE_SELECTION, Switch).value = True
        self._update_selection_summary()
        self._save_current_config()
        self._log_event(
            (
                "Explorer selection updated: "
                f"{len(self._selected_paths)} path(s) selected"
            ),
            "INFO",
        )
        self._set_status(
            (
                "Selection cleared; using path field."
                if not self._selected_paths
                else f"Using {len(self._selected_paths)} selected path(s)."
            ),
            "green" if self._selected_paths else "yellow",
        )

    # ── Event handlers: persist on every change ────────────────────────────────

    def on_input_changed(self, _event: Input.Changed) -> None:
        """Persist config whenever any input field changes."""
        if _event.input.id in {
            "ins-filter-search",
            "ins-filter-types",
            "ins-filter-langs",
        }:
            self._refresh_filtered_inspector_rows()
            return
        self._update_selection_summary()
        self._save_current_config()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle Track Inspector dropdown changes."""
        if event.select.id == "ins-sort-by":
            self._refresh_filtered_inspector_rows()

    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Handle switch changes and persist config."""
        if event.switch.id in {"sw-ins-default", "sw-ins-forced", "sw-ins-sort-desc"}:
            self._refresh_filtered_inspector_rows()
            return
        if event.switch.id == "sw-no-subs" and event.value:
            self.query_one(self._SEL_INPUT_KEEP_SUBS, Input).disabled = True
        elif event.switch.id == "sw-no-subs" and not event.value:
            self.query_one(self._SEL_INPUT_KEEP_SUBS, Input).disabled = False
        elif event.switch.id == "sw-save-log":
            self._save_log_file = bool(event.value)
            self._log_event(
                "File logging enabled." if event.value else "File logging disabled.",
                "INFO",
            )
        elif event.switch.id == "sw-use-selection":
            self._update_selection_summary()
        self._save_current_config()

    # ── Reactive watches ───────────────────────────────────────────────────────

    def watch_is_busy(self, busy: bool) -> None:
        """Toggle UI interactivity and progress row while workers are active."""
        self.query_one("#btn-inspect").disabled = busy
        self.query_one("#btn-run").disabled = busy or not check_dependencies()
        self.query_one("#btn-cancel").disabled = not busy
        self.query_one("#btn-reset").disabled = busy
        self.query_one("#btn-help").disabled = busy
        progress_row = self.query_one("#progress-row")
        if busy:
            progress_row.add_class("visible")
        else:
            progress_row.remove_class("visible")
        self._apply_panel_heights()

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get_run_config(self) -> RunConfig:
        """Collect all form values into a config dict for running/inspecting."""

        def tokens(input_id: str) -> list[str]:
            """Return normalized whitespace tokens from an Input field."""
            raw = self.query_one(input_id, Input).value.strip()
            return [t for t in raw.split() if t] if raw else []

        def sw(switch_id: str) -> bool:
            """Read one Switch value to keep the run-config mapping concise."""
            return self.query_one(switch_id, Switch).value

        def val(input_id: str) -> str | None:
            """Return trimmed input text or None so empty fields mean ""unset""."""
            v = self.query_one(input_id, Input).value.strip()
            return v if v else None

        return {
            "path": self.query_one(self._SEL_INPUT_PATH, Input).value.strip(),
            "recursive": sw("#sw-recursive"),
            "keep_audio_langs": tokens("#inp-keep-audio"),
            "keep_sub_langs": tokens(self._SEL_INPUT_KEEP_SUBS),
            "no_subs": sw("#sw-no-subs"),
            "remove_named": tokens("#inp-remove-named"),
            "default_audio": val("#inp-default-audio"),
            "default_subs": val("#inp-default-subs"),
            "auto_default": sw("#sw-auto-default"),
            "fix_missing_default": sw("#sw-fix-defaults"),
            "sync_title_to_filename": sw("#sw-sync-title"),
            "protect_single_audio": sw("#sw-protect-audio"),
            "protect_single_sub": sw("#sw-protect-sub"),
            "dry_run": sw("#sw-dry-run"),
            "in_place": sw("#sw-in-place"),
            "save_log_file": sw("#sw-save-log"),
            "jobs": resolve_jobs(
                self.query_one("#inp-jobs", Input).value,
                fallback=default_jobs(),
            ),
            "use_selection": sw(self._SEL_SWITCH_USE_SELECTION),
            "selected_paths": sorted(self._selected_paths),
        }

    def _write_log(self, msg: str | Text, style: str = "") -> None:
        """Write a message to the main output log."""
        log = self.query_one("#main-log", RichLog)
        if style and isinstance(msg, str):
            log.write(Text(msg, style=style))
        else:
            log.write(msg)
        if self.query_one(self._SEL_SWITCH_AUTO_SCROLL, Switch).value:
            try:
                log.scroll_end()
            except (NoMatches, AttributeError):
                pass

    def _set_status(self, msg: str, style: str = "dim") -> None:
        self.query_one("#status-bar", StatusBar).set_status(msg, style)

    def _reset_perf_counters(self) -> None:
        """Reset per-operation cache performance counters."""
        with self._cache_stats_lock:
            self._cache_hits = 0
            self._cache_misses = 0

    def _record_cache_lookup(self, hit: bool) -> None:
        """Record one track metadata cache lookup outcome."""
        with self._cache_stats_lock:
            if hit:
                self._cache_hits += 1
            else:
                self._cache_misses += 1

    def _cache_stats_snapshot(self) -> tuple[int, int]:
        """Return current cache hit/miss counters."""
        with self._cache_stats_lock:
            return self._cache_hits, self._cache_misses

    def _perf_note(self) -> str:
        """Return a compact performance note for progress/status UI."""
        hits, misses = self._cache_stats_snapshot()
        lookups = hits + misses
        if lookups <= 0:
            return f"jobs {self._active_jobs} • cache n/a (0/0)"
        pct = (hits / lookups) * 100
        return f"jobs {self._active_jobs} • cache {pct:.0f}% ({hits}/{lookups})"

    def _load_track_cache(self) -> None:
        """Load persisted track metadata cache from disk."""
        try:
            data: object = json.loads(_track_cache_path().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return

        data_map = cast(dict[object, object], data)
        raw_saved_at = data_map.get("saved_at")
        saved_at = int(raw_saved_at) if isinstance(raw_saved_at, (int, float)) else 0
        entries_obj = data_map.get("entries")
        if not isinstance(entries_obj, dict):
            return

        now_ts = int(time.time())
        loaded: dict[str, tuple[int, int, list[TrackInfo], str | None, int]] = {}
        entries = cast(dict[object, object], entries_obj)
        modified = False
        for raw_path, entry_obj in entries.items():
            path_key = str(raw_path)
            if not isinstance(entry_obj, dict):
                modified = True
                continue
            entry = cast(dict[object, object], entry_obj)
            raw_mtime = entry.get("mtime_ns")
            raw_size = entry.get("size")
            raw_title = entry.get("title")
            tracks_obj = entry.get("tracks")
            if not isinstance(raw_mtime, int) or not isinstance(raw_size, int):
                modified = True
                continue
            if not isinstance(tracks_obj, list):
                modified = True
                continue
            if isinstance(raw_title, str):
                title = raw_title.strip() or None
            else:
                title = None
                if raw_title is not None:
                    modified = True
            raw_last_accessed = entry.get("last_accessed")
            if isinstance(raw_last_accessed, (int, float)):
                last_accessed = int(raw_last_accessed)
            else:
                last_accessed = saved_at if saved_at > 0 else now_ts
                modified = True
            if now_ts - last_accessed > self._track_cache_ttl_seconds:
                modified = True
                continue
            tracks_payload = cast(list[object], tracks_obj)
            tracks: list[TrackInfo] = []
            valid = True
            for track_obj in tracks_payload:
                if not isinstance(track_obj, dict):
                    valid = False
                    break
                track_payload_raw = cast(dict[object, object], track_obj)
                track_payload = {str(k): v for k, v in track_payload_raw.items()}
                try:
                    tracks.append(_track_from_payload(track_payload))
                except (TypeError, ValueError):
                    valid = False
                    break
            if valid:
                loaded[path_key] = (raw_mtime, raw_size, tracks, title, last_accessed)
            else:
                modified = True

        if len(loaded) > self._track_cache_max_entries:
            keep = sorted(
                loaded.items(),
                key=lambda item: item[1][4],
                reverse=True,
            )[: self._track_cache_max_entries]
            loaded = dict(keep)
            modified = True

        if loaded:
            with self._track_cache_lock:
                self._track_cache = loaded
                self._track_cache_dirty = modified

    def _compact_track_cache_locked(self) -> bool:
        """Apply TTL and max-entry compaction to the in-memory track cache."""
        if not self._track_cache:
            return False

        changed = False
        now_ts = int(time.time())
        expired_keys = [
            key
            for key, (
                _mtime_ns,
                _size,
                _tracks,
                _title,
                last_accessed,
            ) in self._track_cache.items()
            if now_ts - last_accessed > self._track_cache_ttl_seconds
        ]
        for key in expired_keys:
            self._track_cache.pop(key, None)
            changed = True

        if len(self._track_cache) > self._track_cache_max_entries:
            stale_keys = sorted(
                self._track_cache,
                key=lambda key: self._track_cache[key][4],
            )[: len(self._track_cache) - self._track_cache_max_entries]
            for key in stale_keys:
                self._track_cache.pop(key, None)
            changed = True

        return changed

    def _persist_track_cache(self, force: bool = False) -> None:
        """Persist in-memory track metadata cache to disk."""
        with self._track_cache_lock:
            compacted = self._compact_track_cache_locked()
            if compacted:
                self._track_cache_dirty = True
            if not self._track_cache:
                self._track_cache_dirty = False
                return
            if not force and not self._track_cache_dirty:
                return

            items = sorted(
                self._track_cache.items(),
                key=lambda item: item[1][4],
                reverse=True,
            )

            payload_entries: dict[str, object] = {}
            for path_key, (mtime_ns, size, tracks, title, last_accessed) in items:
                payload_entries[path_key] = {
                    "mtime_ns": mtime_ns,
                    "size": size,
                    "title": title,
                    "last_accessed": last_accessed,
                    "tracks": [_track_to_payload(track) for track in tracks],
                }

            payload: dict[str, object] = {
                "version": 1,
                "saved_at": time.time(),
                "entries": payload_entries,
            }

        try:
            _track_cache_path().write_text(
                json.dumps(payload, separators=(",", ":")),
                encoding="utf-8",
            )
            with self._track_cache_lock:
                self._track_cache_dirty = False
        except OSError:
            pass

    def _invalidate_track_cache(self, path: Path) -> None:
        """Remove one path from track cache and mark cache as dirty."""
        with self._track_cache_lock:
            removed = self._track_cache.pop(str(path), None)
            if removed is not None:
                self._track_cache_dirty = True

    def _get_tracks_cached(
        self, mkv_path: Path
    ) -> tuple[list[TrackInfo], int, str | None]:
        """Return cached track metadata when file size/mtime is unchanged."""
        stat = mkv_path.stat()
        cache_key = str(mkv_path)
        now_ts = int(time.time())
        with self._track_cache_lock:
            cached = self._track_cache.get(cache_key)
        if cached is None or cached[0] != stat.st_mtime_ns or cached[1] != stat.st_size:
            self._record_cache_lookup(False)
            tracks, title = get_tracks_and_title(mkv_path)
            with self._track_cache_lock:
                self._track_cache[cache_key] = (
                    stat.st_mtime_ns,
                    stat.st_size,
                    tracks,
                    title,
                    now_ts,
                )
                self._track_cache_dirty = True
            return tracks, stat.st_size, title

        with self._track_cache_lock:
            current = self._track_cache.get(cache_key)
            if current is not None:
                self._track_cache[cache_key] = (
                    current[0],
                    current[1],
                    current[2],
                    current[3],
                    now_ts,
                )
        self._record_cache_lookup(True)
        return cached[2], stat.st_size, cached[3]

    def _source_description(self, cfg: RunConfig) -> str:
        """Describe the active input source for logs and headers."""
        if cfg["use_selection"]:
            selected_paths = [
                Path(p).expanduser()
                for p in cfg["selected_paths"]
                if Path(p).expanduser().exists()
            ]
            if not selected_paths:
                return "explorer selection (empty)"
            preview = ", ".join(path.name or str(path) for path in selected_paths[:3])
            if len(selected_paths) > 3:
                preview += f", +{len(selected_paths) - 3} more"
            return f"explorer selection — {preview}"
        return f"path input — {cfg['path']}"

    def _log_event(self, msg: str, level: str = "INFO", to_output: bool = True) -> None:
        """Write one structured log line to the UI and optional log file."""
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARN": logging.WARNING,
            "ERROR": logging.ERROR,
        }
        style_map = {
            "DEBUG": "dim",
            "INFO": "cyan",
            "WARN": "yellow",
            "ERROR": self._SUMMARY_ERROR_STYLE,
        }
        lvl = level.upper()
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] [{lvl}] {msg}"

        if self._save_log_file:
            try:
                self._ensure_file_logger()
                self._file_logger.log(level_map.get(lvl, logging.INFO), msg)
            except OSError:
                pass

        if to_output:
            self._write_log(line, style_map.get(lvl, ""))

    @staticmethod
    def _summary_line(label: str, value: str, value_style: str = "bold") -> Text:
        line = Text()
        line.append(f"  {label:<22}", style=MkvCleanerApp._SUMMARY_LABEL_STYLE)
        line.append(value, style=value_style)
        return line

    @staticmethod
    def _summary_header(title: str, subtitle: str | None = None) -> Text:
        header = Text()
        header.append(f"  {title}", style=MkvCleanerApp._SUMMARY_LABEL_STYLE)
        if subtitle:
            header.append(f"  {subtitle}", style="dim")
        return header

    @staticmethod
    def _summary_section(title: str) -> Text:
        return Text(f"  {title}", style=MkvCleanerApp._SUMMARY_SECTION_STYLE)

    @staticmethod
    def _track_label(track: TrackInfo) -> str:
        name = track.name or "—"
        return f"Track {track.tid}  lang={track.lang}  name={name}"

    def _render_live_summary_intro(
        self, source_label: str, total: int, dry_run: bool
    ) -> None:
        """Initialize the live summary pane before a run starts processing files."""
        slog = self.query_one(self._SEL_SUMMARY_LOG, RichLog)
        slog.clear()
        mode = "DRY RUN" if dry_run else "LIVE RUN"
        slog.write(
            self._summary_header("Live run summary", f"— {mode} — {source_label}")
        )
        slog.write(self._summary_line("Files queued", str(total)))
        slog.write("")

    def _render_live_file_changes(
        self,
        src: Path,
        summary: RunSummary,
        state: str,
        dry_run: bool,
        before_size: int,
        after_size: int | None = None,
        error: str | None = None,
    ) -> list[tuple[str, int, str, str, bool, bool]]:
        """Append one per-file visual summary block and return default-flag changes."""
        slog = self.query_one(self._SEL_SUMMARY_LOG, RichLog)

        default_changes: list[tuple[str, int, str, str, bool, bool]] = []
        for t in summary["audio_keep"]:
            old_default = t.default
            new_default = summary["audio_defaults"].get(t.tid, old_default)
            if new_default != old_default:
                default_changes.append(
                    ("audio", t.tid, t.lang, t.name or "—", old_default, new_default)
                )
        for t in summary["subs_keep"]:
            old_default = t.default
            new_default = summary["sub_defaults"].get(t.tid, old_default)
            if new_default != old_default:
                default_changes.append(
                    (
                        "subtitle",
                        t.tid,
                        t.lang,
                        t.name or "—",
                        old_default,
                        new_default,
                    )
                )

        removed_audio = summary["audio_removed"]
        removed_subs = summary["subs_removed"]

        if state == "error":
            badge = Text("  ✗ ERROR", style=self._SUMMARY_ERROR_STYLE)
        elif state == "skipped":
            badge = Text("  ✓ SKIPPED", style="bold yellow")
        elif dry_run:
            badge = Text("  ◌ DRY RUN", style="bold cyan")
        else:
            badge = Text("  ✅ WRITTEN", style=self._SUMMARY_SUCCESS_STYLE)

        slog.write(Text("─" * 64, style="dim"))
        header = Text()
        header.append(f"  {src.name}", style=self._SUMMARY_LABEL_STYLE)
        header.append(f"  ({fmt_size(before_size)})", style="dim")
        slog.write(header)
        slog.write(badge)

        if state == "error":
            slog.write(Text(f"    • {error or 'Unknown error'}", style="red"))
            slog.write("")
            return default_changes

        if removed_audio or removed_subs:
            slog.write(self._summary_section("  Removed tracks".strip()))
            for t in removed_audio:
                slog.write(
                    Text(f"    ✂ audio    {self._track_label(t)}", style="yellow")
                )
            for t in removed_subs:
                slog.write(
                    Text(f"    ✂ subtitle {self._track_label(t)}", style="yellow")
                )
        else:
            slog.write(Text("    • No tracks removed", style="dim"))

        if default_changes:
            slog.write(self._summary_section("  Default flag changes".strip()))
            for ttype, tid, lang, name, old, new in default_changes:
                old_s = "default" if old else "non-default"
                new_s = "default" if new else "non-default"
                slog.write(
                    Text(
                        (
                            f"    ★ {ttype:<8} Track {tid}  lang={lang}  "
                            f"name={name}  {old_s} → {new_s}"
                        ),
                        style="magenta",
                    )
                )
        else:
            slog.write(Text("    • No default-flag changes", style="dim"))

        if summary["title_changed"]:
            title_target = summary["title_target"] or "—"
            slog.write(self._summary_section("  File title".strip()))
            slog.write(Text(f"    ★ title → {title_target}", style="magenta"))

        if after_size is not None and not dry_run:
            saved = before_size - after_size
            slog.write(
                Text(
                    (
                        f"    • Size: {fmt_size(before_size)} → {fmt_size(after_size)}  "
                        f"({fmt_delta(saved, before_size)})"
                    ),
                    style="dim",
                )
            )

        slog.write("")
        if self.query_one(self._SEL_SWITCH_AUTO_SCROLL, Switch).value:
            try:
                slog.scroll_end()
            except (NoMatches, AttributeError):
                pass
        return default_changes

    def _build_outcome_log_line(self, outcome: RunWorkerOutcome, dry_run: bool) -> str:
        """Build a concise per-file progress line for the output log."""
        audio_removed = sum(1 for t in outcome.result.removed if t[0] == "audio")
        sub_removed = sum(1 for t in outcome.result.removed if t[0] == "subtitle")
        default_flips = len(outcome.result.default_changes)
        title_changed = outcome.summary["title_changed"]
        title_target = outcome.summary["title_target"] or "—"

        if outcome.state == "error":
            return f"✗ {outcome.path.name}: failed" + (
                f" ({outcome.error})" if outcome.error else ""
            )

        if outcome.state == "skipped":
            return (
                f"✓ {outcome.path.name}: no changes needed"
                f" (audio rm={audio_removed}, subs rm={sub_removed},"
                f" defaults={default_flips}, title={'1' if title_changed else '0'})"
            )

        if outcome.state == "dry_run" or dry_run:
            return (
                f"◌ {outcome.path.name}: preview ready"
                f" (audio rm={audio_removed}, subs rm={sub_removed},"
                f" defaults={default_flips}, title={title_target if title_changed else '0'})"
            )

        if outcome.state == "written":
            if outcome.after_size is not None and outcome.before_size > 0:
                saved = outcome.before_size - outcome.after_size
                return (
                    f"✅ {outcome.path.name}: written"
                    f" (audio rm={audio_removed}, subs rm={sub_removed},"
                    f" defaults={default_flips}, title={title_target if title_changed else '0'}, {fmt_delta(saved, outcome.before_size)})"
                )
            return (
                f"✅ {outcome.path.name}: written"
                f" (audio rm={audio_removed}, subs rm={sub_removed},"
                f" defaults={default_flips}, title={title_target if title_changed else '0'})"
            )

        if outcome.state == "cancelled":
            return f"⚠ {outcome.path.name}: cancelled"

        return f"• {outcome.path.name}: state={outcome.state}"

    def _validate_path(self, cfg: RunConfig) -> Path | None:
        if cfg["use_selection"]:
            existing_selection = [
                Path(p).expanduser()
                for p in cfg["selected_paths"]
                if Path(p).expanduser().exists()
            ]
            if not existing_selection:
                self._write_log(
                    "✗  Selection is empty or no selected paths exist.",
                    self._SUMMARY_ERROR_STYLE,
                )
                self._set_status("Selection is empty or invalid.", "red")
                return None
            raw = cfg["path"]
            if raw:
                candidate = Path(raw).expanduser()
                if candidate.exists():
                    self._log_event(
                        (
                            "Resolved source from selection mode: "
                            f"path_field={candidate} selected_paths={len(existing_selection)}"
                        ),
                        "INFO",
                    )
                    return candidate
            self._log_event(
                (
                    "Resolved source from selection mode: "
                    f"first_selected={existing_selection[0]} "
                    f"total_selected={len(existing_selection)}"
                ),
                "INFO",
            )
            return existing_selection[0]

        raw = cfg["path"]
        if not raw:
            self._write_log("⚠  No path specified.", "bold yellow")
            self._set_status("Enter a file or folder path first.", "yellow")
            return None
        p = Path(raw).expanduser()
        if not p.exists():
            self._write_log(f"✗  Path not found: {p}", self._SUMMARY_ERROR_STYLE)
            self._set_status(f"Path not found: {p}", "red")
            return None
        self._log_event(f"Resolved source from path input: {p}", "INFO")
        return p

    # ── Actions ────────────────────────────────────────────────────────────────

    def action_inspect(self) -> None:
        """Validate current options and start track-inspection worker."""
        if self.is_busy:
            return
        cfg = self._get_run_config()
        p = self._validate_path(cfg)
        if p is None:
            return
        source_label = self._source_description(cfg)
        self._log_event(f"Inspect requested for {source_label}", "INFO")
        self._run_inspect(p, cfg, source_label)

    def action_run(self) -> None:
        """Validate current options and start processing worker."""
        if self.is_busy:
            return
        cfg = self._get_run_config()
        p = self._validate_path(cfg)
        if p is None:
            return
        source_label = self._source_description(cfg)
        self._log_event(
            (
                f"Run requested for {source_label} dry_run={cfg['dry_run']} "
                f"in_place={cfg['in_place']} recursive={cfg['recursive']}"
            ),
            "INFO",
        )
        self._run_process(p, cfg, source_label)

    def action_explorer(self) -> None:
        """Open the modal file explorer for multi-path selection."""
        if self.is_busy:
            return
        start = self._open_explorer_at()
        self.push_screen(
            PathExplorerScreen(start, initial_selected=sorted(self._selected_paths)),
            self._on_explorer_closed,
        )

    def action_cancel(self) -> None:
        """Request cancellation of the active inspect/run operation."""
        if not self.is_busy:
            return
        self._cancel_event.set()
        with self._active_process_lock:
            active = list(self._active_processes)
        terminated = 0
        for proc in active:
            if proc.poll() is None:
                try:
                    proc.terminate()
                    terminated += 1
                except OSError:
                    pass
        self._log_event(
            (
                "Cancel requested by user. "
                f"active_processes={len(active)} terminated={terminated}"
            ),
            "WARN",
        )
        self._set_status("Cancel requested…", "yellow")

    def action_clear_log(self) -> None:
        """Clear both output and summary log panes."""
        self.query_one("#main-log", RichLog).clear()
        self.query_one(self._SEL_SUMMARY_LOG, RichLog).clear()
        self._log_event("Cleared output and summary panes.", "INFO")

    def _clear_inspector_filters(self) -> None:
        """Reset track inspector filter controls to defaults."""
        self.query_one(self._SEL_INS_FILTER_SEARCH, Input).value = ""
        self.query_one(self._SEL_INS_FILTER_TYPES, Input).value = ""
        self.query_one(self._SEL_INS_FILTER_LANGS, Input).value = ""
        self.query_one(self._SEL_INS_SORT_BY, Select).value = "file"
        self.query_one(self._SEL_INS_SORT_DESC, Switch).value = False
        self.query_one(self._SEL_INS_FILTER_DEFAULT, Switch).value = False
        self.query_one(self._SEL_INS_FILTER_FORCED, Switch).value = False
        self._refresh_filtered_inspector_rows()
        self._set_status("Inspector filters cleared.", "green")

    # ── Button handlers ────────────────────────────────────────────────────────

    def action_reset(self) -> None:
        """Reset all form fields to default values."""
        input_defaults = {
            "inp-keep-audio": "",
            "inp-keep-subs": "",
            "inp-remove-named": "",
            "inp-default-audio": "",
            "inp-default-subs": "",
            "inp-jobs": str(default_jobs()),
        }
        for widget_id, default in input_defaults.items():
            try:
                self.query_one(f"#{widget_id}", Input).value = default
            except NoMatches:
                pass

        switch_defaults = {
            "sw-recursive": False,
            "sw-no-subs": False,
            "sw-auto-default": True,
            "sw-fix-defaults": True,
            "sw-protect-audio": True,
            "sw-protect-sub": True,
            "sw-dry-run": False,
            "sw-in-place": False,
            "sw-save-log": False,
        }
        for widget_id, default in switch_defaults.items():
            try:
                self.query_one(f"#{widget_id}", Switch).value = default
            except NoMatches:
                pass

        self.query_one(self._SEL_INPUT_KEEP_SUBS, Input).disabled = False
        self._save_log_file = False
        self._selected_paths.clear()
        self.query_one(self._SEL_SWITCH_USE_SELECTION, Switch).value = False

        self._update_selection_summary()
        self._save_current_config()
        self._log_event("Reset all options to defaults.", "INFO")
        self._set_status("Options reset to defaults.", "green")

    def action_help(self) -> None:
        """Open inline help so users can verify options without leaving the current run context."""
        self.push_screen(HelpScreen())

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Route sidebar button presses to the matching action handler."""
        if event.button.id == "btn-inspect":
            self.action_inspect()
        elif event.button.id == "btn-run":
            self.action_run()
        elif event.button.id == "btn-explorer":
            self.action_explorer()
        elif event.button.id == "btn-cancel":
            self.action_cancel()
        elif event.button.id == "btn-reset":
            self.action_reset()
        elif event.button.id == "btn-help":
            self.action_help()
        elif event.button.id == "btn-ins-filter-clear":
            self._clear_inspector_filters()

    def on_key(self, event: Key) -> None:
        """Allow Esc/Q to cancel only while a worker is active."""
        if self.is_busy and event.key in {"escape", "q"}:
            self.action_cancel()
            event.stop()

    # pylint: disable=consider-using-with
    def _run_subprocess_cancellable(self, cmd: list[str]) -> tuple[int, str, str, bool]:
        """Run a subprocess that can be cancelled via self._cancel_event."""
        self._log_event(f"Launching subprocess: {' '.join(cmd)}", "DEBUG", False)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        with self._active_process_lock:
            self._active_processes.add(proc)
        try:
            while True:
                if self._cancel_event.is_set():
                    self._log_event(
                        "Cancellation detected while subprocess running.",
                        "WARN",
                        False,
                    )
                    try:
                        proc.terminate()
                    except OSError:
                        pass
                    try:
                        stdout, stderr = proc.communicate(timeout=2)
                    except subprocess.TimeoutExpired:
                        try:
                            proc.kill()
                        except OSError:
                            pass
                        stdout, stderr = proc.communicate()
                    code = proc.returncode
                    return (code if code is not None else 1), stdout, stderr, True

                try:
                    stdout, stderr = proc.communicate(timeout=0.2)
                    break
                except subprocess.TimeoutExpired:
                    continue

            self._log_event(
                f"Subprocess finished with return code {proc.returncode or 0}.",
                "DEBUG",
                False,
            )
            code = proc.returncode
            return (code if code is not None else 0), stdout, stderr, False
        finally:
            with self._active_process_lock:
                self._active_processes.discard(proc)

    # ── Worker: Inspect ────────────────────────────────────────────────────────

    @work(thread=True, exclusive=True)
    def _run_inspect(self, path: Path, cfg: RunConfig, source_label: str) -> None:
        self._cancel_event.clear()
        recursive = cfg["recursive"]
        self.call_from_thread(setattr, self, "is_busy", True)
        self.call_from_thread(
            self._log_event, f"Inspect scan started for {source_label}", "INFO"
        )
        self.call_from_thread(self._set_status, "Scanning files…", "cyan")

        table = self._track_table()

        def _clear_all():
            table.clear()
            self._inspect_rows.clear()
            self._inspect_source_rows.clear()
            self._file_row_keys.clear()
            self._file_row_offsets.clear()
            self._update_inspector_filter_summary()

        self.call_from_thread(_clear_all)

        self.call_from_thread(
            lambda: setattr(
                self.query_one("#tabs", TabbedContent), "active", "tab-inspect"
            )
        )

        try:
            selected_paths = [Path(p).expanduser() for p in cfg["selected_paths"]]
            if cfg["use_selection"]:
                files = collect_files_from_selection(selected_paths, recursive)
                self.call_from_thread(
                    self._log_event,
                    (
                        "Inspect running from explicit selection "
                        f"({len(selected_paths)} path(s))"
                    ),
                    "INFO",
                )
            else:
                files = collect_files(path, recursive)
        except ValueError as e:
            self.call_from_thread(self._log_event, str(e), "ERROR")
            self.call_from_thread(self._set_status, str(e), "red")
            self.call_from_thread(setattr, self, "is_busy", False)
            return

        total_files = len(files)
        self.call_from_thread(
            self._log_event,
            f"Inspect discovered {total_files} file(s). Reading tracks.",
            "INFO",
        )
        self.call_from_thread(
            self._log_event,
            (
                "Inspect config: "
                f"jobs={cfg['jobs']} recursive={cfg['recursive']} "
                f"use_selection={cfg['use_selection']}"
            ),
            "INFO",
        )

        self._last_inspected_path = path
        self._last_inspected_source_label = source_label
        self._last_inspected_time = time.time()
        self.call_from_thread(self._update_inspect_header, True)

        pb = self.query_one("#progress-bar", ProgressBar)
        pl = self.query_one("#progress-label", Static)
        pe = self.query_one("#progress-eta", Static)
        self.call_from_thread(pb.update, total=total_files, progress=0)
        self.call_from_thread(pe.update, "")

        _start_time = time.monotonic()
        last_ui_update = 0.0
        cancelled = False
        jobs = cfg["jobs"]
        self._active_jobs = jobs
        self._reset_perf_counters()
        self.call_from_thread(
            self._log_event,
            f"Inspect worker pool: {jobs} parallel job(s).",
            "INFO",
        )

        def _probe(mkv_file: Path) -> tuple[Path, str, list[TrackInfo], str | None]:
            try:
                tracks, size_bytes, _title = self._get_tracks_cached(mkv_file)
                return mkv_file, fmt_size(size_bytes), tracks, None
            except (RuntimeError, OSError, TypeError, ValueError) as exc:
                return mkv_file, "?", [], str(exc)

        completed = 0
        executor = ThreadPoolExecutor(max_workers=jobs)
        future_map = {
            executor.submit(_probe, file_path): (order_idx, file_path)
            for order_idx, file_path in enumerate(files)
        }
        queued_preview = ", ".join(file_path.name for file_path in files[:8])
        if len(files) > 8:
            queued_preview += f", +{len(files) - 8} more"
        self.call_from_thread(
            self._log_event,
            f"Inspect queued files: {queued_preview}"
            if queued_preview
            else "Inspect queued files: <none>",
            "INFO",
        )
        row_batch: list[tuple[Path, str, list[TrackInfo], str]] = []
        last_row_flush = 0.0

        def _flush_row_batch(force: bool = False) -> None:
            nonlocal last_row_flush
            if not row_batch:
                return
            now = time.monotonic()
            if (
                not force
                and len(row_batch) < self._row_batch_size
                and (now - last_row_flush) < self._row_flush_interval
            ):
                return
            payload = list(row_batch)
            row_batch.clear()
            last_row_flush = now

            def _apply_rows() -> None:
                for file_path, size_str, tracks, prefix in payload:
                    self._add_file_rows(file_path, size_str, tracks, prefix)

            self.call_from_thread(_apply_rows)

        try:
            for future in as_completed(future_map):
                if self._cancel_event.is_set():
                    cancelled = True
                    break

                _, _ = future_map[future]
                file_path, size_str, tracks, error = future.result()
                completed += 1
                now = time.monotonic()
                if (
                    completed == 1
                    or completed == total_files
                    or (now - last_ui_update) >= self._progress_update_interval
                ):
                    elapsed = now - _start_time
                    if completed > 1 and elapsed > 2:
                        rate = elapsed / completed
                        remaining = rate * (total_files - completed)
                        eta_str = (
                            f"  ETA {remaining:.0f}s"
                            if remaining < 60
                            else f"  ETA {remaining / 60:.1f}m"
                        )
                    else:
                        eta_str = ""
                    perf_note = self._perf_note()
                    self.call_from_thread(
                        pl.update,
                        (
                            f"  {completed}/{total_files}  "
                            f"{100 * completed // total_files}%  •  {perf_note}"
                        ),
                    )
                    self.call_from_thread(pe.update, eta_str)
                    self.call_from_thread(
                        pb.update, total=total_files, progress=completed
                    )
                    last_ui_update = now

                if error is not None:
                    row_batch.append((file_path, "?", [], ""))
                    _flush_row_batch(force=False)
                    self.call_from_thread(
                        self._log_event,
                        f"Inspect failed for {file_path.name}: {error}",
                        "ERROR",
                    )
                    continue

                self.call_from_thread(
                    self._log_event,
                    (
                        f"Inspected {file_path.name}: "
                        f"{len(tracks)} track(s), size={size_str}"
                    ),
                    "DEBUG",
                    False,
                )
                row_batch.append((file_path, size_str, tracks, ""))
                _flush_row_batch(force=False)
        finally:
            executor.shutdown(wait=not cancelled, cancel_futures=cancelled)

        _flush_row_batch(force=True)

        self.call_from_thread(self._auto_fit_data_columns)
        self._pending_inspector_autofit = 0
        self._last_inspected_time = time.time()
        self._last_inspected_source_label = source_label
        self._last_inspect_config_hash = _inspect_config_hash(cfg)
        self.call_from_thread(self._update_inspect_header, False)
        self.call_from_thread(self._persist_inspect_state)
        self._persist_track_cache()
        if cancelled:
            self.call_from_thread(self._log_event, "Inspect cancelled.", "WARN")
            self.call_from_thread(self._set_status, "Inspect cancelled.", "yellow")
        else:
            self.call_from_thread(
                self._log_event,
                f"Inspect complete. {len(files)} file(s) loaded in inspector.",
                "INFO",
            )
            self.call_from_thread(
                self._set_status,
                f"Inspect done — {len(files)} file(s).  •  {self._perf_note()}",
                "green",
            )
            self.call_from_thread(
                self._log_event,
                f"Inspect perf snapshot: {self._perf_note()}",
                "INFO",
            )
        self.call_from_thread(setattr, self, "is_busy", False)

    # ── Worker: Process ────────────────────────────────────────────────────────

    def _run_process_one_file(self, src: Path, cfg: RunConfig) -> RunWorkerOutcome:
        """Process one file and return a UI-ready outcome payload."""
        self.call_from_thread(
            self._log_event,
            f"Starting file: {src.name}",
            "INFO",
        )
        result = FileSummary(path=src)
        empty_summary: RunSummary = {
            "audio_keep": [],
            "audio_removed": [],
            "subs_keep": [],
            "subs_removed": [],
            "audio_defaults": {},
            "sub_defaults": {},
            "defaults_changed": False,
            "title_target": None,
            "title_changed": False,
        }

        try:
            tracks, src_size_bytes, current_title = self._get_tracks_cached(src)
        except (RuntimeError, OSError, TypeError, ValueError) as exc:
            try:
                before_size = src.stat().st_size
            except OSError:
                before_size = 0
            result.errored = True
            return RunWorkerOutcome(
                path=src,
                result=result,
                summary=empty_summary,
                state="error",
                before_size=before_size,
                error=f"Could not read tracks: {exc}",
                log_messages=[
                    ("ERROR", f"Could not read tracks for {src.name}: {exc}"),
                ],
            )

        result.src_size = src_size_bytes
        size_str = fmt_size(result.src_size)
        dst = src.with_stem(src.stem + ".cleaned")
        desired_title = _desired_output_title(src, dst, cfg["in_place"])
        title_changed = bool(
            cfg["sync_title_to_filename"]
            and desired_title is not None
            and current_title != desired_title
            and tracks
        )
        title_skipped = bool(
            cfg["sync_title_to_filename"]
            and desired_title is not None
            and current_title != desired_title
            and not tracks
        )
        if title_skipped:
            self.call_from_thread(
                self._log_event,
                f"{src.name}: skipping title sync because no tracks were detected",
                "WARN",
            )

        cmd, summary = build_mkvmerge_cmd(
            src,
            dst,
            tracks,
            cfg,
            title=desired_title if title_changed else None,
        )

        nothing_removed = not summary["audio_removed"] and not summary["subs_removed"]
        defaults_changed = summary["defaults_changed"]

        result.removed = [
            ("audio", t.tid, t.lang, t.name) for t in summary["audio_removed"]
        ] + [("subtitle", t.tid, t.lang, t.name) for t in summary["subs_removed"]]
        result.default_changes = [
            (
                "audio",
                t.tid,
                t.lang,
                t.name,
                t.default,
                summary["audio_defaults"].get(t.tid, t.default),
            )
            for t in summary["audio_keep"]
            if summary["audio_defaults"].get(t.tid, t.default) != t.default
        ] + [
            (
                "subtitle",
                t.tid,
                t.lang,
                t.name,
                t.default,
                summary["sub_defaults"].get(t.tid, t.default),
            )
            for t in summary["subs_keep"]
            if summary["sub_defaults"].get(t.tid, t.default) != t.default
        ]

        if nothing_removed and not defaults_changed and not title_changed:
            result.skipped = True
            preview_prefix = "◌ PREVIEW " if cfg["dry_run"] else ""
            return RunWorkerOutcome(
                path=src,
                result=result,
                summary=summary,
                state="skipped",
                before_size=result.src_size,
                display_tracks=tracks,
                display_size=size_str,
                display_prefix=preview_prefix,
            )

        if cfg["dry_run"]:
            preview_tracks = build_preview_tracks_from_summary(tracks, summary)
            return RunWorkerOutcome(
                path=src,
                result=result,
                summary=summary,
                state="dry_run",
                before_size=result.src_size,
                display_tracks=preview_tracks,
                display_size=size_str,
                display_prefix="◌ PREVIEW ",
            )

        if self._cancel_event.is_set():
            return RunWorkerOutcome(
                path=src,
                result=result,
                summary=summary,
                state="cancelled",
                before_size=result.src_size,
            )

        metadata_only = nothing_removed and (defaults_changed or title_changed)
        if metadata_only and self._optional_tools.get("mkvpropedit", False):
            self.call_from_thread(
                self._log_event,
                f"{src.name}: metadata-only path selected (mkvpropedit)",
                "INFO",
            )
            target = src if cfg["in_place"] else dst
            if not cfg["in_place"]:
                try:
                    shutil.copy2(src, target)
                except OSError as copy_err:
                    result.errored = True
                    return RunWorkerOutcome(
                        path=src,
                        result=result,
                        summary=summary,
                        state="error",
                        before_size=result.src_size,
                        error=f"copy failed: {copy_err}",
                        log_messages=[
                            (
                                "ERROR",
                                f"Could not prepare metadata-only target {src.name}: {copy_err}",
                            )
                        ],
                    )

            meta_cmd = build_mkvpropedit_cmd(
                target,
                tracks,
                summary,
                title=desired_title if title_changed else None,
            )
            if len(meta_cmd) > 2:
                self.call_from_thread(
                    self._log_event,
                    f"{src.name}: running mkvpropedit ({len(meta_cmd)} arg(s))",
                    "INFO",
                )
                returncode, _, stderr, was_cancelled = self._run_subprocess_cancellable(
                    meta_cmd
                )
                if was_cancelled:
                    return RunWorkerOutcome(
                        path=src,
                        result=result,
                        summary=summary,
                        state="cancelled",
                        before_size=result.src_size,
                    )
                if returncode not in (0, 1):
                    result.errored = True
                    err = stderr.strip()
                    return RunWorkerOutcome(
                        path=src,
                        result=result,
                        summary=summary,
                        state="error",
                        before_size=result.src_size,
                        error=f"mkvpropedit failed: {err}",
                        log_messages=[
                            ("ERROR", f"mkvpropedit failed for {src.name}: {err}")
                        ],
                    )

                result.dst_size = target.stat().st_size
                final_path = src if cfg["in_place"] else target
                self._invalidate_track_cache(target)
                self._invalidate_track_cache(final_path)
                try:
                    final_tracks, final_size_bytes, _final_title = (
                        self._get_tracks_cached(final_path)
                    )
                    final_size = fmt_size(final_size_bytes)
                except (RuntimeError, OSError, TypeError, ValueError):
                    final_tracks = tracks
                    final_size = size_str
                return RunWorkerOutcome(
                    path=src,
                    result=result,
                    summary=summary,
                    state="written",
                    before_size=result.src_size,
                    after_size=result.dst_size,
                    display_tracks=final_tracks,
                    display_size=final_size,
                )

            self.call_from_thread(
                self._log_event,
                f"{src.name}: running mkvmerge ({len(cmd)} arg(s))",
                "INFO",
            )
        returncode, _, stderr, was_cancelled = self._run_subprocess_cancellable(cmd)
        if was_cancelled:
            return RunWorkerOutcome(
                path=src,
                result=result,
                summary=summary,
                state="cancelled",
                before_size=result.src_size,
            )
        if returncode not in (0, 1):
            result.errored = True
            err = stderr.strip()
            return RunWorkerOutcome(
                path=src,
                result=result,
                summary=summary,
                state="error",
                before_size=result.src_size,
                error=f"mkvmerge failed: {err}",
                log_messages=[("ERROR", f"mkvmerge failed for {src.name}: {err}")],
            )

        result.dst_size = dst.stat().st_size
        final_path = dst
        if cfg["in_place"]:
            src.unlink()
            dst.rename(src)
            final_path = src

        self._invalidate_track_cache(final_path)
        try:
            final_tracks, final_size_bytes, _final_title = self._get_tracks_cached(
                final_path
            )
            final_size = fmt_size(final_size_bytes)
        except (RuntimeError, OSError, TypeError, ValueError):
            final_tracks = tracks
            final_size = size_str

        return RunWorkerOutcome(
            path=src,
            result=result,
            summary=summary,
            state="written",
            before_size=result.src_size,
            after_size=result.dst_size,
            display_tracks=final_tracks,
            display_size=final_size,
        )

    @work(thread=True, exclusive=True)
    def _run_process(self, path: Path, cfg: RunConfig, source_label: str) -> None:
        self._cancel_event.clear()
        self.call_from_thread(setattr, self, "is_busy", True)

        dry_prefix = "[DRY RUN] " if cfg["dry_run"] else ""
        self.call_from_thread(
            self._log_event, f"{dry_prefix}Run started for {source_label}", "INFO"
        )
        self.call_from_thread(self._set_status, "Processing…", "cyan")

        try:
            selected_paths = [Path(p).expanduser() for p in cfg["selected_paths"]]
            if cfg["use_selection"]:
                files = collect_files_from_selection(selected_paths, cfg["recursive"])
                self.call_from_thread(
                    self._log_event,
                    (f"Run using explicit selection ({len(selected_paths)} path(s))"),
                    "INFO",
                )
            else:
                files = collect_files(path, cfg["recursive"])
        except ValueError as e:
            self.call_from_thread(self._log_event, str(e), "ERROR")
            self.call_from_thread(self._set_status, str(e), "red")
            self.call_from_thread(setattr, self, "is_busy", False)
            return

        total = len(files)
        self.call_from_thread(
            self._log_event, f"Run discovered {total} file(s).", "INFO"
        )
        self.call_from_thread(
            self._log_event,
            (
                "Run config: "
                f"jobs={cfg['jobs']} recursive={cfg['recursive']} "
                f"dry_run={cfg['dry_run']} in_place={cfg['in_place']} "
                f"use_selection={cfg['use_selection']}"
            ),
            "INFO",
        )

        def _prep_inspector():
            table = self._track_table()
            table.clear()
            self._inspect_rows.clear()
            self._inspect_source_rows.clear()
            self._file_row_keys.clear()
            self._file_row_offsets.clear()
            self._update_inspector_filter_summary()
            self.query_one(self._SEL_TABS, TabbedContent).active = "tab-summary"

        self.call_from_thread(_prep_inspector)
        self.call_from_thread(
            self._render_live_summary_intro, source_label, total, cfg["dry_run"]
        )

        self._last_inspected_path = path
        self._last_inspected_source_label = source_label
        self._last_inspected_time = time.time()
        self.call_from_thread(self._update_inspect_header, True)

        pb = self.query_one("#progress-bar", ProgressBar)
        pl = self.query_one("#progress-label", Static)
        pe = self.query_one("#progress-eta", Static)
        self.call_from_thread(pb.update, total=total, progress=0)
        self.call_from_thread(pe.update, "")

        _start_time = time.monotonic()
        last_ui_update = 0.0
        self._active_jobs = cfg["jobs"]
        self._reset_perf_counters()
        results: list[FileSummary] = []
        cancelled = False
        row_batch: list[tuple[Path, str, list[TrackInfo], str]] = []
        last_row_flush = 0.0
        emitted = 0
        last_detail_update = 0.0
        last_live_log_update = 0.0
        detail_every = 1 if total <= 200 else self._run_detail_batch_size

        def _flush_row_batch(force: bool = False) -> None:
            nonlocal last_row_flush
            if not row_batch:
                return
            now = time.monotonic()
            if (
                not force
                and len(row_batch) < self._row_batch_size
                and (now - last_row_flush) < self._row_flush_interval
            ):
                return
            payload = list(row_batch)
            row_batch.clear()
            last_row_flush = now

            def _apply_rows() -> None:
                for batch_src, batch_size, batch_tracks, batch_prefix in payload:
                    self._add_file_rows(
                        batch_src, batch_size, batch_tracks, batch_prefix
                    )

            self.call_from_thread(_apply_rows)

        jobs = cfg["jobs"]
        self.call_from_thread(
            self._log_event,
            f"Run worker pool: {jobs} parallel job(s).",
            "INFO",
        )

        executor = ThreadPoolExecutor(max_workers=jobs)
        future_map = {
            executor.submit(self._run_process_one_file, src, cfg): (idx, src)
            for idx, src in enumerate(files)
        }
        queued_preview = ", ".join(src.name for src in files[:8])
        if len(files) > 8:
            queued_preview += f", +{len(files) - 8} more"
        self.call_from_thread(
            self._log_event,
            f"Run queued files: {queued_preview}"
            if queued_preview
            else "Run queued files: <none>",
            "INFO",
        )
        completed = 0

        try:
            for future in as_completed(future_map):
                _, _ = future_map[future]
                if self._cancel_event.is_set():
                    cancelled = True
                    break

                outcome = future.result()
                completed += 1

                now = time.monotonic()
                if (
                    completed == 1
                    or completed == total
                    or (now - last_ui_update) >= self._progress_update_interval
                ):
                    elapsed = now - _start_time
                    if completed > 1 and elapsed > 3:
                        rate = elapsed / completed
                        remaining = rate * (total - completed)
                        eta_str = (
                            f"  ETA {remaining:.0f}s"
                            if remaining < 60
                            else f"  ETA {remaining / 60:.1f}m"
                        )
                    else:
                        eta_str = ""
                    perf_note = self._perf_note()
                    self.call_from_thread(
                        pl.update,
                        (
                            f"  {completed}/{total}  "
                            f"{100 * completed // total}%  •  {perf_note}"
                        ),
                    )
                    self.call_from_thread(pe.update, eta_str)
                    self.call_from_thread(pb.update, total=total, progress=completed)
                    last_ui_update = now

                if outcome.state == "cancelled":
                    cancelled = True
                    break

                for level, msg in outcome.log_messages:
                    self.call_from_thread(self._log_event, msg, level)

                results.append(outcome.result)
                emitted += 1
                self.call_from_thread(
                    self._log_event,
                    self._build_outcome_log_line(outcome, cfg["dry_run"]),
                    "INFO" if outcome.state != "error" else "ERROR",
                )

                should_render_detail = (
                    outcome.state == "error"
                    or emitted == total
                    or emitted % detail_every == 0
                    or (now - last_detail_update) >= self._live_detail_interval
                )
                if should_render_detail:
                    self.call_from_thread(
                        self._render_live_file_changes,
                        outcome.path,
                        outcome.summary,
                        outcome.state,
                        cfg["dry_run"],
                        outcome.before_size,
                        outcome.after_size,
                        outcome.error,
                    )
                    last_detail_update = now

                should_log_checkpoint = (
                    completed == 1
                    or completed == total
                    or (now - last_live_log_update) >= self._live_log_interval
                )
                if should_log_checkpoint:
                    self.call_from_thread(
                        self._log_event,
                        (
                            f"Run progress: {completed}/{total} files completed"
                            f"  •  {self._perf_note()}"
                        ),
                        "INFO",
                    )
                    last_live_log_update = now

                if outcome.display_tracks or outcome.state == "error":
                    row_batch.append(
                        (
                            outcome.path,
                            outcome.display_size,
                            outcome.display_tracks,
                            outcome.display_prefix,
                        )
                    )
                    _flush_row_batch(force=False)

                if cancelled:
                    break
        finally:
            executor.shutdown(wait=not cancelled, cancel_futures=cancelled)

        _flush_row_batch(force=True)

        self.call_from_thread(self._auto_fit_data_columns)
        self._pending_inspector_autofit = 0
        self.call_from_thread(self._render_summary, results, cfg["dry_run"], True)
        self._last_inspected_path = path
        self._last_inspected_source_label = source_label
        self._last_inspected_time = time.time()
        self._last_inspect_config_hash = _inspect_config_hash(cfg)
        self.call_from_thread(self._update_inspect_header, False)
        self.call_from_thread(self._persist_inspect_state)
        self._persist_track_cache()
        if cancelled:
            self.call_from_thread(self._log_event, "Run cancelled.", "WARN")
            self.call_from_thread(
                self._set_status,
                f"Run cancelled.  •  {self._perf_note()}",
                "yellow",
            )
        else:
            self.call_from_thread(
                self._log_event,
                f"Run perf snapshot: {self._perf_note()}",
                "INFO",
            )
        self.call_from_thread(setattr, self, "is_busy", False)

    def _render_summary(
        self,
        results: list[FileSummary],
        dry_run: bool,
        preserve_existing: bool = False,
    ) -> None:
        processed = [r for r in results if not r.skipped and not r.errored]
        skipped = [r for r in results if r.skipped]
        errored = [r for r in results if r.errored]

        total_src = sum(r.src_size for r in processed)
        total_dst = sum(r.dst_size for r in processed)
        total_saved = total_src - total_dst

        audio_rm = sum(1 for r in results for t in r.removed if t[0] == "audio")
        sub_rm = sum(1 for r in results for t in r.removed if t[0] == "subtitle")
        default_flip_count = sum(len(r.default_changes) for r in results)

        slog = self.query_one(self._SEL_SUMMARY_LOG, RichLog)
        if not preserve_existing:
            slog.clear()
            slog.write(
                self._summary_header("Run summary", "— cleaned up and ready to inspect")
            )
        else:
            slog.write(Text("═" * 64, style="dim"))
            slog.write(self._summary_header("Overall totals", "— run complete"))
        slog.write("")
        slog.write(self._summary_section("FILES"))
        slog.write(self._summary_line("Total", str(len(results))))
        slog.write(self._summary_line("Processed", str(len(processed))))
        slog.write(self._summary_line("Skipped", str(len(skipped))))
        if errored:
            slog.write(
                self._summary_line(
                    "Errored", str(len(errored)), self._SUMMARY_ERROR_STYLE
                )
            )

        if processed and not dry_run:
            slog.write("")
            slog.write(self._summary_section("SPACE"))
            slog.write(self._summary_line("Before", fmt_size(total_src)))
            slog.write(self._summary_line("After", fmt_size(total_dst)))
            slog.write(
                self._summary_line(
                    "Saved",
                    fmt_delta(total_saved, total_src),
                    self._SUMMARY_SUCCESS_STYLE,
                )
            )

        if audio_rm or sub_rm:
            slog.write("")
            slog.write(self._summary_section("TRACKS REMOVED"))
            if audio_rm:
                slog.write(self._summary_line("Audio", str(audio_rm)))
            if sub_rm:
                slog.write(self._summary_line("Subtitles", str(sub_rm)))

        if default_flip_count:
            slog.write("")
            slog.write(self._summary_section("DEFAULT FLAGS CHANGED"))
            slog.write(self._summary_line("Tracks", str(default_flip_count)))

        if errored:
            slog.write("")
            slog.write(self._summary_section("ERRORED FILES"))
            for r in errored:
                slog.write(Text(f"  • {r.path.name}", style="red"))

        self.query_one(self._SEL_TABS, TabbedContent).active = "tab-summary"

        if dry_run:
            status = "Dry run complete"
        else:
            saved_text = fmt_delta(total_saved, total_src) if total_src else "0 MB (0%)"
            status = f"Done — saved {saved_text}"
        status = f"{status}  •  {self._perf_note()}"
        self._set_status(status, "bold green")
        self._log_event(
            f"Run complete — {len(processed)} processed, {len(skipped)} skipped"
            + (f", {len(errored)} errored" if errored else ""),
            "INFO",
        )


# ── CLI mode ─────────────────────────────────────────────────────────────────


def _format_track_flags(track: TrackInfo) -> str:
    """Return default/forced flag markers for one track."""
    flags: list[str] = []
    if track.default:
        flags.append("default")
    if track.forced:
        flags.append("forced")
    return f"[{', '.join(flags)}]" if flags else ""


def _print_tracks_cli(
    tracks: list[TrackInfo], mkv_path: Path, file_size: int | None = None
) -> None:
    """Print a plain terminal table of tracks for CLI inspect mode."""
    size_str = f"  ({fmt_size(file_size)})" if file_size is not None else ""
    print(f"\n{'─' * 62}")
    print(f"  File : {mkv_path.name}{size_str}")
    print(f"{'─' * 62}")
    if not tracks:
        print("  (no tracks found)")
        return

    tag_strings = [_format_track_flags(t) for t in tracks]
    tag_width = max((len(s) for s in tag_strings), default=0)
    for t, tag_str in zip(tracks, tag_strings):
        print(
            f"  {tag_str:<{tag_width}}  Track {t.tid:>2}  {t.ttype:<12} lang={t.lang:<6}  "
            f"codec={t.codec:<20}  name={t.name or '—'}"
        )
    print()


def _process_file_cli(
    src: Path,
    options: Mapping[str, object],
) -> FileSummary:
    """Process one MKV file in non-TUI mode and print progress to stdout."""
    result = FileSummary(path=src)

    try:
        tracks, current_title = get_tracks_and_title(src)
    except (RuntimeError, OSError, TypeError, ValueError) as exc:
        print(f"[SKIP] {src.name} — could not read tracks: {exc}\n")
        result.errored = True
        return result

    result.src_size = src.stat().st_size
    _print_tracks_cli(tracks, src, file_size=result.src_size)

    sync_title_to_filename = bool(options.get("sync_title_to_filename", True))
    dry_run = bool(options.get("dry_run", False))
    in_place = bool(options.get("in_place", False))

    dst = src.with_stem(src.stem + ".cleaned")
    desired_title = _desired_output_title(src, dst, in_place)
    title_changed = bool(
        sync_title_to_filename
        and desired_title is not None
        and current_title != desired_title
        and tracks
    )
    title_skipped = bool(
        sync_title_to_filename
        and desired_title is not None
        and current_title != desired_title
        and not tracks
    )
    cmd, summary = build_mkvmerge_cmd(
        src, dst, tracks, options, title=desired_title if title_changed else None
    )

    nothing_removed = not summary["audio_removed"] and not summary["subs_removed"]
    defaults_changed = summary["defaults_changed"]

    if nothing_removed and not defaults_changed and not title_changed:
        if title_skipped:
            print("  ⚠  Skipping title sync because no tracks were detected.")
        print("  ✓  Nothing to do — file already matches criteria. Skipping.\n")
        result.skipped = True
        return result

    for t in summary["audio_removed"]:
        print(
            f"  ✂  Remove audio    : Track {t.tid}  lang={t.lang}  name={t.name or '—'}"
        )
    for t in summary["subs_removed"]:
        print(
            f"  ✂  Remove subtitle : Track {t.tid}  lang={t.lang}  name={t.name or '—'}"
        )

    audio_defs = summary["audio_defaults"]
    sub_defs = summary["sub_defaults"]
    for t in summary["audio_keep"]:
        if audio_defs.get(t.tid) and not t.default:
            print(
                f"  ★  Set default audio    : Track {t.tid}  lang={t.lang}  name={t.name or '—'}"
            )
    for t in summary["subs_keep"]:
        if sub_defs.get(t.tid) and not t.default:
            print(
                f"  ★  Set default subtitle : Track {t.tid}  lang={t.lang}  name={t.name or '—'}"
            )

    if title_changed:
        print(f"  ★  Set title          : {desired_title}")

    result.removed = [
        ("audio", t.tid, t.lang, t.name) for t in summary["audio_removed"]
    ] + [("subtitle", t.tid, t.lang, t.name) for t in summary["subs_removed"]]

    if dry_run:
        print(f"\n  [DRY RUN] Would write → {dst.name}\n")
        return result

    metadata_only = nothing_removed and (defaults_changed or title_changed)
    optional_tools = check_optional_tools()
    if metadata_only and optional_tools.get("mkvpropedit", False):
        target = src if in_place else dst
        if not in_place:
            try:
                shutil.copy2(src, target)
            except OSError as exc:
                print(
                    f"  [ERROR] could not create output file for metadata edit: {exc}\n"
                )
                result.errored = True
                return result

        cmd = build_mkvpropedit_cmd(
            target,
            tracks,
            summary,
            title=desired_title if title_changed else None,
        )
        if len(cmd) > 2:
            print(f"\n  ⚡ Metadata-only update → {target.name} ...")
            try:
                run = _run_subprocess_with_timeout(
                    cmd,
                    timeout=_SUBPROCESS_WRITE_TIMEOUT_SECONDS,
                )
            except RuntimeError as exc:
                print(f"  [ERROR] mkvpropedit failed:\n{exc}\n")
                result.errored = True
                return result
            if run.returncode not in (0, 1):
                print(f"  [ERROR] mkvpropedit failed:\n{run.stderr.strip()}\n")
                result.errored = True
                return result

            result.dst_size = target.stat().st_size
            saved = result.src_size - result.dst_size
            if in_place:
                print("  ✅ Done (updated original metadata in-place)")
            else:
                print(f"  ✅ Done → {target.name}")
            print(f"     Before : {fmt_size(result.src_size)}")
            print(f"     After  : {fmt_size(result.dst_size)}")
            print(f"     Saved  : {fmt_delta(saved, result.src_size)}\n")
            return result

    print(f"\n  ⏳ Writing → {dst.name} ...")
    try:
        run = _run_subprocess_with_timeout(
            cmd,
            timeout=_SUBPROCESS_WRITE_TIMEOUT_SECONDS,
        )
    except RuntimeError as exc:
        print(f"  [ERROR] mkvmerge failed:\n{exc}\n")
        result.errored = True
        return result
    if run.returncode not in (0, 1):
        print(f"  [ERROR] mkvmerge failed:\n{run.stderr.strip()}\n")
        result.errored = True
        return result

    result.dst_size = dst.stat().st_size
    saved = result.src_size - result.dst_size

    if in_place:
        src.unlink()
        dst.rename(src)
        print("  ✅ Done (replaced original)")
    else:
        print(f"  ✅ Done → {dst.name}")

    print(f"     Before : {fmt_size(result.src_size)}")
    print(f"     After  : {fmt_size(result.dst_size)}")
    print(f"     Saved  : {fmt_delta(saved, result.src_size)}\n")
    return result


def _print_folder_summary_cli(results: list[FileSummary], dry_run: bool) -> None:
    """Print aggregate summary for CLI folder runs."""
    total = len(results)
    processed = [r for r in results if not r.skipped and not r.errored]
    skipped = [r for r in results if r.skipped]
    errored = [r for r in results if r.errored]

    total_src = sum(r.src_size for r in processed)
    total_dst = sum(r.dst_size for r in processed)
    total_saved = total_src - total_dst

    print("=" * 62)
    print("  SUMMARY")
    print("=" * 62)
    print(f"  Files found      : {total}")
    print(f"  Files processed  : {len(processed)}")
    print(f"  Files skipped    : {len(skipped)}  (already matched criteria)")
    if errored:
        print(f"  Files errored    : {len(errored)}")

    if processed and not dry_run:
        print()
        print(f"  Total before     : {fmt_size(total_src)}")
        print(f"  Total after      : {fmt_size(total_dst)}")
        print(f"  Total saved      : {fmt_delta(total_saved, total_src)}")

    all_removed = [r for r in results for _ in r.removed]
    if all_removed:
        audio_rm = sum(1 for r in results for t in r.removed if t[0] == "audio")
        sub_rm = sum(1 for r in results for t in r.removed if t[0] == "subtitle")
        print()
        if audio_rm:
            print(f"  Audio tracks removed    : {audio_rm}")
        if sub_rm:
            print(f"  Subtitle tracks removed : {sub_rm}")

    if errored:
        print()
        print("  Errored files:")
        for r in errored:
            print(f"    • {r.path.name}")

    print("=" * 62)


def _bool_with_default(value: bool | None, default: bool) -> bool:
    """Resolve tri-state argparse booleans where None means use default."""
    return default if value is None else bool(value)


def _add_bool_toggle(
    parser: argparse.ArgumentParser,
    name: str,
    on_help: str,
    off_help: str,
) -> None:
    """Register paired --foo / --no-foo flags with None default."""
    flag = name.replace("_", "-")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{flag}", dest=name, action="store_true", help=on_help)
    group.add_argument(f"--no-{flag}", dest=name, action="store_false", help=off_help)
    parser.set_defaults(**{name: None})


def _build_parser() -> argparse.ArgumentParser:
    """Build a shared parser for TUI and CLI modes."""
    parser = argparse.ArgumentParser(
        description="MKV Cleaner — interactive TUI and non-interactive CLI.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="",
        help="Path to an MKV file or folder.",
    )

    parser.add_argument(
        "--cli",
        "--no-tui",
        dest="cli",
        action="store_true",
        help="Run in non-interactive CLI mode (no TUI).",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect tracks only. No files are written.",
    )
    parser.add_argument(
        "--keep-audio",
        nargs="+",
        metavar="LANG",
        help="Language codes to keep for audio tracks.",
    )
    parser.add_argument(
        "--keep-subs",
        nargs="+",
        metavar="LANG",
        help="Language codes to keep for subtitle tracks.",
    )
    parser.add_argument(
        "--remove-named",
        nargs="+",
        metavar="NAME",
        help="Remove tracks whose name contains any token (case-insensitive).",
    )
    parser.add_argument(
        "--default-audio",
        metavar="LANG_OR_NAME",
        help="Set default audio by language code or name match.",
    )
    parser.add_argument(
        "--default-subs",
        metavar="LANG_OR_NAME",
        help="Set default subtitle by language code or name match.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Parallel workers for inspect metadata reads (default: auto).",
    )

    _add_bool_toggle(
        parser,
        "recursive",
        "Search subfolders when path is a folder.",
        "Do not search subfolders when path is a folder.",
    )
    _add_bool_toggle(
        parser, "no_subs", "Strip all subtitle tracks.", "Keep subtitle tracks."
    )
    _add_bool_toggle(
        parser, "dry_run", "Preview only; do not write files.", "Write output files."
    )
    _add_bool_toggle(
        parser,
        "in_place",
        "Replace original files after writing cleaned output.",
        "Keep originals and write .cleaned.mkv outputs.",
    )
    _add_bool_toggle(
        parser,
        "auto_default",
        "Auto-promote a kept track when default track gets removed.",
        "Disable auto-promotion of default tracks.",
    )
    _add_bool_toggle(
        parser,
        "fix_missing_default",
        "Assign a default track when none exists.",
        "Do not auto-assign default tracks when missing.",
    )
    _add_bool_toggle(
        parser,
        "sync_title_to_filename",
        "Sync the file title to the filename.",
        "Do not sync the file title to the filename.",
    )
    _add_bool_toggle(
        parser,
        "protect_single_audio",
        "Keep a file's sole audio track even if filters would remove it.",
        "Allow removing a file's sole audio track.",
    )
    _add_bool_toggle(
        parser,
        "protect_single_sub",
        "Keep a file's sole subtitle track even if filters would remove it.",
        "Allow removing a file's sole subtitle track.",
    )
    _add_bool_toggle(
        parser,
        "save_log_file",
        "Enable TUI file logging to .mkv-cleaner-tui.log.",
        "Disable TUI file logging.",
    )
    _add_bool_toggle(
        parser,
        "auto_scroll",
        "Enable auto-scroll in TUI logs/tables.",
        "Disable auto-scroll in TUI logs/tables.",
    )

    return parser


def _run_cli_mode(args: argparse.Namespace) -> int:
    """Execute the cleaner in non-interactive CLI mode."""
    if not args.path:
        print("[ERROR] CLI mode requires a file or folder path.")
        return 2

    path = Path(args.path).expanduser()
    if not path.exists():
        print(f"[ERROR] Path not found: {path}")
        return 2

    check_dependencies()

    recursive = _bool_with_default(args.recursive, False)
    dry_run = _bool_with_default(args.dry_run, False)
    in_place = _bool_with_default(args.in_place, False)
    no_subs = _bool_with_default(args.no_subs, False)
    auto_default = _bool_with_default(args.auto_default, True)
    fix_missing_default = _bool_with_default(args.fix_missing_default, True)
    sync_title_to_filename = _bool_with_default(args.sync_title_to_filename, True)
    protect_single_audio = _bool_with_default(args.protect_single_audio, True)
    protect_single_sub = _bool_with_default(args.protect_single_sub, True)
    jobs = resolve_jobs(args.jobs, fallback=default_jobs())

    try:
        files = collect_files(path, recursive)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 2

    is_folder = path.is_dir()

    if args.inspect:
        if is_folder:
            print(f"\nInspecting {len(files)} file(s) in: {path}")
            print(f"Using {jobs} parallel worker(s) for metadata inspection.")
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            future_map = {
                executor.submit(get_tracks, file_path): (order_idx, file_path)
                for order_idx, file_path in enumerate(files)
            }
            pending: dict[
                int, tuple[Path, list[TrackInfo] | None, Exception | None]
            ] = {}
            next_emit = 0
            for future in as_completed(future_map):
                order_idx, file_path = future_map[future]
                try:
                    pending[order_idx] = (file_path, future.result(), None)
                except (RuntimeError, OSError, TypeError, ValueError) as exc:
                    pending[order_idx] = (file_path, None, exc)

                while next_emit in pending:
                    emit_path, emit_tracks, emit_error = pending.pop(next_emit)
                    next_emit += 1
                    if emit_error is not None:
                        print(f"[SKIP] {emit_path.name} — {emit_error}\n")
                        continue
                    _print_tracks_cli(
                        cast(list[TrackInfo], emit_tracks),
                        emit_path,
                        file_size=emit_path.stat().st_size,
                    )
        return 0

    if not any(
        [
            args.keep_audio,
            args.keep_subs,
            no_subs,
            args.remove_named,
            args.default_audio,
            args.default_subs,
            fix_missing_default,
            sync_title_to_filename,
        ]
    ):
        print("[ERROR] Nothing to do — specify at least one actionable option:")
        print("  --keep-audio, --keep-subs, --no-subs, --remove-named,")
        print("  --default-audio, --default-subs")
        return 2

    print(f"\nFound {len(files)} MKV file(s).")
    if dry_run:
        print("*** DRY RUN — no files will be written ***")

    results: list[FileSummary] = []
    for file_path in files:
        file_options: dict[str, object] = {
            "keep_audio_langs": args.keep_audio or [],
            "keep_sub_langs": args.keep_subs or [],
            "no_subs": no_subs,
            "remove_named": args.remove_named or [],
            "auto_default": auto_default,
            "fix_missing_default": fix_missing_default,
            "default_audio": args.default_audio,
            "default_subs": args.default_subs,
            "sync_title_to_filename": sync_title_to_filename,
            "protect_single_audio": protect_single_audio,
            "protect_single_sub": protect_single_sub,
            "dry_run": dry_run,
            "in_place": in_place,
        }
        result = _process_file_cli(
            src=file_path,
            options=file_options,
        )
        results.append(result)

    if is_folder:
        _print_folder_summary_cli(results, dry_run=dry_run)
    else:
        print("Done.")

    return 1 if any(r.errored for r in results) else 0


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    """Parse arguments and launch either CLI mode or TUI mode."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.cli:
        raise SystemExit(_run_cli_mode(args))

    overrides: dict[str, object] = {}
    for key in (
        "recursive",
        "no_subs",
        "auto_default",
        "fix_missing_default",
        "sync_title_to_filename",
        "protect_single_audio",
        "protect_single_sub",
        "dry_run",
        "in_place",
        "save_log_file",
        "jobs",
        "auto_scroll",
    ):
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value

    if args.path:
        overrides["path"] = args.path
    if args.keep_audio is not None:
        overrides["keep_audio_langs"] = " ".join(args.keep_audio)
    if args.keep_subs is not None:
        overrides["keep_sub_langs"] = " ".join(args.keep_subs)
    if args.remove_named is not None:
        overrides["remove_named"] = " ".join(args.remove_named)
    if args.default_audio is not None:
        overrides["default_audio"] = args.default_audio
    if args.default_subs is not None:
        overrides["default_subs"] = args.default_subs

    app = MkvCleanerApp(initial_path=args.path, initial_overrides=overrides)
    app.run()


if __name__ == "__main__":
    main()
