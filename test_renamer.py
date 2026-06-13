#!/usr/bin/env python3
"""
Extensive test suite for mkv_renamer episode detection and renaming logic.

Usage:
    python test_renamer.py                  # Run all tests
    python test_renamer.py --list           # List all tests
    python test_renamer.py --grep "10"      # Run tests matching keyword

To add new tests, append tuples to the TEST_CASES list below:
    ("Description of what this tests", "filename_without_extension", "E01")

where the expected value is:
    - The expected episode token (e.g. "E01", "SP01", "OVA02")
    - None if no episode should be detected (treated as movie/no episode)
"""

from __future__ import annotations

import sys
import argparse

from mkv_renamer import get_episode_info

sys.path.insert(0, ".")


# =============================================================================
# TEST CASES
# =============================================================================
# Format: (description, filename_stem, expected_token_or_None)
#
# Add your own tests by appending tuples here.
# The filename_stub should NOT include the extension (that's added internally).
# =============================================================================

TEST_CASES: list[tuple[str, str, str | None]] = [
    # -------------------------------------------------------------------------
    # Custom
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # SeasonEpisode pattern (SxxExx)
    # -------------------------------------------------------------------------
    ("SeasonEpisode basic S01E01", "Show S01E01 1080p", "E01"),
    ("SeasonEpisode S01E10", "Show S01E10 1080p", "E10"),
    ("SeasonEpisode S01E99", "Show S01E99 1080p", "E99"),
    ("SeasonEpisode S01E100", "Show S01E100 1080p", "E100"),
    ("SeasonEpisode S04E01", "[SubGroup] Anime S04E01 1080p", "E01"),
    ("SeasonEpisode S04E10", "[SubGroup] Anime S04E10 1080p", "E10"),
    ("SeasonEpisode S12E01", "Show S12E01.mkv", "E01"),
    ("SeasonEpisode with version", "Show S01E01v2 1080p", "E01"),
    ("SeasonEpisode with version S04E10v2", "Show S04E10v2 1080p", "E10"),
    ("SeasonEpisode fractional E01.5", "Show S01E01.5 1080p", "E01"),
    ("SeasonEpisode dot-separated", "Show.Name.S01E01.1080p", "E01"),
    ("SeasonEpisode dot-separated E10", "Show.Name.S01E10.1080p", "E10"),
    ("SeasonEpisode lowercase s01e01", "Show s01e01 1080p", "E01"),
    ("SeasonEpisode compact S01E01-1080p", "Show S01E01-1080p", "E01"),
    ("SeasonEpisode S01E10 double episode (not supported)", "Show S01E10S01E11", None),
    (
        "SeasonEpisode S04E10 in name",
        "That Time I Got Reincarnated as a Slime S04E10",
        "E10",
    ),
    # -------------------------------------------------------------------------
    # LeadingEpisode pattern (starts with E/EP/Episode)
    # -------------------------------------------------------------------------
    ("LeadingEpisode E01", "E01 Title", "E01"),
    ("LeadingEpisode E10", "E10 Title", "E10"),
    ("LeadingEpisode EP05", "EP05 Title", "E05"),
    ("LeadingEpisode EP10", "EP10", "E10"),
    ("LeadingEpisode Episode_10", "Episode_10", "E10"),
    ("LeadingEpisode E01 with text after", "E01 Some Text", "E01"),
    ("LeadingEpisode E99", "E99 Finale", "E99"),
    # -------------------------------------------------------------------------
    # ExplicitEpisode pattern (E/EP/Episode in middle or end)
    # -------------------------------------------------------------------------
    ("ExplicitEpisode Show E01", "Show E01 1080p", "E01"),
    ("ExplicitEpisode Show E10", "Show E10 1080p", "E10"),
    ("ExplicitEpisode Show EP10 Title", "Show EP10 Title", "E10"),
    ("ExplicitEpisode Show EP01", "Show EP01", "E01"),
    ("ExplicitEpisode Show Episode 03", "Show Episode 03", "E03"),
    # -------------------------------------------------------------------------
    # SpecialEpisode pattern (OVA, OAD, SP, etc.)
    # -------------------------------------------------------------------------
    ("SpecialEpisode SP01", "Show SP01", "SP01"),
    ("SpecialEpisode SP01 with text", "Show SP01 Title", "SP01"),
    ("SpecialEpisode OVA02", "Show OVA02", "OVA02"),
    ("SpecialEpisode OVA02 Unreleased", "Show OVA02 Unreleased", "OVA02"),
    ("SpecialEpisode NCOP01", "Show NCOP01", "NCOP01"),
    ("SpecialEpisode OAD", "Show OAD", "OAD"),
    ("SpecialEpisode Special", "Show SPECIAL", "SPECIAL"),
    ("SpecialEpisode ED01", "Show ED01", "ED01"),
    ("SpecialEpisode OP02", "Show OP02", "OP02"),
    ("SpecialEpisode PV", "Show PV", "PV"),
    # -------------------------------------------------------------------------
    # BracketEpisode pattern ([01], [10])
    # -------------------------------------------------------------------------
    ("BracketEpisode Show [01]", "Show [01] [1080p]", "E01"),
    ("BracketEpisode Show [10]", "Show [10] [1080p]", "E10"),
    ("BracketEpisode Show [01] only", "Show [01]", "E01"),
    ("BracketEpisode version [10v2]", "Show [10v2] [1080p]", "E10"),
    ("BracketEpisode [01] with spaces", "Show [ 01 ]", "E01"),
    # -------------------------------------------------------------------------
    # ParentheticalEpisode pattern ((01))
    # -------------------------------------------------------------------------
    ("ParentheticalEpisode (01)", "Show (01) 1080p", "E01"),
    ("ParentheticalEpisode (10)", "Show (10) 1080p", "E10"),
    ("ParentheticalEpisode (01) only - stripped by tag removal", "Show (01)", None),
    # -------------------------------------------------------------------------
    # BareEpisode pattern (- 01, - 10)
    # -------------------------------------------------------------------------
    ("BareEpisode Show - 01", "Show - 01", "E01"),
    ("BareEpisode Show - 10", "Show - 10", "E10"),
    ("BareEpisode Show - 10 1080p", "Show - 10 1080p", "E10"),
    ("BareEpisode Show - 10 Title", "Show - 10 Episode Title", "E10"),
    ("BareEpisode em-dash", "Show – 01", "E01"),
    ("BareEpisode em-dash E10", "Show – 10", "E10"),
    ("BareEpisode erai-raws ep 1", "[Erai-raws] Show - 1 [1080p]", "E01"),
    ("BareEpisode erai-raws ep 9", "[Erai-raws] Show - 9 [1080p]", "E09"),
    ("BareEpisode erai-raws ep 10", "[Erai-raws] Show - 10 [1080p]", "E10"),
    # -------------------------------------------------------------------------
    # CompactEpisode pattern (Show-01)
    # -------------------------------------------------------------------------
    ("CompactEpisode Show-01", "Show-01", "E01"),
    ("CompactEpisode Show-10", "Show-10", "E10"),
    ("CompactEpisode with extension", "Show-10", "E10"),
    # -------------------------------------------------------------------------
    # StandaloneEpisode pattern (01.mkv)
    # -------------------------------------------------------------------------
    ("StandaloneEpisode 01", "01", "E01"),
    ("StandaloneEpisode 10", "10", "E10"),
    ("StandaloneEpisode 99", "99", "E99"),
    ("StandaloneEpisode 100", "100", "E100"),
    # -------------------------------------------------------------------------
    # EpisodeTitle pattern (Show - 01 - Title)
    # -------------------------------------------------------------------------
    ("EpisodeTitle Show - 01 - Title", "Show - 01 - Episode Title", "E01"),
    ("EpisodeTitle Show - 10 - Title", "Show - 10 - Tenth Episode", "E10"),
    ("EpisodeTitle em-dash", "Show - 10 – Title", "E10"),
    # -------------------------------------------------------------------------
    # Series names with numbers in them (should NOT match those numbers)
    # -------------------------------------------------------------------------
    ("Series 100 Girlfriends ep 01", "100 Girlfriends - 01", "E01"),
    ("Series 100 Girlfriends ep 10", "100 Girlfriends - 10", "E10"),
    ("Series 86 ep 01", "86 - 01", "E01"),
    ("Series 86 ep 23", "86 - 23", "E23"),
    ("Series Gundam 00 ep 01", "Gundam 00 - 01", "E01"),
    ("Series 3-gatsu ep 05", "3-gatsu no Lion - 05", "E05"),
    # -------------------------------------------------------------------------
    # Version tags
    # -------------------------------------------------------------------------
    ("Version tag v2 in bare", "Show - 01v2 [1080p]", "E01"),
    ("Version tag v2 in season", "Show S01E10v2 1080p", "E10"),
    ("Version tag v2 in bracket", "Show [10v2] [1080p]", "E10"),
    ("Version tag v3", "Show - 01v3", "E01"),
    # -------------------------------------------------------------------------
    # Large episode numbers
    # -------------------------------------------------------------------------
    ("Large episode 99", "Show - 99", "E99"),
    ("Large episode 100", "Show - 100", "E100"),
    ("Large episode S01E100", "Show S01E100 1080p", "E100"),
    ("Large episode 200", "Show - 200", "E200"),
    # -------------------------------------------------------------------------
    # Dot-separated filenames
    # -------------------------------------------------------------------------
    ("Dot-separated Show.Name.E01.1080p", "Show.Name.E01.1080p", "E01"),
    ("Dot-separated S01E01", "Show.Name.S01E01.1080p", "E01"),
    ("Dot-separated S01E10", "Show.Name.S01E10.1080p", "E10"),
    # -------------------------------------------------------------------------
    # Resolution and codec info in filenames
    # -------------------------------------------------------------------------
    ("Resolution 1080p in brackets", "[Group] Show - 01 [1080p].mkv", "E01"),
    ("Resolution 720p in brackets", "[Group] Show - 01 [720p].mkv", "E01"),
    ("HEVC/H.265 codec", "[Group] Show - 01 [1080p][HEVC].mkv", "E01"),
    ("Multiple tags", "[Group] Show - 01 [1080p][HEVC][Multi Sub].mkv", "E01"),
    # -------------------------------------------------------------------------
    # Release group tags
    # -------------------------------------------------------------------------
    ("Leading group tag", "[SubGroup] Show - 01", "E01"),
    ("Multiple group tags", "[Group1][Group2] Show - 01", "E01"),
    ("Parenthesized group tag", "(SubGroup) Show - 01", "E01"),
    ("Braced group tag", "{SubGroup} Show - 01", "E01"),
    ("Group tag with CRC", "[SubGroup] Show - 01 [1080p][84E0FCD2]", "E01"),
    # -------------------------------------------------------------------------
    # Trailing text and titles
    # -------------------------------------------------------------------------
    ("Trailing episode title", "Show - 01 - My Episode Title [1080p]", "E01"),
    ("Trailing text no dash", "Show - 01 Episode Title [1080p]", "E01"),
    ("Title with numbers after dash", "Show - 10 - Episode Title", "E10"),
    # -------------------------------------------------------------------------
    # Filename formats without any episode info (should return None)
    # -------------------------------------------------------------------------
    ("No episode - movie", "Movie Title", None),
    ("No episode - just group", "[SubGroup] Release", None),
    ("No episode - just title", "Some Random Title", None),
    ("Empty after cleaning", "[Group]()", None),
    # -------------------------------------------------------------------------
    # Original bug scenario: mixed format renumbering correctness
    # -------------------------------------------------------------------------
    ("Erai-raws format - 1", "[Erai-raws] Show - 1 [1080p]", "E01"),
    ("Erai-raws format - 10", "[Erai-raws] Show - 10 [1080p]", "E10"),
    ("ToonsHub format S01E10", "[ToonsHub] Show S01E10 1080p", "E10"),
]


def run_detection_test(description: str, filename: str, expected: str | None) -> bool:
    """Run a single episode detection test. Returns True if passed."""
    result = get_episode_info(filename)
    detected = result.token if result else None
    passed = detected == expected
    status = "PASS" if passed else "FAIL"
    expected_disp = "<NONE>" if expected is None else expected
    detected_disp = "<NONE>" if detected is None else detected
    print(
        f"  {status:4s} | {description:50s} | expected={expected_disp:6s} | got={detected_disp:6s}"
    )
    return passed


def run_all_tests() -> None:
    """Run all registered test cases and report results."""
    total = len(TEST_CASES)
    passed = 0
    failed_cases: list[tuple[int, str, str, str | None, str | None]] = []

    print(f"\n{'=' * 100}")
    print(f"  Running {total} episode detection tests")
    print(f"{'=' * 100}\n")

    for idx, (description, filename, expected) in enumerate(TEST_CASES, 1):
        result = get_episode_info(filename)
        detected = result.token if result else None
        if detected == expected:
            passed += 1
            print(
                f"  PASS | {description:50s} | expected={str(expected):6s} | got={str(detected):6s}"
            )
        else:
            failed_cases.append((idx, description, filename, expected, detected))
            print(
                f"  FAIL | {description:50s} | expected={str(expected):6s} | got={str(detected):6s}"
            )

    print(f"\n{'=' * 100}")
    print(f"  Results: {passed}/{total} passed, {len(failed_cases)} failed")
    print(f"{'=' * 100}")

    if failed_cases:
        print("\n  FAILED TESTS:")
        for idx, desc, filename, expected, detected in failed_cases:
            print(f"    #{idx} {desc}")
            print(f"        filename: {filename}")
            print(f"        expected: {expected}, got: {detected}")
        print()


def list_tests() -> None:
    """List all registered test cases."""
    print(f"\nRegistered test cases ({len(TEST_CASES)}):\n")
    for idx, (description, filename, expected) in enumerate(TEST_CASES, 1):
        exp_str = "<NONE>" if expected is None else expected
        print(f"  {idx:3d}. [{exp_str:6s}] {description}")
        print(f"       filename: {filename[:80]}")
    print()


def grep_and_run(pattern: str) -> None:
    """Run tests matching a grep pattern in the description."""
    matching = [
        (desc, fn, exp)
        for desc, fn, exp in TEST_CASES
        if pattern.lower() in desc.lower() or pattern.lower() in fn.lower()
    ]
    if not matching:
        print(f"\n  No tests match pattern: {pattern}\n")
        return

    print(f"\n  Running {len(matching)} tests matching '{pattern}':\n")
    total = len(matching)
    passed = 0
    for desc, fn, exp in matching:
        if run_detection_test(desc, fn, exp):
            passed += 1
    print(f"\n  Results: {passed}/{total} passed")


def main() -> None:
    """Main entry point for the test suite."""
    parser = argparse.ArgumentParser(description="Test suite for mkv_renamer")
    parser.add_argument("--list", action="store_true", help="List all test cases")
    parser.add_argument("--grep", type=str, help="Run tests matching pattern")
    args = parser.parse_args()

    if args.list:
        list_tests()
        return

    if args.grep:
        grep_and_run(args.grep)
        return

    run_detection_test("", "", None)  # Warm up imports
    run_all_tests()


if __name__ == "__main__":
    main()
