#!/usr/bin/env python3
"""Convert `lychee --format json` output into SARIF for trunk.

lychee reports one `[path]:` header followed by its broken links, so no single
output line carries both the source path and the failure — and trunk's `output:
regex` requires a `path` capture group in every match. Reading the JSON
`error_map` (which *is* keyed by source path) sidesteps that shape mismatch.

Reads lychee JSON on stdin, writes SARIF on stdout.
"""

import argparse
import json
import os
import sys

RULE_ID = "broken-link"
# SARIF 2.1.0 requires `run.tool.driver.name`. trunk composes its own label from the
# trunk.yaml linter name plus `ruleId` and never reads this, but a conformant consumer
# rejects a run without it.
TOOL_NAME = "lychee"
# lychee's own "did this run fail" test is `error_map.is_empty() && timeout_map.is_empty()`,
# so either counter can be what earned a rc 2. This parser renders `error_map` only, so a
# nonzero `timeouts` would produce no results at all; unreachable while `--offline` is set
# (a filesystem check cannot time out) but silently fail-open if that flag were dropped.
FAILURE_COUNTERS = ("errors", "timeouts")


def relativize(path: str, root: str) -> str:
    """Make an absolute scan path repo-relative, which is what SARIF wants.

    lychee is pointed at the absolute workspace, so its `error_map` keys come
    back absolute; trunk cannot map those onto files. Anything already relative,
    or somehow outside the root, is passed through untouched.
    """
    if not root:
        return path
    try:
        rel = os.path.relpath(path, root)
    except ValueError:  # different drive on Windows
        return path
    return path if rel.startswith(os.pardir) else rel.replace(os.sep, "/")


def to_result(path: str, entry: dict) -> dict:
    """Render one `error_map` entry as a SARIF result."""
    # `span` is absent when lychee cannot attribute the link to a position
    # (e.g. a link recovered from a source it could not re-scan); anchor the
    # finding at the top of the file rather than dropping it.
    span = entry.get("span") or {}
    status = (entry.get("status") or {}).get("text") or "broken link"
    url = entry.get("url", "")
    return {
        "level": "error",
        "locations": [
            {
                "physicalLocation": {
                    "artifactLocation": {"uri": path},
                    "region": {
                        "startLine": span.get("line", 1),
                        "startColumn": span.get("column", 1),
                    },
                }
            }
        ],
        "message": {"text": f"{status}: {url}"},
        "ruleId": RULE_ID,
    }


def read_report(raw: str) -> dict:
    """Decode lychee's report, refusing any shape this parser cannot render.

    `success_codes: [0, 2]` makes the SARIF `results` array the only signal that
    can fail this lint, so an unreadable report has to fail the *parser* rather
    than emit a clean empty run — otherwise an upstream output-format change
    waves every broken link through a green CI. Requiring the key is safe: 0.24.2
    writes the full report on both 0 (clean, `error_map: {}`) and 2 (failures),
    and a fatal lychee exits 1, which `success_codes` rejects before the parser
    is ever invoked.
    """
    if not raw:
        raise ValueError("lychee wrote no report to stdout")
    try:
        report = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"lychee report is not JSON: {exc}") from exc
    if not isinstance(report, dict) or "error_map" not in report:
        raise ValueError("lychee report has no 'error_map' key; output format changed?")
    error_map = report["error_map"]
    if not isinstance(error_map, dict):
        raise ValueError(f"lychee 'error_map' is {type(error_map).__name__}, not an object")
    # Every 0.24.2 `error_map` value is an array of failure objects — including the
    # synthetic span-less `url: "error:"` entry lychee writes when it cannot read an
    # input at all. Any other type either drops silently to zero results (`{}`, `""`)
    # or dies inside to_result() with a bare AttributeError; name it here instead.
    for path, entries in error_map.items():
        if not isinstance(entries, list):
            raise ValueError(f"lychee 'error_map[{path}]' is {type(entries).__name__}, not a list")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="", help="workspace root, to relativize paths")
    args = ap.parse_args()

    try:
        report = read_report(sys.stdin.read().strip())
        results = [
            to_result(relativize(path, args.root), entry)
            for path, entries in report["error_map"].items()
            for entry in entries
        ]
        # A shape check on `error_map` alone still lets an *empty* value through, and
        # `[]` is a list. lychee counts every failure it found independently, so a
        # nonzero counter with nothing to show for it means the report carried failures
        # this parser could not render. Deliberately not `errors == len(results)`:
        # `error_map` values are Rust HashSets, so two byte-identical failures under one
        # key collapse, and an exact match would redden a valid report.
        counted = [name for name in FAILURE_COUNTERS if report.get(name)]
        if counted and not results:
            raise ValueError(
                f"lychee counted failures ({', '.join(counted)}) but the report yielded "
                "no SARIF results; output format changed?"
            )
    except ValueError as exc:
        # trunk copies parser stderr verbatim into its failure report and turns a
        # nonzero parser exit into a visible red lint, so this is the loud path.
        print(f"lychee_to_sarif: {exc}", file=sys.stderr)
        return 1
    sarif = {
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/sarif-2.1/schema/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [{"tool": {"driver": {"name": TOOL_NAME}}, "results": results}],
    }
    print(json.dumps(sarif, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
