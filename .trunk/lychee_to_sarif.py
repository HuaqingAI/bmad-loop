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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="", help="workspace root, to relativize paths")
    args = ap.parse_args()

    raw = sys.stdin.read().strip()
    # A clean run still exits 0 with a full JSON report, but guard the empty
    # case so a no-op invocation cannot crash the parser and fail the lint.
    report = json.loads(raw) if raw else {}
    results = [
        to_result(relativize(path, args.root), entry)
        for path, entries in (report.get("error_map") or {}).items()
        for entry in entries
    ]
    sarif = {
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [{"results": results}],
    }
    print(json.dumps(sarif, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
