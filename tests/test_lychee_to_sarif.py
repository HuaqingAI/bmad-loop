"""Unit tests for .trunk/lychee_to_sarif.py — the JSON→SARIF parser behind the
`lychee` trunk linter.

`.trunk/trunk.yaml` gives that linter `success_codes: [0, 2]`, and 2 is exactly
lychee's "I found broken links" exit — so the SARIF `results` array this script
emits is the *only* thing that can fail the lint. A parser that stopped
recognizing lychee's report shape would emit a well-formed empty SARIF and wave
broken links through a green CI, which is what the error_map test below guards.
That fixture is trimmed from a real `lychee --offline --include-fragments
--format json` run (0.24.2), not hand-invented from the parser's own reading.

`.trunk` is not a package, so the module is loaded off sys.path the way
tests/test_seed_skills.py loads scripts/.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

TRUNK_DIR = Path(__file__).resolve().parent.parent / ".trunk"
sys.path.insert(0, str(TRUNK_DIR))

import lychee_to_sarif  # noqa: E402


def _run(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    payload: str,
    root: str = "",
) -> tuple[int, dict]:
    """Drive main() end to end on `payload`, returning (exit code, parsed SARIF)."""
    argv = ["lychee_to_sarif.py"] + (["--root", root] if root else [])
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(sys, "stdin", io.StringIO(payload))
    rc = lychee_to_sarif.main()
    return rc, json.loads(capsys.readouterr().out)


def test_error_map_entry_becomes_a_located_sarif_result(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """A broken link in lychee's report must reach SARIF as one located result.

    This is the load-bearing assertion: renaming `error_map`, or reshaping the
    entry, silently drops to zero results and the lint goes green.
    """
    root = str(tmp_path)
    report = {
        "error_map": {
            str(tmp_path / "docs" / "src.md"): [
                {
                    "url": f"file://{tmp_path}/docs/target.md#no-such-anchor",
                    "status": {"text": "Cannot find fragment", "details": "Cannot find fragment"},
                    "span": {"line": 3, "column": 37},
                }
            ]
        }
    }

    rc, sarif = _run(monkeypatch, capsys, json.dumps(report), root)

    assert rc == 0
    (result,) = sarif["runs"][0]["results"]
    location = result["locations"][0]["physicalLocation"]
    assert location["artifactLocation"]["uri"] == "docs/src.md"
    assert location["region"] == {"startLine": 3, "startColumn": 37}
    assert result["level"] == "error"
    assert result["ruleId"] == lychee_to_sarif.RULE_ID
    assert "Cannot find fragment" in result["message"]["text"]


def test_missing_span_anchors_the_finding_at_the_top_of_the_file(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """An unattributable link keeps its result rather than being dropped."""
    report = {"error_map": {str(tmp_path / "docs" / "src.md"): [{"url": "missing.md"}]}}

    _, sarif = _run(monkeypatch, capsys, json.dumps(report), str(tmp_path))

    (result,) = sarif["runs"][0]["results"]
    region = result["locations"][0]["physicalLocation"]["region"]
    assert region == {"startLine": 1, "startColumn": 1}


def test_relativize_maps_scan_paths_into_the_repo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Absolute scan paths become repo-relative; anything else passes through.

    trunk cannot map an absolute URI back onto a file, so a regression here
    reports findings against paths that resolve to nothing.
    """
    # relpath() consults the cwd for a relative input, so pin it outside the root.
    monkeypatch.chdir(tmp_path.parent)
    root = str(tmp_path)

    assert lychee_to_sarif.relativize(str(tmp_path / "docs" / "a.md"), root) == "docs/a.md"
    assert lychee_to_sarif.relativize("docs/a.md", root) == "docs/a.md"
    outside = str(tmp_path.parent / "elsewhere.md")
    assert lychee_to_sarif.relativize(outside, root) == outside
    assert lychee_to_sarif.relativize(outside, "") == outside


def test_empty_input_yields_a_conformant_empty_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A no-op invocation emits valid SARIF instead of crashing the lint."""
    rc, sarif = _run(monkeypatch, capsys, "")

    assert rc == 0
    assert sarif["version"] == "2.1.0"
    (run,) = sarif["runs"]
    assert run["results"] == []
    # SARIF 2.1.0 requires run.tool.driver.name. trunk composes its own label
    # from the trunk.yaml linter name plus ruleId and never reads this, but a
    # conformant consumer rejects a run without it.
    assert run["tool"]["driver"]["name"] == "lychee"
