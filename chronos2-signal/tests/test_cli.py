"""Tests for the command line entry points.

Every command must run offline and must refuse, with a reason, anything that
would need data or weights the project does not have.
"""

from __future__ import annotations

import json

from chronos2_signal.cli import main


def test_global_flags_work_in_either_position(capsys):
    """``--json`` is accepted before or after the subcommand.

    Both spellings are natural to type, so both must parse; a value given
    before the subcommand must not be clobbered by the subcommand's copy.
    """
    assert main(["--json", "verify"]) == 0
    before = json.loads(capsys.readouterr().out)
    assert main(["verify", "--json"]) == 0
    after = json.loads(capsys.readouterr().out)
    assert before["design_fingerprint"] == after["design_fingerprint"]

    assert main(["verify"]) == 0
    plain = capsys.readouterr().out
    assert plain.startswith("design_version:")


def test_verify_reports_no_problems(capsys):
    assert main(["--json", "verify"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["problems"] == []
    assert payload["status"] == "design_only_unvalidated"
    assert payload["is_validated"] is False
    assert payload["channels"] == {
        "task_channels": 12,
        "calendar_channels": 5,
        "economic_features": 20,
        "preprocessing_columns": 21,
    }
    assert payload["registered_variants"] == ["B0", "C128", "C256", "C512", "U256"]
    # Absent optional packages are expected and reported, not hidden.
    assert "chronos-forecasting" in payload["absent_optional_packages"]
    assert len(payload["design_fingerprint"]) == 64


def test_schedule_reports_the_available_chronology(capsys):
    assert main(["--json", "schedule", "--latest-session", "2026-09-25"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["test_origins"] == 60
    assert payload["labelled_origins"] > payload["development_origins"]
    assert payload["label"] == "post-checkpoint study window"
    assert payload["earliest_origin"] == "2025-10-31"
    for fold in payload["folds"]:
        assert fold["fit_sessions"] >= 80
        assert fold["calibration_sessions"] == 30
        assert fold["validation_sessions"] == 20


def test_schedule_labels_a_pre_checkpoint_window_as_diagnostic(capsys):
    assert (
        main(
            [
                "--json",
                "schedule",
                "--latest-session",
                "2026-09-25",
                "--earliest-origin",
                "2024-07-01",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["label"].startswith("DIAGNOSTIC")
    assert "not evidence" in payload["label"]


def test_schedule_reports_insufficient_history_without_inventing_folds(capsys):
    """A short window yields no folds rather than a shortened block."""
    assert main(["--json", "schedule", "--latest-session", "2025-12-19"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["folds"] == []
    assert payload["development_origins"] == 0
    assert any("collect more dates" in note for note in payload["notes"])


def test_audit_dry_run_makes_no_request(capsys):
    assert (
        main(["--json", "audit", "--roster", "config/candidate_roster.example.csv"]) == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"].startswith("dry run")
    assert "results" not in payload
    assert len(payload["roster_hash"]) == 64
    intervals = {request["interval"] for request in payload["requests"]}
    assert intervals == {"1h", "1d", "15m"}
    assert payload["request_options"]["auto_adjust"] is False
    assert payload["request_options"]["prepost"] is False
    assert payload["request_options"]["actions"] is True
    assert "SPY" in payload["context_symbols"]


def test_audit_rejects_a_missing_roster(capsys):
    assert main(["audit", "--roster", "does/not/exist.csv"]) == 2
    assert "roster error" in capsys.readouterr().err


def test_smoke_refuses_without_the_model_extra(capsys):
    """No checkpoint means no forecast, and a pointer to the missing extra.

    Written to pass either way: the expected state here is that the model extra
    is absent, but the assertions still hold if someone installs it.
    """
    exit_code = main(["--json", "smoke"])
    payload = json.loads(capsys.readouterr().out)
    if payload.get("status") == "unavailable":
        assert exit_code == 2
        assert "chronos-forecasting is not installed" in payload["reason"]
        assert "do not substitute" in payload["next_step"]
    else:  # pragma: no cover - only when the model extra is installed
        assert exit_code in (0, 1)
        assert payload["shape"] == payload["expected_shape"]


def test_demo_study_writes_artifacts_and_says_what_they_are(tmp_path, capsys):
    exit_code = main(
        [
            "demo-study",
            "--variant",
            "B0",
            "--out",
            str(tmp_path),
            "--bootstrap-samples",
            "50",
        ]
    )
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "RESEARCH / UNVALIDATED" in output
    assert "Promotion gates" in output

    report = json.loads((tmp_path / "demo-study-B0.json").read_text(encoding="utf-8"))
    assert report["gates"]["passed"] is False
    assert report["output_label"] == "RESEARCH / UNVALIDATED"
    assert any("SYNTHETIC DATA" in note for note in report["notes"])
    assert set(report["controls"]) == {"momentum", "spy_exposure_matched", "cash"}
    assert (tmp_path / "demo-study-B0.md").is_file()
