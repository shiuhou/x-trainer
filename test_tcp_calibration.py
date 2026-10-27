"""Offline calibration input provenance and pivot numerical checks."""

import json

import pytest

from tcp_calibration import load_samples, rotation_matrix_xyz, solve_pivot


def samples():
    tcp, fixed = (5.0, -3.0, 197.0), (300.0, 100.0, 200.0)
    result = []
    for angles in [
        (0, 0, 0),
        (40, 0, 0),
        (0, 40, 0),
        (0, 0, 40),
        (-30, 20, 10),
        (20, -40, 50),
    ]:
        r = rotation_matrix_xyz(*angles)
        p = [fixed[i] - sum(r[i][j] * tcp[j] for j in range(3)) for i in range(3)]
        result.append([*p, *angles])
    return result


def test_pivot_recovers_synthetic_translation_only():
    result = solve_pivot(samples())
    assert result.tcp_flange_mm == pytest.approx([5, -3, 197], abs=1e-8)
    assert result.rms_residual_mm < 1e-8


@pytest.mark.parametrize(
    "metadata",
    [{}, {"tool": 1, "user": 0}, {"tool": 0, "user": 1}, {"tool": False, "user": 0}],
)
def test_load_rejects_unknown_or_nonbase_flange_frames(tmp_path, metadata):
    p = tmp_path / "samples.json"
    p.write_text(json.dumps(dict(metadata, samples=[{"pose": s} for s in samples()])))
    with pytest.raises(ValueError, match="user=0, tool=0"):
        load_samples(p)


def test_load_accepts_explicit_base_flange_frames(tmp_path):
    p = tmp_path / "samples.json"
    p.write_text(
        json.dumps({"user": 0, "tool": 0, "samples": [{"pose": s} for s in samples()]})
    )
    assert len(load_samples(p)) == 6


def test_identical_rotations_rejected():
    with pytest.raises(ValueError, match="diversity"):
        solve_pivot([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 6)


def test_sample_frame_cannot_override_document(tmp_path):
    p = tmp_path / "samples.json"
    p.write_text(
        json.dumps(
            {
                "user": 0,
                "tool": 0,
                "samples": [{"pose": s, "tool": 1} for s in samples()],
            }
        )
    )
    with pytest.raises(ValueError, match="consistent"):
        load_samples(p)


def test_audit_svd_and_heldout_synthetic(tmp_path):
    from tcp_calibration_audit import audit

    p = tmp_path / "samples.json"
    p.write_text(
        json.dumps({"user": 0, "tool": 0, "samples": [{"pose": s} for s in samples()]})
    )
    result = audit(p)
    assert result["design_rank"] == 6
    assert result["normal_equation_vs_svd_translation_mm"] < 1e-8
    assert max(result["leave_one_out_errors_mm"]) < 1e-8
    assert not result["calibration_accepted"]
    assert result["independent_physical_error_mm"] is None


def test_collector_rejects_tool_before_constructing_client(monkeypatch, tmp_path):
    import collect_tcp_samples as collector

    def forbidden(*args, **kwargs):
        pytest.fail("invalid frames must not construct a hardware client")

    monkeypatch.setattr(collector, "NovaClient", forbidden)
    with pytest.raises(SystemExit):
        collector.main(["--tool", "1", "--output", str(tmp_path / "new.json")])
