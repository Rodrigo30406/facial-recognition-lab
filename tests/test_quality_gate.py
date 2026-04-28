import numpy as np

from eleccia_vision.application.quality_gate import (
    FaceObservation,
    QualityGateThresholds,
    build_angle_plan,
    classify_bucket,
    evaluate_quality_gate,
    next_target_bucket,
)


def test_build_angle_plan_distributes_across_all_buckets() -> None:
    plan = build_angle_plan(12)
    assert sum(plan.values()) == 12
    assert all(plan[bucket] >= 2 for bucket in ("center", "left", "right"))
    assert all(plan[bucket] >= 2 for bucket in ("up", "down"))


def test_next_target_bucket_prioritizes_largest_deficit() -> None:
    plan = {"center": 3, "left": 2, "right": 2, "up": 1, "down": 1}
    captured = {"center": 3, "left": 1, "right": 2, "up": 1, "down": 0}
    assert next_target_bucket(captured, plan) == "left"


def test_classify_bucket_from_pose() -> None:
    assert classify_bucket(yaw=-20.0, pitch=0.0) == "left"
    assert classify_bucket(yaw=20.0, pitch=0.0) == "right"
    assert classify_bucket(yaw=0.0, pitch=-15.0) == "up"
    assert classify_bucket(yaw=0.0, pitch=15.0) == "down"
    assert classify_bucket(yaw=0.0, pitch=0.0) == "center"


def test_quality_gate_green_when_target_angle_matches() -> None:
    frame = np.full((240, 320, 3), 130, dtype=np.uint8)
    obs = FaceObservation(
        bbox=(96.0, 48.0, 224.0, 192.0),
        det_score=0.95,
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
    )
    plan = {"center": 1, "left": 0, "right": 0, "up": 0, "down": 0}
    captured = {"center": 0, "left": 0, "right": 0, "up": 0, "down": 0}

    out = evaluate_quality_gate(
        frame,
        obs,
        QualityGateThresholds(min_sharpness=0.0),
        captured,
        plan,
    )
    assert out.status == "green"
    assert out.current_bucket == "center"


def test_quality_gate_yellow_when_quality_ok_but_wrong_angle() -> None:
    frame = np.full((240, 320, 3), 130, dtype=np.uint8)
    obs = FaceObservation(
        bbox=(96.0, 48.0, 224.0, 192.0),
        det_score=0.95,
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
    )
    plan = {"center": 0, "left": 1, "right": 0, "up": 0, "down": 0}
    captured = {"center": 0, "left": 0, "right": 0, "up": 0, "down": 0}

    out = evaluate_quality_gate(
        frame,
        obs,
        QualityGateThresholds(min_sharpness=0.0),
        captured,
        plan,
    )
    assert out.status == "yellow"
    assert "Mueve a:" in out.reason


def test_quality_gate_red_when_face_too_small() -> None:
    frame = np.full((240, 320, 3), 130, dtype=np.uint8)
    obs = FaceObservation(
        bbox=(150.0, 110.0, 170.0, 130.0),
        det_score=0.95,
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
    )
    plan = {"center": 1, "left": 0, "right": 0, "up": 0, "down": 0}
    captured = {"center": 0, "left": 0, "right": 0, "up": 0, "down": 0}

    out = evaluate_quality_gate(frame, obs, QualityGateThresholds(min_face_ratio=0.10), captured, plan)
    assert out.status == "red"
    assert "Acercate" in out.reason
