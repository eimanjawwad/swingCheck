import numpy as np

# Simple heuristic classifier based on wrist trajectory
# exactly what the TA suggested

def classify_stroke(keypoints_list):
    """
    keypoints_list: list of per-frame dicts from pose_extractor
    returns: "forehand", "backhand", "serve", or "unknown"
    """
    wrist_x = []
    wrist_y = []

    for frame in keypoints_list:
        if frame["landmarks"] is None:
            continue
        lm = frame["landmarks"]
        wrist_x.append(lm["right_wrist"]["x"])
        wrist_y.append(lm["right_wrist"]["y"])

    if len(wrist_x) < 10:
        return "unknown"

    wrist_x = np.array(wrist_x)
    wrist_y = np.array(wrist_y)

    # Serve: wrist moves significantly upward (y decreases in image coords)
    y_range = wrist_y.max() - wrist_y.min()
    y_drop  = wrist_y[0] - wrist_y.min()   # how much wrist rose
    if y_drop > 0.25 and y_drop / (y_range + 1e-6) > 0.6:
        return "serve"

    # Forehand vs backhand: direction of wrist sweep
    # Forehand (right-handed): wrist moves left-to-right across body
    # Backhand: wrist moves right-to-left
    x_start = np.mean(wrist_x[:len(wrist_x)//3])
    x_end   = np.mean(wrist_x[-len(wrist_x)//3:])
    delta_x = x_end - x_start

    if delta_x > 0.05:
        return "forehand"
    elif delta_x < -0.05:
        return "backhand"
    else:
        return "unknown"


def evaluate_form(keypoints_list, stroke_type):
    """
    Returns list of feedback strings based on joint angles at contact frame.
    Reference ranges from biomechanics literature.
    """
    # Find contact frame = frame with lowest wrist y (highest point) for serve,
    # or frame with most extreme wrist x for groundstrokes
    best_frame = None
    for frame in keypoints_list:
        if frame["angles"] and frame["landmarks"]:
            best_frame = frame
            break   # fallback: just use first valid frame

    if best_frame is None or not best_frame["angles"]:
        return ["Could not compute angles — no valid frame found"]

    angles = best_frame["angles"]
    feedback = []

    REFERENCE = {
        "forehand": {
            "right_elbow":    (130, 170, "Elbow"),
            "right_shoulder": (70,  110, "Shoulder"),
            "right_knee":     (140, 170, "Knee"),
        },
        "backhand": {
            "left_elbow":     (150, 180, "Elbow"),
            "right_shoulder": (60,  100, "Shoulder"),
            "right_knee":     (140, 170, "Knee"),
        },
        "serve": {
            "right_elbow":    (90,  140, "Elbow"),
            "right_shoulder": (80,  130, "Shoulder"),
            "right_knee":     (130, 160, "Knee"),
        },
    }

    refs = REFERENCE.get(stroke_type, REFERENCE["forehand"])
    for joint, (lo, hi, label) in refs.items():
        val = angles.get(joint)
        if val is None:
            continue
        if lo <= val <= hi:
            feedback.append(f"✅ {label}: {val:.0f}° (target {lo}–{hi}°) — good")
        elif val < lo:
            feedback.append(f"⚠️  {label}: {val:.0f}° (target {lo}–{hi}°) — too bent")
        else:
            feedback.append(f"⚠️  {label}: {val:.0f}° (target {lo}–{hi}°) — too extended")

    return feedback if feedback else ["No angle data available"]