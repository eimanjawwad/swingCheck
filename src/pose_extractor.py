import cv2
import mediapipe as mp
import numpy as np
import sys
from pathlib import Path

# NEW
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def extract_pose(video_path: str, output_path: str = None):
    """
    Run MediaPipe pose extraction on a video.
    Saves annotated video and returns per-frame keypoint data.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if output_path is None:
        output_path = str(Path(video_path).stem) + "_pose.mp4"

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    all_keypoints = []   # list of dicts, one per frame

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,       
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # MediaPipe needs RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            frame_data = {"frame": frame_idx, "landmarks": None, "angles": None}

            if results.pose_landmarks:
                # Draw skeleton overlay
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

                # Extract raw landmark coords (normalized 0-1)
                lm = results.pose_landmarks.landmark
                keypoints = {
                    name: {
                        "x": lm[idx].x, "y": lm[idx].y,
                        "z": lm[idx].z, "vis": lm[idx].visibility
                    }
                    for name, idx in KEYPOINT_IDS.items()
                }

                # Compute joint angles
                angles = compute_angles(keypoints, w, h)
                frame_data["landmarks"] = keypoints
                frame_data["angles"] = angles

                # Overlay angle values on frame
                overlay_angles(frame, angles, keypoints, w, h)

            # Frame counter
            cv2.putText(frame, f"Frame {frame_idx}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            out.write(frame)
            all_keypoints.append(frame_data)
            frame_idx += 1

    cap.release()
    out.release()
    print(f"Done! Saved to {output_path} ({frame_idx} frames processed)")
    return all_keypoints


# ── Key joints for tennis 
KEYPOINT_IDS = {
    "left_shoulder":  11, "right_shoulder": 12,
    "left_elbow":     13, "right_elbow":    14,
    "left_wrist":     15, "right_wrist":    16,
    "left_hip":       23, "right_hip":      24,
    "left_knee":      25, "right_knee":     26,
    "left_ankle":     27, "right_ankle":    28,
}


def angle_between(a, b, c):
    """Angle at point b, given points a-b-c (in pixel coords)."""
    ba = np.array([a[0]-b[0], a[1]-b[1]], dtype=float)
    bc = np.array([c[0]-b[0], c[1]-b[1]], dtype=float)
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))


def compute_angles(kp, w, h):
    """Compute tennis-relevant joint angles from normalized keypoints."""
    def px(name):
        return (kp[name]["x"] * w, kp[name]["y"] * h)

    angles = {}
    try:
        # Right elbow angle (shoulder → elbow → wrist)
        angles["right_elbow"] = angle_between(
            px("right_shoulder"), px("right_elbow"), px("right_wrist"))
        # Left elbow
        angles["left_elbow"] = angle_between(
            px("left_shoulder"), px("left_elbow"), px("left_wrist"))
        # Right shoulder (hip → shoulder → elbow)
        angles["right_shoulder"] = angle_between(
            px("right_hip"), px("right_shoulder"), px("right_elbow"))
        # Right knee (hip → knee → ankle)
        angles["right_knee"] = angle_between(
            px("right_hip"), px("right_knee"), px("right_ankle"))
        # Left knee
        angles["left_knee"] = angle_between(
            px("left_hip"), px("left_knee"), px("left_ankle"))
    except Exception:
        pass
    return angles


def overlay_angles(frame, angles, kp, w, h):
    """Draw angle values next to the relevant joints."""
    label_positions = {
        "right_elbow":    ("right_elbow",    (15, 0)),
        "right_shoulder": ("right_shoulder", (15, 0)),
        "right_knee":     ("right_knee",     (15, 0)),
    }
    for angle_name, (joint, offset) in label_positions.items():
        if angle_name not in angles:
            continue
        x = int(kp[joint]["x"] * w) + offset[0]
        y = int(kp[joint]["y"] * h) + offset[1]
        cv2.putText(frame, f"{angle_name.split('_')[1]}: {angles[angle_name]:.0f}°",
                    (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2)


# Run
if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "data/pro/forehand/clip1.mp4"
    output = sys.argv[1].replace(".mp4", "_pose.mp4") if len(sys.argv) > 1 else "output/clip1_pose.mp4"
    keypoints = extract_pose(video, output)
    print(f"Extracted keypoints from {len(keypoints)} frames")