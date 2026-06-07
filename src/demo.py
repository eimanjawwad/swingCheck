import sys
import time
from pose_extractor import extract_pose
from classifier import classify_stroke, evaluate_form

def run_demo(video_path):
    print(f"\n🎾 SwingCheck — analyzing {video_path}\n")
    print("=" * 50)

    # Stage 1: Pose extraction
    print("📍 Stage 1: Pose extraction...")
    output_path = video_path.replace(".mp4", "_demo_output.mp4")
    keypoints, timing = extract_pose(video_path, output_path)

    print(f"\n⏱  Timing Report:")
    print(f"   Total processing time : {timing['total_s']}s")
    print(f"   Per-frame latency     : {timing['per_frame_ms']}ms")
    print(f"   Effective throughput  : {timing['fps']} FPS")
    print(f"   Frames processed      : {timing['frames']}")

    # Stage 2: Stroke classification
    print("\n🏸 Stage 2: Stroke classification...")
    t0 = time.time()
    stroke = classify_stroke(keypoints)
    t_classify = (time.time() - t0) * 1000
    print(f"   Detected stroke type  : {stroke.upper()}")
    print(f"   Classification time   : {t_classify:.1f}ms")

    # Stage 3: Form feedback
    print("\n📊 Stage 3: Form feedback...")
    feedback = evaluate_form(keypoints, stroke)
    for line in feedback:
        print(f"   {line}")

    print(f"\n✅ Output saved to: {output_path}")
    print("=" * 50)

if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "data/self/forehand/clip1.mp4"
    run_demo(video)