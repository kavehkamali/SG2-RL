#!/usr/bin/env bash
# Record APF path-planning videos (A: polyline only, B: gripper follows path).
# Run on the Isaac + UWLab machine (e.g. tai). Uses run_on_tai.sh for PYTHONPATH, assets, and cwd.
#
# Usage:
#   chmod +x scripts/run_apf_path_videos.sh
#   ./scripts/run_apf_path_videos.sh
# Optional env:
#   VIDEO_LENGTH_A=300 VIDEO_LENGTH_B=480 EXTRA_ARGS="--headless"
#   UWLAB_CLOUD_ASSETS_DIR=...   # or SG2_CLOUD_ASSETS_DIR (see run_on_tai.sh)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export SG2_RL="${SG2_RL:-$ROOT}"
RUN="${ROOT}/scripts/run_on_tai.sh"
chmod +x "${RUN}" 2>/dev/null || true

VIDEO_LENGTH_A="${VIDEO_LENGTH_A:-300}"
VIDEO_LENGTH_B="${VIDEO_LENGTH_B:-480}"
# Pass through any extra args (e.g. --headless) to both recordings.
EXTRA=("$@")

echo "[sg2_rl] APF video A (path polyline only), length=${VIDEO_LENGTH_A}"
"${RUN}" record_path_apf_visual_only.py --headless --video_length "${VIDEO_LENGTH_A}" "${EXTRA[@]}"

echo "[sg2_rl] APF video B (path + gripper IK), length=${VIDEO_LENGTH_B}"
"${RUN}" record_path_apf_follow_gripper.py --headless --video_length "${VIDEO_LENGTH_B}" "${EXTRA[@]}"

echo "[sg2_rl] Done. MP4s under ${ROOT}/artifacts/videos/ (latest apf_path_* folders)."
