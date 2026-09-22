"""Extract fixed-length MediaPipe Holistic landmarks from WLASL videos."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm


POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
FACE_LANDMARKS = 468
HOLISTIC_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task"
FRAME_SIZE = (640, 480)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("WSLA100/wlasl100-new/WLASL_100"),
        help="Dataset root containing train, val, and test directories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("WSLA100/processed_landmarks"),
        help="Directory where compressed landmark files will be written.",
    )
    parser.add_argument(
        "--modality",
        choices=("hands", "hand_face", "hand_face_pose"),
        default="hands",
        help="Landmark groups to save for each frame.",
    )
    parser.add_argument("--frames", type=int, default=30, help="Output frames per video.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many videos in total; useful for local smoke tests.",
    )
    return parser.parse_args()


def landmark_array(landmarks: object, count: int) -> np.ndarray:
    """Return xyz coordinates, using zeros when a landmark group is absent."""
    if landmarks is None or len(landmarks) == 0:
        return np.zeros((count, 3), dtype=np.float32)
    points = landmarks if isinstance(landmarks, list) else landmarks.landmark
    coordinates = np.asarray([[point.x, point.y, point.z] for point in points], dtype=np.float32)
    if len(coordinates) != count:
        padded = np.zeros((count, 3), dtype=np.float32)
        padded[: min(count, len(coordinates))] = coordinates[:count]
        return padded
    return coordinates


def frame_features(results: object, modality: str) -> np.ndarray:
    parts = [
        landmark_array(results.left_hand_landmarks, HAND_LANDMARKS),
        landmark_array(results.right_hand_landmarks, HAND_LANDMARKS),
    ]
    if modality in {"hand_face", "hand_face_pose"}:
        parts.append(landmark_array(results.face_landmarks, FACE_LANDMARKS))
    if modality == "hand_face_pose":
        parts.append(landmark_array(results.pose_landmarks, POSE_LANDMARKS))
    return np.concatenate(parts).reshape(-1)


def sample_frame_indices(frame_count: int, target_frames: int) -> np.ndarray:
    if frame_count <= 0:
        raise ValueError("Video contains no readable frames")
    return np.linspace(0, frame_count - 1, target_frames).round().astype(int)


class TasksHolisticAdapter:
    def __init__(self, landmarker: object, image_format: object) -> None:
        self.landmarker = landmarker
        self.image_format = image_format

    def process(self, rgb_frame: np.ndarray) -> object:
        image = mp.Image(image_format=self.image_format, data=rgb_frame)
        return self.landmarker.detect(image)

    def close(self) -> None:
        try:
            self.landmarker.close()
        except RuntimeError:
            pass


def create_holistic(model_path: Path) -> object:
    if hasattr(mp, "solutions"):
        return mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False,
        )

    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import HolisticLandmarker, HolisticLandmarkerOptions

    if not model_path.exists():
        print(f"Downloading MediaPipe Holistic model to: {model_path}")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HOLISTIC_MODEL_URL, model_path)
    options = HolisticLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(model_path)))
    return TasksHolisticAdapter(HolisticLandmarker.create_from_options(options), mp.ImageFormat.SRGB)


def extract_video(video_path: Path, holistic: object, target_frames: int, modality: str) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sample_frame_indices(frame_count, target_frames)
    features = []
    next_index = 0
    for frame_index in range(frame_count):
        success, frame = capture.read()
        if not success:
            break
        if frame_index != indices[next_index]:
            continue
        frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)
        features.append(frame_features(results, modality))
        if next_index == len(indices) - 1:
            break
        next_index += 1
    capture.release()
    if not features:
        raise ValueError("No readable sampled frames")
    if len(features) < target_frames:
        features.extend([features[-1]] * (target_frames - len(features)))
    if len(features) > target_frames:
        features = features[:target_frames]
    return np.asarray(features, dtype=np.float32)


def video_paths(input_root: Path, limit: int | None) -> list[Path]:
    paths = sorted(input_root.glob("*/**/*.mp4"))
    return paths if limit is None else paths[:limit]


def main() -> None:
    args = parse_args()
    if args.frames < 1:
        raise ValueError("--frames must be at least 1")
    videos = video_paths(args.input_root, args.limit)
    if not videos:
        raise FileNotFoundError(f"No .mp4 files found below {args.input_root}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    holistic = create_holistic(args.output_root / "holistic_landmarker.task")
    failures = []
    try:
        for video_path in tqdm(videos, desc=f"Extracting {args.modality}"):
            relative_path = video_path.relative_to(args.input_root).with_suffix(".npz")
            output_path = args.output_root / args.modality / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.exists():
                continue
            try:
                keypoints = extract_video(video_path, holistic, args.frames, args.modality)
                np.savez_compressed(
                    output_path,
                    keypoints=keypoints,
                    label=video_path.parent.name,
                    video_id=video_path.stem,
                    split=video_path.parts[-3],
                )
            except (OSError, ValueError, cv2.error) as error:
                failures.append(f"{video_path}: {error}")
    finally:
        holistic.close()

    print(f"Processed: {len(videos) - len(failures)} / {len(videos)}")
    if failures:
        print("Failures:")
        print("\n".join(failures))


if __name__ == "__main__":
    main()