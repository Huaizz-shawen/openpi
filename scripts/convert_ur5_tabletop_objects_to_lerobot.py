"""
Convert local UR5 tabletop pick-and-place sessions to LeRobot format for OpenPI.

Expected raw layout:
  <raw_root>/
    cola_01/
      data/dataset_*.json
      videos/front_camera_*.mp4
      videos/left_camera_*.mp4
      videos/right_camera_*.mp4
    medicine_01/
      ...
    lemon_01/
      ...

The output dataset stores:
- observation.state: 7D UR5 state from arm_left joint positions
- action: next-step absolute joint positions
- observation.images.cam_front
- observation.images.cam_left
- task: per-episode language instruction derived from the folder prefix
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


OBJECT_NAMES = ("cola", "medicine", "lemon")


@dataclasses.dataclass(frozen=True)
class SessionData:
    session_id: str
    object_name: str
    json_path: Path
    front_video: Path
    left_video: Path

    @property
    def task(self) -> str:
        return f"pick {self.object_name} from table and place it to the basket."


def _find_video(videos_dir: Path, prefix: str) -> Path:
    candidates = sorted(videos_dir.glob(f"{prefix}_camera_*.mp4"))
    if not candidates:
        raise FileNotFoundError(f"No video found for prefix={prefix!r} under {videos_dir}")
    return candidates[0]


def _parse_object_name(session_dir: Path) -> str | None:
    for object_name in OBJECT_NAMES:
        if session_dir.name.startswith(f"{object_name}_"):
            return object_name
    return None


def discover_sessions(raw_root: Path) -> list[SessionData]:
    sessions: list[SessionData] = []
    for session_dir in sorted(raw_root.iterdir()):
        if not session_dir.is_dir():
            continue

        object_name = _parse_object_name(session_dir)
        if object_name is None:
            continue

        data_dir = session_dir / "data"
        videos_dir = session_dir / "videos"
        json_files = sorted(data_dir.glob("dataset_*.json"))
        if not json_files:
            continue

        try:
            front = _find_video(videos_dir, "front")
            left = _find_video(videos_dir, "left")
        except FileNotFoundError:
            continue

        sessions.append(
            SessionData(
                session_id=session_dir.name,
                object_name=object_name,
                json_path=json_files[0],
                front_video=front,
                left_video=left,
            )
        )
    return sessions


def split_sessions(
    sessions: list[SessionData],
    val_ratio: float,
    seed: int,
) -> tuple[list[SessionData], list[SessionData]]:
    if val_ratio <= 0:
        return list(sessions), []

    grouped: dict[str, list[SessionData]] = defaultdict(list)
    for session in sessions:
        grouped[session.object_name].append(session)

    train: list[SessionData] = []
    val: list[SessionData] = []
    rng = random.Random(seed)

    for object_name in OBJECT_NAMES:
        object_sessions = list(grouped[object_name])
        rng.shuffle(object_sessions)
        if not object_sessions:
            continue

        n_val = max(1, int(round(len(object_sessions) * val_ratio)))
        n_val = min(n_val, len(object_sessions) - 1) if len(object_sessions) > 1 else 0
        val.extend(object_sessions[:n_val])
        train.extend(object_sessions[n_val:])

    train.sort(key=lambda s: s.session_id)
    val.sort(key=lambda s: s.session_id)
    return train, val


def _resize_rgb(frame_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    import cv2

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if frame_rgb.shape[1] == width and frame_rgb.shape[0] == height:
        return frame_rgb
    return cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_AREA)


def _load_json_entries(json_path: Path) -> list[dict]:
    with json_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    data = obj.get("data", [])
    if not isinstance(data, list) or not data:
        raise ValueError(f"No valid data entries in {json_path}")
    return data


def _extract_state(entry: dict) -> np.ndarray:
    joints = entry["joints"]["arm_left"]
    pos = joints["joint_positions"]
    state = np.asarray(pos, dtype=np.float32)
    if state.shape != (7,):
        raise ValueError(f"Expected 7D arm_left joint_positions, got {state.shape}")
    return state


def _iter_video_frames(cap) -> Iterable[np.ndarray]:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame


def _build_episode_arrays(
    session: SessionData,
    image_width: int,
    image_height: int,
    normalize_gripper: bool,
) -> dict[str, list[np.ndarray]]:
    entries = _load_json_entries(session.json_path)
    states = [_extract_state(e) for e in entries]

    actions = states[1:] + [states[-1]]

    if normalize_gripper:
        g_min, g_max = 0.011764705882352941, 0.8705882352941177

        def _norm(v: float) -> float:
            if g_max <= g_min:
                return float(v)
            return float((v - g_min) / (g_max - g_min))

        states = [np.concatenate([s[:6], np.asarray([_norm(float(s[6]))], dtype=np.float32)]).astype(np.float32) for s in states]
        actions = [np.concatenate([a[:6], np.asarray([_norm(float(a[6]))], dtype=np.float32)]).astype(np.float32) for a in actions]

    import cv2

    cap_front = cv2.VideoCapture(str(session.front_video))
    cap_left = cv2.VideoCapture(str(session.left_video))
    if not cap_front.isOpened():
        raise RuntimeError(f"Failed to open front video for {session.session_id}")
    if not cap_left.isOpened():
        raise RuntimeError(f"Failed to open left video for {session.session_id}")

    try:
        front_iter = _iter_video_frames(cap_front)
        left_iter = _iter_video_frames(cap_left)
        last_front = None
        last_left = None

        out_front: list[np.ndarray] = []
        out_left: list[np.ndarray] = []

        for _ in range(len(states)):
            front = next(front_iter, None)
            left = next(left_iter, None)
            if front is None:
                if last_front is None:
                    raise RuntimeError(f"front video has no frames: {session.front_video}")
                front = last_front
            if left is None:
                if last_left is None:
                    raise RuntimeError(f"left video has no frames: {session.left_video}")
                left = last_left
            last_front = front
            last_left = left
            out_front.append(_resize_rgb(front, image_width, image_height))
            out_left.append(_resize_rgb(left, image_width, image_height))

    finally:
        cap_front.release()
        cap_left.release()

    return {
        "state": states,
        "action": actions,
        "front": out_front,
        "left": out_left,
    }


def _create_dataset(
    repo_id: str,
    fps: int,
    image_h: int,
    image_w: int,
    image_writer_threads: int,
    image_writer_processes: int,
    camera_dtype: str,
):
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    out_dir = Path(HF_LEROBOT_HOME) / repo_id
    if out_dir.exists():
        shutil.rmtree(out_dir)

    image_shape = (image_h, image_w, 3)
    image_names = ["height", "width", "channel"]
    if camera_dtype == "video":
        image_shape = (3, image_h, image_w)
        image_names = ["channels", "height", "width"]

    ds = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="ur5e",
        fps=fps,
        features={
            "observation.state": {
                "dtype": "float32",
                "shape": (7,),
                "names": [[
                    "shoulder_pan_joint",
                    "shoulder_lift_joint",
                    "elbow_joint",
                    "wrist_1_joint",
                    "wrist_2_joint",
                    "wrist_3_joint",
                    "left_finger_joint",
                ]],
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": [[
                    "shoulder_pan_joint",
                    "shoulder_lift_joint",
                    "elbow_joint",
                    "wrist_1_joint",
                    "wrist_2_joint",
                    "wrist_3_joint",
                    "left_finger_joint",
                ]],
            },
            "observation.images.cam_front": {
                "dtype": camera_dtype,
                "shape": image_shape,
                "names": image_names,
            },
            "observation.images.cam_left": {
                "dtype": camera_dtype,
                "shape": image_shape,
                "names": image_names,
            },
        },
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )
    return ds


def _write_dataset(
    sessions: list[SessionData],
    repo_id: str,
    fps: int,
    image_w: int,
    image_h: int,
    normalize_gripper: bool,
    image_writer_threads: int,
    image_writer_processes: int,
    camera_dtype: str,
):
    if not sessions:
        return

    ds = _create_dataset(
        repo_id=repo_id,
        fps=fps,
        image_h=image_h,
        image_w=image_w,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
        camera_dtype=camera_dtype,
    )

    for session in sessions:
        arrays = _build_episode_arrays(
            session,
            image_width=image_w,
            image_height=image_h,
            normalize_gripper=normalize_gripper,
        )
        n = len(arrays["state"])
        for i in range(n):
            ds.add_frame(
                {
                    "observation.state": arrays["state"][i],
                    "action": arrays["action"][i],
                    "observation.images.cam_front": arrays["front"][i],
                    "observation.images.cam_left": arrays["left"][i],
                    "task": session.task,
                }
            )
        ds.save_episode()


def _default_repo_id(raw_root: Path, split: str, camera_dtype: str) -> str:
    suffix = "video" if camera_dtype == "video" else "image"
    return str(raw_root / "lerobot" / f"ur5_tabletop_3obj_frontleft_{split}_{suffix}")


def main():
    parser = argparse.ArgumentParser(description="Convert UR5 tabletop object sessions to LeRobot format.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--repo-id-train", type=str, default="")
    parser.add_argument("--repo-id-val", type=str, default="")
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--image-height", type=int, default=180)
    parser.add_argument("--normalize-gripper", action="store_true", help="Normalize gripper from observed range to [0, 1].")
    parser.add_argument("--camera-dtype", choices=["image", "video"], default="video")
    parser.add_argument("--image-writer-threads", type=int, default=2)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Only print split summary, do not write dataset.")
    args = parser.parse_args()

    repo_id_train = args.repo_id_train or _default_repo_id(args.raw_root, "train", args.camera_dtype)
    repo_id_val = args.repo_id_val or _default_repo_id(args.raw_root, "val", args.camera_dtype)

    sessions = discover_sessions(args.raw_root)
    if not sessions:
        raise RuntimeError(f"No valid object sessions found under {args.raw_root}")

    train_sessions, val_sessions = split_sessions(sessions, val_ratio=args.val_ratio, seed=args.seed)

    print(f"Found sessions: {len(sessions)}")
    for object_name in OBJECT_NAMES:
        count = sum(1 for session in sessions if session.object_name == object_name)
        print(f"  {object_name}: {count}")
    print(f"Train sessions: {len(train_sessions)}")
    print(f"Val sessions: {len(val_sessions)}")
    print(f"Train repo_id: {repo_id_train}")
    if val_sessions:
        print(f"Val repo_id: {repo_id_val}")

    if args.dry_run:
        return

    _write_dataset(
        sessions=train_sessions,
        repo_id=repo_id_train,
        fps=args.fps,
        image_w=args.image_width,
        image_h=args.image_height,
        normalize_gripper=args.normalize_gripper,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
        camera_dtype=args.camera_dtype,
    )

    if val_sessions:
        _write_dataset(
            sessions=val_sessions,
            repo_id=repo_id_val,
            fps=args.fps,
            image_w=args.image_width,
            image_h=args.image_height,
            normalize_gripper=args.normalize_gripper,
            image_writer_threads=args.image_writer_threads,
            image_writer_processes=args.image_writer_processes,
            camera_dtype=args.camera_dtype,
        )

    print("Conversion completed.")


if __name__ == "__main__":
    main()
