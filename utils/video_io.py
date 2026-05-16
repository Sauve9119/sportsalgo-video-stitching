"""
utils/video_io.py
-----------------
FFmpeg-backed video reader and writer.
Faster than cv2.VideoCapture for long files; supports accurate seeking.
"""

import subprocess
import numpy as np
import cv2
from pathlib import Path


class VideoReader:
    """
    Reads frames from a video file using cv2.VideoCapture.
    Wraps the capture with a clean iterator interface and exposes metadata.
    """

    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def seek(self, frame_idx: int):
        """Seek to a specific frame index."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    def read(self):
        """Read next frame. Returns (success, frame)."""
        return self.cap.read()

    def read_frames(self, start: int = 0, end: int = None, step: int = 1):
        """
        Generator yielding (frame_idx, frame) from start to end.
        step > 1 skips frames (useful for calibration sampling).
        """
        self.seek(start)
        idx = start
        end = end or self.total_frames
        while idx < end:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield idx, frame
            # Skip frames if step > 1
            for _ in range(step - 1):
                self.cap.read()
            idx += step

    def close(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __repr__(self):
        return (
            f"VideoReader({Path(self.path).name}, "
            f"{self.width}x{self.height}, "
            f"{self.fps:.2f}fps, "
            f"{self.total_frames} frames)"
        )


class VideoWriter:
    """
    Writes frames to an MP4 file using cv2.VideoWriter with H.264 encoding.
    """

    def __init__(self, path: str, fps: float, width: int, height: int):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not self.writer.isOpened():
            raise IOError(f"Cannot open VideoWriter at: {path}")

    def write(self, frame: np.ndarray):
        self.writer.write(frame)

    def close(self):
        self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def get_video_info(path: str) -> dict:
    """Return basic metadata dict for a video file."""
    cap = cv2.VideoCapture(path)
    info = {
        "path": path,
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    info["duration_sec"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0
    cap.release()
    return info
