import argparse
from pathlib import Path

import cv2
import numpy as np
from skimage.transform import estimate_transform

from utils.mediapipe_utils import run_mediapipe


def compute_center_size(landmarks, scale=1.4):
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])

    old_size = (right - left + bottom - top) / 2.0
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    size = max(1.0, old_size * scale)
    return center, size


def compute_center_size_from_bbox(bbox, scale=1.4):
    left, top, right, bottom = bbox
    old_size = (right - left + bottom - top) / 2.0
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    size = max(1.0, old_size * scale)
    return center, size


def crop_face_tform(center, size, image_size=224):
    src_pts = np.array([
        [center[0] - size / 2, center[1] - size / 2],
        [center[0] - size / 2, center[1] + size / 2],
        [center[0] + size / 2, center[1] - size / 2],
    ])
    dst_pts = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])

    tform = estimate_transform("similarity", src_pts, dst_pts)
    return tform.params[:2]


def warp_affine(image_bgr, M, dsize):
    return cv2.warpAffine(
        image_bgr,
        M,
        dsize,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def center_crop_resize(image_bgr, size):
    h, w = image_bgr.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = image_bgr[y0 : y0 + side, x0 : x0 + side]
    return cv2.resize(crop, (size, size))


def process_video(
    input_path: Path,
    output_size=224,
    scale=1.4,
    smooth_alpha=0.9,
    min_alpha=0.4,
    fast_threshold=20.0,
    max_size_change=0.15,
    detector="mediapipe",
):
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    out_path = input_path.with_name(input_path.stem + "_cropped_face" + input_path.suffix)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (output_size, output_size))

    last_M = None
    smooth_center = None
    smooth_size = None

    face_app = None
    if detector == "insightface":
        try:
            from insightface.app import FaceAnalysis
        except Exception as exc:
            raise RuntimeError("insightface is not available in this environment") from exc
        face_app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        face_app.prepare(ctx_id=0, det_size=(640, 640))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        center = None
        size = None

        if detector == "mediapipe":
            kpts = run_mediapipe(frame)
            if kpts is not None:
                center, size = compute_center_size(kpts[..., :2], scale=scale)
        else:
            faces = face_app.get(frame)
            if faces:
                face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                center, size = compute_center_size_from_bbox(face.bbox, scale=scale)

        if center is not None:
            if smooth_center is None:
                smooth_center = center
                smooth_size = size
            else:
                delta = np.linalg.norm(center - smooth_center)
                t = min(1.0, delta / max(1.0, fast_threshold))
                alpha = smooth_alpha * (1.0 - t) + min_alpha * t

                smooth_center = alpha * smooth_center + (1.0 - alpha) * center

                max_up = smooth_size * (1.0 + max_size_change)
                max_down = smooth_size * (1.0 - max_size_change)
                size = float(np.clip(size, max_down, max_up))
                smooth_size = alpha * smooth_size + (1.0 - alpha) * size

            M = crop_face_tform(smooth_center, smooth_size, image_size=output_size)
            last_M = M

        if last_M is not None:
            cropped = warp_affine(frame, last_M, (output_size, output_size))
        else:
            cropped = center_crop_resize(frame, output_size)

        out.write(cropped)

    cap.release()
    out.release()

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop/align face from a video using MediaPipe or InsightFace")
    parser.add_argument("--input_path", required=True, help="Path to input video")
    parser.add_argument("--output_size", type=int, default=224, help="Output square size")
    parser.add_argument("--scale", type=float, default=1.4, help="Crop scale around face")
    parser.add_argument("--smooth_alpha", type=float, default=0.9, help="EMA smoothing for crop (0-1, higher is smoother)")
    parser.add_argument("--min_alpha", type=float, default=0.4, help="Minimum alpha during fast motion (0-1, lower follows faster)")
    parser.add_argument("--fast_threshold", type=float, default=20.0, help="Pixels of motion to reduce smoothing")
    parser.add_argument("--max_size_change", type=float, default=0.15, help="Max per-frame relative size change (0-1)")
    parser.add_argument("--detector", choices=["mediapipe", "insightface"], default="insightface", help="Face detector backend")
    args = parser.parse_args()

    process_video(
        Path(args.input_path),
        output_size=args.output_size,
        scale=args.scale,
        smooth_alpha=args.smooth_alpha,
        min_alpha=args.min_alpha,
        fast_threshold=args.fast_threshold,
        max_size_change=args.max_size_change,
        detector=args.detector,
    )
