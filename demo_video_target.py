import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.transform import estimate_transform

from src.smirk_encoder import SmirkEncoder
from utils.mediapipe_utils import run_mediapipe


def crop_face(frame, landmarks, scale=1.0, image_size=224):
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])

    h, w, _ = frame.shape
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])

    size = int(old_size * scale)

    # crop image
    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)

    return tform


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


def precrop_video(
    input_path: Path,
    output_path: Path,
    output_size=224,
    scale=1.4,
    smooth_alpha=0.9,
    min_alpha=0.3,
    fast_threshold=25.0,
    max_size_change=0.2,
    detector="insightface",
):
    print(f"---> RUNNING CROP ALIGN (using {detector})")
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (output_size, output_size))

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

    return output_path


def extract_auto_ref_image(video_path: Path, out_dir: Path, step: int = 10) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle = total // 2 if total > 0 else 0

    candidates = list(range(0, total, max(1, step)))
    candidates = sorted(candidates, key=lambda x: abs(x - middle))

    best_frame = None
    best_score = -1.0

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
    img_area = float(h * w)

    for idx in candidates:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        kpts = run_mediapipe(frame)
        if kpts is None:
            continue
        pts = kpts[..., :2]
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        face_area = max(1.0, (x1 - x0) * (y1 - y0))
        area_ratio = face_area / img_area
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        center_dx = abs(cx - w / 2) / (w / 2)
        center_dy = abs(cy - h / 2) / (h / 2)
        center_penalty = (center_dx + center_dy) / 2.0
        # sharpness (variance of Laplacian) helps avoid blurry frames
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_norm = min(1.0, sharpness / 200.0)

        score = area_ratio - 0.3 * center_penalty + 0.2 * sharpness_norm

        if score > best_score:
            best_score = score
            best_frame = frame.copy()

        if best_score > 0.15:
            break

    if best_frame is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError("Could not read any frame for auto ref.")
        best_frame = frame

    cap.release()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'auto_ref.jpg'
    cv2.imwrite(str(out_path), best_frame)
    return out_path


def run_mica_and_get_identity(
    mica_repo_dir: Path,
    ref_image_path: Path,
    mica_checkpoint: Path,
    keep_outputs: bool = False,
) -> np.ndarray:
    print("---> RUNNING MICA FACE DETECTOR")
    mica_repo_dir = Path(mica_repo_dir).resolve()
    demo_py = mica_repo_dir / 'demo.py'
    if not demo_py.exists():
        raise FileNotFoundError(f"MICA demo.py not found at {demo_py}")

    mica_checkpoint = Path(mica_checkpoint).resolve()

    tmp_root = Path(tempfile.mkdtemp(prefix='mica_run_'))
    input_dir = tmp_root / 'demo' / 'input'
    output_dir = tmp_root / 'demo' / 'output'
    arcface_dir = tmp_root / 'demo' / 'arcface'
    for d in [input_dir, output_dir, arcface_dir]:
        d.mkdir(parents=True, exist_ok=True)

    ref_basename = 'ref_image' + ref_image_path.suffix.lower()
    ref_copy = input_dir / ref_basename
    shutil.copy2(ref_image_path, ref_copy)

    cmd = [
        sys.executable,
        str(demo_py),
        '-i', str(input_dir),
        '-o', str(output_dir),
        '-a', str(arcface_dir),
        '-m', str(mica_checkpoint),
    ]

    try:
        subprocess.run(cmd, check=True, cwd=str(mica_repo_dir))
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to run MICA demo.py: {e}")

    stem = Path(ref_basename).stem
    identity_npy = output_dir / stem / 'identity.npy'
    if not identity_npy.exists():
        raise FileNotFoundError(f"MICA identity not found at {identity_npy}")

    identity = np.load(identity_npy)

    if not keep_outputs:
        try:
            shutil.rmtree(tmp_root)
        except Exception:
            pass

    return identity


def _interpolate_tensor(a, b, t):
    return a * (1.0 - t) + b * t


def _fill_missing_sequence(seq, interpolate_fn):
    indices = [i for i, v in enumerate(seq) if v is not None]
    if not indices:
        return seq

    for i in range(len(seq)):
        if seq[i] is not None:
            continue
        prev_idx = max([j for j in indices if j < i], default=None)
        next_idx = min([j for j in indices if j > i], default=None)

        if prev_idx is None:
            seq[i] = seq[next_idx]
        elif next_idx is None:
            seq[i] = seq[prev_idx]
        else:
            t = (i - prev_idx) / float(next_idx - prev_idx)
            seq[i] = interpolate_fn(seq[prev_idx], seq[next_idx], t)

    return seq


def _iter_target_videos(target_txt: Path, videos_root: Path):
    with target_txt.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            name = Path(line).name
            prefix = name.split("-Scene-")[0]
            yield videos_root / prefix / name


def _resolve_video_list(args):
    if args.target_txt:
        target_txt = Path(args.target_txt)
        videos_root = Path(args.videos_root)
        return list(_iter_target_videos(target_txt, videos_root))
    return [Path(args.input_path)]


def process_single_video(input_path: Path, args, smirk_encoder, mica_shape):
    print(f"---> PROCESSING VIDEO: {input_path}")
    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else Path(tempfile.mkdtemp(prefix="precrop_"))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_video = tmp_dir / f"{input_path.stem}_cropped{input_path.suffix}"
    print(f"---> TMP CROPPED VIDEO: {tmp_video}")

    precrop_video(
        input_path=input_path,
        output_path=tmp_video,
        output_size=args.crop_output_size,
        scale=args.crop_scale,
        smooth_alpha=args.crop_smooth_alpha,
        min_alpha=args.crop_min_alpha,
        fast_threshold=args.crop_fast_threshold,
        max_size_change=args.crop_max_size_change,
        detector=args.crop_detector,
    )

    try:
        cap = cv2.VideoCapture(str(tmp_video))
        if not cap.isOpened():
            raise RuntimeError(f"Error opening video file: {tmp_video}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)

        outputs_list = []
        tracked_frames_mask = []

        while True:
            ret, image = cap.read()
            if not ret:
                break

            kpt_mediapipe = run_mediapipe(image)

            cropped_image = image
            cropped_kpt_mediapipe = kpt_mediapipe[..., :2] if kpt_mediapipe is not None else None

            cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            cropped_image_rgb = cv2.resize(cropped_image_rgb, (224, 224))
            cropped_tensor = torch.tensor(cropped_image_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            cropped_tensor = cropped_tensor.to(args.device)

            outputs = smirk_encoder(cropped_tensor)
            outputs_list.append({k: v.detach().cpu() for k, v in outputs.items()})
            tracked_frames_mask.append(1)

        cap.release()

        def interp_outputs(a, b, t):
            return {k: _interpolate_tensor(a[k], b[k], t) for k in a}

        outputs_list = _fill_missing_sequence(outputs_list, interp_outputs)

        if all(v is None for v in outputs_list):
            raise RuntimeError("No valid detections were found in the video.")

        def _to_numpy_row(t):
            if isinstance(t, torch.Tensor):
                t = t.detach().cpu().numpy()
            return np.reshape(t, (1, -1))

        tracked_frames_mask_np = np.array(tracked_frames_mask, dtype=np.float32).reshape(-1, 1)

        exp_np = np.concatenate([_to_numpy_row(o['expression_params']) for o in outputs_list], axis=0)
        pose_np = np.concatenate([_to_numpy_row(o['pose_params']) for o in outputs_list], axis=0)
        eyes_np = np.concatenate([_to_numpy_row(o['eyelid_params']) for o in outputs_list], axis=0)
        jaw_np = np.concatenate([_to_numpy_row(o['jaw_params']) for o in outputs_list], axis=0)
        shape_np = mica_shape.detach().cpu().numpy().reshape(-1)

        npz_out_dir = Path(args.npz_out_dir)
        npz_out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = npz_out_dir / f"{input_path.stem}.npz"

        np.savez(
            npz_path,
            exp=exp_np,
            pose=pose_np,
            eyes=eyes_np,
            jaw=jaw_np,
            shape=shape_np,
            tracked_frames_mask=tracked_frames_mask_np,
            fps=np.array(video_fps, dtype=np.float32),
        )

        return npz_path
    finally:
        if not args.keep_tmp:
            try:
                if tmp_video.exists():
                    tmp_video.unlink()
            except Exception:
                pass
            if not args.tmp_dir:
                try:
                    shutil.rmtree(tmp_dir)
                except Exception:
                    pass


def _get_mica_shape(args, input_path: Path):
    if args.mica_identity:
        mica_shape_np = np.load(args.mica_identity)
    else:
        tmp_auto = None
        if args.mica_auto_ref:
            tmp_auto = Path(tempfile.mkdtemp(prefix='mica_auto_ref_'))
            ref_img_path = extract_auto_ref_image(input_path, tmp_auto)
        elif args.mica_ref_frame is not None:
            tmp_auto = Path(tempfile.mkdtemp(prefix='mica_ref_frame_'))
            cap_ref = cv2.VideoCapture(str(input_path))
            if not cap_ref.isOpened():
                raise RuntimeError('Error opening video file for reference frame')
            total = int(cap_ref.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            frame_idx = max(0, min(total - 1, args.mica_ref_frame)) if total > 0 else max(0, args.mica_ref_frame)
            cap_ref.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap_ref.read()
            cap_ref.release()
            if not ok:
                raise RuntimeError('Could not read reference frame from video')
            tmp_auto.mkdir(parents=True, exist_ok=True)
            ref_img_path = tmp_auto / 'ref_frame.jpg'
            cv2.imwrite(str(ref_img_path), frame)
        else:
            ref_img_path = Path(args.mica_ref_image)

        mica_shape_np = run_mica_and_get_identity(
            mica_repo_dir=Path(args.mica_repo),
            ref_image_path=Path(ref_img_path),
            mica_checkpoint=Path(args.mica_checkpoint),
            keep_outputs=args.mica_keep,
        )

        if tmp_auto is not None and not args.mica_keep:
            try:
                shutil.rmtree(tmp_auto)
            except Exception:
                pass

    return torch.tensor(mica_shape_np, dtype=torch.float32, device=args.device).view(1, -1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', type=str, default='samples/mead_90.png', help='Path to the input image/video')
    parser.add_argument('--target_txt', type=str, help='Path to a txt file listing target video filenames')
    parser.add_argument('--videos_root', type=str, default='/mnt/GSAMegaRepo/talkcuts/1_scene/cliped_videos', help='Root folder for scene clips')
    parser.add_argument('--npz_out_dir', type=str, default='/mnt/GSAMegaRepo/talkcuts/5_head_tracking/npz', help='Folder to save output npz files')
    parser.add_argument('--tmp_dir', type=str, default='', help='Optional temp folder for precropped videos')
    parser.add_argument('--keep_tmp', action='store_true', help='Keep temporary precropped videos')

    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
    parser.add_argument('--checkpoint', type=str, default='trained_models/SMIRK_em1.pt', help='Path to the checkpoint')

    parser.add_argument('--crop_output_size', type=int, default=224, help='Precrops output size')
    parser.add_argument('--crop_scale', type=float, default=1.4, help='Precrops scale')
    parser.add_argument('--crop_smooth_alpha', type=float, default=0.9, help='Precrops EMA smoothing')
    parser.add_argument('--crop_min_alpha', type=float, default=0.3, help='Precrops minimum alpha')
    parser.add_argument('--crop_fast_threshold', type=float, default=25.0, help='Precrops fast motion threshold')
    parser.add_argument('--crop_max_size_change', type=float, default=0.2, help='Precrops max size change')
    parser.add_argument('--crop_detector', choices=['mediapipe', 'insightface'], default='insightface', help='Precrops detector')

    # --- MICA integration ---
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--mica_identity', type=str, help='Path to a precomputed MICA identity .npy file (shape code)')
    group.add_argument('--mica_ref_image', type=str, help='Path to a reference face image to compute MICA identity')
    group.add_argument('--mica_auto_ref', action='store_true', help='Automatically pick a reference face frame from the input video')
    group.add_argument('--mica_ref_frame', type=int, help='Use a specific frame index from the input video as MICA reference')

    parser.add_argument('--mica_repo', type=str, default='MICA', help='Path to the MICA repo (folder containing demo.py)')
    parser.add_argument('--mica_checkpoint', type=str, default='MICA/data/pretrained/mica.tar', help='Path to the MICA pretrained checkpoint')
    parser.add_argument('--mica_keep', action='store_true', help='Keep temporary MICA outputs (for debugging)')

    args = parser.parse_args()

    smirk_encoder = SmirkEncoder().to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    checkpoint_encoder = {k.replace('smirk_encoder.', ''): v for k, v in checkpoint.items() if 'smirk_encoder' in k}
    smirk_encoder.load_state_dict(checkpoint_encoder)
    smirk_encoder.eval()

    video_paths = _resolve_video_list(args)
    for input_path in video_paths:
        if not input_path.exists():
            print(f"Skipping missing video: {input_path}")
            continue

        npz_path = Path(args.npz_out_dir) / f"{input_path.stem}.npz"
        if npz_path.exists():
            print(f"Already processed, skipping: {input_path} -> {npz_path}")
            continue

        try:
            mica_shape = _get_mica_shape(args, input_path)
            npz_path = process_single_video(input_path, args, smirk_encoder, mica_shape)
            print(f"Saved: {npz_path}")
        except Exception as exc:
            print(f"Failed on {input_path}: {exc}")

