
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# --- Import SMIRK demo_video dependencies ---
import cv2
from skimage.transform import estimate_transform  # keep only for estimating; we'll warp via cv2
# from skimage.transform import warp  # removed usage

from src.smirk_encoder import SmirkEncoder
from src.FLAME.FLAME import FLAME
from src.renderer.renderer import Renderer
import src.utils.masking as masking_utils
from utils.mediapipe_utils import run_mediapipe
from datasets.base_dataset import create_mask
import torch.nn.functional as F


def crop_face_tform(frame, landmarks, scale=1.0, image_size=224):
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])

    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])

    size = int(old_size * scale)

    # crop image
    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)

    # Return 2x3 affine matrix (src->dst) and its inverse (dst->src)
    M = tform.params[:2]  # 2x3
    Minv = cv2.invertAffineTransform(M)
    return M, Minv


def warp_affine(image_bgr, M, dsize):
    # cv2.warpAffine expects width,height in dsize
    return cv2.warpAffine(image_bgr, M, dsize, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


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
        score = area_ratio - 0.3 * center_penalty

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
    mica_repo_dir = Path(mica_repo_dir).resolve()
    demo_py = mica_repo_dir / 'demo.py'
    if not demo_py.exists():
        raise FileNotFoundError(f"MICA demo.py not found at {demo_py}")

    mica_checkpoint = Path(mica_checkpoint).resolve()  # ensure absolute

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


def main():
    parser = argparse.ArgumentParser(
        description='SMIRK video tracker with fixed FLAME shape from MICA identity.'
    )
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input video')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
    parser.add_argument('--checkpoint', type=str, default='trained_models/SMIRK_em1.pt', help='Path to the SMIRK checkpoint')
    parser.add_argument('--crop', action='store_true', help='Crop the face using mediapipe')
    parser.add_argument('--out_path', type=str, default='output', help='Path to save the output (created if not exists)')
    parser.add_argument('--use_smirk_generator', action='store_true', help='Use SMIRK neural image-to-image translator')
    parser.add_argument('--render_orig', action='store_true', help='Render results at original video resolution')

    # Perf & logging controls
    parser.add_argument('--max_frames', type=int, default=0, help='Process at most this many frames (0 = all)')
    parser.add_argument('--frame_stride', type=int, default=1, help='Process every Nth frame (>=1). N>1 will duplicate frames in output.')
    parser.add_argument('--log_every', type=int, default=50, help='Log progress every N frames')

    # --- MICA integration args ---
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--mica_identity', type=str, help='Path to a precomputed MICA identity .npy file (shape code)')
    group.add_argument('--mica_ref_image', type=str, help='Path to a reference face image to compute MICA identity')
    group.add_argument('--mica_auto_ref', action='store_true', help='Automatically pick a reference face frame from the input video')

    parser.add_argument('--mica_repo', type=str, default='MICA', help='Path to the MICA repo (folder containing demo.py)')
    parser.add_argument('--mica_checkpoint', type=str, default='MICA/data/pretrained/mica.tar', help='Path to the MICA pretrained checkpoint')
    parser.add_argument('--mica_keep', action='store_true', help='Keep temporary MICA outputs (for debugging)')

    args = parser.parse_args()

    device = args.device
    torch.backends.cudnn.benchmark = True

    input_image_size = 224

    smirk_encoder = SmirkEncoder().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_encoder = {k.replace('smirk_encoder.', ''): v for k, v in checkpoint.items() if 'smirk_encoder' in k}
    smirk_encoder.load_state_dict(checkpoint_encoder)
    smirk_encoder.eval()

    use_smirk_generator = args.use_smirk_generator
    if use_smirk_generator:
        from src.smirk_generator import SmirkGenerator
        smirk_generator = SmirkGenerator(in_channels=6, out_channels=3, init_features=32, res_blocks=5).to(device)
        checkpoint_generator = {k.replace('smirk_generator.', ''): v for k, v in checkpoint.items() if 'smirk_generator' in k}
        smirk_generator.load_state_dict(checkpoint_generator)
        smirk_generator.eval()
        face_probabilities = masking_utils.load_probabilities_per_FLAME_triangle()

    flame = FLAME().to(device)
    renderer = Renderer().to(device)

    # ----------------------- obtain MICA identity (shape) ----------------------- #
    if args.mica_identity:
        mica_shape_np = np.load(args.mica_identity)
    else:
        if args.mica_auto_ref:
            tmp_auto = Path(tempfile.mkdtemp(prefix='mica_auto_ref_'))
            auto_img = extract_auto_ref_image(Path(args.input_path), tmp_auto)
            ref_img_path = auto_img
        else:
            ref_img_path = Path(args.mica_ref_image)

        mica_shape_np = run_mica_and_get_identity(
            mica_repo_dir=Path(args.mica_repo),
            ref_image_path=Path(ref_img_path),
            mica_checkpoint=Path(args.mica_checkpoint),
            keep_outputs=args.mica_keep,
        )

        if args.mica_auto_ref:
            try:
                shutil.rmtree(tmp_auto)
            except Exception:
                pass

    mica_shape = torch.tensor(mica_shape_np, dtype=torch.float32, device=device).view(1, -1)

    # ----------------------- open video ----------------------- #
    cap = cv2.VideoCapture(args.input_path)
    if not cap.isOpened():
        print('Error opening video file')
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # output size
    if args.render_orig:
        out_width = video_width
        out_height = video_height
    else:
        out_width = input_image_size
        out_height = input_image_size

    out_width *= (3 if use_smirk_generator else 2)

    out_dir = Path(args.out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (Path(args.input_path).stem + '_mica.mp4')
    cap_out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (out_width, out_height))

    processed = 0
    frame_idx = 0

    print(f"[INFO] Starting processing: total_frames={total_frames}, stride={args.frame_stride}, max_frames={args.max_frames}")

    with torch.inference_mode():
        while True:
            ok, image = cap.read()
            if not ok:
                break

            if frame_idx % args.frame_stride != 0:
                frame_idx += 1
                continue

            kpt_mediapipe = run_mediapipe(image)

            if args.crop:
                if (kpt_mediapipe is None):
                    print('[WARN] No landmarks found; skipping frame.')
                    frame_idx += 1
                    continue

                pts2d = kpt_mediapipe[..., :2]
                M, Minv = crop_face_tform(image, pts2d, scale=1.4, image_size=input_image_size)
                # Crop with fast cv2
                cropped_image = warp_affine(image, M, (224, 224))

                # Map the mediapipe points to cropped space (homogeneous)
                pts_h = np.hstack([pts2d, np.ones((pts2d.shape[0], 1))])
                cropped_kpt_mediapipe = (M @ pts_h.T).T
            else:
                cropped_image = image
                cropped_kpt_mediapipe = (kpt_mediapipe[..., :2] if kpt_mediapipe is not None else None)
                Minv = None

            # to RGB and tensor
            cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            cropped_image_rgb = cv2.resize(cropped_image_rgb, (224, 224))
            cropped_image_t = torch.from_numpy(cropped_image_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            cropped_image_t = cropped_image_t.to(device, non_blocking=True)

            # --- SMIRK encoder ---
            outputs = smirk_encoder(cropped_image_t)
            outputs['shape'] = mica_shape  # inject fixed identity

            # --- FLAME + render ---
            flame_output = flame.forward(outputs)
            renderer_output = renderer.forward(flame_output['vertices'], outputs['cam'],
                                               landmarks_fan=flame_output['landmarks_fan'],
                                               landmarks_mp=flame_output['landmarks_mp'])

            rendered_img = renderer_output['rendered_img']  # (1,3,224,224)

            if args.render_orig:
                if args.crop and Minv is not None:
                    # project the 224x224 render back to original size using inverse affine
                    rendered_img_numpy = (rendered_img.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                    rendered_img_bgr = cv2.cvtColor(rendered_img_numpy, cv2.COLOR_RGB2BGR)
                    rendered_img_orig_bgr = warp_affine(rendered_img_bgr, Minv, (video_width, video_height))
                    rendered_img_orig_rgb = cv2.cvtColor(rendered_img_orig_bgr, cv2.COLOR_BGR2RGB)
                    rendered_img_orig = torch.from_numpy(rendered_img_orig_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                else:
                    rendered_img_orig = F.interpolate(rendered_img, (video_height, video_width), mode='bilinear').cpu()

                full_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                full_image_t = torch.from_numpy(full_image_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                grid = torch.cat([full_image_t, rendered_img_orig], dim=3)
            else:
                grid = torch.cat([cropped_image_t.cpu(), rendered_img.cpu()], dim=3)

            if use_smirk_generator:
                if (kpt_mediapipe is None):
                    print('[WARN] No landmarks for smirk generator; skipping generator view.')
                else:
                    mask_ratio_mul = 5
                    mask_ratio = 0.01
                    mask_dilation_radius = 10

                    hull_mask = create_mask(cropped_kpt_mediapipe, (224, 224))

                    rendered_mask = 1 - (rendered_img == 0).all(dim=1, keepdim=True).float()
                    tmask_ratio = mask_ratio * mask_ratio_mul

                    npoints, _ = masking_utils.mesh_based_mask_uniform_faces(
                        renderer_output['transformed_vertices'],
                        flame_faces=flame.faces_tensor,
                        face_probabilities=face_probabilities,
                        mask_ratio=tmask_ratio
                    )

                    pmask = torch.zeros_like(rendered_mask)
                    rsing = torch.randint(0, 2, (npoints.size(0),)).to(npoints.device) * 2 - 1
                    rscale = torch.rand((npoints.size(0),)).to(npoints.device) * (mask_ratio_mul - 1) + 1
                    rbound = (npoints.size(1) * (1 / mask_ratio_mul) * (rscale ** rsing)).long()

                    for bi in range(npoints.size(0)):
                        pmask[bi, :, npoints[bi, :rbound[bi], 1], npoints[bi, :rbound[bi], 0]] = 1

                    hull_mask = torch.from_numpy(hull_mask).type(dtype=torch.float32).unsqueeze(0)

                    extra_points = cropped_image_t.cpu() * pmask.cpu()
                    masked_img = masking_utils.masking(cropped_image_t.cpu(), hull_mask, extra_points, mask_dilation_radius, rendered_mask=rendered_mask.cpu())

                    smirk_generator_input = torch.cat([rendered_img.cpu(), masked_img], dim=1)
                    reconstructed_img = smirk_generator(smirk_generator_input.to(device)).cpu()

                    if args.render_orig:
                        if args.crop and Minv is not None:
                            reconstructed_numpy = (reconstructed_img.squeeze(0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                            reconstructed_bgr = cv2.cvtColor(reconstructed_numpy, cv2.COLOR_RGB2BGR)
                            reconstructed_orig_bgr = warp_affine(reconstructed_bgr, Minv, (video_width, video_height))
                            reconstructed_orig_rgb = cv2.cvtColor(reconstructed_orig_bgr, cv2.COLOR_BGR2RGB)
                            reconstructed_orig = torch.from_numpy(reconstructed_orig_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                        else:
                            reconstructed_orig = F.interpolate(reconstructed_img, (video_height, video_width), mode='bilinear')

                        grid = torch.cat([grid, reconstructed_orig], dim=3)
                    else:
                        grid = torch.cat([grid, reconstructed_img], dim=3)

            grid_numpy = (grid.squeeze(0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            grid_bgr = cv2.cvtColor(grid_numpy, cv2.COLOR_RGB2BGR)
            cap_out.write(grid_bgr)

            processed += 1
            if processed % args.log_every == 0:
                print(f"[INFO] Processed {processed} frames (frame_idx={frame_idx})")

            if args.max_frames > 0 and processed >= args.max_frames:
                break

            frame_idx += 1

    cap.release()
    cap_out.release()

    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
