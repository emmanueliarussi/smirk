
import os
import sys
import glob
import argparse
import collections
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2

from skimage.transform import estimate_transform  # only to estimate sim3; warps via cv2

# SMIRK / FLAME pieces
from src.smirk_encoder import SmirkEncoder
from src.FLAME.FLAME import FLAME
from src.renderer.renderer import Renderer
from utils.mediapipe_utils import run_mediapipe


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def extract_audio_ffmpeg(input_video: str, output_wav: str, target_sr: int = 16000):
    cmd = ["ffmpeg", "-y", "-i", input_video, "-ar", str(target_sr), "-ac", "1", "-vn", output_wav]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print(f"[WARN] ffmpeg failed: {input_video} -> {output_wav}")


def crop_face_tform(frame, landmarks, scale=1.0, image_size=224):
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    size = int(old_size * scale)
    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2],
                        [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    dst_pts = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, dst_pts)
    M = tform.params[:2]
    Minv = cv2.invertAffineTransform(M)
    return M, Minv


def warp_affine(image_bgr, M, dsize):
    return cv2.warpAffine(image_bgr, M, dsize, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def grab_frame_to_file(video_path: Path, frame_idx: int, out_jpg: Path) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total == 0:
        cap.release(); return False
    frame_idx = max(0, min(total-1, frame_idx))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False
    if run_mediapipe(frame) is None:
        return False
    cv2.imwrite(str(out_jpg), frame)
    return True


def run_mica_and_get_identity(mica_repo_dir: Path, ref_image_path: Path, mica_checkpoint: Path) -> np.ndarray:
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
    ref_copy = input_dir / ('ref_image' + ref_image_path.suffix.lower())
    import shutil; shutil.copy2(ref_image_path, ref_copy)
    cmd = [sys.executable, str(demo_py), '-i', str(input_dir), '-o', str(output_dir), '-a', str(arcface_dir), '-m', str(mica_checkpoint)]
    try:
        subprocess.run(cmd, check=True, cwd=str(mica_repo_dir))
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"MICA demo.py failed: {e}")
    identity_npy = output_dir / 'ref_image' / 'identity.npy'
    if not identity_npy.exists():
        raise FileNotFoundError(f"MICA identity not found at {identity_npy}")
    ident = np.load(identity_npy)
    try:
        shutil.rmtree(tmp_root)
    except Exception:
        pass
    return ident


def get_mica_identity_with_retries(video_path: Path, mica_repo: Path, mica_ckpt: Path, attempts: int = 8) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video for MICA: {video_path}")
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()
    if total <= 0:
        print(f"[ERROR] No frames in video for MICA: {video_path}")
        return None
    centers = [0.5, 0.25, 0.75, 0.1, 0.9, 0.33, 0.66, 0.05, 0.95]
    candidates = []
    for c in centers:
        idx = int(total * c)
        if 0 <= idx < total and idx not in candidates:
            candidates.append(idx)
    candidates = candidates[:max(1, attempts)]
    tmp_root = Path(tempfile.mkdtemp(prefix='mica_ref_'))
    try:
        for i, idx in enumerate(candidates, 1):
            ref_img = tmp_root / f"ref_{idx}.jpg"
            if not grab_frame_to_file(video_path, idx, ref_img):
                continue
            try:
                ident = run_mica_and_get_identity(mica_repo, ref_img, mica_ckpt)
                return ident
            except Exception as e:
                print(f"[WARN] MICA attempt {i}/{len(candidates)} failed at frame {idx}: {e}")
                continue
        print(f"[ERROR] All MICA attempts failed for {video_path}")
        return None
    finally:
        import shutil; shutil.rmtree(tmp_root, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description="Robust batch SMIRK tracking with MICA identity. Skips completed files, continues on failures.")
    ap.add_argument('--input_folder', required=True)
    ap.add_argument('--out_path', default='results')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--crop', action='store_true')
    # Video export controls
    ap.add_argument('--export_video', action='store_true', help='Export visualization for all videos')
    ap.add_argument('--export_every', type=int, default=50, help='Heartbeat: export video every Nth video (0=disable)')
    ap.add_argument('--render_orig', action='store_true')
    ap.add_argument('--frame_stride', type=int, default=1)
    ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--overwrite', action='store_true', help='Overwrite existing npz/wav if present')
    # MICA
    grp = ap.add_mutually_exclusive_group(required=False)
    grp.add_argument('--mica_identity', type=str)
    grp.add_argument('--mica_ref_image', type=str)
    ap.add_argument('--mica_auto_ref', action='store_true')
    ap.add_argument('--mica_repo', type=str, default='MICA')
    ap.add_argument('--mica_checkpoint', type=str, default='MICA/data/pretrained/mica.tar')
    ap.add_argument('--mica_retries', type=int, default=8)
    ap.add_argument('--reuse_last_mica_on_fail', action='store_true')
    args = ap.parse_args()

    device = args.device
    torch.backends.cudnn.benchmark = True

    out_npz = Path(args.out_path) / 'npz'
    out_wav = Path(args.out_path) / 'wav'
    out_vid = Path(args.out_path) / 'mp4'
    ensure_dir(out_npz); ensure_dir(out_wav)
    if args.export_video or (args.export_every and args.export_every > 0):
        ensure_dir(out_vid)

    smirk_encoder = SmirkEncoder().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    enc_w = {k.replace('smirk_encoder.', ''): v for k, v in ckpt.items() if 'smirk_encoder' in k}
    smirk_encoder.load_state_dict(enc_w)
    smirk_encoder.eval()

    flame = FLAME().to(device)
    renderer = Renderer().to(device) if (args.export_video or (args.export_every and args.export_every > 0)) else None

    mica_identity_global = None
    if args.mica_identity:
        mica_identity_global = np.load(args.mica_identity)

    videos = sorted(glob.glob(os.path.join(args.input_folder, '*.mp4')))
    if not videos:
        print(f"[WARN] No .mp4 files in {args.input_folder}")
        return

    print(f"[INFO] {len(videos)} videos")

    last_ident = mica_identity_global.copy() if mica_identity_global is not None else None

    for i, vp in enumerate(videos, 1):
        base = os.path.splitext(os.path.basename(vp))[0]
        npz_path = (out_npz / f"{base}.npz")
        wav_path = (out_wav / f"{base}.wav")
        # Skip if outputs exist and not overwriting
        if not args.overwrite and npz_path.exists() and wav_path.exists():
            print(f"[SKIP] {base} (already has NPZ and WAV)")
            continue

        # Decide whether to export video for this file (heartbeat or all)
        do_export_video = args.export_video or (args.export_every and args.export_every > 0 and i % args.export_every == 0)

        print(f"[{i}/{len(videos)}] {base}" + (f"  [export_video]" if do_export_video else ""))

        try:
            # Identity
            if mica_identity_global is not None:
                mica_np = mica_identity_global
            else:
                if args.mica_ref_image and not args.mica_auto_ref:
                    try:
                        mica_np = run_mica_and_get_identity(Path(args.mica_repo), Path(args.mica_ref_image), Path(args.mica_checkpoint))
                    except Exception as e:
                        print(f"[ERROR] MICA ref-image failed: {e}")
                        if args.reuse_last_mica_on_fail and last_ident is not None:
                            print("[INFO] Reusing last identity")
                            mica_np = last_ident
                        else:
                            print("[SKIP] No identity; skipping video")
                            continue
                else:
                    mica_np = get_mica_identity_with_retries(Path(vp), Path(args.mica_repo), Path(args.mica_checkpoint), attempts=max(1, args.mica_retries))
                    if mica_np is None:
                        if args.reuse_last_mica_on_fail and last_ident is not None:
                            print("[INFO] Reusing last identity")
                            mica_np = last_ident
                        else:
                            print("[SKIP] No identity; skipping video")
                            continue
            last_ident = mica_np.copy()
            mica_t = torch.tensor(mica_np, dtype=torch.float32, device=device).view(1, -1)

            cap = cv2.VideoCapture(vp)
            if not cap.isOpened():
                print(f"[WARN] cannot open {vp}")
                continue
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if do_export_video:
                w = W if args.render_orig else 224
                h = H if args.render_orig else 224
                w *= 2
                out_video_path = str((out_vid / f"{base}_mica.mp4").resolve())
                writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            else:
                writer = None

            store = collections.defaultdict(list)
            processed = 0
            idx = 0

            with torch.inference_mode():
                while True:
                    ok, frame_bgr = cap.read()
                    if not ok: break
                    if idx % args.frame_stride != 0:
                        idx += 1; continue

                    kpts = run_mediapipe(frame_bgr)
                    if args.crop:
                        if kpts is None:
                            idx += 1; continue
                        pts2d = kpts[..., :2]
                        M, Minv = crop_face_tform(frame_bgr, pts2d, scale=1.4, image_size=224)
                        crop_bgr = warp_affine(frame_bgr, M, (224, 224))
                    else:
                        crop_bgr = cv2.resize(frame_bgr, (224, 224)); Minv = None

                    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                    img_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    img_t = img_t.to(device, non_blocking=True)

                    raw = smirk_encoder(img_t)
                    raw['shape'] = mica_t

                    key_map = {
                        'expression_params': 'exp',
                        'pose_params':       'pose',
                        'eyelid_params':     'eyes',
                        'jaw_params':        'jaw',
                        'cam':               'cam',
                        'exp':               'exp',
                        'pose':              'pose',
                        'eyes':              'eyes',
                        'jaw':               'jaw',
                    }
                    for src, dst in key_map.items():
                        if src in raw:
                            store[dst].append(raw[src].detach().cpu())

                    if writer is not None:
                        flame_out = FLAME().to(device).forward(raw) if False else None  # placeholder; leave for clarity
                        flame_out = flame.forward(raw)
                        render_out = renderer.forward(flame_out['vertices'], raw['cam'], landmarks_fan=flame_out['landmarks_fan'], landmarks_mp=flame_out['landmarks_mp'])
                        rendered = render_out['rendered_img']
                        if args.render_orig and Minv is not None:
                            rend_np = (rendered.squeeze(0).permute(1,2,0).cpu().numpy()*255.0).astype(np.uint8)
                            rend_bgr = cv2.cvtColor(rend_np, cv2.COLOR_RGB2BGR)
                            back = warp_affine(rend_bgr, Minv, (W,H))
                            left = frame_bgr; right = back
                        else:
                            left = cv2.resize(frame_bgr, (224,224))
                            rend_np = (rendered.squeeze(0).permute(1,2,0).cpu().numpy()*255.0).astype(np.uint8)
                            right = cv2.cvtColor(rend_np, cv2.COLOR_RGB2BGR)
                        panel = np.concatenate([left, right], axis=1)
                        writer.write(panel)

                    processed += 1; idx += 1
                    if processed % args.log_every == 0:
                        print(f"[{base}] {processed}/{total}")

            cap.release()
            if writer is not None:
                writer.release()

            stacked = {k: torch.cat(v, dim=0).numpy() for k, v in store.items()}
            if 'pose' in stacked and 'neck' not in stacked:
                stacked['neck'] = stacked['pose']
            stacked['mica_shape'] = np.array(mica_np).reshape(-1)

            np.savez(str(npz_path.resolve()), **stacked)
            extract_audio_ffmpeg(vp, str(wav_path.resolve()))
            done_msg = f"[DONE] {base}: npz -> {npz_path} | wav -> {wav_path}"
            if do_export_video:
                done_msg += f" | mp4 -> {out_video_path}"
            print(done_msg)

        except KeyboardInterrupt:
            print("\n[INTERRUPTED]"); raise
        except Exception as e:
            print(f"[ERROR] {base}: {e}")
            continue


if __name__ == "__main__":
    main()
