import torch
import cv2
import numpy as np
from skimage.transform import estimate_transform, warp, SimilarityTransform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from src.smirk_encoder import SmirkEncoder
from src.FLAME.FLAME import FLAME
from src.renderer.renderer import Renderer
import argparse
import os
import src.utils.masking as masking_utils
from utils.mediapipe_utils import run_mediapipe
from datasets.base_dataset import create_mask
import torch.nn.functional as F


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', type=str, default='samples/mead_90.png', help='Path to the input image/video')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
    parser.add_argument('--checkpoint', type=str, default='trained_models/SMIRK_em1.pt', help='Path to the checkpoint')
    parser.add_argument('--crop', action='store_true', help='Crop the face using mediapipe')
    parser.add_argument('--out_path', type=str, default='output', help='Path to save the output (will be created if not exists)')
    parser.add_argument('--use_smirk_generator', action='store_true', help='Use SMIRK neural image to image translator to reconstruct the image')
    parser.add_argument('--render_orig', action='store_true', help='Present the result w.r.t. the original image/video size')

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

    input_image_size = 224
    

    # ----------------------- initialize configuration ----------------------- #
    smirk_encoder = SmirkEncoder().to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    checkpoint_encoder = {k.replace('smirk_encoder.', ''): v for k, v in checkpoint.items() if 'smirk_encoder' in k} # checkpoint includes both smirk_encoder and smirk_generator

    smirk_encoder.load_state_dict(checkpoint_encoder)
    smirk_encoder.eval()

    if args.use_smirk_generator:
        from src.smirk_generator import SmirkGenerator
        smirk_generator = SmirkGenerator(in_channels=6, out_channels=3, init_features=32, res_blocks=5).to(args.device)

        checkpoint_generator = {k.replace('smirk_generator.', ''): v for k, v in checkpoint.items() if 'smirk_generator' in k} # checkpoint includes both smirk_encoder and smirk_generator
        smirk_generator.load_state_dict(checkpoint_generator)
        smirk_generator.eval()

        # load also triangle probabilities for sampling points on the image
        face_probabilities = masking_utils.load_probabilities_per_FLAME_triangle()  


    # ---- visualize the results ---- #

    flame = FLAME().to(args.device)
    renderer = Renderer().to(args.device)

    # ----------------------- obtain MICA identity (shape) ----------------------- #
    if args.mica_identity:
        mica_shape_np = np.load(args.mica_identity)
    else:
        tmp_auto = None
        if args.mica_auto_ref:
            tmp_auto = Path(tempfile.mkdtemp(prefix='mica_auto_ref_'))
            ref_img_path = extract_auto_ref_image(Path(args.input_path), tmp_auto)
        elif args.mica_ref_frame is not None:
            tmp_auto = Path(tempfile.mkdtemp(prefix='mica_ref_frame_'))
            cap_ref = cv2.VideoCapture(args.input_path)
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

    mica_shape = torch.tensor(mica_shape_np, dtype=torch.float32, device=args.device).view(1, -1)


    cap = cv2.VideoCapture(args.input_path)

    if not cap.isOpened():
        print('Error opening video file')
        exit()

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # calculate size of output video
    if args.render_orig:
        out_width = video_width
        out_height = video_height
    else:
        out_width = input_image_size
        out_height = input_image_size

    if args.use_smirk_generator:
        out_width *= 3
    else:
        out_width *= 2

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    output_video_path = f"{args.out_path}/{args.input_path.split('/')[-1].split('.')[0]}.mp4"
    cap_out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        video_fps,
        (out_width, out_height)
    )

    outputs_list = []
    tform_list = [] if args.crop else None
    cropped_kpt_list = [] if args.use_smirk_generator else None
    tracked_frames_mask = []

    # ----------------------- Pass 1: compute outputs on successful detections ----------------------- #
    while True:
        ret, image = cap.read()
        if not ret:
            break

        kpt_mediapipe = run_mediapipe(image)

        if args.crop:
            if kpt_mediapipe is None:
                outputs_list.append(None)
                tform_list.append(None)
                if cropped_kpt_list is not None:
                    cropped_kpt_list.append(None)
                tracked_frames_mask.append(0)
                continue

            kpt_mediapipe = kpt_mediapipe[..., :2]
            tform = crop_face(image, kpt_mediapipe, scale=1.4, image_size=input_image_size)
            cropped_image = warp(image, tform.inverse, output_shape=(224, 224), preserve_range=True).astype(np.uint8)

            cropped_kpt_mediapipe = np.dot(
                tform.params,
                np.hstack([kpt_mediapipe, np.ones([kpt_mediapipe.shape[0], 1])]).T
            ).T
            cropped_kpt_mediapipe = cropped_kpt_mediapipe[:, :2]
        else:
            cropped_image = image
            cropped_kpt_mediapipe = kpt_mediapipe[..., :2] if kpt_mediapipe is not None else None

        cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        cropped_image_rgb = cv2.resize(cropped_image_rgb, (224, 224))
        cropped_tensor = torch.tensor(cropped_image_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        cropped_tensor = cropped_tensor.to(args.device)

        outputs = smirk_encoder(cropped_tensor)
        outputs_list.append({k: v.detach().cpu() for k, v in outputs.items()})
        tracked_frames_mask.append(1)

        if args.crop:
            tform_list.append(tform.params.copy())
        if cropped_kpt_list is not None:
            cropped_kpt_list.append(cropped_kpt_mediapipe)

    cap.release()

    # ----------------------- Fill missing frames by interpolation ----------------------- #
    def interp_outputs(a, b, t):
        return {k: _interpolate_tensor(a[k], b[k], t) for k in a}

    outputs_list = _fill_missing_sequence(outputs_list, interp_outputs)

    if all(v is None for v in outputs_list):
        print('No valid detections were found in the video. Exiting...')
        exit()

    if args.crop:
        tform_list = _fill_missing_sequence(
            tform_list,
            lambda a, b, t: _interpolate_tensor(a, b, t)
        )

    if cropped_kpt_list is not None:
        cropped_kpt_list = _fill_missing_sequence(
            cropped_kpt_list,
            lambda a, b, t: _interpolate_tensor(a, b, t)
        )

    # ----------------------- Export parameter tensors ----------------------- #
    def _to_numpy_row(t):
        if isinstance(t, torch.Tensor):
            t = t.detach().cpu().numpy()
        return np.reshape(t, (1, -1))

    num_frames = len(outputs_list)
    tracked_frames_mask_np = np.array(tracked_frames_mask, dtype=np.float32).reshape(-1, 1)

    exp_np = np.concatenate([_to_numpy_row(o['expression_params']) for o in outputs_list], axis=0)
    pose_np = np.concatenate([_to_numpy_row(o['pose_params']) for o in outputs_list], axis=0)
    eyes_np = np.concatenate([_to_numpy_row(o['eyelid_params']) for o in outputs_list], axis=0)
    jaw_np = np.concatenate([_to_numpy_row(o['jaw_params']) for o in outputs_list], axis=0)
    shape_np = mica_shape.detach().cpu().numpy().reshape(-1)

    npz_path = os.path.splitext(output_video_path)[0] + '.npz'
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

    # ----------------------- Pass 2: render all frames ----------------------- #
    cap = cv2.VideoCapture(args.input_path)
    if not cap.isOpened():
        print('Error opening video file')
        exit()

    frame_idx = 0
    while True:
        ret, image = cap.read()
        if not ret:
            break

        outputs_cpu = outputs_list[frame_idx]
        if outputs_cpu is None:
            frame_idx += 1
            continue

        outputs = {k: v.to(args.device) for k, v in outputs_cpu.items()}
        outputs['shape'] = mica_shape

        if args.crop:
            tform = SimilarityTransform(matrix=tform_list[frame_idx])
            cropped_image = warp(image, tform.inverse, output_shape=(224, 224), preserve_range=True).astype(np.uint8)
            cropped_kpt_mediapipe = cropped_kpt_list[frame_idx] if cropped_kpt_list is not None else None
        else:
            cropped_image = image
            cropped_kpt_mediapipe = cropped_kpt_list[frame_idx] if cropped_kpt_list is not None else None

        cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        cropped_image_rgb = cv2.resize(cropped_image_rgb, (224, 224))
        cropped_tensor = torch.tensor(cropped_image_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        cropped_tensor = cropped_tensor.to(args.device)

        flame_output = flame.forward(outputs)
        renderer_output = renderer.forward(
            flame_output['vertices'],
            outputs['cam'],
            landmarks_fan=flame_output['landmarks_fan'],
            landmarks_mp=flame_output['landmarks_mp']
        )

        rendered_img = renderer_output['rendered_img']

        if args.render_orig:
            if args.crop:
                rendered_img_numpy = (rendered_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
                rendered_img_orig = warp(rendered_img_numpy, tform, output_shape=(video_height, video_width), preserve_range=True).astype(np.uint8)
                rendered_img_orig = torch.Tensor(rendered_img_orig).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            else:
                rendered_img_orig = F.interpolate(rendered_img, (video_height, video_width), mode='bilinear').cpu()

            full_image = torch.Tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            grid = torch.cat([full_image, rendered_img_orig], dim=3)
        else:
            grid = torch.cat([cropped_tensor, rendered_img], dim=3)

        # ---- create the neural renderer reconstructed img ---- #
        if args.use_smirk_generator:
            if cropped_kpt_mediapipe is None:
                print('Could not find landmarks for the image using mediapipe and cannot create the hull mask for the smirk generator. Exiting...')
                exit()

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

            hull_mask = torch.from_numpy(hull_mask).type(dtype=torch.float32).unsqueeze(0).to(args.device)

            extra_points = cropped_tensor * pmask
            masked_img = masking_utils.masking(cropped_tensor, hull_mask, extra_points, mask_dilation_radius, rendered_mask=rendered_mask)

            smirk_generator_input = torch.cat([rendered_img, masked_img], dim=1)

            reconstructed_img = smirk_generator(smirk_generator_input)

            if args.render_orig:
                if args.crop:
                    reconstructed_img_numpy = (reconstructed_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
                    reconstructed_img_orig = warp(reconstructed_img_numpy, tform, output_shape=(video_height, video_width), preserve_range=True).astype(np.uint8)
                    reconstructed_img_orig = torch.Tensor(reconstructed_img_orig).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                else:
                    reconstructed_img_orig = F.interpolate(reconstructed_img, (video_height, video_width), mode='bilinear').cpu()

                grid = torch.cat([grid, reconstructed_img_orig], dim=3)
            else:
                grid = torch.cat([grid, reconstructed_img], dim=3)

        grid_numpy = grid.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0
        grid_numpy = grid_numpy.astype(np.uint8)
        grid_numpy = cv2.cvtColor(grid_numpy, cv2.COLOR_BGR2RGB)
        cap_out.write(grid_numpy)

        frame_idx += 1

    cap.release()
    cap_out.release()

