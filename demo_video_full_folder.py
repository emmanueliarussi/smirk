import os
import glob
import argparse
import torch
import numpy as np
import torchaudio
import torchaudio.transforms as T
import cv2
import collections
from utils.mediapipe_utils import run_mediapipe
from skimage.transform import estimate_transform, warp
from src.smirk_encoder import SmirkEncoder
from src.FLAME.FLAME import FLAME


def crop_face(frame, landmarks, scale=1.0, image_size=224):
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])

    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    size = int(old_size * scale)

    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    dst_pts = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, dst_pts)
    return tform


def extract_audio(input_video, output_wav, target_sr=16000):
    import subprocess
    subprocess.run([
        "ffmpeg", "-y", "-i", input_video,
        "-ar", str(target_sr), "-ac", "1", "-vn", output_wav
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, required=True, help='Folder with input videos')
    parser.add_argument('--out_path', type=str, default='results', help='Base path for npz and wav outputs')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--crop', action='store_true')

    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_path, 'npz'), exist_ok=True)
    os.makedirs(os.path.join(args.out_path, 'wav'), exist_ok=True)

    input_image_size = 224

    smirk_encoder = SmirkEncoder().to(args.device)
    flame = FLAME().to(args.device)

    checkpoint = torch.load(args.checkpoint)
    encoder_weights = {k.replace('smirk_encoder.', ''): v for k, v in checkpoint.items() if 'smirk_encoder' in k}
    smirk_encoder.load_state_dict(encoder_weights)
    smirk_encoder.eval()

    video_files = sorted(glob.glob(os.path.join(args.input_folder, '*.mp4')))

    for video_path in video_files:
        print(f"Processing {video_path}")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Failed to open {video_path}")
            continue

        smirk_outputs = collections.defaultdict(list)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            kpt_mediapipe = run_mediapipe(frame)

            if args.crop:
                if kpt_mediapipe is None:
                    print(f"No landmarks found for {video_path}")
                    break
                tform = crop_face(frame, kpt_mediapipe[..., :2], scale=1.4, image_size=input_image_size)
                cropped = warp(frame, tform.inverse, output_shape=(224, 224), preserve_range=True).astype(np.uint8)
            else:
                cropped = cv2.resize(frame, (224, 224))

            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            cropped_tensor = torch.tensor(cropped).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            cropped_tensor = cropped_tensor.to(args.device)

            with torch.no_grad():
                smirk_out = smirk_encoder(cropped_tensor)

            for k, v in smirk_out.items():
                smirk_outputs[k].append(v.detach().cpu())

        cap.release()

        # stack outputs frame by frame
        smirk_outputs = {k: torch.cat(v, dim=0).numpy() for k, v in smirk_outputs.items()}

        base_name = os.path.splitext(os.path.basename(video_path))[0]
        npz_out_path = os.path.join(args.out_path, 'npz', f"{base_name}.npz")
        np.savez(npz_out_path, **smirk_outputs)

        # Extract audio
        wav_out_path = os.path.join(args.out_path, 'wav', f"{base_name}.wav")
        extract_audio(video_path, wav_out_path)
