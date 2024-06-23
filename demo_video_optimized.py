import torch
import cv2
import numpy as np
from skimage.transform import estimate_transform, warp
from src.smirk_encoder import SmirkEncoder
from src.FLAME.FLAME import FLAME
from src.renderer.renderer import Renderer
import argparse
import os
import src.utils.masking as masking_utils
from utils.mediapipe_utils import run_mediapipe
from datasets.base_dataset import create_mask
import torch.nn.functional as F

import concurrent.futures

import time
import glob


def process_image(image):    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    image_tensor = image_tensor.to('cuda')
    
    return image_tensor

def post_process_image(image):
    image = image.squeeze(0).permute(1,2,0).numpy()*255.0
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def process_images_in_parallel(images, max_workers=4):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = [x for x in executor.map(process_image, images)]
    
    return results

def post_process_image_in_parallel(images, max_workers=4):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = [x for x in executor.map(post_process_image, images)]
    
    return results

def forward_pass(video_path, output_dir, flame, renderer, smirk_encoder):
    
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print('Error opening video file')
        exit()

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path_name = video_path.split('/')[-1].split('.')[0]
    flame_output_dir = f'{output_dir}/flame_outputs'
    video_output_dir = f'{output_dir}/video_outputs'
    out_path_video = f'{video_output_dir}/{out_path_name}'
    out_path_flame = f'{flame_output_dir}/{out_path_name}'

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(flame_output_dir, exist_ok=True)
    os.makedirs(video_output_dir, exist_ok=True)

    cap_out = cv2.VideoWriter(f"{out_path_video}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (224*2, 224))

    # Get frames from the video
    frames = []
    ret = True
    while ret:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    # Release the video capture object
    cap.release()

    processed_frames = process_images_in_parallel(frames)

    batch_tensor = torch.cat(processed_frames)

    # split the frames into batches of 32
    batch_size = 32

    visualized_images = []
    flame_outputs = []
    pose, shape, expression, eyelid, jaw, cam = np.zeros((len(batch_tensor), 3)), np.zeros((len(batch_tensor), 300)), np.zeros((len(batch_tensor), 50)), np.zeros((len(batch_tensor), 2)), np.zeros((len(batch_tensor), 3)), np.zeros((len(batch_tensor), 3))

    for i in range(0, len(batch_tensor), batch_size):
        batch = batch_tensor[i:i+batch_size]
        outputs = smirk_encoder(batch) # dict_keys(['pose_params', 'cam', 'shape_params', 'expression_params', 'eyelid_params', 'jaw_params'])

        flame_output = flame.forward(outputs)

        renderer_output = renderer.forward(flame_output['vertices'], outputs['cam'],
                                            landmarks_fan=flame_output['landmarks_fan'], landmarks_mp=flame_output['landmarks_mp'])

        rendered_img = renderer_output['rendered_img']

        grid = torch.cat([batch, rendered_img], dim=3).detach().cpu()

        visualized_images.extend([grid[i] for i in range(grid.shape[0])])
        
        pose[i:i+batch_size] = outputs['pose_params'].detach().cpu().numpy()
        shape[i:i+batch_size] = outputs['shape_params'].detach().cpu().numpy()
        expression[i:i+batch_size] = outputs['expression_params'].detach().cpu().numpy()
        eyelid[i:i+batch_size] = outputs['eyelid_params'].detach().cpu().numpy()
        jaw[i:i+batch_size] = outputs['jaw_params'].detach().cpu().numpy()
        cam[i:i+batch_size] = outputs['cam'].detach().cpu().numpy()


    # write flame outputs to a numpy file
    np.savez(f'{out_path_flame}.npz', pose=pose, shape=shape, expression=expression, eyelid=eyelid, jaw=jaw, cam=cam)
    
    visualized_images = post_process_image_in_parallel(visualized_images)
    # write the frames to the output video
    for image in visualized_images:
        cap_out.write(image)

    cap_out.release()





if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', type=str, default='samples/mead_90.png', help='Path to the input image/video')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
    parser.add_argument('--checkpoint', type=str, default='trained_models/SMIRK_em1.pt', help='Path to the checkpoint')
    parser.add_argument('--crop', action='store_true', help='Crop the face using mediapipe')
    parser.add_argument('--out_path', type=str, default='output', help='Path to save the output (will be created if not exists)')
    parser.add_argument('--use_smirk_generator', action='store_true', help='Use SMIRK neural image to image translator to reconstruct the image')
    parser.add_argument('--render_orig', action='store_true', help='Present the result w.r.t. the original image/video size')

    args = parser.parse_args()

    input_image_size = 224
    

    # ----------------------- initialize configuration ----------------------- #
    smirk_encoder = SmirkEncoder().to(args.device)
    checkpoint = torch.load(args.checkpoint)
    checkpoint_encoder = {k.replace('smirk_encoder.', ''): v for k, v in checkpoint.items() if 'smirk_encoder' in k} # checkpoint includes both smirk_encoder and smirk_generator

    smirk_encoder.load_state_dict(checkpoint_encoder)
    smirk_encoder.eval()


    # ---- visualize the results ---- #

    flame = FLAME().to(args.device)
    renderer = Renderer().to(args.device)

    # ---- Run the model on the input ---- #

    videos = os.listdir(args.input_path)
    videos = [os.path.join(args.input_path, video) for video in videos if video.endswith('.mp4')]
    print("Found {} videos".format(len(videos)))
    
    import tqdm
    for video in tqdm.tqdm(videos):
        forward_pass(video, args.out_path, flame, renderer, smirk_encoder)
