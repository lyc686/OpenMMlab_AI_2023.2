# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import cv2
import mmcv
#from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.apis import inference_detector, init_detector

import numpy as np
import torch




def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection video demo')
    parser.add_argument('video', help='Video file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cpu', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument('--out', type=str, help='Output video file')
    parser.add_argument('--show', action='store_true', help='Show video')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='The interval of show (s), 0 is block')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.out or args.show, \
        ('Please specify at least one operation (save/show the '
         'video) with the argument "--out" or "--show"')

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    
    video_reader = mmcv.VideoReader(args.video)
    video_writer = None

    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.out, fourcc, video_reader.fps,
            (video_reader.width, video_reader.height))   

    for frame in track_iter_progress(video_reader):
        result = inference_detector(model, frame)        
        mask = None
        masks = result[1][0]
        for i in range(len(masks)):
            if result[0][0][i][-1] >= args.score_thr:
                if not mask is None:
                    mask = mask | masks[i]
                else:
                    mask = masks[i]
                    
        masked_b = frame[:, :, 0] * mask
        masked_g = frame[:, :, 1] * mask
        masked_r = frame[:, :, 2] * mask
        masked = np.concatenate([masked_b[:, :, None],masked_g[:, :, None],masked_r[:, :, None]], axis=2)
        
        un_mask = 1 - mask
        frame_b = frame[:, :, 0] * un_mask
        frame_g = frame[:, :, 1] * un_mask
        frame_r = frame[:, :, 2] * un_mask    

        frame = np.concatenate([frame_b[:, :, None],frame_g[:, :, None],frame_r[:, :, None]], axis=2).astype(np.uint8)
        frame = mmcv.bgr2gray(frame, keepdim=True)
        frame = np.concatenate([frame, frame, frame], axis=2)
        
        frame += masked
        
        if args.show:
            cv2.namedWindow('video', 0)
            mmcv.imshow(frame, 'video', args.wait_time)
        if args.out:
            video_writer.write(frame)
        
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
                    
if __name__ == '__main__':
    main()

"""
终端执行指令
python .\splash_video.py --out splash_video.mp4 .\test_video.mp4 .\configs\balloon.py .\checkpoints\latest.pth
"""
    
    