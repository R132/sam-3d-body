#!/usr/bin/env python3
"""Run MHR pose optimization on a random subset of Sapiens frames.

Saves optimization before/after visuals and meshes to the output folder.
"""
import os
import sys
import json
import random
import argparse

import numpy as np
import cv2
import pyrootutils

# ensure repository root is on PYTHONPATH (like demo.py)
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".sl"],
    pythonpath=True,
    dotenv=True,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sapiens_json", required=True)
    parser.add_argument("--images_dir", default="")
    parser.add_argument("--out_dir", default=".")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--num_iters", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--checkpoint", default="/home/gj/Projects/2026.05.07-SMA3/checkpoints/sam-3d-body/sam-3d-body-vith/model.ckpt")
    parser.add_argument("--mhr_path", default="/home/gj/Projects/2026.05.07-SMA3/checkpoints/MHR/assets/mhr_model.pt")
    args = parser.parse_args()

    try:
        from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
        from pose_optimization import optimize_mhr_pose
    except Exception as e:
        print('IMPORT_FAIL', e)
        sys.exit(1)

    # Load model
    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    print('Loading SAM3D model on', device)
    model, cfg = load_sam_3d_body(args.checkpoint, device=device, mhr_path=args.mhr_path)
    estimator = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)

    with open(args.sapiens_json, 'r') as f:
        data = json.load(f)
    frames = data.get('frames', [])
    if len(frames) == 0:
        print('No frames in JSON')
        return

    images_dir = args.images_dir if args.images_dir else os.path.dirname(args.sapiens_json)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    random.seed(0)
    idxs = random.sample(range(len(frames)), min(args.num_samples, len(frames)))
    print('Selected indices:', idxs)

    for i in idxs:
        fr = frames[i]
        image_name = fr.get('image_name')
        instances = fr.get('instances', [])
        if len(instances) == 0:
            print('skip no instances for', image_name)
            continue
        kps = instances[0].get('keypoints')
        if kps is None:
            print('skip no keypoints for', image_name)
            continue

        img_path = os.path.join(images_dir, image_name)
        if not os.path.exists(img_path):
            print('image not found', img_path)
            continue

        print('\nProcessing', image_name)
        outputs = estimator.process_one_image(img_path, bbox_thr=0.8, use_mask=False)
        if len(outputs) == 0:
            print('no outputs for', image_name)
            continue
        first = outputs[0]

        init_params = {
            'global_rot': first.get('global_rot'),
            'body_pose_params': first.get('body_pose'),
            'hand_pose_params': first.get('hand'),
            'scale_params': first.get('scale'),
            'shape_params': first.get('shape'),
            'expr_params': first.get('expr_params') if 'expr_params' in first else first.get('expr'),
            'pred_cam_t': first.get('pred_cam_t'),
            'focal_length': float(first.get('focal_length', 1.0)),
        }

        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        f = init_params.get('focal_length', 1.0)
        cam_K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], dtype=np.float32)

        out_prefix = os.path.join(out_dir, f'opt_{os.path.splitext(image_name)[0]}')
        try:
            res = optimize_mhr_pose(
                mhr_head=estimator.model.head_pose,
                init_params=init_params,
                sapiens_kps2d=np.array(kps),
                cam_K=cam_K,
                img=img,
                faces=estimator.faces,
                out_prefix=out_prefix,
                device=device,
                num_iters=args.num_iters,
                lr=args.lr,
            )
            print('Result:', res)
        except Exception as e:
            print('OPTIMIZE_FAIL', image_name, type(e).__name__, e)

    print('Done')


if __name__ == '__main__':
    main()
