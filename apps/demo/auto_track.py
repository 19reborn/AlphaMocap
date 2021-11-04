'''
  @ Date: 2021-06-25 15:59:35
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-28 10:32:24
  @ FilePath: /EasyMocapRelease/apps/demo/auto_track.py
'''
from easymocap.assignment.track import Track2D, Track3D
import os
import json

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('out', type=str)
    parser.add_argument('--track3d', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    cfg = {
        'path': args.path,
        'out': args.out,
        'WINDOW_SIZE': 10,
        'MIN_FRAMES': 10,
        'SMOOTH_SIZE': 10
    }
    if args.track3d:
        tracker = Track3D(with2d=False, **cfg)
    else:
        tracker = Track2D(**cfg)
    tracker.auto_track()

    ## draw 3D pose and reprojection

    file_path = os.path.join(args.out,'keypoints3d')
    for frame in range(60):
        path = os.path.join(file_path,str(frame).rjust(6,'0')+'.json')
        assert os.path.exists(path)
        with open(path ,'r') as json_file:
            data = json.load(json_file)