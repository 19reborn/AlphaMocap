@echo off

python scripts/preprocess/extract_video.py D:\Study\2021_Research\animation\experiments\EasyMocap-master\datasets\mvmp --no2d

python apps/demo/mvmp.py E:\study\EasyMocap\datasets\mvmp --out D:\Study\2021_Research\animation\experiments\EasyMocap-master\datasets\mvmp/output --annot annots --cfg config/exp/mvmp1f.yml --undis --vis_det --vis_repro

python apps/demo/auto_track.py D:\Study\2021_Research\animation\experiments\EasyMocap-master\datasets\mvmp/output D:\Study\2021_Research\animation\experiments\EasyMocap-master\datasets\mvmp/output-track --track3d

python apps/demo/smpl_from_keypoints.py D:\Study\2021_Research\animation\experiments\EasyMocap-master\datasets\mvmp --skel D:\Study\2021_Research\animation\experiments\EasyMocap-master\datasets\mvmp/output-track/keypoints3d --out D:\Study\2021_Research\animation\experiments\EasyMocap-master\datasets\mvmp/output-track/smpl --verbose --opts smooth_poses 1e1

pause