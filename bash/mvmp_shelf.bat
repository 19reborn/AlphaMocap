@echo off
cmd /k python D:\Study\2021_Research\animation\experiments\EasyMocap-master/scripts/preprocess/extract_video.py E:\study\EasyMocap\datasets\shelf --openpose D:\Study\2021\frontier\2\openpose 

cmd /k python D:\Study\2021_Research\animation\experiments\EasyMocap-master/apps/demo/mvmp.py E:\study\EasyMocap\datasets\shelf --out E:\study\EasyMocap\datasets\shelf/output --annot annots --cfg config/exp/mvmp1f_ori.yml --undis --vis_det --vis_repro

cmd /k python D:\Study\2021_Research\animation\experiments\EasyMocap-master/apps/demo/auto_track.py E:\study\EasyMocap\datasets\shelf/output E:\study\EasyMocap\datasets\shelf/output-track --track3d

cmd /k python D:\Study\2021_Research\animation\experiments\EasyMocap-master/apps/demo/smpl_from_keypoints.py E:\study\EasyMocap\datasets\shelf --skel E:\study\EasyMocap\datasets\shelf/output-track/keypoints3d --out E:\study\EasyMocap\datasets\shelf/output-track/smpl --verbose --opts smooth_poses 1e1

pause