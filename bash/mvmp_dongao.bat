@echo off
powershell python D:\Study\2021_Research\animation\experiments\EasyMocap-master/scripts/preprocess/extract_video.py E:\study\EasyMocap\datasets\dongao --openpose D:\Study\2021\frontier\2\openpose
powershell python D:\Study\2021_Research\animation\experiments\EasyMocap-master/apps/demo/mvmp.py E:\study\EasyMocap\datasets\dongao --out E:\study\EasyMocap\datasets\dongao/output --annot annots --cfg config/exp/mvmp1f_dongao.yml --undis --vis_det --vis_repro --vis3d
powershell python D:\Study\2021_Research\animation\experiments\EasyMocap-master/apps/demo/auto_track.py E:\study\EasyMocap\datasets\dongao/output E:\study\EasyMocap\datasets\dongao/output-track --track3d
powershell python D:\Study\2021_Research\animation\experiments\EasyMocap-master/apps/demo/smpl_from_keypoints.py E:\study\EasyMocap\datasets\dongao --skel E:\study\EasyMocap\datasets\dongao/output-track/keypoints3d --out E:\study\EasyMocap\datasets\dongao/output-track/smpl --verbose --opts smooth_poses 1e1

pause   